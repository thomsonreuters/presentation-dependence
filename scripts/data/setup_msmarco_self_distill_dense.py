#!/usr/bin/env python
r"""Materialize BGE-dense MS MARCO train candidates for self-distill silver (train-side).

Companion to ``scripts/data/setup_msmarco_self_distill.py`` (BM25). Reuses the **same**
deterministic qid order and query texts as the BM25 self-distill channel so the only
varied axis is the first-stage retriever for the BGE-trained OC-SFT control.

Outputs::

    data/msmarco-train-selfdistill-seed42-bge/
    ├── qids_30k.txt              # copied from BM25 channel (same prefix)
    ├── topics.tsv                # same queries as BM25 materialization
    ├── qrels.txt                 # copied or regenerated from ir_datasets
    ├── run.msmarco-v1-passage.bge-base-en-v1.5.selfdistill.txt
    ├── fixture.jsonl
    └── dataset_meta.yaml

Dense retrieval uses pyserini ``FaissSearcher`` (exact flat) or Lucene HNSW dense
(``msmarco-v1-passage.bge-base-en-v1.5.hnsw``, acceptable for training candidates).
Passage **text** is resolved from the Lucene ``msmarco-v1-passage`` index by docid.

Usage::

    # Reuse qids/topics from the BM25 channel; materialize 30K on a high-RAM box:
    uv run python scripts/data/setup_msmarco_self_distill_dense.py --query-count 30000

    # HNSW index (less RAM; default in the historical training build):
    uv run python scripts/data/setup_msmarco_self_distill_dense.py --query-count 30000 \\
        --search-index msmarco-v1-passage.bge-base-en-v1.5.hnsw --backend hnsw
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from presentation_dependence.utils.pyserini_index import pyserini_import, require_prebuilt_index

if __package__:
    from .setup_msmarco_self_distill import (
        DEFAULT_HITS,
        DEFAULT_SEED,
        append_materialized_qids,
        fixture_qids,
        qids_prefix_filename,
        read_qids,
        write_ir_qrels_for_qids,
        write_topics_tsv,
    )
else:
    from setup_msmarco_self_distill import (
        DEFAULT_HITS,
        DEFAULT_SEED,
        append_materialized_qids,
        fixture_qids,
        qids_prefix_filename,
        read_qids,
        write_ir_qrels_for_qids,
        write_topics_tsv,
    )

DEFAULT_BM25_DIR = Path("data/msmarco-train-selfdistill-seed42")
DEFAULT_OUT_DIR = Path("data/msmarco-train-selfdistill-seed42-bge")
DEFAULT_TEXT_INDEX = "msmarco-v1-passage"
DEFAULT_FAISS_INDEX = "msmarco-v1-passage.bge-base-en-v1.5"
DEFAULT_HNSW_INDEX = "msmarco-v1-passage.bge-base-en-v1.5.hnsw"
BGE_ENCODER = "BAAI/bge-base-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages:"
RETRIEVER_TOKEN = "bge-base-en-v1.5"
RUN_NAME = "run.msmarco-v1-passage.bge-base-en-v1.5.selfdistill.txt"


def _load_topics_tsv(path: Path) -> dict[str, str]:
    topics: dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            qid, text = line.split("\t", 1)
            topics[qid] = text
    return topics


def _copy_prefix_files(*, bm25_dir: Path, out_dir: Path, query_count: int, force: bool) -> list[str]:
    """Copy qid prefix files + sample order from the BM25 channel."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        f"sample_qids_seed{DEFAULT_SEED}.txt",
        qids_prefix_filename(query_count),
        "qids_1k.txt",
        "qids_30k.txt",
        "qids_100k.txt",
    ):
        src = bm25_dir / name
        dst = out_dir / name
        if src.exists() and (force or not dst.exists()):
            shutil.copy2(src, dst)
    order_path = out_dir / f"sample_qids_seed{DEFAULT_SEED}.txt"
    if not order_path.exists():
        fallback = out_dir / qids_prefix_filename(query_count)
        if not fallback.exists():
            fallback = bm25_dir / qids_prefix_filename(query_count)
        if fallback.exists():
            shutil.copy2(fallback, order_path)
        else:
            raise FileNotFoundError(
                f"Missing {order_path} and {fallback}; stage sample_qids_seed{DEFAULT_SEED}.txt or "
                f"{qids_prefix_filename(query_count)} to S3."
            )
    return read_qids(order_path)[:query_count]


def _make_dense_searcher(
    *,
    backend: str,
    search_index: str,
    ef_search: int,
) -> Any:
    """Return a searcher object with ``.search(query, k)`` and ``.doc(docid)``."""
    if backend == "faiss":
        with pyserini_import():
            from pyserini.encode import AutoQueryEncoder
            from pyserini.search.faiss import FaissSearcher

        encoder = AutoQueryEncoder(
            encoder_dir=BGE_ENCODER,
            device="cpu",
            prefix=BGE_QUERY_PREFIX,
            l2_norm=True,
        )
        return require_prebuilt_index(FaissSearcher.from_prebuilt_index(search_index, encoder), search_index)

    if backend == "hnsw":
        with pyserini_import():
            from pyserini.search.lucene import LuceneSearcher

        searcher = require_prebuilt_index(LuceneSearcher.from_prebuilt_index(search_index), search_index)
        # Lucene HNSW dense path uses ONNX encoder baked into the index search.
        if hasattr(searcher, "set_ef_search"):
            searcher.set_ef_search(ef_search)
        return searcher

    raise ValueError(f"unknown backend {backend!r}; expected faiss or hnsw")


def write_dataset_meta_dense(
    *,
    out_dir: Path,
    search_index: str,
    text_index: str,
    backend: str,
    seed: int,
    hits: int,
    materialized_count: int,
    n_qrel_rows: int,
    bm25_dir: Path,
) -> None:
    meta = {
        "dataset": out_dir.name,
        "corpus": text_index,
        "source": "scripts/data/setup_msmarco_self_distill_dense.py",
        "fetched": str(date.today()),
        "paired_bm25_channel": str(bm25_dir),
        "sample": {
            "strategy": "deterministic-shuffle-prefix (shared with BM25 channel)",
            "seed": seed,
            "materialized_query_count": materialized_count,
            "qid_order_file": f"sample_qids_seed{seed}.txt",
            "note": "Same qid prefix as msmarco-train-selfdistill-seed42; only first stage differs.",
        },
        "first_stage": {
            RETRIEVER_TOKEN: {
                "retriever": "dense",
                "model": "BgeBaseEn15",
                "backend": "faiss-flat" if backend == "faiss" else "faiss-hnsw",
                "encoder": BGE_ENCODER,
                "encoder_class": "auto",
                "query_prefix": BGE_QUERY_PREFIX,
                "l2_norm": True,
                "search_index": search_index,
                "text_index": text_index,
                "hits": hits,
                "ef_search": 1000 if backend == "hnsw" else None,
                "note": (
                    "BGE dense first stage for MS MARCO train self-distill. "
                    "Exact flat Faiss for reproducibility; HNSW acceptable for training candidates."
                ),
            }
        },
        "search_indexes": {RETRIEVER_TOKEN: search_index},
        "topics": {"source": f"copied from {bm25_dir}/topics.tsv", "n": materialized_count},
        "qrels": {
            "source": f"copied/regenerated from {bm25_dir}",
            "n_rows": n_qrel_rows,
            "note": "Self-distill teacher configs set eval.strict_qrels_filter=false.",
        },
    }
    with open(out_dir / "dataset_meta.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(meta, f, sort_keys=False, default_flow_style=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--query-count", type=int, default=30000)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--bm25-dir", type=Path, default=DEFAULT_BM25_DIR, help="Source for qids/topics/qrels.")
    p.add_argument("--hits", type=int, default=DEFAULT_HITS)
    p.add_argument(
        "--backend",
        choices=("faiss", "hnsw"),
        default="faiss",
        help="faiss=exact flat (default, robust); hnsw=Lucene HNSW (fragile in Python loop).",
    )
    p.add_argument("--search-index", default=None, help="Override pyserini dense search index id.")
    p.add_argument("--text-index", default=DEFAULT_TEXT_INDEX)
    p.add_argument("--ef-search", type=int, default=1000, help="HNSW efSearch (hnsw backend only).")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main() -> int:  # noqa: C901
    args = parse_args()
    if args.query_count <= 0:
        print("[msmarco-selfdistill-dense][FATAL] --query-count must be positive", file=sys.stderr)
        return 2

    search_index = args.search_index or (DEFAULT_FAISS_INDEX if args.backend == "faiss" else DEFAULT_HNSW_INDEX)
    out_dir: Path = args.out_dir
    bm25_dir: Path = args.bm25_dir

    try:
        with pyserini_import():
            from pyserini.search.lucene import LuceneSearcher
    except Exception as e:
        print(f"[msmarco-selfdistill-dense][FATAL] pyserini import failed: {e}", file=sys.stderr)
        return 2

    if not (bm25_dir / "topics.tsv").exists():
        print(f"[msmarco-selfdistill-dense][FATAL] missing {bm25_dir}/topics.tsv", file=sys.stderr)
        return 2

    target_qids = _copy_prefix_files(bm25_dir=bm25_dir, out_dir=out_dir, query_count=args.query_count, force=args.force)
    topics = _load_topics_tsv(bm25_dir / "topics.tsv")
    missing = [q for q in target_qids if q not in topics]
    if missing:
        print(f"[msmarco-selfdistill-dense][FATAL] {len(missing)} qids missing from topics.tsv", file=sys.stderr)
        return 2

    fixture_path = out_dir / "fixture.jsonl"
    run_path = out_dir / RUN_NAME
    topics_path = out_dir / "topics.tsv"
    qrels_path = out_dir / "qrels.txt"

    print(f"[msmarco-selfdistill-dense] out_dir       = {out_dir}", flush=True)
    print(f"[msmarco-selfdistill-dense] bm25_dir      = {bm25_dir}", flush=True)
    print(f"[msmarco-selfdistill-dense] query_count   = {args.query_count}", flush=True)
    print(f"[msmarco-selfdistill-dense] backend       = {args.backend}", flush=True)
    print(f"[msmarco-selfdistill-dense] search_index  = {search_index}", flush=True)
    print(f"[msmarco-selfdistill-dense] text_index    = {args.text_index}", flush=True)

    write_topics_tsv(topics_path, topics, target_qids)
    if bm25_dir.joinpath("qrels.txt").exists():
        shutil.copy2(bm25_dir / "qrels.txt", qrels_path)
        n_qrel_rows = sum(1 for _ in open(qrels_path, encoding="utf-8"))
    else:
        import ir_datasets

        ds = ir_datasets.load("msmarco-passage/train")
        n_qrel_rows = write_ir_qrels_for_qids(ds, set(target_qids), qrels_path)

    dense_searcher = _make_dense_searcher(backend=args.backend, search_index=search_index, ef_search=args.ef_search)
    text_searcher = require_prebuilt_index(LuceneSearcher.from_prebuilt_index(args.text_index), args.text_index)

    class _DenseTextSearcher:
        """Adapter: dense rank + Lucene doc text lookup."""

        def search(self, query: str, k: int):
            return dense_searcher.search(query, k=k)

        def doc(self, docid: str):
            return text_searcher.doc(docid)

    t0 = datetime.now(timezone.utc)
    written, skipped = append_materialized_qids(
        qids=target_qids,
        topics=topics,
        searcher=_DenseTextSearcher(),
        fixture_path=fixture_path,
        run_path=run_path,
        hits=args.hits,
        force=args.force,
    )
    elapsed = (datetime.now(timezone.utc) - t0).total_seconds()
    materialized = len(fixture_qids(fixture_path))

    write_dataset_meta_dense(
        out_dir=out_dir,
        search_index=search_index,
        text_index=args.text_index,
        backend=args.backend,
        seed=DEFAULT_SEED,
        hits=args.hits,
        materialized_count=materialized,
        n_qrel_rows=n_qrel_rows,
        bm25_dir=bm25_dir,
    )

    print(
        f"[msmarco-selfdistill-dense] wrote={written} skipped_existing={skipped} "
        f"materialized={materialized} elapsed_s={elapsed:.1f}",
        flush=True,
    )
    print(f"[msmarco-selfdistill-dense] fixture      = {fixture_path}", flush=True)
    print(f"[msmarco-selfdistill-dense] run          = {run_path}", flush=True)

    # Guard against silent empty retrieval (e.g. HNSW LuceneSearcher without dense mode).
    n_zero = 0
    n_checked = 0
    with open(fixture_path, encoding="utf-8") as f:
        for line in f:
            if n_checked >= 200:
                break
            rec = json.loads(line)
            n_checked += 1
            if not rec.get("passages"):
                n_zero += 1
    if n_checked and n_zero == n_checked:
        print(
            "[msmarco-selfdistill-dense][FATAL] all sampled fixture rows have 0 passages — "
            "dense retrieval did not run (try --backend faiss).",
            file=sys.stderr,
        )
        return 2
    if n_zero > n_checked // 2:
        print(
            f"[msmarco-selfdistill-dense][WARN] {n_zero}/{n_checked} sampled rows have 0 passages",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
