#!/usr/bin/env python
"""Materialize deterministic MS MARCO train candidates for self-distill silver.

Build one extendable, non-redundant candidate fixture for the self-distillation
teacher pass:

    data/msmarco-train-selfdistill-seed42/
    ├── sample_qids_seed42.txt     # full deterministic shuffled qid order
    ├── qids_1k.txt                # first 1K qids in that order
    ├── qids_30k.txt               # first 30K qids in that order
    ├── qids_100k.txt              # first 100K qids in that order
    ├── qids_30k_to_100k.txt       # 100K prefix suffix after the first 30K
    ├── topics.tsv                 # materialized qid -> query text
    ├── qrels.txt                  # empty placeholder unless qrels are available
    ├── run.msmarco-v1-passage.bm25-default.selfdistill.txt
    ├── fixture.jsonl              # FixtureLoader input, append-only by qid
    └── dataset_meta.yaml

The 1K pilot, 30K pass, and later 100K+ pass must be nested prefixes of the
same shuffled train query order. That way:

* 1K teacher work is not thrown away when extending to 30K.
* 30K teacher work is not thrown away when extending to 100K.
* Every config can point at the same growing ``fixture.jsonl`` and constrain
  itself with a cheap ``qids_<N>.txt`` prefix file via ``qids_to_run_path``.
* Extension jobs can use explicit delta files (for example
  ``qids_30k_to_100k.txt``) so paper/provenance says exactly which new qids
  were run, without relying on an external job-runner resume.

Usage
-----
Create / extend the 1K pilot:

    uv run python -m scripts.data.setup_msmarco_self_distill --query-count 1000

Extend the same fixture to 30K:

    uv run python -m scripts.data.setup_msmarco_self_distill --query-count 30000

Later extension is identical:

    uv run python -m scripts.data.setup_msmarco_self_distill --query-count 100000

Then create the non-overlapping delta file:

    uv run python scripts/data/make_qid_delta.py \
        --base data/msmarco-train-selfdistill-seed42/qids_30k.txt \
        --target data/msmarco-train-selfdistill-seed42/qids_100k.txt \
        --out data/msmarco-train-selfdistill-seed42/qids_30k_to_100k.txt \
        --sample-order data/msmarco-train-selfdistill-seed42/sample_qids_seed42.txt \
        --target-count 100000

The script skips qids already present in ``fixture.jsonl`` unless ``--force``
is passed. It only uses Pyserini locally; the container reads the prebuilt
``fixture.jsonl`` through ``FixtureLoader``.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections.abc import Mapping
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from presentation_dependence.utils.pyserini_index import pyserini_import, require_prebuilt_index


DEFAULT_OUT_DIR = Path("data/msmarco-train-selfdistill-seed42")
DEFAULT_INDEX = "msmarco-v1-passage"
DEFAULT_TOPICS = "msmarco-passage/train"
DEFAULT_SEED = 42
DEFAULT_HITS = 100
RETRIEVER_TOKEN = "bm25-default"
RUN_NAME = "run.msmarco-v1-passage.bm25-default.selfdistill.txt"


def _topic_text(payload: Any) -> str:
    if isinstance(payload, Mapping):
        text = payload.get("title") or payload.get("text") or payload.get("query") or ""
    else:
        text = str(payload)
    text = " ".join(str(text).split())
    if not text:
        raise ValueError(f"Empty topic payload: {payload!r}")
    return text


def normalize_topics(raw_topics: Mapping[Any, Any]) -> dict[str, str]:
    """Return string-qid -> query text mapping from a Pyserini topics dict."""
    return {str(qid): _topic_text(payload) for qid, payload in raw_topics.items()}


def load_topics(source: str, key: str) -> tuple[dict[str, str], Any | None, str]:
    """Load query topics from ir_datasets (default) or Pyserini.

    Pyserini exposes DL/dev topics but not the MS MARCO passage train query
    set. ``ir_datasets`` does expose ``msmarco-passage/train`` and is already
    in this project, so it is the default source for self-distill training data.
    Returns ``(topics, dataset_obj_or_none, source_description)``.
    """
    if source == "pyserini":
        from pyserini.search import get_topics

        return normalize_topics(get_topics(key)), None, f"pyserini.get_topics({key!r})"

    if source == "ir_datasets":
        import ir_datasets

        ds = ir_datasets.load(key)
        topics = {str(q.query_id): " ".join(str(q.text).split()) for q in ds.queries_iter()}
        return topics, ds, f"ir_datasets.load({key!r}).queries_iter()"

    # auto: try Pyserini first for backwards compatibility with DL/BEIR keys,
    # then fall back to ir_datasets for MS MARCO train.
    try:
        from pyserini.search import get_topics

        return normalize_topics(get_topics(key)), None, f"pyserini.get_topics({key!r})"
    except Exception:
        import ir_datasets

        ds = ir_datasets.load(key)
        topics = {str(q.query_id): " ".join(str(q.text).split()) for q in ds.queries_iter()}
        return topics, ds, f"ir_datasets.load({key!r}).queries_iter()"


def deterministic_qid_order(qids: list[str], *, seed: int) -> list[str]:
    """Stable shuffled qid order. Prefixes of this order define 1K/30K/100K."""
    out = sorted(map(str, qids))
    random.Random(int(seed)).shuffle(out)
    return out


def read_qids(path: Path) -> list[str]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def write_qids(path: Path, qids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for qid in qids:
            f.write(f"{qid}\n")


def qids_prefix_filename(n: int) -> str:
    return f"qids_{n // 1000}k.txt" if n >= 1000 and n % 1000 == 0 else f"qids_{n}.txt"


def fixture_qids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out: set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            out.add(str(rec["qid"]))
    return out


def _doc_text_from_raw(raw: str) -> str:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    if "contents" in parsed:
        return str(parsed["contents"])
    if "text" in parsed:
        text = str(parsed["text"])
        if "title" in parsed and parsed["title"]:
            return f"{parsed['title']} {text}"
        return text
    return raw


def _fetch_doc_text(searcher: Any, docid: str) -> str:
    doc = searcher.doc(docid)
    if doc is None:
        return ""
    return _doc_text_from_raw(doc.raw())


def write_topics_tsv(path: Path, topics: dict[str, str], qids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for qid in qids:
            f.write(f"{qid}\t{topics[qid]}\n")


def append_materialized_qids(
    *,
    qids: list[str],
    topics: dict[str, str],
    searcher: Any,
    fixture_path: Path,
    run_path: Path,
    hits: int,
    force: bool = False,
) -> tuple[int, int]:
    """Append missing qids to fixture/run. Returns (written, skipped_existing)."""
    existing = set() if force else fixture_qids(fixture_path)
    mode = "w" if force else "a"
    fixture_path.parent.mkdir(parents=True, exist_ok=True)
    run_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    with open(fixture_path, mode, encoding="utf-8") as fixture_f, open(run_path, mode, encoding="utf-8") as run_f:
        for qid in qids:
            if qid in existing:
                skipped += 1
                continue
            query = topics[qid]
            hits_ = searcher.search(query, k=hits)
            passages: list[dict[str, Any]] = []
            for rank_i, hit in enumerate(hits_, start=1):
                docid = str(hit.docid)
                text = _fetch_doc_text(searcher, docid)
                if not text:
                    continue
                score = float(hit.score)
                run_f.write(f"{qid} Q0 {docid} {rank_i} {score:.6f} {RETRIEVER_TOKEN}\n")
                passages.append(
                    {
                        "pid": docid,
                        "text": text,
                        "score": score,
                        "rank": rank_i,
                    }
                )
            fixture_f.write(json.dumps({"qid": qid, "query": query, "passages": passages}, ensure_ascii=False) + "\n")
            written += 1
    return written, skipped


def write_empty_qrels_if_missing(path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("", encoding="utf-8")
    return 0


def write_ir_qrels_for_qids(dataset: Any, qids: set[str], path: Path) -> int:
    """Write TREC qrels for selected qids when ir_datasets provides them."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for qrel in dataset.qrels_iter():
            qid = str(qrel.query_id)
            if qid not in qids:
                continue
            f.write(f"{qid} 0 {qrel.doc_id} {int(qrel.relevance)}\n")
            n += 1
    return n


def write_dataset_meta(
    *,
    out_dir: Path,
    topics_key: str,
    topics_source: str,
    index: str,
    seed: int,
    hits: int,
    materialized_count: int,
    full_topic_count: int,
    n_qrel_rows: int,
) -> None:
    meta = {
        "dataset": out_dir.name,
        "corpus": index,
        "source": "scripts/data/setup_msmarco_self_distill.py",
        "fetched": str(date.today()),
        "sample": {
            "strategy": "deterministic-shuffle-prefix",
            "seed": seed,
            "topics_key": topics_key,
            "topics_source": topics_source,
            "full_topic_count": full_topic_count,
            "materialized_query_count": materialized_count,
            "qid_order_file": f"sample_qids_seed{seed}.txt",
            "note": (
                "Use prefix files qids_1k.txt, qids_30k.txt, qids_100k.txt, ... "
                "to extend without changing the sample. Prefix nesting avoids "
                "redundant teacher generation when scaling silver data."
            ),
        },
        "first_stage": {
            RETRIEVER_TOKEN: {
                "retriever": "bm25",
                "variant": "msmarco-tuned",
                "k1": 0.82,
                "b": 0.68,
                "hits": hits,
                "index": index,
                "query_side": False,
                "note": (
                    "LuceneSearcher over msmarco-v1-passage. Pyserini's BM25 "
                    "for this index uses MS MARCO-tuned params (k1=0.82, b=0.68)."
                ),
            }
        },
        "topics": {"source": topics_source, "n": materialized_count},
        "qrels": {
            "source": "ir_datasets qrels when available; otherwise empty placeholder",
            "n_rows": n_qrel_rows,
            "note": "Self-distill teacher configs set eval.strict_qrels_filter=false; qrels are provenance.",
        },
    }
    with open(out_dir / "dataset_meta.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(meta, f, sort_keys=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--query-count", type=int, required=True, help="Prefix size to materialize, e.g. 1000 or 30000.")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--hits", type=int, default=DEFAULT_HITS)
    p.add_argument("--topics", default=DEFAULT_TOPICS, help="Topic source key (default: ir_datasets MS MARCO train).")
    p.add_argument(
        "--topics-source",
        default="ir_datasets",
        choices=("ir_datasets", "pyserini", "auto"),
        help="Where to load query texts from. Use ir_datasets for MS MARCO train.",
    )
    p.add_argument("--index", default=DEFAULT_INDEX, help="Pyserini Lucene prebuilt index.")
    p.add_argument("--force", action="store_true", help="Rewrite fixture/run from scratch for the requested prefix.")
    p.add_argument("--write-prefix", action="append", type=int, default=[1000, 30000], help="Also write qids_<N>.txt.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.query_count <= 0:
        print("[msmarco-selfdistill][FATAL] --query-count must be positive", file=sys.stderr)
        return 2
    if args.hits <= 0:
        print("[msmarco-selfdistill][FATAL] --hits must be positive", file=sys.stderr)
        return 2

    try:
        with pyserini_import():
            from pyserini.search.lucene import LuceneSearcher
    except Exception as e:  # pragma: no cover - env-specific
        print(f"[msmarco-selfdistill][FATAL] pyserini import failed: {e}", file=sys.stderr)
        print("Hint: `uv sync && export JAVA_HOME=...`", file=sys.stderr)
        return 2

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    topics, dataset_obj, topics_source = load_topics(args.topics_source, args.topics)
    if args.query_count > len(topics):
        print(
            f"[msmarco-selfdistill][FATAL] requested {args.query_count} queries but only "
            f"{len(topics)} topics are available from {args.topics!r}",
            file=sys.stderr,
        )
        return 2

    order_path = out_dir / f"sample_qids_seed{args.seed}.txt"
    if order_path.exists() and not args.force:
        order = read_qids(order_path)
        if set(order) - set(topics):
            print(
                f"[msmarco-selfdistill][FATAL] existing qid order {order_path} contains qids "
                "not present in current topic source. Use --force only if you intend to resample.",
                file=sys.stderr,
            )
            return 2
    else:
        order = deterministic_qid_order(list(topics), seed=args.seed)
        write_qids(order_path, order)

    target_qids = order[: args.query_count]
    for n in sorted(set(args.write_prefix + [args.query_count])):
        if n <= len(order):
            write_qids(out_dir / qids_prefix_filename(n), order[:n])

    topics_path = out_dir / "topics.tsv"
    qrels_path = out_dir / "qrels.txt"
    fixture_path = out_dir / "fixture.jsonl"
    run_path = out_dir / RUN_NAME

    print(f"[msmarco-selfdistill] out_dir       = {out_dir}", flush=True)
    print(
        f"[msmarco-selfdistill] topics       = {args.topics} via {args.topics_source} ({len(topics)} available)",
        flush=True,
    )
    print(f"[msmarco-selfdistill] query_count  = {args.query_count}", flush=True)
    print(f"[msmarco-selfdistill] seed         = {args.seed}", flush=True)
    print(f"[msmarco-selfdistill] hits         = {args.hits}", flush=True)
    print(f"[msmarco-selfdistill] fixture      = {fixture_path}", flush=True)

    write_topics_tsv(topics_path, topics, target_qids)
    if dataset_obj is not None and hasattr(dataset_obj, "qrels_iter"):
        n_qrel_rows = write_ir_qrels_for_qids(dataset_obj, set(target_qids), qrels_path)
    else:
        n_qrel_rows = write_empty_qrels_if_missing(qrels_path)

    searcher = require_prebuilt_index(LuceneSearcher.from_prebuilt_index(args.index), args.index)
    t0 = datetime.now(timezone.utc)
    written, skipped = append_materialized_qids(
        qids=target_qids,
        topics=topics,
        searcher=searcher,
        fixture_path=fixture_path,
        run_path=run_path,
        hits=args.hits,
        force=args.force,
    )
    elapsed = (datetime.now(timezone.utc) - t0).total_seconds()
    materialized = len(fixture_qids(fixture_path))

    write_dataset_meta(
        out_dir=out_dir,
        topics_key=args.topics,
        topics_source=topics_source,
        index=args.index,
        seed=args.seed,
        hits=args.hits,
        materialized_count=materialized,
        full_topic_count=len(topics),
        n_qrel_rows=n_qrel_rows,
    )

    print(
        f"[msmarco-selfdistill] wrote={written} skipped_existing={skipped} "
        f"materialized={materialized} elapsed_s={elapsed:.1f}",
        flush=True,
    )
    print(f"[msmarco-selfdistill] qids prefix  = {out_dir / qids_prefix_filename(args.query_count)}", flush=True)
    print(f"[msmarco-selfdistill] run          = {run_path}", flush=True)
    print(f"[msmarco-selfdistill] topics.tsv   = {topics_path}", flush=True)
    print(
        f"[msmarco-selfdistill] qrels.txt    = {qrels_path} ({n_qrel_rows} rows; strict_qrels_filter=false)", flush=True
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
