#!/usr/bin/env python
r"""Fetch topics + qrels + BM25 run for any pyserini-supported dataset.

Produces the layout consumed by presentation_dependence:

    <out>/topics.tsv
    <out>/qrels.txt
    <out>/run.<retriever-token>.<slug>.txt
    <out>/dataset_meta.yaml

The canonical reproduction orchestrator passes ``--run-name`` so the sorted
run is written to the exact filename declared by the dataset population.

Requires Java 21 (pyserini is a JPype wrapper over Lucene). First invocation
of an unseen prebuilt index downloads ~1.5 GB into ~/.cache/pyserini/.

Example — add TREC DL 2020 passage:

    uv run python -m scripts.data.fetch_pyserini_dataset \\
        --topics dl20 \\
        --qrels dl20-passage \\
        --index msmarco-v1-passage \\
        --slug dl20 \\
        --out data/dl20-passage/

Then register the BM25 run as an evaluatable baseline:

    uv run python scripts/import_run_as_baseline.py \\
        -r data/dl20-passage/run.msmarco-v1-passage.bm25-default.dl20_sorted.txt \\
        -q data/dl20-passage/qrels.txt \\
        -i bm25-dl20-baseline
    uv run python scripts/run_eval.py -e bm25-dl20-baseline

Pyserini topic/qrels keys for common datasets:
    | Dataset               | --topics        | --qrels           |
    | --------------------- | --------------- | ----------------- |
    | TREC DL 2019 passage  | dl19-passage    | dl19-passage      |
    | TREC DL 2020 passage  | dl20            | dl20-passage      |
    | BEIR trec-covid       | beir-v1.0.0-trec-covid-test | beir-v1.0.0-trec-covid-test |

Full list: https://github.com/castorini/pyserini/blob/master/docs/prebuilt-indexes.md
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any

import yaml
from pyserini.search import get_qrels, get_topics

from presentation_dependence.utils.dataset_meta import META_FILENAME
from presentation_dependence.utils.trec import ensure_sorted_run_file


def write_topics_tsv(topics_name: str, out_path: Path) -> int:
    topics = get_topics(topics_name)
    if not topics:
        raise RuntimeError(
            f"pyserini.get_topics({topics_name!r}) returned empty. "
            "Check the key against https://github.com/castorini/pyserini/blob/master/pyserini/2cr/"
        )
    with open(out_path, "w") as f:
        for qid, q in topics.items():
            if isinstance(q, dict):
                text = q.get("title") or q.get("text") or q.get("query") or ""
            else:
                text = str(q)
            if not text:
                raise ValueError(f"Empty query text for qid={qid}; topic payload={q!r}")
            f.write(f"{qid}\t{text}\n")
    return len(topics)


def write_qrels_trec(qrels_name: str, out_path: Path) -> int:
    qrels = get_qrels(qrels_name)
    if not qrels:
        raise RuntimeError(f"pyserini.get_qrels({qrels_name!r}) returned empty.")
    n = 0
    with open(out_path, "w") as f:
        for qid, docs in qrels.items():
            for docid, rel in docs.items():
                f.write(f"{qid} 0 {docid} {rel}\n")
                n += 1
    return n


def infer_bm25_params(index: str, *, query_side: bool = False) -> dict[str, Any]:
    """Return the BM25 variant + k1/b that pyserini's BM25 CLI applies
    to `index` for recording in dataset_meta.yaml.

    Args:
        index: pyserini prebuilt index key (e.g. 'msmarco-v1-passage',
            'beir-v1.0.0-nq.flat').
        query_side: if True, the caller passed `--bm25qs`, which applies BM25
            scoring to the query tokens to build the query vector instead of
            using bag-of-words. It changes the variant recorded in the
            metadata but not k1/b, which stay whatever the index implies.
            Worth considering for collections whose queries are long enough
            that term weighting on the query side matters; bag-of-words is
            the convention for short-query benchmarks such as DL19 and BEIR.

    Returns: a dict with `variant`, `k1`, `b`, and `note`, recorded
    per-retriever-token under `first_stage:` in `dataset_meta.yaml`.

    - `msmarco-v1-passage` → MS MARCO-tuned (k1=0.82, b=0.68).
    - BEIR `.flat` indices → vanilla Anserini defaults (k1=0.9, b=0.4).
    - Anything else → unknown; caller should verify by reading pyserini's
      Lucene log line ("setting k1=..., b=...") after running BM25 once.
    """
    if query_side:
        known = index.startswith("msmarco-") or index.startswith("beir-") or ".flat" in index
        k1, b = (0.82, 0.68) if index.startswith("msmarco-") else (0.9, 0.4) if known else (None, None)
        params: dict[str, Any] = {
            "variant": "query-side",
            "k1": k1,
            "b": b,
            "note": (
                "pyserini's `--bm25qs` flag applies BM25 scoring to the query "
                "tokens to form the query vector, then inner-products with the "
                "document vectors, instead of the default bag-of-words. k1/b "
                "are unchanged from what the index implies. Metadata keeps both "
                "variants per dataset and records which one each "
                "retriever-token used."
            ),
        }
        if not known:
            params["_unverified"] = True
        return params
    if index.startswith("msmarco-"):
        return {
            "variant": "msmarco-tuned",
            "k1": 0.82,
            "b": 0.68,
            "note": (
                "pyserini's `--bm25` CLI flag applies MS MARCO-tuned params "
                "(k1=0.82, b=0.68) to msmarco-v1-passage. Matches the "
                "first-stage input in every major DL19/DL20 reranking paper."
            ),
        }
    if index.startswith("beir-") or ".flat" in index:
        return {
            "variant": "vanilla",
            "k1": 0.9,
            "b": 0.4,
            "note": (
                "pyserini applies vanilla Anserini BM25 defaults "
                "(k1=0.9, b=0.4) for BEIR `.flat` indices. Matches BEIR "
                "paper + Anserini 2CR first-stage numbers."
            ),
        }
    return {
        "variant": "unknown",
        "k1": None,
        "b": None,
        "_unverified": True,
        "note": (
            "Unknown BM25 variant for this index — verify by reading "
            "pyserini's Lucene log line 'setting k1=..., b=...' and "
            "updating dataset_meta.yaml by hand."
        ),
    }


def write_dataset_meta(
    out_dir: Path,
    *,
    dataset: str,
    corpus: str,
    topics_key: str,
    qrels_key: str,
    retriever_token: str,
    n_topics: int,
    n_qrel_rows: int,
    hits: int,
    remove_query: bool = False,
    query_side: bool = False,
) -> None:
    """Emit (or merge into) `dataset_meta.yaml` for a pyserini-fetched dataset.

    If the file already exists (e.g. fetching a second retriever into an
    existing dataset dir), merge the new retriever-token entry
    added under `first_stage`, other top-level fields are preserved.
    """
    meta_path = out_dir / META_FILENAME
    existing = {}
    if meta_path.exists():
        with open(meta_path, "r") as f:
            existing = yaml.safe_load(f) or {}

    params = infer_bm25_params(corpus, query_side=query_side)
    fs_all = existing.get("first_stage", {}) or {}
    fs_all[retriever_token] = {
        "retriever": "bm25",
        "variant": params["variant"],
        "k1": params["k1"],
        "b": params["b"],
        "hits": hits,
        "index": corpus,
        "remove_query": remove_query,
        "query_side": query_side,
        "note": params["note"],
        **({"_unverified": True} if params.get("_unverified") else {}),
    }

    merged = {
        "dataset": dataset,
        "corpus": corpus,
        "source": f"scripts/data/fetch_pyserini_dataset.py (topics={topics_key}, qrels={qrels_key})",
        "fetched": str(date.today()),
        "first_stage": fs_all,
        "topics": {"source": f"pyserini.get_topics({topics_key!r})", "n": n_topics},
        "qrels": {"source": f"pyserini.get_qrels({qrels_key!r})", "n_rows": n_qrel_rows},
    }
    if "notes" in existing:
        merged["notes"] = existing["notes"]
    with open(meta_path, "w") as f:
        yaml.safe_dump(merged, f, sort_keys=False)


def run_bm25(
    index: str,
    topics: str,
    out_run: Path,
    hits: int,
    *,
    remove_query: bool = False,
    query_side: bool = False,
) -> None:
    # `--bm25` is bag-of-words (default, used for DL19/20 + BEIR).
    # `--bm25qs` is query-side BM25; see the --query-side-bm25 flag.
    # The two flags are mutually exclusive in pyserini; pick one.
    bm25_flag = "--bm25qs" if query_side else "--bm25"
    cmd = [
        sys.executable,
        "-m",
        "pyserini.search.lucene",
        "--index",
        index,
        "--topics",
        topics,
        "--output",
        str(out_run),
        bm25_flag,
        "--hits",
        str(hits),
    ]
    if remove_query:
        cmd.append("--remove-query")
    print(f"      $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--topics", required=True, help="pyserini topics key (e.g. 'dl20').")
    parser.add_argument("--qrels", required=True, help="pyserini qrels key (e.g. 'dl20-passage').")
    parser.add_argument("--index", required=True, help="pyserini prebuilt index key (e.g. 'msmarco-v1-passage').")
    parser.add_argument("--slug", required=True, help="Short name used in run filename (e.g. 'dl20').")
    parser.add_argument("--out", required=True, help="Output directory (created if missing).")
    parser.add_argument(
        "--run-name",
        default=None,
        help=(
            "Exact final run filename under --out. The fetched run is sorted "
            "into this path. Default: derive run.<retriever-token>.<slug>.txt."
        ),
    )
    parser.add_argument("--hits", type=int, default=1000, help="BM25 cutoff (default: 1000).")
    parser.add_argument(
        "--skip-bm25",
        action="store_true",
        help="Only fetch topics + qrels; skip the Lucene BM25 search.",
    )
    parser.add_argument(
        "--retriever-token",
        default=None,
        help=(
            "Retriever token used in the run filename and as the meta key. "
            "Default: 'bm25-default' for msmarco-* indices (reflects tuned "
            "params), 'bm25' otherwise (vanilla defaults, BEIR convention)."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Re-fetch even if outputs (topics.tsv + qrels.txt + run file + "
            "dataset_meta.yaml) already exist. Default: skip to make the "
            "script safe to re-run as part of a larger driver."
        ),
    )
    parser.add_argument(
        "--remove-query",
        action="store_true",
        help=(
            "Pass `--remove-query` to pyserini's Lucene search. Needed for "
            "ArguAna (counter-argument retrieval, where each query is also a "
            "doc in the corpus) and similar tasks; without it, BM25 ranks "
            "the query document itself #1 and nDCG@10 drops by ~0.10. "
            "Anserini 2CR uses this flag for arguana; default off everywhere "
            "else."
        ),
    )
    parser.add_argument(
        "--query-side-bm25",
        action="store_true",
        help=(
            "Pass `--bm25qs` instead of `--bm25` to pyserini's Lucene search. "
            "Uses BM25 scoring on the query side to form the query vector "
            "rather than the default bag-of-words. Consider it for "
            "collections with long queries, where term weighting on the "
            "query side matters. For DL19/20 and BEIR, stick with the "
            "default bag-of-words."
        ),
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    retriever_token = args.retriever_token or (
        "bm25-default" if args.index.startswith("msmarco-") else "bm25qs" if args.query_side_bm25 else "bm25"
    )

    topics_path = out_dir / "topics.tsv"
    qrels_path = out_dir / "qrels.txt"
    if args.run_name:
        requested_name = Path(args.run_name)
        if (
            requested_name.is_absolute()
            or requested_name.name != args.run_name
            or requested_name.name in {"topics.tsv", "qrels.txt", META_FILENAME}
        ):
            parser.error("--run-name must be a non-reserved filename directly under --out")
        run_path = out_dir / requested_name.name
    else:
        # Short form avoids dots-in-index issues. The corpus/index remains in
        # dataset_meta.yaml rather than being encoded in the filename.
        run_path = out_dir / f"run.{retriever_token}.{args.slug}.txt"
    meta_path = out_dir / "dataset_meta.yaml"

    # Idempotency: if everything's on disk already, skip. Makes this safe
    # to re-run from the canonical reproduction-data orchestrator.
    # `--force` overrides. Each subsequent fetch for a *different*
    # retriever into the same dir still works via `meta_for_run` merge.
    if not args.force and all(p.exists() for p in (topics_path, qrels_path, run_path, meta_path)):
        existing_meta = yaml.safe_load(meta_path.read_text()) or {}
        if retriever_token in (existing_meta.get("first_stage") or {}):
            print(
                f"[skip] {out_dir} already has topics + qrels + run + meta for "
                f"retriever '{retriever_token}'. Use --force to re-fetch."
            )
            return

    print(f"[1/3] topics: {topics_path}")
    n_topics = write_topics_tsv(args.topics, topics_path)
    print(f"      {n_topics} queries")

    print(f"[2/3] qrels: {qrels_path}")
    n_qrels = write_qrels_trec(args.qrels, qrels_path)
    print(f"      {n_qrels} qrel rows")

    if args.skip_bm25:
        print("[3/3] --skip-bm25 set; exiting (no dataset_meta.yaml written).")
        return

    bm25_mode = "query-side (--bm25qs)" if args.query_side_bm25 else "bag-of-words (--bm25)"
    print(
        f"[3/3] BM25 [{bm25_mode}] (index={args.index}, hits={args.hits}, remove_query={args.remove_query}): {run_path}"
    )
    raw_run_path = out_dir / f".{run_path.name}.raw" if args.run_name else run_path
    raw_run_path.unlink(missing_ok=True)
    run_bm25(
        args.index,
        args.topics,
        raw_run_path,
        args.hits,
        remove_query=args.remove_query,
        query_side=args.query_side_bm25,
    )
    sorted_path = ensure_sorted_run_file(raw_run_path)
    if args.run_name:
        sorted_path.replace(run_path)
        raw_run_path.unlink(missing_ok=True)
        sorted_path = run_path
    print(f"      sorted: {sorted_path}")

    # `retriever_token` is already computed above: it drove both the run
    # filename and will drive the meta key, so `meta_for_run()` can find
    # the right first-stage entry later.
    write_dataset_meta(
        out_dir,
        dataset=out_dir.name,
        corpus=args.index,
        topics_key=args.topics,
        qrels_key=args.qrels,
        retriever_token=retriever_token,
        n_topics=n_topics,
        n_qrel_rows=n_qrels,
        hits=args.hits,
        remove_query=args.remove_query,
        query_side=args.query_side_bm25,
    )
    print(f"      meta: {out_dir / 'dataset_meta.yaml'}")

    print()
    print("Next: register as an evaluatable baseline:")
    print(
        f"  uv run python scripts/import_run_as_baseline.py \\\n"
        f"      -r {sorted_path} \\\n"
        f"      -q {qrels_path} \\\n"
        f"      -i bm25-{args.slug}-baseline"
    )
    print(f"  uv run python scripts/run_eval.py -e bm25-{args.slug}-baseline")


if __name__ == "__main__":
    main()
