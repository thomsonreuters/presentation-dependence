#!/usr/bin/env python
"""Build a `FixtureLoader`-compatible JSONL for silver-data dry runs.

The default source is TREC-DL'19 passage (the standard NIST-judged eval set).
Samples up to ``--candidates-per-query`` NIST-graded passages per query so each
silver call has a grade for post-hoc comparison.

Output schema (one JSON object per line):

    {"qid": "<qid>", "query": "<text>",
     "passages": [{"pid": "<pid>", "text": "<passage text>"}]}

Reads from ``ir_datasets`` without Pyserini or Java.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import ir_datasets


LOGGER = logging.getLogger("build_silver_fixture")


def main() -> None:  # noqa: C901
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="msmarco-passage/trec-dl-2019/judged",
        help="ir_datasets dataset id (must expose queries + qrels + docs).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output JSONL path (will be overwritten).",
    )
    parser.add_argument(
        "--candidates-per-query",
        type=int,
        default=25,
        help="Max passages per query to include.",
    )
    parser.add_argument(
        "--query-count",
        type=int,
        default=None,
        help="Optional cap on number of queries.",
    )
    parser.add_argument(
        "--trec-run",
        type=Path,
        default=None,
        help=(
            "Optional TREC run file. When set, the fixture's candidate pool "
            "for each query becomes the top-N entries from this run (sorted "
            "by rank ASC). NIST grades are still attached when available; "
            "missing grades are emitted as -1 (TREC treats them as 0)."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    rng = random.Random(args.seed)

    LOGGER.info("Loading ir_datasets %s ...", args.dataset)
    ds = ir_datasets.load(args.dataset)

    queries = {q.query_id: q.text for q in ds.queries_iter()}
    LOGGER.info("Loaded %d queries", len(queries))

    qrels_by_qid: dict[str, list[tuple[str, int]]] = {}
    for qrel in ds.qrels_iter():
        qrels_by_qid.setdefault(qrel.query_id, []).append((qrel.doc_id, int(qrel.relevance)))
    LOGGER.info(
        "Loaded qrels for %d queries (%d total grades)",
        len(qrels_by_qid),
        sum(len(v) for v in qrels_by_qid.values()),
    )

    needed_doc_ids: set[str] = set()
    selected_per_qid: dict[str, list[tuple[str, int]]] = {}

    if args.trec_run is not None:
        # Use a TREC run file as the candidate source. Keep up to
        # `--candidates-per-query` entries per qid in the order the run file
        # provides; NIST grades are looked up if present, otherwise -1.
        run_by_qid: dict[str, list[str]] = {}
        with args.trec_run.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 4:
                    continue
                run_qid, _q0, docid, rank = parts[0], parts[1], parts[2], parts[3]
                try:
                    rank_i = int(rank)
                except ValueError:
                    continue
                run_by_qid.setdefault(run_qid, []).append((rank_i, docid))
        qids = sorted(run_by_qid)
        if args.query_count is not None:
            rng.shuffle(qids)
            qids = qids[: args.query_count]
        for qid in qids:
            ranked = sorted(run_by_qid[qid], key=lambda rd: rd[0])
            kept_docs = [doc_id for _r, doc_id in ranked[: args.candidates_per_query]]
            grade_lookup = dict(qrels_by_qid.get(qid, []))
            selected_per_qid[qid] = [(doc_id, grade_lookup.get(doc_id, -1)) for doc_id in kept_docs]
            needed_doc_ids.update(kept_docs)
    else:
        qids = sorted(qrels_by_qid)
        if args.query_count is not None:
            rng.shuffle(qids)
            qids = qids[: args.query_count]

        for qid in qids:
            graded = list(qrels_by_qid[qid])
            rng.shuffle(graded)
            kept = graded[: args.candidates_per_query]
            if not kept:
                continue
            selected_per_qid[qid] = kept
            needed_doc_ids.update(doc_id for doc_id, _ in kept)

    LOGGER.info(
        "Need to fetch %d unique passage texts ...",
        len(needed_doc_ids),
    )
    docstore = ds.docs_store()
    doc_text: dict[str, str] = {}
    for doc_id in needed_doc_ids:
        doc = docstore.get(doc_id)
        if doc is None:
            continue
        doc_text[doc_id] = doc.text

    LOGGER.info("Resolved %d/%d passages.", len(doc_text), len(needed_doc_ids))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_queries = 0
    n_pairs = 0
    with args.out.open("w", encoding="utf-8") as f:
        for qid, kept in selected_per_qid.items():
            if qid not in queries:
                continue
            passages = []
            for doc_id, grade in kept:
                text = doc_text.get(doc_id)
                if text is None:
                    continue
                passages.append({"pid": doc_id, "text": text, "nist_grade": grade})
            if not passages:
                continue
            row = {"qid": qid, "query": queries[qid], "passages": passages}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_queries += 1
            n_pairs += len(passages)

    LOGGER.info(
        "Wrote %s : %d queries, %d pairs (%.1f avg / query).",
        args.out,
        n_queries,
        n_pairs,
        (n_pairs / n_queries) if n_queries else 0.0,
    )


if __name__ == "__main__":
    main()
