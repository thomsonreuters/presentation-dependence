#!/usr/bin/env python
"""Compute reranking metrics for a silver-data run vs. NIST qrels.

Reports nDCG@10 / nDCG@100 / MAP / R@1000 / RR@10 for:

  1. The first-stage retrieval (the fixture's input order, e.g. BM25 top-100).
  2. The silver teacher's reranked order.
  3. An oracle reranker that sorts by the NIST grade itself (caps at 1.0).

The silver `score_normalized` is used as the rerank score. Ties are broken by
preserving first-stage order (stable sort). Unjudged docs are treated as
non-relevant by ``pytrec_eval`` per TREC convention.

Output:
  - JSON metrics summary printed to stdout.
  - Optional ``--out`` path: writes a JSON file with per-method metrics and
    per-query nDCG@10 deltas for forensics.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import pytrec_eval


def load_qrels(qrels_path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    with qrels_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            qid, _zero, doc_id, grade = parts[0], parts[1], parts[2], parts[3]
            try:
                grade_i = int(grade)
            except ValueError:
                continue
            qrels.setdefault(qid, {})[doc_id] = grade_i
    return qrels


def fixture_order(fixture_path: Path) -> dict[str, list[str]]:
    """Return ``{qid: [pid_in_fixture_order]}`` for the first-stage retrieval."""
    out: dict[str, list[str]] = {}
    with fixture_path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            out[str(rec["qid"])] = [str(p["pid"]) for p in rec["passages"]]
    return out


def silver_scores(labels_path: Path) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = defaultdict(dict)
    with labels_path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            score = rec.get("score_normalized")
            if score is None:
                continue
            out[str(rec["query_id"])][str(rec["doc_id"])] = float(score)
    return dict(out)


def order_to_run(orders: dict[str, list[str]]) -> dict[str, dict[str, float]]:
    """Convert ``{qid: [pid_in_order]}`` to a pytrec_eval-compatible run.

    Higher score = higher rank. We assign ``score = -rank_index`` (offset by a
    large constant) so that ``argsort desc`` reproduces the input order.
    """
    return {qid: {pid: float(len(pids) - i) for i, pid in enumerate(pids)} for qid, pids in orders.items()}


def evaluate(qrels: dict[str, dict[str, int]], run: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    metrics = {"ndcg_cut.10", "ndcg_cut.100", "map", "recall.1000", "recip_rank"}
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, metrics)
    return evaluator.evaluate(run)


def aggregate(eval_out: dict[str, dict[str, float]]) -> dict[str, float]:
    summary: dict[str, list[float]] = defaultdict(list)
    for _qid, vals in eval_out.items():
        for metric, value in vals.items():
            summary[metric].append(value)
    return {m: statistics.mean(v) for m, v in summary.items() if v}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--qrels", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    labels_path = args.run_dir / "silver_labels.jsonl"
    if not labels_path.is_file():
        raise SystemExit(f"silver_labels.jsonl not found at {labels_path}")

    qrels = load_qrels(args.qrels)
    first_stage = fixture_order(args.fixture)
    silver = silver_scores(labels_path)

    # 1) First-stage retrieval (rank by fixture order).
    first_stage_run = order_to_run(first_stage)

    # 2) Silver rerank: stable sort by score (desc), tie-break = first-stage order.
    silver_run: dict[str, dict[str, float]] = {}
    for qid, pids in first_stage.items():
        scored = silver.get(qid, {})
        # Reranked order: score desc, original index asc as tie-break.
        ranked = sorted(
            range(len(pids)),
            key=lambda i: (-scored.get(pids[i], -1.0), i),
        )
        ordered_pids = [pids[i] for i in ranked]
        silver_run[qid] = {pid: float(len(ordered_pids) - i) for i, pid in enumerate(ordered_pids)}

    # 3) Oracle: sort by NIST grade desc (within the candidate pool), tie-break by first-stage.
    oracle_run: dict[str, dict[str, float]] = {}
    for qid, pids in first_stage.items():
        graded = qrels.get(qid, {})
        ranked = sorted(
            range(len(pids)),
            key=lambda i: (-graded.get(pids[i], 0), i),
        )
        ordered_pids = [pids[i] for i in ranked]
        oracle_run[qid] = {pid: float(len(ordered_pids) - i) for i, pid in enumerate(ordered_pids)}

    methods = {
        "bm25_baseline": first_stage_run,
        "silver": silver_run,
        "oracle_nist": oracle_run,
    }
    summaries = {name: aggregate(evaluate(qrels, run)) for name, run in methods.items()}

    print("=== mean metrics across", len(qrels), "queries ===")
    print(f"{'method':<25} {'nDCG@10':>9} {'nDCG@100':>9} {'MAP':>7} {'R@1000':>8} {'RR@10':>8}")
    for name, m in summaries.items():
        print(
            f"{name:<25} {m.get('ndcg_cut_10', 0):>9.4f} {m.get('ndcg_cut_100', 0):>9.4f} "
            f"{m.get('map', 0):>7.4f} {m.get('recall_1000', 0):>8.4f} {m.get('recip_rank', 0):>8.4f}"
        )

    # Per-query nDCG@10 deltas (silver - bm25), useful for forensics.
    bm25_per_q = evaluate(qrels, first_stage_run)
    silver_per_q = evaluate(qrels, silver_run)
    deltas: list[tuple[str, float, float, float]] = []
    for qid in sorted(set(bm25_per_q) | set(silver_per_q)):
        a = bm25_per_q.get(qid, {}).get("ndcg_cut_10", 0.0)
        b = silver_per_q.get(qid, {}).get("ndcg_cut_10", 0.0)
        deltas.append((qid, a, b, b - a))
    deltas_sorted = sorted(deltas, key=lambda d: d[3])
    print("\n=== per-query nDCG@10 (silver - bm25) — bottom 5, top 5 ===")
    for row in deltas_sorted[:5] + deltas_sorted[-5:]:
        print(f"  qid={row[0]:>10}  bm25={row[1]:.3f}  silver={row[2]:.3f}  Δ={row[3]:+.3f}")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "n_queries": len(qrels),
                    "summaries": summaries,
                    "per_query_ndcg10": [
                        {
                            "qid": q,
                            "bm25": a,
                            "silver": b,
                            "delta": d,
                        }
                        for q, a, b, d in deltas
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
