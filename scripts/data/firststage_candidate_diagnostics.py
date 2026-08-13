#!/usr/bin/env python
r"""Candidate-source diagnostics: recall@100 + gold-rank histogram (no GPU).

CPU-only confound diagnostics for the Stage-1 retriever-robustness panel.
Given a first-stage TREC run + qrels, computes, per (surface x first stage) cell:

* ``recall@k`` of the first stage: the candidate ceiling a reranker can
  reach (a reranker cannot recover gold the first stage dropped).
* the gold-rank histogram in the top-k: where gold sits before reranking.
  If off-shelf base tau-PSI stays substantial even when gold is already
  pre-clustered near the top under a dense first stage, the order-instability
  is attributable to the scorer rather than to BM25 candidate ordering.

Reads only the run file + qrels (no pyserini, no torch, no GPU). Emits one merged
JSON keyed by cell label; safe to call repeatedly to accumulate cells.

Usage -- single cell (matches the hint printed by fetch_dense_sparse_candidates.py)::

    uv run python scripts/data/firststage_candidate_diagnostics.py \\
        --run data/dl19-passage/run.bge-base-en-v1.5.dl19_sorted.txt \\
        --qrels data/dl19-passage/qrels.txt \\
        --label dl19:bge-base-en-v1.5

Usage -- many cells into one report::

    uv run python scripts/data/firststage_candidate_diagnostics.py \\
        --cell data/dl19-passage/run.bge-base-en-v1.5.dl19_sorted.txt:data/dl19-passage/qrels.txt:dl19:bge \\
        --cell data/dl19-passage/run.splade-pp-ed.dl19_sorted.txt:data/dl19-passage/qrels.txt:dl19:splade \\
        --out runs/stage1-firststage-diagnostics/diagnostics.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "runs" / "stage1-firststage-diagnostics" / "diagnostics.json"


def _parse_qrels(path: Path) -> dict[str, set[str]]:
    """Qid -> {gold docid}, gold = rel > 0 (binary + graded both collapse here)."""
    gold: dict[str, set[str]] = defaultdict(set)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            qid, _zero, docid, rel = parts[0], parts[1], parts[2], parts[3]
            try:
                if int(rel) > 0:
                    gold[qid].add(docid)
            except ValueError:
                continue
    return dict(gold)


def _parse_run_ranks(path: Path, k: int) -> dict[str, dict[str, int]]:
    """Qid -> {docid: rank (1-based, ascending)} truncated to top-k."""
    per: dict[str, list[tuple[int, str]]] = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            qid, _q0, docid, rank = parts[0], parts[1], parts[2], parts[3]
            try:
                rk = int(rank)
            except ValueError:
                continue
            per[qid].append((rk, docid))
    out: dict[str, dict[str, int]] = {}
    for qid, entries in per.items():
        entries.sort()
        topk = entries[:k]
        out[qid] = {docid: i + 1 for i, (_rk, docid) in enumerate(topk)}
    return out


def _bucket(rank: int, width: int = 10) -> str:
    lo = ((rank - 1) // width) * width + 1
    hi = lo + width - 1
    return f"{lo:02d}-{hi:02d}"


def compute_cell(run_path: Path, qrels_path: Path, k: int) -> dict[str, Any]:
    gold = _parse_qrels(qrels_path)
    ranks = _parse_run_ranks(run_path, k)

    # Buckets across the top-k, plus a not-in-top-k overflow.
    n_buckets = (k + 9) // 10
    rank_hist: dict[str, int] = {_bucket(b * 10 + 1): 0 for b in range(n_buckets)}
    rank_hist["not_in_topk"] = 0
    best_rank_hist: dict[str, int] = {_bucket(b * 10 + 1): 0 for b in range(n_buckets)}
    best_rank_hist["no_gold_in_topk"] = 0

    per_query_recall: list[float] = []
    n_queries_scored = 0
    n_queries_with_gold = 0
    gold_total = 0
    gold_found = 0
    gold_in_top10 = 0
    cand_counts: list[int] = []

    for qid, q_ranks in ranks.items():
        cand_counts.append(len(q_ranks))
        q_gold = gold.get(qid, set())
        n_queries_scored += 1
        if not q_gold:
            continue
        n_queries_with_gold += 1
        gold_total += len(q_gold)
        found_ranks = [q_ranks[d] for d in q_gold if d in q_ranks]
        gold_found += len(found_ranks)
        gold_in_top10 += sum(1 for r in found_ranks if r <= 10)
        per_query_recall.append(len(found_ranks) / len(q_gold))

        for d in q_gold:
            r = q_ranks.get(d)
            if r is None:
                rank_hist["not_in_topk"] += 1
            else:
                rank_hist[_bucket(r)] += 1
        if found_ranks:
            best_rank_hist[_bucket(min(found_ranks))] += 1
        else:
            best_rank_hist["no_gold_in_topk"] += 1

    macro_recall = sum(per_query_recall) / len(per_query_recall) if per_query_recall else 0.0
    return {
        "k": k,
        "run_path": str(run_path),
        "qrels_path": str(qrels_path),
        f"recall_at_{k}_macro": round(macro_recall, 4),
        f"recall_at_{k}_micro": round(gold_found / gold_total, 4) if gold_total else 0.0,
        "frac_gold_in_top10": round(gold_in_top10 / gold_found, 4) if gold_found else 0.0,
        "n_queries": n_queries_scored,
        "n_queries_with_gold": n_queries_with_gold,
        "gold_total": gold_total,
        "gold_found_in_topk": gold_found,
        "mean_candidates_per_query": round(sum(cand_counts) / len(cand_counts), 1) if cand_counts else 0.0,
        "gold_rank_histogram": rank_hist,
        "best_gold_rank_histogram": best_rank_hist,
    }


def _parse_cell(spec: str) -> tuple[Path, Path, str]:
    """Parse ``run:qrels:label`` (label may itself contain ':')."""
    parts = spec.split(":")
    if len(parts) < 3:
        raise argparse.ArgumentTypeError(f"--cell must be run:qrels:label, got {spec!r}")
    run_path = Path(parts[0])
    qrels_path = Path(parts[1])
    label = ":".join(parts[2:])
    return run_path, qrels_path, label


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cell", action="append", default=[], help="run:qrels:label triple. Repeatable.")
    p.add_argument("--run", default=None, help="Single-cell run path (use with --qrels/--label).")
    p.add_argument("--qrels", default=None, help="Single-cell qrels path.")
    p.add_argument("--label", default=None, help="Single-cell label (e.g. 'dl19:bge-base-en-v1.5').")
    p.add_argument("--k", type=int, default=100, help="Candidate cutoff (default: 100).")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Merged JSON output path.")
    args = p.parse_args()

    cells: list[tuple[Path, Path, str]] = [_parse_cell(c) for c in args.cell]
    if args.run:
        if not (args.qrels and args.label):
            p.error("--run requires --qrels and --label")
        cells.append((Path(args.run), Path(args.qrels), args.label))
    if not cells:
        p.error("provide at least one --cell or a --run/--qrels/--label triple")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {}
    if args.out.exists():
        try:
            report = json.loads(args.out.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            report = {}
    report.setdefault("k", args.k)
    report["generated"] = str(date.today())
    report.setdefault("cells", {})

    for run_path, qrels_path, label in cells:
        if not run_path.exists():
            print(f"[diag][WARN] missing run for {label}: {run_path}")
            continue
        if not qrels_path.exists():
            print(f"[diag][WARN] missing qrels for {label}: {qrels_path}")
            continue
        stats = compute_cell(run_path, qrels_path, args.k)
        report["cells"][label] = stats
        print(
            f"[diag] {label:36s} recall@{args.k}={stats[f'recall_at_{args.k}_macro']:.4f} "
            f"frac_gold_top10={stats['frac_gold_in_top10']:.3f} "
            f"(q={stats['n_queries_with_gold']}, gold_found={stats['gold_found_in_topk']}/{stats['gold_total']})"
        )

    args.out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[diag] wrote {len(report['cells'])} cell(s) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
