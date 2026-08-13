#!/usr/bin/env python
"""Sanity checks for iterative self-distillation silver labels.

This implements three pre-training gates:

* Check A: same-seed v1-vs-v0 silver Pearson correlation.
* Check B: per-query, per-doc two-permutation variance ratio.
* Check C: silver-as-prediction nDCG@10 quality preservation.

The input files must share the same ``(query_id, doc_id)`` universe. For the
canonical use case, pass the original instruct-teacher K=10 silver as
``--v0-silver`` and the
new v1 OC-SFT-teacher K=2 silver as ``--v1-silver``; the script compares
``score_raw_vector[0]`` for Check A/C and ``score_raw_vector[0:2]`` for Check B.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


Key = tuple[str, str]


def _load_silver(path: Path) -> dict[Key, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"silver file not found: {path}")
    out: dict[Key, dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            rec = json.loads(line)
            key = (str(rec["query_id"]), str(rec["doc_id"]))
            if key in out:
                raise ValueError(f"{path}: duplicate (query_id, doc_id) at line {line_no}: {key}")
            out[key] = rec
    if not out:
        raise ValueError(f"{path}: no silver records found")
    return out


def _score_at(rec: dict[str, Any], index: int) -> float:
    raw = rec.get("score_raw_vector")
    if isinstance(raw, list):
        if 0 <= index < len(raw):
            value = float(raw[index])
        else:
            raise ValueError(
                f"record qid={rec.get('query_id')!r} doc={rec.get('doc_id')!r}: "
                f"score_raw_vector length {len(raw)} has no index {index}"
            )
    elif index == 0 and rec.get("score_continuous") is not None:
        value = float(rec["score_continuous"])
    else:
        raise ValueError(
            f"record qid={rec.get('query_id')!r} doc={rec.get('doc_id')!r}: missing score_raw_vector for nonzero index"
        )
    if not math.isfinite(value):
        raise ValueError(f"non-finite silver score for qid={rec.get('query_id')!r} doc={rec.get('doc_id')!r}")
    return value


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys):
        raise ValueError("pearson inputs must have matching lengths")
    if len(xs) < 2:
        return float("nan")
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    if den_x == 0.0 or den_y == 0.0:
        return float("nan")
    return num / (den_x * den_y)


def _mean_two_perm_variance_by_query(
    records: dict[Key, dict[str, Any]],
    keys: set[Key],
    *,
    index_a: int,
    index_b: int,
) -> float:
    by_qid: dict[str, list[float]] = defaultdict(list)
    for key in keys:
        a = _score_at(records[key], index_a)
        b = _score_at(records[key], index_b)
        mean = (a + b) / 2.0
        by_qid[key[0]].append(((a - mean) ** 2 + (b - mean) ** 2) / 2.0)
    if not by_qid:
        return float("nan")
    per_query = [sum(vals) / len(vals) for vals in by_qid.values() if vals]
    return sum(per_query) / len(per_query) if per_query else float("nan")


def _load_qrels(path: Path) -> dict[str, dict[str, int]]:
    if not path.is_file():
        raise FileNotFoundError(f"qrels file not found: {path}")
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 4:
                raise ValueError(f"{path}: malformed qrels line {line_no}: {line.rstrip()!r}")
            qid, _, doc_id, rel = parts[:4]
            qrels[str(qid)][str(doc_id)] = int(rel)
    return dict(qrels)


def _dcg(rels: list[int], k: int) -> float:
    return sum(float(rel) / math.log2(rank + 2) for rank, rel in enumerate(rels[:k]))


def _mean_ndcg_at_k(
    records: dict[Key, dict[str, Any]],
    keys: set[Key],
    qrels: dict[str, dict[str, int]],
    *,
    score_index: int,
    k: int,
) -> tuple[float, int]:
    by_qid: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for key in keys:
        qid, doc_id = key
        if qid in qrels:
            by_qid[qid].append((doc_id, _score_at(records[key], score_index)))

    vals: list[float] = []
    for qid, scored_docs in by_qid.items():
        judged = qrels.get(qid) or {}
        ideal_rels = sorted((rel for rel in judged.values() if rel > 0), reverse=True)
        ideal = _dcg(ideal_rels, k)
        if ideal <= 0.0:
            continue
        ranked = sorted(scored_docs, key=lambda item: (-item[1], item[0]))
        rels = [int(judged.get(doc_id, 0)) for doc_id, _score in ranked]
        vals.append(_dcg(rels, k) / ideal)
    return (sum(vals) / len(vals), len(vals)) if vals else (float("nan"), 0)


def _finite_leq(value: float, threshold: float) -> bool:
    return math.isfinite(value) and value <= threshold


def _finite_geq(value: float, threshold: float) -> bool:
    return math.isfinite(value) and value >= threshold


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the iterative silver gate."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--v0-silver", required=True, type=Path, help="Original teacher silver JSONL.")
    p.add_argument("--v1-silver", required=True, type=Path, help="OC-SFT-v1 teacher silver JSONL.")
    p.add_argument("--qrels", type=Path, default=None, help="Optional qrels for Check C silver-as-prediction nDCG.")
    p.add_argument("--score-index-a", type=int, default=0, help="Primary raw-vector index. Default: 0.")
    p.add_argument("--score-index-b", type=int, default=1, help="Second raw-vector index for Check B. Default: 1.")
    p.add_argument("--ndcg-k", type=int, default=10, help="nDCG cutoff for Check C. Default: 10.")
    p.add_argument("--min-pearson", type=float, default=0.60, help="Check A warning/fail threshold.")
    p.add_argument("--max-variance-ratio", type=float, default=0.60, help="Check B warning/fail threshold.")
    p.add_argument("--max-quality-drop", type=float, default=0.005, help="Allowed v1-v0 nDCG drop.")
    p.add_argument("--json-out", type=Path, default=None, help="Optional path for a machine-readable report.")
    p.add_argument(
        "--enforce-thresholds",
        action="store_true",
        help="Exit 1 when any available check misses its configured threshold.",
    )
    return p.parse_args()


def main() -> int:
    """Run the silver sanity checks and return a process exit code."""
    args = parse_args()
    try:
        v0 = _load_silver(args.v0_silver)
        v1 = _load_silver(args.v1_silver)
        common = set(v0) & set(v1)
        if not common:
            raise ValueError("silver files have no common (query_id, doc_id) pairs")

        v0_scores = [_score_at(v0[key], args.score_index_a) for key in common]
        v1_scores = [_score_at(v1[key], args.score_index_a) for key in common]
        pearson = _pearson(v0_scores, v1_scores)

        v0_var = _mean_two_perm_variance_by_query(v0, common, index_a=args.score_index_a, index_b=args.score_index_b)
        v1_var = _mean_two_perm_variance_by_query(v1, common, index_a=args.score_index_a, index_b=args.score_index_b)
        variance_ratio = v1_var / v0_var if v0_var > 0.0 else float("nan")

        ndcg_v0 = ndcg_v1 = quality_drop = float("nan")
        ndcg_queries = 0
        if args.qrels is not None:
            qrels = _load_qrels(args.qrels)
            ndcg_v0, ndcg_queries = _mean_ndcg_at_k(v0, common, qrels, score_index=args.score_index_a, k=args.ndcg_k)
            ndcg_v1, ndcg_queries_v1 = _mean_ndcg_at_k(v1, common, qrels, score_index=args.score_index_a, k=args.ndcg_k)
            ndcg_queries = min(ndcg_queries, ndcg_queries_v1)
            quality_drop = ndcg_v0 - ndcg_v1
    except Exception as e:
        print(f"[iter-silver][FATAL] {e}", file=sys.stderr)
        return 2

    check_a_pass = _finite_geq(pearson, args.min_pearson)
    check_b_pass = _finite_leq(variance_ratio, args.max_variance_ratio)
    coverage_pass = not (set(v0) - set(v1)) and not (set(v1) - set(v0))
    check_c_pass = True
    if args.qrels is not None:
        check_c_pass = math.isfinite(quality_drop) and quality_drop <= args.max_quality_drop

    report = {
        "v0_silver": str(args.v0_silver),
        "v1_silver": str(args.v1_silver),
        "common_pairs": len(common),
        "only_v0_pairs": len(set(v0) - set(v1)),
        "only_v1_pairs": len(set(v1) - set(v0)),
        "score_index_a": args.score_index_a,
        "score_index_b": args.score_index_b,
        "pearson": pearson,
        "v0_mean_two_perm_variance": v0_var,
        "v1_mean_two_perm_variance": v1_var,
        "variance_ratio_v1_over_v0": variance_ratio,
        "ndcg_k": args.ndcg_k,
        "ndcg_queries": ndcg_queries,
        "v0_silver_ndcg": ndcg_v0,
        "v1_silver_ndcg": ndcg_v1,
        "quality_drop_v0_minus_v1": quality_drop,
        "checks": {
            "coverage": "PASS" if coverage_pass else "FAIL",
            "correlation": "PASS" if check_a_pass else "FAIL",
            "variance_ratio": "PASS" if check_b_pass else "FAIL",
            "quality": "PASS" if check_c_pass else "FAIL",
        },
    }

    print(f"[iter-silver] v0 records       = {len(v0)}")
    print(f"[iter-silver] v1 records       = {len(v1)}")
    print(f"[iter-silver] common pairs     = {len(common)}")
    print(
        f"[iter-silver] only v0 / v1     = {report['only_v0_pairs']} / {report['only_v1_pairs']} "
        f"({'PASS' if coverage_pass else 'FAIL'})"
    )
    print()
    print(
        f"[iter-silver] Check A Pearson  = {pearson:.6f} "
        f"({'PASS' if check_a_pass else 'FAIL'}; threshold >= {args.min_pearson:.3f})"
    )
    print(
        f"[iter-silver] Check B var      = v0 {v0_var:.8f}  v1 {v1_var:.8f}  "
        f"ratio {variance_ratio:.6f} "
        f"({'PASS' if check_b_pass else 'FAIL'}; threshold <= {args.max_variance_ratio:.3f})"
    )
    if args.qrels is not None:
        print(
            f"[iter-silver] Check C nDCG@{args.ndcg_k} = "
            f"v0 {ndcg_v0:.6f}  v1 {ndcg_v1:.6f}  drop {quality_drop:.6f} "
            f"({'PASS' if check_c_pass else 'FAIL'}; max drop {args.max_quality_drop:.3f}; "
            f"queries {ndcg_queries})"
        )
    else:
        print("[iter-silver] Check C nDCG    = SKIP (pass --qrels to enable)")

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"[iter-silver] json report    = {args.json_out}")

    if args.enforce_thresholds and not (coverage_pass and check_a_pass and check_b_pass and check_c_pass):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
