#!/usr/bin/env python3
"""Quantile-match pointwise silver to the batched-teacher shape per query.

For each query, pointwise documents are sorted by score (document id breaks
ties) and assigned the sorted batched-teacher scores. The output
preserves every strict pointwise ordering, exactly matches the batched score
multiset per query, and changes no query/document coverage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "msmarco-train-selfdistill-seed42"
DEFAULT_REPORT = ROOT / "build" / "reproduction" / "analysis" / "staged_context" / "recalibration_build.json"
FILES = (
    (
        "train_29500",
        DATA / "silver_labels_qwen3_4b_k1_seed0_train_29500.jsonl",
        DATA / "silver_labels_qwen3_4b_pointwise_b1_train_29500.jsonl",
        DATA / "silver_labels_qwen3_4b_pointwise_b1_quantile_to_k1_train_29500.jsonl",
    ),
    (
        "heldout_500",
        DATA / "silver_labels_qwen3_4b_k1_seed0_heldout_500.jsonl",
        DATA / "silver_labels_qwen3_4b_pointwise_b1_heldout_500.jsonl",
        DATA / "silver_labels_qwen3_4b_pointwise_b1_quantile_to_k1_heldout_500.jsonl",
    ),
)


def grouped_records(path: Path) -> Iterator[tuple[str, list[dict[str, Any]]]]:
    """Yield raw records grouped by contiguous query id."""
    current: str | None = None
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            qid = str(record["query_id"])
            if current is None:
                current = qid
            if qid != current:
                if qid in seen:
                    raise ValueError(f"{path}: non-contiguous qid {qid}")
                seen.add(current)
                yield current, rows
                current, rows = qid, []
            rows.append(record)
    if current is not None:
        yield current, rows


def quantile_map(
    batched: list[dict[str, Any]], pointwise: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Map one pointwise vector onto the exact sorted batched vector."""
    batched_docs = [str(record["doc_id"]) for record in batched]
    pointwise_docs = [str(record["doc_id"]) for record in pointwise]
    if batched_docs != pointwise_docs:
        raise ValueError("candidate order/universe differs")
    target = np.sort(np.asarray([float(record["score_continuous"]) for record in batched]))
    source = np.asarray([float(record["score_continuous"]) for record in pointwise])
    order = sorted(
        range(len(pointwise)),
        key=lambda index: (source[index], pointwise_docs[index]),
    )
    mapped = np.empty_like(source)
    mapped[np.asarray(order)] = target
    if not np.array_equal(np.sort(mapped), target):
        raise AssertionError("mapped score multiset differs from batched target")
    if np.any(np.diff(mapped[np.asarray(order)]) < 0):
        raise AssertionError("quantile map inverted a pointwise ordering")

    output: list[dict[str, Any]] = []
    for record, score in zip(pointwise, mapped, strict=True):
        transformed = dict(record)
        transformed["score_continuous"] = float(score)
        transformed["score_raw_vector"] = [float(score)]
        transformed["score_grade_vector"] = None
        output.append(transformed)
    rho = float(np.corrcoef(rankdata(source), rankdata(mapped))[0, 1])
    return output, {
        "pointwise_mean": float(source.mean()),
        "batched_target_mean": float(target.mean()),
        "mapped_mean": float(mapped.mean()),
        "pointwise_mapped_spearman": rho,
        "pointwise_sd": float(source.std(ddof=0)),
        "batched_target_sd": float(target.std(ddof=0)),
        "mapped_sd": float(mapped.std(ddof=0)),
    }


def build_one(
    name: str,
    batched_path: Path,
    pointwise_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Build one paired cohort and return verification statistics."""
    batched_iter = grouped_records(batched_path)
    pointwise_iter = grouped_records(pointwise_path)
    sentinel = object()
    n_queries = 0
    n_records = 0
    query_stats: list[dict[str, float]] = []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output:
        while True:
            left = next(batched_iter, sentinel)
            right = next(pointwise_iter, sentinel)
            if left is sentinel or right is sentinel:
                if left is not right:
                    raise ValueError(f"{name}: input query counts differ")
                break
            batched_qid, batched = left
            pointwise_qid, pointwise = right
            if batched_qid != pointwise_qid:
                raise ValueError(f"{name}: qid mismatch {batched_qid} != {pointwise_qid}")
            transformed, stats = quantile_map(batched, pointwise)
            query_stats.append(stats)
            for record in transformed:
                output.write(json.dumps(record, ensure_ascii=False))
                output.write("\n")
            n_queries += 1
            n_records += len(transformed)
    if not query_stats:
        raise ValueError(f"{name}: no records")
    summary: dict[str, Any] = {
        "cohort": name,
        "batched_source": str(batched_path.relative_to(ROOT)),
        "pointwise_source": str(pointwise_path.relative_to(ROOT)),
        "output": str(output_path.relative_to(ROOT)),
        "n_queries": n_queries,
        "n_records": n_records,
        "output_bytes": output_path.stat().st_size,
        "contract": {
            "candidate_parity": True,
            "batched_score_multiset_exact_per_query": True,
            "strict_pointwise_order_inversions": 0,
            "tie_break": "document id ascending",
            "score_grade_vector": ("set to null because remapped scores are not model grade probabilities"),
        },
    }
    for key in query_stats[0]:
        values = np.asarray([row[key] for row in query_stats])
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name}: non-finite verification statistic {key}")
        summary[f"{key}_query_mean"] = float(values.mean())
        summary[f"{key}_query_median"] = float(np.median(values))
    return summary


def parse_args() -> argparse.Namespace:
    """Parse overwrite and report arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing remapped JSONL files.",
    )
    return parser.parse_args()


def main() -> int:
    """Build both cohorts and write a provenance/verification sidecar."""
    args = parse_args()
    for _, _, _, output in FILES:
        if output.exists() and not args.force:
            raise FileExistsError(f"{output} exists; pass --force to rebuild")
    summaries = [build_one(*spec) for spec in FILES]
    payload = {
        "schema_version": 1,
        "method": (
            "per-query rank quantile matching: sorted pointwise documents receive the sorted batched-teacher scores"
        ),
        "summaries": summaries,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.report.relative_to(ROOT)}")
    for summary in summaries:
        print(f"{summary['cohort']}: {summary['n_queries']} queries, {summary['n_records']} records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
