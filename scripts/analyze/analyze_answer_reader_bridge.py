#!/usr/bin/env python
"""Reduce canonical answer-reader rows at the 4B QA anchor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from presentation_dependence.reproduction.analysis_inputs import (
    VARIANT_ALIASES,
    load_per_consumer,
    summarize_seeded_metric,
)

SLM_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = SLM_ROOT / "build/reproduction/analysis/answer_reader_bridge/answer_reader_bridge.json"
CANONICAL = SLM_ROOT / "build/reproduction/multi-document-qa/downstream-eval/per_consumer.json"

METRICS = (
    "canonical_em",
    "canonical_f1",
    "mean_answer_flip_rate",
    "gold_in_topk_rate",
    "gold_slot_mean",
    "gold_slot_var",
)
GATE_KEYS = (
    "seed42_posaug_beats_oc_on_flip_with_level_em_f1",
    "all_seeds_posaug_beats_oc_on_flip",
    "all_seeds_k1_beats_oc_on_flip",
    "all_seeds_debiasfirst_beats_oc_on_flip",
    "seed42_posaug_beats_oc_on_flip_all_six_4b_cells",
)


def reduce_canonical(path: Path, k: int) -> dict:
    """Reduce reproduction answer-reader rows."""
    rows = [
        row
        for row in load_per_consumer(path)
        if row.get("consumer") == "answer-reader"
        and row.get("dataset") == "hotpotqa"
        and int(row.get("reader_k", -1)) == k
    ]
    aggregates = {}
    for metric in METRICS:
        summary = summarize_seeded_metric(
            rows,
            metric,
            consumer="answer-reader",
            datasets={"hotpotqa"},
            variants=set(VARIANT_ALIASES),
        )
        for variant, result in summary["across_seed"].items():
            aggregates.setdefault(VARIANT_ALIASES[variant], {})[metric] = {
                "mean": result["mean"],
                "sample_sd": result["sample_sd"],
                "n_seeds": len(result["by_seed"]),
            }
    return {
        "basis": {
            "source": str(path),
            "dataset": "HotpotQA",
            "reader_k": k,
        },
        "aggregates": aggregates,
        "provenance": {
            "canonical_rows": [row["run_dir"] for row in rows],
        },
    }


def main() -> int:
    """Run the canonical answer-reader bridge reduction."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--canonical", type=Path, default=CANONICAL)
    args = parser.parse_args()

    report = reduce_canonical(args.canonical, args.k)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
