#!/usr/bin/env python3
"""Reduce canonical Table 2 Climate-FEVER verdict-flip rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from presentation_dependence.reproduction.analysis_inputs import (
    VARIANT_ALIASES,
    load_per_consumer,
    summarize_seeded_metric,
)

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "build" / "reproduction" / "analysis" / "table2_completion"
REPORT_DIR = ROOT / "build" / "reproduction" / "reporting" / "table2_completion"
OUT_JSON = OUT_DIR / "verdict_flip_multiseed.json"
OUT_DOC = REPORT_DIR / "verdict_flip_multiseed.md"
SEEDS = (42, 43, 44)
ARMS = ("k1sft", "k10sft", "debiasfirst", "posaug", "ocsft")
CANONICAL = ROOT / "build/reproduction/multi-document-qa/downstream-eval/per_consumer.json"


def analyze_canonical(path: Path) -> dict[str, Any]:
    """Reduce reproduction verdict-consumer rows."""
    summary = summarize_seeded_metric(
        load_per_consumer(path),
        "mean_verdict_flip_rate",
        consumer="verdict-reader",
        variants=set(VARIANT_ALIASES),
    )
    per_seed = {VARIANT_ALIASES[variant]: values for variant, values in summary["per_seed"].items()}
    across_seed = {VARIANT_ALIASES[variant]: values for variant, values in summary["across_seed"].items()}
    missing = [
        {
            "arm": arm,
            "seed": seed,
            "exp_id": f"canonical:{arm}:seed{seed}:climate-fever",
        }
        for arm in ARMS
        for seed in SEEDS
        if str(seed) not in per_seed.get(arm, {})
    ]
    return {
        "analysis": "Table 2 Climate-FEVER verdict flip, seeds 42-44",
        "status": "blocked" if missing else "complete",
        "missing": missing,
        "protocol": {
            "source": str(path),
            "seed_aggregation": "mean and sample SD over seeds 42-44",
        },
        "per_seed": per_seed,
        "across_seed": across_seed,
    }


def render(payload: dict[str, Any]) -> str:
    """Render either the missing-row receipt or the final values."""
    lines = [
        "# Verdict flip bridge at three seeds",
        "",
        f"Status: **{payload['status']}**.",
        "",
    ]
    if payload["missing"]:
        lines.extend(
            [
                "The following canonical rows are missing:",
                "",
            ]
        )
        lines.extend(f"- `{row['exp_id']}`" for row in payload["missing"])
        lines.append("")
        return "\n".join(lines)

    labels = {
        "k1sft": "K=1 SFT",
        "k10sft": "K=10 SFT",
        "debiasfirst": "DebiasFirst",
        "posaug": "Shuffled-view augmentation",
        "ocsft": "OC-SFT",
    }
    lines.extend(
        [
            "| Variant | Seed 42 | Seed 43 | Seed 44 | Verdict flip mean +/- SD |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for arm in ARMS:
        seeds = payload["per_seed"][arm]
        summary = payload["across_seed"][arm]
        lines.append(
            f"| {labels[arm]} | {seeds['42']:.4f} | {seeds['43']:.4f} | "
            f"{seeds['44']:.4f} | **{summary['mean']:.4f} +/- "
            f"{summary['sample_sd']:.4f}** |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    """Write the canonical verdict reduction."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, default=CANONICAL)
    args = parser.parse_args()

    payload = analyze_canonical(args.canonical)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    OUT_DOC.write_text(render(payload), encoding="utf-8")
    print(
        json.dumps(
            {"status": payload["status"], "missing": payload["missing"]},
            indent=2,
        )
    )
    return 0 if payload["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
