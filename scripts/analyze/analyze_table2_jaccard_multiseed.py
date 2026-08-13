#!/usr/bin/env python3
"""Reduce canonical Table 2 retained-set Jaccard rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from presentation_dependence.reproduction.analysis_inputs import (
    load_per_consumer,
    summarize_seeded_metric,
)

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "build" / "reproduction" / "analysis" / "table2_completion"
REPORT_DIR = ROOT / "build" / "reproduction" / "reporting" / "table2_completion"
OUT_JSON = OUT_DIR / "jaccard_multiseed.json"
OUT_DOC = REPORT_DIR / "jaccard_multiseed.md"
CANONICAL = ROOT / "build/reproduction/passage-reranking/downstream-eval/per_consumer.json"


def analyze_canonical(path: Path) -> dict[str, Any]:
    """Summarize reproduction passage-downstream rows for Table 2."""
    rows = load_per_consumer(path)
    variants = {
        "k1-sft",
        "k10-sft",
        "shuffled-view-augmentation",
        "debias-first",
        "oc-sft",
    }
    aliases = {
        "k1-sft": "k1sft",
        "k10-sft": "k10sft",
        "shuffled-view-augmentation": "posaug",
        "debias-first": "debiasfirst",
        "oc-sft": "oc_sft",
    }
    tuned = summarize_seeded_metric(
        rows,
        "mean_pairwise_jaccard",
        consumer="retained-set-threshold",
        variants=variants,
    )
    matched = summarize_seeded_metric(
        rows,
        "matched_retention_mean_pairwise_jaccard",
        consumer="retained-set-threshold",
        variants=variants,
    )
    return {
        "analysis": "Table 2 retained-set Jaccard, 18 collections",
        "status": "complete",
        "source": str(path),
        "across_seed": {aliases[variant]: values for variant, values in tuned["across_seed"].items()},
        "across_seed_matched_retention": {
            aliases[variant]: values for variant, values in matched["across_seed"].items()
        },
    }


def render_canonical(payload: dict[str, Any]) -> str:
    """Render a compact Jaccard receipt from reproduction rows."""
    lines = [
        "# Table 2 retained-set Jaccard",
        "",
        f"Reproduction source: `{payload['source']}`",
        "",
        "| Variant | F1-tuned Jaccard | Matched-retention Jaccard |",
        "| --- | ---: | ---: |",
    ]
    labels = {
        "k1sft": "K=1 SFT",
        "k10sft": "K=10 SFT",
        "posaug": "Shuffled-view augmentation",
        "debiasfirst": "DebiasFirst",
        "oc_sft": "OC-SFT",
    }
    for arm, label in labels.items():
        tuned = payload["across_seed"][arm]
        matched = payload["across_seed_matched_retention"][arm]
        lines.append(
            f"| {label} | {tuned['mean']:.4f} +/- {tuned['sample_sd']:.4f} | "
            f"{matched['mean']:.4f} +/- {matched['sample_sd']:.4f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    """Write the canonical seeded Jaccard reduction and report."""
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
    OUT_DOC.write_text(render_canonical(payload), encoding="utf-8")
    print(json.dumps(payload["across_seed"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
