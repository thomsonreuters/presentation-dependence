#!/usr/bin/env python3
"""Reduce canonical Table 2 preference-pair flip rows."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from presentation_dependence.reproduction.analysis_inputs import (
    load_per_consumer,
    summarize_seeded_metric,
)

ROOT = Path(__file__).resolve().parents[2]
TRAINED_ARMS = ("k1_sft", "k10_sft", "position_augmentation", "debiasfirst", "oc_sft")
SURFACE_LABELS = {
    "rewardbench2": "RewardBench-2",
    "nectar": "Nectar",
    "ppe-math": "PPE-MATH",
    "ppe-mmlu-pro": "PPE-MMLU-Pro",
    "rmbench": "RM-Bench",
}
SURFACE_KEYS = tuple(SURFACE_LABELS)
OUT_DIR = ROOT / "build" / "reproduction" / "analysis" / "table2_completion"
REPORT_DIR = ROOT / "build" / "reproduction" / "reporting" / "table2_completion"
OUT_JSON = OUT_DIR / "pair_flip_multiseed.json"
OUT_DOC = REPORT_DIR / "pair_flip_multiseed.md"
CANONICAL = ROOT / "build/reproduction/response-ranking/downstream-eval/per_consumer.json"


def analyze_canonical(path: Path) -> dict[str, Any]:
    """Reduce reproduction response-selection rows for Table 2."""
    rows = load_per_consumer(path)
    datasets = set(SURFACE_KEYS)
    variant_map = {
        "k1-sft": "k1_sft",
        "k10-sft": "k10_sft",
        "shuffled-view-augmentation": "position_augmentation",
        "debias-first": "debiasfirst",
        "oc-sft": "oc_sft",
    }
    trained_counts = Counter(
        (str(row.get("variant")), int(row["training_seed"]), str(row.get("dataset")))
        for row in rows
        if row.get("variant") in variant_map
        and row.get("training_seed") is not None
        and row.get("consumer") == "argmax-response-selection"
        and row.get("dataset") in datasets
    )
    expected_trained = {
        (variant, seed, dataset) for variant in variant_map for seed in (42, 43, 44) for dataset in datasets
    }
    missing_trained = sorted(expected_trained - set(trained_counts))
    duplicate_trained = sorted(key for key, count in trained_counts.items() if count != 1)
    if missing_trained or duplicate_trained:
        raise ValueError(
            f"Incomplete Table 2 pair-flip matrix: missing={missing_trained[:5]} duplicate={duplicate_trained[:5]}"
        )
    off_shelf_counts = Counter(
        str(row.get("dataset"))
        for row in rows
        if row.get("variant") == "off-shelf"
        and row.get("consumer") == "argmax-response-selection"
        and row.get("dataset") in datasets
    )
    if set(off_shelf_counts) != datasets or any(count != 1 for count in off_shelf_counts.values()):
        raise ValueError(f"Incomplete off-shelf pair-flip matrix: counts={dict(sorted(off_shelf_counts.items()))}")
    summary = summarize_seeded_metric(
        rows,
        "preference_pair_flip_rate",
        consumer="argmax-response-selection",
        datasets=datasets,
        variants=set(variant_map),
    )
    by_seed = {variant_map[variant]: values for variant, values in summary["per_seed"].items()}
    across = {
        variant_map[variant]: {
            "mean": values["mean"],
            "sample_sd": values["sample_sd"],
            "n_training_seeds": len(values["by_seed"]),
        }
        for variant, values in summary["across_seed"].items()
    }
    per_collection = {}
    cells = {}
    off_by_collection = {}
    for dataset in SURFACE_KEYS:
        cell = summarize_seeded_metric(
            rows,
            "preference_pair_flip_rate",
            consumer="argmax-response-selection",
            datasets={dataset},
            variants=set(variant_map),
        )
        per_collection[dataset] = {
            variant_map[variant]: {
                "mean": values["mean"],
                "sample_sd": values["sample_sd"],
            }
            for variant, values in cell["across_seed"].items()
        }
        cells[dataset] = {
            variant_map[variant]: {seed: {"mean_preference_pair_flip": value} for seed, value in values.items()}
            for variant, values in cell["per_seed"].items()
        }
        off_value = next(
            float(row["metrics"]["preference_pair_flip_rate"])
            for row in rows
            if row.get("variant") == "off-shelf" and row.get("dataset") == dataset
        )
        off_by_collection[dataset] = {"mean_preference_pair_flip": off_value}
    off_values = [
        float(row["metrics"]["preference_pair_flip_rate"])
        for row in rows
        if row.get("variant") == "off-shelf" and row.get("dataset") in datasets
    ]
    return {
        "analysis": "Table 2 preference-pair flip, five-collection mean",
        "status": "complete",
        "protocol": {
            "source": str(path),
            "collections": list(SURFACE_KEYS),
        },
        "off_shelf": {
            "by_collection": off_by_collection,
            "five_collection_mean": statistics.fmean(off_values),
            "seed_basis": "single evaluation",
        },
        "cells": cells,
        "per_collection_across_seed": per_collection,
        "five_collection_by_seed": by_seed,
        "across_seed": across,
    }


def render(payload: dict[str, Any]) -> str:
    """Render the repository receipt and compact values."""
    lines = [
        "# Table 2 pair flip: five-collection, three-seed result",
        "",
        "The value is the equal mean over RewardBench-2, Nectar, PPE-MATH,",
        "PPE-MMLU-Pro and RM-Bench. Trained rows are mean +/- sample SD over seeds 42-44;",
        "off-the-shelf is a single evaluation because it has no training seed.",
        "",
        "| Variant | Seed 42 | Seed 43 | Seed 44 | Five-collection pair flip |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    off = payload["off_shelf"]["five_collection_mean"]
    lines.append(f"| Off the shelf | - | - | - | {off:.4f} |")
    labels = {
        "k1_sft": "K=1 SFT",
        "k10_sft": "K=10 SFT",
        "position_augmentation": "Shuffled-view augmentation",
        "debiasfirst": "DebiasFirst",
        "oc_sft": "OC-SFT",
    }
    for arm in TRAINED_ARMS:
        seeds = payload["five_collection_by_seed"][arm]
        summary = payload["across_seed"][arm]
        lines.append(
            f"| {labels[arm]} | {seeds['42']:.4f} | {seeds['43']:.4f} | "
            f"{seeds['44']:.4f} | **{summary['mean']:.4f} +/- "
            f"{summary['sample_sd']:.4f}** |"
        )
    lines.extend(
        [
            "",
            "## Per-collection seed-42 check",
            "",
            "| Collection | Off shelf | K=1 SFT | K=10 SFT | Augmentation | DebiasFirst | OC-SFT |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for surface in SURFACE_KEYS:
        row = payload["cells"][surface]
        lines.append(
            f"| {SURFACE_LABELS[surface]} | "
            f"{payload['off_shelf']['by_collection'][surface]['mean_preference_pair_flip']:.3f} | "
            f"{row['k1_sft']['42']['mean_preference_pair_flip']:.3f} | "
            f"{row['k10_sft']['42']['mean_preference_pair_flip']:.3f} | "
            f"{row['position_augmentation']['42']['mean_preference_pair_flip']:.3f} | "
            f"{row['debiasfirst']['42']['mean_preference_pair_flip']:.3f} | "
            f"{row['oc_sft']['42']['mean_preference_pair_flip']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Per-collection three-seed values",
            "",
            "| Collection | K=1 SFT | K=10 SFT | Augmentation | DebiasFirst | OC-SFT |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for surface in SURFACE_KEYS:
        row = payload["per_collection_across_seed"][surface]
        lines.append(
            f"| {SURFACE_LABELS[surface]} | "
            f"{row['k1_sft']['mean']:.3f} +/- {row['k1_sft']['sample_sd']:.3f} | "
            f"{row['k10_sft']['mean']:.3f} +/- {row['k10_sft']['sample_sd']:.3f} | "
            f"{row['position_augmentation']['mean']:.3f} +/- "
            f"{row['position_augmentation']['sample_sd']:.3f} | "
            f"{row['debiasfirst']['mean']:.3f} +/- "
            f"{row['debiasfirst']['sample_sd']:.3f} | "
            f"{row['oc_sft']['mean']:.3f} +/- {row['oc_sft']['sample_sd']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    """Write the canonical pair-flip reduction."""
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
    print(json.dumps(payload["across_seed"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
