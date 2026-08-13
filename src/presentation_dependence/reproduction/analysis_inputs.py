"""Read reproduction outputs for figure and table analysis scripts."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any, Iterable


VARIANT_ALIASES = {
    "k1-sft": "k1sft",
    "k10-sft": "k10sft",
    "debias-first": "debiasfirst",
    "shuffled-view-augmentation": "posaug",
    "oc-sft": "ocsft",
}


def load_per_consumer(path: Path) -> list[dict[str, Any]]:
    """Load per_consumer rows from a downstream output file."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("per_consumer")
    if not isinstance(rows, list):
        raise ValueError(f"{path} has no per_consumer rows")
    return rows


def summarize_seeded_metric(
    rows: Iterable[dict[str, Any]],
    metric: str,
    *,
    consumer: str | None = None,
    datasets: set[str] | None = None,
    variants: set[str] | None = None,
) -> dict[str, Any]:
    """Equal-average datasets per seed, then summarize across seeds."""
    grouped: dict[tuple[str, int], list[float]] = {}
    for row in rows:
        seed = row.get("training_seed")
        variant = str(row.get("variant"))
        if seed is None or (consumer and row.get("consumer") != consumer):
            continue
        if datasets is not None and row.get("dataset") not in datasets:
            continue
        if variants is not None and variant not in variants:
            continue
        value = (row.get("metrics") or {}).get(metric)
        if value is not None:
            grouped.setdefault((variant, int(seed)), []).append(float(value))
    per_seed: dict[str, dict[str, float]] = {}
    for (variant, seed), values in grouped.items():
        per_seed.setdefault(variant, {})[str(seed)] = statistics.fmean(values)
    across_seed = {}
    for variant, seeded in per_seed.items():
        values = [seeded[key] for key in sorted(seeded)]
        across_seed[variant] = {
            "by_seed": seeded,
            "mean": statistics.fmean(values),
            "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
        }
    return {"per_seed": per_seed, "across_seed": across_seed}
