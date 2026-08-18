"""Aggregate downstream results and write their output files."""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

from .errors import DownstreamStageError


def downstream_stage(pipeline: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the pipeline downstream_eval block."""
    value = pipeline.get("downstream_eval")
    if not isinstance(value, Mapping) or "source_stage" not in value:
        raise DownstreamStageError("Pipeline downstream_eval must be a mapping with source_stage")
    return value


def aggregate_downstream_rows(
    rows: list[dict[str, Any]], measures: list[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Equal-average datasets, then summarize across training seeds."""
    grouped: dict[tuple[str, int | None], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["variant"], row["training_seed"])].append(row)

    def mean_values(values: list[Any]) -> float | None:
        finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
        return statistics.fmean(finite) if finite else None

    per_seed = []
    for (variant, seed), group in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1] or -1)):
        per_seed.append(
            {
                "variant": variant,
                "training_seed": seed,
                "datasets": len(group),
                "metrics": {metric: mean_values([row["metrics"].get(metric) for row in group]) for metric in measures},
            }
        )
    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in per_seed:
        by_variant[row["variant"]].append(row)
    aggregate = []
    for variant, group in sorted(by_variant.items()):
        aggregate.append(
            {
                "variant": variant,
                "training_seeds": [row["training_seed"] for row in group if row["training_seed"] is not None],
                "metrics": {
                    metric: {
                        "mean": mean_values([row["metrics"].get(metric) for row in group]),
                        "sample_sd": (
                            statistics.stdev(values)
                            if len(
                                values := [
                                    float(row["metrics"][metric])
                                    for row in group
                                    if row["metrics"].get(metric) is not None
                                    and math.isfinite(float(row["metrics"][metric]))
                                ]
                            )
                            > 1
                            else None
                        ),
                    }
                    for metric in measures
                },
            }
        )
    return per_seed, aggregate


def write_downstream_outputs(
    destination: Path,
    outputs: Mapping[str, Any],
    result: Mapping[str, Any],
) -> None:
    """Write downstream result files and the provenance manifest."""
    destination.mkdir(parents=True, exist_ok=True)
    for key in ("per_consumer", "per_seed", "aggregate"):
        payload = {"schema_version": 1, key: result[key]}
        if result.get("reporting_cohorts"):
            payload["reporting_cohorts"] = result["reporting_cohorts"]
        (destination / str(outputs[key])).write_text(
            json.dumps(
                payload,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    (destination / str(outputs["run_manifest"])).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "reporting_cohorts": result.get("reporting_cohorts", {}),
                "runs": [
                    {
                        key: row[key]
                        for key in (
                            "config_id",
                            "variant",
                            "training_seed",
                            "dataset",
                            "run_dir",
                        )
                    }
                    for row in result["per_consumer"]
                ],
                "controls": [
                    {
                        key: row[key]
                        for key in (
                            "config_id",
                            "variant",
                            "dataset",
                            "consumer",
                            "run_dir",
                            "metrics_file",
                        )
                    }
                    for row in result.get("controls", [])
                ],
                "missing": result["missing"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
