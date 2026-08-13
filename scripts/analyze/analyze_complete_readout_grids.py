#!/usr/bin/env python
"""Reduce the complete primary-18 readout and placeholder grids.

Example::

    uv run python -m scripts.analyze.analyze_complete_readout_grids
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
# Importable as `scripts.*` whether this runs as a module or as a file path.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze.analyze_grade_scalarizations import analyze_run

DEFAULT_STUDY_ROOT = ROOT / "build" / "reproduction" / "studies" / "setup-readout-silver" / "complete-readout-grid"
CANONICAL_READOUT_ROOT = ROOT / "build" / "reproduction" / "studies" / "setup-readout-silver" / "readout"
STUDY_PATH = ROOT / "configs" / "studies" / "setup-readout-silver.yaml"
READOUTS = {
    "argmax-grade": "argmax_grade",
    "yes-no-probability": "binary_relevance",
    "expected-grade": "expected_grade",
}
PLACEHOLDERS = ("0", "1", "2", "3")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON mapping: {path}")
    return value


def _metric(metrics: dict[str, Any]) -> float:
    for key in ("mean_ndcg_cut_10", "ndcg_cut_10"):
        if key in metrics:
            return float(metrics[key])
    raise KeyError("metrics artifact has no nDCG@10 value")


def _run_dirs(
    project_root: Path,
    materialization: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Path]:
    expected = {str(row["config"]) for row in materialization["configs"]}
    declared = {str(row["config_id"]): row.get("run_dir") for row in manifest.get("runs") or []}
    if set(declared) != expected:
        raise ValueError(
            "Run manifest does not match materialization: "
            f"missing={sorted(expected - set(declared))}, "
            f"extra={sorted(set(declared) - expected)}"
        )
    blank = sorted(config_id for config_id, run_dir in declared.items() if not run_dir)
    if blank:
        raise ValueError(f"Run manifest has {len(blank)} blank run directories; first={blank[0]}")
    resolved = {}
    for config_id, raw in declared.items():
        path = Path(str(raw))
        path = path if path.is_absolute() else project_root / path
        if not path.is_dir():
            raise FileNotFoundError(f"Missing declared run directory: {path}")
        resolved[config_id] = path
    return resolved


def _winner_counts(
    rows: Iterable[dict[str, Any]],
    conditions: tuple[str, ...],
) -> dict[str, Any]:
    counts = dict.fromkeys(conditions, 0)
    strict = dict.fromkeys(conditions, 0)
    ties = 0
    n_rows = 0
    for row in rows:
        n_rows += 1
        values = {condition: float(row[condition]) for condition in conditions}
        best = max(values.values())
        winners = [condition for condition, value in values.items() if np.isclose(value, best, rtol=0.0, atol=1e-12)]
        if len(winners) > 1:
            ties += 1
        for winner in winners:
            counts[winner] += 1
        if len(winners) == 1:
            strict[winners[0]] += 1
    return {
        "n_cells": n_rows,
        "best_including_ties": counts,
        "strict_wins": strict,
        "tied_cells": ties,
    }


def _means(
    rows: Iterable[dict[str, Any]],
    conditions: tuple[str, ...],
) -> dict[str, float]:
    materialized = list(rows)
    return {condition: float(np.mean([float(row[condition]) for row in materialized])) for condition in conditions}


def _by_model(
    rows: list[dict[str, Any]],
    conditions: tuple[str, ...],
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["model"])].append(row)
    return {
        model: {
            "means": _means(model_rows, conditions),
            "winners": _winner_counts(model_rows, conditions),
        }
        for model, model_rows in sorted(grouped.items())
    }


def _bold(values: dict[str, float], condition: str) -> str:
    rendered = f"{values[condition]:.3f}"
    best = max(values.values())
    return rf"\textbf{{{rendered}}}" if np.isclose(values[condition], best, rtol=0.0, atol=1e-12) else rendered


def _latex_rows(
    rows: list[dict[str, Any]],
    conditions: tuple[str, ...],
) -> list[str]:
    output = []
    for row in rows:
        values = {condition: float(row[condition]) for condition in conditions}
        cells = " & ".join(_bold(values, condition) for condition in conditions)
        output.append(f"{row['paper_name']} & {row['dataset']} & {cells} \\\\")
    return output


def _series(
    rows: list[dict[str, Any]],
    *,
    conditions: tuple[str, ...],
    panel: str,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["model"])].append(row)
    output = []
    for model, model_rows in grouped.items():
        ordered = sorted(model_rows, key=lambda row: str(row["dataset"]))
        for condition in conditions:
            output.append(
                {
                    "name": f"{panel}:{model}:{condition}",
                    "x": [str(row["dataset"]) for row in ordered],
                    "y": [float(row[condition]) for row in ordered],
                }
            )
    return output


def analyze(
    *,
    project_root: Path = ROOT,
    study_root: Path = DEFAULT_STUDY_ROOT,
) -> dict[str, Any]:
    """Build complete grids from explicit new runs plus the frozen readout rows."""
    study = yaml.safe_load(STUDY_PATH.read_text(encoding="utf-8"))
    condition = study["conditions"]["complete-readout-grid"]
    models = condition["models"]
    materialization = _read_json(study_root / "materialization.json")
    manifest = _read_json(study_root / "run_manifest.json")
    run_dirs = _run_dirs(project_root, materialization, manifest)

    placeholder_values: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    readout_rows: list[dict[str, Any]] = []
    for row in materialization["configs"]:
        config_id = str(row["config"])
        run_dir = run_dirs[config_id]
        metrics = _read_json(run_dir / "metrics.json")
        model = str(row["model"])
        dataset = str(row["dataset"])
        placeholder = str(row["placeholder"])
        placeholder_values[(model, dataset)][placeholder] = _metric(metrics)
        if placeholder != str(condition["readout_placeholder"]):
            continue
        scalarizations = analyze_run(run_dir, measures={"ndcg_cut_10"})
        values = {
            readout: float(scalarizations[scalarizer]["mean_ndcg_cut_10"]) for readout, scalarizer in READOUTS.items()
        }
        # Keep the run's canonical expected-grade metric. Recomputing scores from
        # persisted probability vectors can move exact/near-exact ties because
        # JSON round-tripping changes their final floating-point bits.
        values["expected-grade"] = _metric(metrics)
        readout_rows.append(
            {
                "model": model,
                "paper_name": str(models[model]["paper_name"]),
                "dataset": dataset,
                **values,
            }
        )

    placeholder_rows = []
    for (model, dataset), values in sorted(placeholder_values.items()):
        missing = set(PLACEHOLDERS) - set(values)
        if missing:
            raise ValueError(f"Incomplete placeholder row {model}/{dataset}: {sorted(missing)}")
        placeholder_rows.append(
            {
                "model": model,
                "paper_name": str(models[model]["paper_name"]),
                "dataset": dataset,
                **{placeholder: values[placeholder] for placeholder in PLACEHOLDERS},
            }
        )

    readout_conditions = tuple(READOUTS)
    readout_rows.sort(key=lambda row: (str(row["model"]), str(row["dataset"])))
    expected_readout_rows = len(models) * 18
    if len(readout_rows) != expected_readout_rows:
        raise ValueError(f"Expected {expected_readout_rows} readout rows, found {len(readout_rows)}")
    expected_placeholder_rows = len(models) * 18
    if len(placeholder_rows) != expected_placeholder_rows:
        raise ValueError(f"Expected {expected_placeholder_rows} placeholder rows, found {len(placeholder_rows)}")

    payload = {
        "schema_version": 1,
        "status": "complete",
        "protocol": {
            "population": str(study["population"]),
            "datasets": 18,
            "models": list(models),
            "serving_width": int(condition["serving_width"]),
            "quality_metric": str(condition["quality_metric"]),
            "presentations": 1,
            "readout_definition": {
                "argmax-grade": "argmax g of P(g)",
                "yes-no-probability": "P(g >= 2)",
                "expected-grade": "E[g] / 3",
            },
        },
        "readout": {
            "rows": readout_rows,
            "means": _means(readout_rows, readout_conditions),
            "winners": _winner_counts(readout_rows, readout_conditions),
            "by_model": _by_model(readout_rows, readout_conditions),
        },
        "placeholder": {
            "rows": placeholder_rows,
            "means": _means(placeholder_rows, PLACEHOLDERS),
            "winners": _winner_counts(placeholder_rows, PLACEHOLDERS),
            "by_model": _by_model(placeholder_rows, PLACEHOLDERS),
        },
        "provenance": {
            "materialization": str(study_root / "materialization.json"),
            "run_manifest": str(study_root / "run_manifest.json"),
        },
    }
    study_root.mkdir(parents=True, exist_ok=True)
    (study_root / "complete_readout_grids.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    series = {
        "series": [
            *_series(
                readout_rows,
                conditions=readout_conditions,
                panel="readout",
            ),
            *_series(
                placeholder_rows,
                conditions=PLACEHOLDERS,
                panel="placeholder",
            ),
        ],
        "metadata": {
            "status": "complete",
            "source": "complete_readout_grids.json",
            "readout_summary": payload["readout"]["winners"],
            "placeholder_summary": payload["placeholder"]["winners"],
        },
    }
    (study_root / "series.json").write_text(
        json.dumps(series, indent=2) + "\n",
        encoding="utf-8",
    )
    CANONICAL_READOUT_ROOT.mkdir(parents=True, exist_ok=True)
    (CANONICAL_READOUT_ROOT / "series.json").write_text(
        json.dumps(series, indent=2) + "\n",
        encoding="utf-8",
    )
    latex = [
        "% Upper block: argmax grade, P(g >= 2), expected grade.",
        *_latex_rows(readout_rows, readout_conditions),
        "",
        "% Lower block: Grade: 0, 1, 2, 3.",
        *_latex_rows(placeholder_rows, PLACEHOLDERS),
        "",
    ]
    (study_root / "table_readout_rows.tex").write_text(
        "\n".join(latex),
        encoding="utf-8",
    )
    markdown = [
        "# Complete readout and placeholder grids",
        "",
        "Status: **complete**.",
        "",
        f"- Readout means: `{payload['readout']['means']}`",
        f"- Readout wins: `{payload['readout']['winners']}`",
        f"- Readout by model: `{payload['readout']['by_model']}`",
        f"- Placeholder means: `{payload['placeholder']['means']}`",
        f"- Placeholder wins: `{payload['placeholder']['winners']}`",
        f"- Placeholder by model: `{payload['placeholder']['by_model']}`",
        "",
        "The upper block uses three scalarizations of the same saved grade "
        "distribution: argmax grade, P(g >= 2), and expected grade.",
        "",
    ]
    (study_root / "report.md").write_text(
        "\n".join(markdown),
        encoding="utf-8",
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, default=DEFAULT_STUDY_ROOT)
    args = parser.parse_args()
    payload = analyze(study_root=args.study_root)
    print(
        "[complete-readout-grid] "
        f"readout_rows={len(payload['readout']['rows'])} "
        f"placeholder_rows={len(payload['placeholder']['rows'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
