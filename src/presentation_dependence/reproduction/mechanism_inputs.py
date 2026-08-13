"""Collect canonical mechanism inputs from exact run-manifest trials."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from presentation_dependence.analysis.amortization import harvest

from .errors import ReproductionError


MECHANISM_ROOT = Path("build/reproduction/studies/mechanism-boundary")
MANIFEST_PATH = MECHANISM_ROOT / "run_manifest.json"


def _read_mapping(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ReproductionError(f"Missing explicit mechanism run manifest: {path}") from exc
    if not isinstance(payload, dict):
        raise ReproductionError(f"Mechanism run manifest must be a mapping: {path}")
    return payload


def _trial(project_root: Path, raw: object, label: str) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise ReproductionError(f"Run directory not declared for {label}")
    path = Path(raw)
    resolved = path if path.is_absolute() else project_root / path
    if not resolved.is_dir():
        raise ReproductionError(f"Declared run directory is missing: {resolved}")
    if not (resolved / "metrics.json").is_file():
        raise ReproductionError(f"Manifest must name an exact trial, not an experiment root: {resolved}")
    return resolved


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _provenance(trial: Path) -> dict[str, Any]:
    paths = (
        trial / "metrics.json",
        trial / "psi" / "psi_metrics.json",
        trial / "psi" / "sc_metrics.json",
    )
    return {
        "run_dir": str(trial),
        "artifacts": [{"path": str(path), "sha256": _digest(path)} for path in paths if path.is_file()],
    }


def _anchor(row: Mapping[str, Any]) -> float:
    for key in ("ndcg_k1_permavg", "ndcg_k1_sc", "ndcg_k1_metrics"):
        if row.get(key) is not None:
            return float(row[key])
    raise ReproductionError("Run has no usable K=1 quality anchor")


def _validated_rows(project_root: Path, manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_rows = manifest.get("runs")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ReproductionError("Mechanism run manifest has no non-empty runs list")
    rows = []
    seen = set()
    for index, raw in enumerate(raw_rows):
        if not isinstance(raw, Mapping):
            raise ReproductionError(f"Mechanism manifest row {index} must be a mapping")
        required = ("config_id", "family", "size", "recipe", "dataset")
        missing = [key for key in required if not isinstance(raw.get(key), str) or not str(raw[key]).strip()]
        if missing:
            raise ReproductionError(f"Mechanism manifest row {index} is missing {missing}")
        config_id = str(raw["config_id"])
        if config_id in seen:
            raise ReproductionError(f"Duplicate mechanism config ID: {config_id}")
        seen.add(config_id)
        base_trial = _trial(project_root, raw.get("base_run_dir"), f"{config_id} base")
        trained_trial = _trial(
            project_root,
            raw.get("trained_run_dir"),
            f"{config_id} trained",
        )
        base = harvest(base_trial)
        trained = harvest(trained_trial, require_self_consistency=False)
        if base is None:
            raise ReproductionError(f"Incomplete base mechanism artifacts: {base_trial}")
        if trained is None:
            raise ReproductionError(f"Incomplete trained mechanism artifacts: {trained_trial}")
        rows.append(
            {
                **{key: raw[key] for key in required},
                "ceiling": raw.get("ceiling"),
                "base": base,
                "trained": trained,
                "base_provenance": _provenance(base_trial),
                "trained_provenance": _provenance(trained_trial),
            }
        )
    return rows


def collect_mechanism_inputs(
    project_root: Path,
    *,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Write both mechanism reducer inputs from explicit exact trials."""
    source = manifest_path or project_root / MANIFEST_PATH
    manifest = _read_mapping(source)
    rows = _validated_rows(project_root, manifest)
    observed = {}
    for row in rows:
        dataset = str(row["dataset"])
        values = (
            _anchor(row["base"]),
            float(row["base"]["ndcg_k10_sc"]),
            _anchor(row["trained"]),
        )
        observed[dataset] = max(observed.get(dataset, 0.0), *values)

    cells = []
    metrics = []
    for row in rows:
        base_anchor = _anchor(row["base"])
        trained_anchor = _anchor(row["trained"])
        base_k10 = float(row["base"]["ndcg_k10_sc"])
        ceiling = float(row["ceiling"]) if row["ceiling"] is not None else observed[str(row["dataset"])]
        reducible = base_k10 - base_anchor
        irreducible = ceiling - base_k10
        worst_permutation = row["trained"]["worst_perm_ndcg"]
        if worst_permutation is None:
            raise ReproductionError(
                f"Metric-robustness collection requires mean_worst_perm_ndcg for {row['config_id']}"
            )
        cells.append(
            {
                "config_id": row["config_id"],
                "family": row["family"],
                "size": row["size"],
                "recipe": row["recipe"],
                "dataset": row["dataset"],
                "reducible_R": reducible,
                "irreducible_I": irreducible,
                "total_gap": ceiling - base_anchor,
                "recovered": trained_anchor - base_anchor,
                "ceiling": ceiling,
                "base_run": row["base_provenance"],
                "trained_run": row["trained_provenance"],
            }
        )
        metrics.append(
            {
                "config_id": row["config_id"],
                "family": row["family"],
                "size": row["size"],
                "recipe": row["recipe"],
                "dataset": row["dataset"],
                "metrics": {
                    "ndcg_cut_10": trained_anchor,
                    "tau_psi": row["trained"]["tau_psi"],
                    "worst_permutation_ndcg": worst_permutation,
                },
                "run": row["trained_provenance"],
            }
        )

    common = {
        "schema_version": 1,
        "source_run_manifest": str(source),
    }
    boundary = {**common, "cells": cells}
    progression = {**common, "rows": metrics}
    calibration = {
        **common,
        "metadata": {
            "x": "tau_psi",
            "y": "ndcg_cut_10",
            "group": "recipe",
        },
        "series": [
            {
                "id": recipe,
                "x": [row["metrics"]["tau_psi"] for row in metrics if row["recipe"] == recipe],
                "y": [row["metrics"]["ndcg_cut_10"] for row in metrics if row["recipe"] == recipe],
            }
            for recipe in sorted({str(row["recipe"]) for row in metrics})
        ],
    }
    boundary_path = project_root / MECHANISM_ROOT / "amortization-law/cells.json"
    metrics_path = project_root / MECHANISM_ROOT / "metric-robustness/rows.json"
    calibration_path = project_root / MECHANISM_ROOT / "calibration/series.json"
    for path, payload in (
        (boundary_path, boundary),
        (metrics_path, progression),
        (calibration_path, calibration),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return {
        "study": "mechanism-boundary",
        "source_run_manifest": str(source),
        "cells": len(cells),
        "metric_rows": len(metrics),
        "outputs": [
            str(boundary_path),
            str(metrics_path),
            str(calibration_path),
        ],
    }
