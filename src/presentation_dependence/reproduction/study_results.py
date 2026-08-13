"""Collect explicit study runs into canonical reducer inputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .direct_results import _metric_means, _psi_means
from .common import sha256_file
from .errors import ReproductionError


ARTIFACT_PATHS = {
    "metrics": ("metrics.json",),
    "robustness": ("psi/psi_metrics.json", "psi_metrics.json"),
    "self_consistency": (
        "self_consistency/self_consistency_metrics.json",
        "self_consistency_metrics.json",
    ),
    "matched_variance": ("matched_variance/matched_variance_metrics.json",),
    "pool_perturbation": ("pool_perturbation/pool_perturbation_metrics.json",),
    "context_decomposition": ("context_decomposition/context_decomposition_metrics.json",),
}
FIRST_STAGE_CONDITIONS = ("first-stage-panel", "first-stage-newsurf")
FIRST_STAGE_OUTPUT = Path("build/reproduction/studies/first-stage-transfer/direct-eval/per_dataset.json")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ReproductionError(f"Expected JSON mapping: {path}")
    return value


def _artifact(run_dir: Path, candidates: tuple[str, ...]) -> tuple[Path, dict[str, Any]] | None:
    matches = [run_dir / candidate for candidate in candidates if (run_dir / candidate).is_file()]
    if len(matches) > 1:
        raise ReproductionError(f"Ambiguous canonical artifacts under {run_dir}: {matches}")
    if not matches:
        return None
    return matches[0], _read_json(matches[0])


def _resolve_run_dir(project_root: Path, raw: str) -> Path:
    path = Path(raw)
    resolved = path if path.is_absolute() else project_root / path
    if not resolved.is_dir():
        raise ReproductionError(f"Declared run directory is missing: {resolved}")
    return resolved


def validate_run_manifest(  # noqa: C901
    project_root: Path,
    materialization: Mapping[str, Any],
    run_manifest: Mapping[str, Any],
) -> list[dict[str, str]]:
    """Validate exact one-to-one config/run declarations."""
    materialized_rows = materialization.get("configs") or []
    if not isinstance(materialized_rows, list):
        raise ReproductionError("Study materialization has no configs list")
    metadata = {str(row["config"]): row for row in materialized_rows if isinstance(row, Mapping) and row.get("config")}
    expected = set(metadata)
    if len(metadata) != len(materialized_rows):
        raise ReproductionError("Study materialization has invalid or duplicate config rows")
    rows = run_manifest.get("runs")
    if not isinstance(rows, list):
        raise ReproductionError("Study run manifest has no runs list")
    if not all(isinstance(row, Mapping) for row in rows):
        raise ReproductionError("Study run manifest rows must be mappings")
    declared = []
    for row in rows:
        config_id = row.get("config_id")
        if not isinstance(config_id, str) or not config_id.strip():
            raise ReproductionError("Study run manifest has blank config ID")
        declared.append(config_id.strip())
    if len(declared) != len(set(declared)):
        raise ReproductionError("Study run manifest has duplicate config IDs")
    if set(declared) != expected:
        raise ReproductionError(
            "Study run manifest/config mismatch: "
            f"missing={sorted(expected - set(declared))}, "
            f"extra={sorted(set(declared) - expected)}"
        )
    normalized = []
    for row in rows:
        config_id = str(row["config_id"]).strip()
        run_dir = row.get("run_dir")
        if not isinstance(run_dir, str) or not run_dir.strip():
            raise ReproductionError(f"Run directory not declared for {config_id}")
        expected_bundle = metadata[config_id].get("bundle_id")
        declared_bundle = row.get("bundle_id")
        if expected_bundle is not None:
            if not isinstance(declared_bundle, str) or declared_bundle.strip() != str(expected_bundle):
                raise ReproductionError(
                    f"Bundle declaration mismatch for {config_id}: "
                    f"expected {expected_bundle!r}, got {declared_bundle!r}"
                )
        elif declared_bundle not in (None, ""):
            raise ReproductionError(f"Unexpected bundle declaration for {config_id}: {declared_bundle!r}")
        normalized_row = {
            "config_id": config_id,
            "run_dir": str(_resolve_run_dir(project_root, run_dir.strip())),
        }
        if expected_bundle is not None:
            normalized_row["bundle_id"] = str(expected_bundle)
        normalized.append(normalized_row)
    return normalized


def collect_study_results(
    project_root: Path,
    *,
    study: str,
    condition: str,
    result_root: Path | None = None,
) -> dict[str, Any]:
    """Collect exact declared runs without run discovery or latest selection."""
    root = result_root or (project_root / "build/reproduction/studies" / study / condition)
    materialization_path = root / "materialization.json"
    manifest_path = root / "run_manifest.json"
    if not materialization_path.is_file():
        raise ReproductionError(f"Missing study materialization receipt: {materialization_path}")
    if not manifest_path.is_file():
        raise ReproductionError(f"Missing explicit study run manifest: {manifest_path}")
    materialization = _read_json(materialization_path)
    manifest = _read_json(manifest_path)
    runs = validate_run_manifest(project_root, materialization, manifest)
    metadata = {str(row["config"]): row for row in materialization.get("configs") or []}
    collected = []
    for run in runs:
        run_dir = Path(run["run_dir"])
        artifacts = {}
        for kind, candidates in ARTIFACT_PATHS.items():
            found = _artifact(run_dir, candidates)
            if found is None:
                continue
            path, payload = found
            artifacts[kind] = {
                "path": str(path),
                "sha256": sha256_file(path),
                "payload": payload,
            }
        if not artifacts:
            raise ReproductionError(f"No canonical artifacts found for {run['config_id']}: {run_dir}")
        collected.append(
            {
                **dict(metadata[run["config_id"]]),
                "run_dir": str(run_dir),
                "artifacts": artifacts,
            }
        )
    result = {
        "schema_version": 1,
        "study": study,
        "condition": condition,
        "source_materialization": str(materialization_path),
        "source_run_manifest": str(manifest_path),
        "rows": collected,
    }
    output = root / "results.json"
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _first_stage_direct_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Convert one collected study member to the direct-eval reducer shape."""
    first_stage = row.get("first_stage")
    if not isinstance(first_stage, str) or not first_stage.strip():
        raise ReproductionError(f"First-stage metadata missing for {row.get('config')}")
    artifacts = row.get("artifacts") or {}
    metrics_artifact = artifacts.get("metrics")
    robustness_artifact = artifacts.get("robustness")
    if not isinstance(metrics_artifact, Mapping):
        raise ReproductionError(f"Metrics artifact missing for {row.get('config')}")
    if not isinstance(robustness_artifact, Mapping):
        raise ReproductionError(f"Robustness artifact missing for {row.get('config')}")
    metrics = metrics_artifact.get("payload")
    robustness = robustness_artifact.get("payload")
    if not isinstance(metrics, Mapping) or not isinstance(robustness, Mapping):
        raise ReproductionError(f"Invalid direct-eval artifacts for {row.get('config')}")
    direct_row = {
        "config_id": str(row["config"]),
        "variant": str(row["variant"]),
        "training_seed": row.get("training_seed"),
        "dataset": str(row["dataset"]),
        "first_stage": first_stage.strip(),
        "run_dir": str(row["run_dir"]),
        "n_queries": int(metrics["n_queries"]),
        "metrics": _metric_means(metrics),
        "robustness": _psi_means(robustness),
    }
    if row.get("bundle_id") is not None:
        direct_row["bundle_id"] = str(row["bundle_id"])
    for key in ("checkpoint_uri", "checkpoint_ref", "checkpoint_artifact_sha256"):
        if row.get(key) is not None:
            direct_row[key] = row[key]
    return direct_row


def collect_first_stage_transfer_results(
    project_root: Path,
) -> dict[str, Any]:
    """Merge both explicit first-stage conditions into one reducer input."""
    rows = []
    sources = []
    for condition in FIRST_STAGE_CONDITIONS:
        collected = collect_study_results(
            project_root,
            study="first-stage-transfer",
            condition=condition,
        )
        rows.extend(_first_stage_direct_row(row) for row in collected["rows"])
        sources.append(
            {
                "condition": condition,
                "materialization": collected["source_materialization"],
                "run_manifest": collected["source_run_manifest"],
            }
        )
    rows.sort(
        key=lambda row: (
            row["first_stage"],
            row["dataset"],
            row["variant"],
            row["training_seed"] if row["training_seed"] is not None else -1,
            row["config_id"],
        )
    )
    config_ids = [row["config_id"] for row in rows]
    if len(config_ids) != len(set(config_ids)):
        raise ReproductionError("First-stage conditions produced duplicate config IDs")
    result = {
        "schema_version": 1,
        "study": "first-stage-transfer",
        "stage": "direct-eval",
        "conditions": list(FIRST_STAGE_CONDITIONS),
        "sources": sources,
        "per_dataset": rows,
    }
    destination = project_root / FIRST_STAGE_OUTPUT
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result
