"""Resolve explicit evidence-source bindings for stage collectors."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import sha256_file
from .errors import ReproductionError


BINDINGS_ROOT = Path("build/reproduction/evidence/source-bindings")


def default_bindings_path(project_root: Path, task: str, stage: str) -> Path:
    """Return the generated binding path for one task/stage."""
    return project_root / BINDINGS_ROOT / task / f"{stage}.json"


def load_stage_bindings(
    project_root: Path,
    task: str,
    stage: str,
    *,
    path: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Load explicit bindings keyed by canonical config ID."""
    source = path or default_bindings_path(project_root, task, stage)
    if not source.is_absolute():
        source = project_root / source
    if not source.is_file():
        return {}
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ReproductionError(f"Unsupported source-binding schema: {source}")
    rows = payload.get("bindings")
    if not isinstance(rows, list):
        raise ReproductionError(f"Source bindings have no rows: {source}")
    indexed = {str(row["config_id"]): dict(row) for row in rows}
    if len(indexed) != len(rows):
        raise ReproductionError(f"Duplicate source-binding config IDs: {source}")
    return indexed


def require_bindings_or_discovery(
    bindings: Mapping[str, Mapping[str, Any]],
    *,
    discover_latest: bool,
    task: str,
    stage: str,
) -> None:
    """Fail with an actionable message when no collection source is declared."""
    if not bindings and not discover_latest:
        raise ReproductionError(
            f"No explicit bindings for {task}/{stage}. Write a binding file "
            f"at build/reproduction/evidence/source-bindings/{task}/"
            f"{stage}.json, or pass --discover-latest for an intentional "
            "legacy scan."
        )


def reject_unknown_bindings(
    bindings: Mapping[str, Mapping[str, Any]],
    expected_ids: set[str],
    *,
    task: str,
    stage: str,
) -> None:
    """Reject binding rows that cannot belong to the canonical stage plan."""
    extra = sorted(set(map(str, bindings)) - expected_ids)
    if extra:
        raise ReproductionError(
            f"Source bindings for {task}/{stage} contain {len(extra)} unknown config IDs: {extra[:10]}"
        )


def resolve_bound_run(
    project_root: Path,
    binding: Mapping[str, Any],
    *,
    required_files: Sequence[str],
    verify_hashes: bool = False,
) -> Path:
    """Resolve and validate one explicitly bound local run directory."""
    source = binding.get("source") or {}
    raw = source.get("run_dir")
    if not raw:
        raise ReproductionError(f"Binding {binding.get('unit_id')} has no local run_dir; fetch the retained unit first")
    run_dir = Path(str(raw))
    if not run_dir.is_absolute():
        run_dir = project_root / run_dir
    missing = [relative for relative in required_files if not (run_dir / relative).is_file()]
    if missing:
        raise ReproductionError(
            f"Bound run is missing {missing}: {run_dir}. Re-run the bound job, or point --runs-root at the tree holding it."
        )
    if verify_hashes:
        expected = source.get("artifact_sha256") or {}
        mismatches = [
            relative
            for relative in required_files
            if expected.get(relative) is not None and sha256_file(run_dir / relative) != expected[relative]
        ]
        if mismatches:
            raise ReproductionError(f"Bound run hash mismatch for {mismatches}: {run_dir}")
    return run_dir


def checkpoint_from_binding(
    binding: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Return a selected checkpoint row from a catalog-only binding."""
    source = binding.get("source") or {}
    if source.get("provider") != "checkpoint-catalog":
        return None
    metadata = binding.get("metadata") or {}
    return {
        "variant": metadata["variant"],
        "training_seed": int(metadata["training_seed"]),
        "lambda": metadata.get("lambda"),
        "selected_step": int(source["selected_step"]),
        "selection_metric": source.get("selection_metric"),
        "selection_rule": source.get("selection_rule"),
        "trial": source.get("trial"),
        "checkpoint_uri": source["checkpoint_uri"],
        "semantic_id": binding["config_id"],
        "legacy_id": source.get("historical_exp_id"),
    }
