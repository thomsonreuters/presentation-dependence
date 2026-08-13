"""Resolve local experiment run directories."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

# src/presentation_dependence/utils/run_paths.py -> the repository root.
_PACKAGE_ROOT = Path(__file__).resolve().parents[3]


def resolve_trial_name() -> str:
    """Resolve a local-first trial name with a legacy launcher alias."""
    return (
        os.environ.get("SLM_TRIAL_NAME")
        or os.environ.get("TRAINING_JOB_NAME")
        or datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )


def default_runs_root() -> Path:
    """Return the run-artifact root to use when a caller names none.

    ``SLM_RUNS_ROOT`` moves the whole tree off the repo, which matters when
    runs outgrow the checkout's disk. Every entrypoint that mints or reads
    ``<root>/<id>/<timestamp>/`` defaults to this, so redirecting the writer
    without redirecting the readers is not possible. An explicit
    ``--runs-root`` still wins.

    The fallback is anchored on the package rather than returned as a relative
    ``runs``: these entrypoints are run from wherever the caller happens to be,
    and a relative default silently writes a second run tree under that
    directory instead of resuming the one in the repository.
    """
    declared = os.environ.get("SLM_RUNS_ROOT")
    if declared and declared.strip():
        return Path(declared.strip()).expanduser()
    return _PACKAGE_ROOT / "runs"


def resolve_run_dir(exp_or_run: str | Path, runs_root: str | Path = "runs") -> Path:
    """Resolve an explicit run dir or latest run for an experiment id."""
    p = Path(exp_or_run)
    if p.is_dir() and (p / "resolved_config.yaml").exists():
        return p

    candidate = Path(runs_root) / str(exp_or_run)
    if candidate.is_dir():
        subs = sorted((d for d in candidate.iterdir() if d.is_dir()), key=os.path.getmtime)
        if subs:
            return subs[-1]

    raise FileNotFoundError(f"Could not resolve run dir from {exp_or_run!r}")


def resolve_psi_cli_run_dir(
    *,
    run_dir: str | Path | None,
    exp_id: str | Path | None,
    runs_root: str | Path = "runs",
) -> Path:
    """Resolve PSI-tool CLI inputs and normalize a trailing ``psi/`` path."""
    if (run_dir is None) == (exp_id is None):
        raise ValueError("Provide exactly one of run_dir or exp_id.")

    if run_dir is not None:
        p = Path(run_dir)
        if p.name == "psi" and (p.parent / "resolved_config.yaml").is_file():
            return p.parent.resolve()

    target = run_dir if run_dir is not None else exp_id
    if target is None:  # guarded by the exactly-one check above
        raise AssertionError("unreachable: run_dir and exp_id cannot both be None")
    resolved = resolve_run_dir(target, runs_root=runs_root).resolve()
    if resolved.name == "psi":
        return resolved.parent
    return resolved


def selected_variants(stage: Mapping[str, Any]) -> set[str]:
    """Return trained and reference IDs declared by an evaluation stage."""
    variants = stage["variants"]
    references = variants.get("references") or []
    return {
        *map(str, variants.get("trained") or []),
        *(str(row["id"]) if isinstance(row, Mapping) else str(row) for row in references),
    }


def latest_complete_trial(
    runs_root: Path,
    config_id: str,
    *,
    required_files: Sequence[str] = ("metrics.json",),
) -> Path | None:
    """Return the latest trial containing every required relative path."""
    experiment = runs_root / config_id
    if not experiment.is_dir():
        return None
    candidates = [
        trial
        for trial in experiment.iterdir()
        if trial.is_dir() and all((trial / relative).is_file() for relative in required_files)
    ]
    return sorted(candidates)[-1] if candidates else None


def latest_run_artifact(
    runs_root: Path,
    config_id: str,
    patterns: Sequence[str],
) -> Path | None:
    """Return the latest file matching trial-relative glob patterns."""
    experiment = runs_root / config_id
    if not experiment.is_dir():
        return None
    matches = [path for pattern in patterns for path in experiment.glob(pattern) if path.is_file()]
    return sorted(matches)[-1] if matches else None
