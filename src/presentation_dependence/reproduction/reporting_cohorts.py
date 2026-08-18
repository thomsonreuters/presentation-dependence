"""Frozen post-hoc reporting cohorts for reproduction collectors."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


class ReportingCohortError(ValueError):
    """A declared reporting cohort is missing or does not match its contract."""


def load_reporting_qids(
    pipeline: Mapping[str, Any],
    project_root: Path,
    dataset: str,
) -> tuple[set[str], dict[str, Any]] | None:
    """Load a dataset's optional post-hoc reporting cohort.

    Reporting cohorts affect reductions only. They do not change materialized
    evaluation configs or the queries sent through inference.
    """
    declarations = pipeline.get("reporting_cohorts") or {}
    raw = declarations.get(dataset)
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ReportingCohortError(f"reporting_cohorts.{dataset} must be a mapping")

    relative_path = str(raw["qids_path"])
    path = project_root / relative_path
    if not path.is_file():
        raise ReportingCohortError(f"Missing reporting cohort: {path}")
    rows = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    qids = set(rows)
    if len(rows) != len(qids):
        raise ReportingCohortError(f"Reporting cohort contains duplicate qids: {path}")

    expected = int(raw["expected_qids"])
    if len(qids) != expected:
        raise ReportingCohortError(f"Reporting cohort {dataset} has {len(qids)} qids; expected {expected}")
    return qids, {
        "id": str(raw["id"]),
        "qids_path": relative_path,
        "n_queries": expected,
        "application": "post-hoc",
    }


def require_cohort_coverage(
    available_qids: set[str],
    cohort_qids: set[str],
    *,
    source: Path,
) -> None:
    """Require every declared cohort qid to be present in a source artifact."""
    missing = sorted(cohort_qids - available_qids)
    if missing:
        raise ReportingCohortError(f"{source} is missing {len(missing)} reporting-cohort qids; first={missing[0]}")
