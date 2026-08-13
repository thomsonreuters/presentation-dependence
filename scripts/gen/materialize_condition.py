#!/usr/bin/env python
"""Materialize one canonical study condition.

Every condition delegates entirely to ``reproduction.study_materialization``, so
the tracked study YAML stays the single owner of the arm set. Nothing here reads
run directories or resolved configs: generation must be reproducible from
tracked source alone.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable, Mapping

from presentation_dependence.reproduction.study_materialization import (
    materialize_grade_one,
    materialize_instrument_width,
    materialize_matched_variance,
    materialize_trained_channel_cross,
)


ROOT = Path(__file__).resolve().parents[2]

CONDITIONS: dict[str, Callable[..., Mapping[str, Any]]] = {
    "grade-one-control": materialize_grade_one,
    "instrument-width": materialize_instrument_width,
    "matched-variance": materialize_matched_variance,
    "trained-channel-cross": materialize_trained_channel_cross,
}


def generate(
    condition: str,
    *,
    project_root: Path | None = None,
) -> Mapping[str, Any]:
    """Materialize one condition and return its materialization result."""
    if condition not in CONDITIONS:
        raise KeyError(f"Unknown condition {condition!r}; choose from {', '.join(sorted(CONDITIONS))}")
    materialize = CONDITIONS[condition]
    root = project_root or ROOT
    return materialize(root)


def config_count(result: Mapping[str, Any]) -> int:
    """Return the number of configs a materialization produced."""
    if "config_count" in result:
        return int(result["config_count"])
    return len(result.get("configs", ()))


def main() -> int:
    """Materialize the requested study condition."""
    parser = argparse.ArgumentParser(description="Materialize one canonical study condition.")
    parser.add_argument("condition", choices=sorted(CONDITIONS))
    args = parser.parse_args()
    result = generate(args.condition)
    print(f"Materialized {config_count(result)} {args.condition} configs and {result.get('sweep')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
