#!/usr/bin/env python
"""Materialize one semantically named appendix study."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from presentation_dependence.reproduction.errors import ReproductionError
from presentation_dependence.reproduction.study_materialization import (
    MATERIALIZERS,
    materialize_study_condition,
)


def parse_args(
    argv: list[str] | None = None,
) -> str:
    """Parse one canonical semantic study."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", choices=sorted(MATERIALIZERS))
    args = parser.parse_args(argv)
    return str(args.study)


def main(argv: list[str] | None = None) -> int:
    """Dispatch to one canonical study materializer."""
    study = parse_args(argv)
    project_root = Path(__file__).resolve().parents[1]
    try:
        result = materialize_study_condition(project_root, study)
    except ReproductionError as exc:
        print(f"[materialize-study][ERROR] {exc}", file=sys.stderr)
        return 2
    print(f"Materialized {result['config_count']} configs and {result['sweep']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
