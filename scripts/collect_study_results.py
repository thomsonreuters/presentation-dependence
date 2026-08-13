#!/usr/bin/env python
"""Collect explicitly declared study runs into canonical result JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from presentation_dependence.reproduction.mechanism_inputs import (
    collect_mechanism_inputs,
)
from presentation_dependence.reproduction.study_results import (
    collect_first_stage_transfer_results,
    collect_study_results,
)


def parse_args() -> argparse.Namespace:
    """Parse canonical study collection arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study")
    parser.add_argument("condition", nargs="?")
    parser.add_argument("--result-root", type=Path)
    args = parser.parse_args()
    aggregate_studies = {"first-stage-transfer", "mechanism-boundary"}
    if args.condition is None and args.study not in aggregate_studies:
        parser.error("condition is required unless collecting an aggregate study")
    if args.result_root is not None and args.condition is None:
        parser.error("--result-root requires a single condition")
    return args


def main() -> int:
    """Collect one study condition without run discovery."""
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    if args.study == "mechanism-boundary" and args.condition is None:
        result = collect_mechanism_inputs(project_root)
        print(json.dumps(result, indent=2))
        return 0
    if args.condition is None:
        result = collect_first_stage_transfer_results(project_root)
        print(
            json.dumps(
                {
                    "study": result["study"],
                    "conditions": result["conditions"],
                    "rows": len(result["per_dataset"]),
                },
                indent=2,
            )
        )
        return 0
    result = collect_study_results(
        project_root,
        study=args.study,
        condition=args.condition,
        result_root=args.result_root,
    )
    print(
        json.dumps(
            {
                "study": result["study"],
                "condition": result["condition"],
                "rows": len(result["rows"]),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
