#!/usr/bin/env python
"""Plan, materialize, or validate canonical reproduction datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from presentation_dependence.reproduction.data_setup import (
    data_plan,
    run_data_setup,
    validate_population_outputs,
)


TASKS = (
    "all",
    "passage-reranking",
    "multi-document-qa",
    "response-ranking",
)


def parse_args() -> argparse.Namespace:
    """Parse canonical data setup arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "run", "validate"), default="plan")
    parser.add_argument("--task", choices=TASKS, default="all")
    parser.add_argument("--include-internal", action="store_true")
    parser.add_argument(
        "--include-gated",
        action="store_true",
        help=(
            "Allow run to execute access-gated population rows after the "
            "operator has obtained the required licenses. Plan always shows "
            "these rows."
        ),
    )
    return parser.parse_args()


def main() -> int:
    """Run canonical data setup without hiding access-gated steps."""
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.command == "plan":
        print(
            json.dumps(
                data_plan(
                    root,
                    task=args.task,
                    include_internal=args.include_internal,
                    include_gated=args.include_gated,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "run":
        return run_data_setup(
            root,
            task=args.task,
            include_internal=args.include_internal,
            include_gated=args.include_gated,
        )
    tasks = TASKS[1:] if args.task == "all" else (args.task,)
    missing = {task: validate_population_outputs(root, task) for task in tasks}
    print(json.dumps(missing, indent=2, sort_keys=True))
    return 2 if any(missing.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
