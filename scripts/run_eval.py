#!/usr/bin/env python
"""Evaluate a completed presentation_dependence experiment run.

Usage:
    python scripts/run_eval.py -e runs/example-passage-b20-psi/20260422_101530/

    # Or pass just an ID: picks the most recent run under runs/<ID>/:
    python scripts/run_eval.py -e example-passage-b20-psi
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from presentation_dependence.eval import EvalManager
from presentation_dependence.utils.run_paths import default_runs_root, resolve_run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-e",
        "--run",
        type=str,
        required=True,
        help="Path to runs/<ID>/<timestamp>/ OR an experiment ID (latest run under runs/<ID>/).",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=default_runs_root(),
        help="Root directory of runs (default: SLM_RUNS_ROOT, else runs/).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip evaluation if metrics.json already exists.",
    )
    args = parser.parse_args()

    load_dotenv()

    run_dir = resolve_run_dir(args.run, runs_root=args.runs_root)
    manager = EvalManager(run_dir=run_dir, skip_existing=args.skip_existing)
    metrics = manager.run()
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
