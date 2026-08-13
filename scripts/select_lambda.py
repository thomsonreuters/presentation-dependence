#!/usr/bin/env python3
"""Discover OC-SFT trajectories and apply the package lambda-selection rule."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from presentation_dependence.self_distill.selection import (
    LambdaResult,
    parse_training_progress,
    select_heldout_lambda,
)
from presentation_dependence.utils.run_paths import default_runs_root


DEFAULT_RUNS_ROOT = default_runs_root()
DEFAULT_GRID = "050:0.5,100:1.0,200:2.0,300:3.0,400:4.0,500:5.0"


def parse_grid(spec: str) -> dict[str, float]:
    """Parse `code:value` lambda pairs."""
    return {code.strip(): float(value) for pair in spec.split(",") for code, value in [pair.split(":")]}


def parse_progress_jsonl(path: Path, *, metric: str) -> tuple[dict[int, float], float | None]:
    """Compatibility wrapper for legacy checkpoint selector scripts."""
    return parse_training_progress(path, metric=metric)


def find_latest_local_progress(runs_root: Path, config: str) -> Path | None:
    """Return the newest local training trajectory for a config."""
    base = runs_root / config
    if not base.is_dir():
        return None
    paths = [
        trial / "student/progress.jsonl" for trial in base.iterdir() if (trial / "student/progress.jsonl").is_file()
    ]
    return max(paths, key=lambda path: path.stat().st_mtime) if paths else None


def _gather_local(
    template: str,
    grid: dict[str, float],
    *,
    runs_root: Path,
    metric: str,
) -> list[LambdaResult]:
    results = []
    for code, lam in grid.items():
        progress = find_latest_local_progress(runs_root, template.format(code=code))
        result = LambdaResult(lam=lam, code=code)
        if progress is None:
            result.missing = True
        else:
            result.traj, result.min_stdev = parse_training_progress(progress, metric=metric)
            result.epoch = int(progress.parent.parent.stat().st_mtime)
        results.append(result)
    return results


def parse_args() -> argparse.Namespace:
    """Parse selector discovery and protocol arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-template", required=True)
    parser.add_argument("--grid", default=DEFAULT_GRID)
    parser.add_argument("--se", type=float, default=0.015)
    parser.add_argument("--conv-step", type=int, default=1000)
    parser.add_argument("--collapse", type=float, default=0.02)
    parser.add_argument("--min-final-step", type=int, default=2000)
    parser.add_argument("--quality-floor", type=float, default=0.40)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--metric", default="student_eval/qrels_ndcg_cut_10")
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--family", default=None)
    parser.add_argument("--k", default=None)
    parser.add_argument("--recipe", default=None)
    parser.add_argument("--emit-cell", action="store_true")
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


def _decision_payload(
    args: argparse.Namespace,
    results: list[LambdaResult],
    decision,
) -> dict:
    return {
        "schema_version": 1,
        "template": args.config_template,
        "family": args.family,
        "k": args.k,
        "recipe": args.recipe,
        "selected_lambda": decision.lambda_star,
        "selected_step": decision.step,
        "selection_metric": decision.metric_value,
        "argmax_lambda": decision.argmax_lambda,
        "argmax_metric": decision.argmax_value,
        "band": decision.band,
        "collapsed": decision.collapsed,
        "missing": decision.missing,
        "confidence": decision.confidence,
        "warnings": decision.warnings,
        "errors": decision.errors,
        "candidates": [
            {
                "lambda": result.lam,
                "code": result.code,
                "trajectory": result.traj,
                "min_stdev": result.min_stdev,
            }
            for result in results
        ],
    }


def main() -> int:
    """Discover evidence, select lambda, and emit a reusable receipt."""
    args = parse_args()
    grid = parse_grid(args.grid)
    results = _gather_local(
        args.config_template,
        grid,
        runs_root=args.runs_root,
        metric=args.metric,
    )
    decision = select_heldout_lambda(
        results,
        standard_error=args.se,
        convergence_step=args.conv_step,
        collapse_floor=args.collapse,
        min_final_step=args.min_final_step,
        quality_floor=args.quality_floor,
    )
    if decision is None:
        print("ERROR: no usable lambda candidate", file=sys.stderr)
        return 2
    payload = _decision_payload(args, results, decision)
    print(
        f"lambda*={decision.lambda_star} step={decision.step} "
        f"metric={decision.metric_value:.5f} "
        f"confidence={decision.confidence}"
    )
    if args.emit_cell:
        print(json.dumps(payload, indent=2, sort_keys=True))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if decision.errors:
        return 2
    return 3 if decision.warnings and args.strict else 0


if __name__ == "__main__":
    raise SystemExit(main())
