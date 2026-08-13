#!/usr/bin/env python
"""Run fixed-order candidate-pool perturbation evaluation.

The config must contain a ``pool_perturbation:`` block.  For each selected
query, the runner scores the canonical top-100, a replacement pool (drop one
uniformly sampled rank and append rank 101), and a drop-only pool.  Metrics are
restricted to the 99 retained documents.

Usage:
    uv run python scripts/run_pool_perturbation.py -e <ID>
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from dotenv import load_dotenv

from presentation_dependence.utils.config import (
    apply_execution_environment,
    apply_overrides,
    load_experiment_config,
    write_resolved_config,
)
from presentation_dependence.utils.run_paths import default_runs_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-e",
        "--exp-config",
        required=True,
        help="Experiment ID or path to configs/experiments/<ID>.yaml.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Dotted-key override. Repeatable.",
    )
    parser.add_argument("--runs-root", type=Path, default=default_runs_root())
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Resume into an existing run directory.",
    )
    args = parser.parse_args()

    load_dotenv()
    config_path, cfg = load_experiment_config(args.exp_config)
    if args.override:
        cfg = apply_overrides(cfg, args.override)
        tmp_dir = Path(tempfile.mkdtemp(prefix="slm_pool_perturbation_"))
        config_path = write_resolved_config(cfg, tmp_dir / config_path.name)

    apply_execution_environment(cfg)

    from presentation_dependence.eval.pool_perturbation import PoolPerturbationRunner

    runner = PoolPerturbationRunner(
        config_path=config_path,
        runs_root=args.runs_root,
        run_dir=args.run_dir,
    )
    metrics = runner.run()
    print(f"[OK] run_dir = {runner.run_dir}")
    for perturbation, block in metrics["aggregate"].items():
        print(
            f"[OK] {perturbation}: n={block['n_queries']} "
            f"pool-PSI={block['mean_pool_psi']:.6f} "
            f"top-{runner.k_cutoff}-flip={block['top_k_set_flip_rate']:.6f} "
            f"delta-nDCG={block['mean_delta_ndcg_at_k_retained']:.6f}"
        )


if __name__ == "__main__":
    main()
