#!/usr/bin/env python
r"""Run a single presentation_dependence experiment locally.

Usage:
    uv run python scripts/run_experiment.py -e configs/experiments/example-passage-b20-psi.yaml
    uv run python scripts/run_experiment.py -e example-passage-b20-psi \\
        --override reranker.batch_size=64 \\
        --override data.k_input=50

``-e`` accepts either a short exp-id (``example-passage-b20-psi``) or an
explicit path. Overrides use the dotted-key syntax every runner shares.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from dotenv import load_dotenv

from presentation_dependence.utils.dry_run import emit_plan
from presentation_dependence.utils.config import (
    apply_execution_environment,
    apply_overrides,
    load_experiment_config,
    write_resolved_config,
)
from presentation_dependence.utils.gpu import log_cuda_diagnostics
from presentation_dependence.utils.run_paths import default_runs_root


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-e",
        "--exp-config",
        type=str,
        required=True,
        help="Exp-id (e.g. example-passage-b20-psi) or path to configs/experiments/<ID>.yaml.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Dotted-key override, e.g. reranker.batch_size=64. Repeatable.",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=default_runs_root(),
        help=(
            "Root directory for run artefacts (default: SLM_RUNS_ROOT, else runs/). "
            "Each run lands in <runs-root>/<ID>/<timestamp>/."
        ),
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help=(
            "Reuse this exact run dir instead of minting a new timestamped one. "
            "Resume path: per-query results already on disk are skipped and only "
            "missing qids are re-ranked. Used by scripts/run_eval_robust.sh so a "
            "re-launch after a crash / sleep / credential expiry never re-pays for "
            "completed queries (matters for paid API rerankers)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve the config, check declared inputs exist, print the plan, and exit. No GPU, no model load.",
    )
    args = parser.parse_args()

    load_dotenv()

    config_path, cfg = load_experiment_config(args.exp_config)
    if args.override:
        cfg = apply_overrides(cfg, args.override)
        # Hand ExperimentManager a resolved YAML on disk so
        # `runs/<ID>/<ts>/resolved_config.yaml` faithfully records what
        # actually ran: overrides and all.
        tmp_dir = Path(tempfile.mkdtemp(prefix="slm_resolved_"))
        config_path = write_resolved_config(cfg, tmp_dir / config_path.name)
        print(f"[run] applied {len(args.override)} override(s); resolved -> {config_path}")

    if args.dry_run:
        raise SystemExit(
            emit_plan(
                "rerank",
                config_path,
                cfg,
                project_root=_project_root(),
                overrides=args.override,
                extra={"runs_root": args.runs_root, "run_dir": args.run_dir},
            )
        )

    apply_execution_environment(cfg)
    log_cuda_diagnostics("[experiment][diagnostics]")

    # Imported here so --dry-run needs neither torch nor an accelerator.
    from presentation_dependence.eval import ExperimentManager

    manager = ExperimentManager(
        config_path=config_path,
        runs_root=args.runs_root,
        run_dir=args.run_dir,
    )
    manager.run()
    print(f"[OK] run_dir = {manager.run_dir}")


if __name__ == "__main__":
    main()
