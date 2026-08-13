#!/usr/bin/env python
"""Run self-distillation student SFT from a YAML config.

Local counterpart of the ``student:`` path in
``scripts/train/entrypoint.py``. Teacher and student jobs stay separate:
the caller points at a completed ``silver_labels.jsonl`` and the matching
``fixture.jsonl``.

When the config declares ``execution.distributed: ddp`` and more than one GPU
is visible, this re-execs itself under single-node torchrun, one worker per
GPU. Set ``SLM_NUM_GPUS`` to override the detected count.
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
from presentation_dependence.utils.gpu import log_cuda_diagnostics, maybe_reexec_torchrun


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-e",
        "--exp-config",
        required=True,
        help="Self-distill SFT exp-id or YAML path under configs/self-distill.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Dotted-key override, e.g. student.training.max_steps=1. Repeatable.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve the config, check declared inputs exist, print the plan, and exit. No GPU, no model load.",
    )
    args = parser.parse_args()

    load_dotenv()

    config_path, cfg = load_experiment_config(args.exp_config, subdir="self-distill")
    if args.override:
        cfg = apply_overrides(cfg, args.override)
        tmp_dir = Path(tempfile.mkdtemp(prefix="slm_student_sft_resolved_"))
        config_path = write_resolved_config(cfg, tmp_dir / config_path.name)
        print(f"[student-sft] applied {len(args.override)} override(s); resolved -> {config_path}")

    if args.dry_run:
        student = cfg.get("student") or {}
        raise SystemExit(
            emit_plan(
                "student-sft",
                config_path,
                cfg,
                project_root=_project_root(),
                overrides=args.override,
                extra={
                    "output_dir": student.get("output_dir"),
                    "base_model": student.get("model_name") or student.get("base_model"),
                },
            )
        )

    apply_execution_environment(cfg)
    # After the environment is applied, so execution.environment can carry the
    # CPU-fallback override, and before torchrun forks: a broken driver should
    # stop one process, not N.
    log_cuda_diagnostics("[student-sft][diagnostics]")

    # Workers re-read `config_path`, which already carries any --override, so
    # the flags are not passed through a second time.
    ddp_exit = maybe_reexec_torchrun(
        cfg,
        ["-e", str(config_path)],
        script=Path(__file__).resolve(),
        log_prefix="[student-sft][ddp]",
    )
    if ddp_exit is not None:
        raise SystemExit(int(ddp_exit))

    from presentation_dependence.self_distill.student import train_student_sft_from_config

    summary = train_student_sft_from_config(config_path)
    if summary.get("is_main_process") is False:
        # Only rank 0 writes artifacts, and only its summary carries the loss
        # fields printed below.
        print(f"[OK] rank {summary.get('rank')} of {summary.get('world_size')} finished; artifacts are rank-0 only.")
        return

    print(f"[OK] output_dir  = {summary['output_dir']}")
    print(f"[OK] checkpoint  = {summary['checkpoint']}")
    print(f"[OK] n_chunks    = {summary['n_chunks']}")
    print(f"[OK] steps       = {summary['global_steps']}")
    print(f"[OK] loss_initial={summary['loss_initial']} loss_final={summary['loss_final']}")
    if summary.get("progress_jsonl"):
        print(f"[OK] progress    = {summary['progress_jsonl']}")


if __name__ == "__main__":
    main()
