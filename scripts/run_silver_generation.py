#!/usr/bin/env python
"""Run any ``configs/silver`` label-generation config locally.

Open-weight ``k_shot_bsc`` configs dispatch to ``KShotBSCTeacher`` and hosted
``closed_model_generated_bsc`` configs dispatch to ``SilverGenerator``.

An open-weight config declaring ``teacher.local_data_parallel_workers: N``
fans out over N GPUs, one replica per GPU, and merges the shards back into the
usual single-run layout.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from presentation_dependence.self_distill.local_dp import run_teacher_local_dp, worker_count
from presentation_dependence.silver_data import (
    SilverConfig,
    SilverConfigKind,
    load_silver_config,
    validate_silver_config,
)
from presentation_dependence.utils.config import apply_execution_environment, apply_overrides, write_resolved_config
from presentation_dependence.utils.gpu import log_cuda_diagnostics, num_gpus
from presentation_dependence.utils.run_paths import default_runs_root


def get_parser() -> argparse.ArgumentParser:
    """Build the unified local silver-generation parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        help="Silver config id (configs/silver/<ID>.yaml) or explicit YAML path.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Dotted-key override, e.g. cost.max_budget_usd=1. Repeatable.",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help=(
            "Optional explicit run directory to resume. By default the "
            "script auto-resumes the most recent existing run dir for the "
            "same experiment_id (when one carries a non-empty "
            "silver_labels.jsonl); pass --new-run to force a fresh dir."
        ),
    )
    parser.add_argument(
        "--new-run",
        action="store_true",
        help="Hosted configs only: force a new run directory and skip auto-resume.",
    )
    parser.add_argument(
        "--runs-root",
        default=None,
        help=(
            "Open-weight configs only: historical KShot output root (default: "
            "SLM_RUNS_ROOT, else runs, producing <root>/self-distill/<id>/<timestamp>/)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve the config, check declared inputs exist, print the plan, and exit. No GPU, no provider call.",
    )
    return parser


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    """Validate and run one local silver-generation config."""
    parser = get_parser()
    args = parser.parse_args(argv)
    load_dotenv()
    try:
        loaded = load_silver_config(args.config)
        loaded = _apply_validated_overrides(loaded, args.override)
        if args.dry_run:
            from presentation_dependence.utils.dry_run import emit_plan

            teacher = loaded.data.get("teacher") or {}
            return emit_plan(
                "silver",
                Path(args.config),
                loaded.data,
                project_root=_project_root(),
                overrides=args.override,
                extra={
                    "silver_kind": loaded.kind.value,
                    "teacher_model": teacher.get("model_id"),
                    "client": teacher.get("client"),
                },
            )
        apply_execution_environment(loaded.data)
        log_cuda_diagnostics("[silver][diagnostics]")
        result = dispatch_silver_config(
            loaded,
            run_dir=args.run_dir,
            new_run=args.new_run,
            runs_root=args.runs_root,
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    if loaded.kind is SilverConfigKind.OPEN_WEIGHT:
        summary = result
        print(f"[OK] silver run_dir = {summary['run_dir']}")
        print(f"[OK] silver_labels = {summary['silver_labels']}")
        print(f"[OK] manifest      = {summary['manifest']}")
    else:
        print(f"[OK] silver run_dir = {result}")
    return 0


def _apply_validated_overrides(
    loaded: SilverConfig,
    overrides: list[str],
) -> SilverConfig:
    if not overrides:
        return loaded
    config = apply_overrides(loaded.data, overrides, strict_top_level=False)
    tmp_dir = Path(tempfile.mkdtemp(prefix="slm_silver_resolved_"))
    config_path = write_resolved_config(config, tmp_dir / loaded.path.name)
    print(f"[silver] applied {len(overrides)} override(s); resolved -> {config_path}")
    return validate_silver_config(config, path=config_path)


def dispatch_silver_config(
    loaded: SilverConfig,
    *,
    run_dir: str | Path | None = None,
    new_run: bool = False,
    runs_root: str | Path | None = None,
    open_teacher_cls: type | None = None,
    hosted_generator_cls: type | None = None,
) -> Any:
    """Dispatch a validated config while preserving each backend's run layout."""
    if run_dir is not None and new_run:
        raise ValueError("--run-dir and --new-run are mutually exclusive")

    if loaded.kind is SilverConfigKind.OPEN_WEIGHT:
        return _dispatch_open_weight(
            loaded,
            run_dir=run_dir,
            new_run=new_run,
            runs_root=runs_root,
            open_teacher_cls=open_teacher_cls,
        )

    if runs_root is not None:
        raise ValueError(
            "--runs-root applies only to open-weight configs; hosted configs use "
            "output.base_dir or runs/silver/<experiment_id>"
        )
    hosted_run_dir = Path(run_dir) if run_dir is not None else None
    if hosted_run_dir is None and not new_run:
        hosted_run_dir = _auto_detect_resume_dir(loaded.data)
        if hosted_run_dir is not None:
            print(f"[silver] resuming existing run dir -> {hosted_run_dir}")
    elif new_run:
        print("[silver] --new-run set; ignoring existing hosted run directories")

    if hosted_generator_cls is None:
        from presentation_dependence.silver_data import SilverGenerator

        hosted_generator_cls = SilverGenerator
    generator = hosted_generator_cls(
        loaded.data,
        config_path=loaded.path,
        run_dir=hosted_run_dir,
    )
    return generator.run()


def _dispatch_open_weight(
    loaded: SilverConfig,
    *,
    run_dir: str | Path | None,
    new_run: bool,
    runs_root: str | Path | None,
    open_teacher_cls: type | None,
) -> Any:
    """Run an open-weight teacher pass, fanning out over GPUs when asked to."""
    if new_run:
        raise ValueError(
            "--new-run applies only to hosted configs; open-weight configs "
            "already create a new run unless --run-dir is supplied"
        )
    n_workers = worker_count(loaded.data)
    visible = num_gpus()
    if n_workers > 1 and visible > 1:
        return _run_open_weight_local_dp(
            loaded,
            run_dir=run_dir,
            runs_root=runs_root,
            n_workers=min(n_workers, visible),
            visible_gpus=visible,
        )
    if n_workers > 1:
        # The configs declare 8 because that is what the study ran on. Spawning
        # 8 replicas against one device would just OOM, so fall back rather
        # than fail: the single-process path produces the same labels.
        print(
            f"[silver] teacher.local_data_parallel_workers={n_workers} but {visible} GPU(s) visible; "
            "running single process. Set SLM_NUM_GPUS to override.",
        )
    if open_teacher_cls is None:
        from presentation_dependence.self_distill.teacher import KShotBSCTeacher

        open_teacher_cls = KShotBSCTeacher
    teacher = open_teacher_cls(
        config_path=loaded.path,
        runs_root=str(runs_root or default_runs_root()),
        run_dir=run_dir,
    )
    return teacher.run()


def _run_open_weight_local_dp(
    loaded: SilverConfig,
    *,
    run_dir: str | Path | None,
    runs_root: str | Path | None,
    n_workers: int,
    visible_gpus: int,
) -> dict:
    """Fan an open-weight teacher pass out over one GPU per worker.

    The single-process path lets ``KShotBSCTeacher`` choose the run directory,
    but the fan-out needs it up front to place the per-worker subdirectories
    under it, so resolve it here using the same layout helper the teacher uses.
    """
    from datetime import datetime

    from presentation_dependence.self_distill.silver_io import silver_run_dir

    cfg = loaded.data
    exp_id = str(cfg.get("id") or cfg.get("experiment_id") or "silver-run")
    if run_dir is not None:
        resolved_run_dir = Path(run_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        resolved_run_dir = silver_run_dir(str(runs_root or default_runs_root()), exp_id, ts)
    resolved_run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[silver] local data parallel: {n_workers} worker(s) over {visible_gpus} visible GPU(s)")
    return run_teacher_local_dp(
        cfg=cfg,
        exp_id=exp_id,
        trial_name="local",
        run_dir=resolved_run_dir,
        tmp_root=Path(tempfile.mkdtemp(prefix="slm_silver_local_dp_")),
        n_workers=n_workers,
        visible_gpus=visible_gpus,
    )


def _auto_detect_resume_dir(cfg: dict) -> Path | None:
    """Find the most recent existing run dir for this experiment, if any.

    The contract is: if a prior run dir exists for `cfg['experiment_id']`
    AND it has a non-empty `silver_labels.jsonl`, prefer to resume it.
    Otherwise return None (caller will create a fresh timestamped dir).
    """
    experiment_id = str(cfg.get("experiment_id") or cfg.get("id") or "silver-run")
    base = Path((cfg.get("output") or {}).get("base_dir") or f"runs/silver/{experiment_id}")
    if not base.is_dir():
        return None
    candidates = [
        d
        for d in base.iterdir()
        if d.is_dir() and (d / "silver_labels.jsonl").is_file() and not (d / ".frozen").exists()
    ]
    if not candidates:
        return None
    # Pick the most recent by mtime; falling back to name-sort for ties.
    candidates.sort(key=lambda p: (p.stat().st_mtime, p.name))
    chosen = candidates[-1]
    if chosen.joinpath("silver_labels.jsonl").stat().st_size == 0:
        return None
    return chosen


if __name__ == "__main__":
    raise SystemExit(main())
