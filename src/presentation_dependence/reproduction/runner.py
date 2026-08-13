"""CLI for silver, training, direct-eval, and downstream reproduction stages."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from presentation_dependence.utils.config import find_configs_root
from presentation_dependence.analysis.cli import run_cli as run_representative_cli

from .appendix import run_cli as run_appendix_cli
from .cohort_silver import (
    collect_qa_silver,
    collect_response_silver,
    materialize_qa_silver,
    materialize_response_silver,
    plan_qa_silver,
    plan_response_silver,
    validate_qa_silver,
    validate_response_silver,
)
from .direct_eval import (
    DirectEvalStageError,
    materialize_direct_eval,
    plan_direct_eval,
    validate_direct_eval,
)
from .direct_results import collect_direct_results
from .downstream_common import DownstreamStageError
from .errors import ReproductionError
from .passage_downstream import (
    collect_passage_downstream,
    materialize_passage_downstream,
    plan_passage_downstream,
    validate_passage_downstream,
)
from .qa_downstream import (
    collect_qa_downstream,
    materialize_qa_downstream,
    plan_qa_downstream,
    validate_qa_downstream,
)
from .pipelines import (
    load_multi_document_qa,
    load_passage_reranking,
    load_response_ranking,
)
from .response_downstream import (
    collect_response_downstream,
    materialize_response_downstream,
    plan_response_downstream,
    validate_response_downstream,
)
from .silver import (
    SilverStageError,
    collect_silver,
    materialize_silver,
    plan_silver,
    validate_silver,
)
from .structural import run_cli as run_structural_cli
from .training import (
    TrainingStageError,
    collect_ablation_checkpoint_catalog,
    collect_task_checkpoint_catalog,
    materialize_training,
    plan_training,
    validate_training,
)


TaskLoader = Callable[[Path], dict[str, Any]]
StageFunction = Callable[..., dict[str, Any]]

TASK_LOADERS: dict[str, TaskLoader] = {
    "passage-reranking": load_passage_reranking,
    "multi-document-qa": load_multi_document_qa,
    "response-ranking": load_response_ranking,
}


def project_root() -> Path:
    """Return the project root."""
    try:
        return find_configs_root().parent
    except FileNotFoundError:
        return Path(__file__).resolve().parents[3]


def build_parser() -> argparse.ArgumentParser:
    """Build the task/stage reproduction CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=sorted(TASK_LOADERS))
    parser.add_argument(
        "stage",
        choices=("silver", "training", "direct-eval", "downstream-eval"),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "materialize"):
        command = commands.add_parser(name)
        command.add_argument("--include-ablations", action="store_true")
        command.add_argument("--include-controls", action="store_true")
    validate = commands.add_parser("validate")
    validate.add_argument("--skip-data", action="store_true")
    validate.add_argument("--include-controls", action="store_true")
    validate.add_argument("--include-ablations", action="store_true")
    execute = commands.add_parser("execute")
    mode = execute.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Print the commands this stage would run.")
    mode.add_argument("--run", action="store_true", help="Run the stage locally.")
    execute.add_argument("--include-controls", action="store_true")
    execute.add_argument("--include-ablations", action="store_true")
    collect = commands.add_parser("collect")
    collect.add_argument("--teacher-run", type=Path)
    collect.add_argument("--training-teacher-run", type=Path)
    collect.add_argument("--heldout-teacher-run", type=Path)
    collect.add_argument("--pointwise-training-teacher-run", type=Path)
    collect.add_argument("--pointwise-heldout-teacher-run", type=Path)
    collect.add_argument("--runs-root", type=Path)
    collect.add_argument("--output-root", type=Path)
    collect.add_argument("--allow-missing", action="store_true")
    collect.add_argument("--include-ablations", action="store_true")
    collect.add_argument("--include-controls", action="store_true")
    collect.add_argument("--source-bindings", type=Path)
    collect.add_argument(
        "--discover-latest",
        action="store_true",
        help="Opt in to legacy latest-run discovery for unbound jobs.",
    )
    collect.add_argument("--verify-hashes", action="store_true")
    commands.add_parser("summarize")
    return parser


def _print(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


# Execute runs every stage in this process tree. These are the local
# counterparts of the cloud launchers the stage used to hand work to.
SWEEP_RUNNER = "scripts/run_sweep.py"
SILVER_RUNNER = "scripts/run_silver_generation.py"
READER_RUNNER = "scripts/run_reader.py"


def _sweep_command(root: Path, sweep: str) -> list[str]:
    """Command that expands a sweep and runs every cell."""
    return [sys.executable, str(root / SWEEP_RUNNER), sweep]


def _silver_command(root: Path, config: str) -> list[str]:
    """Command that runs one label-generation config."""
    return [sys.executable, str(root / SILVER_RUNNER), "-c", config]


def _launch(
    args: argparse.Namespace,
    command: list[str],
    *,
    payload_key: str = "command",
) -> int:
    if args.dry_run:
        _print(
            {
                "task": args.task,
                "stage": args.stage,
                "mode": "dry-run",
                payload_key: command,
                "executed": False,
            }
        )
        return 0
    return subprocess.run(command, check=False).returncode


def _print_file(path: Path, error: type[RuntimeError], label: str) -> int:
    if not path.is_file():
        raise error(f"No {label} found: {path}")
    print(path.read_text(encoding="utf-8"), end="")
    return 0


def _passage_silver(args: argparse.Namespace, root: Path) -> int:
    pipeline = load_passage_reranking(root)
    if args.command == "plan":
        _print(plan_silver(pipeline))
    elif args.command == "materialize":
        _print(materialize_silver(pipeline, root))
    elif args.command == "validate":
        _print(validate_silver(pipeline, root, check_data=not args.skip_data))
    elif args.command == "execute":
        materialize_silver(pipeline, root)
        validation = validate_silver(pipeline, root, check_data=True)
        return _launch(args, _silver_command(root, validation["teacher_config"]))
    elif args.command == "collect":
        if not args.teacher_run:
            raise SilverStageError("collect requires --teacher-run")
        _print(collect_silver(pipeline, root, teacher_run=args.teacher_run))
    else:
        return _print_file(
            root / str(pipeline["silver"]["outputs"]["root"]) / str(pipeline["silver"]["outputs"]["manifest"]),
            SilverStageError,
            "silver manifest",
        )
    return 0


def _cohort_silver(  # noqa: C901
    args: argparse.Namespace, root: Path
) -> int:
    qa = args.task == "multi-document-qa"
    pipeline = load_multi_document_qa(root) if qa else load_response_ranking(root)
    plan_fn = plan_qa_silver if qa else plan_response_silver
    materialize_fn = materialize_qa_silver if qa else materialize_response_silver
    validate_fn = validate_qa_silver if qa else validate_response_silver
    collect_fn = collect_qa_silver if qa else collect_response_silver
    if args.command == "plan":
        _print(plan_fn(pipeline))
    elif args.command == "materialize":
        _print(materialize_fn(pipeline, root))
    elif args.command == "validate":
        _print(validate_fn(pipeline, root, check_data=not args.skip_data))
    elif args.command == "execute":
        materialize_fn(pipeline, root)
        validation = validate_fn(pipeline, root, check_data=True)
        commands = [_silver_command(root, config) for config in validation["teacher_configs"]]
        if args.dry_run:
            _print(
                {
                    "task": args.task,
                    "stage": args.stage,
                    "mode": "dry-run",
                    "commands": commands,
                    "executed": False,
                }
            )
            return 0
        for command in commands:
            completed = subprocess.run(command, check=False)
            if completed.returncode:
                return completed.returncode
    elif args.command == "collect":
        if not args.training_teacher_run or not args.heldout_teacher_run:
            raise SilverStageError("collect requires --training-teacher-run and --heldout-teacher-run")
        needs_pointwise = any(
            str(product.get("labels")) == "pointwise-single" for product in pipeline["silver"]["products"]
        )
        if needs_pointwise and (not args.pointwise_training_teacher_run or not args.pointwise_heldout_teacher_run):
            raise SilverStageError(
                "collect requires --pointwise-training-teacher-run and "
                "--pointwise-heldout-teacher-run for the B=1 controls"
            )
        _print(
            collect_fn(
                pipeline,
                root,
                training_teacher_run=args.training_teacher_run,
                heldout_teacher_run=args.heldout_teacher_run,
                pointwise_training_teacher_run=args.pointwise_training_teacher_run,
                pointwise_heldout_teacher_run=args.pointwise_heldout_teacher_run,
            )
        )
    else:
        return _print_file(
            root / str(pipeline["silver"]["outputs"]["root"]) / str(pipeline["silver"]["outputs"]["manifest"]),
            SilverStageError,
            "silver manifest",
        )
    return 0


def _training(args: argparse.Namespace, root: Path) -> int:
    pipeline = TASK_LOADERS[args.task](root)
    if args.command == "plan":
        _print(plan_training(pipeline, include_ablations=args.include_ablations))
    elif args.command == "materialize":
        _print(
            materialize_training(
                pipeline,
                root,
                include_ablations=args.include_ablations,
            )
        )
    elif args.command == "validate":
        _print(
            validate_training(
                pipeline,
                root,
                include_ablations=args.include_ablations,
                check_data=not args.skip_data,
            )
        )
    elif args.command == "execute":
        materialize_training(
            pipeline,
            root,
            include_ablations=args.include_ablations,
        )
        validation = validate_training(
            pipeline,
            root,
            include_ablations=args.include_ablations,
        )
        sweep = validation["ablations"]["sweep"] if args.include_ablations else validation["sweep"]
        return _launch(args, _sweep_command(root, sweep))
    elif args.command == "collect":
        collector = collect_ablation_checkpoint_catalog if args.include_ablations else collect_task_checkpoint_catalog
        collector_kwargs = {
            "runs_root": args.runs_root,
            "output_root": args.output_root,
        }
        if not args.include_ablations:
            collector_kwargs.update(
                {
                    "source_bindings": args.source_bindings,
                    "discover_latest": args.discover_latest,
                    "verify_hashes": args.verify_hashes,
                }
            )
        _print(
            collector(
                pipeline,
                root,
                **collector_kwargs,
            )
        )
    else:
        return _print_file(
            root
            / str(pipeline["training"]["outputs"]["root"])
            / str(pipeline["training"]["outputs"]["checkpoint_catalog"]),
            TrainingStageError,
            "checkpoint catalog",
        )
    return 0


def _direct_eval(args: argparse.Namespace, root: Path) -> int:
    pipeline = TASK_LOADERS[args.task](root)
    if args.command == "plan":
        _print(plan_direct_eval(pipeline, root, include_controls=args.include_controls))
    elif args.command == "materialize":
        _print(materialize_direct_eval(pipeline, root, include_controls=args.include_controls))
    elif args.command == "validate":
        _print(
            validate_direct_eval(
                pipeline,
                root,
                check_data=not args.skip_data,
                include_controls=args.include_controls,
            )
        )
    elif args.command == "execute":
        materialize_direct_eval(
            pipeline,
            root,
            include_controls=args.include_controls,
        )
        validation = validate_direct_eval(pipeline, root, include_controls=args.include_controls)
        return _launch(args, _sweep_command(root, validation["sweep"]))
    elif args.command == "collect":
        _print(
            collect_direct_results(
                pipeline,
                root,
                runs_root=args.runs_root,
                output_root=args.output_root,
                allow_missing=args.allow_missing,
                source_bindings=args.source_bindings,
                discover_latest=args.discover_latest,
                verify_hashes=args.verify_hashes,
                include_controls=args.include_controls,
            )
        )
    else:
        return _print_file(
            root
            / str(pipeline["direct_eval"]["outputs"]["root"])
            / str(pipeline["direct_eval"]["outputs"]["aggregate"]),
            DirectEvalStageError,
            "direct-eval aggregate",
        )
    return 0


def _downstream_functions(
    task: str,
) -> tuple[StageFunction, StageFunction, StageFunction, StageFunction]:
    if task == "passage-reranking":
        return (
            plan_passage_downstream,
            materialize_passage_downstream,
            validate_passage_downstream,
            collect_passage_downstream,
        )
    if task == "multi-document-qa":
        return (
            plan_qa_downstream,
            materialize_qa_downstream,
            validate_qa_downstream,
            collect_qa_downstream,
        )
    return (
        plan_response_downstream,
        materialize_response_downstream,
        validate_response_downstream,
        collect_response_downstream,
    )


def _execute_qa_downstream(
    args: argparse.Namespace,
    root: Path,
    materialize_fn: StageFunction,
    pipeline: dict[str, Any],
) -> int:
    configs = materialize_fn(pipeline, root)["configs"]
    # No staging step locally: the reader reads the QA bundle and the scorer
    # parquet from disk, which is where materialize just wrote them.
    commands = [[sys.executable, str(root / READER_RUNNER), "--config", config] for config in configs]
    if args.dry_run:
        _print(
            {
                "task": args.task,
                "stage": args.stage,
                "mode": "dry-run",
                "commands": commands,
                "executed": False,
            }
        )
        return 0
    for command in commands:
        completed = subprocess.run(command, check=False)
        if completed.returncode:
            return completed.returncode
    return 0


def _downstream(args: argparse.Namespace, root: Path) -> int:
    pipeline = TASK_LOADERS[args.task](root)
    plan_fn, materialize_fn, validate_fn, collect_fn = _downstream_functions(args.task)
    if args.command == "plan":
        _print(plan_fn(pipeline, root))
    elif args.command == "materialize":
        _print(materialize_fn(pipeline, root))
    elif args.command == "validate":
        _print(validate_fn(pipeline, root, check_data=not args.skip_data))
    elif args.command == "execute":
        if args.task == "multi-document-qa":
            return _execute_qa_downstream(args, root, materialize_fn, pipeline)
        if args.dry_run:
            _print(
                {
                    "task": args.task,
                    "stage": args.stage,
                    "mode": "dry-run",
                    "inference_jobs": 0,
                    "action": "reduce direct-eval score logs offline",
                    "submitted": False,
                }
            )
        else:
            _print(collect_fn(pipeline, root))
    elif args.command == "collect":
        _print(
            collect_fn(
                pipeline,
                root,
                runs_root=args.runs_root,
                output_root=args.output_root,
                allow_missing=args.allow_missing,
                source_bindings=args.source_bindings,
                discover_latest=args.discover_latest,
                verify_hashes=args.verify_hashes,
            )
        )
    else:
        return _print_file(
            root
            / str(pipeline["downstream_eval"]["outputs"]["root"])
            / str(pipeline["downstream_eval"]["outputs"]["aggregate"]),
            DownstreamStageError,
            "downstream aggregate",
        )
    return 0


STAGE_HANDLERS: dict[str, Callable[[argparse.Namespace, Path], int]] = {
    "training": _training,
    "direct-eval": _direct_eval,
    "downstream-eval": _downstream,
}

COMMAND_FAMILIES: dict[str, tuple[str, Callable[[list[str], Path], int]]] = {
    "structural": (
        "tracked-source and clean-checkout audits",
        run_structural_cli,
    ),
    "appendix": ("appendix program planning and execution", run_appendix_cli),
    "representative": (
        "offline representative analysis checks",
        run_representative_cli,
    ),
}


def print_root_help() -> None:
    """Print the complete task and command-family surface."""
    print("usage: study.py <task> <stage> <command> [options]")
    print("       study.py <family> <command> [options]")
    print()
    print("Task pipeline:")
    print("  task     " + " | ".join(sorted(TASK_LOADERS)))
    print("  stage    silver | training | direct-eval | downstream-eval")
    print("  command  plan | materialize | validate | execute | collect | summarize")
    print()
    print("Command families:")
    width = max(map(len, COMMAND_FAMILIES))
    for name, (description, _handler) in COMMAND_FAMILIES.items():
        print(f"  {name:<{width}}  {description}")
    print()
    print("Use 'study.py <family> --help' or")
    print("'study.py <task> <stage> <command> --help' for details.")


def run(argv: list[str], *, root: Path | None = None) -> int:
    """Run one reproduction CLI command."""
    resolved_root = root or project_root()
    try:
        if not argv or argv[0] in {"-h", "--help"}:
            print_root_help()
            return 0
        if argv[0] in COMMAND_FAMILIES:
            return COMMAND_FAMILIES[argv[0]][1](argv[1:], resolved_root)
        args = build_parser().parse_args(argv)
        if args.stage == "silver":
            handler = _passage_silver if args.task == "passage-reranking" else _cohort_silver
        else:
            handler = STAGE_HANDLERS[args.stage]
        return handler(args, resolved_root)
    except ReproductionError as exc:
        print(f"[reproduction][ERROR] {exc}", file=sys.stderr)
        return 2


def main(argv: list[str] | None = None) -> int:
    """Run the reproduction CLI entrypoint."""
    return run(list(sys.argv[1:] if argv is None else argv))
