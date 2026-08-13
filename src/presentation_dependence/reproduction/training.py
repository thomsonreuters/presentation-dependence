"""Plan, materialize, validate, and collect reproduction training runs."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any, Mapping


from presentation_dependence.self_distill.selection import best_checkpoint
from presentation_dependence.utils.config import load_yaml_mapping
from presentation_dependence.utils.sweeps import write_config, write_sweep
from .common import prepare_generated_directory
from .errors import TrainingStageError
from .source_bindings import (
    checkpoint_from_binding,
    load_stage_bindings,
    reject_unknown_bindings,
    require_bindings_or_discovery,
    resolve_bound_run,
)


def _training(pipeline: Mapping[str, Any]) -> Mapping[str, Any]:
    value = pipeline.get("training")
    if not isinstance(value, Mapping):
        raise TrainingStageError("training must be a mapping")
    return value


def _jobs(pipeline: Mapping[str, Any]) -> list[dict[str, Any]]:
    training = _training(pipeline)
    jobs: list[dict[str, Any]] = []
    for variant in training["variants"]:
        for seed, seed_config in training["seeds"].items():
            if variant["id"] == "oc-sft":
                for candidate in variant["lambda_grid"]:
                    jobs.append(
                        {
                            "variant": variant["id"],
                            "seed": int(seed),
                            "view_seeds": seed_config["view_seeds"],
                            "lambda_code": candidate["code"],
                            "lambda": candidate["value"],
                            "template": variant["template_pattern"].format(code=candidate["code"]),
                            "silver": variant["silver"],
                            "view_seeded": True,
                        }
                    )
            else:
                jobs.append(
                    {
                        "variant": variant["id"],
                        "seed": int(seed),
                        "view_seeds": seed_config["view_seeds"],
                        "template": variant["template"],
                        "silver": variant["silver"],
                        "view_seeded": bool(variant.get("view_seeded")),
                    }
                )
    return jobs


def _ablation_jobs(pipeline: Mapping[str, Any]) -> list[dict[str, Any]]:
    training = _training(pipeline)
    base_variant = next(variant for variant in training["variants"] if variant["id"] == "k1-sft")
    seed_configs = training["seeds"]
    return [
        {
            "variant": ablation["id"],
            "seed": int(seed),
            "view_seeds": seed_configs[int(seed)]["view_seeds"],
            "template": ablation["template"],
            "silver": ablation.get("silver", base_variant["silver"]),
            "view_seeded": False,
            "patch": ablation.get("patch") or {},
            "ablation": True,
        }
        for ablation in training["optional_ablations"]["variants"]
        if ablation.get("runnable")
        for seed in ablation.get("seeds", [42])
    ]


def _config_id(pipeline: Mapping[str, Any], job: Mapping[str, Any]) -> str:
    suffix = f"--lambda{job['lambda_code']}" if job.get("lambda_code") else ""
    return f"{pipeline['task']['id']}-training--{job['variant']}{suffix}--seed{job['seed']}"


def plan_training(pipeline: Mapping[str, Any], *, include_ablations: bool = False) -> dict[str, Any]:
    """Return planned training jobs and optional ablations without writing files."""
    training = _training(pipeline)
    jobs = [{**job, "id": _config_id(pipeline, job)} for job in _jobs(pipeline)]
    ablation_jobs = [
        {**job, "id": _config_id(pipeline, job)} for job in (_ablation_jobs(pipeline) if include_ablations else [])
    ]
    return {
        "task": pipeline["task"]["id"],
        "stage": "training",
        "jobs": jobs,
        "job_count": len(jobs),
        "ablation_jobs": ablation_jobs,
        "ablation_job_count": len(ablation_jobs),
        "variants": sorted({job["variant"] for job in jobs}),
        "training_seeds": sorted({job["seed"] for job in jobs}),
        "optional_ablations": (training["optional_ablations"]["variants"] if include_ablations else []),
        "optional_ablations_enabled": include_ablations,
        "checkpoint_selection": training["checkpoint_selection"],
    }


def _set_view_seeds(student: dict[str, Any], view_seeds: list[int]) -> None:
    objective = student.get("objective") or {}
    if "view_seeds" in objective:
        objective["view_seeds"] = view_seeds
    augmentation = (student.get("data") or {}).get("permutation_augmentation") or {}
    if augmentation.get("enabled"):
        augmentation["seeds"] = view_seeds


def _deep_merge(target: dict[str, Any], patch: Mapping[str, Any]) -> None:
    for key, value in patch.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = value


def _materialized_config(
    pipeline: Mapping[str, Any],
    project_root: Path,
    job: Mapping[str, Any],
) -> dict[str, Any]:
    template_path = project_root / str(job["template"])
    if not template_path.is_file():
        raise TrainingStageError(f"Missing training template: {template_path}")
    config = load_yaml_mapping(template_path)
    config_id = _config_id(pipeline, job)
    config["id"] = config_id
    _deep_merge(config, job.get("patch") or {})
    student = config["student"]
    student["output_dir"] = f"runs/self-distill/{config_id}/student"
    student["training"]["seed"] = int(job["seed"])
    if job["view_seeded"]:
        _set_view_seeds(student, list(job["view_seeds"]))
    if job.get("lambda") is not None:
        student["objective"]["lambda"] = float(job["lambda"])
    return config


def _materialize_job_set(
    pipeline: Mapping[str, Any],
    project_root: Path,
    jobs: list[dict[str, Any]],
    root: Path,
    *,
    sweep_id: str,
) -> tuple[list[dict[str, Any]], Path]:
    training = _training(pipeline)
    configs_dir = prepare_generated_directory(root / str(training["outputs"]["configs_dir"]))
    records = []
    for job in jobs:
        config = _materialized_config(pipeline, project_root, job)
        path = configs_dir / f"{config['id']}.yaml"
        write_config(path, config)
        records.append({**job, "id": config["id"], "config": str(path)})
    description = (
        f"{pipeline['task']['id']} training sweep"
        if sweep_id.endswith("training-primary")
        else f"{pipeline['task']['id']} training ablation sweep"
    )
    sweep_path = write_sweep(
        root / str(training["outputs"]["sweep"]),
        {
            "id": sweep_id,
            "description": description,
            "execution": {"max_parallel": training["execution"]["max_parallel"]},
            "jobs": [{"exp_id": row["config"]} for row in records],
        },
    )
    return records, sweep_path


def materialize_training(
    pipeline: Mapping[str, Any],
    project_root: Path,
    *,
    build_root: Path | None = None,
    include_ablations: bool = False,
) -> dict[str, Any]:
    """Write training configs and their launch sweep."""
    training = _training(pipeline)
    root = build_root or project_root / str(training["outputs"]["root"])
    records, sweep_path = _materialize_job_set(
        pipeline,
        project_root,
        _jobs(pipeline),
        root,
        sweep_id=f"{pipeline['task']['id']}-training-primary",
    )
    result = {
        **plan_training(pipeline, include_ablations=include_ablations),
        "configs": records,
        "sweep": str(sweep_path),
    }
    if include_ablations:
        ablation_root = root.parent / "training-ablations"
        ablation_records, ablation_sweep = _materialize_job_set(
            pipeline,
            project_root,
            _ablation_jobs(pipeline),
            ablation_root,
            sweep_id=f"{pipeline['task']['id']}-training-ablations",
        )
        result["ablation_root"] = str(ablation_root)
        result["ablation_configs"] = ablation_records
        result["ablation_sweep"] = str(ablation_sweep)
    (root / "materialization.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _resolve_silver_input_path(
    project_root: Path,
    config: Mapping[str, Any],
    student: Mapping[str, Any],
    key: str,
) -> Path:
    """Resolve repository products separately from channel-local filenames."""
    silver_path = Path(str(student[key]))
    if silver_path.is_absolute():
        return silver_path
    if len(silver_path.parts) > 1:
        return project_root / silver_path
    fixture = Path(str(student["fixture_path"]))
    channel = config.get("execution", {}).get("fixture_channel")
    base = fixture.parent if fixture.parent != Path(".") else project_root / "data" / str(channel)
    return base / silver_path


def _validate_training_tree(
    pipeline: Mapping[str, Any],
    project_root: Path,
    root: Path,
    jobs: list[dict[str, Any]],
    *,
    check_data: bool,
) -> dict[str, Any]:
    training = _training(pipeline)
    configs = sorted((root / str(training["outputs"]["configs_dir"])).glob("*.yaml"))
    expected = len(jobs)
    if len(configs) != expected:
        raise TrainingStageError(f"Expected {expected} training configs, found {len(configs)}")
    ids: set[str] = set()
    for path in configs:
        config = load_yaml_mapping(path)
        config_id = str(config["id"])
        if config_id in ids:
            raise TrainingStageError(f"Duplicate training config ID: {config_id}")
        ids.add(config_id)
        if "AE" in config_id or "A8c" in config_id:
            raise TrainingStageError(f"Legacy experiment token in generated config ID: {config_id}")
        student = config["student"]
        for key in ("silver_labels_path", "eval_silver_labels_path"):
            silver_path = _resolve_silver_input_path(project_root, config, student, key)
            if check_data and not silver_path.is_file():
                raise TrainingStageError(f"Missing silver input referenced by {path}: {silver_path}")
    sweep_path = root / str(training["outputs"]["sweep"])
    sweep = load_yaml_mapping(sweep_path)
    if len(sweep["jobs"]) != expected:
        raise TrainingStageError("Training sweep size does not match config count")
    return {
        "configs": len(configs),
        "unique_ids": len(ids),
        "sweep": str(sweep_path),
    }


def validate_training(
    pipeline: Mapping[str, Any],
    project_root: Path,
    *,
    build_root: Path | None = None,
    include_ablations: bool = False,
    check_data: bool = True,
) -> dict[str, Any]:
    """Validate primary and optional isolated training trees."""
    training = _training(pipeline)
    root = build_root or project_root / str(training["outputs"]["root"])
    result = {
        "task": pipeline["task"]["id"],
        "stage": "training",
        **_validate_training_tree(
            pipeline,
            project_root,
            root,
            _jobs(pipeline),
            check_data=check_data,
        ),
        "selection_protocol": training["checkpoint_selection"]["selection_protocol"],
    }
    if include_ablations:
        ablation_root = root.parent / "training-ablations"
        result["ablations"] = _validate_training_tree(
            pipeline,
            project_root,
            ablation_root,
            _ablation_jobs(pipeline),
            check_data=check_data,
        )
    return result


def _candidate_variant_jobs(pipeline: Mapping[str, Any], project_root: Path) -> list[dict[str, Any]]:
    jobs = []
    for planned in plan_training(pipeline)["jobs"]:
        template_config = load_yaml_mapping(project_root / str(planned["template"]))
        legacy_id = str(template_config["id"])
        jobs.append(
            {
                "variant": planned["variant"],
                "training_seed": int(planned["seed"]),
                "semantic_id": planned["id"],
                "legacy_id": legacy_id,
                "lambda": (float(planned["lambda"]) if planned.get("lambda") is not None else None),
            }
        )
    return jobs


TRAINING_SUMMARY = Path("student") / "training_summary.json"


def _student_trials(runs_root: Path, exp_id: str) -> tuple[list[Path], Path | None]:
    """Return this experiment's completed student runs, in both layouts.

    Returns ``(trials, flat)``. ``trials`` are container-shaped runs, one
    directory per launch under ``<root>/<exp_id>/<trial>/``. ``flat`` is the
    local shape, which has no trial level: the materialized config points
    ``student.output_dir`` at ``runs/self-distill/<exp_id>/student``, so the
    run directory is ``<root>/self-distill/<exp_id>`` itself. Kept apart
    because "latest" only orders the timestamped ones.
    """
    trials: list[Path] = []
    experiment = runs_root / exp_id
    if experiment.is_dir():
        trials = [trial for trial in experiment.iterdir() if trial.is_dir() and (trial / TRAINING_SUMMARY).is_file()]
    flat = runs_root / "self-distill" / exp_id
    return trials, flat if (flat / TRAINING_SUMMARY).is_file() else None


def _find_existing_training_run(runs_root: Path, semantic_id: str, legacy_id: str | None, seed: int) -> Path | None:
    """Return the run to collect for this job, preferring a timestamped trial."""
    exp_ids = [semantic_id, legacy_id, f"{legacy_id}-seed{seed}" if legacy_id else None]
    trials: list[Path] = []
    flats: list[Path] = []
    for exp_id in exp_ids:
        if not exp_id:
            continue
        found, flat = _student_trials(runs_root, str(exp_id))
        trials.extend(found)
        if flat is not None:
            flats.append(flat)
    if trials:
        return sorted(trials)[-1]
    return flats[0] if flats else None


def _checkpoint_uri(trial: Path, selected_step: int, summary: Mapping[str, Any] | None = None) -> str:
    """Return the checkpoint directory for a selected step.

    The run records where it put its checkpoints, so read that rather than
    guess. ``training_summary.json``'s ``checkpoint`` is the last step written,
    and its parent is the root every step shares.

    Two roots are possible and they need opposite treatment. A local run names
    no ``checkpoint.dir``, so the trainer keeps steps under the run itself and
    the recorded root is usable as-is. A container run is given an explicit
    ``checkpoint.dir`` outside the output tree, so its recorded root is a
    container-internal absolute path that means nothing in this checkout; those
    are reconstructed from the parallel ``checkpoints/<exp_id>/<trial>/`` tree
    instead. "Inside the collected run" is the test, because that is exactly
    what makes the recorded path portable.

    Deliberately does not probe the filesystem: a run that saved no checkpoint,
    or whose checkpoints were pruned, would otherwise fall through and be
    described with the wrong layout.
    """
    name = f"checkpoint-step-{selected_step:06d}"
    recorded = (summary or {}).get("checkpoint")
    if recorded:
        root = Path(str(recorded)).parent
        if not root.is_absolute() or root.is_relative_to(trial):
            return f"{root.as_posix()}/{name}/"
    return f"checkpoints/{trial.parent.name}/{trial.name}/student/{name}/"


def _recorded_lambda(project_root: Path, selection: Mapping[str, Any]) -> tuple[float, str] | None:
    """Return the lambda recorded for this pipeline's declared decision cell.

    The selection protocol in ``self_distill.selection`` needs per-lambda score
    standard deviations for its collapse check, which live in run artifacts
    rather than in tracked evidence, so it cannot be replayed here. The tracked
    decision record is the authoritative output of that protocol.
    """
    cell = selection.get("decision_cell")
    if not cell:
        return None
    path = project_root / str(selection["decisions"])
    if not path.is_file():
        raise TrainingStageError(f"Missing lambda decision record: {path}")
    family, k = str(cell["family"]), str(cell["k"])
    matches = {
        float(row["lambda_star"])
        for row in json.loads(path.read_text(encoding="utf-8"))
        if row.get("family") == family and row.get("k") == k and row.get("lambda_star") is not None
    }
    if not matches:
        raise TrainingStageError(f"No recorded lambda decision for {family!r} {k!r} in {path}")
    if len(matches) > 1:
        raise TrainingStageError(
            f"Conflicting recorded lambda decisions for {family!r} {k!r} in {path}: {sorted(matches)}"
        )
    return matches.pop(), f"recorded-decision:{family} {k}"


def _select_lambda(
    candidates: list[dict[str, Any]],
    selection: Mapping[str, Any],
    destination: Path,
    project_root: Path,
) -> dict[str, Any]:
    receipt_path = destination / "lambda_selection.json"
    expected = selection.get("expected_lambda", selection.get("selected_lambda"))
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        selected = receipt.get("selected_lambda", receipt.get("lambda_star"))
        if selected is None:
            raise TrainingStageError(f"Lambda receipt has no selected lambda: {receipt_path}")
        return {
            **receipt,
            "selected_lambda": float(selected),
            "source": str(receipt_path),
        }
    by_lambda: dict[float, list[float]] = {}
    for candidate in candidates:
        if candidate["lambda"] is not None:
            by_lambda.setdefault(float(candidate["lambda"]), []).append(float(candidate["selection_metric"]))
    if not by_lambda:
        raise TrainingStageError("No completed OC-SFT lambda candidates")
    rule = str(selection["rule"])
    means = {lam: statistics.fmean(values) for lam, values in by_lambda.items()}
    recorded = _recorded_lambda(project_root, selection)
    if recorded is not None:
        selected, source = recorded
        if expected is not None and selected != float(expected):
            raise TrainingStageError(f"Recorded lambda {selected} disagrees with declared expected_lambda {expected}")
    elif expected is not None:
        selected, source = float(expected), "expected-paper-anchor"
    else:
        raise TrainingStageError(
            f"Selection rule {rule!r} has no tracked decision: declare "
            "decision_cell or expected_lambda, or place a "
            f"lambda_selection.json at {receipt_path}. Recomputing the "
            "selection here is not supported, because the protocol's 1-SE "
            "band and collapse rules need run-level score deviations."
        )
    if selected not in by_lambda:
        raise TrainingStageError(f"Selected lambda {selected} has no completed candidates")
    receipt = {
        "schema_version": 1,
        "rule": rule,
        "selected_lambda": selected,
        "expected_lambda": float(expected) if expected is not None else None,
        "matches_expected": (selected == float(expected) if expected is not None else None),
        "source": source,
        "mean_selection_metric_by_lambda": {str(lam): value for lam, value in sorted(means.items())},
    }
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return receipt


def _completed_candidate(
    job: Mapping[str, Any],
    trial: Path,
    metric: str,
) -> dict[str, Any]:
    summary = json.loads((trial / "student/training_summary.json").read_text(encoding="utf-8"))
    try:
        step, value = best_checkpoint(summary, metric)
    except ValueError as exc:
        raise TrainingStageError(str(exc)) from exc
    return {
        **job,
        "selected_step": step,
        "selection_metric": value,
        "trial": trial.name,
        "checkpoint_uri": _checkpoint_uri(trial, step, summary),
    }


def collect_ablation_checkpoint_catalog(
    pipeline: Mapping[str, Any],
    project_root: Path,
    *,
    runs_root: Path | None = None,
    output_root: Path | None = None,
) -> dict[str, Any]:
    """Collect isolated ablation checkpoints into a separate catalog."""
    training = _training(pipeline)
    source_root = runs_root or project_root / "runs"
    root = output_root or (project_root / str(training["outputs"]["root"])).parent / "training-ablations"
    root.mkdir(parents=True, exist_ok=True)
    metric = str(training["checkpoint_selection"]["metric"])
    rows = []
    missing = []
    for job in _ablation_jobs(pipeline):
        semantic_id = _config_id(pipeline, job)
        trial = _find_existing_training_run(source_root, semantic_id, None, int(job["seed"]))
        if trial is None:
            missing.append(
                {
                    "variant": job["variant"],
                    "training_seed": job["seed"],
                    "semantic_id": semantic_id,
                }
            )
            continue
        candidate = _completed_candidate(
            {
                "variant": job["variant"],
                "training_seed": int(job["seed"]),
                "semantic_id": semantic_id,
                "legacy_id": None,
                "lambda": None,
            },
            trial,
            metric,
        )
        rows.append(
            {
                "variant": candidate["variant"],
                "training_seed": candidate["training_seed"],
                "selected_step": candidate["selected_step"],
                "selection_metric": candidate["selection_metric"],
                "trial": candidate["trial"],
                "checkpoint_uri": candidate["checkpoint_uri"],
                "selection_rule": "heldout-argmax",
            }
        )
    catalog = {
        "schema_version": 1,
        "task": pipeline["task"]["id"],
        "stage": "training-ablations",
        "checkpoints": rows,
        "missing": missing,
    }
    (root / str(training["outputs"]["checkpoint_catalog"])).write_text(
        json.dumps(catalog, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return catalog


def collect_task_checkpoint_catalog(
    pipeline: Mapping[str, Any],
    project_root: Path,
    *,
    runs_root: Path | None = None,
    output_root: Path | None = None,
    source_bindings: Path | None = None,
    discover_latest: bool = False,
    verify_hashes: bool = False,
) -> dict[str, Any]:
    """Collect selected checkpoints from explicit bindings or opted-in discovery."""
    training = _training(pipeline)
    source_root = runs_root or project_root / "runs"
    destination = output_root or project_root / str(training["outputs"]["root"])
    destination.mkdir(parents=True, exist_ok=True)
    completed: list[dict[str, Any]] = []
    candidate_missing: list[dict[str, Any]] = []
    metric = str(training["checkpoint_selection"]["metric"])
    bindings = load_stage_bindings(
        project_root,
        str(pipeline["task"]["id"]),
        "training",
        path=source_bindings,
    )
    require_bindings_or_discovery(
        bindings,
        discover_latest=discover_latest,
        task=str(pipeline["task"]["id"]),
        stage="training",
    )
    candidate_jobs = _candidate_variant_jobs(pipeline, project_root)
    reject_unknown_bindings(
        bindings,
        {str(job["semantic_id"]) for job in candidate_jobs},
        task=str(pipeline["task"]["id"]),
        stage="training",
    )
    for job in candidate_jobs:
        binding = bindings.get(str(job["semantic_id"]))
        catalog_candidate = checkpoint_from_binding(binding) if binding else None
        if catalog_candidate:
            completed.append(catalog_candidate)
            continue
        if binding:
            trial = resolve_bound_run(
                project_root,
                binding,
                required_files=("student/training_summary.json",),
                verify_hashes=verify_hashes,
            )
        elif discover_latest:
            trial = _find_existing_training_run(
                source_root,
                job["semantic_id"],
                job["legacy_id"],
                job["training_seed"],
            )
        else:
            trial = None
        if trial is None:
            candidate_missing.append(dict(job))
            continue
        completed.append(_completed_candidate(job, trial, metric))
    oc_candidates = [row for row in completed if row["variant"] == "oc-sft"]
    lambda_selection = _select_lambda(
        oc_candidates,
        training["checkpoint_selection"],
        destination,
        project_root,
    )
    selected_lambda = float(lambda_selection["selected_lambda"])
    selected_candidates = [
        row for row in completed if row["variant"] != "oc-sft" or float(row["lambda"]) == selected_lambda
    ]
    rows = []
    for candidate in selected_candidates:
        rows.append(
            {
                "variant": candidate["variant"],
                "training_seed": candidate["training_seed"],
                "selected_step": candidate["selected_step"],
                "selection_metric": candidate["selection_metric"],
                "lambda": candidate["lambda"],
                "trial": candidate["trial"],
                "checkpoint_uri": candidate["checkpoint_uri"],
                "selection_rule": training["checkpoint_selection"]["rule"],
            }
        )
    selected_keys = {(row["variant"], int(row["training_seed"])) for row in rows}
    expected_variants = [str(variant["id"]) for variant in training["variants"]]
    expected_seeds = [int(seed) for seed in training["seeds"]]
    missing = [
        {
            "variant": variant,
            "training_seed": seed,
            "selected_lambda": (selected_lambda if variant == "oc-sft" else None),
        }
        for variant in expected_variants
        for seed in expected_seeds
        if (variant, seed) not in selected_keys
    ]
    catalog = {
        "schema_version": 1,
        "task": pipeline["task"]["id"],
        "stage": "training",
        "checkpoint_selection": training["checkpoint_selection"],
        "lambda_selection": lambda_selection,
        "checkpoints": rows,
        "missing": missing,
        "candidate_missing": candidate_missing,
    }
    (destination / str(training["outputs"]["checkpoint_catalog"])).write_text(
        json.dumps(catalog, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return catalog
