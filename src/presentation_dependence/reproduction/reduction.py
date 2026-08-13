"""Plan inference-free downstream reductions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from presentation_dependence.eval.dataset_catalog import load_dataset_population
from presentation_dependence.utils.run_paths import (
    latest_complete_trial,
    selected_variants,
)

from .direct_eval import plan_direct_eval
from .downstream_common import DownstreamStageError, downstream_stage


def plan_reduction(
    pipeline: Mapping[str, Any],
    project_root: Path,
    *,
    consumer: str,
) -> dict[str, Any]:
    """Expand selected direct-evaluation jobs into offline reductions."""
    stage = downstream_stage(pipeline)
    selected = selected_variants(stage)
    task = str(pipeline["task"]["id"])
    jobs = []
    for source in plan_direct_eval(pipeline, project_root)["jobs"]:
        if source["variant"] not in selected:
            continue
        parts = [f"{task}-downstream", str(source["variant"])]
        if source.get("training_seed") is not None:
            parts.append(f"seed{source['training_seed']}")
        parts.append(str(source["dataset"]))
        jobs.append(
            {
                "id": "--".join(parts),
                "variant": source["variant"],
                "training_seed": source.get("training_seed"),
                "dataset": source["dataset"],
                "source_config_id": source["id"],
                "kind": source["kind"],
                "order_invariant": bool(source.get("order_invariant")),
            }
        )
    return {
        "task": task,
        "stage": "downstream-eval",
        "consumer": consumer,
        "jobs": jobs,
        "job_count": len(jobs),
        "inference_jobs": 0,
    }


def materialize_reduction(
    pipeline: Mapping[str, Any],
    project_root: Path,
    *,
    consumer: str,
    build_root: Path | None = None,
) -> dict[str, Any]:
    """Write the reduction plan and return its output paths."""
    stage = downstream_stage(pipeline)
    plan = plan_reduction(pipeline, project_root, consumer=consumer)
    root = build_root or project_root / str(stage["outputs"]["root"])
    root.mkdir(parents=True, exist_ok=True)
    path = root / str(stage["outputs"].get("reduction_plan", "reduction_plan.json"))
    path.write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {**plan, "root": str(root), "reduction_plan": str(path)}


def stage_population(pipeline: Mapping[str, Any], project_root: Path) -> list[dict[str, Any]]:
    """Load the downstream stage dataset population."""
    stage = downstream_stage(pipeline)
    return load_dataset_population(str(stage["population"]), configs_root=project_root / "configs")


def validate_reduction(
    pipeline: Mapping[str, Any],
    project_root: Path,
    *,
    consumer: str,
    build_root: Path | None = None,
    check_data: bool = True,
    check_source_runs: bool = False,
    data_keys: tuple[str, ...] = ("qrels_path",),
) -> dict[str, Any]:
    """Validate a reduction plan, qrels, and optional direct-eval score logs."""
    stage = downstream_stage(pipeline)
    plan = plan_reduction(pipeline, project_root, consumer=consumer)
    root = build_root or project_root / str(stage["outputs"]["root"])
    plan_name = str(stage["outputs"].get("reduction_plan", "reduction_plan.json"))
    missing_data = []
    if check_data:
        for dataset in stage_population(pipeline, project_root):
            for key in data_keys:
                path = project_root / str(dataset[key])
                if not path.is_file():
                    missing_data.append(str(path))
    missing_runs = []
    if check_source_runs:
        runs_root = project_root / "runs"
        missing_runs = [
            job["source_config_id"]
            for job in plan["jobs"]
            if latest_complete_trial(runs_root, job["source_config_id"]) is None
        ]
    plan_missing = not (root / plan_name).is_file()
    if plan_missing or missing_data or missing_runs:
        raise DownstreamStageError(
            f"{pipeline['task']['id']} reduction validation failed: "
            f"plan={int(plan_missing)}, data={len(missing_data)}, "
            f"source_runs={len(missing_runs)}"
        )
    return {
        "task": pipeline["task"]["id"],
        "stage": "downstream-eval",
        "jobs": len(plan["jobs"]),
        "inference_jobs": 0,
        "data_checked": check_data,
        "source_runs_checked": check_source_runs,
        "valid": True,
    }
