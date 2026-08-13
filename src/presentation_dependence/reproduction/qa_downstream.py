"""Frozen-reader downstream evaluation for multi-document QA."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from presentation_dependence.utils.run_paths import latest_run_artifact, selected_variants

from .common import prepare_generated_directory
from .direct_eval import plan_direct_eval
from .downstream_common import (
    DownstreamStageError,
    aggregate_downstream_rows,
    downstream_stage,
    write_downstream_outputs,
)
from .pipelines import load_passage_reranking
from .reduction import stage_population
from .source_bindings import (
    load_stage_bindings,
    reject_unknown_bindings,
    require_bindings_or_discovery,
    resolve_bound_run,
)


def _job_id(job: Mapping[str, Any]) -> str:
    parts = [
        "multi-document-qa-downstream",
        str(job["variant"]),
    ]
    if job.get("training_seed") is not None:
        parts.append(f"seed{job['training_seed']}")
    parts.append(str(job["dataset"]))
    return "--".join(parts)


def plan_qa_downstream(pipeline: Mapping[str, Any], project_root: Path) -> dict[str, Any]:
    """Expand direct-evaluation scorers into frozen-reader jobs."""
    stage = downstream_stage(pipeline)
    selected = selected_variants(stage)
    answer_controls = set(map(str, stage.get("controls") or []))
    direct = plan_direct_eval(pipeline, project_root, include_controls=bool(answer_controls))
    answer_jobs = [
        {
            "id": _job_id(job),
            "variant": job["variant"],
            "training_seed": job.get("training_seed"),
            "dataset": job["dataset"],
            "scorer_config_id": job["id"],
            "kind": job["kind"],
            "consumer": "answer-reader",
            "phase": "R3",
        }
        for job in direct["jobs"]
        if job["variant"] in selected | answer_controls
    ]
    passage = load_passage_reranking(project_root)
    passage["direct_eval"]["checkpoint_catalog"] = stage["verdict_checkpoint_catalog"]
    verdict_jobs = [
        {
            "id": _job_id(job),
            "variant": job["variant"],
            "training_seed": job.get("training_seed"),
            "dataset": job["dataset"],
            "scorer_config_id": job["id"],
            "kind": job["kind"],
            "consumer": "verdict-reader",
            "phase": "V3",
        }
        for job in plan_direct_eval(passage, project_root)["jobs"]
        if job["variant"] in selected and job["dataset"] == stage["protocol"]["verdict_dataset"]
    ]
    jobs = [*answer_jobs, *verdict_jobs]
    return {
        "task": pipeline["task"]["id"],
        "stage": "downstream-eval",
        "consumer": "frozen-reader",
        "jobs": jobs,
        "job_count": len(jobs),
    }


def _reader_config(job: Mapping[str, Any], stage: Mapping[str, Any]) -> dict[str, Any]:
    is_verdict = job["phase"] == "V3"
    config = {
        "id": job["id"],
        "phase": job["phase"],
        "reader": dict(stage["verdict_reader"] if is_verdict else stage["reader"]),
        "dataset": job["dataset"],
        "k_values": list(stage["protocol"]["verdict_k_values" if is_verdict else "k_values"]),
        "canonical_perm": int(stage["protocol"]["canonical_presentation"]),
        "scorer": {
            "exp_id": job["scorer_config_id"],
            "channel_id": job["scorer_config_id"],
            "variant": job["variant"],
            "training_seed": job.get("training_seed"),
        },
        "execution": dict(stage["execution"]),
    }
    if is_verdict:
        config["verdict_qids_filename"] = "verdict_qids.txt"
        config["scorer"]["score_log_filename"] = "beta_gamma_scores.parquet"
    return config


def materialize_qa_downstream(
    pipeline: Mapping[str, Any],
    project_root: Path,
    *,
    build_root: Path | None = None,
) -> dict[str, Any]:
    """Write reader launch configs and their sweep manifest."""
    stage = downstream_stage(pipeline)
    plan = plan_qa_downstream(pipeline, project_root)
    root = build_root or project_root / str(stage["outputs"]["root"])
    config_dir = prepare_generated_directory(root / str(stage["outputs"]["configs_dir"]))
    config_paths = []
    for job in plan["jobs"]:
        path = config_dir / f"{job['id']}.yaml"
        path.write_text(
            yaml.safe_dump(
                _reader_config(job, stage),
                sort_keys=False,
                allow_unicode=True,
            ),
            encoding="utf-8",
        )
        config_paths.append(str(path))
    sweep_path = root / str(stage["outputs"]["sweep"])
    sweep_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "task": pipeline["task"]["id"],
                "stage": "downstream-eval",
                "launcher": "scripts/run_reader.py",
                "configs": config_paths,
                "max_parallel": int(stage["execution"]["max_parallel"]),
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return {
        **plan,
        "root": str(root),
        "configs": config_paths,
        "sweep": str(sweep_path),
    }


def _population(pipeline: Mapping[str, Any], project_root: Path) -> list[dict[str, Any]]:
    return stage_population(pipeline, project_root)


def _score_log(runs_root: Path, config_id: str) -> Path | None:
    return latest_run_artifact(runs_root, config_id, ("*/psi/beta_gamma_scores.parquet",))


def validate_qa_downstream(
    pipeline: Mapping[str, Any],
    project_root: Path,
    *,
    build_root: Path | None = None,
    check_data: bool = True,
    check_scorer_runs: bool = False,
) -> dict[str, Any]:
    """Validate reader configs, QA sidecars, and optionally scorer score logs."""
    stage = downstream_stage(pipeline)
    plan = plan_qa_downstream(pipeline, project_root)
    root = build_root or project_root / str(stage["outputs"]["root"])
    missing_configs = [
        job["id"]
        for job in plan["jobs"]
        if not (root / str(stage["outputs"]["configs_dir"]) / f"{job['id']}.yaml").is_file()
    ]
    missing_data: list[str] = []
    if check_data:
        for dataset in _population(pipeline, project_root):
            data_dir = (project_root / str(dataset["run_path"])).parent
            for filename in ("fixture.jsonl", "qrels.txt", "answers.jsonl"):
                path = data_dir / filename
                if not path.is_file():
                    missing_data.append(str(path))
        verdict_dir = project_root / "data/beir-v1.0.0-climate-fever-test"
        for filename in (
            "fixture.jsonl",
            "qrels.txt",
            "verdicts.jsonl",
            "verdict_qids.txt",
        ):
            path = verdict_dir / filename
            if not path.is_file():
                missing_data.append(str(path))
    missing_scorers = []
    if check_scorer_runs:
        runs_root = project_root / "runs"
        missing_scorers = [
            job["scorer_config_id"] for job in plan["jobs"] if _score_log(runs_root, job["scorer_config_id"]) is None
        ]
    if missing_configs or missing_data or missing_scorers:
        raise DownstreamStageError(
            "QA downstream validation failed: "
            f"configs={len(missing_configs)}, data={len(missing_data)}, "
            f"scorer_runs={len(missing_scorers)}"
        )
    return {
        "task": pipeline["task"]["id"],
        "stage": "downstream-eval",
        "configs": len(plan["jobs"]),
        "primary_jobs": len(plan["jobs"]),
        "data_checked": check_data,
        "scorer_runs_checked": check_scorer_runs,
        "valid": True,
    }


def _latest_reader_metrics(runs_root: Path, job: Mapping[str, Any]) -> Path | None:
    relative = {
        "R3": "*/reader/reader_metrics.json",
        "V3": "*/verdict/verdict_metrics.json",
    }[job["phase"]]
    return latest_run_artifact(runs_root, str(job["id"]), (relative,))


def _bound_reader_metrics(
    project_root: Path,
    binding: Mapping[str, Any],
    job: Mapping[str, Any],
    *,
    verify_hashes: bool,
) -> Path:
    relative = {
        "R3": "reader/reader_metrics.json",
        "V3": "verdict/verdict_metrics.json",
    }[job["phase"]]
    run_dir = resolve_bound_run(
        project_root,
        binding,
        required_files=(relative,),
        verify_hashes=verify_hashes,
    )
    return run_dir / relative


def collect_qa_downstream(
    pipeline: Mapping[str, Any],
    project_root: Path,
    *,
    runs_root: Path | None = None,
    output_root: Path | None = None,
    allow_missing: bool = False,
    source_bindings: Path | None = None,
    discover_latest: bool = False,
    verify_hashes: bool = False,
) -> dict[str, Any]:
    """Collect frozen-reader metrics into downstream result files."""
    stage = downstream_stage(pipeline)
    plan = plan_qa_downstream(pipeline, project_root)
    source = runs_root or project_root / "runs"
    bindings = load_stage_bindings(
        project_root,
        str(pipeline["task"]["id"]),
        "downstream-eval",
        path=source_bindings,
    )
    require_bindings_or_discovery(
        bindings,
        discover_latest=discover_latest,
        task=str(pipeline["task"]["id"]),
        stage="downstream-eval",
    )
    reject_unknown_bindings(
        bindings,
        {str(job["id"]) for job in plan["jobs"]},
        task=str(pipeline["task"]["id"]),
        stage="downstream-eval",
    )
    rows = []
    missing = []
    for job in plan["jobs"]:
        binding = bindings.get(str(job["id"]))
        path = (
            _bound_reader_metrics(
                project_root,
                binding,
                job,
                verify_hashes=verify_hashes,
            )
            if binding
            else (_latest_reader_metrics(source, job) if discover_latest else None)
        )
        if path is None:
            missing.append(job["id"])
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        is_verdict = job["phase"] == "V3"
        headline_k = str(stage["protocol"]["verdict_headline_k" if is_verdict else "headline_k"])
        measures = list(
            map(
                str,
                stage["evaluation"]["verdict_measures" if is_verdict else "measures"],
            )
        )
        corpus = payload["corpus"][headline_k]
        rows.append(
            {
                "config_id": job["id"],
                "variant": job["variant"],
                "training_seed": job["training_seed"],
                "dataset": job["dataset"],
                "consumer": job["consumer"],
                "consumer_model": payload["meta"]["reader"],
                "reader_k": int(headline_k),
                "n_queries": int(payload["meta"]["n_queries"]),
                "metrics": {metric: float(corpus[metric]) for metric in measures},
                "run_dir": str(path.parent.parent),
            }
        )
    if missing and not allow_missing:
        raise DownstreamStageError(f"Missing {len(missing)} QA downstream runs; first={missing[0]}")
    per_seed = []
    aggregate = []
    for consumer, measure_key in (
        ("answer-reader", "measures"),
        ("verdict-reader", "verdict_measures"),
    ):
        consumer_rows = [row for row in rows if row["consumer"] == consumer]
        seed_rows, aggregate_rows = aggregate_downstream_rows(
            consumer_rows,
            list(map(str, stage["evaluation"][measure_key])),
        )
        for row in [*seed_rows, *aggregate_rows]:
            row["consumer"] = consumer
        per_seed.extend(seed_rows)
        aggregate.extend(aggregate_rows)
    result = {
        "schema_version": 1,
        "task": pipeline["task"]["id"],
        "stage": "downstream-eval",
        "expected_jobs": len(plan["jobs"]),
        "completed_jobs": len(rows),
        "missing": missing,
        "per_consumer": rows,
        "per_seed": per_seed,
        "aggregate": aggregate,
    }
    destination = output_root or project_root / str(stage["outputs"]["root"])
    write_downstream_outputs(destination, stage["outputs"], result)
    return result
