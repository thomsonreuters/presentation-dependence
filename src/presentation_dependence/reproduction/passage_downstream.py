"""Retained-set downstream evaluation for passage reranking."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from presentation_dependence.eval.aligned_scores import load_aligned_scores
from presentation_dependence.eval.retained_set_reduction import (
    common_query_ids,
    development_retention,
    evaluate_matched_retention,
    has_expected_presentations,
    reduce_retained_set,
    repeat_single_presentation,
)
from presentation_dependence.utils.run_paths import latest_complete_trial
from presentation_dependence.utils.trec import read_qrels

from .downstream_common import (
    DownstreamStageError,
    aggregate_downstream_rows,
    downstream_stage,
    write_downstream_outputs,
)
from .reduction import (
    materialize_reduction,
    plan_reduction,
    stage_population,
    validate_reduction,
)
from .source_bindings import (
    load_stage_bindings,
    reject_unknown_bindings,
    require_bindings_or_discovery,
    resolve_bound_run,
)


def plan_passage_downstream(pipeline: Mapping[str, Any], project_root: Path) -> dict[str, Any]:
    """Plan one retained-set reduction per selected direct-eval job."""
    return plan_reduction(
        pipeline,
        project_root,
        consumer=str(downstream_stage(pipeline)["consumer"]["id"]),
    )


def materialize_passage_downstream(
    pipeline: Mapping[str, Any],
    project_root: Path,
    *,
    build_root: Path | None = None,
) -> dict[str, Any]:
    """Write the deterministic retained-set reduction plan."""
    return materialize_reduction(
        pipeline,
        project_root,
        consumer=str(downstream_stage(pipeline)["consumer"]["id"]),
        build_root=build_root,
    )


def validate_passage_downstream(
    pipeline: Mapping[str, Any],
    project_root: Path,
    *,
    build_root: Path | None = None,
    check_data: bool = True,
    check_source_runs: bool = False,
) -> dict[str, Any]:
    """Validate the plan, qrels, and optional direct-eval score-log runs."""
    return validate_reduction(
        pipeline,
        project_root,
        consumer=str(downstream_stage(pipeline)["consumer"]["id"]),
        build_root=build_root,
        check_data=check_data,
        check_source_runs=check_source_runs,
    )


def _pair_trained_queries(entries: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for entry in entries:
        job = entry["job"]
        if job["kind"] == "trained":
            grouped.setdefault((int(job["training_seed"]), str(job["dataset"])), []).append(entry)
    for group in grouped.values():
        common = common_query_ids([entry["scores"] for entry in group])
        for entry in group:
            entry["scores"] = {qid: entry["scores"][qid] for qid in sorted(common)}


def _resolve_source_run(
    project_root: Path,
    source: Path,
    job: Mapping[str, Any],
    bindings: Mapping[str, Mapping[str, Any]],
    *,
    discover_latest: bool,
    verify_hashes: bool,
) -> Path | None:
    binding = bindings.get(str(job["source_config_id"]))
    if binding:
        return resolve_bound_run(
            project_root,
            binding,
            required_files=binding.get("required_artifacts") or ("metrics.json",),
            verify_hashes=verify_hashes,
        )
    if discover_latest:
        return latest_complete_trial(source, job["source_config_id"])
    return None


def _apply_matched_retention(
    rows: list[dict[str, Any]],
    prepared_by_id: Mapping[str, Mapping[str, Any]],
) -> None:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row["kind"] == "trained":
            grouped.setdefault((int(row["training_seed"]), str(row["dataset"])), []).append(row)
    for group in grouped.values():
        reference = next((row for row in group if row["variant"] == "k1-sft"), None)
        if reference is None:
            continue
        prepared = prepared_by_id[reference["config_id"]]
        target = development_retention(prepared)
        for row in group:
            matched = evaluate_matched_retention(prepared_by_id[row["config_id"]], target)
            row["matched_retention_threshold"] = matched["threshold"]
            row["matched_retention_target"] = target
            row["metrics"]["matched_retention_mean_pairwise_jaccard"] = matched["test_aggregate"][
                "mean_mean_pairwise_jaccard"
            ]
            row["metrics"]["matched_retention_set_flip_rate"] = matched["test_aggregate"]["mean_set_flip_rate"]


def collect_passage_downstream(
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
    """Reduce direct score logs into retained-set downstream metrics."""
    stage = downstream_stage(pipeline)
    plan = plan_passage_downstream(pipeline, project_root)
    population = {str(row["id"]): row for row in stage_population(pipeline, project_root)}
    source = runs_root or project_root / "runs"
    bindings = load_stage_bindings(
        project_root,
        str(pipeline["task"]["id"]),
        "direct-eval",
        path=source_bindings,
    )
    require_bindings_or_discovery(
        bindings,
        discover_latest=discover_latest,
        task=str(pipeline["task"]["id"]),
        stage="direct-eval",
    )
    jobs = plan["jobs"]
    reject_unknown_bindings(
        bindings,
        {str(job["source_config_id"]) for job in jobs},
        task=str(pipeline["task"]["id"]),
        stage="direct-eval",
    )
    measures = list(map(str, stage["evaluation"]["measures"]))
    protocol = stage["protocol"]
    entries = []
    missing = []
    for job in jobs:
        run_dir = _resolve_source_run(
            project_root,
            source,
            job,
            bindings,
            discover_latest=discover_latest,
            verify_hashes=verify_hashes,
        )
        if run_dir is None:
            missing.append(job["id"])
            continue
        scores = load_aligned_scores(run_dir)
        if not scores:
            missing.append(job["id"])
            continue
        if job["order_invariant"]:
            scores = repeat_single_presentation(scores, int(protocol["order_invariant_presentations"]))
        if not has_expected_presentations(scores, int(protocol["scorer_presentations"])):
            missing.append(job["id"])
            continue
        dataset = population[job["dataset"]]
        entries.append(
            {
                "job": job,
                "run_dir": run_dir,
                "scores": scores,
                "qrels": read_qrels(project_root / str(dataset["qrels_path"])),
            }
        )
    _pair_trained_queries(entries)
    rows = []
    prepared_by_id = {}
    base_measures = [measure for measure in measures if not measure.startswith("matched_retention_")]
    for entry in entries:
        job = entry["job"]
        try:
            reduced = reduce_retained_set(
                entry["scores"],
                entry["qrels"],
                relevance_cutoff=int(stage["consumer"]["relevance_cutoff"]),
                development_fraction=float(protocol["development_fraction"]),
                split_seed=int(protocol["split_seed"]),
                grid_size=int(protocol["threshold_grid_size"]),
            )
        except ValueError as exc:
            raise DownstreamStageError(str(exc)) from exc
        prepared_by_id[job["id"]] = reduced
        aggregate = reduced["test_aggregate"]
        rows.append(
            {
                "config_id": job["id"],
                "source_config_id": job["source_config_id"],
                "variant": job["variant"],
                "training_seed": job["training_seed"],
                "dataset": job["dataset"],
                "kind": job["kind"],
                "consumer": stage["consumer"]["id"],
                "threshold": reduced["threshold"],
                "development_mean_f1": reduced["development_mean_f1"],
                "development_queries": reduced["development_queries"],
                "test_queries": reduced["test_queries"],
                "metrics": {measure: aggregate[f"mean_{measure}"] for measure in base_measures},
                "run_dir": str(entry["run_dir"]),
            }
        )
        for measure in measures:
            rows[-1]["metrics"].setdefault(measure, None)
    _apply_matched_retention(rows, prepared_by_id)
    if missing and not allow_missing:
        raise DownstreamStageError(f"Missing {len(missing)} passage downstream sources; first={missing[0]}")
    per_seed, aggregate = aggregate_downstream_rows(rows, measures)
    result = {
        "schema_version": 1,
        "task": pipeline["task"]["id"],
        "stage": "downstream-eval",
        "expected_jobs": len(jobs),
        "completed_jobs": len(rows),
        "missing": missing,
        "per_consumer": rows,
        "per_seed": per_seed,
        "aggregate": aggregate,
    }
    destination = output_root or project_root / str(stage["outputs"]["root"])
    write_downstream_outputs(destination, stage["outputs"], result)
    return result
