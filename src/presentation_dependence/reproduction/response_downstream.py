"""Response-selection downstream reductions over direct-evaluation scores."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any, Mapping

from presentation_dependence.eval.aligned_scores import load_aligned_scores
from presentation_dependence.eval.response_selection import (
    load_fixture_pids,
    response_selection_metrics,
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
from .reporting_cohorts import load_reporting_qids, require_cohort_coverage
from .source_bindings import (
    load_stage_bindings,
    reject_unknown_bindings,
    require_bindings_or_discovery,
    resolve_bound_run,
)


def plan_response_downstream(pipeline: Mapping[str, Any], project_root: Path) -> dict[str, Any]:
    """Plan one selection reduction for each selected direct-eval job."""
    return plan_reduction(pipeline, project_root, consumer="response-selection")


def materialize_response_downstream(
    pipeline: Mapping[str, Any],
    project_root: Path,
    *,
    build_root: Path | None = None,
) -> dict[str, Any]:
    """Write the deterministic reduction plan; no inference config is needed."""
    return materialize_reduction(
        pipeline,
        project_root,
        consumer="response-selection",
        build_root=build_root,
    )


def validate_response_downstream(
    pipeline: Mapping[str, Any],
    project_root: Path,
    *,
    build_root: Path | None = None,
    check_data: bool = True,
    check_source_runs: bool = False,
) -> dict[str, Any]:
    """Validate the plan, response datasets, and optional direct-eval score logs."""
    return validate_reduction(
        pipeline,
        project_root,
        consumer="response-selection",
        build_root=build_root,
        check_data=check_data,
        check_source_runs=check_source_runs,
        data_keys=("run_path", "qrels_path"),
    )


def _order_invariant_metrics(
    run_dir: Path,
    reporting_qids: set[str] | None = None,
) -> dict[str, float | int]:
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    quality = float(metrics["mean_ndcg_cut_1"])
    n_queries = int(metrics["n_queries"])
    if reporting_qids is not None:
        per_query_path = run_dir / "all_queries_eval_results.jsonl"
        if per_query_path.is_file():
            per_query = {}
            for line in per_query_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    if "ndcg_cut_1" in row:
                        per_query[str(row["qid"])] = float(row["ndcg_cut_1"])
            require_cohort_coverage(set(per_query), reporting_qids, source=per_query_path)
            quality = statistics.fmean(per_query[qid] for qid in reporting_qids)
            n_queries = len(reporting_qids)
        elif n_queries != len(reporting_qids):
            raise DownstreamStageError(
                f"{run_dir} has {n_queries} aggregate queries but reporting requires "
                f"{len(reporting_qids)}; refetch with --no-prune-detailed"
            )
    return {
        "n_queries": n_queries,
        "selection_flip_rate": 0.0,
        "selected_quality_flip_rate": 0.0,
        "preference_pair_flip_rate": 0.0,
        "mean_selected_quality": quality,
        "canonical_selected_quality": quality,
        "benchmark_score_range": 0.0,
    }


def collect_response_downstream(
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
    """Reduce response selections and write downstream result files."""
    stage = downstream_stage(pipeline)
    plan = plan_response_downstream(pipeline, project_root)
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
    rows = []
    missing = []
    for job in jobs:
        binding = bindings.get(str(job["source_config_id"]))
        if binding:
            run_dir = resolve_bound_run(
                project_root,
                binding,
                required_files=binding.get("required_artifacts") or ("metrics.json",),
                verify_hashes=verify_hashes,
            )
        elif discover_latest:
            run_dir = latest_complete_trial(source, job["source_config_id"])
        else:
            run_dir = None
        if run_dir is None:
            missing.append(job["id"])
            continue
        dataset = population[job["dataset"]]
        reporting = load_reporting_qids(pipeline, project_root, str(job["dataset"]))
        reporting_qids = reporting[0] if reporting is not None else None
        if job["order_invariant"]:
            metrics = _order_invariant_metrics(run_dir, reporting_qids)
        else:
            aligned = load_aligned_scores(run_dir)
            if not aligned:
                missing.append(job["id"])
                continue
            fixture = load_fixture_pids(project_root / str(dataset["run_path"]))
            qrels = read_qrels(project_root / str(dataset["qrels_path"]))
            if reporting_qids is not None:
                require_cohort_coverage(set(aligned), reporting_qids, source=run_dir)
                require_cohort_coverage(
                    set(fixture),
                    reporting_qids,
                    source=project_root / str(dataset["run_path"]),
                )
                aligned = {qid: aligned[qid] for qid in reporting_qids}
                fixture = {qid: fixture[qid] for qid in reporting_qids}
                qrels = {qid: qrels.get(qid, {}) for qid in reporting_qids}
            try:
                metrics = response_selection_metrics(
                    aligned,
                    fixture,
                    qrels,
                )
            except ValueError as exc:
                raise DownstreamStageError(str(exc)) from exc
        rows.append(
            {
                "config_id": job["id"],
                "source_config_id": job["source_config_id"],
                "variant": job["variant"],
                "training_seed": job["training_seed"],
                "dataset": job["dataset"],
                "consumer": "argmax-response-selection",
                "n_queries": metrics["n_queries"],
                "metrics": {metric: float(metrics[metric]) for metric in measures},
                "run_dir": str(run_dir),
                "reporting_cohort": reporting[1] if reporting is not None else None,
            }
        )
    if missing and not allow_missing:
        raise DownstreamStageError(f"Missing {len(missing)} response downstream sources; first={missing[0]}")
    headline_datasets = set(map(str, stage["protocol"]["headline_datasets"]))
    headline_rows = [row for row in rows if row["dataset"] in headline_datasets]
    per_seed, aggregate = aggregate_downstream_rows(headline_rows, measures)
    result = {
        "schema_version": 1,
        "task": pipeline["task"]["id"],
        "stage": "downstream-eval",
        "reporting_cohorts": {
            str(row["dataset"]): row["reporting_cohort"] for row in rows if row.get("reporting_cohort") is not None
        },
        "expected_jobs": len(jobs),
        "completed_jobs": len(rows),
        "aggregation_datasets": sorted(headline_datasets),
        "missing": missing,
        "per_consumer": rows,
        "per_seed": per_seed,
        "aggregate": aggregate,
    }
    destination = output_root or project_root / str(stage["outputs"]["root"])
    write_downstream_outputs(destination, stage["outputs"], result)
    return result
