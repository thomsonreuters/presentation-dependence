"""Collect and aggregate direct-evaluation run metrics."""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

from presentation_dependence.eval.dataset_catalog import load_dataset_population
from presentation_dependence.utils.run_paths import latest_complete_trial

from .direct_eval import DirectEvalStageError, plan_direct_eval
from .reporting_cohorts import load_reporting_qids, require_cohort_coverage
from .source_bindings import (
    load_stage_bindings,
    reject_unknown_bindings,
    require_bindings_or_discovery,
    resolve_bound_run,
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise DirectEvalStageError(f"Expected JSON object: {path}")
    return value


def _metric_means(metrics: Mapping[str, Any]) -> dict[str, float]:
    return {
        str(key).removeprefix("mean_"): float(value)
        for key, value in metrics.items()
        if str(key).startswith("mean_") and isinstance(value, (int, float))
    }


def _psi_means(psi: Mapping[str, Any]) -> dict[str, float | int | None]:
    aggregate = psi.get("aggregate") or {}
    keys = (
        "mean_tau_based_psi",
        "mean_kendall_tau",
        "mean_zeng_psi",
        "mean_score_variance",
        "mean_rank_variance",
        "mean_per_perm_ndcg",
        "mean_worst_perm_ndcg",
        "mean_best_perm_ndcg",
        "K_permutations",
    )
    return {key: aggregate.get(key) for key in keys}


def _reporting_metric_means(
    trial: Path,
    metrics: Mapping[str, Any],
    reporting_qids: set[str],
) -> dict[str, float]:
    path = trial / "all_queries_eval_results.jsonl"
    if not path.is_file():
        if int(metrics["n_queries"]) == len(reporting_qids):
            return _metric_means(metrics)
        raise DirectEvalStageError(
            f"{trial} needs {path.name} for the post-hoc {len(reporting_qids)}-query cohort; "
            "refetch with --no-prune-detailed"
        )
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[str(row["qid"])] = row
    require_cohort_coverage(set(rows), reporting_qids, source=path)
    result = {}
    for aggregate_key in metrics:
        if not str(aggregate_key).startswith("mean_"):
            continue
        per_query_key = str(aggregate_key).removeprefix("mean_")
        values = [
            float(rows[qid][per_query_key])
            for qid in reporting_qids
            if isinstance(rows[qid].get(per_query_key), int | float)
        ]
        if len(values) != len(reporting_qids):
            raise DirectEvalStageError(f"{path} has {len(values)}/{len(reporting_qids)} values for {per_query_key}")
        result[per_query_key] = statistics.fmean(values)
    return result


def _reporting_psi_means(
    trial: Path,
    psi: Mapping[str, Any],
    reporting_qids: set[str],
) -> dict[str, float | int | None]:
    path = trial / "psi/psi_per_query.json"
    aggregate = psi.get("aggregate") or {}
    if not path.is_file():
        if int(aggregate.get("n_queries", -1)) == len(reporting_qids):
            return _psi_means(psi)
        raise DirectEvalStageError(
            f"{trial} needs psi/psi_per_query.json for the post-hoc {len(reporting_qids)}-query cohort"
        )
    rows = _read_json(path)
    require_cohort_coverage(set(rows), reporting_qids, source=path)
    field_map = {
        "mean_tau_based_psi": "tau_based_psi",
        "mean_kendall_tau": "kendall_tau",
        "mean_zeng_psi": "zeng_psi",
        "mean_score_variance": "score_variance",
        "mean_rank_variance": "rank_variance",
        "mean_per_perm_ndcg": "per_perm_ndcg_mean",
        "mean_worst_perm_ndcg": "per_perm_ndcg_min",
        "mean_best_perm_ndcg": "per_perm_ndcg_max",
    }
    result: dict[str, float | int | None] = {"K_permutations": aggregate.get("K_permutations")}
    for aggregate_key, per_query_key in field_map.items():
        values = [
            float(rows[qid][per_query_key])
            for qid in reporting_qids
            if isinstance(rows[qid].get(per_query_key), int | float) and math.isfinite(float(rows[qid][per_query_key]))
        ]
        result[aggregate_key] = statistics.fmean(values) if values else None
    return result


def _reporting_self_consistency(
    trial: Path,
    headline: Mapping[str, Any],
    reporting_qids: set[str],
) -> dict[str, Any]:
    path = trial / "psi/sc_per_query.json"
    if not path.is_file():
        by_k = headline.get("by_K") or {}
        if by_k and all(int(block.get("n_queries", -1)) == len(reporting_qids) for block in by_k.values()):
            return dict(by_k)
        raise DirectEvalStageError(
            f"{trial} needs psi/sc_per_query.json for the post-hoc {len(reporting_qids)}-query cohort"
        )
    per_query_by_k = _read_json(path)
    result = {}
    for k, per_query in per_query_by_k.items():
        require_cohort_coverage(set(per_query), reporting_qids, source=path)
        metrics = {}
        metric_names = sorted({str(metric) for qid in reporting_qids for metric in (per_query[qid] or {})})
        for metric in metric_names:
            values = [
                float(per_query[qid][metric])
                for qid in reporting_qids
                if isinstance(per_query[qid].get(metric), int | float)
            ]
            metrics[metric] = {
                "mean": statistics.fmean(values) if values else None,
                "std": statistics.stdev(values) if len(values) > 1 else 0.0 if values else None,
                "n": len(values),
            }
        result[str(k)] = {
            "n_queries": len(reporting_qids),
            "n_docs_total": None,
            "metrics": metrics,
        }
    return result


def _collect_row(
    job: Mapping[str, Any],
    trial: Path,
    reporting: tuple[set[str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    metrics = _read_json(trial / "metrics.json")
    psi_path = trial / "psi/psi_metrics.json"
    reporting_qids = reporting[0] if reporting is not None else None
    metric_means = (
        _reporting_metric_means(trial, metrics, reporting_qids)
        if reporting_qids is not None
        else _metric_means(metrics)
    )
    psi = _read_json(psi_path) if psi_path.is_file() else None
    robustness = (
        _reporting_psi_means(trial, psi, reporting_qids)
        if reporting_qids is not None and psi is not None
        else _psi_means(psi)
        if psi is not None
        else None
    )
    row = {
        "config_id": job["id"],
        "variant": job["variant"],
        "training_seed": job.get("training_seed"),
        "dataset": job["dataset"],
        "kind": job["kind"],
        "run_dir": str(trial),
        "n_queries": len(reporting_qids) if reporting_qids is not None else int(metrics["n_queries"]),
        "metrics": metric_means,
        "robustness": robustness,
        "reporting_cohort": reporting[1] if reporting is not None else None,
    }
    if job.get("first_stage") is not None:
        row["first_stage"] = str(job["first_stage"])
    sc_path = trial / "psi/sc_metrics.json"
    if sc_path.is_file():
        sc = _read_json(sc_path)
        row["self_consistency"] = (
            _reporting_self_consistency(trial, sc, reporting_qids) if reporting_qids is not None else sc.get("by_K")
        )
    return row


def _aggregate_rows(
    rows: list[dict[str, Any]], quality_metric: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_seed: dict[tuple[str, int | None], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_seed[(row["variant"], row["training_seed"])].append(row)
    per_seed: list[dict[str, Any]] = []
    for (variant, seed), group in sorted(by_seed.items(), key=lambda item: (item[0][0], item[0][1] or -1)):
        tau_values = [
            float(row["robustness"]["mean_tau_based_psi"])
            for row in group
            if row["robustness"] and row["robustness"].get("mean_tau_based_psi") is not None
        ]
        per_seed.append(
            {
                "variant": variant,
                "training_seed": seed,
                "datasets": len(group),
                "quality_metric": quality_metric,
                "quality_mean": statistics.mean(row["metrics"][quality_metric] for row in group),
                "tau_psi_mean": (statistics.mean(tau_values) if tau_values else None),
            }
        )
    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in per_seed:
        by_variant[row["variant"]].append(row)
    aggregate: list[dict[str, Any]] = []
    for variant, group in sorted(by_variant.items()):
        quality = [float(row["quality_mean"]) for row in group]
        tau = [float(row["tau_psi_mean"]) for row in group if row["tau_psi_mean"] is not None]
        aggregate.append(
            {
                "variant": variant,
                "training_seeds": [row["training_seed"] for row in group if row["training_seed"] is not None],
                "quality_metric": quality_metric,
                "quality_mean": statistics.mean(quality),
                "quality_sample_sd": (statistics.stdev(quality) if len(quality) > 1 else None),
                "tau_psi_mean": statistics.mean(tau) if tau else None,
                "tau_psi_sample_sd": (statistics.stdev(tau) if len(tau) > 1 else None),
            }
        )
    return per_seed, aggregate


def collect_direct_results(
    pipeline: Mapping[str, Any],
    project_root: Path,
    *,
    runs_root: Path | None = None,
    output_root: Path | None = None,
    allow_missing: bool = False,
    source_bindings: Path | None = None,
    discover_latest: bool = False,
    verify_hashes: bool = False,
    include_controls: bool = False,
) -> dict[str, Any]:
    """Collect direct-eval runs from bindings or opted-in discovery."""
    plan = plan_direct_eval(pipeline, project_root, include_controls=include_controls)
    direct = pipeline["direct_eval"]
    population = {
        str(row["id"]): row
        for row in load_dataset_population(
            str(direct["population"]),
            configs_root=project_root / "configs",
        )
    }
    source_root = runs_root or project_root / "runs"
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
        {str(job["id"]) for job in jobs},
        task=str(pipeline["task"]["id"]),
        stage="direct-eval",
    )
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for job in jobs:
        require_psi = not bool(job.get("order_invariant"))
        if require_psi and str(pipeline["task"]["id"]) in {"passage-reranking", "multi-document-qa"}:
            required = (
                "metrics.json",
                "psi/psi_metrics.json",
                "psi/beta_gamma_scores.parquet",
            )
        else:
            required = ("metrics.json", "psi/psi_metrics.json") if require_psi else ("metrics.json",)
        binding = bindings.get(str(job["id"]))
        if binding:
            trial = resolve_bound_run(
                project_root,
                binding,
                required_files=required,
                verify_hashes=verify_hashes,
            )
        elif discover_latest:
            trial = latest_complete_trial(source_root, job["id"], required_files=required)
        else:
            trial = None
        if trial is None:
            missing.append(job["id"])
            continue
        if str(job["dataset"]) not in population:
            raise DirectEvalStageError(f"Unknown direct-eval dataset: {job['dataset']}")
        reporting = load_reporting_qids(pipeline, project_root, str(job["dataset"]))
        rows.append(_collect_row(job, trial, reporting))
    if missing and not allow_missing:
        raise DirectEvalStageError(f"Missing {len(missing)} direct-eval runs; first={missing[0]}")
    quality_metric = str(direct["evaluation"]["headline_quality"])
    per_seed, aggregate = _aggregate_rows(rows, quality_metric)
    destination = output_root or project_root / str(direct["outputs"]["root"])
    destination.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": 1,
        "task": pipeline["task"]["id"],
        "stage": "direct-eval",
        "quality_metric": quality_metric,
        "reporting_cohorts": {
            str(row["dataset"]): row["reporting_cohort"] for row in rows if row.get("reporting_cohort") is not None
        },
        "expected_jobs": len(jobs),
        "completed_jobs": len(rows),
        "missing": missing,
        "per_dataset": rows,
        "per_seed": per_seed,
        "aggregate": aggregate,
    }
    for key, filename in (
        ("per_dataset", direct["outputs"]["per_dataset"]),
        ("per_seed", direct["outputs"]["per_seed"]),
        ("aggregate", direct["outputs"]["aggregate"]),
    ):
        (destination / str(filename)).write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "task": pipeline["task"]["id"],
                    "reporting_cohorts": result["reporting_cohorts"],
                    key: result[key],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    (destination / str(direct["outputs"]["run_manifest"])).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task": pipeline["task"]["id"],
                "reporting_cohorts": result["reporting_cohorts"],
                "runs": [
                    {
                        key: row[key]
                        for key in (
                            "config_id",
                            "variant",
                            "training_seed",
                            "dataset",
                            "run_dir",
                        )
                    }
                    for row in rows
                ],
                "missing": missing,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return result
