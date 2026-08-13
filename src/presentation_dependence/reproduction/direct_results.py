"""Collect and aggregate direct-evaluation run metrics."""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

from presentation_dependence.utils.run_paths import latest_complete_trial

from .direct_eval import DirectEvalStageError, plan_direct_eval
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


def _collect_row(job: Mapping[str, Any], trial: Path) -> dict[str, Any]:
    metrics = _read_json(trial / "metrics.json")
    psi_path = trial / "psi/psi_metrics.json"
    row = {
        "config_id": job["id"],
        "variant": job["variant"],
        "training_seed": job.get("training_seed"),
        "dataset": job["dataset"],
        "kind": job["kind"],
        "run_dir": str(trial),
        "n_queries": int(metrics["n_queries"]),
        "metrics": _metric_means(metrics),
        "robustness": (_psi_means(_read_json(psi_path)) if psi_path.is_file() else None),
    }
    if job.get("first_stage") is not None:
        row["first_stage"] = str(job["first_stage"])
    sc_path = trial / "psi/sc_metrics.json"
    if sc_path.is_file():
        row["self_consistency"] = _read_json(sc_path).get("by_K")
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
        rows.append(_collect_row(job, trial))
    if missing and not allow_missing:
        raise DirectEvalStageError(f"Missing {len(missing)} direct-eval runs; first={missing[0]}")
    direct = pipeline["direct_eval"]
    quality_metric = str(direct["evaluation"]["headline_quality"])
    per_seed, aggregate = _aggregate_rows(rows, quality_metric)
    destination = output_root or project_root / str(direct["outputs"]["root"])
    destination.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": 1,
        "task": pipeline["task"]["id"],
        "stage": "direct-eval",
        "quality_metric": quality_metric,
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
