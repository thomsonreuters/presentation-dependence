"""General analysis primitives for representative appendix checks."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from presentation_dependence.eval.score_distribution import summarize_aligned_query
from presentation_dependence.eval.threshold_stability import (
    mean_pairwise_jaccard as _mean_pairwise_jaccard,
)
from presentation_dependence.reader.verdict_eval import (
    verdict_flip_rate as _verdict_flip_rate,
)


def _finite_vector(values: Sequence[object], *, label: str) -> np.ndarray:
    parsed = np.asarray([float(str(value)) for value in values], dtype=float)
    if parsed.ndim != 1 or parsed.size == 0 or not np.all(np.isfinite(parsed)):
        raise ValueError(f"{label} requires a non-empty finite vector")
    return parsed


def paired_bootstrap_ci(differences: Sequence[object], *, samples: int = 20000, seed: int = 4401) -> list[float]:
    """Paired query-bootstrap percentile interval for representative checks."""
    values = _finite_vector(differences, label="bootstrap differences")
    if samples < 1:
        raise ValueError("bootstrap samples must be positive")
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=float)
    chunk_size = max(1, min(samples, 2_000_000 // values.size))
    for start in range(0, samples, chunk_size):
        stop = min(samples, start + chunk_size)
        indices = rng.integers(
            0,
            values.size,
            size=(stop - start, values.size),
        )
        estimates[start:stop] = values[indices].mean(axis=1)
    return [float(value) for value in np.quantile(estimates, [0.025, 0.975])]


def _parse_hierarchical_values(per_dataset: Mapping[str, Sequence[object]]) -> dict[str, np.ndarray]:
    """Validate and normalize dataset-indexed bootstrap values."""
    parsed: dict[str, np.ndarray] = {}
    for dataset, values in per_dataset.items():
        name = str(dataset)
        if name in parsed:
            raise ValueError(f"duplicate dataset after string normalization: {name!r}")
        parsed[name] = _finite_vector(values, label=f"{name} differences")
    if not parsed:
        raise ValueError("hierarchical bootstrap requires a dataset")
    return parsed


def _hierarchical_bootstrap_ci_from_parsed(
    parsed: Mapping[str, np.ndarray],
    *,
    samples: int,
    seed: int,
) -> list[float]:
    """Bootstrap the equal-dataset mean from validated query vectors."""
    if samples < 1:
        raise ValueError("bootstrap samples must be positive")
    datasets = sorted(parsed)
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=float)
    for sample in range(samples):
        selected = rng.choice(datasets, size=len(datasets), replace=True)
        dataset_means = []
        for dataset in selected:
            values = parsed[str(dataset)]
            indices = rng.integers(0, values.size, size=values.size)
            dataset_means.append(float(np.mean(values[indices])))
        estimates[sample] = float(np.mean(dataset_means))
    return [float(value) for value in np.quantile(estimates, [0.025, 0.975])]


def hierarchical_bootstrap_ci(
    per_dataset: Mapping[str, Sequence[object]],
    *,
    samples: int = 20000,
    seed: int = 4401,
) -> list[float]:
    """Resample datasets equally, then paired queries within each dataset."""
    return _hierarchical_bootstrap_ci_from_parsed(
        _parse_hierarchical_values(per_dataset),
        samples=samples,
        seed=seed,
    )


def tau_psi_from_aligned_scores(
    queries: Mapping[str, Mapping[str, Sequence[float]]],
) -> dict[str, Any]:
    """Recompute query-balanced tau-PSI from retained aligned score vectors."""
    if not queries:
        raise ValueError("tau-PSI requires at least one query")
    per_query = {
        str(query_id): float(summarize_aligned_query(scores)["tau_psi_docid_tiebreak"])
        for query_id, scores in queries.items()
    }
    return {
        "tau_psi": statistics.fmean(per_query.values()),
        "per_query": per_query,
        "n_queries": len(per_query),
    }


def verdict_flip_rate(values: Sequence[str | None]) -> float:
    """Delegate verdict instability to the production reader implementation."""
    return _verdict_flip_rate(values)


def mean_pairwise_jaccard(sets: Sequence[Sequence[object]]) -> float:
    """Delegate retained-set stability to the established implementation."""
    parsed = [{str(value) for value in values} for values in sets]
    value = float(_mean_pairwise_jaccard(parsed))
    if not math.isfinite(value):
        raise ValueError("Jaccard requires at least two retained sets")
    return value


def gain_histogram(gains: Sequence[object], *, bins: Sequence[object]) -> dict[str, Any]:
    """Per-query gain histogram and non-negative share for representative checks."""
    parsed = _finite_vector(gains, label="gain histogram")
    edges = _finite_vector(bins, label="histogram bins")
    if len(edges) < 2 or not np.all(np.diff(edges) > 0):
        raise ValueError("histogram bins must be strictly increasing")
    counts, resolved_edges = np.histogram(parsed, bins=edges)
    centers = (resolved_edges[:-1] + resolved_edges[1:]) / 2.0
    return {
        "mean_gain": float(parsed.mean()),
        "median_gain": float(np.median(parsed)),
        "nonnegative_fraction": float(np.mean(parsed >= 0)),
        "negative_fraction": float(np.mean(parsed < 0)),
        "bin_edges": resolved_edges.tolist(),
        "bin_centers": centers.tolist(),
        "counts": counts.astype(int).tolist(),
        "n_queries": len(parsed),
    }


def clustered_linear_calibration(
    x_values: Sequence[object],
    y_values: Sequence[object],
    groups: Sequence[object],
) -> dict[str, Any]:
    """OLS with CR1 errors clustered by model (representative calibration)."""
    from scipy.stats import t

    x = _finite_vector(x_values, label="calibration x")
    y = _finite_vector(y_values, label="calibration y")
    parsed_groups = np.asarray([str(value) for value in groups])
    if len(x) != len(y) or len(x) != len(parsed_groups):
        raise ValueError("calibration x, y, and groups must align")
    if len(x) <= 2 or np.allclose(x, x[0]):
        raise ValueError("clustered calibration requires varying cells")
    design = np.column_stack([np.ones(len(x)), x])
    bread = np.linalg.inv(design.T @ design)
    beta = bread @ (design.T @ y)
    residual = y - design @ beta
    unique = sorted(set(parsed_groups))
    if len(unique) < 2:
        raise ValueError("clustered calibration requires multiple groups")
    meat = np.zeros((2, 2), dtype=float)
    for group in unique:
        selected = parsed_groups == group
        score = design[selected].T @ residual[selected]
        meat += np.outer(score, score)
    n = len(x)
    k = design.shape[1]
    g = len(unique)
    covariance = (g / (g - 1)) * ((n - 1) / (n - k)) * bread @ meat @ bread
    slope_se = float(np.sqrt(max(0.0, covariance[1, 1])))
    critical = float(t.ppf(0.975, g - 1))
    return {
        "n_cells": n,
        "n_clusters": g,
        "intercept": float(beta[0]),
        "slope": float(beta[1]),
        "slope_se": slope_se,
        "slope_ci95": [
            float(beta[1] - critical * slope_se),
            float(beta[1] + critical * slope_se),
        ],
        "covariance": covariance.tolist(),
        "correction": "CR1: G/(G-1) * (N-1)/(N-k), t critical with G-1 df",
    }


def compute_analysis(operation: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Dispatch one declared analysis operation."""
    if operation == "hierarchical-bootstrap":
        parsed = _parse_hierarchical_values(payload["per_dataset"])
        samples = int(payload.get("samples", 20000))
        seed = int(payload.get("seed", 4401))
        return {
            # Match the bootstrap estimand: every dataset contributes one mean,
            # independent of how many paired query differences it contains.
            "mean": statistics.fmean(float(values.mean()) for values in parsed.values()),
            "ci95": _hierarchical_bootstrap_ci_from_parsed(
                parsed,
                samples=samples,
                seed=seed,
            ),
        }
    if operation == "mean":
        values = _finite_vector(payload["values"], label="mean")
        return {
            "mean": float(values.mean()),
            "sample_sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "n": len(values),
            "values": values.tolist(),
        }
    if operation == "pairwise-jaccard":
        return {
            "mean_pairwise_jaccard": mean_pairwise_jaccard(payload["sets"]),
            "presentations": len(payload["sets"]),
        }
    if operation == "gain-histogram":
        return gain_histogram(payload["gains"], bins=payload["bins"])
    if operation == "clustered-calibration":
        return clustered_linear_calibration(payload["x"], payload["y"], payload["groups"])
    raise ValueError(f"unknown analysis operation: {operation}")
