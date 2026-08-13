"""Score-level diagnostics for the fixed-weights co-presence estimand."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np


RANDOM_PERMUTATION_RE = re.compile(r"^permutation_(\d+)_random_s(\d+)$")


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _rank_result_scores(path: Path) -> dict[str, float]:
    """Read doc scores from a RankResult without reconstructing input order."""
    payload = _read_json(path)
    passages = payload.get("top_k_psgs")
    scores = payload.get("scores_init_order")
    if not isinstance(passages, list) or not isinstance(scores, list):
        raise ValueError(f"{path} has no scalar-score RankResult")
    if len(passages) != len(scores):
        raise ValueError(f"{path}: {len(passages)} ranked passages but {len(scores)} input scores")
    # top_k_psgs is sorted, but each entry carries the same scalar score as its
    # input-order counterpart.  Reading that explicit score avoids incorrectly
    # zipping sorted pids with scores_init_order.
    result = {str(item["pid"]): float(item["score"]) for item in passages}
    if len(result) != len(passages):
        raise ValueError(f"{path} contains duplicate document ids")
    return result


def load_b1_scores(run_dir: Path) -> dict[str, dict[str, float]]:
    """Load one B=1 scalar score per query and document."""
    per_query = run_dir / "per_query_results"
    if not per_query.is_dir():
        raise FileNotFoundError(f"missing unpruned B=1 detail: {per_query}")
    output: dict[str, dict[str, float]] = {}
    for qdir in sorted(path for path in per_query.iterdir() if path.is_dir()):
        detail = qdir / "detailed_results.json"
        if detail.is_file():
            output[qdir.name] = _rank_result_scores(detail)
    if not output:
        raise ValueError(f"no B=1 score-bearing queries in {per_query}")
    return output


def _aligned_scores(path: Path, *, k: int) -> dict[str, float]:
    payload = _read_json(path)
    if payload.get("perturbation") != "random_shuffle":
        raise ValueError(f"{path} is not random-shuffle aligned scores")
    scores = payload.get("scores")
    if not isinstance(scores, dict):
        raise ValueError(f"{path} has no scores mapping")
    output: dict[str, float] = {}
    for docid, values in scores.items():
        if not isinstance(values, list) or len(values) < k:
            continue
        output[str(docid)] = float(np.mean(np.asarray(values[:k], dtype=float)))
    return output


def _permutation_scores(qdir: Path, *, k: int) -> dict[str, float]:
    """Fallback reconstruction from the first K random-permutation RankResults."""
    candidates: list[tuple[int, int, Path]] = []
    for pdir in qdir.iterdir():
        if not pdir.is_dir():
            continue
        match = RANDOM_PERMUTATION_RE.match(pdir.name)
        detail = pdir / "detailed_results.json"
        if match and detail.is_file():
            candidates.append((int(match.group(1)), int(match.group(2)), detail))
    # The SC protocol truncates in stored permutation/seed order.
    candidates.sort(key=lambda item: (item[0], item[1]))
    if len(candidates) < k:
        return {}
    by_doc: dict[str, list[float]] = {}
    for _, _, detail in candidates[:k]:
        for docid, score in _rank_result_scores(detail).items():
            by_doc.setdefault(docid, []).append(score)
    return {docid: float(np.mean(values)) for docid, values in by_doc.items() if len(values) == k}


def load_b20_order_averaged_scores(
    run_dir: Path,
    *,
    k: int = 10,
) -> tuple[dict[str, dict[str, float]], dict[str, int]]:
    """Load mean-per-document B=20 scores and source-coverage counts."""
    per_query = run_dir / "psi" / "per_query_results"
    if not per_query.is_dir():
        raise FileNotFoundError(f"missing unpruned B=20 PSI detail: {per_query}")
    output: dict[str, dict[str, float]] = {}
    aligned_count = 0
    fallback_count = 0
    incomplete_count = 0
    for qdir in sorted(path for path in per_query.iterdir() if path.is_dir()):
        aligned_path = qdir / "aligned_scores.json"
        scores: dict[str, float] = {}
        if aligned_path.is_file():
            scores = _aligned_scores(aligned_path, k=k)
            if scores:
                aligned_count += 1
        if not scores:
            scores = _permutation_scores(qdir, k=k)
            if scores:
                fallback_count += 1
        if scores:
            output[qdir.name] = scores
        else:
            incomplete_count += 1
    if not output:
        raise ValueError(f"no complete K={k} score-bearing queries in {per_query}")
    return output, {
        "queries_from_aligned_scores": aligned_count,
        "queries_from_permutation_details": fallback_count,
        "queries_incomplete": incomplete_count,
        "queries_loaded": len(output),
    }


def summarize_score_pairs(
    queries: Mapping[
        str,
        tuple[Mapping[str, float], Mapping[str, float]],
    ],
) -> dict[str, Any]:
    """Summarize c_hat = mean_K f_B20 - f_B1 over aligned query/doc pairs.

    The score-scale normalizer is the population standard deviation of the
    pooled B=1 and order-averaged B=20 score values.  The offset variance uses
    document weighting: each query offset is repeated for every aligned
    document in that query, which makes the decomposition obey the law of total
    variance even if candidate counts differ.
    """
    b1_all: list[float] = []
    b20_all: list[float] = []
    c_all: list[float] = []
    offsets_repeated: list[float] = []
    residuals: list[float] = []
    docs_per_query: list[int] = []
    query_offsets: dict[str, float] = {}

    for qid, (b1_scores, b20_scores) in queries.items():
        docids = sorted(set(b1_scores) & set(b20_scores))
        if not docids:
            continue
        b1 = np.asarray([b1_scores[docid] for docid in docids], dtype=float)
        b20 = np.asarray([b20_scores[docid] for docid in docids], dtype=float)
        c_hat = b20 - b1
        offset = float(c_hat.mean())
        b1_all.extend(b1.tolist())
        b20_all.extend(b20.tolist())
        c_all.extend(c_hat.tolist())
        offsets_repeated.extend([offset] * len(docids))
        residuals.extend((c_hat - offset).tolist())
        docs_per_query.append(len(docids))
        query_offsets[qid] = offset

    if not c_all:
        raise ValueError("no aligned query/document score pairs")
    b1_array = np.asarray(b1_all)
    b20_array = np.asarray(b20_all)
    c_array = np.asarray(c_all)
    offsets_array = np.asarray(offsets_repeated)
    residual_array = np.asarray(residuals)
    pooled_score_sd = float(np.std(np.concatenate([b1_array, b20_array]), ddof=0))
    if pooled_score_sd == 0.0:
        raise ValueError("score-scale normalizer is zero")
    c_variance = float(np.var(c_array, ddof=0))
    offset_variance = float(np.var(offsets_array, ddof=0))
    residual_variance = float(np.var(residual_array, ddof=0))
    correlation = (
        float(np.corrcoef(b1_array, b20_array)[0, 1])
        if np.std(b1_array) > 0 and np.std(b20_array) > 0
        else float("nan")
    )
    return {
        "n_queries": len(docs_per_query),
        "n_documents": len(c_all),
        "docs_per_query_min": min(docs_per_query),
        "docs_per_query_max": max(docs_per_query),
        "pooled_score_sd": pooled_score_sd,
        "b1_score_sd": float(np.std(b1_array, ddof=0)),
        "b20_order_averaged_score_sd": float(np.std(b20_array, ddof=0)),
        "signed_mean_c_hat": float(c_array.mean()),
        "signed_mean_c_hat_over_score_sd": float(c_array.mean() / pooled_score_sd),
        "mean_absolute_c_hat": float(np.mean(np.abs(c_array))),
        "mean_absolute_c_hat_over_score_sd": float(np.mean(np.abs(c_array)) / pooled_score_sd),
        "b1_b20_score_correlation": correlation,
        "variance_c_hat": c_variance,
        "variance_query_offset": offset_variance,
        "variance_within_query_residual": residual_variance,
        "query_offset_variance_fraction": (float(offset_variance / c_variance) if c_variance > 0 else float("nan")),
        "within_query_variance_fraction": (float(residual_variance / c_variance) if c_variance > 0 else float("nan")),
        "query_offsets": query_offsets,
    }


def summarize_dataset_balanced(
    datasets: Mapping[
        str,
        Mapping[str, tuple[Mapping[str, float], Mapping[str, float]]],
    ],
) -> dict[str, Any]:
    """Summarize scores with equal total weight for every dataset.

    Within a dataset every aligned document receives equal weight.  This makes
    the suite summary follow a macro-over-datasets convention while retaining a
    per-document estimand inside each dataset.
    """
    b1_values: list[float] = []
    b20_values: list[float] = []
    c_values: list[float] = []
    offset_values: list[float] = []
    residual_values: list[float] = []
    weights: list[float] = []
    n_queries = 0
    n_documents = 0

    valid: dict[
        str,
        list[tuple[np.ndarray, np.ndarray, np.ndarray, float]],
    ] = {}
    for dataset, queries in datasets.items():
        records: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]] = []
        for b1_scores, b20_scores in queries.values():
            docids = sorted(set(b1_scores) & set(b20_scores))
            if not docids:
                continue
            b1 = np.asarray([b1_scores[docid] for docid in docids], dtype=float)
            b20 = np.asarray([b20_scores[docid] for docid in docids], dtype=float)
            c_hat = b20 - b1
            records.append((b1, b20, c_hat, float(c_hat.mean())))
        if records:
            valid[dataset] = records
    if not valid:
        raise ValueError("no aligned dataset/query/document score pairs")

    dataset_weight = 1.0 / len(valid)
    for records in valid.values():
        dataset_docs = sum(len(record[0]) for record in records)
        document_weight = dataset_weight / dataset_docs
        for b1, b20, c_hat, offset in records:
            b1_values.extend(b1.tolist())
            b20_values.extend(b20.tolist())
            c_values.extend(c_hat.tolist())
            offset_values.extend([offset] * len(c_hat))
            residual_values.extend((c_hat - offset).tolist())
            weights.extend([document_weight] * len(c_hat))
            n_queries += 1
            n_documents += len(c_hat)

    b1_array = np.asarray(b1_values)
    b20_array = np.asarray(b20_values)
    c_array = np.asarray(c_values)
    offset_array = np.asarray(offset_values)
    residual_array = np.asarray(residual_values)
    weight_array = np.asarray(weights)

    def weighted_mean(values: np.ndarray) -> float:
        return float(np.average(values, weights=weight_array))

    def weighted_var(values: np.ndarray) -> float:
        mean = weighted_mean(values)
        return float(np.average((values - mean) ** 2, weights=weight_array))

    b1_mean = weighted_mean(b1_array)
    b20_mean = weighted_mean(b20_array)
    b1_var = weighted_var(b1_array)
    b20_var = weighted_var(b20_array)
    covariance = float(
        np.average(
            (b1_array - b1_mean) * (b20_array - b20_mean),
            weights=weight_array,
        )
    )
    # Give each arm half of the score-scale normalizer's mass.
    pooled_mean = 0.5 * (b1_mean + b20_mean)
    pooled_variance = 0.5 * float(np.average((b1_array - pooled_mean) ** 2, weights=weight_array)) + 0.5 * float(
        np.average((b20_array - pooled_mean) ** 2, weights=weight_array)
    )
    pooled_sd = float(np.sqrt(pooled_variance))
    c_mean = weighted_mean(c_array)
    c_variance = weighted_var(c_array)
    offset_variance = weighted_var(offset_array)
    residual_variance = weighted_var(residual_array)
    return {
        "weighting": "equal datasets, then equal aligned documents within dataset",
        "n_datasets": len(valid),
        "n_queries": n_queries,
        "n_documents": n_documents,
        "pooled_score_sd": pooled_sd,
        "b1_score_sd": float(np.sqrt(b1_var)),
        "b20_order_averaged_score_sd": float(np.sqrt(b20_var)),
        "signed_mean_c_hat": c_mean,
        "signed_mean_c_hat_over_score_sd": c_mean / pooled_sd,
        "mean_absolute_c_hat": weighted_mean(np.abs(c_array)),
        "mean_absolute_c_hat_over_score_sd": (weighted_mean(np.abs(c_array)) / pooled_sd),
        "b1_b20_score_correlation": covariance / np.sqrt(b1_var * b20_var),
        "variance_c_hat": c_variance,
        "variance_query_offset": offset_variance,
        "variance_within_query_residual": residual_variance,
        "query_offset_variance_fraction": offset_variance / c_variance,
        "within_query_variance_fraction": residual_variance / c_variance,
    }


def compare_run_scores(
    b1_run: Path,
    b20_run: Path,
    *,
    k: int = 10,
) -> tuple[dict[str, Any], dict[str, int]]:
    """Load, align, and summarize one dataset's B=1 and B=20 score artifacts."""
    b1 = load_b1_scores(b1_run)
    b20, coverage = load_b20_order_averaged_scores(b20_run, k=k)
    common_qids = sorted(set(b1) & set(b20))
    queries = {qid: (b1[qid], b20[qid]) for qid in common_qids}
    diagnostics = summarize_score_pairs(queries)
    coverage.update(
        {
            "queries_b1": len(b1),
            "queries_b20": len(b20),
            "queries_aligned": len(common_qids),
            "queries_b1_only": len(set(b1) - set(b20)),
            "queries_b20_only": len(set(b20) - set(b1)),
        }
    )
    return diagnostics, coverage
