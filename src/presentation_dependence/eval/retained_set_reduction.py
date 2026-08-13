"""End-to-end retained-set threshold reduction over aligned scorer outputs."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from presentation_dependence.eval.threshold_stability import (
    build_grid,
    evaluate_at_threshold,
    mean_canonical_retention,
    tune_threshold_f1,
    tune_threshold_retention,
)


def repeat_single_presentation(
    scores: dict[str, dict[str, list[float]]], count: int
) -> dict[str, dict[str, list[float]]]:
    """Represent an order-invariant scorer as identical presentations."""
    return {
        qid: {docid: ([values[0]] * count if len(values) == 1 else values) for docid, values in documents.items()}
        for qid, documents in scores.items()
    }


def has_expected_presentations(scores: Mapping[str, Mapping[str, list[float]]], expected: int) -> bool:
    """Return whether every aligned document has exactly `expected` scores."""
    lengths = {len(values) for documents in scores.values() for values in documents.values()}
    return lengths == {expected}


def common_query_ids(
    score_maps: list[Mapping[str, Any]],
) -> set[str]:
    """Return the paired query intersection across scorer variants."""
    if not score_maps:
        return set()
    return set.intersection(*(set(scores) for scores in score_maps))


def _split_qids(qids: list[str], development_fraction: float, seed: int) -> tuple[list[str], list[str]]:
    order = sorted(qids)
    np.random.default_rng(seed).shuffle(order)
    n_development = max(1, int(round(len(order) * development_fraction)))
    return sorted(order[:n_development]), sorted(order[n_development:])


def reduce_retained_set(
    scores: dict[str, dict[str, list[float]]],
    qrels: Mapping[str, Mapping[str, int]],
    *,
    relevance_cutoff: int,
    development_fraction: float,
    split_seed: int,
    grid_size: int,
) -> dict[str, Any]:
    """Tune a presentation-0 threshold on dev and evaluate held-out stability."""
    qids = [qid for qid, documents in scores.items() if documents]
    if len(qids) < 2:
        raise ValueError("Retained-set reduction needs at least two queries")
    development_qids, test_qids = _split_qids(qids, development_fraction, split_seed)
    canonical = {
        qid: {docid: values[0] for docid, values in documents.items() if values} for qid, documents in scores.items()
    }
    relevant = {
        qid: {
            docid
            for docid, grade in qrels.get(qid, {}).items()
            if grade >= relevance_cutoff and docid in scores.get(qid, {})
        }
        for qid in qids
    }
    grid = build_grid(
        [score for documents in canonical.values() for score in documents.values()],
        n=grid_size,
    )
    tuned = tune_threshold_f1(
        {qid: canonical[qid] for qid in development_qids},
        {qid: relevant[qid] for qid in development_qids},
        grid,
    )
    test = evaluate_at_threshold(
        {qid: scores[qid] for qid in test_qids},
        relevant,
        float(tuned["tau_star"]),
    )
    return {
        "threshold": float(tuned["tau_star"]),
        "development_mean_f1": float(tuned["tau_star_mean_f1"]),
        "development_queries": len(development_qids),
        "test_queries": len(test_qids),
        "test_aggregate": test["aggregate"],
        "_development_qids": development_qids,
        "_test_qids": test_qids,
        "_canonical": canonical,
        "_relevant": relevant,
        "_scores": scores,
        "_grid": grid,
    }


def development_retention(prepared: Mapping[str, Any]) -> float:
    """Return mean dev retention at a prepared cell's F1 threshold."""
    return mean_canonical_retention(
        {qid: prepared["_canonical"][qid] for qid in prepared["_development_qids"]},
        float(prepared["threshold"]),
    )


def evaluate_matched_retention(prepared: Mapping[str, Any], target: float) -> dict[str, Any]:
    """Retune a prepared cell to a shared dev-retention target."""
    tuned = tune_threshold_retention(
        {qid: prepared["_canonical"][qid] for qid in prepared["_development_qids"]},
        target,
        prepared["_grid"],
    )
    test = evaluate_at_threshold(
        {qid: prepared["_scores"][qid] for qid in prepared["_test_qids"]},
        prepared["_relevant"],
        float(tuned["tau_star"]),
    )
    return {
        "threshold": float(tuned["tau_star"]),
        "development_mean_retention": float(tuned["tau_star_mean_retention"]),
        "test_aggregate": test["aggregate"],
    }
