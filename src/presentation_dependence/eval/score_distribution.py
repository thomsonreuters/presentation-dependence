"""Score-scale diagnostics for aligned scorer presentations.

These metrics separate two quantities that raw score variance can conflate:

* residual variance: how much one document's score changes across presentations;
* score dispersion: how far documents are separated within one presentation.

Adjacent-rank gaps are computed after sorting each presentation independently.
They describe the score geometry seen by a one-presentation consumer.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from presentation_dependence.eval.psi import mean_pairwise_tau
from presentation_dependence.eval.topk_stability import mean_pairwise_rbo


def adjacent_rank_margins(scores: Mapping[str, float]) -> np.ndarray:
    """Return non-negative score gaps between adjacent ranks.

    Scores are ordered descending with lexical document id as the deterministic
    tie-break. The tie-break does not change gap values, but pins rank semantics
    for callers that use the same ordering to compute rank stability.
    """
    if len(scores) < 2:
        raise ValueError("adjacent-rank margins require at least two documents")
    ordered = sorted(
        ((str(doc_id), float(score)) for doc_id, score in scores.items()),
        key=lambda item: (-item[1], item[0]),
    )
    values = np.asarray([score for _, score in ordered], dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("scores must be finite")
    return values[:-1] - values[1:]


def summarize_aligned_query(
    scores: Mapping[str, Sequence[float]],
) -> dict[str, Any]:
    """Summarize one query's documents across aligned presentations.

    ``scores[doc_id][presentation]`` must be rectangular, finite, and contain at
    least two documents and two presentations.
    """
    if len(scores) < 2:
        raise ValueError("aligned scores require at least two documents")

    doc_ids = sorted(str(doc_id) for doc_id in scores)
    vectors = [np.asarray(scores[doc_id], dtype=float) for doc_id in doc_ids]
    lengths = {len(vector) for vector in vectors}
    if len(lengths) != 1:
        raise ValueError("aligned score vectors must have equal lengths")
    n_presentations = lengths.pop()
    if n_presentations < 2:
        raise ValueError("aligned scores require at least two presentations")

    matrix = np.stack(vectors, axis=0)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("aligned scores must be finite")

    # Existing PSI estimand: E_d Var_pi[s(d, pi)].
    residual_variance = float(np.mean(np.var(matrix, axis=1, ddof=0)))

    # A query-wide score offset can vary by presentation without changing any
    # rank. Remove that offset before forming the scale-normalized residual.
    centered = matrix - np.mean(matrix, axis=0, keepdims=True)
    centered_residual_variance = float(np.mean(np.var(centered, axis=1, ddof=0)))

    presentation_variances = np.var(matrix, axis=0, ddof=0)
    score_dispersion_variance = float(np.mean(presentation_variances))
    score_dispersion_std = float(np.mean(np.sqrt(presentation_variances)))
    order_marginal_variance = float(np.var(np.mean(matrix, axis=1), ddof=0))

    all_margins: list[float] = []
    top10_internal_margins: list[float] = []
    cutoff10_margins: list[float] = []
    rankings: list[list[str]] = []
    for presentation in range(n_presentations):
        score_map = {doc_id: float(matrix[index, presentation]) for index, doc_id in enumerate(doc_ids)}
        rankings.append(sorted(score_map, key=lambda doc_id: (-score_map[doc_id], doc_id)))
        margins = adjacent_rank_margins(score_map)
        all_margins.extend(float(value) for value in margins)
        top10_internal_margins.extend(float(value) for value in margins[:9])
        if len(margins) >= 10:
            cutoff10_margins.append(float(margins[9]))

    all_array = np.asarray(all_margins, dtype=float)
    top10_array = np.asarray(top10_internal_margins, dtype=float)
    cutoff10_array = np.asarray(cutoff10_margins, dtype=float)
    mean_tau = mean_pairwise_tau(rankings)
    mean_rbo_at_10 = mean_pairwise_rbo(
        rankings,
        depth=min(10, len(doc_ids)),
        persistence=0.9,
    )

    normalized_residual = (
        float(np.sqrt(centered_residual_variance / score_dispersion_variance))
        if score_dispersion_variance > 0
        else None
    )

    return {
        "n_documents": len(doc_ids),
        "n_presentations": n_presentations,
        "tau_psi_docid_tiebreak": float((1.0 - mean_tau) / 2.0),
        "mean_rbo_at_10": mean_rbo_at_10,
        "rbo_instability_at_10": 1.0 - mean_rbo_at_10,
        "residual_score_variance": residual_variance,
        "centered_residual_score_variance": centered_residual_variance,
        "score_dispersion_variance": score_dispersion_variance,
        "score_dispersion_std": score_dispersion_std,
        "order_marginal_score_variance": order_marginal_variance,
        "normalized_residual_rmse": normalized_residual,
        "mean_adjacent_margin_all": float(np.mean(all_array)),
        "median_adjacent_margin_all": float(np.median(all_array)),
        "mean_adjacent_margin_top10_internal": float(np.mean(top10_array)),
        "median_adjacent_margin_top10_internal": float(np.median(top10_array)),
        "mean_rank10_cutoff_margin": (float(np.mean(cutoff10_array)) if cutoff10_array.size else None),
        "median_rank10_cutoff_margin": (float(np.median(cutoff10_array)) if cutoff10_array.size else None),
        "exact_adjacent_tie_rate": float(np.mean(all_array == 0.0)),
        "score_p10": float(np.quantile(matrix, 0.10)),
        "score_p50": float(np.quantile(matrix, 0.50)),
        "score_p90": float(np.quantile(matrix, 0.90)),
    }
