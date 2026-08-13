"""Rank-space K-shot self-consistency for generative-listwise rerankers.

Score-space self-consistency (``presentation_dependence.eval.self_consistency``) averages
per-doc scalars across K permutations, then reranks. Generative-listwise bases
(e.g. RankZephyr) emit a textual permutation without per-document scalar scores
(``scores_init_order is None``), so score-space SC is undefined for them.
Instead aggregates the K output rankings with Borda mean rank and evaluates the
aggregate against qrels.

Aggregation (Borda / mean-rank):
    For each doc, take its 0-indexed position in every ranking it appears in,
    average those positions, and sort docs ascending by mean position (lower =
    better). Ties broken by docid for determinism. This is the rank-space
    counterpart to score averaging. It approximates the exact Kemeny central
    ranking used by Tang et al. at the B=20 list lengths evaluated here.

The K ``random_shuffle`` outputs produced by
:class:`presentation_dependence.eval.psi_manager.PsiExperimentRunner` are the rankings
needed for this calculation. No additional reranker calls are required.

K=1 reference:
    ``by_K[1]`` is the Borda aggregate of one shuffled pass, not the BM25-order
    pass. The SC lift uses the BM25-order base pass from ``ExperimentManager``.
    Callers must pass that nDCG to :func:`compute_sc_lift`; it cannot be
    recovered from shuffled rankings alone.

Returns the same ``by_K`` / ``headline_K`` / ``protocol`` schema as
``derive_sc_metrics`` so ``split_headline_and_verbose`` works on both.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pytrec_eval  # type: ignore


def borda_aggregate(rankings: Sequence[Sequence[str]]) -> list[str]:
    """Aggregate K rankings into one by mean rank position (Borda / mean-rank).

    Each ranking lists document IDs from best to worst. A document's mean position is the
    average of its 0-indexed positions across the rankings in which it appears
    (all rankings cover the same doc pool in the intended single-window use, so
    every doc appears in every ranking). Docs are sorted ascending by mean
    position; ties are broken by docid so the result is deterministic.
    """
    positions: dict[str, list[int]] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking):
            positions.setdefault(str(doc), []).append(rank)
    mean_pos = {doc: float(np.mean(pos)) for doc, pos in positions.items()}
    return sorted(mean_pos, key=lambda d: (mean_pos[d], d))


def _borda_run_for_K(
    rankings_per_qid: Mapping[str, Sequence[Sequence[str]]],
    K: int,
) -> dict[str, dict[str, float]]:
    """Build a pytrec_eval run from the first K rankings for each query.

    Queries with fewer than K rankings are excluded so every aggregate uses the
    same number of presentations.
    """
    run: dict[str, dict[str, float]] = {}
    for qid, rankings in rankings_per_qid.items():
        if len(rankings) < K:
            continue
        aggregate = borda_aggregate(rankings[:K])
        if not aggregate:
            continue
        # Descending score = better rank. -idx is strictly decreasing and only
        # the ordering matters to pytrec_eval's ndcg_cut.
        run[str(qid)] = {doc: float(-idx) for idx, doc in enumerate(aggregate)}
    return run


def derive_rank_sc_metrics(
    *,
    rankings_per_qid: Mapping[str, Sequence[Sequence[str]]],
    qrels: Mapping[str, Mapping[str, int]],
    measures: Sequence[str] | None = None,
    k_subsets: Sequence[int] | None = None,
) -> dict:
    """Compute rank-space (Borda) K-shot self-consistency metrics.

    Parameters
    ----------
    rankings_per_qid:
        ``{qid: [ranking_0, …, ranking_{K-1}]}`` where each ranking is a list
        of document IDs from best to worst. Position ``i`` must refer to the same
        permutation across queries; the PSI driver appends rankings in seed order.
    qrels:
        Parsed TREC qrels ``{qid: {docid: rel}}``.
    measures:
        ``pytrec_eval`` measure names (default ``["ndcg_cut_10"]``).
    k_subsets:
        K' truncation values; defaults to ``[full_K]``. Each K' ≤ full_K
        produces an independent Borda aggregate of the first K' rankings.

    Returns:
    -------
    dict
        ``{"by_K": {K: {n_queries, n_docs_total, metrics, per_query}}, ...},
        "headline_K": int | None, "protocol": {...}}``. This matches
        ``self_consistency.derive_sc_metrics`` and can be passed to
        ``split_headline_and_verbose``.
    """
    measures_list = list(measures) if measures is not None else ["ndcg_cut_10"]
    if not measures_list:
        raise ValueError("derive_rank_sc_metrics: `measures` must be non-empty")

    if not rankings_per_qid:
        return {
            "by_K": {},
            "headline_K": None,
            "protocol": _protocol_block(measures_list, requested_k=list(k_subsets or []), full_K=None),
        }

    full_K = max((len(r) for r in rankings_per_qid.values()), default=0)
    if k_subsets is None:
        ks = [full_K]
    else:
        ks = sorted({int(k) for k in k_subsets if 1 <= int(k) <= full_K})
    if not ks:
        return {
            "by_K": {},
            "headline_K": None,
            "protocol": _protocol_block(measures_list, requested_k=list(k_subsets or []), full_K=full_K),
        }

    qrels_dict: dict = {qid: dict(rels) for qid, rels in qrels.items()}
    measure_set = set(measures_list)
    by_K: dict[int, dict] = {}
    for K in ks:
        run = {qid: docs for qid, docs in _borda_run_for_K(rankings_per_qid, K).items() if qid in qrels_dict}
        if not run:
            by_K[int(K)] = {
                "n_queries": 0,
                "n_docs_total": 0,
                "metrics": {m: {"mean": None, "std": None, "n": 0} for m in measures_list},
                "per_query": {},
            }
            continue

        evaluator = pytrec_eval.RelevanceEvaluator(qrels_dict, measure_set)
        per_query = evaluator.evaluate(run)

        metrics_block: dict[str, dict] = {}
        for m in measures_list:
            vals = [
                float(per_query[qid][m]) for qid in per_query if m in per_query[qid] and per_query[qid][m] is not None
            ]
            if vals:
                metrics_block[m] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                    "n": len(vals),
                }
            else:
                metrics_block[m] = {"mean": None, "std": None, "n": 0}

        by_K[int(K)] = {
            "n_queries": len(per_query),
            "n_docs_total": sum(len(v) for v in run.values()),
            "metrics": metrics_block,
            "per_query": dict(per_query),
        }

    headline_K = max(by_K.keys()) if by_K else None
    return {
        "by_K": by_K,
        "headline_K": headline_K,
        "protocol": _protocol_block(measures_list, requested_k=list(k_subsets or []), full_K=full_K),
    }


def compute_sc_lift(
    rank_sc: dict,
    *,
    base_ndcg: float | None,
    measure: str = "ndcg_cut_10",
    k_aggregate: int | None = None,
) -> dict:
    """Compute nDCG lift over the K=1 BM25-order base pass.

    ``base_ndcg`` is the single-pass BM25-order quality (the
    ``ExperimentManager`` ``metrics.json`` mean for ``measure``). It is the
    K=1 reference and differs from ``by_K[1]``, which is one shuffled pass.
    ``k_aggregate`` defaults to the run's ``headline_K``.
    """
    by_K = rank_sc.get("by_K") or {}
    if k_aggregate is None:
        k_aggregate = rank_sc.get("headline_K")
    agg_block = by_K.get(int(k_aggregate)) if k_aggregate is not None else None
    agg_mean = (agg_block or {}).get("metrics", {}).get(measure, {}).get("mean") if agg_block else None
    lift = float(agg_mean) - float(base_ndcg) if (agg_mean is not None and base_ndcg is not None) else None
    return {
        "measure": measure,
        "k_aggregate": k_aggregate,
        "k_aggregate_ndcg": agg_mean,
        "k1_base_ndcg": base_ndcg,
        "k1_reference": "BM25-order single pass (ExperimentManager base metrics.json)",
        "sc_lift": lift,
    }


def _protocol_block(measures: Sequence[str], *, requested_k: list, full_K: int | None) -> dict:
    return {
        "aggregation": (
            "Borda / mean-rank: average each doc's 0-indexed position across the "
            "first K' random_shuffle output rankings; rerank ascending by mean "
            "position (ties by docid); evaluate via pytrec_eval."
        ),
        "space": "rank (no per-doc scalar; generative-listwise output rankings)",
        "kemeny_note": (
            "Borda is the cheap mean-rank approximation to an exact Kemeny "
            "central ranking (Tang et al. permutation self-consistency); adequate at B=20 list length."
        ),
        "k1_lift_reference": (
            "SC lift is computed against the BM25-order base pass (passed in "
            "separately); by_K[1] is a single shuffled pass, not that reference."
        ),
        "k_truncation": "First K' rankings in stored (seed) order; K' > full_K dropped.",
        "measures": list(measures),
        "ndcg_gain": "linear (rel / log2(i+2)); matches trec_eval ndcg_cut and EvalManager",
        "requested_k_subsets": list(requested_k),
        "full_K_observed": full_K,
        "metric_std_ddof": 1,
    }
