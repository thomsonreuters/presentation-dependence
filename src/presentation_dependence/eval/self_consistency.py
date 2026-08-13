"""Derive K-shot self-consistency metrics from per-permutation reranker scores.

For each query, self-consistency scores K shuffles of the candidate list,
averages each document's scores, and ranks by the mean. This matches
``Qwen3Reranker._rank_self_consistency``. The ``random_shuffle`` presentations
produced by :class:`presentation_dependence.eval.psi_manager.PsiExperimentRunner` use the
same Fisher-Yates seeding as the native path, so their aligned score vectors
support the same aggregation without additional reranker calls.

Consumes aligned per-document score lists
``(qid → docid → [s_1, …, s_K])`` plus parsed qrels and returns
``pytrec_eval``-derived metrics for one or more truncated subsets ``K' ≤ K``.
The PSI driver feeds it the in-memory data from a live run; the post-hoc
script (``scripts/analyze/derive_self_consistency_from_psi.py``) reconstructs
the same shape from on-disk artefacts.

The per-permutation nDCG distribution in ``psi.evaluate_psi`` measures the K
individual outputs. It is distinct from self-consistency, which averages score
vectors before ranking. A comparison with a separately launched native
self-consistency run requires matching seeds, populated ``scores_init_order``,
and rank-preserving post-processing. Both paths default to ``range(K)`` seeds,
and monotone transforms such as multiplying grade scores by 3 do not affect the
ranking. The default configurations satisfy these conditions for
batched-pointwise and scoring-listwise rerankers.

K-subset truncation rule: ``scores[:K']`` uses the first K' stored entries,
which the PSI driver appends in seed order. From a stored K=10 run, re-aggregate
to get the K' ∈ {2, 3, 5, 10} curve.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pytrec_eval  # type: ignore


def _mean_scores_truncated(
    scores_for_qid: Mapping[str, Sequence[float]],
    K: int,
) -> dict[str, float]:
    """Average each docid's scores over the first ``K`` shots.

    Documents with fewer than K scores are excluded. This can occur when a
    permutation failed for a docid (rare but possible if the reranker truncates
    the candidate list). pytrec_eval assigns no ranking credit to a document
    absent from the run.
    """
    out: dict[str, float] = {}
    for docid, scores in scores_for_qid.items():
        if len(scores) >= K:
            out[docid] = float(np.mean(scores[:K]))
    return out


def derive_sc_metrics(  # noqa: C901
    *,
    scores_per_qid: Mapping[str, Mapping[str, Sequence[float]]],
    qrels: Mapping[str, Mapping[str, int]],
    measures: Sequence[str] | None = None,
    k_subsets: Sequence[int] | None = None,
) -> dict:
    """Compute K-shot self-consistency quality metrics for one or more K subsets.

    Parameters
    ----------
    scores_per_qid:
        ``{qid: {docid: [score_for_perm_0, …, score_for_perm_{K-1}]}}``. Score
        order must be the same across docids within a query (i.e. position ``i``
        in every doc's list is the score from the same permutation).
    qrels:
        Parsed TREC qrels ``{qid: {docid: rel}}`` (e.g. from
        ``pytrec_eval.parse_qrel``).
    measures:
        ``pytrec_eval`` measure names (default ``["ndcg_cut_10"]``). Use the
        same names as ``eval.measures`` in the experiment YAML so the SC
        numbers are directly comparable to the K=1 baseline `metrics.json`.
    k_subsets:
        SC truncation values to compute. Defaults to ``[full_K]`` where
        ``full_K`` is the maximum number of scores stored per doc. Each
        requested ``K'`` ≤ full_K produces an independent SC reranking using
        the first ``K'`` per-doc scores; ``K' > full_K`` is silently dropped.

    Returns:
    -------
    dict
        Schema::

            {
              "by_K": {
                K: {
                  "n_queries": int,
                  "n_docs_total": int,
                  "metrics": {
                    measure: {"mean": float, "std": float, "n": int},
                    ...
                  },
                  "per_query": {qid: {measure: float, ...}},
                },
                ...
              },
              "headline_K": int | None,            # max evaluated K, for convenience
              "protocol": {...},                   # provenance + caveats
            }

        ``per_query`` is the verbose detail (one row per qid). Drivers
        typically split the headline ``by_K[*].metrics`` and the verbose
        ``per_query`` blocks into two artifact files (sc_metrics.json and
        sc_per_query.json), following the ``psi_metrics.json`` and
        ``psi_per_query.json`` convention.
    """
    measures_list = list(measures) if measures is not None else ["ndcg_cut_10"]
    if not measures_list:
        raise ValueError("derive_sc_metrics: `measures` must be non-empty")

    if not scores_per_qid:
        return {
            "by_K": {},
            "headline_K": None,
            "protocol": _protocol_block(measures_list, requested_k=list(k_subsets or []), full_K=None),
        }

    full_K = 0
    for q_scores in scores_per_qid.values():
        for s in q_scores.values():
            if len(s) > full_K:
                full_K = len(s)

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
        run: dict[str, dict[str, float]] = {}
        for qid, q_scores in scores_per_qid.items():
            if qid not in qrels_dict:
                continue
            mean_scores = _mean_scores_truncated(q_scores, K)
            if mean_scores:
                run[qid] = mean_scores

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


def _protocol_block(measures: Sequence[str], *, requested_k: list, full_K: int | None) -> dict:
    return {
        "aggregation": (
            "Mean of per-doc scores across the first K' random_shuffle permutations "
            "(seed-order); rerank by descending mean score; evaluate via pytrec_eval."
        ),
        "matches_native_sc_when": (
            "(a) PSI random_shuffle seeds match native self_consistency seeds "
            "(both default to range(K) — Fisher-Yates with same seed + same list "
            "length yields the same input permutation regardless of element "
            "identity); (b) reranker.scores_init_order is populated; (c) any "
            "post-aggregation in the native path is monotone (e.g. ×3.0 grade "
            "rescaling does not affect ranking)."
        ),
        "k_truncation": "First K' seeds in stored order; K' > full_K silently dropped.",
        "measures": list(measures),
        "ndcg_gain": "linear (rel / log2(i+2)); matches trec_eval ndcg_cut and EvalManager",
        "requested_k_subsets": list(requested_k),
        "full_K_observed": full_K,
        "score_aggregation_ddof": 0,
        "metric_std_ddof": 1,
    }


def split_headline_and_verbose(metrics: dict) -> tuple[dict, dict]:
    """Split :func:`derive_sc_metrics` output into ``(headline, per_query)``.

    Mirrors the ``psi_metrics.json`` / ``psi_per_query.json`` split: the
    headline file carries aggregates and protocol metadata; the verbose file
    carries the full ``per_query`` rows for each K. Drivers should write both.
    """
    by_K = metrics.get("by_K") or {}
    headline_by_K: dict[int, dict] = {}
    per_query_by_K: dict[int, dict] = {}
    for K, block in by_K.items():
        block_dict = dict(block)
        per_query = block_dict.pop("per_query", {})
        headline_by_K[int(K)] = block_dict
        per_query_by_K[int(K)] = per_query
    headline = {
        "by_K": headline_by_K,
        "headline_K": metrics.get("headline_K"),
        "protocol": metrics.get("protocol"),
    }
    return headline, per_query_by_K
