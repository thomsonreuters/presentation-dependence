"""Compute permutation-sensitivity metrics from aligned rankings and scores.

Reports five metric families.

1. Per-document score variance across K permutations is the primary scalar for
   pointwise and scoring-listwise rerankers. Per-document rank variance is the
   fallback for generative-listwise rerankers that do not emit scores.
2. Mean Kendall τ is computed over all C(K,2) pairs of output rankings.
3. Two position-sensitivity scalars are kept distinct:

   - ``zeng_psi`` implements Zeng et al., Eq. 1:
     ``1 - min(s) / max(s)``, where ``s`` contains the position-specific
     nDCG@10 values. It is scale-invariant and ranges from 0 (equal bucket
     scores) to 1 (worst-case collapse). Zeng et al. use PSI ≥ 0.03 as the
     notable-bias threshold. The per-query form requires at least two non-empty
     buckets with a positive maximum and uses their mean nDCG values.
     ``zeng_psi_corpus`` first averages nDCG by bucket across queries and is
     comparable to Zeng et al.'s leaderboard value. Their protocol places an answer
     within a passage; this project places a relevant document within a
     candidate list. The formula is shared, but the perturbation protocols are
     different and must be reported with the value.
   - ``tau_based_psi`` maps mean pairwise Kendall τ to [0,1] with
     ``(1 - mean_tau) / 2``. It does not require qrels or position buckets.

4. Stratified Δ-nDCG follows Liu et al.:
   ``nDCG(top ∪ bottom) - nDCG(middle)``. Without input-position labels, the
   code reports ``max_k nDCG@k - min_k nDCG@k`` as a stability proxy. The
   stratified value can be negative; the max-min value cannot. Aggregation keeps
   the two forms separate and sets ``mean_delta_ndcg`` only for homogeneous
   runs.
5. The per-permutation nDCG summary reports the mean, minimum, and maximum
   single-pass quality as ``mean_per_perm_ndcg``,
   ``mean_worst_perm_ndcg``, and ``mean_best_perm_ndcg``. These values are not
   self-consistency nDCG. For rerankers with scalar scores, Jensen's inequality
   places mean-then-rank self-consistency above the mean per-permutation nDCG.
   Computing it requires aligned score-level records.

nDCG uses linear gain, ``rel / log2(i+2)``, matching trec_eval's ``ndcg_cut``.
The implementation in ``trec_eval-9.0.7/m_ndcg_cut.c`` and the binary and
graded pytrec_eval parity tests pin this convention.

Performs no I/O or configuration parsing. The PSI driver generates the
permutations and passes aligned rankings, scores, and position labels to
``evaluate_psi``. The current evaluation-time recommendation is K=10 to 20. The
pure-Python Kendall τ implementation is O(n²), which is practical for reranking
depths up to 200.

Typed inputs (qid → ...):
    rankings_over_K[qid][k] = [docid_best, ..., docid_worst]   # after rerank
    scores_over_K[qid][docid] = [score_k1, ..., score_kK]      # optional
    injection_positions[qid][k] = "top" | "middle" | "bottom"  # optional;
                                                                 # applies to random or targeted inputs
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from math import nan
from typing import Literal

import numpy as np


Bucket = Literal["top", "middle", "bottom"]


# ---------------------------------------------------------------------------
# Low-level primitives (pure functions; stateless; tested in isolation).
# ---------------------------------------------------------------------------


def kendall_tau(a: list[str], b: list[str]) -> float:
    """Kendall τ_a between two rankings, restricted to the intersection.

    Strict orderings (no ties within a single ranking), so τ_a = τ_b.
    Returns `nan` when the intersection has <2 items (τ is undefined there).

    Complexity is O(n²). At the typical reranking depth of n≤200, this requires
    at most 40,000 comparisons. Set and position-map constructions are hoisted
    out of the inner loop so setup stays O(n) for large K × query counts.
    """
    b_set = set(b)
    common = [d for d in a if d in b_set]
    if len(common) < 2:
        return nan
    common_set = set(common)
    pos_a = {d: i for i, d in enumerate(a) if d in common_set}
    pos_b = {d: i for i, d in enumerate(b) if d in common_set}
    concordant = discordant = 0
    for i, j in combinations(common, 2):
        sign_a = pos_a[i] - pos_a[j]
        sign_b = pos_b[i] - pos_b[j]
        prod = sign_a * sign_b
        if prod > 0:
            concordant += 1
        elif prod < 0:
            discordant += 1
    n = len(common)
    denom = n * (n - 1) // 2
    return (concordant - discordant) / denom


def mean_pairwise_tau(rankings: list[list[str]]) -> float:
    """Mean Kendall τ across all C(K,2) pairs. `nan` if no finite pair exists."""
    if len(rankings) < 2:
        return nan
    taus = [kendall_tau(a, b) for a, b in combinations(rankings, 2)]
    finite = [t for t in taus if not np.isnan(t)]
    return float(np.mean(finite)) if finite else nan


def per_doc_rank_variance(rankings: list[list[str]]) -> dict[str, float]:
    """Population variance (ddof=0) of rank-position across K rankings, for
    each doc present in ALL K rankings (strict intersection).

    Position maps make each document lookup O(1), rather than O(n) through
    ``list.index``, with O(n) setup for each ranking.
    """
    if len(rankings) < 2:
        return {}
    pos_maps = [{d: i for i, d in enumerate(r)} for r in rankings]
    common = set(rankings[0])
    for r in rankings[1:]:
        common &= set(r)
    out: dict[str, float] = {}
    for d in common:
        positions = [pmap[d] for pmap in pos_maps]
        out[d] = float(np.var(positions, ddof=0))
    return out


def gold_rank_variance(rankings: list[list[str]], qrels_for_q: dict[str, int]) -> float:
    """Mean rank-position variance over qrels-positive documents only.

    This is a top-focused companion to full-list Kendall τ / τ-PSI. In QA,
    distractor-tail churn can move full-list τ even when the gold support
    passages are stable near the top, so report this relevance-focused scalar
    alongside the full-list metrics. Lower is more stable.
    """
    gold_docs = {str(doc_id) for doc_id, rel in qrels_for_q.items() if int(rel) > 0}
    if not gold_docs or len(rankings) < 2:
        return nan
    positions_by_doc: dict[str, list[int]] = {doc_id: [] for doc_id in gold_docs}
    for ranking in rankings:
        pos_by_doc = {str(doc_id): pos for pos, doc_id in enumerate(ranking)}
        for doc_id in gold_docs:
            if doc_id in pos_by_doc:
                positions_by_doc[doc_id].append(pos_by_doc[doc_id])
    variances = [float(np.var(pos, ddof=0)) for pos in positions_by_doc.values() if len(pos) >= 2]
    return float(np.mean(variances)) if variances else nan


def per_doc_score_variance(
    scores: dict[str, list[float]],
) -> dict[str, float]:
    """Population variance of per-permutation scores, one entry per doc.

    `scores[d] = [s_1, ..., s_K]`. Docs with <2 scores are skipped
    (variance undefined / useless).
    """
    out: dict[str, float] = {}
    for d, vals in scores.items():
        if vals is None or len(vals) < 2:
            continue
        out[d] = float(np.var(vals, ddof=0))
    return out


def ndcg_at_k(
    ranking: list[str],
    qrels_for_q: dict[str, int],
    k: int = 10,
) -> float:
    """Compute graded nDCG@k with linear relevance gains.

    Gains are ``rel``, without the ``2^rel - 1`` transform, and discounts are
    ``log2(i+2)``. This matches pytrec_eval ``ndcg_cut`` for non-negative
    integer qrels. The local implementation avoids file I/O for one ranking and
    is cross-checked against pytrec_eval.
    """
    if not ranking:
        return 0.0
    top = ranking[:k]
    dcg = 0.0
    for i, d in enumerate(top):
        rel = qrels_for_q.get(d, 0)
        if rel > 0:
            dcg += rel / np.log2(i + 2)  # positions 0 → log2(2)=1
    ideal_rels = sorted(qrels_for_q.values(), reverse=True)[:k]
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_rels) if rel > 0)
    return float(dcg / idcg) if idcg > 0 else 0.0


# ---------------------------------------------------------------------------
# Headline harness.
# ---------------------------------------------------------------------------


def _psi_from_tau(mean_tau: float) -> float:
    """τ-based PSI as the [0,1] remap of mean pairwise Kendall τ.

    `psi = (1 - mean_tau) / 2`. Clamped to [0, 1] to absorb float noise at
    the endpoints (identical rankings can produce 1 + ε due to intersection
    edge cases). Returns `nan` when τ itself is undefined. This quantity is
    distinct from Zeng et al.'s PSI.
    """
    if np.isnan(mean_tau):
        return nan
    return float(np.clip((1.0 - mean_tau) / 2.0, 0.0, 1.0))


def _delta_ndcg_stratified(
    ndcg_K: list[float],
    buckets: list[Bucket],
) -> float | None:
    """Δ-nDCG = mean nDCG(top ∪ bottom) − mean nDCG(middle).

    Liu-style stratified form. Returns None unless both the extremes and
    the middle have at least one observation.
    """
    if len(ndcg_K) != len(buckets):
        raise ValueError(f"ndcg_K ({len(ndcg_K)}) and buckets ({len(buckets)}) length mismatch")
    extremes = [n for n, b in zip(ndcg_K, buckets) if b in ("top", "bottom")]
    middles = [n for n, b in zip(ndcg_K, buckets) if b == "middle"]
    if not extremes or not middles:
        return None
    return float(np.mean(extremes) - np.mean(middles))


def _bucket_means(
    ndcg_K: list[float],
    buckets: list[Bucket],
) -> dict[str, float]:
    """Group `ndcg_K` by bucket label, return {bucket: mean_nDCG}.

    Empty buckets are absent from the output (caller decides how to handle).
    """
    if len(ndcg_K) != len(buckets):
        raise ValueError(f"ndcg_K ({len(ndcg_K)}) and buckets ({len(buckets)}) length mismatch")
    groups: dict[str, list[float]] = {}
    for n, b in zip(ndcg_K, buckets):
        groups.setdefault(b, []).append(n)
    return {b: float(np.mean(vs)) for b, vs in groups.items() if vs}


def zeng_psi(bucket_means: dict[str, float]) -> float | None:
    """Zeng et al., Eq. 1 (§3.3): `PSI = 1 − min(s) / max(s)`.

    `bucket_means` maps each position bucket to the mean quality score for
    that bucket (e.g. nDCG@10). Zeng et al. define `s` as one
    position-specific evaluation score per bucket. PSI is the worst-case
    relative degradation across those buckets.

    Returns `None` (statistic undefined) when:
        - fewer than 2 non-empty buckets are present,
        - max(s) ≤ 0 (the published definition requires `max(s) > 0`; otherwise
          the formula either divides by 0 or yields a meaningless 0/0).

    Clamped to [0, 1] to absorb float noise; identical buckets produce
    exactly 0.
    """
    if len(bucket_means) < 2:
        return None
    vals = list(bucket_means.values())
    mx = max(vals)
    if mx <= 0:
        return None
    mn = min(vals)
    return float(np.clip(1.0 - mn / mx, 0.0, 1.0))


def evaluate_psi(
    *,
    rankings_over_K: dict[str, list[list[str]]],
    scores_over_K: dict[str, dict[str, list[float]]] | None = None,
    qrels: dict[str, dict[str, int]] | None = None,
    injection_positions: dict[str, list[Bucket]] | None = None,
    k_cutoff: int = 10,
) -> dict:
    """Compute the four robustness metrics over K rerank permutations.

    Args:
        rankings_over_K: qid → list of K output rankings, each best→worst docids.
        scores_over_K: qid → {docid → [K scalar scores]} built from each run's
            ``scores_init_order`` (aligned by docid across permutations). This
            input is required for rerankers that expose per-document scalars,
            including classical pointwise CE, scoring-head models such as Jina,
            and batched-pointwise models. In those cases,
            ``mean_score_variance`` is a
            primary robustness readout alongside τ-PSI. When absent (no scalars),
            per-query ``score_variance`` is omitted and ``mean_rank_variance``
            supplements Kendall τ and τ-based PSI. This is the expected path for
            generative-listwise rankings such as RankZephyr, or any result where
            ``scores_init_order`` is consistently ``None``.
        qrels: TREC-style parsed qrels `{qid: {docid: rel}}`. Required to
            compute Δ-nDCG; if absent, that field is None.
        injection_positions: qid → list of K bucket labels describing where
            the relevant doc was placed in the reranker's *input* on each
            permutation. Expected labels: "top" | "middle" | "bottom". When
            absent AND qrels present, compute `max − min` Δ-nDCG
            (per-query stability) as a fallback. The caller (driver) must
            choose buckets consistent with the reported
            perturbation protocol; labels may describe observational random
            draws or a separate targeted placement design.
        k_cutoff: nDCG cutoff (default 10, matching the project's headline).

    Returns:
        {
          "per_query":   {qid: {...}, ...},
          "aggregate":   {headline mean/std over queries, n_queries, K},
          "protocol":    {formula IDs + form flags; for reproducibility audit}
        }

    Notes:
        - All per-query metrics drop to `nan` when the input is degenerate
          (K<2, empty intersection, etc.). Aggregates skip NaN rows.
        - Rank variance uses 0-indexed ranks (position 0 = best). Units are
          squared-ranks; compare ratios across models, not absolute numbers.
        - Score variance inherits the reranker's output scale; compare it only
          within a model. For pointwise models it should sit near the
          numerical noise floor (~deterministic scores across permutations).
          For batched-pointwise and scoring-head models it reflects permutation
          sensitivity of logits. Report it with
          ``tau_based_psi`` as co-primary robustness signals for PA-GRPO.
        - Rank variance is the ranking-level fallback whenever scalars are
          unavailable.
    """
    if not rankings_over_K:
        return _empty_result(k_cutoff, reason="no queries provided")

    per_query: dict[str, dict] = {}
    for qid, rankings in rankings_over_K.items():
        per_query[qid] = _evaluate_one_query(
            qid=qid,
            rankings=rankings,
            scores=(scores_over_K or {}).get(qid),
            qrels_for_q=(qrels or {}).get(qid),
            buckets=(injection_positions or {}).get(qid),
            k_cutoff=k_cutoff,
        )

    aggregate = _aggregate(per_query)
    aggregate["K_permutations"] = _modal_K(per_query)
    aggregate["k_cutoff_for_ndcg"] = k_cutoff

    return {
        "per_query": per_query,
        "aggregate": aggregate,
        "protocol": {
            "formula_set": "slm-ranking-robustness-v1",
            "zeng_psi_formula": "1 - min(s) / max(s); Zeng et al. Eq. 1, §3.3",
            "zeng_psi_input": (
                "s = {mean nDCG@k per position bucket}; per-query uses "
                "this query's bucket means, corpus uses global bucket means"
            ),
            "zeng_psi_caveat": (
                "Zeng et al. use buckets at positions INSIDE a long passage "
                "(SQuAD-PosQ/FineWeb-PosQ); our buckets are positions inside "
                "the candidate list. Formula transfers but protocols differ."
            ),
            "tau_based_psi_formula": "(1 - mean_tau) / 2; Tang-style τ->[0,1] remap",
            "tau_based_psi_note": (
                "NOT Zeng PSI. Reported because it captures listwise "
                "consistency (τ spread) without requiring qrels or buckets."
            ),
            "delta_ndcg_source": "Liu et al. (lost-in-the-middle)",
            "delta_ndcg_aggregation": (
                "stratified and max_min forms aggregated separately; "
                "combined mean_delta_ndcg populated only when form is homogeneous"
            ),
            "per_perm_ndcg_summary": (
                "mean_per_perm_ndcg / mean_worst_perm_ndcg / mean_best_perm_ndcg "
                "summarise the K nDCG@k values per query (mean / min / max), "
                "averaged across queries. Bracket self-consistency nDCG from "
                "below by Jensen — NOT a substitute for SC, which requires "
                "averaging score vectors per docid before ranking."
            ),
            "kendall_tau_variant": "tau_a (strict orders, no ties)",
            "ndcg_gain": "linear (rel / log2(i+2)); matches trec_eval ndcg_cut",
            "rank_variance_ddof": 0,
            "gold_rank_variance_note": (
                "Mean population rank-position variance over qrels-positive documents only; "
                "top-focused stability readout for QA/multi-gold settings."
            ),
            "score_variance_ddof": 0,
        },
    }


def _evaluate_one_query(
    *,
    qid: str,
    rankings: list[list[str]],
    scores: dict[str, list[float]] | None,
    qrels_for_q: dict[str, int] | None,
    buckets: list[Bucket] | None,
    k_cutoff: int,
) -> dict:
    K = len(rankings)
    out: dict = {
        "qid": qid,
        "K": K,
        "score_variance": None,
        "rank_variance": nan,
        "kendall_tau": nan,
        "tau_based_psi": nan,
        "zeng_psi": None,
        "bucket_means_ndcg": None,
        "gold_rank_variance": nan,
        "ndcg_K": None,
        "delta_ndcg": None,
        "delta_ndcg_form": "na",
        "per_perm_ndcg_mean": None,
        "per_perm_ndcg_min": None,
        "per_perm_ndcg_max": None,
    }

    if K < 2:
        return out  # nothing useful to compute

    # --- rank / score variance -------------------------------------------
    rank_vars = per_doc_rank_variance(rankings)
    out["rank_variance"] = float(np.mean(list(rank_vars.values()))) if rank_vars else nan

    if scores:
        score_vars = per_doc_score_variance(scores)
        out["score_variance"] = float(np.mean(list(score_vars.values()))) if score_vars else None

    # --- Kendall τ / tau-based PSI (NOT Zeng PSI) ------------------------
    mean_tau = mean_pairwise_tau(rankings)
    out["kendall_tau"] = mean_tau
    out["tau_based_psi"] = _psi_from_tau(mean_tau)

    # --- Δ-nDCG + Zeng PSI ------------------------------------------------
    if qrels_for_q is not None:
        out["gold_rank_variance"] = gold_rank_variance(rankings, qrels_for_q)
        ndcg_K = [ndcg_at_k(r, qrels_for_q, k=k_cutoff) for r in rankings]
        out["ndcg_K"] = ndcg_K
        if ndcg_K:
            out["per_perm_ndcg_mean"] = float(np.mean(ndcg_K))
            out["per_perm_ndcg_min"] = float(np.min(ndcg_K))
            out["per_perm_ndcg_max"] = float(np.max(ndcg_K))
        if buckets:
            out["delta_ndcg"] = _delta_ndcg_stratified(ndcg_K, buckets)
            out["delta_ndcg_form"] = "stratified"
            # Zeng PSI needs per-bucket means; only defined for stratified runs.
            bucket_means = _bucket_means(ndcg_K, buckets)
            out["bucket_means_ndcg"] = bucket_means
            out["zeng_psi"] = zeng_psi(bucket_means)
        else:
            out["delta_ndcg"] = float(max(ndcg_K) - min(ndcg_K))
            out["delta_ndcg_form"] = "max_min"
            # Zeng PSI is undefined without bucket means.

    return out


def _aggregate(per_query: dict[str, dict]) -> dict:  # noqa: C901
    """Per-metric mean/std/n across queries.

    Δ-nDCG is split by form. `stratified` can be negative; `max_min` is always
    non-negative. Pooling them would make a mixed-run mean uninterpretable.
    We always emit per-form keys, and only populate the top-level
    `mean_delta_ndcg` when the run is form-homogeneous.

    Zeng PSI is computed two ways:
    - `mean_zeng_psi` / `std_zeng_psi` average per-query PSI and describe its
      spread. Zeng et al. do not report this form.
    - `zeng_psi_corpus` matches Zeng et al.'s Table 1: pool all
      (nDCG, bucket) observations across queries, compute mean nDCG per
      bucket globally, then Eq. 1. This is the leaderboard-comparable number.
    """
    if not per_query:
        return {"n_queries": 0}

    simple_keys = (
        "score_variance",
        "rank_variance",
        "gold_rank_variance",
        "kendall_tau",
        "tau_based_psi",
        "zeng_psi",
    )
    agg: dict = {"n_queries": len(per_query)}
    for k in simple_keys:
        vals = _finite_values(per_query, k)
        if vals:
            agg[f"mean_{k}"] = float(np.mean(vals))
            agg[f"std_{k}"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            agg[f"n_{k}"] = len(vals)
        else:
            agg[f"mean_{k}"] = None
            agg[f"std_{k}"] = None
            agg[f"n_{k}"] = 0

    # Per-permutation nDCG distribution summary across queries.
    # Naming: per-query stores (mean/min/max) of `ndcg_K`; aggregate exposes
    #     mean_per_perm_ndcg   = E_q[ mean_k nDCG@k(q, perm_k) ]   (expected single-shot)
    #     mean_worst_perm_ndcg = E_q[ min_k  nDCG@k(q, perm_k) ]   (worst permutation)
    #     mean_best_perm_ndcg  = E_q[ max_k  nDCG@k(q, perm_k) ]   (best permutation)
    # These bracket SC nDCG@k from below (Jensen): a self-consistency rerank
    # that means score vectors per docid then sorts is ≥ mean_per_perm_ndcg
    # in expectation. SC is computed externally
    # (post-hoc on detailed_results.json), not here, because evaluate_psi()
    # operates on rankings and ndcg values, not raw score vectors.
    per_perm_pairs = (
        ("per_perm_ndcg_mean", "mean_per_perm_ndcg"),
        ("per_perm_ndcg_min", "mean_worst_perm_ndcg"),
        ("per_perm_ndcg_max", "mean_best_perm_ndcg"),
    )
    for src_key, dst_key in per_perm_pairs:
        vals = _finite_values(per_query, src_key)
        if vals:
            agg[dst_key] = float(np.mean(vals))
            agg[f"std_{dst_key[len('mean_') :]}"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            agg[f"n_{dst_key[len('mean_') :]}"] = len(vals)
        else:
            agg[dst_key] = None
            agg[f"std_{dst_key[len('mean_') :]}"] = None
            agg[f"n_{dst_key[len('mean_') :]}"] = 0

    # --- Δ-nDCG: always split by form; never pool ---------------------------
    forms_counter: defaultdict[str, int] = defaultdict(int)
    per_form_vals: defaultdict[str, list[float]] = defaultdict(list)
    for row in per_query.values():
        form = row["delta_ndcg_form"]
        forms_counter[form] += 1
        v = row["delta_ndcg"]
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            per_form_vals[form].append(float(v))

    agg["delta_ndcg_forms"] = dict(forms_counter)
    for form in ("stratified", "max_min"):
        vals = per_form_vals.get(form, [])
        if vals:
            agg[f"mean_delta_ndcg_{form}"] = float(np.mean(vals))
            agg[f"std_delta_ndcg_{form}"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            agg[f"n_delta_ndcg_{form}"] = len(vals)
        else:
            agg[f"mean_delta_ndcg_{form}"] = None
            agg[f"std_delta_ndcg_{form}"] = None
            agg[f"n_delta_ndcg_{form}"] = 0

    # Populate the combined field only for homogeneous runs. Consumers that
    # ignore the form flag must not receive an average over incompatible forms.
    observed_forms = {f for f, c in forms_counter.items() if c > 0 and f != "na"}
    if len(observed_forms) == 1:
        form = next(iter(observed_forms))
        agg["mean_delta_ndcg"] = agg[f"mean_delta_ndcg_{form}"]
        agg["std_delta_ndcg"] = agg[f"std_delta_ndcg_{form}"]
        agg["n_delta_ndcg"] = agg[f"n_delta_ndcg_{form}"]
        agg["delta_ndcg_form_homogeneous"] = form
    else:
        agg["mean_delta_ndcg"] = None
        agg["std_delta_ndcg"] = None
        agg["n_delta_ndcg"] = sum(agg[f"n_delta_ndcg_{f}"] for f in ("stratified", "max_min"))
        agg["delta_ndcg_form_homogeneous"] = None

    # --- Zeng corpus-level PSI (leaderboard-comparable) -------------------
    # Zeng et al.'s Table 1 computes PSI once per (model, dataset) by first
    # averaging nDCG per bucket across all queries, then applying Eq. 1 to
    # those bucket means. We emit that as `zeng_psi_corpus` whenever any
    # query in the run supplied stratified buckets.
    corpus_bucket_vals: defaultdict[str, list[float]] = defaultdict(list)
    for row in per_query.values():
        if row["delta_ndcg_form"] != "stratified":
            continue
        bmeans = row.get("bucket_means_ndcg")
        if not bmeans:
            continue
        for bucket, val in bmeans.items():
            corpus_bucket_vals[bucket].append(float(val))
    corpus_bucket_means = {b: float(np.mean(vs)) for b, vs in corpus_bucket_vals.items() if vs}
    agg["zeng_psi_corpus_bucket_means"] = corpus_bucket_means or None
    agg["zeng_psi_corpus"] = zeng_psi(corpus_bucket_means) if corpus_bucket_means else None
    agg["n_zeng_psi_corpus_queries"] = sum(
        1 for r in per_query.values() if r["delta_ndcg_form"] == "stratified" and r.get("bucket_means_ndcg")
    )
    return agg


def _finite_values(per_query: dict[str, dict], key: str) -> list[float]:
    """Helper: extract finite (not None, not NaN) values of `key` across rows."""
    out: list[float] = []
    for row in per_query.values():
        v = row[key]
        if v is None:
            continue
        if isinstance(v, float) and np.isnan(v):
            continue
        out.append(float(v))
    return out


def _modal_K(per_query: dict[str, dict]) -> int:
    if not per_query:
        return 0
    Ks = [row["K"] for row in per_query.values()]
    vals, counts = np.unique(Ks, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _empty_result(k_cutoff: int, reason: str) -> dict:
    return {
        "per_query": {},
        "aggregate": {"n_queries": 0, "K_permutations": 0, "reason": reason},
        "protocol": {
            "formula_set": "slm-ranking-robustness-v1",
            "k_cutoff_for_ndcg": k_cutoff,
        },
    }
