"""Measure the stability of thresholded document sets.

The metrics convert each reranker's absolute scores into a retained set,
``S = {doc : score >= tau}``, then compare that set across input
permutations. Yields a reader-free downstream decision for reranking,
parallel to evaluating answer stability after the QA reader.

Performs no I/O or configuration parsing. Consumes aligned per-document scores,
``scores_over_perms[docid] = [s_perm0, ..., s_perm{M-1}]``, and the reachable
relevant-document set. Canonical reproduction reductions provide these inputs
and write the retained artifacts.

The metric families are:

1. Set stability: mean pairwise Jaccard and the fraction of ever-retained
   documents whose inclusion changes.
2. Decision-quality stability: the mean and standard deviation of retained-set
   precision, recall, and F1 across permutations.
3. Operating-point quality: precision, recall, F1, and retention in the
   canonical presentation.
4. Matched-retention control: metrics after tuning ``tau`` to a shared mean
   retention target, usually the F1-tuned retention of single-order distillation. This separates a
   change in set size from a change in set stability.
5. Calibration variance: per-document score variance across permutations, used
   as the T4 calibration axis.
6. Rank stability: Kendall τ and τ-PSI of the rankings induced by the same
   scores, used as the second T4 axis.

Conventions:

- ``relevant`` contains qrels-positive documents, at the caller's relevance
  cutoff (label >= 1 by default), restricted to the scored candidate pool. The
  recall denominator excludes relevant documents that the first stage did not
  retrieve.
- Two empty retained sets are treated as identical (Jaccard = 1.0).
- Variances use population ddof=0 (matches ``psi.per_doc_score_variance``);
  metric spreads across permutations use sample ddof=1.
"""

from __future__ import annotations

from math import nan
from typing import Mapping, Sequence

import numpy as np

from presentation_dependence.eval.psi import mean_pairwise_tau


# ---------------------------------------------------------------------------
# Per-permutation primitives (single retained set).
# ---------------------------------------------------------------------------


def retained_set(perm_scores: Mapping[str, float], tau: float) -> set[str]:
    """Docs retained at threshold ``tau`` for a single permutation.

    ``S = {doc : score >= tau}``. Ties at exactly ``tau`` are retained.
    """
    return {str(d) for d, s in perm_scores.items() if float(s) >= tau}


def prf(retained: set[str], relevant: set[str]) -> dict[str, float]:
    """Precision / recall / F1 of a retained set against the relevant set.

    Empty retained set -> precision 0.0 (nothing returned, nothing correct).
    Empty relevant set  -> recall 0.0 (no relevant docs reachable). F1 is the
    harmonic mean, 0.0 when precision+recall == 0.
    """
    n_ret = len(retained)
    n_rel = len(relevant)
    tp = len(retained & relevant)
    precision = tp / n_ret if n_ret else 0.0
    recall = tp / n_rel if n_rel else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def retention_rate(retained: set[str], n_candidates: int) -> float:
    """Fraction of the candidate pool retained: ``|S| / |candidates|``.

    Distinct from precision/recall: two arms can share F1 while retaining
    different numbers of documents. Returns ``nan`` when the pool is empty.
    """
    if n_candidates <= 0:
        return nan
    return float(len(retained) / n_candidates)


# ---------------------------------------------------------------------------
# Across-permutation set stability (headline metrics).
# ---------------------------------------------------------------------------


def sets_over_perms(
    scores_over_perms: Mapping[str, Sequence[float]],
    tau: float,
) -> list[set[str]]:
    """Build the retained set for each permutation, in permutation order.

    ``scores_over_perms[docid] = [s_perm0, ..., s_perm{M-1}]`` (docid-aligned;
    entry ``i`` is the same permutation across docids). Returns a list of M
    retained sets.
    """
    if not scores_over_perms:
        return []
    m = max(len(v) for v in scores_over_perms.values())
    out: list[set[str]] = [set() for _ in range(m)]
    for docid, vals in scores_over_perms.items():
        d = str(docid)
        for i, s in enumerate(vals):
            if float(s) >= tau:
                out[i].add(d)
    return out


def _membership_matrix(sets: Sequence[set[str]]) -> np.ndarray:
    """(M perms x D docs) boolean membership matrix over the union of docs."""
    docids = sorted(set().union(*sets)) if sets else []
    idx = {d: j for j, d in enumerate(docids)}
    mat = np.zeros((len(sets), len(docids)), dtype=bool)
    for i, s in enumerate(sets):
        for d in s:
            mat[i, idx[d]] = True
    return mat


def mean_pairwise_jaccard(sets: Sequence[set[str]]) -> float:
    """Mean Jaccard similarity of the retained set across all C(M,2) perm pairs.

    A value of 1.0 means that every permutation retains the same documents. Two
    empty sets count as identical. The membership-matrix implementation remains
    tractable at M=500.
    """
    if len(sets) < 2:
        return nan
    mat = _membership_matrix(sets).astype(np.int64)  # (M, D)
    inter = mat @ mat.T  # (M, M) pairwise intersection counts
    sizes = np.diag(inter)
    union = sizes[:, None] + sizes[None, :] - inter
    with np.errstate(invalid="ignore", divide="ignore"):
        jac = np.where(union > 0, inter / union, 1.0)  # both-empty -> identical
    iu = np.triu_indices(len(sets), k=1)
    return float(np.mean(jac[iu]))


def set_flip_rate(sets: Sequence[set[str]]) -> float:
    """Fraction of *ever-retained* docs whose in/out decision changes.

    For each doc, ``f_d`` = fraction of permutations in which it is retained. A
    doc is *ever-retained* if ``f_d > 0`` and *unstable* if ``0 < f_d < 1``.
    Flip-rate = (#unstable) / (#ever-retained). 0.0 = every retained doc is
    retained under all permutations; higher = more set churn. ``nan`` when no
    doc is ever retained.
    """
    if len(sets) < 2:
        return nan
    mat = _membership_matrix(sets)  # (M, D) bool
    if mat.shape[1] == 0:
        return nan
    f = mat.mean(axis=0)  # per-doc in-fraction
    ever = f > 0
    if not ever.any():
        return nan
    unstable = ever & (f < 1.0)
    return float(unstable.sum() / ever.sum())


def prf_over_perms(sets: Sequence[set[str]], relevant: set[str]) -> dict[str, dict[str, float]]:
    """Per-permutation P/R/F1 -> {mean, std (ddof=1)} across permutations.

    The standard deviation measures decision-quality variation across input
    permutations.
    """
    if not sets:
        return {k: {"mean": nan, "std": nan} for k in ("precision", "recall", "f1")}
    rows = [prf(s, relevant) for s in sets]
    out: dict[str, dict[str, float]] = {}
    for key in ("precision", "recall", "f1"):
        vals = np.array([r[key] for r in rows], dtype=float)
        out[key] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        }
    return out


def retention_over_perms(sets: Sequence[set[str]], n_candidates: int) -> dict[str, float]:
    """Per-permutation retention rate -> {mean, std (ddof=1)} across permutations."""
    if not sets or n_candidates <= 0:
        return {"mean": nan, "std": nan}
    vals = np.array([retention_rate(s, n_candidates) for s in sets], dtype=float)
    return {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
    }


def mean_score_variance(scores_over_perms: Mapping[str, Sequence[float]]) -> float:
    """Mean per-doc population variance (ddof=0) of scores across permutations.

    Thresholding uses absolute scores, so documents may cross the cutoff even
    when their relative ranks remain stable. Documents with fewer than two
    scores are excluded. Returns ``nan`` when no variance is measurable.
    """
    variances: list[float] = []
    for vals in scores_over_perms.values():
        if vals is not None and len(vals) >= 2:
            variances.append(float(np.var(np.asarray(vals, dtype=float), ddof=0)))
    return float(np.mean(variances)) if variances else nan


# Limit the Kendall-τ calculation to the first ``max_perms`` presentations in
# deterministic seed order. At M=500, all pairings require about 125,000
# comparisons per query. The deterministic prefix provides a paired, stable
# subsample. Set-based metrics continue to use every presentation.
RANK_STABILITY_MAX_PERMS = 64


def rank_stability(
    scores_over_perms: Mapping[str, Sequence[float]],
    max_perms: int = RANK_STABILITY_MAX_PERMS,
) -> dict[str, float]:
    """Kendall-tau / tau-PSI of the *ordering* induced by the per-perm scores.

    Builds the descending-score ranking for each permutation and reuses
    ``psi.mean_pairwise_tau``. ``tau_psi = (1 - mean_tau) / 2`` is the rank-space
    stability the T4 probe correlates set-stability against. Ties broken by
    docid for determinism. Uses at most ``max_perms`` permutations (seed-order)
    for the pairwise-tau mean to stay tractable at M=500.
    """
    if not scores_over_perms:
        return {"mean_kendall_tau": nan, "tau_psi": nan}
    docids = list(scores_over_perms.keys())
    m = max(len(v) for v in scores_over_perms.values())
    m = min(m, max_perms) if max_perms and max_perms > 0 else m
    rankings: list[list[str]] = []
    for i in range(m):
        scored = [(str(d), float(scores_over_perms[d][i])) for d in docids if i < len(scores_over_perms[d])]
        scored.sort(key=lambda kv: (-kv[1], kv[0]))
        rankings.append([d for d, _ in scored])
    mean_tau = mean_pairwise_tau(rankings)
    tau_psi = nan if np.isnan(mean_tau) else float(np.clip((1.0 - mean_tau) / 2.0, 0.0, 1.0))
    return {"mean_kendall_tau": mean_tau, "tau_psi": tau_psi}


# ---------------------------------------------------------------------------
# Threshold tuning + sweep (cross-query, canonical order).
# ---------------------------------------------------------------------------


def build_grid(scores: Sequence[float], n: int = 41) -> list[float]:
    """Threshold grid spanning the score range (inclusive of the extremes).

    Uses evenly spaced percentiles of the pooled canonical scores so the grid
    adapts to each scorer's absolute-score scale. A shared numerical cutoff is
    not comparable across different score scales. Degenerate scores produce a
    one-point grid.
    """
    arr = np.asarray([float(s) for s in scores], dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return [0.0]
    lo, hi = float(arr.min()), float(arr.max())
    if hi <= lo:
        return [lo]
    return list(np.linspace(lo, hi, n))


def tune_threshold_f1(
    canonical_scores_per_q: Mapping[str, Mapping[str, float]],
    relevant_by_q: Mapping[str, set[str]],
    grid: Sequence[float],
) -> dict:
    """Pick ``tau*`` maximising mean canonical-order set-F1 over the dev queries.

    The frozen per-scorer operating point: tune on the canonical
    (unpermuted) order, then freeze. Returns ``tau_star``, the per-tau mean-F1
    curve, and the mean P/R/F1 at ``tau*``. Ties on F1 break toward the *higher*
    threshold (smaller, more precise retained set).
    """
    qids = [q for q in canonical_scores_per_q if canonical_scores_per_q[q]]
    by_tau: list[dict] = []
    best = {"tau": nan, "f1": -1.0}
    for tau in grid:
        ps, rs, fs, rets = [], [], [], []
        for q in qids:
            scores = canonical_scores_per_q[q]
            ret = retained_set(scores, tau)
            m = prf(ret, relevant_by_q.get(q, set()))
            ps.append(m["precision"])
            rs.append(m["recall"])
            fs.append(m["f1"])
            rets.append(retention_rate(ret, len(scores)))
        mean_f1 = float(np.mean(fs)) if fs else 0.0
        mean_ret = float(np.nanmean(rets)) if rets else nan
        row = {
            "tau": float(tau),
            "mean_precision": float(np.mean(ps)) if ps else 0.0,
            "mean_recall": float(np.mean(rs)) if rs else 0.0,
            "mean_f1": mean_f1,
            "mean_retention": mean_ret,
        }
        by_tau.append(row)
        # >= so that on ties we keep advancing to the higher tau.
        if mean_f1 >= best["f1"]:
            best = {"tau": float(tau), "f1": mean_f1}
    return {
        "tau_star": best["tau"],
        "tau_star_mean_f1": best["f1"],
        "n_dev_queries": len(qids),
        "by_tau": by_tau,
    }


def mean_canonical_retention(
    canonical_scores_per_q: Mapping[str, Mapping[str, float]],
    tau: float,
) -> float:
    """Mean ``|S| / |candidates|`` over queries at ``tau`` on the canonical order."""
    rates: list[float] = []
    for scores in canonical_scores_per_q.values():
        if not scores:
            continue
        rates.append(retention_rate(retained_set(scores, tau), len(scores)))
    return float(np.nanmean(rates)) if rates else nan


def tune_threshold_retention(
    canonical_scores_per_q: Mapping[str, Mapping[str, float]],
    target_retention: float,
    grid: Sequence[float],
) -> dict:
    """Pick ``tau`` whose mean canonical retention is closest to ``target_retention``.

    The matched-retention control: equalise retained-set size across arms so a
    set-flip gain cannot be an artifact of a smaller (or larger) retained set.
    On equal absolute error, prefer the *higher* threshold (smaller set).
    """
    if not np.isfinite(target_retention):
        return {
            "tau_star": nan,
            "tau_star_mean_retention": nan,
            "target_retention": float(target_retention) if target_retention is not None else nan,
            "abs_error": nan,
            "n_dev_queries": 0,
            "by_tau": [],
        }
    qids = [q for q in canonical_scores_per_q if canonical_scores_per_q[q]]
    by_tau: list[dict] = []
    best = {"tau": nan, "retention": nan, "err": float("inf")}
    for tau in grid:
        rets = []
        for q in qids:
            scores = canonical_scores_per_q[q]
            rets.append(retention_rate(retained_set(scores, tau), len(scores)))
        mean_ret = float(np.nanmean(rets)) if rets else nan
        err = abs(mean_ret - target_retention) if np.isfinite(mean_ret) else float("inf")
        by_tau.append({"tau": float(tau), "mean_retention": mean_ret, "abs_error": err})
        # Prefer closer; on ties advance to higher tau.
        if err < best["err"] - 1e-15 or (
            abs(err - best["err"]) <= 1e-15 and (not np.isfinite(best["tau"]) or tau >= best["tau"])
        ):
            best = {"tau": float(tau), "retention": mean_ret, "err": err}
    return {
        "tau_star": best["tau"],
        "tau_star_mean_retention": best["retention"],
        "target_retention": float(target_retention),
        "abs_error": best["err"] if best["err"] != float("inf") else nan,
        "n_dev_queries": len(qids),
        "by_tau": by_tau,
    }


def evaluate_at_threshold(
    scores_over_perms_per_q: Mapping[str, Mapping[str, Sequence[float]]],
    relevant_by_q: Mapping[str, set[str]],
    tau: float,
    *,
    include_rank_calibration: bool = True,
) -> dict:
    """Aggregate the full threshold-set metric suite at a frozen ``tau``.

    Computes, per query, the headline set-stability (Jaccard, flip-rate),
    decision-quality stability (per-perm P/R/F1 mean + spread), the canonical
    (perm 0) operating-point P/R/F1 and retention rate, calibration variance,
    and rank stability; then averages across queries. Per-query rows are
    returned for the T4 decoupling correlation.

    ``include_rank_calibration=False`` skips the threshold-independent rank
    stability and calibration variance (they don't change with ``tau``) so the
    :func:`sweep` can scan the grid cheaply.
    """
    per_query: dict[str, dict] = {}
    for qid, scores_over_perms in scores_over_perms_per_q.items():
        if not scores_over_perms:
            continue
        relevant = relevant_by_q.get(qid, set())
        n_cand = len(scores_over_perms)
        sets = sets_over_perms(scores_over_perms, tau)
        prf_perm = prf_over_perms(sets, relevant)
        ret_perm = retention_over_perms(sets, n_cand)
        canonical = {d: vals[0] for d, vals in scores_over_perms.items() if len(vals) >= 1}
        canon_ret = retained_set(canonical, tau)
        canon_prf = prf(canon_ret, relevant)
        row = {
            "qid": qid,
            "n_candidates": n_cand,
            "n_relevant_reachable": len(relevant),
            "n_retained_canonical": len(canon_ret),
            "mean_pairwise_jaccard": mean_pairwise_jaccard(sets),
            "set_flip_rate": set_flip_rate(sets),
            "precision_mean": prf_perm["precision"]["mean"],
            "precision_std": prf_perm["precision"]["std"],
            "recall_mean": prf_perm["recall"]["mean"],
            "recall_std": prf_perm["recall"]["std"],
            "f1_mean": prf_perm["f1"]["mean"],
            "f1_std": prf_perm["f1"]["std"],
            "retention_mean": ret_perm["mean"],
            "retention_std": ret_perm["std"],
            "canonical_precision": canon_prf["precision"],
            "canonical_recall": canon_prf["recall"],
            "canonical_f1": canon_prf["f1"],
            "canonical_retention": retention_rate(canon_ret, n_cand),
        }
        if include_rank_calibration:
            rank = rank_stability(scores_over_perms)
            row["mean_score_variance"] = mean_score_variance(scores_over_perms)
            row["kendall_tau"] = rank["mean_kendall_tau"]
            row["tau_psi"] = rank["tau_psi"]
        per_query[qid] = row
    aggregate = _aggregate_per_query(per_query, tau)
    return {"tau": float(tau), "per_query": per_query, "aggregate": aggregate}


_AGG_KEYS = (
    "mean_pairwise_jaccard",
    "set_flip_rate",
    "precision_mean",
    "precision_std",
    "recall_mean",
    "recall_std",
    "f1_mean",
    "f1_std",
    "retention_mean",
    "retention_std",
    "canonical_precision",
    "canonical_recall",
    "canonical_f1",
    "canonical_retention",
    "n_retained_canonical",
    "n_candidates",
    "mean_score_variance",
    "kendall_tau",
    "tau_psi",
)


def _finite(per_query: Mapping[str, dict], key: str) -> list[float]:
    out: list[float] = []
    for row in per_query.values():
        v = row.get(key)
        if v is None:
            continue
        if isinstance(v, float) and np.isnan(v):
            continue
        out.append(float(v))
    return out


def _aggregate_per_query(per_query: Mapping[str, dict], tau: float) -> dict:
    agg: dict = {"tau": float(tau), "n_queries": len(per_query)}
    for key in _AGG_KEYS:
        vals = _finite(per_query, key)
        agg[f"mean_{key}"] = float(np.mean(vals)) if vals else None
        agg[f"n_{key}"] = len(vals)
    return agg


def sweep(
    scores_over_perms_per_q: Mapping[str, Mapping[str, Sequence[float]]],
    relevant_by_q: Mapping[str, set[str]],
    grid: Sequence[float],
) -> list[dict]:
    """Aggregate metrics across the full threshold grid.

    Returns one aggregate row per operating point so the conclusion does not
    depend on a single selected threshold.
    """
    return [
        evaluate_at_threshold(scores_over_perms_per_q, relevant_by_q, tau, include_rank_calibration=False)["aggregate"]
        for tau in grid
    ]


def decoupling_correlations(per_query: Mapping[str, dict]) -> dict:
    """Correlate set churn with rank instability and score variance.

    Correlates per-query set churn (``1 - mean_pairwise_jaccard``) against the
    rank-instability (``tau_psi``) and the calibration variance
    (``mean_score_variance``). The T4 analysis uses these correlations to
    compare rank and calibration explanations of set stability. Association
    with τ-PSI alone supports a rank-stability explanation; additional
    association with score variance supports a calibration-stability effect.
    """
    rows = list(per_query.values())
    churn, taupsi, var = [], [], []
    for r in rows:
        j = r.get("mean_pairwise_jaccard")
        tp = r.get("tau_psi")
        sv = r.get("mean_score_variance")
        if any(x is None or (isinstance(x, float) and np.isnan(x)) for x in (j, tp, sv)):
            continue
        churn.append(1.0 - float(j))
        taupsi.append(float(tp))
        var.append(float(sv))
    n = len(churn)
    return {
        "n_queries": n,
        "spearman_churn_vs_tau_psi": _spearman(churn, taupsi) if n >= 3 else None,
        "spearman_churn_vs_score_variance": _spearman(churn, var) if n >= 3 else None,
        "spearman_tau_psi_vs_score_variance": _spearman(taupsi, var) if n >= 3 else None,
    }


def _spearman(a: Sequence[float], b: Sequence[float]) -> float | None:
    """Spearman rank correlation (no scipy dep). ``None`` if either is constant."""
    if len(a) < 3 or len(a) != len(b):
        return None
    ra = _rankdata(a)
    rb = _rankdata(b)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def _rankdata(x: Sequence[float]) -> np.ndarray:
    """Average-rank of values (ties shared), mirroring scipy.stats.rankdata."""
    arr = np.asarray(x, dtype=float)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(len(arr), dtype=float)
    ranks[order] = np.arange(1, len(arr) + 1, dtype=float)
    # average ties
    _, inv, counts = np.unique(arr, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    return (sums / counts)[inv]
