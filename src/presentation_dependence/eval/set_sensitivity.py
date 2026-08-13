"""Set-sensitivity variance components for batched-pointwise score logs.

Zero-GPU re-analysis of an existing ``k_input > B`` permutation run (the headline
tau-PSI setup): every ``random_shuffle`` permutation reshuffles the whole
candidate list and the scorer re-chunks it into groups of ``B`` documents, so a
document's ``B-1`` companions change every permutation. Measures how much of a
document's per-permutation score variance is **not** explained by its readout
slot or its chunk index: the residual that the slot (beta) and within-chunk
order (gamma) channels of the ``k_input == B`` beta/gamma runs cannot reach,
which is the set-composition + within-chunk-prefix-order channel.

Facts the estimator relies on (see ``presentation_dependence.eval.psi_manager``):

* The input order of ``random_shuffle`` permutation ``i`` is
  ``random.Random(seeds[i]).shuffle(first_stage_passages[:k_input])``: a pure
  index permutation, reproducible from the first-stage candidate order alone
  (``reconstruct_positions``). No GPU, no rerank.
* ``psi/per_query_results/<qid>/aligned_scores.json`` stores, per document, its
  scalar score in **seed order** for the ``random_shuffle`` permutations only:
  exactly the M = K uniform-random shots we want (``middle_injection`` shots are
  excluded there, which is correct: they bias positions).
* Scoring is deterministic (greedy logprob readout), so there is no per-call
  sampling noise: the within-document score variance is entirely
  slot + chunk-index + companion-set + within-chunk-order.

Estimator. For one query we fit the additive model

    score(d, perm) = mu_d + a_slot[slot] + g_chunk[chunk] + eps(d, perm)

by ordinary least squares with document, slot, and chunk fixed effects (the
slot/chunk effects are pooled across the query's documents, where they are
well-estimated; ``mu_d`` is the per-document mean). The residual ``eps`` is the
set-composition + within-chunk-order channel. We report, per document averaged:

* ``total_var``: within-document score variance (after ``mu_d`` only),
* ``resid_var_slot``: residual after ``mu_d`` + slot,
* ``resid_var``: residual after ``mu_d`` + slot + chunk (the headline),

and ``residual_fraction = resid_var / total_var``.

Caveats (carried into the generated doc):

* Observationally, set-composition and within-chunk prefix-order stay entangled
  (both are "who else is in the chunk"); Step 0 is suggestive, not clean — that
  is what the instrument fixes.
* ``mu_d`` is removed from M = K (typically 10) shots, so ``total_var`` and
  ``resid_var`` are both deflated by ~``(M-1)/M``; this factor is common to every
  dataset and cancels in the ArguAna-vs-control contrast that is the whole point.
* Chunk index is not seen by the model (each chunk is an independent forward), so
  ``g_chunk`` is expected to absorb ~nothing. It is still regressed out to
  measure and control any systematic score differences between chunks.
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np


# A document is usable in the estimator only when it carries a full set of M
# permutation scores (one per random_shuffle seed); partial docs are skipped.
DocRows = dict[str, list[tuple[float, int, int]]]  # pid -> [(score, slot, chunk), ...]


def reconstruct_positions(
    first_stage_pids: Sequence[str],
    seeds: Sequence[int],
) -> list[dict[str, int]]:
    """Reproduce each ``random_shuffle`` permutation's input position map.

    Mirrors ``PsiExperimentRunner._random_shuffle``: a deterministic Fisher-Yates
    shuffle of the (already ``k_input``-truncated) first-stage list, seeded per
    permutation. Returns ``[{pid: position}, ...]`` in seed order.

    ``random.shuffle`` permutes purely by index, so shuffling the pid list yields
    the identical permutation as the runner's shuffle of the passage dicts.
    """
    pids = [str(p) for p in first_stage_pids]
    out: list[dict[str, int]] = []
    for seed in seeds:
        order = list(pids)
        random.Random(int(seed)).shuffle(order)
        out.append({pid: pos for pos, pid in enumerate(order)})
    return out


def assemble_doc_rows(
    first_stage_pids: Sequence[str],
    seeds: Sequence[int],
    aligned_scores: Mapping[str, Sequence[float]],
    batch_size: int,
) -> DocRows:
    """Join reconstructed positions with seed-ordered scores into per-doc rows.

    Parameters
    ----------
    first_stage_pids:
        The query's first-stage candidate pids, **already truncated to**
        ``k_input`` and in first-stage order (what the runner shuffled).
    seeds:
        The ``random_shuffle`` seeds, in order (must match the score vectors).
    aligned_scores:
        ``{pid: [score_seed0, score_seed1, ...]}`` from ``aligned_scores.json``.
    batch_size:
        ``B`` = ``reranker.docs_per_score_forward``; slot = ``pos % B``,
        chunk = ``pos // B``.

    Returns a mapping ``pid -> [(score, slot, chunk), ...]`` over the perms, only
    for pids that (a) appear in the first-stage list and (b) have one score per
    seed. Pids missing or with the wrong number of scores are skipped.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    positions = reconstruct_positions(first_stage_pids, seeds)
    n_perms = len(seeds)
    rows: DocRows = {}
    for pid in (str(p) for p in first_stage_pids):
        scores = aligned_scores.get(pid)
        if scores is None or len(scores) != n_perms:
            continue
        doc_rows: list[tuple[float, int, int]] = []
        ok = True
        for perm_idx in range(n_perms):
            pos = positions[perm_idx].get(pid)
            if pos is None:
                ok = False
                break
            score = float(scores[perm_idx])
            if not math.isfinite(score):
                ok = False
                break
            doc_rows.append((score, pos % batch_size, pos // batch_size))
        if ok and doc_rows:
            rows[pid] = doc_rows
    return rows


def _one_hot(labels: Sequence[int]) -> np.ndarray:
    """Dense one-hot matrix (n_rows, n_levels) over the distinct integer labels."""
    uniq = sorted(set(labels))
    index = {lvl: j for j, lvl in enumerate(uniq)}
    mat = np.zeros((len(labels), len(uniq)), dtype=np.float64)
    for i, lvl in enumerate(labels):
        mat[i, index[lvl]] = 1.0
    return mat


def _ols_residuals(y: np.ndarray, design: np.ndarray) -> np.ndarray:
    """Residuals ``y - X beta_hat`` via least squares (min-norm, rank-safe).

    The design is intentionally collinear (overlapping fixed-effect blocks each
    carry a constant); ``lstsq`` returns the minimum-norm solution, for which the
    fitted values (and the residuals) are unique regardless.
    """
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    return y - design @ beta


@dataclass(frozen=True)
class QuerySetSensitivity:
    """Per-query set-sensitivity variance components (means over documents)."""

    query_id: str
    n_docs: int
    n_perms: int
    total_var: float
    resid_var_slot: float
    resid_var: float
    explained_slot: float
    explained_chunk: float
    residual_fraction: float | None
    n_rel_docs: int
    resid_var_rel: float | None
    resid_var_nonrel: float | None
    total_var_rel: float | None
    total_var_nonrel: float | None


def compute_query_metrics(
    query_id: str,
    doc_rows: DocRows,
    *,
    rel_pids: Iterable[str] | None = None,
) -> QuerySetSensitivity | None:
    """Variance components for one query from its per-doc ``(score, slot, chunk)``.

    Fits document + slot + chunk fixed effects jointly (OLS), then measures the
    per-document residual / total within-doc variance. Returns ``None`` when the
    query has too few documents to fit (need >= 2 docs with rows).
    """
    pids = [pid for pid, rows in doc_rows.items() if rows]
    if len(pids) < 2:
        return None

    # Flatten to row-aligned arrays, tagging each row with its document index.
    scores: list[float] = []
    slots: list[int] = []
    chunks: list[int] = []
    doc_idx: list[int] = []
    n_perms_set: set[int] = set()
    for d, pid in enumerate(pids):
        rows = doc_rows[pid]
        n_perms_set.add(len(rows))
        for score, slot, chunk in rows:
            scores.append(score)
            slots.append(slot)
            chunks.append(chunk)
            doc_idx.append(d)

    y = np.asarray(scores, dtype=np.float64)
    doc_oh = _one_hot(doc_idx)
    slot_oh = _one_hot(slots)
    chunk_oh = _one_hot(chunks)

    # Nested models: doc-only (= within-doc centering), +slot, +slot+chunk.
    resid_doc = _ols_residuals(y, doc_oh)
    resid_doc_slot = _ols_residuals(y, np.hstack([doc_oh, slot_oh]))
    resid_full = _ols_residuals(y, np.hstack([doc_oh, slot_oh, chunk_oh]))

    rel = {str(p) for p in (rel_pids or [])}
    doc_arr = np.asarray(doc_idx)

    def _per_doc_mean(resid: np.ndarray, mask: np.ndarray | None = None) -> float | None:
        sel = resid if mask is None else resid[mask]
        groups = doc_arr if mask is None else doc_arr[mask]
        if sel.size == 0:
            return None
        # Mean over documents of that document's mean squared residual.
        per_doc: list[float] = []
        for d in np.unique(groups):
            vals = sel[groups == d]
            per_doc.append(float(np.mean(vals * vals)))
        return float(np.mean(per_doc)) if per_doc else None

    total_var = _per_doc_mean(resid_doc) or 0.0
    resid_var_slot = _per_doc_mean(resid_doc_slot) or 0.0
    resid_var = _per_doc_mean(resid_full) or 0.0

    rel_mask = np.asarray([pids[d] in rel for d in doc_idx])
    nonrel_mask = ~rel_mask
    n_rel = int(sum(1 for pid in pids if pid in rel))

    residual_fraction = (resid_var / total_var) if total_var > 0 else None

    return QuerySetSensitivity(
        query_id=str(query_id),
        n_docs=len(pids),
        n_perms=max(n_perms_set) if n_perms_set else 0,
        total_var=total_var,
        resid_var_slot=resid_var_slot,
        resid_var=resid_var,
        explained_slot=max(total_var - resid_var_slot, 0.0),
        explained_chunk=max(resid_var_slot - resid_var, 0.0),
        residual_fraction=residual_fraction,
        n_rel_docs=n_rel,
        resid_var_rel=_per_doc_mean(resid_full, rel_mask) if n_rel else None,
        resid_var_nonrel=_per_doc_mean(resid_full, nonrel_mask),
        total_var_rel=_per_doc_mean(resid_doc, rel_mask) if n_rel else None,
        total_var_nonrel=_per_doc_mean(resid_doc, nonrel_mask),
    )


SUMMARY_METRICS: tuple[str, ...] = (
    "total_var",
    "resid_var_slot",
    "resid_var",
    "explained_slot",
    "explained_chunk",
    "residual_fraction",
    "resid_var_rel",
    "resid_var_nonrel",
    "total_var_rel",
    "total_var_nonrel",
)


def summarize_per_query(per_query: Sequence[QuerySetSensitivity]) -> dict:
    """Aggregate per-query metrics to query-mean summary fields."""
    if not per_query:
        raise ValueError("No per-query set-sensitivity metrics to summarize")
    summary: dict[str, object] = {
        "n_queries": len(per_query),
        "n_docs_total": int(sum(q.n_docs for q in per_query)),
        "n_docs_min": min(q.n_docs for q in per_query),
        "n_docs_max": max(q.n_docs for q in per_query),
        "n_perms_min": min(q.n_perms for q in per_query),
        "n_perms_max": max(q.n_perms for q in per_query),
        "n_rel_docs_total": int(sum(q.n_rel_docs for q in per_query)),
    }
    for metric in SUMMARY_METRICS:
        vals = [getattr(q, metric) for q in per_query]
        vals = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
        summary[metric] = float(np.mean(vals)) if vals else None
    # A corpus-level residual fraction from the pooled means is more stable than
    # the mean of per-query ratios (which over-weights low-variance queries).
    tv = summary.get("total_var")
    rv = summary.get("resid_var")
    slot = summary.get("explained_slot")
    chunk = summary.get("explained_chunk")
    summary["residual_fraction_pooled"] = (
        float(rv) / float(tv) if isinstance(tv, float) and isinstance(rv, float) and tv > 0 else None
    )
    # Additive shares of within-document score variance (after mu_d). These three
    # sum to 1 when the nested OLS residuals are well-behaved. ``share_companion``
    # is the observational residual after slot + chunk: companion identity
    # entangled with within-chunk order; Step 0 cannot split those further.
    if isinstance(tv, float) and tv > 0:
        summary["share_slot"] = float(slot) / tv if isinstance(slot, float) else None
        summary["share_chunk"] = float(chunk) / tv if isinstance(chunk, float) else None
        summary["share_companion"] = float(rv) / tv if isinstance(rv, float) else None
    else:
        summary["share_slot"] = None
        summary["share_chunk"] = None
        summary["share_companion"] = None
    return summary


def per_query_to_dicts(per_query: Sequence[QuerySetSensitivity]) -> list[dict]:
    """Convert per-query dataclasses to JSON-serializable dicts."""
    return [asdict(row) for row in per_query]
