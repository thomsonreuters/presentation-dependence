"""Inverse-propensity slot weights for the DebiasFirst-mechanism port.

This is the *only* new ingredient over shuffled-view augmentation
(``objective.type: supervised_mse`` with ``data.permutation_augmentation``)
needed to reproduce DebiasFirst (Qiao et al., 2026, arXiv:2604.03642) as a
batched-pointwise baseline. With :func:`uniform_weights` the objective reduces
to that augmentation arm exactly.

Paper vs. this port:
  - Qiao+2026 (arXiv:2604.03642v1 §4.1) defines a *2-D transition* propensity
    ``omega_{i, pi-bar}`` over (input position i -> model-predicted output
    position pi-bar), estimated from the model's own rankings on shuffled
    inputs (their Eq. 5), and applies it to a *pairwise* FIRST-style
    learning-to-rank loss (their Eq. 4): ``L = lambda * L_Rank-IPS + L_LM``
    (their Eq. 6).
  - This port uses batched-pointwise expected-grade regression (per-doc expected
    grade -> MSE vs. silver), so we port the mechanism to its 1-D, data-only
    form: ``p(slot | relevant)`` is the marginal distribution of *relevant
    documents over their first-stage (canonical BM25) input slots*, estimated
    once from the training qrels + first-stage order; the per-(doc, slot) MSE
    is then weighted by ``1 / p(slot)``. Same intent (down-weight the
    over-represented early slots, up-weight the rare late ones so the model
    treats positions more symmetrically), adapted to our loss.

The functions here are pure (no torch, no I/O) so they are unit-testable on CPU
and can be reused at train time, where the weights are estimated once before the
loop and written to ``ips_slot_propensity.json`` in the run directory.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol


class _SlotExample(Protocol):
    """Structural type for what the estimator reads off a training example.

    :class:`presentation_dependence.self_distill.student_data.GroupedSilverExample`
    satisfies it. We read ``doc_ids`` in first-stage order and ``qrels``
    (doc_id -> graded relevance).
    """

    doc_ids: list[str]
    qrels: dict[str, int]


@dataclass(slots=True)
class SlotPropensity:
    """Result of :func:`estimate_slot_propensity`.

    ``weights[s]`` is the per-(doc, slot) loss multiplier for within-chunk
    slot ``s`` (0-indexed); it is what the loss consumes. ``propensity[s]`` is
    the smoothed ``p(slot=s | relevant)`` it was derived from. The remaining
    fields are diagnostics for logging / pre-registration.
    """

    chunk_size: int
    counts: list[int]
    propensity: list[float]
    weights: list[float]
    total_positives: int
    n_examples: int
    n_empty_slots: int
    relevance_threshold: int
    smoothing_eps: float
    clip: float | None

    def as_dict(self) -> dict[str, object]:
        return {
            "chunk_size": self.chunk_size,
            "counts": list(self.counts),
            "propensity": list(self.propensity),
            "weights": list(self.weights),
            "total_positives": self.total_positives,
            "n_examples": self.n_examples,
            "n_empty_slots": self.n_empty_slots,
            "relevance_threshold": self.relevance_threshold,
            "smoothing_eps": self.smoothing_eps,
            "clip": self.clip,
        }


def estimate_slot_propensity(
    examples: Iterable[_SlotExample],
    *,
    chunk_size: int,
    relevance_threshold: int = 1,
    smoothing_eps: float = 1e-3,
    clip: float | None = None,
) -> SlotPropensity:
    """Estimate ``p(slot | relevant)`` and the IPS loss weights from training data.

    For every example, each document at first-stage list index ``idx`` falls in
    within-chunk slot ``idx % chunk_size``. That is the value
    :func:`presentation_dependence.self_distill.student_data.regression_chunks` records as
    ``RegressionChunk.canonical_slots``, so the weight a document receives here
    is the weight the loss applies to it, whatever order it is presented in.
    Note this is the *dense* first-stage index, not the BM25 rank, which is
    sparse whenever the candidate pool was subsampled. A document counts toward
    its slot iff it is qrels-relevant (graded rel ``>= relevance_threshold``).

    The smoothed propensity is
    ``p[s] = (counts[s] + eps) / sum_s (counts[s] + eps)`` (Laplace smoothing,
    so empty slots never blow up the inverse), and the weights are
    ``w[s] = (1 / p[s])`` normalized so ``mean(w) == 1`` (keeps the overall
    loss scale identical to Pos-Aug-only; only the *relative* per-slot emphasis
    changes). ``clip`` optionally caps each weight before re-normalizing, for
    stability against a single near-empty slot.

    ``examples`` are training examples in first-stage (BM25) order carrying gold
    ``qrels``, and ``chunk_size`` is ``B``, the documents per scored input list.
    ``relevance_threshold`` is the minimum graded relevance for a document to
    count (binary MS MARCO qrels are rel=1). ``smoothing_eps`` is the Laplace
    additive smoothing on the slot counts and must be > 0. ``clip`` optionally
    caps each weight before the final mean-1 normalization.
    """
    if chunk_size < 2:
        raise ValueError(f"chunk_size must be >= 2, got {chunk_size}")
    if smoothing_eps <= 0.0:
        raise ValueError(f"smoothing_eps must be > 0, got {smoothing_eps}")
    if clip is not None and clip <= 0.0:
        raise ValueError(f"clip must be > 0 when set, got {clip}")

    counts = [0] * chunk_size
    total_positives = 0
    n_examples = 0
    for ex in examples:
        n_examples += 1
        qrels = ex.qrels or {}
        for idx, doc_id in enumerate(ex.doc_ids):
            if int(qrels.get(doc_id, 0)) >= relevance_threshold:
                counts[idx % chunk_size] += 1
                total_positives += 1

    if total_positives == 0:
        raise ValueError(
            "no qrels-relevant documents found in the training examples; "
            "cannot estimate IPS slot propensity (check qrels_path / "
            "relevance_threshold / candidate set overlap)"
        )

    smoothed = [c + smoothing_eps for c in counts]
    denom = sum(smoothed)
    propensity = [s / denom for s in smoothed]

    inv = [1.0 / p for p in propensity]
    if clip is not None:
        inv = [min(v, clip) for v in inv]
    mean_inv = sum(inv) / len(inv)
    weights = [v / mean_inv for v in inv]

    return SlotPropensity(
        chunk_size=chunk_size,
        counts=counts,
        propensity=propensity,
        weights=weights,
        total_positives=total_positives,
        n_examples=n_examples,
        n_empty_slots=sum(1 for c in counts if c == 0),
        relevance_threshold=relevance_threshold,
        smoothing_eps=smoothing_eps,
        clip=clip,
    )


def uniform_weights(chunk_size: int) -> list[float]:
    """All-ones weights of length ``chunk_size`` — the IPS no-op (== Pos-Aug-only)."""
    if chunk_size < 2:
        raise ValueError(f"chunk_size must be >= 2, got {chunk_size}")
    return [1.0] * chunk_size


def format_slot_table(prop: SlotPropensity) -> str:
    """One-line-per-slot human-readable table for logs / CLI output."""
    lines = [
        f"IPS slot propensity (chunk_size={prop.chunk_size}, "
        f"positives={prop.total_positives}, examples={prop.n_examples}, "
        f"empty_slots={prop.n_empty_slots}, eps={prop.smoothing_eps}, clip={prop.clip})",
        f"  {'slot':>4} {'count':>8} {'p(slot|rel)':>12} {'weight':>10}",
    ]
    for s in range(prop.chunk_size):
        lines.append(f"  {s:>4} {prop.counts[s]:>8} {prop.propensity[s]:>12.5f} {prop.weights[s]:>10.4f}")
    return "\n".join(lines)


# Re-exported for callers that only need the field name.
__all__ = [
    "SlotPropensity",
    "estimate_slot_propensity",
    "uniform_weights",
    "format_slot_table",
]
