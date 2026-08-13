"""Top-k ranking stability metrics for repeated scorer presentations."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations
from math import nan

import numpy as np


def rank_biased_overlap(
    ranking_a: Sequence[str],
    ranking_b: Sequence[str],
    *,
    depth: int = 10,
    persistence: float = 0.9,
) -> float:
    """Return extrapolated finite-depth rank-biased overlap.

    The Webber et al. extrapolated form is
    ``(1-p) * sum_d A_d p^(d-1) + A_k p^k``, where ``A_d`` is prefix
    overlap at depth ``d`` and ``k`` is ``depth``. It is 1 for identical
    prefixes and 0 for disjoint prefixes. Both rankings must contain at least
    ``depth`` unique documents so the top-k estimand is unambiguous.
    """
    if depth < 1:
        raise ValueError("RBO depth must be positive")
    if not 0.0 < persistence < 1.0:
        raise ValueError("RBO persistence must be between zero and one")
    if len(ranking_a) < depth or len(ranking_b) < depth:
        raise ValueError("rankings must contain at least depth documents")

    prefix_a = [str(doc_id) for doc_id in ranking_a[:depth]]
    prefix_b = [str(doc_id) for doc_id in ranking_b[:depth]]
    if len(set(prefix_a)) != depth or len(set(prefix_b)) != depth:
        raise ValueError("rankings must not contain duplicate documents")

    seen_a: set[str] = set()
    seen_b: set[str] = set()
    weighted_agreement = 0.0
    agreement_at_depth = 0.0
    for index, (doc_a, doc_b) in enumerate(
        zip(prefix_a, prefix_b, strict=True),
        start=1,
    ):
        seen_a.add(doc_a)
        seen_b.add(doc_b)
        agreement_at_depth = len(seen_a & seen_b) / index
        weighted_agreement += (1.0 - persistence) * agreement_at_depth * persistence ** (index - 1)
    return float(weighted_agreement + agreement_at_depth * persistence**depth)


def mean_pairwise_rbo(
    rankings: Sequence[Sequence[str]],
    *,
    depth: int = 10,
    persistence: float = 0.9,
) -> float:
    """Average RBO over every pair of repeated rankings."""
    if len(rankings) < 2:
        return nan
    values = [
        rank_biased_overlap(
            ranking_a,
            ranking_b,
            depth=depth,
            persistence=persistence,
        )
        for ranking_a, ranking_b in combinations(rankings, 2)
    ]
    return float(np.mean(values))
