"""Identity reranker: return the first-stage order unchanged.

Smoke-test path for dataloading, TREC run writing, and eval without a GPU or
model download. Numbers equal the first-stage retriever (typically BM25).
"""

from __future__ import annotations

import time

from presentation_dependence.rerankers.base import Passage, Query, RankResult, Reranker


class IdentityReranker(Reranker):
    # Smoke-test reranker: exposes BM25-style scalar scores for the input order.
    # τ-PSI geometry still keys off the class name, not this paradigm value.
    paradigm = "pointwise"

    def rank(self, query: Query, passages: list[Passage]) -> RankResult:
        del query  # unused
        t0 = time.perf_counter()
        top_k = [
            {"pid": p["pid"], "text": p["text"], "score": float(len(passages) - i)} for i, p in enumerate(passages)
        ]
        elapsed = time.perf_counter() - t0
        return {
            "top_k_psgs": top_k,
            "scores_init_order": [float(len(passages) - i) for i in range(len(passages))],
            "prompting_runtimes": [elapsed],
            "paradigm": self.paradigm,
        }
