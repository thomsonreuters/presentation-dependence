"""Shared helpers for building project ``RankResult`` payloads."""

from __future__ import annotations

from presentation_dependence.rerankers.base import Passage, RankResult, RerankerParadigm


def scores_to_rank_result(
    scores: list[float],
    passages: list[Passage],
    elapsed: float,
    paradigm: RerankerParadigm,
    *,
    model_name: str = "reranker",
) -> RankResult:
    """Map input-order scalar scores to the shared ``RankResult`` contract."""
    if len(scores) != len(passages):
        raise ValueError(f"{model_name} produced {len(scores)} scores for {len(passages)} passages")

    order = sorted(range(len(passages)), key=lambda i: (-scores[i], i))
    top_k = [
        {
            "pid": passages[i]["pid"],
            "text": passages[i]["text"],
            "score": float(scores[i]),
        }
        for i in order
    ]
    return {
        "top_k_psgs": top_k,
        "scores_init_order": [float(s) for s in scores],
        "prompting_runtimes": [elapsed],
        "paradigm": paradigm,
    }
