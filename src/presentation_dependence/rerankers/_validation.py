"""Debug/test validators for reranker contract boundaries."""

from __future__ import annotations

from collections import Counter

from presentation_dependence.rerankers.base import Passage, RankResult


def validate_rank_result(result: RankResult, passages: list[Passage]) -> None:
    """Validate a ``RankResult`` against the input passages it reranked."""
    input_pids = [str(p["pid"]) for p in passages]
    input_pid_set = set(input_pids)

    top = result.get("top_k_psgs")
    if top is None:
        raise ValueError("RankResult missing top_k_psgs")

    top_pids = [str(p["pid"]) for p in top]
    if not passages and top_pids:
        raise ValueError("RankResult for empty input must have empty top_k_psgs")

    unknown = sorted(set(top_pids) - input_pid_set)
    if unknown:
        raise ValueError(f"RankResult top_k_psgs contains pids not present in input: {unknown}")

    dupes = sorted(pid for pid, count in Counter(top_pids).items() if count > 1)
    if dupes:
        raise ValueError(f"RankResult top_k_psgs contains duplicate pids: {dupes}")

    scores = result.get("scores_init_order")
    if scores is not None and len(scores) != len(passages):
        raise ValueError(
            f"RankResult scores_init_order length {len(scores)} does not match input length {len(passages)}"
        )

    runtimes = result.get("prompting_runtimes")
    if runtimes is None:
        raise ValueError("RankResult missing prompting_runtimes")

    paradigm = result.get("paradigm")
    if paradigm is None:
        raise ValueError("RankResult missing paradigm")
