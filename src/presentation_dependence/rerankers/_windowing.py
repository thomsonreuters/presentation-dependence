"""Pure helpers for generative listwise window reranking.

Apache-2.0 source and modification notice: this module adapts the end-to-start
window schedule from ``sunnweiwei/RankGPT`` revision
``0d62bc3855c7c118048a7c47c18e719b938e291a`` as validated pure helpers shared
across the generative-listwise wrappers. Unlike upstream, the local schedule
always adds a final ``(0, window_size)`` window when the stride does not land
exactly on zero; it also validates dimensions and handles empty/small pools
explicitly. The reviewed upstream source carries no copyright or attribution
header and the repository has no root ``NOTICE`` file.

See ``THIRD_PARTY_NOTICES.md``; licence text at
``third_party_licenses/Apache-2.0.txt``.
"""

from __future__ import annotations

from presentation_dependence.rerankers.base import Passage


def apply_window_permutation(
    passages: list[Passage],
    start: int,
    end: int,
    local_perm: list[int],
) -> list[Passage]:
    """Return a copy with ``passages[start:end]`` reordered by ``local_perm``."""
    window = passages[start:end]
    if len(local_perm) != len(window):
        raise ValueError(
            f"local_perm length {len(local_perm)} does not match window size {len(window)} for slice [{start}, {end})"
        )
    reordered_window = [window[i] for i in local_perm]
    return passages[:start] + reordered_window + passages[end:]


def sliding_window_bubble_schedule(n: int, window_size: int, stride: int) -> list[tuple[int, int]]:
    """Return end-to-start overlapping windows for listwise bubble reranking."""
    if window_size <= 0 or stride <= 0:
        raise ValueError(f"window_size and stride must be > 0 (got W={window_size}, S={stride})")
    if n <= 0:
        return []
    if n <= window_size:
        return [(0, n)]

    schedule: list[tuple[int, int]] = []
    start = n - window_size
    end = n
    while start > 0:
        schedule.append((start, end))
        start -= stride
        end -= stride
    schedule.append((0, window_size))
    return schedule
