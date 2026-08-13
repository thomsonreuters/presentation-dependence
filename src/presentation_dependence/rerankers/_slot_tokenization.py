"""Tokenization helpers that locate each slot's readout position in a shared prompt.

Used by every batched-pointwise readout that reads logits at fixed positions
rather than decoding: the grade skeleton (`self_distill.student`,
`rerankers.qwen3` grade modes) and the Qwen3 setwise Yes/No mode.
"""

from __future__ import annotations

from typing import Any


# Stands in for the answer token while the slot boundary is located. Only its
# character length is used, so changing this string shifts the computed span.
_DUMMY_ANSWER = "Yes"


def _left_truncate_token_ids(ids: list[int], max_length: int) -> tuple[list[int], int]:
    """Keep the last ``max_length`` tokens."""
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    if len(ids) <= max_length:
        return ids, 0
    drop = len(ids) - max_length
    return ids[drop:], drop


def tokenize_tail_preserving(
    tokenizer: Any,
    text: str,
    *,
    max_length: int,
    add_special_tokens: bool = False,
) -> dict[str, Any]:
    """Encode with left truncation while restoring tokenizer state."""
    if not hasattr(tokenizer, "truncation_side"):
        return tokenizer(
            text,
            add_special_tokens=add_special_tokens,
            truncation=True,
            max_length=max_length,
        )
    previous = tokenizer.truncation_side
    tokenizer.truncation_side = "left"
    try:
        return tokenizer(
            text,
            add_special_tokens=add_special_tokens,
            truncation=True,
            max_length=max_length,
        )
    finally:
        tokenizer.truncation_side = previous


def _slot_position_from_offsets(  # noqa: C901
    offsets: list[Any],
    *,
    slot_char: int,
) -> int:
    """Find the final prefix token before the slot's answer position."""
    position = -1
    for token_index, span in enumerate(offsets):
        if span is None:
            continue
        start, end = int(span[0]), int(span[1])
        if end == 0 and start == 0:
            continue
        if end <= slot_char:
            position = token_index
        else:
            break
    if position < 0:
        raise ValueError("could not align slot boundary to token spans")

    dummy_span_end = slot_char + len(_DUMMY_ANSWER)
    found_dummy = False
    for token_index in range(position + 1, len(offsets)):
        span = offsets[token_index]
        if span is None:
            continue
        start, end = int(span[0]), int(span[1])
        if start == 0 and end == 0:
            continue
        if start < dummy_span_end and end > slot_char:
            found_dummy = True
            break
    if not found_dummy:
        raise ValueError("no token overlaps dummy answer span after prefix boundary")
    return position


def prepare_single_forward(
    tokenizer: Any,
    full_text: str,
    slot_prefixes: list[str],
    max_length: int,
) -> tuple[list[int], list[int]]:
    """Tokenize one shared prompt and locate each slot's readout position."""
    original_positions: list[int] = []
    encoded: Any | None = None
    try:
        encoded = tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=False,
            return_offsets_mapping=True,
        )
    except (TypeError, ValueError):
        encoded = None

    offsets = encoded.get("offset_mapping") if encoded is not None else None
    encoded_ids = encoded["input_ids"] if encoded is not None else None
    full_ids: list[int]
    if offsets is not None and encoded_ids is not None and len(offsets) == len(encoded_ids):
        full_ids = list(encoded_ids)
        for prefix in slot_prefixes:
            if len(prefix) > len(full_text) or full_text[: len(prefix)] != prefix:
                raise ValueError("internal: slot prefix is not a prefix of full prompt")
            original_positions.append(_slot_position_from_offsets(offsets, slot_char=len(prefix)))
    else:
        full_ids = tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]
        for prefix in slot_prefixes:
            prefix_ids = tokenizer(
                prefix,
                add_special_tokens=False,
                truncation=False,
            )["input_ids"]
            if not prefix_ids:
                raise ValueError("empty slot prefix tokenization")
            prefix_length = len(prefix_ids)
            if len(full_ids) < prefix_length or full_ids[:prefix_length] != prefix_ids:
                raise ValueError("slot prefix tokenization mismatch vs full prompt")
            original_positions.append(prefix_length - 1)

    input_ids, offset = _left_truncate_token_ids(full_ids, max_length)
    positions = [position - offset for position in original_positions]
    if any(position < 0 or position >= len(input_ids) for position in positions):
        raise ValueError("slot readout positions outside truncated window")
    return input_ids, positions
