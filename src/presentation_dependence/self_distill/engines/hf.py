"""HuggingFace transformers backend for the logit-skeleton engine.

Performance characteristics
---------------------------
Per-call cost on this stack: ``O(B × L_max²)`` attention (sdpa with
longest-in-batch padding). On a 4B model at ``L_max ≈ 3.2K`` (MS MARCO
B=20 with default geometry), one forward is ~4 sec on an L40S.

The big idle cost relative to vLLM is that the body of the prefix
(~3K tokens shared across all B prefixes within a chunk) is recomputed
once per row. vLLM's auto-prefix-caching collapses that to once per chunk;
the HF engine here cannot.
"""

from __future__ import annotations

from typing import Any, Sequence

from presentation_dependence.self_distill.engines.base import LogitSkeletonEngine
from presentation_dependence.self_distill.readout import expected_grade, grade_probabilities


def _left_truncate_ids(ids: list[int], max_length: int) -> tuple[list[int], int]:
    """Left-truncate ``ids`` to ``max_length``; return (ids, dropped_prefix_len)."""
    if max_length <= 0:
        raise ValueError(f"max_length must be positive, got {max_length}")
    if len(ids) <= max_length:
        return ids, 0
    offset = len(ids) - max_length
    return ids[offset:], offset


def compute_skeleton_slot_positions(  # noqa: C901
    tokenizer: Any,
    full_text: str,
    slot_prefixes: Sequence[str],
    max_length: int,
) -> tuple[list[int], list[int]]:
    """Tokenize the full grade skeleton once and locate each slot's read position.

    Returns ``(input_ids, read_positions)`` where ``read_positions[i]`` is the
    index, in the (left-truncated) token sequence, of the last token of
    ``slot_prefixes[i]`` — i.e. the position whose *next-token* logits give
    slot ``i``'s grade distribution. Each ``slot_prefixes[i]`` must be a string
    prefix of ``full_text`` (the skeleton builder guarantees this).

    This is the single-forward analogue of scoring each nested prefix
    separately: under causal masking the logits at ``read_positions[i]`` of one
    forward over ``full_text`` are identical to the final-token logits of a
    forward over ``slot_prefixes[i]`` alone — **provided the chunk fits within
    ``max_length``**. If left-truncation drops body tokens, the per-prefix path
    (which truncates each prefix independently) and this path diverge; we raise
    rather than silently return non-equivalent positions.

    Prefer ``offset_mapping`` on the full text (handles BPE/SentencePiece
    merges); fall back to token-prefix-length matching for tokenizers without
    offset support (e.g. test doubles).
    """
    enc: Any | None = None
    try:
        enc = tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=False,
            return_offsets_mapping=True,
        )
    except (TypeError, ValueError):
        enc = None

    offsets = enc.get("offset_mapping") if enc is not None else None
    if enc is not None:
        full_ids = list(enc["input_ids"])
    else:
        full_ids = list(tokenizer(full_text, add_special_tokens=False, truncation=False)["input_ids"])

    positions_orig: list[int] = []
    if offsets is not None and len(offsets) == len(full_ids):
        for prefix in slot_prefixes:
            if len(prefix) > len(full_text) or full_text[: len(prefix)] != prefix:
                raise ValueError("slot prefix is not a prefix of the full skeleton")
            cut = len(prefix)  # char index where this slot's dummy grade begins
            pos = -1
            for t, span in enumerate(offsets):
                if span is None:
                    continue
                st, en = int(span[0]), int(span[1])
                if st == 0 and en == 0:
                    continue
                if en <= cut:
                    pos = t
                else:
                    break
            if pos < 0:
                raise ValueError("could not align slot prefix to token spans")
            positions_orig.append(pos)
    else:
        for prefix in slot_prefixes:
            p_ids = list(tokenizer(prefix, add_special_tokens=False, truncation=False)["input_ids"])
            n = len(p_ids)
            if n == 0 or len(full_ids) < n or full_ids[:n] != p_ids:
                raise ValueError("slot prefix tokenization is not a prefix of the full skeleton")
            positions_orig.append(n - 1)

    input_ids, offset = _left_truncate_ids(full_ids, max_length)
    positions = [p - offset for p in positions_orig]
    if any(p < 0 or p >= len(input_ids) for p in positions):
        raise ValueError(
            "skeleton slot position fell outside the left-truncated window; "
            "single-forward readout is exact only when the chunk fits in max_length"
        )
    return input_ids, positions


def last_real_token_positions(inputs: dict, *, padding_side: str, device) -> Any:
    """Return per-row indices for logits at the final non-pad token.

    ``attention_mask.sum(dim=1) - 1`` is only correct for right padding. With
    left padding, the final real token is at the final tensor column for every
    row. Centralizing this avoids a subtle slot-logit bug where shorter rows in
    a left-padded batch read logits from the middle of the prompt.
    """
    import torch

    if padding_side == "left":
        return torch.full(
            (inputs["input_ids"].shape[0],),
            inputs["input_ids"].shape[1] - 1,
            dtype=torch.long,
            device=device,
        )
    return inputs["attention_mask"].sum(dim=1) - 1


class HFLogitSkeletonEngine(LogitSkeletonEngine):
    """Score prefix-tail next-token logits via a HuggingFace forward pass.

    Parameters
    ----------
    model :
        A loaded ``AutoModelForCausalLM`` already moved to its target device
        and ``.eval()``-ed. The engine does not own model lifecycle.
    tokenizer :
        Matching ``AutoTokenizer``. Must have ``padding_side="left"`` and
        ``truncation_side="left"`` so the slot tail (``"... Grade: "``) is
        preserved when prompts overflow ``max_length``.
    max_length : int
        Truncation upper bound applied to each prefix. Should match the
        reranker's ``reranker.max_length`` config.
    batch_size : int
        How many prefixes to pack in one HF forward. Defaults to the
        reranker's ``docs_per_score_forward`` (B). Larger batches help on
        large GPUs; smaller batches lower per-call peak memory.
    """

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        max_length: int,
        batch_size: int,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        if self.max_length <= 0:
            raise ValueError(f"max_length must be positive, got {self.max_length}")
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")

    def _final_logits(self, prefixes: Sequence[str]) -> Any:
        """Run the forward pass for every prefix and return final-token logits.

        Returns a 2D tensor ``[N, V]`` on the model's device, where row ``i``
        is the next-token logits at the tail of ``prefixes[i]``. Used by both
        :meth:`score_prefixes` (expected-grade readout) and
        :meth:`score_prefixes_argmax_grade` (Path A argmax).
        """
        import torch

        prev_trunc = getattr(self.tokenizer, "truncation_side", None)
        if prev_trunc is not None:
            self.tokenizer.truncation_side = "left"
        try:
            encoded = [
                self.tokenizer(
                    prefix,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=self.max_length,
                )["input_ids"]
                for prefix in prefixes
            ]
        finally:
            if prev_trunc is not None:
                self.tokenizer.truncation_side = prev_trunc

        chunks: list[Any] = []
        for start in range(0, len(encoded), self.batch_size):
            batch_ids = encoded[start : start + self.batch_size]
            inputs = self.tokenizer.pad(
                [{"input_ids": ids, "attention_mask": [1] * len(ids)} for ids in batch_ids],
                padding=True,
                pad_to_multiple_of=8,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = self.model(**inputs).logits

            last_positions = last_real_token_positions(
                inputs,
                padding_side=getattr(self.tokenizer, "padding_side", "right"),
                device=logits.device,
            )
            row_idx = torch.arange(logits.shape[0], device=logits.device)
            chunks.append(logits[row_idx, last_positions, :].detach())
        return torch.cat(chunks, dim=0)

    def score_prefixes(
        self,
        prefixes: Sequence[str],
        grade_token_ids: Sequence[int],
        grade_values: Sequence[float] = (0.0, 1.0, 2.0, 3.0),
    ) -> list[float]:
        if not prefixes:
            return []

        final_logits = self._final_logits(prefixes)
        expected = expected_grade(final_logits, grade_token_ids, grade_values)
        return [float(x) for x in expected.detach().cpu().float().tolist()]

    def score_prefix_probabilities(
        self,
        prefixes: Sequence[str],
        grade_token_ids: Sequence[int],
    ) -> list[list[float]]:
        """Return per-prefix grade probability vectors."""
        if not prefixes:
            return []
        final_logits = self._final_logits(prefixes)
        probs = grade_probabilities(final_logits, grade_token_ids)
        return [[float(v) for v in row] for row in probs.detach().cpu().float().tolist()]

    def score_skeleton_probabilities(
        self,
        full_text: str,
        slot_prefixes: Sequence[str],
        grade_token_ids: Sequence[int],
    ) -> list[list[float]]:
        """Single-forward multi-slot readout (one forward reads all B slots).

        Bit-identical to ``score_prefix_probabilities(slot_prefixes, …)`` when
        the chunk fits within ``max_length`` (causal-mask equivalence; see
        :func:`compute_skeleton_slot_positions`), but encodes the shared B-doc
        body exactly once instead of once per nested prefix.
        """
        if not slot_prefixes:
            return []
        import torch

        input_ids, positions = compute_skeleton_slot_positions(
            self.tokenizer, full_text, list(slot_prefixes), self.max_length
        )
        batch = torch.tensor([input_ids], dtype=torch.long, device=self.model.device)
        with torch.no_grad():
            logits = self.model(input_ids=batch).logits[0]  # [seq_len, vocab]
        pos_idx = torch.tensor(positions, dtype=torch.long, device=logits.device)
        selected = logits.index_select(0, pos_idx)  # [B, vocab]
        probs = grade_probabilities(selected, grade_token_ids)  # [B, G]
        return [[float(v) for v in row] for row in probs.detach().cpu().float().tolist()]

    def score_prefixes_argmax_grade(
        self,
        prefixes: Sequence[str],
        grade_token_ids: Sequence[int],
        grade_values: Sequence[float] = (0.0, 1.0, 2.0, 3.0),
    ) -> list[float]:
        """Path A readout: argmax over the four grade-token logits at each tail."""
        if not prefixes:
            return []
        if len(grade_token_ids) != len(grade_values):
            raise ValueError(
                f"grade_token_ids ({len(grade_token_ids)}) and grade_values "
                f"({len(grade_values)}) must have the same length"
            )

        import torch

        final_logits = self._final_logits(prefixes)
        grade_logits = final_logits[:, [int(g) for g in grade_token_ids]]
        idx = torch.argmax(grade_logits, dim=-1).detach().cpu().tolist()
        vals = [float(v) for v in grade_values]
        return [vals[int(i)] for i in idx]
