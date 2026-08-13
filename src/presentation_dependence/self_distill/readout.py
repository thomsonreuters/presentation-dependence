"""Continuous-readout primitives for K-shot BSC self-distillation.

K=15 averaged labels are continuous values in ``[0, 3]``, e.g.
``(3 + 3 + 2 + ... + 1) / 10 = 2.4``. The teacher emits a continuous prediction
for each slot. An integer readout collapses the grade distribution to its mode
and removes the variation used by BSC distillation.

At the position where the model would emit a grade digit ``{0, 1, 2, 3}``:

1. Take the raw logits over the full vocabulary at that position.
2. Slice out the four logits corresponding to the digit tokens.
3. Softmax over those four to a probability distribution.
4. Compute expected value: ``E[g] = Σ_g  g · P(g)``.

The result is a continuous value in ``[0, 3]`` for each slot.

Two backends:

- vLLM ``prompt_logprobs``: production path, deferred until vLLM is wired into
  ``presentation_dependence`` (pyproject.toml line 11 has it intentionally unpinned).
- HF transformers: works directly on the per-position logit tensor returned
  by ``model.generate(..., output_logits=True)`` or ``model.forward(...)``.
  Slower than vLLM at scale, but functionally equivalent and immediately
  available in this stack.

Defines only the calculation. Where to read the logits from (generation step,
fixed prefix position, etc.) is up to the caller: each reranker wrapper knows
its own grade-emission positions.
"""

from __future__ import annotations

from typing import Sequence

# Type aliases (avoid hard torch import at module top so this file is
# importable in environments without torch: useful for tests on the schema).
TensorLike = "torch.Tensor"  # type: ignore[assignment]


_DEFAULT_GRADE_STRINGS: tuple[str, ...] = ("0", "1", "2", "3")
_DEFAULT_GRADE_VALUES: tuple[float, ...] = (0.0, 1.0, 2.0, 3.0)


def expected_grade(
    logits,
    grade_token_ids: Sequence[int],
    grade_values: Sequence[float] = _DEFAULT_GRADE_VALUES,
):
    """Expected grade ``E[g] = Σ_g g · softmax(logits[grade_ids])[g]``.

    Pure / I/O-free. Caller supplies a logits tensor; production path
    (HF forward, generate, or vLLM ``prompt_logprobs``) is out of scope here.

    Parameters
    ----------
    logits : torch.Tensor
        Shape ``(..., vocab_size)``. The last dim is the vocabulary axis.
        Any leading shape is preserved (use this to score B slots in one call).
    grade_token_ids : Sequence[int]
        Length ``G``. Token IDs of the grade-emission digits, e.g.
        ``[id("0"), id("1"), id("2"), id("3")]``. Must come from the teacher's
        own tokenizer; see :func:`resolve_grade_token_ids`.
    grade_values : Sequence[float], default ``(0.0, 1.0, 2.0, 3.0)``
        Length ``G``. Numeric values associated with each grade. The default uses
        ``E[g] = Σ_{g∈{0,1,2,3}} g · P(g)``.

    Returns:
    -------
    torch.Tensor
        Shape ``logits.shape[:-1]``. Values in
        ``[min(grade_values), max(grade_values)]`` — clipped only by the
        softmax + dot product, not by an explicit clamp, so it stays
        differentiable for the future student-SFT MSE loss.

    Notes:
    -----
    The function is differentiable end-to-end: softmax + linear combination
    has well-defined gradients with respect to ``logits``, allowing student MSE
    training against continuous teacher labels.
    """
    import torch

    if len(grade_token_ids) != len(grade_values):
        raise ValueError(
            f"grade_token_ids ({len(grade_token_ids)}) and grade_values ({len(grade_values)}) must have the same length"
        )
    if len(grade_token_ids) == 0:
        raise ValueError("grade_token_ids must not be empty")

    if not isinstance(logits, torch.Tensor):
        raise TypeError(f"logits must be a torch.Tensor, got {type(logits).__name__}")
    if logits.dim() < 1:
        raise ValueError(f"logits must have at least one dimension, got shape {tuple(logits.shape)}")

    probs = grade_probabilities(logits, grade_token_ids)
    values = torch.as_tensor(list(grade_values), dtype=probs.dtype, device=probs.device)
    return (probs * values).sum(dim=-1)


def grade_probabilities(
    logits,
    grade_token_ids: Sequence[int],
):
    """Return ``softmax(logits[grade_token_ids])`` over grade-token logits.

    This exposes the probability vector already used internally by
    :func:`expected_grade`, preserving any leading dimensions of ``logits``.
    """
    import torch

    if len(grade_token_ids) == 0:
        raise ValueError("grade_token_ids must not be empty")
    if not isinstance(logits, torch.Tensor):
        raise TypeError(f"logits must be a torch.Tensor, got {type(logits).__name__}")
    if logits.dim() < 1:
        raise ValueError(f"logits must have at least one dimension, got shape {tuple(logits.shape)}")

    ids = torch.as_tensor(list(grade_token_ids), dtype=torch.long, device=logits.device)
    grade_logits = logits.index_select(-1, ids)
    return torch.nn.functional.softmax(grade_logits, dim=-1)


def resolve_grade_token_ids(
    tokenizer,
    *,
    grade_strings: Sequence[str] = _DEFAULT_GRADE_STRINGS,
    with_space_prefix: bool = False,
) -> list[int]:
    """Resolve a list of token IDs for each grade digit.

    Each grade string must encode to exactly one token; otherwise the readout
    cannot extract a single logit for that grade. Raises ``ValueError`` rather
    than silently using only the first sub-token.

    Parameters
    ----------
    tokenizer : transformers.PreTrainedTokenizerBase
        The teacher's tokenizer. Same instance the wrapper uses for the
        batched-PW grade prompt.
    grade_strings : Sequence[str], default ``("0", "1", "2", "3")``
        Grade strings to encode.
    with_space_prefix : bool, default ``False``
        When True, encode each grade with a leading space (``" 0"``) — many
        BPE tokenizers split bare digits and digits-with-space-prefix into
        different vocab entries; choose the one that matches the prompt
        layout's actual whitespace context.

    Returns:
    -------
    list[int]
        Token IDs in the same order as ``grade_strings``.

    Raises:
    ------
    ValueError
        If any grade string does not encode to exactly one token.
    """
    out: list[int] = []
    for g in grade_strings:
        s = (" " + g) if with_space_prefix else g
        # tokenizer.__call__ prepends the SentencePiece leading-space token on
        # some tokenizers (e.g. Zephyr / LLaMA family), producing 2 tokens for
        # bare single-digit strings. tokenizer.encode() does not exhibit this
        # artefact, so we prefer it and fall back to __call__ only when encode
        # is unavailable (non-standard tokenizer). Both paths require the result
        # to be exactly one token for the slot-logit readout to be well-defined.
        if hasattr(tokenizer, "encode"):
            ids = tokenizer.encode(s, add_special_tokens=False)
        else:
            ids = tokenizer(s, add_special_tokens=False)["input_ids"]
        if not isinstance(ids, list):
            ids = list(ids)
        if len(ids) != 1:
            raise ValueError(
                f"Grade string {s!r} tokenises to {len(ids)} tokens "
                f"(ids={ids}); continuous readout requires exactly one token "
                "per grade. Try with_space_prefix=True or check the tokenizer."
            )
        out.append(int(ids[0]))
    if len(set(out)) != len(out):
        raise ValueError(
            f"Resolved duplicate token ids for grades {list(grade_strings)}: {out}. "
            "The tokenizer maps multiple grade digits to the same id; readout would be ambiguous."
        )
    return out
