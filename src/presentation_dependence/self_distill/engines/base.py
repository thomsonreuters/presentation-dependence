"""Abstract logit-skeleton engine for self-distillation scoring.

Self-distill teachers reduce continuous-grade scoring to the same primitive:
take a list of prompt prefixes (each ending right before a grade-digit slot)
and return the next-token expected value over the four grade tokens.
Concrete backends in ``hf.py`` and ``vllm.py`` implement that contract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence


class LogitSkeletonEngine(ABC):
    """Score next-token expected grade at each of a batch of prompt prefixes.

    Implementations decide how to dispatch the forward pass: HF batched
    forward over a padded tensor, vLLM ``generate`` with ``logprobs=N``, etc.
    Callers (reranker wrappers) stay engine-agnostic: build a list of prefix
    strings, pass them in, get one expected-grade scalar per prefix back.
    """

    @abstractmethod
    def score_prefixes(
        self,
        prefixes: Sequence[str],
        grade_token_ids: Sequence[int],
        grade_values: Sequence[float] = (0.0, 1.0, 2.0, 3.0),
    ) -> list[float]:
        """Return the next-token expected grade at each prefix's tail.

        Parameters
        ----------
        prefixes : Sequence[str]
            Full-text prompt prefixes. Each ends immediately before a slot's
            dummy grade digit (e.g. ``"...[1] Grade: "``). The implementation
            schedules a forward pass that exposes the next-token logits at
            the position right after the prefix.
        grade_token_ids : Sequence[int]
            Token IDs of the four grade digits, in the same order as
            ``grade_values``. Resolve once per tokenizer via
            :func:`presentation_dependence.self_distill.readout.resolve_grade_token_ids`.
        grade_values : Sequence[float], default ``(0.0, 1.0, 2.0, 3.0)``
            Numeric grade scale. Output values are in
            ``[min(grade_values), max(grade_values)]``.

        Returns:
        -------
        list[float]
            One expected-grade scalar per input prefix, in input order.
            Caller must apply any downstream scaling (e.g. dividing by
            ``max(grade_values)`` to fit the ``[0, 1]`` reranker contract).
        """
        raise NotImplementedError

    def score_prefixes_argmax_grade(
        self,
        prefixes: Sequence[str],
        grade_token_ids: Sequence[int],
        grade_values: Sequence[float] = (0.0, 1.0, 2.0, 3.0),
    ) -> list[float]:
        """Path A readout: return ``argmax_g P(g | prefix)`` mapped to its grade value.

        Same forward as :meth:`score_prefixes`, but the discrete grade with the
        highest next-token probability is emitted instead of the soft expected
        value. Used by prompt-based rerankers that want fixed-integer "Path A"
        scoring (decoded grade) as a non-continuous control on the continuous expected-grade readout.

        Default implementation raises ``NotImplementedError``; concrete backends
        implement it by reusing the same prefix-encoding / forward path as
        :meth:`score_prefixes` and reading the same per-grade logprob slice.
        """
        raise NotImplementedError

    def score_prefix_probabilities(
        self,
        prefixes: Sequence[str],
        grade_token_ids: Sequence[int],
    ) -> list[list[float]]:
        """Return next-token grade probability vectors at each prefix's tail.

        Each inner list is ``softmax(logits[grade_token_ids])`` in the same
        order as ``grade_token_ids``. Callers use this for vector-loss silver
        labels; scalar expected-grade scoring remains available through
        :meth:`score_prefixes`.
        """
        raise NotImplementedError

    def score_skeleton_probabilities(
        self,
        full_text: str,
        slot_prefixes: Sequence[str],
        grade_token_ids: Sequence[int],
    ) -> list[list[float]]:
        """Single-forward multi-slot readout: score all B slots in one pass.

        Given the full grade skeleton ``full_text`` (one sequence containing the
        shared B-doc body plus all B dummy-graded slots) and the per-slot string
        prefixes, return one grade-probability vector per slot, reading the B
        interior slot positions from a single forward instead of dispatching B
        nested-prefix sequences.

        Equivalent to :meth:`score_prefix_probabilities` on ``slot_prefixes``
        under causal masking when the chunk fits in the backend's max length.
        Optional; backends that can't read interior positions need not implement
        it (callers fall back to :meth:`score_prefix_probabilities`).
        """
        raise NotImplementedError

    def close(self) -> None:
        """Optional teardown hook. vLLM owns a GPU process and benefits from
        explicit cleanup; HF impl is a no-op.
        """
        return None
