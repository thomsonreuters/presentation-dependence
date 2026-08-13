"""Frozen reader generation backends for the QA reader bridge.

Generate answers from a pinned prompt with greedy, temperature-zero decoding.
Fixed decoding isolates changes caused by the input context.

``VLLMReaderEngine`` is the production backend. It loads a frozen instruction
model and follows the construction rules in ``self_distill.engines.vllm``,
including the spawn guard, rope-scaling normalization, and tf5 compatibility.
Using a model family different from the scorer supports clean attribution.

``EchoReaderEngine`` is a deterministic stub for offline tests. It delegates to
a caller-supplied function over the chat messages.

Both consume chat messages (``[{"role": ..., "content": ...}]``) built by
``reader.prompt.build_reader_messages`` and return raw generated strings;
answer extraction is done by ``reader.prompt.extract_answer``.
"""

from __future__ import annotations

import abc
from typing import Any, Callable, Sequence

Messages = Sequence[dict[str, str]]


class ReaderEngine(abc.ABC):
    """Generate a raw answer string per chat-message conversation."""

    name: str = "reader"

    @abc.abstractmethod
    def generate_batch(self, conversations: Sequence[Messages]) -> list[str]:
        """Return one generated string per conversation, in order."""

    def close(self) -> None:  # pragma: no cover - optional cleanup hook
        return None


class EchoReaderEngine(ReaderEngine):
    """Deterministic, dependency-free reader for offline smoke and tests.

    ``responder`` maps a conversation (list of messages) to an answer string. The
    default echoes the user message's first passage line, which is enough to make
    the pipeline plumbing testable without a GPU.
    """

    def __init__(self, responder: Callable[[Messages], str] | None = None, *, name: str = "echo-reader"):
        self.name = name
        self._responder = responder or self._default_responder

    @staticmethod
    def _default_responder(conversation: Messages) -> str:
        user = next((m["content"] for m in reversed(list(conversation)) if m.get("role") == "user"), "")
        for line in user.splitlines():
            line = line.strip()
            if line.startswith("[1]"):
                return line[3:].strip().split("\n", 1)[0]
        return ""

    def generate_batch(self, conversations: Sequence[Messages]) -> list[str]:
        return [self._responder(conv) for conv in conversations]


class VLLMReaderEngine(ReaderEngine):
    """Frozen instruct LLM reader via vLLM greedy generation.

    Parameters mirror the scorer-side vLLM engine. Decoding is pinned to
    ``temperature=0`` (greedy) with a fixed ``max_tokens`` so the only thing that
    varies across reader runs is the input context.
    """

    def __init__(
        self,
        *,
        model_name: str,
        max_model_len: int = 8192,
        max_tokens: int = 64,
        dtype: str = "auto",
        gpu_memory_utilization: float = 0.9,
        enable_prefix_caching: bool = True,
        tensor_parallel_size: int = 1,
        enable_thinking: bool | None = None,
        choices: Sequence[str] | None = None,
        extra_llm_kwargs: dict[str, Any] | None = None,
    ):
        # Use the scorer engine's spawn guard, rope-scaling normalization, and
        # tf5 compatibility helpers so both vLLM paths share one implementation.
        from presentation_dependence.self_distill.engines.vllm import (
            _ensure_vllm_multiproc_spawn,
            _normalise_rope_scaling_for_vllm,
            _raise_if_unsupported_gemma4_vllm,
            _resolve_rope_scaling_override,
            _supports_rope_scaling_constructor_arg,
        )

        _ensure_vllm_multiproc_spawn()
        try:
            import vllm
            from transformers import AutoTokenizer
            from vllm import LLM, SamplingParams
            from vllm.sampling_params import StructuredOutputsParams
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise ImportError(
                "VLLMReaderEngine requires 'vllm' and 'transformers'. "
                "Run `uv sync --extra vllm`. For offline plumbing use EchoReaderEngine."
            ) from exc

        self.model_name = str(model_name)
        self.name = self.model_name
        _raise_if_unsupported_gemma4_vllm(self.model_name, vllm)
        self.max_tokens = int(max_tokens)
        self.enable_thinking = enable_thinking
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        llm_kwargs: dict[str, Any] = {
            "model": self.model_name,
            "max_model_len": int(max_model_len),
            "dtype": dtype,
            "gpu_memory_utilization": gpu_memory_utilization,
            "enable_prefix_caching": enable_prefix_caching,
            "tensor_parallel_size": int(tensor_parallel_size),
        }
        rope_scaling = (
            _resolve_rope_scaling_override(self.model_name) if _supports_rope_scaling_constructor_arg(vllm) else None
        )
        if rope_scaling is not None:
            llm_kwargs.setdefault("rope_scaling", rope_scaling)
        llm_kwargs.setdefault("hf_overrides", _normalise_rope_scaling_for_vllm)
        if extra_llm_kwargs:
            llm_kwargs.update(extra_llm_kwargs)

        self.llm = LLM(**llm_kwargs)
        # Greedy decoding with a fixed limit produces one deterministic response
        # per conversation.
        sampling_kwargs: dict[str, Any] = {
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
        }
        if choices:
            sampling_kwargs["structured_outputs"] = StructuredOutputsParams(choice=[str(choice) for choice in choices])
        self.sampling_params = SamplingParams(**sampling_kwargs)

    def _render_prompt(self, conversation: Messages) -> str:
        kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
        if self.enable_thinking is not None:
            # Qwen3 honours enable_thinking; other templates ignore unknown kwargs.
            kwargs["enable_thinking"] = bool(self.enable_thinking)
        try:
            return self.tokenizer.apply_chat_template(list(conversation), **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking", None)
            return self.tokenizer.apply_chat_template(list(conversation), **kwargs)

    def generate_batch(self, conversations: Sequence[Messages]) -> list[str]:
        prompts = [self._render_prompt(conv) for conv in conversations]
        outputs = self.llm.generate(prompts, self.sampling_params)
        return [out.outputs[0].text if out.outputs else "" for out in outputs]
