"""Inference-engine backends for self-distillation logit-skeleton scoring.

Each engine receives prompt prefixes ending immediately before a grade slot and
returns the next-token expected value over ``{0, 1, 2, 3}``. Two backends
implement this contract:

- :class:`HFLogitSkeletonEngine`: HuggingFace ``transformers`` forward pass
  (default; works with the existing stack on any GPU/CPU/MPS).
- :class:`VLLMLogitSkeletonEngine`: vLLM ``LLM.generate`` with ``logprobs=N``,
  optimised for the within-chunk prefix-sharing pattern (B=20 prefixes that
  share ~99% of their tokens). Opt-in via ``[vllm]`` extra.

Reranker wrappers (:class:`Qwen3Reranker`, :class:`Qwen3InstructGradeReranker`)
build prompts and delegate per-prefix scoring to the configured engine.
"""

from presentation_dependence.self_distill.engines.base import LogitSkeletonEngine
from presentation_dependence.self_distill.engines.hf import HFLogitSkeletonEngine

__all__ = [
    "HFLogitSkeletonEngine",
    "LogitSkeletonEngine",
    "make_engine",
]


def make_engine(kind: str, **kwargs):
    """Construct a :class:`LogitSkeletonEngine` by name.

    Parameters
    ----------
    kind : str
        ``"hf"`` for HuggingFace transformers, or ``"vllm"`` for vLLM.
    **kwargs
        Engine-specific constructor arguments. See the relevant subclass.

    Notes:
    -----
    The vLLM path is imported lazily so the HF stack remains usable without
    the optional ``[vllm]`` extra installed.
    """
    if kind == "hf":
        return HFLogitSkeletonEngine(**kwargs)
    if kind == "vllm":
        from presentation_dependence.self_distill.engines.vllm import VLLMLogitSkeletonEngine

        return VLLMLogitSkeletonEngine(**kwargs)
    raise ValueError(f"Unknown inference engine kind: {kind!r}. Expected one of: 'hf', 'vllm'.")
