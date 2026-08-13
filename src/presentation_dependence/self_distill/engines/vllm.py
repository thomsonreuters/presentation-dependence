"""vLLM backend for the logit-skeleton engine.

The readout requires the four grade-token log probabilities at each slot. vLLM
``prompt_logprobs`` reports probabilities for tokens inside the prompt. For
next-token probabilities after a prefix, this engine uses
``SamplingParams(max_tokens=1, logprobs=N)``. Both paths apply the same
expected-grade calculation to the four grade-token log probabilities.

Prefix-cache behavior
---------------------
Within one teacher chunk, the B=20 prefixes share ~99% of their tokens
(system prompt + instruction + B documents + assistant prefix + slot
header), differing only in the slot tail (``[i] Grade: ``). vLLM with
``enable_prefix_caching=True`` auto-detects this and computes the body
once per chunk. Combined with FlashAttention 2's varlen kernel, a chunk
that costs ``B × L²`` tokens-of-attention on HF runs in roughly
``L² + B × ΔL²`` on vLLM, where ΔL is the small slot suffix. Empirically
this is a 3-5× per-chunk speedup for this prompt shape.

vLLM is opt-in via the ``[vllm]`` extra. Importing this module does not require
vLLM; only constructing
:class:`VLLMLogitSkeletonEngine` does. The HF stack remains usable on
machines without ``vllm``.
"""

from __future__ import annotations

import math
import os
import re
from typing import Any, Sequence

from presentation_dependence.self_distill.engines._tf5_compat import ensure_tf5_vllm_compat
from presentation_dependence.self_distill.engines.base import LogitSkeletonEngine


# Apply the transformers-5 → vllm-0.10 shim at module import time, before any
# vllm code path can touch a tokenizer. See ``_tf5_compat.py`` for context.
ensure_tf5_vllm_compat()


_DEFAULT_TOP_LOGPROBS = 50
_MISSING_GRADE_LOGPROB = -1e9


def _normalise_rope_scaling_for_vllm(config: Any) -> Any:
    """Patch HF-style RoPE config aliases to what vLLM 0.10 expects.

    Some newer checkpoints (observed: ``google/gemma-4-E4B-it``) ship
    ``rope_scaling`` with a ``type`` key but no ``rope_type`` key. vLLM 0.10's
    ``ModelConfig`` validator requires ``rope_type`` and fails during
    construction before the model can load. Transformers accepts both spellings;
    copying ``type`` to ``rope_type`` preserves the checkpoint semantics while
    keeping the patch local to vLLM construction.
    """
    rope_scaling = getattr(config, "rope_scaling", None)
    if isinstance(rope_scaling, dict) and "rope_type" not in rope_scaling and "type" in rope_scaling:
        rope_scaling["rope_type"] = rope_scaling["type"]
    return config


def _resolve_rope_scaling_override(model_name: str) -> dict[str, Any] | None:
    """Return a vLLM-compatible ``rope_scaling`` dict for checkpoints needing it."""
    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_name)
    except Exception:
        return None

    rope_scaling = getattr(config, "rope_scaling", None)
    if not isinstance(rope_scaling, dict):
        return None

    out = dict(rope_scaling)
    if "rope_type" not in out and "type" in out:
        out["rope_type"] = out["type"]
    return out


def _version_tuple(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", version)[:3])


def _is_gemma4_model_name(model_name: str) -> bool:
    lowered = model_name.lower()
    return "gemma-4" in lowered or "gemma4" in lowered


def _raise_if_unsupported_gemma4_vllm(model_name: str, vllm_module: Any) -> None:
    """Fail loud for Gemma-4 on vLLM builds that predate Gemma-4 support.

    The retained Qwen-compatible container stack pins vLLM 0.10.2, which validates only the older
    ``rope_scaling`` schema. Gemma-4 checkpoints use Transformers-v5
    ``rope_parameters`` with per-layer-type settings; shoehorning this into a
    flat ``rope_scaling`` dict would be semantically wrong. Use HF readout or
    a vLLM>=0.19 container with native Gemma-4 support.
    """
    version = str(getattr(vllm_module, "__version__", "0"))
    if _is_gemma4_model_name(model_name) and _version_tuple(version) < (0, 19, 0):
        raise RuntimeError(
            f"Gemma-4 vLLM inference is not supported by vLLM {version}. "
            "Gemma-4 uses Transformers-v5 rope_parameters; vLLM <0.19 expects "
            "the older rope_scaling schema and fails during ModelConfig validation. "
            "Use reranker.inference_engine='hf' for Gemma-4, or upgrade the "
            "container to vLLM>=0.19 with native Gemma-4 support before retrying."
        )


def _supports_task_constructor_arg(vllm_module: Any) -> bool:
    """Return whether this vLLM version accepts ``LLM(..., task=...)``.

    vLLM 0.10 needs ``task="generate"`` for Qwen3-Reranker metadata; the
    0.19 API rejects the argument and relies on newer model loading semantics.
    """
    version = str(getattr(vllm_module, "__version__", "0"))
    return _version_tuple(version) < (0, 19, 0)


def _supports_rope_scaling_constructor_arg(vllm_module: Any) -> bool:
    """Return whether this vLLM version accepts ``LLM(..., rope_scaling=...)``."""
    version = str(getattr(vllm_module, "__version__", "0"))
    return _version_tuple(version) < (0, 19, 0)


def _ensure_vllm_multiproc_spawn() -> None:
    """Force vLLM's worker multiprocessing to use ``spawn`` instead of ``fork``.

    Symptom this fixes: in any container where the PyTorch build
    pre-imports torch + touches CUDA before our code runs), vLLM's v1
    EngineCore subprocess raises ::

        RuntimeError: Cannot re-initialize CUDA in forked subprocess.
        To use CUDA with multiprocessing, you must use the 'spawn' start method.

    Setting ``VLLM_WORKER_MULTIPROC_METHOD=spawn`` BEFORE ``import vllm``
    makes vLLM honour the spawn method when it boots its engine workers.
    Idempotent — uses ``setdefault`` so an explicit ``execution.environment`` or
    operator-set value still wins.

    Reference: vLLM #5639, vLLM docs "Distributed Serving" section. Reproduced
    on a single-GPU PyTorch 2.5 image during a parity smoke.
    """
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


class VLLMLogitSkeletonEngine(LogitSkeletonEngine):
    """Score prefix-tail next-token logits via vLLM ``LLM.generate``.

    Parameters
    ----------
    model_name : str
        HuggingFace model id (e.g. ``"Qwen/Qwen3-Reranker-4B"``).
    max_model_len : int, default ``4096``
        Upper bound on prompt length seen by vLLM. Set to match the
        reranker's ``max_length``. Smaller values give vLLM's scheduler
        more concurrent-request memory headroom.
    dtype : str, default ``"auto"``
        vLLM dtype string (``"auto"``, ``"bfloat16"``, ``"float16"``).
        ``"auto"`` resolves to bf16 on Ampere+, fp16 elsewhere.
    gpu_memory_utilization : float, default ``0.9``
        Fraction of GPU memory vLLM may use. Lower if running alongside
        another process; higher if vLLM is the only GPU consumer.
    enable_prefix_caching : bool, default ``True``
        Detect shared prompt prefixes and reuse their KV cache. B=20 requests
        share a long body, so prefix caching supplies most of the measured
        speedup. Disable it only for debugging.
    tensor_parallel_size : int, default ``1``
        Number of GPUs used to shard the model. The tracked public configs keep
        this at 1; TP>1 needs a separately reviewed compatibility profile.
    top_logprobs : int, default ``50``
        How many top-token logprobs vLLM should return at the generated
        position. Must be large enough to safely contain all four grade
        token IDs in the worst case. The grade digits are normally in the
        top 20 for a prompted grade task; 50 provides additional margin.
    extra_llm_kwargs : dict | None, default ``None``
        Forwarded verbatim to ``vllm.LLM(...)``. Use for vLLM-specific
        flags not covered by the named parameters above.
    """

    def __init__(  # noqa: C901
        self,
        *,
        model_name: str,
        max_model_len: int = 4096,
        dtype: str = "auto",
        gpu_memory_utilization: float = 0.9,
        enable_prefix_caching: bool = True,
        tensor_parallel_size: int = 1,
        top_logprobs: int = _DEFAULT_TOP_LOGPROBS,
        tokenizer: Any = None,
        max_length: int | None = None,
        extra_llm_kwargs: dict[str, Any] | None = None,
        lora_path: str | None = None,
        lora_adapters: Sequence[str] | None = None,
        max_lora_rank: int = 16,
    ):
        # Set VLLM_WORKER_MULTIPROC_METHOD=spawn before importing vllm so the
        # engine subprocess avoids the fork-with-CUDA crash in a container.
        # Must run before ``import vllm`` to take effect.
        _ensure_vllm_multiproc_spawn()
        try:
            import vllm
            from vllm import LLM, SamplingParams
            from vllm.inputs import TokensPrompt
        except ImportError as e:
            raise ImportError(
                "vLLM backend requested but the 'vllm' package is not installed. "
                "Run `uv sync --extra vllm` to install. "
                "If you only need the HuggingFace path, set "
                "`reranker.inference_engine: hf` in your config."
            ) from e
        self.model_name = str(model_name)
        _raise_if_unsupported_gemma4_vllm(self.model_name, vllm)
        self._TokensPrompt = TokensPrompt
        self.tokenizer = tokenizer
        self.max_length = int(max_length) if max_length is not None else int(max_model_len)
        if self.max_length <= 0:
            raise ValueError(f"max_length must be positive, got {self.max_length}")
        self.max_lora_rank = int(max_lora_rank)
        # Multi-adapter support (shared-base bundles): one base engine can serve
        # several LoRA adapters + the base itself, swapping per ``generate`` call
        # via distinct ``LoRARequest``s. ``lora_adapters`` (a list of adapter
        # paths/dirs) takes precedence; ``lora_path`` is the legacy single-adapter
        # form (kept bit-for-bit: when only it is given, that adapter is the
        # default active request, matching the original single-adapter behaviour).
        # Adapters are keyed by their resolved path so callers select by path
        # (``set_active_adapter(path)``); ``None`` selects the base (no adapter).
        adapter_paths: list[str] = []
        if lora_adapters:
            adapter_paths = [str(p) for p in lora_adapters if p]
        elif lora_path:
            adapter_paths = [str(lora_path)]
        # De-dup while preserving order (distinct mounts only).
        seen: set[str] = set()
        adapter_paths = [p for p in adapter_paths if not (p in seen or seen.add(p))]
        self.lora_path = str(lora_path) if lora_path else None
        self._lora_requests: dict[str, Any] = {}
        if adapter_paths:
            from vllm.lora.request import LoRARequest

            for i, path in enumerate(adapter_paths, start=1):
                self._lora_requests[path] = LoRARequest(
                    lora_name=f"reranker-adapter-{i}",
                    lora_int_id=i,
                    lora_path=path,
                )
        # Active selection. Legacy single-adapter (``lora_path`` only): default to
        # that adapter so existing single-runs are unchanged. Multi-adapter
        # (``lora_adapters``): default to the base (None); the bundle driver sets
        # the active adapter per surface before scoring.
        self._active_lora_request: Any = None
        if lora_adapters is None and self.lora_path is not None:
            self._active_lora_request = self._lora_requests[self.lora_path]

        self.max_model_len = int(max_model_len)
        self.top_logprobs = int(top_logprobs)
        if self.top_logprobs < 4:
            raise ValueError(f"top_logprobs must be >= 4 to cover all grade tokens, got {self.top_logprobs}")

        llm_kwargs: dict[str, Any] = {
            "model": self.model_name,
            "max_model_len": self.max_model_len,
            "dtype": dtype,
            "gpu_memory_utilization": gpu_memory_utilization,
            "enable_prefix_caching": enable_prefix_caching,
            "tensor_parallel_size": int(tensor_parallel_size),
            # vLLM enforces ``SamplingParams.logprobs <= LLM.max_logprobs``,
            # defaulting to 20 since v0.10. Our default ``top_logprobs=50``
            # is chosen to safely contain all four grade tokens even when
            # the model puts unrelated tokens at top-1..top-4. Pass through
            # here so the LLM accepts our SamplingParams, as confirmed by the
            # parity smoke.
            "max_logprobs": self.top_logprobs,
            # Force generative task. Qwen3-Reranker-4B and related reranker
            # models ship sentence-transformers/text-ranking metadata; if a
            # future vLLM release auto-selects a pooling task, next-token
            # logprobs would no longer come from the LM head. Keep the task
            # explicit on vLLM versions whose constructor supports it because
            # this engine's contract is causal-LM logits. vLLM 0.19 removed
            # this LLM(...) kwarg, so below we only set it when accepted.
        }
        if _supports_task_constructor_arg(vllm):
            llm_kwargs["task"] = "generate"
        if self._lora_requests:
            # vLLM gates LoRA at engine init. ``max_lora_rank`` MUST be >= the
            # LoRA r in adapter_config.json (we default to 16 to match the
            # self-distill student configs). Setting it lower → vLLM raises
            # at request time. ``max_loras`` = the number of distinct adapters
            # this engine may serve concurrently (1 for a single-adapter run, N
            # for a shared-base bundle); each adds only a small per-rank LoRA
            # cache on top of the one base load.
            llm_kwargs["enable_lora"] = True
            llm_kwargs["max_lora_rank"] = self.max_lora_rank
            llm_kwargs["max_loras"] = len(self._lora_requests)
        if extra_llm_kwargs:
            llm_kwargs.update(extra_llm_kwargs)

        rope_scaling = (
            _resolve_rope_scaling_override(self.model_name) if _supports_rope_scaling_constructor_arg(vllm) else None
        )
        if rope_scaling is not None:
            llm_kwargs.setdefault("rope_scaling", rope_scaling)
        llm_kwargs.setdefault("hf_overrides", _normalise_rope_scaling_for_vllm)

        self.llm = LLM(**llm_kwargs)
        # max_tokens=1 + temperature=0 makes vLLM run a single deterministic
        # forward and exposes the grade-token logprobs for the expected-grade
        # readout.
        self.sampling_params = SamplingParams(
            max_tokens=1,
            temperature=0.0,
            logprobs=self.top_logprobs,
        )
        self._n_missing_grade_tokens = 0  # surfaced in close() / tests

    @property
    def adapter_paths(self) -> list[str]:
        """Paths of the LoRA adapters this engine can serve (empty if base-only)."""
        return list(self._lora_requests.keys())

    def set_active_adapter(self, adapter_path: str | None) -> None:
        """Select which loaded adapter the next ``generate`` calls use.

        ``None`` selects the base model (no adapter). A non-None path MUST be one
        of the adapters this engine was built with (``adapter_paths``); selecting
        an unknown adapter is a hard error rather than a silent base fallback.
        Used by the shared-base bundle driver to swap adapters between surfaces.
        """
        if adapter_path is None:
            self._active_lora_request = None
            return
        path = str(adapter_path)
        req = self._lora_requests.get(path)
        if req is None:
            raise ValueError(
                f"set_active_adapter: {path!r} is not a loaded adapter; "
                f"engine was built with {sorted(self._lora_requests)}. "
                f"Pass it via lora_adapters at engine construction."
            )
        self._active_lora_request = req

    def score_prefixes(
        self,
        prefixes: Sequence[str],
        grade_token_ids: Sequence[int],
        grade_values: Sequence[float] = (0.0, 1.0, 2.0, 3.0),
    ) -> list[float]:
        if not prefixes:
            return []
        if len(grade_token_ids) != len(grade_values):
            raise ValueError(
                f"grade_token_ids ({len(grade_token_ids)}) and grade_values "
                f"({len(grade_values)}) must have the same length"
            )

        # Pre-tokenize with ``add_special_tokens=False`` and pass token IDs via
        # ``TokensPrompt``. vLLM's text-prompt path
        # tokenizes with ``add_special_tokens=True`` by default, which silently
        # prepends a BOS token to every prefix and shifts the model's view of
        # the input. Pre-tokenization with the same options
        # as ``HFLogitSkeletonEngine`` (``add_special_tokens=False``,
        # left-truncation to ``max_length``) guarantees both backends see the
        # identical token sequence.
        # vLLM's ``max_model_len`` includes the generated token. We generate
        # exactly one token to expose next-token logprobs, so the prompt itself
        # must be at most ``max_model_len - 1``. Without this headroom, a prompt
        # truncated to exactly 4096 with ``SamplingParams(max_tokens=1)`` fails
        # at runtime with ``4097 > max_model_len``. Observed first on qid=733756
        # during the 30K local-DP production run.
        prompt_max_length = min(self.max_length, self.max_model_len - 1)
        if prompt_max_length <= 0:
            raise ValueError(f"vLLM max_model_len={self.max_model_len} leaves no room for max_tokens=1")

        prev_trunc = getattr(self.tokenizer, "truncation_side", None)
        if prev_trunc is not None:
            self.tokenizer.truncation_side = "left"
        try:
            tokenized_prompts = [
                self._TokensPrompt(
                    prompt_token_ids=self.tokenizer(
                        prefix,
                        add_special_tokens=False,
                        truncation=True,
                        max_length=prompt_max_length,
                    )["input_ids"]
                )
                for prefix in prefixes
            ]
        finally:
            if prev_trunc is not None:
                self.tokenizer.truncation_side = prev_trunc

        generate_kwargs: dict[str, Any] = {"use_tqdm": False}
        if self._active_lora_request is not None:
            generate_kwargs["lora_request"] = self._active_lora_request
        outputs = self.llm.generate(tokenized_prompts, self.sampling_params, **generate_kwargs)
        if len(outputs) != len(prefixes):
            raise RuntimeError(
                f"vLLM returned {len(outputs)} outputs for {len(prefixes)} prefixes; request/result count mismatch."
            )

        results: list[float] = []
        grade_ids = [int(g) for g in grade_token_ids]
        grade_vals = [float(v) for v in grade_values]
        for out in outputs:
            logprobs_dict = self._extract_first_generated_logprobs(out)
            grade_logprobs = self._lookup_grade_logprobs(logprobs_dict, grade_ids)
            results.append(_softmax_expected_value(grade_logprobs, grade_vals))
        return results

    def score_prefixes_argmax_grade(
        self,
        prefixes: Sequence[str],
        grade_token_ids: Sequence[int],
        grade_values: Sequence[float] = (0.0, 1.0, 2.0, 3.0),
    ) -> list[float]:
        """Path A readout: argmax over the four grade-token logprobs at each tail.

        Same vLLM forward + ``logprobs`` extraction as :meth:`score_prefixes`;
        the only difference is the readout: pick the integer grade with the
        highest probability instead of taking the soft expected value.
        """
        if not prefixes:
            return []
        if len(grade_token_ids) != len(grade_values):
            raise ValueError(
                f"grade_token_ids ({len(grade_token_ids)}) and grade_values "
                f"({len(grade_values)}) must have the same length"
            )

        # Re-tokenize and dispatch (same as score_prefixes; we reuse the same
        # SamplingParams since they only differ in how we *read* the logprobs).
        prompt_max_length = min(self.max_length, self.max_model_len - 1)
        if prompt_max_length <= 0:
            raise ValueError(f"vLLM max_model_len={self.max_model_len} leaves no room for max_tokens=1")

        prev_trunc = getattr(self.tokenizer, "truncation_side", None)
        if prev_trunc is not None:
            self.tokenizer.truncation_side = "left"
        try:
            tokenized_prompts = [
                self._TokensPrompt(
                    prompt_token_ids=self.tokenizer(
                        prefix,
                        add_special_tokens=False,
                        truncation=True,
                        max_length=prompt_max_length,
                    )["input_ids"]
                )
                for prefix in prefixes
            ]
        finally:
            if prev_trunc is not None:
                self.tokenizer.truncation_side = prev_trunc

        generate_kwargs: dict[str, Any] = {"use_tqdm": False}
        if self._active_lora_request is not None:
            generate_kwargs["lora_request"] = self._active_lora_request
        outputs = self.llm.generate(tokenized_prompts, self.sampling_params, **generate_kwargs)
        if len(outputs) != len(prefixes):
            raise RuntimeError(
                f"vLLM returned {len(outputs)} outputs for {len(prefixes)} prefixes; request/result count mismatch."
            )

        grade_ids = [int(g) for g in grade_token_ids]
        grade_vals = [float(v) for v in grade_values]
        results: list[float] = []
        for out in outputs:
            logprobs_dict = self._extract_first_generated_logprobs(out)
            grade_logprobs = self._lookup_grade_logprobs(logprobs_dict, grade_ids)
            best_idx = max(range(len(grade_logprobs)), key=lambda i: grade_logprobs[i])
            results.append(grade_vals[best_idx])
        return results

    def score_prefix_probabilities(
        self,
        prefixes: Sequence[str],
        grade_token_ids: Sequence[int],
    ) -> list[list[float]]:
        """Return per-prefix grade probability vectors from vLLM logprobs."""
        if not prefixes:
            return []

        prompt_max_length = min(self.max_length, self.max_model_len - 1)
        if prompt_max_length <= 0:
            raise ValueError(f"vLLM max_model_len={self.max_model_len} leaves no room for max_tokens=1")

        prev_trunc = getattr(self.tokenizer, "truncation_side", None)
        if prev_trunc is not None:
            self.tokenizer.truncation_side = "left"
        try:
            tokenized_prompts = [
                self._TokensPrompt(
                    prompt_token_ids=self.tokenizer(
                        prefix,
                        add_special_tokens=False,
                        truncation=True,
                        max_length=prompt_max_length,
                    )["input_ids"]
                )
                for prefix in prefixes
            ]
        finally:
            if prev_trunc is not None:
                self.tokenizer.truncation_side = prev_trunc

        generate_kwargs: dict[str, Any] = {"use_tqdm": False}
        if self._active_lora_request is not None:
            generate_kwargs["lora_request"] = self._active_lora_request
        outputs = self.llm.generate(tokenized_prompts, self.sampling_params, **generate_kwargs)
        if len(outputs) != len(prefixes):
            raise RuntimeError(
                f"vLLM returned {len(outputs)} outputs for {len(prefixes)} prefixes; request/result count mismatch."
            )

        grade_ids = [int(g) for g in grade_token_ids]
        results: list[list[float]] = []
        for out in outputs:
            logprobs_dict = self._extract_first_generated_logprobs(out)
            grade_logprobs = self._lookup_grade_logprobs(logprobs_dict, grade_ids)
            results.append(_softmax_probabilities(grade_logprobs))
        return results

    def score_skeleton_probabilities(
        self,
        full_text: str,
        slot_prefixes: Sequence[str],
        grade_token_ids: Sequence[int],
    ) -> list[list[float]]:
        """Single-forward multi-slot readout via vLLM ``prompt_logprobs``.

        Submits the full grade skeleton as *one* request and reads the
        prompt-position logprobs at each slot's dummy-grade token, instead of
        dispatching B nested-prefix sequences. Bit-identical to
        :meth:`score_prefix_probabilities` (causal-mask equivalence) when the
        chunk fits in ``max_model_len``.

        Note: this uses ``prompt_logprobs`` (logprobs of tokens *in* the prompt)
        because that is the interior-position read this method needs. The
        per-prefix :meth:`score_prefixes` path still uses ``max_tokens=1`` +
        ``logprobs`` (next-token after the prefix); the two are
        different-but-equivalent reads of the same distribution.
        """
        if not slot_prefixes:
            return []
        if self.tokenizer is None:
            raise RuntimeError("score_skeleton_probabilities requires a tokenizer on the engine")
        from vllm import SamplingParams

        from presentation_dependence.self_distill.engines.hf import compute_skeleton_slot_positions

        prompt_max_length = min(self.max_length, self.max_model_len - 1)
        if prompt_max_length <= 0:
            raise ValueError(f"vLLM max_model_len={self.max_model_len} leaves no room for max_tokens=1")

        prev_trunc = getattr(self.tokenizer, "truncation_side", None)
        if prev_trunc is not None:
            self.tokenizer.truncation_side = "left"
        try:
            input_ids, positions = compute_skeleton_slot_positions(
                self.tokenizer, full_text, list(slot_prefixes), prompt_max_length
            )
        finally:
            if prev_trunc is not None:
                self.tokenizer.truncation_side = prev_trunc

        sampling = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=self.top_logprobs)
        generate_kwargs: dict[str, Any] = {"use_tqdm": False}
        if self._active_lora_request is not None:
            generate_kwargs["lora_request"] = self._active_lora_request
        outputs = self.llm.generate([self._TokensPrompt(prompt_token_ids=input_ids)], sampling, **generate_kwargs)
        if len(outputs) != 1:
            raise RuntimeError(f"vLLM returned {len(outputs)} outputs for a single skeleton prompt")

        prompt_lps = self._extract_prompt_logprobs(outputs[0])
        grade_ids = [int(g) for g in grade_token_ids]
        seq_len = len(input_ids)
        results: list[list[float]] = []
        for pos in positions:
            read = pos + 1  # the slot's grade token follows its prefix's last token
            if read >= seq_len or read >= len(prompt_lps) or prompt_lps[read] is None:
                self._n_missing_grade_tokens += len(grade_ids)
                results.append([1.0 / len(grade_ids)] * len(grade_ids))
                continue
            grade_logprobs = self._lookup_grade_logprobs(prompt_lps[read], grade_ids)
            results.append(_softmax_probabilities(grade_logprobs))
        return results

    # ------------------------------------------------------------------
    # vLLM RequestOutput parsing: defensive against minor API drift
    # across vLLM versions (0.6+).
    # ------------------------------------------------------------------

    def _extract_prompt_logprobs(self, request_output: Any) -> list[dict[int, float] | None]:
        """Pull per-prompt-position ``{token_id: logprob}`` dicts (or None)."""
        plps = getattr(request_output, "prompt_logprobs", None)
        if plps is None:
            raise RuntimeError(
                "vLLM RequestOutput has no `prompt_logprobs`; was "
                "SamplingParams.prompt_logprobs set on the skeleton request?"
            )
        out: list[dict[int, float] | None] = []
        for entry in plps:
            if entry is None:
                out.append(None)
                continue
            d: dict[int, float] = {}
            for tok_id, lp in entry.items():
                d[int(tok_id)] = float(lp.logprob) if hasattr(lp, "logprob") else float(lp)
            out.append(d)
        return out

    def _extract_first_generated_logprobs(self, request_output: Any) -> dict[int, float]:
        """Pull the dict ``{token_id: logprob}`` for the single generated token."""
        outs = getattr(request_output, "outputs", None)
        if not outs:
            raise RuntimeError("vLLM RequestOutput has no `outputs`; can't extract logprobs.")
        completion = outs[0]
        logprobs_seq = getattr(completion, "logprobs", None)
        if not logprobs_seq:
            raise RuntimeError(
                "vLLM CompletionOutput has empty `logprobs`. Did SamplingParams.logprobs "
                "get overridden to None somewhere?"
            )
        first = logprobs_seq[0]
        if first is None:
            raise RuntimeError("vLLM returned None for the first generated-token logprobs.")
        # vLLM 0.6+ returns dict[int, Logprob] where Logprob has `.logprob`.
        # Older versions returned dict[int, float]. Handle both.
        out: dict[int, float] = {}
        for tok_id, lp in first.items():
            if hasattr(lp, "logprob"):
                out[int(tok_id)] = float(lp.logprob)
            else:
                out[int(tok_id)] = float(lp)
        return out

    def _lookup_grade_logprobs(
        self,
        logprobs_dict: dict[int, float],
        grade_token_ids: Sequence[int],
    ) -> list[float]:
        """Read the four grade-token logprobs from a top-N response.

        If a grade token is absent from the top-N (rare for a prompted
        grade task), we substitute a very negative logprob so its softmax
        weight is effectively zero. We track misses so callers / tests can
        expose them through ``n_missing_grade_tokens``.
        """
        out: list[float] = []
        for gid in grade_token_ids:
            if gid in logprobs_dict:
                out.append(logprobs_dict[gid])
            else:
                self._n_missing_grade_tokens += 1
                out.append(_MISSING_GRADE_LOGPROB)
        return out

    @property
    def n_missing_grade_tokens(self) -> int:
        """Total number of (prefix, grade-token) lookups that fell outside
        ``top_logprobs``. Should stay 0 in practice for prompted grade
        tasks; nonzero values are a flag to bump ``top_logprobs`` or
        investigate the prompt.
        """
        return self._n_missing_grade_tokens


def _softmax_expected_value(logprobs: Sequence[float], values: Sequence[float]) -> float:
    """Numerically stable softmax over ``logprobs`` weighted by ``values``.

    Mirrors :func:`presentation_dependence.self_distill.readout.expected_grade` for the
    HF path, but operates on plain Python floats instead of a torch tensor.
    vLLM returns logprobs as scalars, so no tensor is needed.
    """
    if not logprobs:
        raise ValueError("logprobs must not be empty")
    probs = _softmax_probabilities(logprobs)
    return sum(p * v for p, v in zip(probs, values))


def _softmax_probabilities(logprobs: Sequence[float]) -> list[float]:
    """Numerically stable softmax over plain-Python logprobs."""
    if not logprobs:
        raise ValueError("logprobs must not be empty")
    m = max(logprobs)
    exps = [math.exp(lp - m) for lp in logprobs]
    z = sum(exps)
    if z <= 0:
        # All logprobs were -inf (every grade missed top-N); return uniform so
        # the readout is defined. ``n_missing_grade_tokens`` surfaces misses.
        return [1.0 / len(logprobs) for _ in logprobs]
    return [e / z for e in exps]
