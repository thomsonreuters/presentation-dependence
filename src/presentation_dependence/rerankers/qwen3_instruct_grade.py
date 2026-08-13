"""Qwen3-4B-Instruct prompted as shared-context batched-PW grade scorer.

Model: ``Qwen/Qwen3-4B-Instruct-2507`` (Apache-2.0); the tracked student configs
use ``Qwen/Qwen3-4B`` (Apache-2.0) as the base. See
``THIRD_PARTY_NOTICES.md`` for model and dataset terms.

Used for K-shot BSC teacher inference in the configuration.

Main self-distillation base: expected-grade readout on the chat-tuned
Qwen3-Instruct model. ``Qwen3Reranker`` instead owns the reranker-specific
yes/no logit path, pairwise template, and ``<think>`` handling. Keeping the
chat-model wrapper separate prevents those contracts from being mixed through a
configuration switch. Add new model families as separate wrappers rather than
modifying ``qwen3.py``.

The K-shot BSC averaging in :mod:`presentation_dependence.self_distill.teacher` shuffles
the candidate set and calls :meth:`rank` K times; this wrapper's only job is
to emit B continuous expected-grade values per shared-context forward pass.


Output scale convention
-----------------------
``scores_init_order`` follows the standard reranker contract: per-doc
scalars in ``[0, 1]`` (expected-grade ``∈ [0, 3]`` divided by 3). The
self-distill teacher driver multiplies by 3 again before persisting
``score_continuous`` into ``silver_labels.jsonl``. The same
convention :class:`Qwen3Reranker._rank_self_consistency` uses for its
mean-grade aggregation.
"""

from __future__ import annotations

import time
from typing import Any

from presentation_dependence.rerankers._rank_result import scores_to_rank_result
from presentation_dependence.rerankers._torch_utils import resolve_device as _resolve_device
from presentation_dependence.rerankers._torch_utils import resolve_dtype as _resolve_dtype
from presentation_dependence.rerankers.base import (
    Passage,
    Query,
    RankResult,
    Reranker,
    apply_vllm_lora_settings,
    parse_lora_config,
)
from presentation_dependence.rerankers.grade_rubrics import (
    DEFAULT_GRADE_RUBRIC_ID as _DEFAULT_GRADE_RUBRIC_ID,
    DEFAULT_INSTRUCTION as _DEFAULT_INSTRUCTION,
    build_pathc_grade_user_body,
    resolve_grade_rubric as _resolve_grade_rubric,
)
from presentation_dependence.rerankers.render_variants import (
    RenderVariant,
    slot_labels_for_chunk as _slot_labels_for_chunk,
)
from presentation_dependence.rerankers.scale_variants import (
    GradeScaleSpec,
    build_scale_grade_user_body,
    scale_spec_for_render_variant,
)
from presentation_dependence.self_distill.engines import HFLogitSkeletonEngine, LogitSkeletonEngine, make_engine
from presentation_dependence.self_distill.readout import resolve_grade_token_ids as _resolve_grade_token_ids
from presentation_dependence.utils.setup_logging import setup_logging


def _build_qwen3_instruct_grade_user_body(
    instruction: str | None,
    query: str,
    passages: list[Passage],
    *,
    max_doc_chars: int,
    grade_rubric_id: str | None = None,
    render_variant: RenderVariant | None = None,
    slot_labels_override: tuple[str, ...] | None = None,
) -> str:
    """User-section body: instruction + query + numbered docs + grade rubric.

    ``slot_labels_override`` (Stage-2 training) sets the per-slot markers
    directly without constructing a :class:`RenderVariant`; it is only honored
    when ``render_variant is None`` (newline separator, default rubric).
    """
    slot_labels = None
    separator = None
    rubric_id = grade_rubric_id
    scale_spec = scale_spec_for_render_variant(render_variant)
    if scale_spec is not None and scale_spec.scheme != "grade_0_3":
        n = len(passages)
        slot_labels = _slot_labels_for_chunk(n, id_scheme="numeric", id_slot_permutation=None)
        return build_scale_grade_user_body(
            instruction,
            query,
            [str(passage["text"]) for passage in passages],
            max_doc_chars=max_doc_chars,
            scale_spec=scale_spec,
            slot_labels=slot_labels,
        )
    if render_variant is None and slot_labels_override is not None:
        n = len(passages)
        if len(slot_labels_override) != n:
            raise ValueError(f"slot_labels_override has {len(slot_labels_override)} entries but {n} passages")
        return build_pathc_grade_user_body(
            instruction,
            query,
            [str(passage["text"]) for passage in passages],
            max_doc_chars=max_doc_chars,
            grade_rubric_id=rubric_id,
            slot_labels=tuple(slot_labels_override),
            separator=None,
        )
    if render_variant is not None:
        n = len(passages)
        slot_labels = render_variant.slot_labels[:n]
        if len(slot_labels) != n:
            slot_labels = _slot_labels_for_chunk(
                n,
                id_scheme=render_variant.id_scheme,
                id_slot_permutation=render_variant.id_slot_permutation[:n]
                if render_variant.id_slot_permutation
                else None,
                rand3_seed=render_variant.rand3_seed,
            )
        separator = render_variant.separator
        rubric_id = render_variant.rubric_id
    return build_pathc_grade_user_body(
        instruction,
        query,
        [str(passage["text"]) for passage in passages],
        max_doc_chars=max_doc_chars,
        grade_rubric_id=rubric_id,
        slot_labels=slot_labels,
        separator=separator,
    )


def _build_qwen3_instruct_grade_skeleton(  # noqa: C901
    tokenizer,
    instruction: str | None,
    query: str,
    passages: list[Passage],
    *,
    max_doc_chars: int,
    chat_template_kwargs: dict[str, Any] | None = None,
    dummy_grade: str = "0",
    prefix_mode: str = "cumulative",
    grade_rubric_id: str | None = None,
    render_variant: RenderVariant | None = None,
    slot_labels_override: tuple[str, ...] | None = None,
    skeleton_slot_labels_override: tuple[str, ...] | None = None,
) -> tuple[str, list[str]]:
    """Render full chat-template prompt + per-slot prefixes.

    Uses ``tokenizer.apply_chat_template(..., add_generation_prompt=True)``
    so the assistant boundary (``<|im_start|>assistant``) and any
    model-specific generation prefix come from the tokenizer config — avoids
    hardcoding chat-template strings that drift between Qwen3 releases.

    The skeleton appended to the assistant prefix::

        Grades:
        [1] Grade: 0
        [2] Grade: 0
        ...
        [B] Grade: 0

    Each prefix in the returned list ends immediately before its slot's dummy
    ``0`` so the caller reads grade-token logits at the last token of the
    prefix in a single batched forward pass — same single-forward pattern as
    :func:`presentation_dependence.rerankers.qwen3._build_qwen3_setwise_grade_skeleton_prefixes`.
    """
    scale_spec = scale_spec_for_render_variant(render_variant)
    effective_rubric_id = render_variant.rubric_id if render_variant is not None else grade_rubric_id
    if scale_spec is not None:
        system_prompt = scale_spec.system_prompt
    else:
        rubric = _resolve_grade_rubric(effective_rubric_id)
        system_prompt = rubric.system_prompt
    if render_variant is not None and (slot_labels_override is not None or skeleton_slot_labels_override is not None):
        raise ValueError("pass either render_variant or slot/skeleton label overrides, not both")
    if skeleton_slot_labels_override is not None and not skeleton_slot_labels_override:
        raise ValueError("skeleton_slot_labels_override must not be empty")
    user_body = _build_qwen3_instruct_grade_user_body(
        instruction,
        query,
        passages,
        max_doc_chars=max_doc_chars,
        grade_rubric_id=effective_rubric_id,
        render_variant=render_variant,
        slot_labels_override=slot_labels_override,
    )
    chat_prefix = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_body},
        ],
        tokenize=False,
        add_generation_prompt=True,
        **(chat_template_kwargs or {}),
    )

    mode = str(prefix_mode).strip().lower()
    if mode == "neutral":
        mode = "independent"
    if mode not in {"cumulative", "independent"}:
        raise ValueError("prefix_mode must be 'cumulative' or 'independent'")

    prompt = chat_prefix + "Grades:\n"
    prefixes: list[str] = []
    if mode == "independent":
        if render_variant is not None:
            slot_markers = render_variant.slot_labels[: len(passages)]
            if len(slot_markers) != len(passages):
                slot_markers = _slot_labels_for_chunk(
                    len(passages),
                    id_scheme=render_variant.id_scheme,
                    id_slot_permutation=render_variant.id_slot_permutation[: len(passages)]
                    if render_variant.id_slot_permutation
                    else None,
                    rand3_seed=render_variant.rand3_seed,
                )
        elif skeleton_slot_labels_override is not None:
            slot_markers = list(skeleton_slot_labels_override)
        elif slot_labels_override is not None:
            slot_markers = list(slot_labels_override)
        else:
            slot_markers = [f"[{i}]" for i in range(1, len(passages) + 1)]
        for marker in slot_markers:
            slot_prefix = f"{marker} Grade: "
            prefixes.append(prompt + slot_prefix)
        full_prompt = prompt + "".join(f"{marker} Grade: \n" for marker in slot_markers)
        return full_prompt, prefixes

    if scale_spec is not None:
        dummy_grade = scale_spec.skeleton_dummy
    dummy = str(dummy_grade)
    if not dummy:
        raise ValueError("dummy_grade must not be empty")
    if render_variant is not None:
        slot_markers = render_variant.slot_labels[: len(passages)]
        if len(slot_markers) != len(passages):
            slot_markers = _slot_labels_for_chunk(
                len(passages),
                id_scheme=render_variant.id_scheme,
                id_slot_permutation=render_variant.id_slot_permutation[: len(passages)]
                if render_variant.id_slot_permutation
                else None,
                rand3_seed=render_variant.rand3_seed,
            )
    elif skeleton_slot_labels_override is not None:
        slot_markers = list(skeleton_slot_labels_override)
    elif slot_labels_override is not None:
        slot_markers = list(slot_labels_override)
    else:
        slot_markers = [f"[{i}]" for i in range(1, len(passages) + 1)]

    for marker in slot_markers:
        slot_prefix = f"{marker} Grade: "
        prompt += slot_prefix + dummy + "\n"
        prefixes.append(prompt[: -(len(dummy) + 1)])
    return prompt, prefixes


class Qwen3InstructGradeReranker(Reranker):
    """Qwen3-Instruct prompted as a B-document continuous-grade scorer.

    Self-distill-only wrapper. Always emits continuous expected-grade scores;
    has no integer-grade or yes/no logit fallback. See module docstring for
    why this lives in its own file rather than as a flag on
    :class:`Qwen3Reranker`.
    """

    paradigm = "batched_pointwise"
    supports_query_batching = True  # rank_query_batch dispatches all queries' prefixes in one engine call
    supports_render_variants = True
    supports_scale_variants = True

    def __init__(self, config: dict):  # noqa: C901
        super().__init__(config)
        self.logger = setup_logging(self.__class__.__name__, config)

        rc = self.reranker_config
        model_name = rc.get("model_name", "Qwen/Qwen3-4B-Instruct-2507")
        requested_device = rc.get("device", "auto")
        self.device = _resolve_device(requested_device)
        dtype = _resolve_dtype(rc.get("dtype", "auto"), self.device)
        revision = rc.get("revision")

        self.instruction = rc.get("instruction", _DEFAULT_INSTRUCTION)
        self.grade_rubric_id = str(rc.get("grade_rubric_id", _DEFAULT_GRADE_RUBRIC_ID)).strip()
        _resolve_grade_rubric(self.grade_rubric_id)
        self.max_length = int(rc.get("max_length", 8192))
        self.docs_per_score_forward = int(rc.get("docs_per_score_forward", 20))
        if self.docs_per_score_forward <= 0:
            raise ValueError("reranker.docs_per_score_forward must be positive")
        self.chunk_assignment = str(rc.get("chunk_assignment", "sorted")).strip().lower()
        if self.chunk_assignment not in {"sorted", "interleaved"}:
            raise ValueError("reranker.chunk_assignment must be 'sorted' or 'interleaved'")
        self.presentation_order = str(rc.get("presentation_order", "forward")).strip().lower()
        if self.presentation_order not in {"forward", "reverse"}:
            raise ValueError("reranker.presentation_order must be 'forward' or 'reverse'")
        self.batch_size = int(rc.get("batch_size", self.docs_per_score_forward))
        if self.batch_size <= 0:
            raise ValueError("reranker.batch_size must be positive")
        self.max_doc_chars = int(rc.get("max_doc_chars", 1200))
        self.grade_prefix_mode = str(rc.get("grade_prefix_mode", "cumulative")).strip().lower()
        if self.grade_prefix_mode == "neutral":
            self.grade_prefix_mode = "independent"
        if self.grade_prefix_mode not in {"cumulative", "independent"}:
            raise ValueError("reranker.grade_prefix_mode must be 'cumulative' or 'independent'")
        self.grade_skeleton_dummy = str(rc.get("grade_skeleton_dummy", "0"))
        if self.grade_prefix_mode == "cumulative" and not self.grade_skeleton_dummy:
            raise ValueError("reranker.grade_skeleton_dummy must not be empty")
        chat_template_kwargs = rc.get("chat_template_kwargs") or {}
        if not isinstance(chat_template_kwargs, dict):
            raise ValueError("reranker.chat_template_kwargs must be a mapping when provided")
        self.chat_template_kwargs = dict(chat_template_kwargs)

        # Grade readout strategy for the per-query rank() path:
        #   "per_prefix": score B nested-prefix sequences per chunk (current
        #                      default; preserved bit-for-bit).
        #   "single_forward": one forward over the full skeleton, reading all B
        #                      slot logits at once. Mathematically equivalent to
        #                      per_prefix when the chunk fits in max_length (causal
        #                      mask), but encodes the shared body once instead of
        #                      B× and drops B→1 sequences per chunk.
        self.grade_readout = str(rc.get("grade_readout", "per_prefix")).strip().lower()
        if self.grade_readout not in {"per_prefix", "single_forward"}:
            raise ValueError("reranker.grade_readout must be 'per_prefix' or 'single_forward'")

        # The wrapper is self-distill-only; the continuous_readout flag exists
        # for surface compatibility with `Qwen3Reranker` configs but is NOT a
        # togglable option here. Reject explicit ``false`` to fail loud rather
        # than silently use a different path the caller didn't expect.
        if rc.get("continuous_readout", True) is False:
            raise ValueError(
                "Qwen3InstructGradeReranker is a self-distill teacher and "
                "always uses continuous readout; set continuous_readout=true "
                "or remove the key. For integer-grade scoring, use Qwen3Reranker "
                "with scoring_mode=setwise_grade_prompt."
            )

        self.inference_engine_kind = str(rc.get("inference_engine", "hf")).strip().lower()
        if self.inference_engine_kind not in {"hf", "vllm"}:
            raise ValueError(
                f"reranker.inference_engine must be one of 'hf', 'vllm' (got {self.inference_engine_kind!r})"
            )
        # grade_readout=single_forward is an HF-only optimization. On vLLM it
        # must read interior positions via prompt_logprobs, which (a) projects
        # the full vocab over every prompt position → OOM / ~1.7x SLOWER than
        # per_prefix, and (b) does NOT reproduce per_prefix scores
        # vLLM's per_prefix is already the memory-light, correct, faster interior reader
        # (single-position projection + prefix-cached body), so reject the
        # combination rather than silently produce wrong/slow scores.
        if self.grade_readout == "single_forward" and self.inference_engine_kind == "vllm":
            raise ValueError(
                "reranker.grade_readout='single_forward' is not supported with "
                "inference_engine='vllm' (slower + not score-equivalent via "
                "prompt_logprobs). "
                "Use grade_readout='per_prefix' on vLLM; single_forward is the HF path."
            )

        self.logger.info(
            "Loading %s on device=%s dtype=%s docs_per_score_forward=%d engine=%s",
            model_name,
            self.device,
            dtype,
            self.docs_per_score_forward,
            self.inference_engine_kind,
        )
        if revision:
            self.logger.info("Pinned HF revision=%s", revision)

        from transformers import AutoTokenizer  # local import: heavy

        # Tokenizer is loaded for both engines: HF needs it for tokenize+pad+
        # forward, vLLM needs it only to resolve grade token IDs (vLLM owns
        # its own internal tokenization for actual prompts).
        tok_kwargs: dict[str, Any] = {"padding_side": "left"}
        if revision:
            tok_kwargs["revision"] = revision
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **tok_kwargs)
        if getattr(self.tokenizer, "pad_token", None) is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Same truncation as the Qwen3-Reranker setwise path:
        # left-truncate so the slot tail (the literal `Grade: ` immediately
        # before the dummy `0`) survives even when shared-context body
        # overflows ``max_length``: otherwise the readout positions drift.
        if hasattr(self.tokenizer, "truncation_side"):
            self.tokenizer.truncation_side = "left"

        # Resolve grade-token ids at init so tokenizer drift fails loud,
        # not silently mid-run. Same helper Qwen3Reranker uses.
        self.grade_token_ids = _resolve_grade_token_ids(self.tokenizer)
        self._scale_readout_cache: dict[str, tuple[list[int], GradeScaleSpec]] = {}

        # Optional LoRA / PEFT adapter(s) on top of the base instruct model. Used
        # to evaluate self-distill student checkpoints through the same scoring
        # pipeline as the offshelf teacher. Single ``lora_path`` and/or
        # multi-adapter ``lora_adapters`` (shared-base bundles) are parsed onto
        # self here; engine wiring + per-surface ``set_active_adapter`` switching
        # are family-agnostic (base Reranker).
        parse_lora_config(self, rc, engine_kind=self.inference_engine_kind)

        if self.inference_engine_kind == "hf":
            from transformers import AutoModelForCausalLM

            model_kwargs: dict[str, Any] = {"dtype": dtype}
            if revision:
                model_kwargs["revision"] = revision
            if rc.get("attn_implementation"):
                model_kwargs["attn_implementation"] = rc["attn_implementation"]
            if "trust_remote_code" in rc:
                model_kwargs["trust_remote_code"] = bool(rc["trust_remote_code"])

            self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
            if self.lora_path is not None:
                from peft import PeftModel  # type: ignore[import-not-found]

                self.logger.info("Loading LoRA adapter from %s", self.lora_path)
                self.model = PeftModel.from_pretrained(self.model, self.lora_path)
                if self.merge_lora:
                    self.logger.info("Merging LoRA into base weights (merge_and_unload)")
                    self.model = self.model.merge_and_unload()
            self.model.to(self.device)
            self.model.eval()
            self.engine: LogitSkeletonEngine = HFLogitSkeletonEngine(
                model=self.model,
                tokenizer=self.tokenizer,
                max_length=self.max_length,
                batch_size=self.batch_size,
            )
        else:  # vllm
            # vLLM owns the model's GPU memory; do NOT also load the HF model
            # or we'll OOM (4B × 2 = 8GB minimum on top of vLLM's KV cache).
            self.model = None
            vllm_settings: dict[str, Any] = dict(rc.get("vllm_settings") or {})
            vllm_settings.setdefault("model_name", model_name)
            vllm_settings.setdefault("max_model_len", self.max_length)
            vllm_settings.setdefault("dtype", str(rc.get("dtype", "auto")))
            # See Qwen3Reranker for context: pass tokenizer + max_length so
            # the engine pre-tokenizes with add_special_tokens=False, matching
            # HFLogitSkeletonEngine bit-for-bit.
            vllm_settings["tokenizer"] = self.tokenizer
            vllm_settings.setdefault("max_length", self.max_length)
            apply_vllm_lora_settings(self, rc, vllm_settings)
            self.engine = make_engine("vllm", **vllm_settings)

    def _build_chunk_skeleton(self, query: Query, chunk: list[Passage]) -> tuple[str, list[str]]:
        """Build (full skeleton prompt, per-slot prefixes) for one B-doc chunk."""
        return _build_qwen3_instruct_grade_skeleton(
            self.tokenizer,
            self.instruction,
            query["text"],
            chunk,
            max_doc_chars=self.max_doc_chars,
            chat_template_kwargs=getattr(self, "chat_template_kwargs", None),
            dummy_grade=getattr(self, "grade_skeleton_dummy", "0"),
            prefix_mode=getattr(self, "grade_prefix_mode", "cumulative"),
            grade_rubric_id=getattr(self, "grade_rubric_id", _DEFAULT_GRADE_RUBRIC_ID),
            render_variant=getattr(self, "render_variant", None),
        )

    def _build_chunk_prefixes(self, query: Query, chunk: list[Passage]) -> list[str]:
        """Build the slot prefixes for one B-doc chunk, ready for the engine."""
        _full_prompt, slot_prefixes = self._build_chunk_skeleton(query, chunk)
        return slot_prefixes

    def score_selected_slots(
        self,
        query: Query,
        requests: list[dict[str, Any]],
    ) -> list[float]:
        """Score one selected grade slot from each independently rendered context.

        A request contains ``passages`` and ``target_skeleton_index``. Optional
        ``document_slot_labels`` and ``skeleton_slot_labels`` decouple the
        document markers from the answer skeleton. This is used by the
        context-decomposition
        scaffold control to show one target document as ``[p]`` while retaining
        a twenty-line cumulative grade skeleton, without inserting fake
        companion documents into the user prompt.
        """
        if not requests:
            return []

        selected_prefixes: list[str] = []
        for request in requests:
            passages = list(request.get("passages") or [])
            if not passages:
                raise ValueError("score_selected_slots requests require at least one passage")
            document_labels_raw = request.get("document_slot_labels")
            document_labels = (
                tuple(str(label) for label in document_labels_raw) if document_labels_raw is not None else None
            )
            skeleton_labels_raw = request.get("skeleton_slot_labels")
            skeleton_labels = (
                tuple(str(label) for label in skeleton_labels_raw) if skeleton_labels_raw is not None else None
            )
            _prompt, prefixes = _build_qwen3_instruct_grade_skeleton(
                self.tokenizer,
                self.instruction,
                query["text"],
                passages,
                max_doc_chars=self.max_doc_chars,
                chat_template_kwargs=getattr(self, "chat_template_kwargs", None),
                dummy_grade=getattr(self, "grade_skeleton_dummy", "0"),
                prefix_mode=getattr(self, "grade_prefix_mode", "cumulative"),
                grade_rubric_id=getattr(
                    self,
                    "grade_rubric_id",
                    _DEFAULT_GRADE_RUBRIC_ID,
                ),
                render_variant=getattr(self, "render_variant", None),
                slot_labels_override=document_labels,
                skeleton_slot_labels_override=skeleton_labels,
            )
            target_index = int(request.get("target_skeleton_index", 0))
            if not 0 <= target_index < len(prefixes):
                raise ValueError(
                    f"target_skeleton_index out of range: {target_index} for {len(prefixes)} skeleton slots"
                )
            selected_prefixes.append(prefixes[target_index])

        grade_token_ids, scale_spec = self._resolve_readout()
        if hasattr(self.engine, "score_prefix_probabilities"):
            vectors = self.engine.score_prefix_probabilities(
                selected_prefixes,
                grade_token_ids,
            )
            if len(vectors) != len(selected_prefixes):
                raise RuntimeError(
                    f"engine returned {len(vectors)} vectors for {len(selected_prefixes)} selected prefixes"
                )
            return self._scores_from_grade_probabilities(
                vectors,
                scale_spec=scale_spec,
            )

        expected_grades = self.engine.score_prefixes(
            selected_prefixes,
            grade_token_ids,
        )
        if len(expected_grades) != len(selected_prefixes):
            raise RuntimeError(
                f"engine returned {len(expected_grades)} scores for {len(selected_prefixes)} selected prefixes"
            )
        if scale_spec is None:
            return [float(grade) / 3.0 for grade in expected_grades]
        return [scale_spec.normalize_expected_grade(float(grade)) for grade in expected_grades]

    def _score_chunk_grade_probabilities(self, query: Query, chunk: list[Passage]) -> list[list[float]]:
        full_prompt, slot_prefixes = self._build_chunk_skeleton(query, chunk)
        grade_token_ids, _ = self._resolve_readout()
        if getattr(self, "grade_readout", "per_prefix") == "single_forward" and hasattr(
            self.engine, "score_skeleton_probabilities"
        ):
            return self.engine.score_skeleton_probabilities(full_prompt, slot_prefixes, grade_token_ids)
        return self.engine.score_prefix_probabilities(slot_prefixes, grade_token_ids)

    def _active_scale_spec(self) -> GradeScaleSpec | None:
        return scale_spec_for_render_variant(getattr(self, "render_variant", None))

    def _resolve_readout(self) -> tuple[list[int], GradeScaleSpec | None]:
        spec = self._active_scale_spec()
        if spec is None:
            return self.grade_token_ids, None
        cached = self._scale_readout_cache.get(spec.scheme)
        if cached is not None:
            return cached[0], cached[1]
        token_ids = _resolve_grade_token_ids(self.tokenizer, grade_strings=spec.grade_strings)
        self._scale_readout_cache[spec.scheme] = (token_ids, spec)
        return token_ids, spec

    @staticmethod
    def _scores_from_grade_probabilities(
        vectors: list[list[float]],
        *,
        scale_spec: GradeScaleSpec | None = None,
    ) -> list[float]:
        """Map P(g) vectors to the standard [0, 1] reranker score contract."""
        if scale_spec is None:
            return [sum(float(i) * float(p) for i, p in enumerate(vector)) / 3.0 for vector in vectors]
        return [
            scale_spec.normalize_expected_grade(
                sum(float(g) * float(p) for g, p in zip(scale_spec.grade_values, vector, strict=True))
            )
            for vector in vectors
        ]

    @staticmethod
    def _grade_probabilities_from_scores(
        scores: list[float],
        *,
        scale_spec: GradeScaleSpec | None = None,
    ) -> list[list[float]]:
        """Compatibility fallback for old scalar-only test engines."""
        if scale_spec is None:
            max_grade = 3.0
            n_grades = 4
        else:
            max_grade = max(scale_spec.grade_values)
            n_grades = len(scale_spec.grade_values)
        vectors: list[list[float]] = []
        for score in scores:
            if scale_spec is None:
                grade_score = max(0.0, min(max_grade, float(score) * max_grade))
            else:
                raw = float(score) * scale_spec.score_divisor + scale_spec.score_offset
                grade_score = max(min(scale_spec.grade_values), min(max_grade, raw))
            lo = int(grade_score)
            hi = min(n_grades - 1, lo + 1) if grade_score > lo else lo
            probs = [0.0] * n_grades
            if lo == hi:
                probs[lo] = 1.0
            else:
                frac = grade_score - lo
                probs[lo] = 1.0 - frac
                probs[hi] = frac
            vectors.append(probs)
        return vectors

    def _score_chunk_grades_continuous(self, query: Query, chunk: list[Passage]) -> list[float]:
        vectors = self._score_chunk_grade_probabilities(query, chunk)
        self._last_grade_probabilities_init_order = vectors
        _, scale_spec = self._resolve_readout()
        return self._scores_from_grade_probabilities(vectors, scale_spec=scale_spec)

    def _chunk_indices(self, n_passages: int) -> list[list[int]]:
        """Return deterministic passage-index groups for one scoring pass."""
        if n_passages <= 0:
            return []
        order = list(range(n_passages))
        if getattr(self, "presentation_order", "forward") == "reverse":
            order.reverse()
        if getattr(self, "chunk_assignment", "sorted") == "sorted":
            return [
                order[start : start + self.docs_per_score_forward]
                for start in range(0, n_passages, self.docs_per_score_forward)
            ]
        n_chunks = (n_passages + self.docs_per_score_forward - 1) // self.docs_per_score_forward
        return [order[offset::n_chunks] for offset in range(n_chunks)]

    def rank(self, query: Query, passages: list[Passage]) -> RankResult:
        if not passages:
            return {
                "top_k_psgs": [],
                "scores_init_order": [],
                "prompting_runtimes": [0.0],
                "paradigm": self.paradigm,
            }

        t0 = time.perf_counter()
        scores: list[float] = [0.0] * len(passages)
        grade_vectors: list[list[float]] = [[] for _ in passages]
        for indices in self._chunk_indices(len(passages)):
            chunk = [passages[index] for index in indices]
            self._last_grade_probabilities_init_order = None
            chunk_scores = self._score_chunk_grades_continuous(query, chunk)
            chunk_vectors = self._last_grade_probabilities_init_order
            if not isinstance(chunk_vectors, list) or len(chunk_vectors) != len(chunk_scores):
                _, scale_spec = self._resolve_readout()
                chunk_vectors = self._grade_probabilities_from_scores(chunk_scores, scale_spec=scale_spec)
            for index, score, vector in zip(indices, chunk_scores, chunk_vectors):
                scores[index] = score
                grade_vectors[index] = vector
        elapsed = time.perf_counter() - t0
        result = scores_to_rank_result(scores, passages, elapsed, self.paradigm, model_name=self.__class__.__name__)
        result["grade_probabilities_init_order"] = grade_vectors
        return result

    def rank_query_batch(  # noqa: C901
        self,
        items: list[tuple[Query, list[Passage]]],
    ) -> list[RankResult]:
        """Cross-query batched scoring.

        Builds slot prefixes for every (query, passages) pair across every
        B-doc chunk, dispatches all of them in a single engine call, then
        unpacks the results back into per-query :class:`RankResult` objects.
        With the vLLM backend this lets one ``LLM.generate`` packing pass
        cover N queries' worth of prefixes — amortising scheduler overhead
        and lifting GPU utilisation past what within-query batching alone
        achieves.

        With the HF backend the savings are marginal because longest-in-batch
        padding makes heterogeneous prompt lengths cost extra. The default
        ``cross_query_batch=1`` in the teacher leaves HF runs unchanged;
        callers explicitly opt in.
        """
        if not items:
            return []

        # Phase 1: build prefixes per (query, chunk), tagging each with the
        # destination ``items`` index so we can stitch results back together
        # after the single engine call.
        all_prefixes: list[str] = []
        plan: list[tuple[int, list[int]]] = []  # (item_idx, passage indices)
        for item_idx, (query, passages) in enumerate(items):
            if not passages:
                continue
            for indices in self._chunk_indices(len(passages)):
                chunk = [passages[index] for index in indices]
                prefixes = self._build_chunk_prefixes(query, chunk)
                if len(prefixes) != len(chunk):
                    raise RuntimeError(
                        f"prefix builder returned {len(prefixes)} prefixes for chunk of "
                        f"{len(chunk)} passages — invariant violated."
                    )
                plan.append((item_idx, indices))
                all_prefixes.extend(prefixes)

        # Phase 2: single engine call covers every (item, chunk, slot).
        t0 = time.perf_counter()
        grade_token_ids, scale_spec = self._resolve_readout()
        if all_prefixes:
            if hasattr(self.engine, "score_prefix_probabilities"):
                grade_vectors = self.engine.score_prefix_probabilities(all_prefixes, grade_token_ids)
                if len(grade_vectors) != len(all_prefixes):
                    raise RuntimeError(
                        f"engine returned {len(grade_vectors)} vectors for {len(all_prefixes)} prefixes; "
                        "request/result count mismatch."
                    )
                unit_scores = self._scores_from_grade_probabilities(grade_vectors, scale_spec=scale_spec)
            else:
                expected_grades = self.engine.score_prefixes(all_prefixes, grade_token_ids)
                if len(expected_grades) != len(all_prefixes):
                    raise RuntimeError(
                        f"engine returned {len(expected_grades)} scores for {len(all_prefixes)} prefixes; "
                        "request/result count mismatch."
                    )
                if scale_spec is None:
                    unit_scores = [g / 3.0 for g in expected_grades]
                else:
                    unit_scores = [scale_spec.normalize_expected_grade(g) for g in expected_grades]
                grade_vectors = self._grade_probabilities_from_scores(unit_scores, scale_spec=scale_spec)
        else:
            grade_vectors = []
            unit_scores = []
        elapsed = time.perf_counter() - t0

        # Phase 3: per-item score vector reconstruction. Each (item, chunk)
        # contributes a contiguous slice of the engine output.
        per_item_scores: list[list[float]] = [[0.0] * len(passages) for _, passages in items]
        per_item_vectors: list[list[list[float]]] = [[[] for _ in passages] for _, passages in items]
        cursor = 0
        for item_idx, indices in plan:
            n = len(indices)
            for index, score, vector in zip(
                indices,
                unit_scores[cursor : cursor + n],
                grade_vectors[cursor : cursor + n],
            ):
                per_item_scores[item_idx][index] = score
                per_item_vectors[item_idx][index] = vector
            cursor += n

        # Phase 4: assemble RankResults. Per-item ``elapsed`` is the engine
        # call wall-clock divided across items so total time is conserved
        # without lying about per-item cost. Empty-passage items get a
        # 0-second short-circuit, matching the rank() contract.
        n_nonempty = max(sum(1 for _, p in items if p), 1)
        per_item_elapsed = elapsed / n_nonempty
        results: list[RankResult] = []
        for (query, passages), scores, vectors in zip(items, per_item_scores, per_item_vectors):
            if not passages:
                results.append(
                    {
                        "top_k_psgs": [],
                        "scores_init_order": [],
                        "grade_probabilities_init_order": [],
                        "prompting_runtimes": [0.0],
                        "paradigm": self.paradigm,
                    }
                )
                continue
            result = scores_to_rank_result(
                scores,
                passages,
                per_item_elapsed,
                self.paradigm,
                model_name=self.__class__.__name__,
            )
            result["grade_probabilities_init_order"] = vectors
            results.append(result)
        return results
