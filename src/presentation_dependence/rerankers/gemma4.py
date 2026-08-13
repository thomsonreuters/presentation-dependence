"""Gemma-4-E4B-it expected-grade reranker.

Model: ``google/gemma-4-E4B-it`` (Apache-2.0 per the model repository metadata).
See ``THIRD_PARTY_NOTICES.md`` for model and dataset terms.

- Path A (:class:`Gemma4GradeReranker`, ``continuous_readout=false``) generates
  a JSON grade object, parses integer grades from 0 to 3, and maps them to [0, 1].
  Same body as ``Qwen3Reranker`` ``setwise_grade_prompt``.
- Expected-grade readout (:class:`Gemma4GradeReranker`, ``continuous_readout=true``) uses a
  fixed grade skeleton and reads E[grade] over {0,1,2,3} in one forward pass.
  The self-distillation teacher uses the same readout
  (:class:`Qwen3InstructGradeReranker`).

The user-body text is identical to the corresponding Qwen3 paths; the chat
template is applied via ``tokenizer.apply_chat_template`` so it follows
whatever Gemma-4 ships in its tokenizer config rather than hardcoded
``<|im_start|>`` strings.  A graceful fallback merges system + user into a
single turn if the tokenizer does not support a system role.

Implementation references
-------------------------
- Prompt scaffold: ``src/presentation_dependence/rerankers/qwen3.py``
  ``_build_setwise_prompt(output_mode="grade")`` /
  ``_build_qwen3_setwise_grade_skeleton_prefixes`` /
  ``_build_qwen3_setwise_yesno_prompt_prefixes``
- E[grade] math: ``src/presentation_dependence/self_distill/readout.py``
"""

from __future__ import annotations

import time
from typing import Any

from presentation_dependence.rerankers._rank_result import scores_to_rank_result as _scores_to_rank_result
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
from presentation_dependence.rerankers.qwen3 import (
    _DEFAULT_INSTRUCTION,
    _SETWISE_SYSTEM_PROMPT,
    _parse_setwise_grades,
)
from presentation_dependence.rerankers.grade_rubrics import (
    build_pathc_grade_user_body,
    resolve_grade_rubric as _resolve_grade_rubric,
)
from presentation_dependence.self_distill.engines import HFLogitSkeletonEngine, LogitSkeletonEngine, make_engine
from presentation_dependence.self_distill.readout import resolve_grade_token_ids as _resolve_grade_token_ids
from presentation_dependence.utils.setup_logging import setup_logging


_DEFAULT_MODEL = "google/gemma-4-E4B-it"
_GRADE_SYSTEM_PROMPT = (
    "You are a search relevance grader. For each numbered document, output one fixed integer "
    "relevance grade in {0, 1, 2, 3}. Do not rank the documents and do not explain."
)
_GEMMA4_GRADE_DUMMY = "0"


# ---------------------------------------------------------------------------
# Chat-template helpers
# ---------------------------------------------------------------------------


def _apply_gemma4_chat_template(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> str:
    """Apply Gemma-4's chat template; falls back to merging system into user if needed.

    Uses ``tokenizer.apply_chat_template(..., tokenize=False,
    add_generation_prompt=True)`` so the returned string ends immediately after
    the assistant-turn start token (ready for appending skeleton / slot lines).
    """
    apply_fn = getattr(tokenizer, "apply_chat_template", None)
    if apply_fn is not None:
        try:
            return str(
                apply_fn(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    **(chat_template_kwargs or {}),
                )
            )
        except (AttributeError, TypeError, ValueError):
            pass
    # Graceful fallback: flatten all message contents into one block.
    parts = [msg["content"] for msg in messages if msg.get("content")]
    return "\n\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# Prompt body builders: identical content to the Qwen3 reference prompts
# ---------------------------------------------------------------------------


def _build_grade_json_user_body(
    instruction: str | None,
    query: str,
    passages: list[Passage],
    *,
    max_doc_chars: int,
) -> str:
    """Grade-JSON user body; mirrors Qwen3Reranker setwise_grade_prompt body.

    Content is identical to the ``output_mode="grade"`` branch of
    :func:`presentation_dependence.rerankers.qwen3._build_setwise_prompt`; only the
    surrounding chat-template framing differs.
    """
    inst = instruction or _DEFAULT_INSTRUCTION
    doc_lines: list[str] = []
    for i, passage in enumerate(passages, start=1):
        text = " ".join(str(passage["text"]).split())
        if max_doc_chars > 0:
            text = text[:max_doc_chars]
        doc_lines.append(f"[{i}] {text}")

    common = (
        f"<Instruct>: {inst}\n"
        f"<Query>: {query}\n\n"
        "<Documents>:\n" + "\n".join(doc_lines) + "\n\n"
        "Evaluate every document independently for relevance to the query, "
        "while comparing documents in the batch.\n"
    )
    instructions = (
        "Assign exactly one integer relevance grade to every document:\n"
        "- 3 = directly and completely answers the query\n"
        "- 2 = strongly relevant but incomplete\n"
        "- 1 = weakly relevant or topical background\n"
        "- 0 = not relevant\n"
        "Return exactly one JSON object and no other text after the assistant prefix.\n"
        "The JSON object must have this shape: "
        '{"grades":[{"id":<document_id>,"grade":<integer_0_to_3>}]}.\n'
        "Use every id from 1 to the number of documents exactly once, including documents graded 0."
    )
    return common + instructions


def _build_grade_skeleton_user_body(
    instruction: str | None,
    query: str,
    passages: list[Passage],
    *,
    max_doc_chars: int,
    grade_rubric_id: str | None = None,
) -> str:
    """Grade-skeleton user body; mirrors Qwen3Reranker continuous_readout=true body.

    Content is identical to
    :func:`presentation_dependence.rerankers.qwen3._build_qwen3_setwise_grade_skeleton_prefixes`'s
    user section when ``grade_rubric_id`` is ``None`` or ``"relevance_v1"``,
    and must stay so: the published passage-reranking numbers were produced
    with that exact text across every base. Any other rubric id, such as
    ``"response_quality_v1"``, routes the body through the shared
    expected-grade rubric builder instead.
    """
    if grade_rubric_id is not None and grade_rubric_id != "relevance_v1":
        return build_pathc_grade_user_body(
            instruction,
            query,
            [str(p["text"]) for p in passages],
            max_doc_chars=max_doc_chars,
            grade_rubric_id=grade_rubric_id,
        )
    inst = instruction or _DEFAULT_INSTRUCTION
    doc_lines: list[str] = []
    for i, passage in enumerate(passages, start=1):
        text = " ".join(str(passage["text"]).split())
        if max_doc_chars > 0:
            text = text[:max_doc_chars]
        doc_lines.append(f"[{i}] {text}")

    return (
        f"<Instruct>: {inst}\n"
        f"<Query>: {query}\n\n"
        "<Documents>:\n" + "\n".join(doc_lines) + "\n\n"
        "Evaluate every document independently for relevance to the query.\n"
        "Assign exactly one integer relevance grade to every document:\n"
        "- 3 = directly and completely answers the query\n"
        "- 2 = strongly relevant but incomplete\n"
        "- 1 = weakly relevant or topical background\n"
        "- 0 = not relevant\n"
        "Output one line per document in the form: "
        "[<id>] Grade: <0|1|2|3>"
    )


# ---------------------------------------------------------------------------
# Full prompt builders (body + chat template + skeleton/slot lines)
# ---------------------------------------------------------------------------


def _build_gemma4_grade_skeleton_prefixes(
    tokenizer: Any,
    instruction: str | None,
    query: str,
    passages: list[Passage],
    *,
    max_doc_chars: int,
    dummy_grade: str = _GEMMA4_GRADE_DUMMY,
    prefix_mode: str = "cumulative",
    chat_template_kwargs: dict[str, Any] | None = None,
    grade_rubric_id: str | None = None,
) -> tuple[str, list[str]]:
    r"""Build Gemma-4 grade-skeleton prompt + per-slot prefixes (expected-grade readout).

    Mirrors :func:`presentation_dependence.rerankers.qwen3._build_qwen3_setwise_grade_skeleton_prefixes`
    but wraps the user body in Gemma-4's chat template instead of hardcoded Qwen3
    ``<|im_start|>`` strings.

    Layout (in the assistant section after the chat template prefix)::

        Grades:
        [1] Grade: 0    ← slot 1 prefix ends just before this "0"
        [2] Grade: 0    ← slot 2 prefix includes slot 1's "0\\n", ends before slot 2's "0"
        ...

    ``grade_rubric_id`` is ``None`` by default (unchanged passage-relevance
    body and system prompt). Set it to score a different item type, e.g.
    ``"response_quality_v1"`` for the response-ranking arm.
    """
    user_body = _build_grade_skeleton_user_body(
        instruction, query, passages, max_doc_chars=max_doc_chars, grade_rubric_id=grade_rubric_id
    )
    non_default_rubric = grade_rubric_id is not None and grade_rubric_id != "relevance_v1"
    system_prompt = (
        _resolve_grade_rubric(grade_rubric_id).system_prompt if non_default_rubric else _SETWISE_SYSTEM_PROMPT
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_body},
    ]
    chat_prefix = _apply_gemma4_chat_template(
        tokenizer,
        messages,
        chat_template_kwargs=chat_template_kwargs,
    )

    mode = str(prefix_mode).strip().lower()
    if mode == "neutral":
        mode = "independent"
    if mode not in {"cumulative", "independent"}:
        raise ValueError("prefix_mode must be 'cumulative' or 'independent'")

    prompt = chat_prefix + "Grades:\n"
    prefixes: list[str] = []
    if mode == "independent":
        for i in range(1, len(passages) + 1):
            slot_prefix = f"[{i}] Grade: "
            prefixes.append(prompt + slot_prefix)
        full_prompt = prompt + "".join(f"[{i}] Grade: \n" for i in range(1, len(passages) + 1))
        return full_prompt, prefixes

    dummy = str(dummy_grade)
    if not dummy:
        raise ValueError("dummy_grade must not be empty")
    for i in range(1, len(passages) + 1):
        slot_prefix = f"[{i}] Grade: "
        prompt += slot_prefix + dummy + "\n"
        prefixes.append(prompt[: -(len(dummy) + 1)])
    return prompt, prefixes


# ---------------------------------------------------------------------------
# Reranker classes
# ---------------------------------------------------------------------------


class Gemma4GradeReranker(Reranker):
    """Gemma-4-E4B-it shared-context grade scorer (Path A JSON + continuous expected-grade readout).

    **Path A** (``continuous_readout=false``): generates a JSON grades object and
    parses integer grades 0–3.  The same generation-and-parse path used by
    :class:`Qwen3Reranker` ``setwise_grade_prompt`` — same user body, same
    parser (:func:`presentation_dependence.rerankers.qwen3._parse_setwise_grades`).

    **Expected-grade readout** (``continuous_readout=true``): fixed grade skeleton +
    single-forward E[grade] readout over {0,1,2,3}.  Identical readout to
    :class:`Qwen3InstructGradeReranker`; adapted to Gemma-4's chat template.

    Used by the cross-family expected-grade base and LoRA runs.
    """

    paradigm = "batched_pointwise"

    def __init__(self, config: dict):  # noqa: C901
        super().__init__(config)
        self.logger = setup_logging(self.__class__.__name__, config)

        rc = self.reranker_config
        model_name = rc.get("model_name", _DEFAULT_MODEL)
        requested_device = rc.get("device", "auto")
        self.device = _resolve_device(requested_device)
        dtype = _resolve_dtype(rc.get("dtype", "auto"), self.device)
        revision = rc.get("revision")

        self.instruction = rc.get("instruction", _DEFAULT_INSTRUCTION)
        # None preserves the exact passage-relevance body/system-prompt this class
        # shipped with; set e.g. "response_quality_v1" for the response-ranking arm.
        self.grade_rubric_id = rc.get("grade_rubric_id")
        self.max_length = int(rc.get("max_length", 8192))
        self.docs_per_score_forward = int(rc.get("docs_per_score_forward", 20))
        if self.docs_per_score_forward <= 0:
            raise ValueError("reranker.docs_per_score_forward must be positive")
        self.batch_size = int(rc.get("batch_size", self.docs_per_score_forward))
        if self.batch_size <= 0:
            raise ValueError("reranker.batch_size must be positive")
        self.max_doc_chars = int(rc.get("max_doc_chars", 1200))
        self.grade_prefix_mode = str(rc.get("grade_prefix_mode", "cumulative")).strip().lower()
        if self.grade_prefix_mode == "neutral":
            self.grade_prefix_mode = "independent"
        if self.grade_prefix_mode not in {"cumulative", "independent"}:
            raise ValueError("reranker.grade_prefix_mode must be 'cumulative' or 'independent'")
        self.grade_skeleton_dummy = str(rc.get("grade_skeleton_dummy", _GEMMA4_GRADE_DUMMY))
        if self.grade_prefix_mode == "cumulative" and not self.grade_skeleton_dummy:
            raise ValueError("reranker.grade_skeleton_dummy must not be empty")
        chat_template_kwargs = rc.get("chat_template_kwargs") or {}
        if not isinstance(chat_template_kwargs, dict):
            raise ValueError("reranker.chat_template_kwargs must be a mapping when provided")
        self.chat_template_kwargs = dict(chat_template_kwargs)
        self.max_new_tokens = int(rc.get("max_new_tokens", 1024))
        self.setwise_missing_grade = rc.get("setwise_missing_grade")
        self.continuous_readout = bool(rc.get("continuous_readout", False))

        self.inference_engine_kind = str(rc.get("inference_engine", "hf")).strip().lower()
        if self.inference_engine_kind not in {"hf", "vllm"}:
            raise ValueError(
                f"reranker.inference_engine must be one of 'hf', 'vllm' (got {self.inference_engine_kind!r})"
            )
        if self.inference_engine_kind == "vllm" and not self.continuous_readout:
            raise ValueError(
                "reranker.inference_engine='vllm' requires reranker.continuous_readout=true "
                "(vLLM is wired only for the continuous-readout path)."
            )

        self.logger.info(
            "Loading %s on device=%s dtype=%s docs_per_score_forward=%d engine=%s continuous_readout=%s",
            model_name,
            self.device,
            dtype,
            self.docs_per_score_forward,
            self.inference_engine_kind,
            self.continuous_readout,
        )
        if revision:
            self.logger.info("Pinned HF revision=%s", revision)

        from transformers import AutoTokenizer  # local import: heavy

        tok_kwargs: dict[str, Any] = {"padding_side": "left"}
        if revision:
            tok_kwargs["revision"] = revision
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **tok_kwargs)
        if getattr(self.tokenizer, "pad_token", None) is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Left-truncation keeps the slot tail (Grade: / Grades: suffix) intact
        # when the shared-context body overflows max_length.
        if hasattr(self.tokenizer, "truncation_side"):
            self.tokenizer.truncation_side = "left"

        # Pre-flight: resolve grade token IDs; fails loud on multi-token grades.
        # Pre-flight invariant: each grade literal must tokenize to one token.
        self.grade_token_ids = _resolve_grade_token_ids(self.tokenizer)
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
            self.engine: LogitSkeletonEngine | None = (
                HFLogitSkeletonEngine(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    max_length=self.max_length,
                    batch_size=self.batch_size,
                )
                if self.continuous_readout
                else None
            )
        else:  # vllm: only valid with continuous_readout=true, validated above
            self.model = None
            vllm_settings: dict[str, Any] = dict(rc.get("vllm_settings") or {})
            vllm_settings.setdefault("model_name", model_name)
            vllm_settings.setdefault("max_model_len", self.max_length)
            vllm_settings.setdefault("dtype", str(rc.get("dtype", "auto")))
            # Pass tokenizer so the engine pre-tokenizes with add_special_tokens=False,
            # matching HFLogitSkeletonEngine bit-for-bit (see Qwen3Reranker docstring).
            vllm_settings["tokenizer"] = self.tokenizer
            vllm_settings.setdefault("max_length", self.max_length)
            apply_vllm_lora_settings(self, rc, vllm_settings)
            self.engine = make_engine("vllm", **vllm_settings)

        # Cross-query batching only meaningful for the continuous-readout path
        # (engine abstraction). JSON-generation path falls back to per-query
        # base-class iteration: same pattern as Qwen3Reranker.
        self.supports_query_batching = self.continuous_readout

    # ------------------------------------------------------------------
    # Expected-grade readout: continuous scoring
    # ------------------------------------------------------------------

    def _build_chunk_grade_prefixes(self, query: Query, chunk: list[Passage]) -> list[str]:
        _full, slot_prefixes = _build_gemma4_grade_skeleton_prefixes(
            self.tokenizer,
            self.instruction,
            query["text"],
            chunk,
            max_doc_chars=self.max_doc_chars,
            dummy_grade=getattr(self, "grade_skeleton_dummy", _GEMMA4_GRADE_DUMMY),
            prefix_mode=getattr(self, "grade_prefix_mode", "cumulative"),
            chat_template_kwargs=getattr(self, "chat_template_kwargs", None),
            grade_rubric_id=getattr(self, "grade_rubric_id", None),
        )
        return slot_prefixes

    def _score_chunk_grades_continuous(self, query: Query, chunk: list[Passage]) -> list[float]:
        vectors = self._score_chunk_grade_probabilities(query, chunk)
        self._last_grade_probabilities_init_order = vectors
        return self._scores_from_grade_probabilities(vectors)

    def _score_chunk_grade_probabilities(self, query: Query, chunk: list[Passage]) -> list[list[float]]:
        if self.grade_token_ids is None or self.engine is None:
            raise RuntimeError(
                "continuous_readout requested but grade_token_ids/engine were "
                "not initialised — internal invariant violated."
            )
        slot_prefixes = self._build_chunk_grade_prefixes(query, chunk)
        return self.engine.score_prefix_probabilities(slot_prefixes, self.grade_token_ids)

    @staticmethod
    def _scores_from_grade_probabilities(vectors: list[list[float]]) -> list[float]:
        return [sum(float(i) * float(p) for i, p in enumerate(vector)) / 3.0 for vector in vectors]

    @staticmethod
    def _grade_probabilities_from_scores(scores: list[float]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for score in scores:
            grade_score = max(0.0, min(3.0, float(score) * 3.0))
            lo = int(grade_score)
            hi = min(3, lo + 1) if grade_score > lo else lo
            probs = [0.0, 0.0, 0.0, 0.0]
            if lo == hi:
                probs[lo] = 1.0
            else:
                frac = grade_score - lo
                probs[lo] = 1.0 - frac
                probs[hi] = frac
            vectors.append(probs)
        return vectors

    # ------------------------------------------------------------------
    # Path A: JSON grade generation + parse
    # ------------------------------------------------------------------

    def _generate_setwise_scores(self, query: Query, chunk: list[Passage]) -> list[float]:
        if self.model is None:
            raise RuntimeError(
                "This scoring path needs the local HF model. The instance is "
                "vLLM-backed (reranker.backend: vllm), which scores through its engine."
            )
        import torch

        user_body = _build_grade_json_user_body(
            self.instruction, query["text"], chunk, max_doc_chars=self.max_doc_chars
        )
        messages = [
            {"role": "system", "content": _SETWISE_SYSTEM_PROMPT},
            {"role": "user", "content": user_body},
        ]
        prompt = _apply_gemma4_chat_template(self.tokenizer, messages)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )
        for key in inputs:
            inputs[key] = inputs[key].to(self.model.device)

        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": False,
        }
        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if pad_token_id is not None:
            generate_kwargs["pad_token_id"] = pad_token_id
        if eos_token_id is not None:
            generate_kwargs["eos_token_id"] = eos_token_id

        with torch.no_grad():
            generated = self.model.generate(**inputs, **generate_kwargs)

        prompt_len = int(inputs["input_ids"].shape[-1])
        new_tokens = generated[0][prompt_len:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return _parse_setwise_grades(
            text,
            len(chunk),
            missing_grade=self.setwise_missing_grade,
        )

    # ------------------------------------------------------------------
    # Reranker interface
    # ------------------------------------------------------------------

    def rank(self, query: Query, passages: list[Passage]) -> RankResult:
        if not passages:
            return {
                "top_k_psgs": [],
                "scores_init_order": [],
                "grade_probabilities_init_order": [],
                "prompting_runtimes": [0.0],
                "paradigm": self.paradigm,
            }

        t0 = time.perf_counter()
        scores: list[float] = []
        grade_vectors: list[list[float]] | None = [] if self.continuous_readout else None
        for start in range(0, len(passages), self.docs_per_score_forward):
            chunk = passages[start : start + self.docs_per_score_forward]
            if self.continuous_readout:
                self._last_grade_probabilities_init_order = None
                chunk_scores = self._score_chunk_grades_continuous(query, chunk)
                chunk_vectors = self._last_grade_probabilities_init_order
                if not isinstance(chunk_vectors, list) or len(chunk_vectors) != len(chunk_scores):
                    chunk_vectors = self._grade_probabilities_from_scores(chunk_scores)
                assert grade_vectors is not None
                grade_vectors.extend(chunk_vectors)
                scores.extend(chunk_scores)
            else:
                scores.extend(self._generate_setwise_scores(query, chunk))
        elapsed = time.perf_counter() - t0
        result = _scores_to_rank_result(scores, passages, elapsed, self.paradigm, model_name=self.__class__.__name__)
        if grade_vectors is not None:
            result["grade_probabilities_init_order"] = grade_vectors
        return result

    def rank_query_batch(  # noqa: C901
        self,
        items: list[tuple[Query, list[Passage]]],
    ) -> list[RankResult]:
        """Cross-query batched scoring for the continuous-readout path only.

        JSON-generation (Path A) falls back to the base-class per-query iteration
        because ``model.generate`` doesn't fit the engine prefix abstraction.
        Mirrors :class:`Qwen3InstructGradeReranker.rank_query_batch` exactly.
        """
        if not items:
            return []
        if not self.continuous_readout:
            return super().rank_query_batch(items)
        if self.grade_token_ids is None or self.engine is None:
            raise RuntimeError(
                "continuous_readout requested but grade_token_ids/engine were "
                "not initialised — internal invariant violated."
            )

        all_prefixes: list[str] = []
        plan: list[tuple[int, int, int]] = []  # (item_idx, chunk_start, n_in_chunk)
        for item_idx, (query, passages) in enumerate(items):
            if not passages:
                continue
            for start in range(0, len(passages), self.docs_per_score_forward):
                chunk = passages[start : start + self.docs_per_score_forward]
                prefixes = self._build_chunk_grade_prefixes(query, chunk)
                if len(prefixes) != len(chunk):
                    raise RuntimeError(
                        f"prefix builder returned {len(prefixes)} prefixes for chunk of "
                        f"{len(chunk)} passages — invariant violated."
                    )
                plan.append((item_idx, start, len(chunk)))
                all_prefixes.extend(prefixes)

        t0 = time.perf_counter()
        if all_prefixes:
            if hasattr(self.engine, "score_prefix_probabilities"):
                grade_vectors = self.engine.score_prefix_probabilities(all_prefixes, self.grade_token_ids)
                if len(grade_vectors) != len(all_prefixes):
                    raise RuntimeError(
                        f"engine returned {len(grade_vectors)} vectors for {len(all_prefixes)} prefixes; "
                        "request/result count mismatch."
                    )
                unit_scores = self._scores_from_grade_probabilities(grade_vectors)
            else:
                expected_grades = self.engine.score_prefixes(all_prefixes, self.grade_token_ids)
                if len(expected_grades) != len(all_prefixes):
                    raise RuntimeError(
                        f"engine returned {len(expected_grades)} scores for {len(all_prefixes)} prefixes; "
                        "request/result count mismatch."
                    )
                unit_scores = [g / 3.0 for g in expected_grades]
                grade_vectors = self._grade_probabilities_from_scores(unit_scores)
        else:
            grade_vectors = []
            unit_scores = []
        elapsed = time.perf_counter() - t0

        per_item_scores: list[list[float]] = [[0.0] * len(passages) for _, passages in items]
        per_item_vectors: list[list[list[float]]] = [[[] for _ in passages] for _, passages in items]
        cursor = 0
        for item_idx, chunk_start, n in plan:
            per_item_scores[item_idx][chunk_start : chunk_start + n] = unit_scores[cursor : cursor + n]
            per_item_vectors[item_idx][chunk_start : chunk_start + n] = grade_vectors[cursor : cursor + n]
            cursor += n

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
            result = _scores_to_rank_result(
                scores,
                passages,
                per_item_elapsed,
                self.paradigm,
                model_name=self.__class__.__name__,
            )
            result["grade_probabilities_init_order"] = vectors
            results.append(result)
        return results
