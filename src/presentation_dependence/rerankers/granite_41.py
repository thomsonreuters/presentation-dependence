"""IBM Granite 4.1 8B expected-grade reranker.

Model: ``ibm-granite/granite-4.1-8b`` (Apache-2.0).

Dense decoder-only checkpoint. Defaults align with the Qwen/Gemma-style
expected-grade readout: **cumulative** grade skeleton via :func:`build_chat_grade_skeleton_prefixes`.
Use ``grade_prefix_mode: independent`` if ablations show slot bias.

Readout: expected grade over next-token probabilities for digits ``0..3``.
The Hugging Face engine is the default; tracked paper-scale configs select
vLLM explicitly for cached, batched inference.
"""

from __future__ import annotations

import random
import re
import time
from typing import Any

from presentation_dependence.rerankers._rank_result import scores_to_rank_result
from presentation_dependence.rerankers._torch_utils import resolve_device as _resolve_device
from presentation_dependence.rerankers._torch_utils import resolve_dtype as _resolve_dtype
from presentation_dependence.rerankers._grade_skeleton import (
    apply_chat_template_or_flatten,
    build_chat_grade_skeleton_prefixes,
    build_grade_skeleton_user_body,
)
from presentation_dependence.rerankers.grade_rubrics import resolve_grade_rubric as _resolve_grade_rubric
from presentation_dependence.rerankers.base import (
    Passage,
    Query,
    RankResult,
    Reranker,
    apply_vllm_lora_settings,
    parse_lora_config,
)
from presentation_dependence.rerankers.qwen3 import _parse_setwise_grades
from presentation_dependence.self_distill.engines import HFLogitSkeletonEngine, LogitSkeletonEngine, make_engine
from presentation_dependence.self_distill.readout import resolve_grade_token_ids as _resolve_grade_token_ids
from presentation_dependence.utils.setup_logging import setup_logging


_DEFAULT_MODEL = "ibm-granite/granite-4.1-8b"
_DEFAULT_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"
_DEFAULT_INFERENCE_ENGINE = "hf"
_SYSTEM_PROMPT = (
    "You are a search relevance grader. For each numbered document, output one fixed integer "
    "relevance grade in {0, 1, 2, 3}. Do not rank the documents and do not explain."
)
_YESNO_SYSTEM_PROMPT = (
    "You are a search relevance judge. For each numbered document, answer only whether it is relevant to the query."
)
_YESNO_DUMMY = "Yes"


def _parse_generated_grade_lines(
    output: str,
    num_docs: int,
    *,
    missing_grade: int | float | None = None,
) -> list[float]:
    """Parse generated ``[id] Grade: <0..3>`` lines into unit scores."""
    text = output.split("</think>", 1)[-1].strip() if "</think>" in output else output.strip()
    scores_by_id: dict[int, float] = {}
    line_pattern = re.compile(
        r"(?:^|\n)\s*(?:\[?(?P<id>\d+)\]?)\s*"
        r"(?:Grade)?\s*:?\s*(?P<grade>[0-3])(?=\D|$)",
        flags=re.IGNORECASE,
    )
    for match in line_pattern.finditer(text):
        idx = int(match.group("id"))
        if 1 <= idx <= num_docs:
            scores_by_id[idx] = float(match.group("grade")) / 3.0
    if not scores_by_id:
        return _parse_setwise_grades(text, num_docs, missing_grade=missing_grade)
    expected = set(range(1, num_docs + 1))
    if missing_grade is not None:
        fill = max(0.0, min(3.0, float(missing_grade))) / 3.0
        for idx in expected - set(scores_by_id):
            scores_by_id[idx] = fill
    if set(scores_by_id) != expected:
        missing = sorted(expected - set(scores_by_id))
        raise ValueError(
            "Generated grade output did not contain exactly one grade per "
            f"document (missing={missing}, output={output!r})"
        )
    return [scores_by_id[index] for index in range(1, num_docs + 1)]


def _apply_granite_chat_template(
    tokenizer: Any,
    messages: list[dict[str, str]],
) -> str:
    return apply_chat_template_or_flatten(tokenizer, messages)


def _build_grade_skeleton_user_body(
    instruction: str | None,
    query: str,
    passages: list[Passage],
    *,
    max_doc_chars: int,
    grade_rubric_id: str | None = None,
) -> str:
    return build_grade_skeleton_user_body(
        instruction=instruction,
        default_instruction=_DEFAULT_INSTRUCTION,
        query=query,
        passages=passages,
        max_doc_chars=max_doc_chars,
        extra_instructions="No explanation.",
        grade_rubric_id=grade_rubric_id,
    )


def _build_granite_grade_skeleton_prefixes(
    tokenizer: Any,
    instruction: str | None,
    query: str,
    passages: list[Passage],
    *,
    max_doc_chars: int,
    prefix_mode: str = "cumulative",
    dummy_grade: str = "0",
    grade_rubric_id: str | None = None,
) -> tuple[str, list[str]]:
    """Build the full fixed-grade skeleton and per-slot readout prefixes.

    ``grade_rubric_id`` defaults to ``None``, which produces the same
    passage-relevance body as ``"relevance_v1"``. Set it (e.g.
    ``"response_quality_v1"``) to score a different item type through the
    shared expected-grade rubric builder for response-ranking cross-family
    evaluation.
    """
    mode = str(prefix_mode).strip().lower()
    if mode == "neutral":
        mode = "independent"
    # System prompt tracks the rubric so a non-default grade_rubric_id (e.g.
    # "response_quality_v1") also swaps the grader framing, not just the body.
    # "relevance_v1" (the legacy default student.py always sets on the
    # tokenizer) is treated as "no rubric" to keep the passage-relevance
    # sweep's exact system prompt.
    non_default_rubric = grade_rubric_id is not None and grade_rubric_id != "relevance_v1"
    system_prompt = _resolve_grade_rubric(grade_rubric_id).system_prompt if non_default_rubric else _SYSTEM_PROMPT
    if mode == "cumulative":
        return build_chat_grade_skeleton_prefixes(
            tokenizer=tokenizer,
            system_prompt=system_prompt,
            instruction=instruction,
            default_instruction=_DEFAULT_INSTRUCTION,
            query=query,
            passages=passages,
            max_doc_chars=max_doc_chars,
            extra_instructions="No explanation.",
            dummy_grade=dummy_grade,
            grade_rubric_id=grade_rubric_id,
        )
    if mode != "independent":
        raise ValueError("prefix_mode must be 'independent' or 'cumulative'")

    user_body = _build_grade_skeleton_user_body(
        instruction,
        query,
        passages,
        max_doc_chars=max_doc_chars,
        grade_rubric_id=grade_rubric_id,
    )
    chat_prefix = _apply_granite_chat_template(
        tokenizer,
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_body},
        ],
    )
    full_prompt = chat_prefix + "Grades:\n" + "".join(f"[{i}] Grade: \n" for i in range(1, len(passages) + 1))
    prefixes = [chat_prefix + "Grades:\n" + f"[{i}] Grade: " for i in range(1, len(passages) + 1)]
    return full_prompt, prefixes


def _build_granite_grade_generation_prompt(
    tokenizer: Any,
    instruction: str | None,
    query: str,
    passages: list[Passage],
    *,
    max_doc_chars: int,
) -> str:
    user_body = _build_grade_skeleton_user_body(
        instruction,
        query,
        passages,
        max_doc_chars=max_doc_chars,
    )
    return (
        _apply_granite_chat_template(
            tokenizer,
            [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_body},
            ],
        )
        + "Grades:\n"
    )


def _build_yesno_user_body(
    instruction: str | None,
    query: str,
    passages: list[Passage],
    *,
    max_doc_chars: int,
) -> str:
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
        "For each document, determine whether it contains an answer to the query "
        "(Yes or No)."
    )


def _build_granite_yesno_prompt_prefixes(
    tokenizer: Any,
    instruction: str | None,
    query: str,
    passages: list[Passage],
    *,
    max_doc_chars: int,
) -> tuple[str, list[str]]:
    """Build Path B yes/no prompt + per-slot prefixes for Granite."""
    user_body = _build_yesno_user_body(
        instruction,
        query,
        passages,
        max_doc_chars=max_doc_chars,
    )
    chat_prefix = _apply_granite_chat_template(
        tokenizer,
        [
            {"role": "system", "content": _YESNO_SYSTEM_PROMPT},
            {"role": "user", "content": user_body},
        ],
    )
    full_prompt = chat_prefix
    prefixes: list[str] = []
    for i in range(1, len(passages) + 1):
        slot_prefix = f"[{i}] Relevant: "
        full_prompt += slot_prefix + _YESNO_DUMMY + "\n"
        prefixes.append(full_prompt[: -(len(_YESNO_DUMMY) + 1)])
    return full_prompt, prefixes


def _resolve_yesno_token_ids(tokenizer: Any) -> tuple[list[int], tuple[float, float]]:
    """Resolve single-token No/Yes ids for Path B probability scoring."""
    for yes_text, no_text in ((_YESNO_DUMMY, "No"), ("yes", "no"), (" Yes", " No"), (" yes", " no")):
        yes_ids = tokenizer(yes_text, add_special_tokens=False)["input_ids"]
        no_ids = tokenizer(no_text, add_special_tokens=False)["input_ids"]
        if len(yes_ids) == 1 and len(no_ids) == 1 and int(yes_ids[0]) != int(no_ids[0]):
            return [int(no_ids[0]), int(yes_ids[0])], (0.0, 1.0)
    raise ValueError(
        "Could not resolve distinct single-token Granite Yes/No ids. "
        "Choose explicit tokenizer-compatible alternatives before using scoring_method='yesno_logit'."
    )


class Granite41GradeReranker(Reranker):
    """Granite 4.1 8B prompted as a B-document expected-grade scorer."""

    paradigm = "batched_pointwise"
    supports_query_batching = True

    def __init__(self, config: dict):  # noqa: C901
        """Initialise tokenizer, grade tokens, and the HF or vLLM skeleton engine."""
        super().__init__(config)
        self.logger = setup_logging(self.__class__.__name__, config)

        rc = self.reranker_config
        model_name = str(rc.get("model_name", _DEFAULT_MODEL))
        requested_device = rc.get("device", "auto")
        self.device = _resolve_device(requested_device)
        dtype = _resolve_dtype(rc.get("dtype", "auto"), self.device)
        revision = rc.get("revision")
        trust_remote_code = bool(rc.get("trust_remote_code", False))

        self.instruction = rc.get("instruction", _DEFAULT_INSTRUCTION)
        # None preserves the exact passage-relevance body/system-prompt this class
        # shipped with; set e.g. "response_quality_v1" for the response-ranking arm.
        self.grade_rubric_id = rc.get("grade_rubric_id")
        self.max_length = int(rc.get("max_length", 32768))
        self.docs_per_score_forward = int(rc.get("docs_per_score_forward", 10))
        if self.docs_per_score_forward <= 0:
            raise ValueError("reranker.docs_per_score_forward must be positive")
        self.batch_size = int(rc.get("batch_size", self.docs_per_score_forward))
        if self.batch_size <= 0:
            raise ValueError("reranker.batch_size must be positive")
        self.max_doc_chars = int(rc.get("max_doc_chars", 1200))
        self.max_new_tokens = int(rc.get("max_new_tokens", 512))
        self.scoring_method = str(rc.get("scoring_method", "continuous_grade")).strip().lower()
        if self.scoring_method not in {"continuous_grade", "discrete_grade", "generate_grades", "yesno_logit"}:
            raise ValueError(
                "reranker.scoring_method must be one of "
                "'continuous_grade' (expected-grade readout), 'discrete_grade' (Path A), "
                "'generate_grades', 'yesno_logit' (Path B)"
            )
        self.generated_missing_grade = rc.get("generated_missing_grade")
        self.grade_prefix_mode = str(rc.get("grade_prefix_mode", "cumulative")).strip().lower()
        if self.grade_prefix_mode == "neutral":
            self.grade_prefix_mode = "independent"
        if self.grade_prefix_mode not in {"independent", "cumulative"}:
            raise ValueError("reranker.grade_prefix_mode must be 'independent' or 'cumulative'")
        self.grade_dummy = str(rc.get("grade_dummy", "0")).strip()
        if self.grade_prefix_mode == "cumulative" and not self.grade_dummy:
            raise ValueError("reranker.grade_dummy must not be empty")

        # Path A (discrete_grade) overrides continuous_readout to False; the
        # discrete-grade readout is by definition non-continuous.
        if self.scoring_method in {"discrete_grade", "yesno_logit"}:
            if rc.get("continuous_readout", False) is True:
                raise ValueError(
                    f"scoring_method={self.scoring_method!r} is incompatible with "
                    "continuous_readout=true. Remove continuous_readout or set it to false."
                )
            self.continuous_readout = False
        else:
            if rc.get("continuous_readout", True) is False:
                raise ValueError(
                    "Granite41GradeReranker scoring_method='continuous_grade' requires "
                    "continuous_readout=true. Use scoring_method: discrete_grade for Path A."
                )
            self.continuous_readout = True

        self.self_consistency_config = config.get("self_consistency") or {}
        self.inference_engine_kind = str(rc.get("inference_engine", _DEFAULT_INFERENCE_ENGINE)).strip().lower()
        if self.inference_engine_kind not in {"hf", "vllm"}:
            raise ValueError(
                f"reranker.inference_engine must be one of 'hf', 'vllm' (got {self.inference_engine_kind!r})"
            )

        self.grade_with_space_prefix = bool(rc.get("grade_with_space_prefix", False))

        self.logger.info(
            "Loading %s on device=%s dtype=%s docs_per_score_forward=%d engine=%s",
            model_name,
            self.device,
            dtype,
            self.docs_per_score_forward,
            self.inference_engine_kind,
        )
        self.logger.info(
            "Granite scoring_method=%s grade_prefix_mode=%s grade_dummy=%s grade_with_space_prefix=%s",
            self.scoring_method,
            self.grade_prefix_mode,
            self.grade_dummy,
            self.grade_with_space_prefix,
        )
        if revision:
            self.logger.info("Pinned HF revision=%s", revision)

        from transformers import AutoTokenizer

        tok_kwargs: dict[str, Any] = {"padding_side": "left", "trust_remote_code": trust_remote_code}
        if revision:
            tok_kwargs["revision"] = revision
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **tok_kwargs)
        if getattr(self.tokenizer, "pad_token", None) is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if hasattr(self.tokenizer, "truncation_side"):
            self.tokenizer.truncation_side = "left"

        self.grade_token_ids = None
        self.yesno_token_ids: list[int] | None = None
        self.yesno_values: tuple[float, float] | None = None
        if self.scoring_method == "yesno_logit":
            self.yesno_token_ids, self.yesno_values = _resolve_yesno_token_ids(self.tokenizer)
        else:
            self.grade_token_ids = _resolve_grade_token_ids(
                self.tokenizer,
                with_space_prefix=self.grade_with_space_prefix,
            )
        parse_lora_config(self, rc, engine_kind=self.inference_engine_kind)

        if self.inference_engine_kind == "hf":
            from transformers import AutoModelForCausalLM

            model_kwargs: dict[str, Any] = {
                "dtype": dtype,
                "trust_remote_code": trust_remote_code,
            }
            if revision:
                model_kwargs["revision"] = revision
            if rc.get("attn_implementation"):
                model_kwargs["attn_implementation"] = rc["attn_implementation"]

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
        else:
            self.model = None
            vllm_settings: dict[str, Any] = dict(rc.get("vllm_settings") or {})
            vllm_settings.setdefault("model_name", model_name)
            vllm_settings.setdefault("max_model_len", self.max_length)
            vllm_settings.setdefault("dtype", str(rc.get("dtype", "auto")))
            vllm_settings["tokenizer"] = self.tokenizer
            vllm_settings.setdefault("max_length", self.max_length)

            extra_llm_kwargs = dict(vllm_settings.get("extra_llm_kwargs") or {})
            extra_llm_kwargs.setdefault("trust_remote_code", trust_remote_code)
            vllm_settings["extra_llm_kwargs"] = extra_llm_kwargs
            apply_vllm_lora_settings(self, rc, vllm_settings)
            self.engine = make_engine("vllm", **vllm_settings)

    def _build_chunk_grade_prefixes(self, query: Query, chunk: list[Passage]) -> list[str]:
        _full, prefixes = _build_granite_grade_skeleton_prefixes(
            self.tokenizer,
            self.instruction,
            query["text"],
            chunk,
            max_doc_chars=self.max_doc_chars,
            prefix_mode=self.grade_prefix_mode,
            dummy_grade=self.grade_dummy,
            grade_rubric_id=getattr(self, "grade_rubric_id", None),
        )
        return prefixes

    def _build_chunk_yesno_prefixes(self, query: Query, chunk: list[Passage]) -> list[str]:
        _full, prefixes = _build_granite_yesno_prompt_prefixes(
            self.tokenizer,
            self.instruction,
            query["text"],
            chunk,
            max_doc_chars=self.max_doc_chars,
        )
        return prefixes

    def _score_chunk_grades_continuous(self, query: Query, chunk: list[Passage]) -> list[float]:
        vectors = self._score_chunk_grade_probabilities(query, chunk)
        self._last_grade_probabilities_init_order = vectors
        return self._scores_from_grade_probabilities(vectors)

    def _score_chunk_grade_probabilities(self, query: Query, chunk: list[Passage]) -> list[list[float]]:
        prefixes = self._build_chunk_grade_prefixes(query, chunk)
        return self.engine.score_prefix_probabilities(prefixes, self.grade_token_ids)

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

    def _score_chunk_grades_discrete(self, query: Query, chunk: list[Passage]) -> list[float]:
        """Path A: argmax over {0,1,2,3} grade logprobs at each slot, scaled to [0,1]."""
        if self.grade_token_ids is None:
            raise RuntimeError("Granite grade_token_ids were not initialised")
        prefixes = self._build_chunk_grade_prefixes(query, chunk)
        argmax_grades = self.engine.score_prefixes_argmax_grade(prefixes, self.grade_token_ids)
        return [g / 3.0 for g in argmax_grades]

    def _score_chunk_yesno(self, query: Query, chunk: list[Passage]) -> list[float]:
        """Path B: normalized P(Yes | {No, Yes}) at each slot."""
        if self.yesno_token_ids is None or self.yesno_values is None:
            raise RuntimeError("Granite yes/no token ids were not initialised")
        prefixes = self._build_chunk_yesno_prefixes(query, chunk)
        return self.engine.score_prefixes(prefixes, self.yesno_token_ids, self.yesno_values)

    def _generate_chunk_grades(self, query: Query, chunk: list[Passage]) -> list[float]:
        if self.model is None:
            raise RuntimeError(
                "This scoring path needs the local HF model. The instance is "
                "vLLM-backed (reranker.backend: vllm), which scores through its engine."
            )
        prompt = _build_granite_grade_generation_prompt(
            self.tokenizer,
            self.instruction,
            query["text"],
            chunk,
            max_doc_chars=self.max_doc_chars,
        )
        if self.inference_engine_kind == "vllm":
            from vllm import SamplingParams

            outputs = self.engine.llm.generate(
                [prompt],
                SamplingParams(max_tokens=self.max_new_tokens, temperature=0.0),
                use_tqdm=False,
            )
            text = outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""
        else:
            import torch

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_length,
            )
            for key in inputs:
                inputs[key] = inputs[key].to(self.model.device)
            with torch.no_grad():
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=getattr(self.tokenizer, "pad_token_id", None),
                    eos_token_id=getattr(self.tokenizer, "eos_token_id", None),
                )
            prompt_len = int(inputs["input_ids"].shape[-1])
            text = self.tokenizer.decode(generated[0][prompt_len:], skip_special_tokens=True)
        return _parse_generated_grade_lines(
            text,
            len(chunk),
            missing_grade=self.generated_missing_grade,
        )

    def _rank_single_pass(self, query: Query, passages: list[Passage]) -> list[float]:
        scores: list[float] = []
        for start in range(0, len(passages), self.docs_per_score_forward):
            chunk = passages[start : start + self.docs_per_score_forward]
            if self.scoring_method == "generate_grades":
                scores.extend(self._generate_chunk_grades(query, chunk))
            elif self.scoring_method == "discrete_grade":
                scores.extend(self._score_chunk_grades_discrete(query, chunk))
            elif self.scoring_method == "yesno_logit":
                scores.extend(self._score_chunk_yesno(query, chunk))
            else:
                scores.extend(self._score_chunk_grades_continuous(query, chunk))
        return scores

    def _rank_self_consistency(self, query: Query, passages: list[Passage]) -> list[float]:
        cfg = self.self_consistency_config
        K = int(cfg.get("K", 1))
        seeds = list(cfg.get("seeds") or list(range(K)))
        if len(seeds) != K:
            raise ValueError(f"self_consistency.seeds ({len(seeds)}) must equal K ({K})")
        if not cfg.get("shuffle_docs", True):
            raise ValueError("Granite41GradeReranker self-consistency requires self_consistency.shuffle_docs=true")

        sums = [0.0] * len(passages)
        counts = [0] * len(passages)
        indexed = list(enumerate(passages))
        for seed in seeds:
            shuffled = list(indexed)
            random.Random(int(seed)).shuffle(shuffled)
            shuffled_passages = [p for _, p in shuffled]
            shuffled_scores = self._rank_single_pass(query, shuffled_passages)
            if len(shuffled_scores) != len(shuffled):
                raise ValueError(
                    f"Granite self-consistency seed={seed} produced {len(shuffled_scores)} "
                    f"scores for {len(shuffled)} passages"
                )
            for (orig_idx, _), score in zip(shuffled, shuffled_scores, strict=True):
                sums[orig_idx] += float(score) * 3.0
                counts[orig_idx] += 1
        return [sums[i] / counts[i] if counts[i] else 0.0 for i in range(len(passages))]

    def rank(self, query: Query, passages: list[Passage]) -> RankResult:
        """Score one query's candidate list with an expected-grade readout."""
        if not passages:
            return {
                "top_k_psgs": [],
                "scores_init_order": [],
                "grade_probabilities_init_order": [],
                "prompting_runtimes": [0.0],
                "paradigm": self.paradigm,
            }

        t0 = time.perf_counter()
        grade_vectors: list[list[float]] | None = None
        if self.self_consistency_config.get("enabled", False):
            scores = self._rank_self_consistency(query, passages)
        elif self.scoring_method == "continuous_grade":
            scores = []
            grade_vectors = []
            for start in range(0, len(passages), self.docs_per_score_forward):
                chunk = passages[start : start + self.docs_per_score_forward]
                self._last_grade_probabilities_init_order = None
                chunk_scores = self._score_chunk_grades_continuous(query, chunk)
                chunk_vectors = self._last_grade_probabilities_init_order
                if not isinstance(chunk_vectors, list) or len(chunk_vectors) != len(chunk_scores):
                    chunk_vectors = self._grade_probabilities_from_scores(chunk_scores)
                grade_vectors.extend(chunk_vectors)
                scores.extend(chunk_scores)
        else:
            scores = self._rank_single_pass(query, passages)
        elapsed = time.perf_counter() - t0
        result = scores_to_rank_result(scores, passages, elapsed, self.paradigm, model_name=self.__class__.__name__)
        if grade_vectors is not None:
            result["grade_probabilities_init_order"] = grade_vectors
        return result

    def rank_query_batch(self, items: list[tuple[Query, list[Passage]]]) -> list[RankResult]:  # noqa: C901
        """Score multiple queries by dispatching all grade prefixes together."""
        if not items:
            return []
        if self.self_consistency_config.get("enabled", False) or self.scoring_method == "generate_grades":
            return super().rank_query_batch(items)

        all_prefixes: list[str] = []
        plan: list[tuple[int, int, int]] = []
        for item_idx, (query, passages) in enumerate(items):
            if not passages:
                continue
            for start in range(0, len(passages), self.docs_per_score_forward):
                chunk = passages[start : start + self.docs_per_score_forward]
                prefixes = (
                    self._build_chunk_yesno_prefixes(query, chunk)
                    if self.scoring_method == "yesno_logit"
                    else self._build_chunk_grade_prefixes(query, chunk)
                )
                if len(prefixes) != len(chunk):
                    raise RuntimeError(
                        f"prefix builder returned {len(prefixes)} prefixes for chunk of "
                        f"{len(chunk)} passages — invariant violated."
                    )
                plan.append((item_idx, start, len(chunk)))
                all_prefixes.extend(prefixes)

        t0 = time.perf_counter()
        if all_prefixes:
            if self.scoring_method == "discrete_grade":
                if self.grade_token_ids is None:
                    raise RuntimeError("Granite grade_token_ids were not initialised")
                raw_grades = self.engine.score_prefixes_argmax_grade(all_prefixes, self.grade_token_ids)
                if len(raw_grades) != len(all_prefixes):
                    raise RuntimeError(
                        f"engine returned {len(raw_grades)} scores for {len(all_prefixes)} prefixes; "
                        "request/result count mismatch."
                    )
                grade_vectors = None
                unit_scores = [g / 3.0 for g in raw_grades]
            elif self.scoring_method == "yesno_logit":
                if self.yesno_token_ids is None or self.yesno_values is None:
                    raise RuntimeError("Granite yes/no token ids were not initialised")
                unit_scores = self.engine.score_prefixes(all_prefixes, self.yesno_token_ids, self.yesno_values)
                if len(unit_scores) != len(all_prefixes):
                    raise RuntimeError(
                        f"engine returned {len(unit_scores)} scores for {len(all_prefixes)} prefixes; "
                        "request/result count mismatch."
                    )
                grade_vectors = None
            else:
                if self.grade_token_ids is None:
                    raise RuntimeError("Granite grade_token_ids were not initialised")
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
            grade_vectors = [] if self.scoring_method == "continuous_grade" else None
            unit_scores = []
        elapsed = time.perf_counter() - t0

        per_item_scores: list[list[float]] = [[0.0] * len(passages) for _, passages in items]
        per_item_vectors: list[list[list[float]]] | None = (
            [[[] for _ in passages] for _, passages in items] if grade_vectors is not None else None
        )
        cursor = 0
        for item_idx, chunk_start, n in plan:
            per_item_scores[item_idx][chunk_start : chunk_start + n] = unit_scores[cursor : cursor + n]
            if per_item_vectors is not None and grade_vectors is not None:
                per_item_vectors[item_idx][chunk_start : chunk_start + n] = grade_vectors[cursor : cursor + n]
            cursor += n

        n_nonempty = max(sum(1 for _, p in items if p), 1)
        per_item_elapsed = elapsed / n_nonempty
        results: list[RankResult] = []
        for item_idx, ((_query, passages), scores) in enumerate(zip(items, per_item_scores, strict=True)):
            if not passages:
                result: RankResult = {
                    "top_k_psgs": [],
                    "scores_init_order": [],
                    "prompting_runtimes": [0.0],
                    "paradigm": self.paradigm,
                }
                if per_item_vectors is not None:
                    result["grade_probabilities_init_order"] = []
                results.append(result)
                continue
            result = scores_to_rank_result(
                scores,
                passages,
                per_item_elapsed,
                self.paradigm,
                model_name=self.__class__.__name__,
            )
            if per_item_vectors is not None:
                result["grade_probabilities_init_order"] = per_item_vectors[item_idx]
            results.append(result)
        return results
