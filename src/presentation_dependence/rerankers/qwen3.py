r"""Qwen3-Reranker-4B official yes/no-logit scorer.

Model: ``Qwen/Qwen3-Reranker-4B`` (Apache-2.0, 32k context).

Implementation source
---------------------

The scoring code mirrors the official Hugging Face model card's
``Using Transformers`` snippet for ``Qwen/Qwen3-Reranker-4B``:

1. Format each pair as
   ``<Instruct>: ...\n<Query>: ...\n<Document>: ...``.
2. Add the official system prefix and assistant ``<think>`` suffix.
3. Run ``AutoModelForCausalLM``.
4. Read final-token logits for the literal ``"yes"`` and ``"no"`` tokens.
5. Return ``P(yes | {yes,no})`` by applying log-softmax over those two logits.

The SentenceTransformers wrapper reports raw logit differences by default;
the official Transformers snippet reports probabilities. We default to the
Transformers output because it is bounded, directly usable as a native scalar
for score-variance PSI, and is the path Qwen documents for custom inference.
Set ``score_output: logit_diff`` to mirror SentenceTransformers-style raw
differences instead.
"""

from __future__ import annotations

import json
import random
import re
import time
from typing import Any

from presentation_dependence.rerankers._torch_utils import resolve_device as _resolve_device
from presentation_dependence.rerankers._torch_utils import resolve_dtype as _resolve_dtype
from presentation_dependence.rerankers._rank_result import scores_to_rank_result as _scores_to_rank_result
from presentation_dependence.rerankers.base import (
    Passage,
    Query,
    RankResult,
    Reranker,
    apply_vllm_lora_settings,
    parse_lora_config,
)
from presentation_dependence.self_distill.engines import HFLogitSkeletonEngine, LogitSkeletonEngine, make_engine
from presentation_dependence.self_distill.engines.hf import last_real_token_positions
from presentation_dependence.self_distill.readout import resolve_grade_token_ids as _resolve_grade_token_ids
from presentation_dependence.utils.setup_logging import setup_logging


_DEFAULT_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"
_SYSTEM_PROMPT = (
    "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
    'Note that the answer can only be "yes" or "no".'
)
_PREFIX = f"<|im_start|>system\n{_SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n"
_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

_SETWISE_SYSTEM_PROMPT = (
    "You are a reranker that scores each numbered document for one query. "
    "Return only document scores. Do not explain or rank the documents."
)
_SETWISE_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def _infer_model_size(model_name: str) -> str | None:
    """Infer Qwen3 reranker size labels from the HF repo name when possible."""
    lowered = model_name.lower()
    for size in ("0.6b", "4b", "8b"):
        if f"reranker-{size}" in lowered or lowered.endswith(size):
            return size
    return None


def format_instruction(instruction: str | None, query: str, doc: str) -> str:
    """Official Qwen3 reranker pair body from the model card."""
    inst = instruction or _DEFAULT_INSTRUCTION
    return f"<Instruct>: {inst}\n<Query>: {query}\n<Document>: {doc}"


# Yes/No setwise dummy answer: same constant as gemma.py so the tokenizer-
# agnostic offset-mapping helpers in that module work for Qwen3 too.
_QWEN3_YESNO_DUMMY = "Yes"


def _build_qwen3_setwise_yesno_prompt_prefixes(
    instruction: str | None,
    query: str,
    passages: list[Passage],
    *,
    max_doc_chars: int,
) -> tuple[str, list[str]]:
    r"""Build shared-context prompt + slot prefixes for Qwen3 yes/no logit scoring.

    Wraps the body in Qwen3's chat template so the dummy ``[i] Relevant: Yes``
    lines land in the assistant section (where the model would normally generate
    its answer). The slot prefix for doc *i* ends immediately before that doc's
    dummy ``Yes``, so the caller reads the yes/no logit at the last token of the
    prefix in a single forward pass.

    Layout::

        <|im_start|>system\n{system}<|im_end|>
        <|im_start|>user
        <Instruct>: ...
        <Query>: ...

        <Documents>:
        [1] text1
        [2] text2

        For each document, determine whether it contains an answer to the query
        (Yes or No):
        <|im_end|>
        <|im_start|>assistant
        <think>\n\n</think>

        [1] Relevant: Yes    ← slot 1 prefix ends just before this Yes
        [2] Relevant: Yes    ← slot 2 prefix includes slot 1's Yes, ends before this one
    """
    inst = instruction or _DEFAULT_INSTRUCTION
    doc_lines: list[str] = []
    for i, passage in enumerate(passages, start=1):
        text = " ".join(str(passage["text"]).split())
        if max_doc_chars > 0:
            text = text[:max_doc_chars]
        doc_lines.append(f"[{i}] {text}")

    user_body = (
        f"<Instruct>: {inst}\n"
        f"<Query>: {query}\n\n"
        "<Documents>:\n" + "\n".join(doc_lines) + "\n\n"
        "For each document, determine whether it contains an answer to the query "
        "(Yes or No):"
    )
    chat_prefix = (
        f"<|im_start|>system\n{_SETWISE_SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_body}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )

    prompt = chat_prefix
    prefixes: list[str] = []
    for i in range(1, len(passages) + 1):
        slot_prefix = f"[{i}] Relevant: "
        prompt += slot_prefix + _QWEN3_YESNO_DUMMY + "\n"
        prefixes.append(prompt[: -(len(_QWEN3_YESNO_DUMMY) + 1)])
    return prompt, prefixes


def _build_setwise_prompt(
    instruction: str | None,
    query: str,
    passages: list[Passage],
    *,
    max_doc_chars: int,
    output_mode: str = "score",
) -> str:
    """Build the shared-context batched-pointwise prompt.

    This is separate from the official pairwise template. The batched-pointwise
    path asks one
    decoder context to see B documents and emit B
    per-doc scalars. The parser below then maps those scalars back to input
    order for score-variance PSI and PA-GRPO reward plumbing.
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
        "Evaluate every document independently for relevance to the query, while comparing documents in the batch.\n"
    )
    if output_mode == "grade":
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
    else:
        instructions = (
            "Use comparable graded scores, not just binary labels:\n"
            "- 1.00 = directly and completely answers the query\n"
            "- 0.75 = strongly relevant but incomplete\n"
            "- 0.50 = partially relevant or topical background\n"
            "- 0.25 = weakly related\n"
            "- 0.00 = not relevant\n"
            "Use intermediate decimal values when useful, and avoid ties unless documents are genuinely equivalent.\n"
            "Return exactly one JSON object and no other text after the assistant prefix.\n"
            "The JSON object must have this shape: "
            '{"scores":[{"id":<document_id>,"score":<number_between_0_and_1>}]}.\n'
            "Use every id from 1 to the number of documents exactly once, including documents scored 0.00. "
            "Scores must be numbers from 0.00 to 1.00."
        )
    body = common + instructions
    return f"<|im_start|>system\n{_SETWISE_SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{body}{_SETWISE_SUFFIX}"


_QWEN3_GRADE_SKELETON_DUMMY = "0"


def _build_qwen3_setwise_grade_skeleton_prefixes(
    instruction: str | None,
    query: str,
    passages: list[Passage],
    *,
    max_doc_chars: int,
    dummy_grade: str = _QWEN3_GRADE_SKELETON_DUMMY,
) -> tuple[str, list[str]]:
    """Build a fixed-grade answer skeleton + per-slot prefixes for continuous readout.

    Companion to :func:`_build_setwise_prompt` (output_mode="grade") but
    structured so we can read the model's logits over ``{0, 1, 2, 3}`` at
    deterministic positions — no JSON generation, no parsing, single forward
    pass.

    Used only when ``Qwen3Reranker`` is configured with
    ``continuous_readout: true`` + ``scoring_mode: setwise_grade_prompt``.

    Layout (in the assistant section, after the ``<think>`` block)::

        Grades:
        [1] Grade: 0
        [2] Grade: 0
        ...
        [B] Grade: 0

    Each slot prefix ends immediately before that slot's dummy ``0`` so the
    caller can read the grade-token logits at the last token of the prefix in
    a single batched forward pass.
    """
    inst = instruction or _DEFAULT_INSTRUCTION
    doc_lines: list[str] = []
    for i, passage in enumerate(passages, start=1):
        text = " ".join(str(passage["text"]).split())
        if max_doc_chars > 0:
            text = text[:max_doc_chars]
        doc_lines.append(f"[{i}] {text}")

    user_body = (
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
    chat_prefix = (
        f"<|im_start|>system\n{_SETWISE_SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_body}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
        "Grades:\n"
    )

    prompt = chat_prefix
    prefixes: list[str] = []
    dummy = str(dummy_grade)
    if not dummy:
        raise ValueError("dummy_grade must not be empty")
    for i in range(1, len(passages) + 1):
        slot_prefix = f"[{i}] Grade: "
        prompt += slot_prefix + dummy + "\n"
        # End the prefix immediately before the dummy "0" + the trailing "\n",
        # so the next token logits at the prefix tail correspond to the grade
        # digit.
        prefixes.append(prompt[: -(len(dummy) + 1)])
    return prompt, prefixes


def _coerce_unit_score(value: Any) -> float:
    score = float(value)
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return score


def _parse_doc_id_key(key: Any) -> int:
    """Parse practical JSON object keys like ``"1"``, ``"#1"``, ``"doc 1"``, or ``"[I1]"``."""
    text = str(key).strip()
    match = re.fullmatch(
        r"(?:#|\[)?(?:(?:doc(?:ument)?|item|i)[_\s-]*)?(\d+)\]?",
        text,
        flags=re.IGNORECASE,
    )
    if match is None:
        raise ValueError(f"Could not parse document id key {key!r}")
    return int(match.group(1))


def _parse_setwise_scores(  # noqa: C901
    output: str,
    num_docs: int,
    *,
    missing_score: float | None = None,
) -> list[float]:
    """Parse setwise JSON-ish output into input-order scores.

    Accepted official shape for our prompt:

        {"scores": [{"id": 1, "score": 0.9}, ...]}

    The parser also accepts two practical variants that LLMs often emit under
    greedy decoding: ``{"1": 0.9, "2": 0.1}`` and a bare score list. Anything
    else raises, so benchmark runs do not silently publish malformed output.
    """
    text = output.strip()
    if not text:
        raise ValueError("Qwen3 setwise output was empty")

    payload: Any | None = None
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end >= start:
        try:
            payload = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            payload = None

    scores_by_id: dict[int, float] = {}
    if isinstance(payload, dict):
        raw_scores = payload.get("scores", payload)
        if isinstance(raw_scores, list):
            for pos, item in enumerate(raw_scores, start=1):
                if isinstance(item, dict):
                    idx = _parse_doc_id_key(item.get("id", item.get("doc_id", pos)))
                    raw_score = item.get("score", item.get("relevance_score"))
                    if raw_score is None:
                        continue
                    score = _coerce_unit_score(raw_score)
                else:
                    idx = pos
                    score = _coerce_unit_score(item)
                scores_by_id[idx] = score
        elif isinstance(raw_scores, dict):
            for key, value in raw_scores.items():
                idx = _parse_doc_id_key(key)
                scores_by_id[idx] = _coerce_unit_score(value)
    elif isinstance(payload, list):
        for pos, value in enumerate(payload, start=1):
            scores_by_id[pos] = _coerce_unit_score(value)

    if not scores_by_id:
        pair_pattern = re.compile(
            r'"?id"?\s*:\s*"?(?P<id>\d+)"?\s*,\s*"?(?:score|relevance_score)"?\s*:\s*"?(?P<score>[+-]?(?:\d+(?:\.\d*)?|\.\d+))"?',
            flags=re.IGNORECASE,
        )
        for match in pair_pattern.finditer(text):
            scores_by_id[int(match.group("id"))] = _coerce_unit_score(match.group("score"))

    if not scores_by_id:
        pattern = re.compile(
            r'(?:\[(?P<bracket>\d+)\]|"?(?P<plain>\d+)"?)\s*[:=]\s*'
            r"(?P<score>[+-]?(?:\d+(?:\.\d*)?|\.\d+))"
        )
        for match in pattern.finditer(text):
            idx = int(match.group("bracket") or match.group("plain"))
            scores_by_id[idx] = _coerce_unit_score(match.group("score"))

    expected = set(range(1, num_docs + 1))
    got = set(scores_by_id)
    if missing_score is not None:
        fill = _coerce_unit_score(missing_score)
        for idx in expected - got:
            scores_by_id[idx] = fill
        got = set(scores_by_id)
    if got != expected:
        missing = sorted(expected - got)
        extra = sorted(got - expected)
        raise ValueError(
            "Qwen3 setwise output did not contain exactly one score per document "
            f"(missing={missing}, extra={extra}, output={output!r})"
        )
    return [scores_by_id[i] for i in range(1, num_docs + 1)]


def _grade_to_score(value: Any) -> float:
    grade = int(float(value))
    if grade < 0:
        grade = 0
    if grade > 3:
        grade = 3
    return grade / 3.0


def _parse_setwise_grades(  # noqa: C901
    output: str,
    num_docs: int,
    *,
    missing_grade: int | float | None = None,
) -> list[float]:
    """Parse fixed relevance grades (0..3) and map them to [0, 1] scores."""
    text = output.strip()
    if not text:
        raise ValueError("Qwen3 setwise grade output was empty")
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()

    payload: Any | None = None
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end >= start:
        try:
            payload = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            payload = None

    scores_by_id: dict[int, float] = {}
    if isinstance(payload, dict):
        raw_grades = payload.get("grades", payload.get("scores", payload))
        if isinstance(raw_grades, list):
            pending_id: int | None = None
            for pos, item in enumerate(raw_grades, start=1):
                if isinstance(item, dict):
                    has_id = "id" in item or "doc_id" in item
                    idx = _parse_doc_id_key(item.get("id", item.get("doc_id", pos)))
                    raw_grade = item.get("grade", item.get("relevance_grade", item.get("score")))
                    if raw_grade is None:
                        pending_id = idx if has_id else None
                        continue
                    if not has_id and pending_id is not None:
                        idx = pending_id
                    score = _grade_to_score(raw_grade)
                    pending_id = None
                else:
                    idx = pos
                    score = _grade_to_score(item)
                    pending_id = None
                scores_by_id[idx] = score
        elif isinstance(raw_grades, dict):
            for key, value in raw_grades.items():
                idx = _parse_doc_id_key(key)
                scores_by_id[idx] = _grade_to_score(value)
    elif isinstance(payload, list):
        for pos, value in enumerate(payload, start=1):
            scores_by_id[pos] = _grade_to_score(value)

    if not scores_by_id:
        pair_pattern = re.compile(
            r'"?id"?\s*:\s*"?(?P<id>\d+)"?\s*,\s*"?(?:grade|relevance_grade|score)"?\s*:\s*"?(?P<grade>[+-]?(?:\d+(?:\.\d*)?|\.\d+))"?',
            flags=re.IGNORECASE,
        )
        for match in pair_pattern.finditer(text):
            scores_by_id[int(match.group("id"))] = _grade_to_score(match.group("grade"))

    expected = set(range(1, num_docs + 1))
    # Some Qwen3 generations continue the id sequence past the current chunk
    # (for example ids 21..100 for a 20-document prompt). Those extras are
    # harmless if every expected id is present, so drop them before validation.
    scores_by_id = {idx: score for idx, score in scores_by_id.items() if idx in expected}
    got = set(scores_by_id)
    if missing_grade is not None:
        fill = _grade_to_score(missing_grade)
        for idx in expected - got:
            scores_by_id[idx] = fill
        got = set(scores_by_id)
    if got != expected:
        missing = sorted(expected - got)
        extra = sorted(got - expected)
        raise ValueError(
            "Qwen3 setwise grade output did not contain exactly one grade per document "
            f"(missing={missing}, extra={extra}, output={output!r})"
        )
    return [scores_by_id[i] for i in range(1, num_docs + 1)]


def _yes_probability_from_logits(logits, *, token_true_id: int, token_false_id: int):
    """Official two-token log-softmax: ``P(yes | {yes,no})``."""
    import torch

    true_vector = logits[:, token_true_id]
    false_vector = logits[:, token_false_id]
    two_class = torch.stack([false_vector, true_vector], dim=1)
    return torch.nn.functional.log_softmax(two_class, dim=1)[:, 1].exp()


def _logit_diff_from_logits(logits, *, token_true_id: int, token_false_id: int):
    """SentenceTransformers-style raw score: ``logit(yes) - logit(no)``."""
    return logits[:, token_true_id] - logits[:, token_false_id]


class Qwen3Reranker(Reranker):
    """Qwen3-Reranker wrapper using the official yes/no output interface."""

    paradigm = "batched_pointwise"

    def __init__(self, config: dict):  # noqa: C901
        super().__init__(config)
        self.logger = setup_logging(self.__class__.__name__, config)

        rc = self.reranker_config
        model_name = rc.get("model_name", "Qwen/Qwen3-Reranker-4B")
        self.model_size = str(rc.get("model_size") or _infer_model_size(model_name) or "unknown")
        requested_device = rc.get("device", "auto")
        self.device = _resolve_device(requested_device)
        dtype = _resolve_dtype(rc.get("dtype", "auto"), self.device)
        revision = rc.get("revision")

        self.instruction = rc.get("instruction", _DEFAULT_INSTRUCTION)
        self.max_length = int(rc.get("max_length", 8192))
        self.docs_per_score_forward = int(rc.get("docs_per_score_forward", 20))
        if self.docs_per_score_forward <= 0:
            raise ValueError("reranker.docs_per_score_forward must be positive")
        self.batch_size = int(rc.get("batch_size", self.docs_per_score_forward))
        if self.batch_size <= 0:
            raise ValueError("reranker.batch_size must be positive")
        self.scoring_mode = str(rc.get("scoring_mode", "official_pairwise")).strip().lower()
        if self.scoring_mode not in {
            "official_pairwise",
            "setwise_prompt",
            "setwise_grade_prompt",
            "setwise_yesno_logit",
        }:
            raise ValueError(
                "reranker.scoring_mode must be one of: official_pairwise, setwise_prompt, "
                "setwise_grade_prompt, setwise_yesno_logit"
            )
        self.score_output = str(rc.get("score_output", "probability")).strip().lower()
        if self.score_output not in {"probability", "logit_diff", "yes_logit"}:
            raise ValueError("reranker.score_output must be one of: probability, logit_diff, yes_logit")
        # Sub-mode for ``setwise_yesno_logit``: which forward strategy to use.
        # "single_forward" (Route 2) is the validated stronger path;
        # "batched_prefixes" (Route 1) is cheaper
        # per-call but B forwards. "auto" tries Route 2 then falls back to Route 1
        # if BPE/SP token boundary alignment fails.
        self.setwise_yesno_forward = str(rc.get("setwise_yesno_forward", "single_forward")).strip().lower()
        if self.setwise_yesno_forward not in {"batched_prefixes", "single_forward", "auto"}:
            raise ValueError("reranker.setwise_yesno_forward must be one of: batched_prefixes, single_forward, auto")
        self.max_doc_chars = int(rc.get("max_doc_chars", 1200))
        self.max_new_tokens = int(rc.get("max_new_tokens", 1024))
        self.grade_skeleton_dummy = str(rc.get("grade_skeleton_dummy", _QWEN3_GRADE_SKELETON_DUMMY))
        if not self.grade_skeleton_dummy:
            raise ValueError("reranker.grade_skeleton_dummy must not be empty")
        self.setwise_missing_score = rc.get("setwise_missing_score")
        self.setwise_missing_grade = rc.get("setwise_missing_grade")
        self.self_consistency_config = config.get("self_consistency") or {}

        # Self-distill continuous-readout mode.
        # OFF by default so existing integer-grade behaviour stays bit-for-bit
        # identical for non-self-distill experiments. Only meaningful for
        # ``setwise_grade_prompt``: in that mode the standard path generates
        # JSON and parses integers; this flag swaps in a fixed grade-skeleton
        # prompt + single-forward expected-grade readout over ``{0,1,2,3}``,
        # so K-shot averaging of continuous values is well defined.
        self.continuous_readout = bool(rc.get("continuous_readout", False))
        if self.continuous_readout and self.scoring_mode != "setwise_grade_prompt":
            raise ValueError(
                "reranker.continuous_readout=true is only valid when "
                f"scoring_mode='setwise_grade_prompt' (got {self.scoring_mode!r})"
            )

        # Inference-engine selector for the continuous-readout path. Default
        # 'hf' keeps existing behaviour identical for any non-vLLM caller;
        # 'vllm' opts into the vLLM backend and skips the HF model load
        # entirely (vLLM owns the GPU instead). Only meaningful when
        # ``continuous_readout=true``: the integer-grade JSON-generation
        # path stays HF-only because vLLM doesn't help generation-bound
        # workloads at our prompt sizes.
        self.inference_engine_kind = str(rc.get("inference_engine", "hf")).strip().lower()
        if self.inference_engine_kind not in {"hf", "vllm"}:
            raise ValueError(
                f"reranker.inference_engine must be one of 'hf', 'vllm' (got {self.inference_engine_kind!r})"
            )
        if self.inference_engine_kind == "vllm" and not self.continuous_readout:
            raise ValueError(
                "reranker.inference_engine='vllm' requires "
                "reranker.continuous_readout=true (vLLM is wired only for the "
                "self-distill continuous-readout path)."
            )

        # This wrapper doubles as a classical pointwise reference when
        # docs_per_score_forward=1; the batched-pointwise path keeps the default.
        self.paradigm = "pointwise" if self.docs_per_score_forward == 1 else "batched_pointwise"

        self.logger.info(
            "Loading %s (size=%s) on device=%s, dtype=%s, scoring_mode=%s, docs_per_score_forward=%d, score_output=%s",
            model_name,
            self.model_size,
            self.device,
            dtype,
            self.scoring_mode,
            self.docs_per_score_forward,
            self.score_output,
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
        # HF defaults to truncation_side="right". Official pairwise scoring reads
        # logits at the final sequence position after a fixed suffix; setwise
        # generation also depends on the prompt tail. Right-truncation drops
        # those tail tokens first, so scores become arbitrary. Prefer dropping
        # early context (same pattern as BGE/mxbai setwise slot encodes).
        if hasattr(self.tokenizer, "truncation_side"):
            self.tokenizer.truncation_side = "left"

        self.token_false_id = int(self.tokenizer.convert_tokens_to_ids("no"))
        self.token_true_id = int(self.tokenizer.convert_tokens_to_ids("yes"))
        if self.token_true_id == self.token_false_id or self.token_true_id < 0 or self.token_false_id < 0:
            raise ValueError(
                "Could not resolve distinct Qwen3 yes/no token ids "
                f"(yes={self.token_true_id}, no={self.token_false_id})"
            )

        # Grade-token ids resolved at init time only when continuous_readout is on
        # so we fail fast on tokenizer drift, not silently mid-run.
        if self.continuous_readout:
            self.grade_token_ids = _resolve_grade_token_ids(self.tokenizer)
        else:
            self.grade_token_ids = None

        self.prefix_tokens = self.tokenizer.encode(_PREFIX, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(_SUFFIX, add_special_tokens=False)
        self._pair_token_budget = self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        if self._pair_token_budget <= 0:
            raise ValueError(
                "reranker.max_length is too small for Qwen3 prefix/suffix "
                f"(max_length={self.max_length}, prefix={len(self.prefix_tokens)}, suffix={len(self.suffix_tokens)})"
            )

        # Optional LoRA / PEFT adapter on top of the base model. Used to evaluate
        # self-distill student checkpoints (and any other PEFT-trained adapter)
        # through the same scoring pipeline that runs offshelf evals: the
        # adapter is loaded by ``peft.PeftModel.from_pretrained`` for HF and by
        # vLLM's native LoRA serving (``LoRARequest``) for vLLM. Default null
        # preserves bit-for-bit identical behaviour for offshelf experiments.
        # LoRA config (single ``lora_path`` and/or multi-adapter ``lora_adapters``
        # for shared-base bundles). ``merge_lora`` (HF-only) bakes the adapter into
        # the base weights. Engine wiring + per-surface adapter switching are
        # family-agnostic (base Reranker.set_active_adapter).
        parse_lora_config(self, rc, engine_kind=self.inference_engine_kind)

        if self.inference_engine_kind == "hf":
            from transformers import AutoModelForCausalLM

            model_kwargs: dict[str, Any] = {"dtype": dtype}
            if revision:
                model_kwargs["revision"] = revision
            if "attn_implementation" in rc and rc["attn_implementation"]:
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
        else:  # vllm: only valid alongside continuous_readout=true, validated above
            self.model = None
            vllm_settings: dict[str, Any] = dict(rc.get("vllm_settings") or {})
            vllm_settings.setdefault("model_name", model_name)
            vllm_settings.setdefault("max_model_len", self.max_length)
            vllm_settings.setdefault("dtype", str(rc.get("dtype", "auto")))
            # Pass our tokenizer + max_length so the engine can pre-tokenize
            # with ``add_special_tokens=False`` (matching HFLogitSkeletonEngine
            # exactly). Without this vLLM's text-prompt path silently prepends
            # BOS, shifting the model's view of the prompt: see
            # ``VLLMLogitSkeletonEngine.score_prefixes`` docstring.
            vllm_settings["tokenizer"] = self.tokenizer
            vllm_settings.setdefault("max_length", self.max_length)
            # max_lora_rank (default 16) matches our student LoRA rank
            # (configs/self-distill/*.yaml: r=16); vLLM rejects ranks > this at
            # request time, so it's an explicit knob rather than a guess.
            apply_vllm_lora_settings(self, rc, vllm_settings)
            self.engine = make_engine("vllm", **vllm_settings)

        # Cross-query batching is only meaningful for the continuous-readout
        # path: other scoring modes (official_pairwise, setwise_prompt JSON,
        # setwise_yesno_logit, JSON-grade) either generate text per chunk or
        # use protocol-specific scoring helpers that don't fit the engine
        # abstraction. The base-class default ``rank_query_batch`` (per-query
        # fallback) handles those modes correctly with zero-throughput
        # benefit, which matches their semantics.
        self.supports_query_batching = bool(self.continuous_readout)

    def _rank_single_pass(self, query: Query, passages: list[Passage]) -> list[float]:
        scores: list[float] = []
        for start in range(0, len(passages), self.docs_per_score_forward):
            chunk = passages[start : start + self.docs_per_score_forward]
            if self.scoring_mode == "setwise_grade_prompt" and self.continuous_readout:
                scores.extend(self._score_chunk_grades_continuous(query, chunk))
            elif self.scoring_mode in {"setwise_prompt", "setwise_grade_prompt"}:
                scores.extend(self._generate_setwise_scores(query, chunk))
            elif self.scoring_mode == "setwise_yesno_logit":
                scores.extend(self._score_chunk_yesno_logits(query, chunk))
            else:
                pair_bodies = [
                    format_instruction(self.instruction, query["text"], passage["text"]) for passage in chunk
                ]
                scores.extend(self._score_pair_bodies(pair_bodies))
        return scores

    # ------------------------------------------------------------------
    # setwise_yesno_logit: Path B (Qwen3 sibling of BGE Gemma).
    # Stuff B docs into one prompt, read logit("yes") - logit("no") at
    # synthetic suffix positions. Implementation borrows the tokenizer-
    # agnostic offset-mapping helpers from gemma.py: those work for any
    # fast tokenizer with offset_mapping support, including Qwen3's.
    # ------------------------------------------------------------------

    def _score_chunk_yesno_logits(self, query: Query, chunk: list[Passage]) -> list[float]:
        mode = str(getattr(self, "setwise_yesno_forward", "single_forward")).strip().lower()
        if mode == "batched_prefixes":
            return self._score_chunk_yesno_logits_batched_prefixes(query, chunk)
        try:
            return self._score_chunk_yesno_logits_single_forward(query, chunk)
        except ValueError as exc:
            if mode == "single_forward":
                raise
            self.logger.debug("setwise yes/no single_forward fallback to batched_prefixes: %s", exc)
            return self._score_chunk_yesno_logits_batched_prefixes(query, chunk)

    def _score_chunk_yesno_logits_single_forward(self, query: Query, chunk: list[Passage]) -> list[float]:
        """Route 2: one prompt with all dummy ``[i] Relevant: Yes`` lines; gather
        logits at fixed slot indices in a single forward pass.
        """
        if self.model is None:
            raise RuntimeError(
                "This scoring path needs the local HF model. The instance is "
                "vLLM-backed (reranker.backend: vllm), which scores through its engine."
            )
        import torch
        from presentation_dependence.rerankers._slot_tokenization import prepare_single_forward

        full_prompt, slot_prefixes = _build_qwen3_setwise_yesno_prompt_prefixes(
            self.instruction, query["text"], chunk, max_doc_chars=self.max_doc_chars
        )
        input_ids, positions = prepare_single_forward(self.tokenizer, full_prompt, slot_prefixes, self.max_length)
        batch = torch.tensor([input_ids], dtype=torch.long, device=self.model.device)
        with torch.no_grad():
            logits = self.model(input_ids=batch).logits[0]

        yes_id, no_id = self.token_true_id, self.token_false_id
        out: list[float] = []
        for pos in positions:
            yes_l = logits[pos, yes_id]
            no_l = logits[pos, no_id]
            diff = yes_l - no_l
            if self.score_output == "probability":
                out.append(float(torch.sigmoid(diff).item()))
            elif self.score_output == "yes_logit":
                out.append(float(yes_l.item()))
            else:
                out.append(float(diff.item()))
        return out

    def _score_chunk_yesno_logits_batched_prefixes(self, query: Query, chunk: list[Passage]) -> list[float]:
        """Route 1: B parallel left-truncated prefixes, one batched forward (B rows)."""
        if self.model is None:
            raise RuntimeError(
                "This scoring path needs the local HF model. The instance is "
                "vLLM-backed (reranker.backend: vllm), which scores through its engine."
            )
        import torch
        from presentation_dependence.rerankers._slot_tokenization import tokenize_tail_preserving

        _full_prompt, slot_prefixes = _build_qwen3_setwise_yesno_prompt_prefixes(
            self.instruction, query["text"], chunk, max_doc_chars=self.max_doc_chars
        )
        encoded = [
            tokenize_tail_preserving(
                self.tokenizer,
                prefix,
                max_length=self.max_length,
                add_special_tokens=False,
            )["input_ids"]
            for prefix in slot_prefixes
        ]
        inputs = self.tokenizer.pad(
            [{"input_ids": ids, "attention_mask": [1] * len(ids)} for ids in encoded],
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
        final_logits = logits[row_idx, last_positions, :]
        yes_logits = final_logits[:, self.token_true_id]
        no_logits = final_logits[:, self.token_false_id]
        diff = yes_logits - no_logits
        if self.score_output == "probability":
            scores_t = torch.sigmoid(diff)
        elif self.score_output == "yes_logit":
            scores_t = yes_logits
        else:
            scores_t = diff
        return [float(x) for x in scores_t.detach().cpu().float().tolist()]

    # ------------------------------------------------------------------
    # Self-distill continuous-readout: fixed grade slots + single forward.
    # The HF and vLLM paths read the same grade-token distribution. This
    # HF-transformers path reads logits at slot positions and
    # computes E[g] = Σ g · softmax(logits[grade_ids])[g], then divides by
    # 3 so the reranker's ``scores_init_order`` stays in [0, 1] like every
    # other Qwen3 scoring path. The teacher driver multiplies back to
    # [0, 3] when persisting silver_labels.jsonl.
    # ------------------------------------------------------------------

    def _build_chunk_grade_prefixes(self, query: Query, chunk: list[Passage]) -> list[str]:
        """Build the slot prefixes for one B-doc chunk under continuous_readout."""
        _full_prompt, slot_prefixes = _build_qwen3_setwise_grade_skeleton_prefixes(
            self.instruction,
            query["text"],
            chunk,
            max_doc_chars=self.max_doc_chars,
            dummy_grade=getattr(self, "grade_skeleton_dummy", _QWEN3_GRADE_SKELETON_DUMMY),
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

    def _rank_self_consistency(self, query: Query, passages: list[Passage]) -> list[float]:
        """K-shot setwise aggregation: shuffle, score, then mean per original doc."""
        cfg = self.self_consistency_config
        K = int(cfg.get("K", 1))
        seeds = list(cfg.get("seeds") or list(range(K)))
        if len(seeds) != K:
            raise ValueError(f"self_consistency.seeds ({len(seeds)}) must equal K ({K})")
        aggregate = str(cfg.get("aggregate", "mean_grade")).strip().lower()
        if aggregate not in {"mean_grade", "mean_score"}:
            raise ValueError("self_consistency.aggregate must be mean_grade or mean_score")
        if not cfg.get("shuffle_docs", True):
            raise ValueError("Qwen3 self-consistency currently requires self_consistency.shuffle_docs=true")

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
                    f"Qwen3 self-consistency shot seed={seed} produced {len(shuffled_scores)} "
                    f"scores for {len(shuffled)} passages"
                )
            for (orig_idx, _), score in zip(shuffled, shuffled_scores, strict=True):
                # For grade prompts scores are grade/3; multiplying preserves
                # the requested [0,3] mean-grade scale without changing order.
                value = float(score) * 3.0 if self.scoring_mode == "setwise_grade_prompt" else float(score)
                sums[orig_idx] += value
                counts[orig_idx] += 1
        return [sums[i] / counts[i] if counts[i] else 0.0 for i in range(len(passages))]

    def _process_inputs(self, pair_bodies: list[str]):
        if self.model is None:
            raise RuntimeError(
                "This scoring path needs the local HF model. The instance is "
                "vLLM-backed (reranker.backend: vllm), which scores through its engine."
            )
        inputs = self.tokenizer(
            pair_bodies,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=self._pair_token_budget,
        )
        input_ids = []
        for ids in inputs["input_ids"]:
            input_ids.append(self.prefix_tokens + list(ids) + self.suffix_tokens)

        batch = self.tokenizer.pad(
            {"input_ids": input_ids},
            padding=True,
            return_tensors="pt",
            max_length=self.max_length,
        )
        for key in batch:
            batch[key] = batch[key].to(self.model.device)
        return batch

    def _score_pair_bodies(self, pair_bodies: list[str]) -> list[float]:
        if self.model is None:
            raise RuntimeError(
                "This scoring path needs the local HF model. The instance is "
                "vLLM-backed (reranker.backend: vllm), which scores through its engine."
            )
        import torch

        scores: list[float] = []
        for start in range(0, len(pair_bodies), self.batch_size):
            batch_bodies = pair_bodies[start : start + self.batch_size]
            inputs = self._process_inputs(batch_bodies)
            with torch.no_grad():
                final_logits = self.model(**inputs).logits[:, -1, :]
                if self.score_output == "probability":
                    batch_scores = _yes_probability_from_logits(
                        final_logits,
                        token_true_id=self.token_true_id,
                        token_false_id=self.token_false_id,
                    )
                else:
                    batch_scores = _logit_diff_from_logits(
                        final_logits,
                        token_true_id=self.token_true_id,
                        token_false_id=self.token_false_id,
                    )
            scores.extend(float(x) for x in batch_scores.detach().cpu().tolist())
        return scores

    def _generate_setwise_scores(self, query: Query, chunk: list[Passage]) -> list[float]:
        if self.model is None:
            raise RuntimeError(
                "This scoring path needs the local HF model. The instance is "
                "vLLM-backed (reranker.backend: vllm), which scores through its engine."
            )
        import torch

        prompt = _build_setwise_prompt(
            self.instruction,
            query["text"],
            chunk,
            max_doc_chars=self.max_doc_chars,
            output_mode="grade" if self.scoring_mode == "setwise_grade_prompt" else "score",
        )
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
        if self.scoring_mode == "setwise_grade_prompt":
            return _parse_setwise_grades(
                text,
                len(chunk),
                missing_grade=self.setwise_missing_grade,
            )
        return _parse_setwise_scores(
            text,
            len(chunk),
            missing_score=self.setwise_missing_score,
        )

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
        grade_vectors: list[list[float]] | None = None
        if self.self_consistency_config.get("enabled", False):
            scores = self._rank_self_consistency(query, passages)
        elif self.scoring_mode == "setwise_grade_prompt" and self.continuous_readout:
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

        result = _scores_to_rank_result(scores, passages, elapsed, self.paradigm, model_name=self.__class__.__name__)
        if grade_vectors is not None:
            result["grade_probabilities_init_order"] = grade_vectors
        return result

    def rank_query_batch(self, items):  # noqa: C901
        """Cross-query batched scoring for the continuous-readout path only.

        Other scoring modes (official_pairwise, setwise_prompt JSON,
        setwise_yesno_logit, JSON-grade integer) fall back to the base-class
        per-query iteration via ``Reranker.rank_query_batch`` because their
        scoring helpers don't fit the engine prefix abstraction.

        Implementation mirrors :class:`Qwen3InstructGradeReranker`: build
        slot prefixes for every (query, chunk), submit them in a single
        ``engine.score_prefixes`` call, then unpack results back into
        per-query :class:`RankResult` objects in input order.
        """
        if not items:
            return []
        if not self.continuous_readout:
            # Other scoring modes don't benefit from cross-query batching;
            # the safe behaviour is the base-class per-query fallback so
            # callers don't accidentally route a non-continuous job through
            # an engine path that doesn't support it.
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
