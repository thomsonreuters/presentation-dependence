"""Skywork-Reward-V2 as an in-family Bradley-Terry (BT) scalar reward model.

Model: ``Skywork/Skywork-Reward-V2-Qwen3-4B`` (Apache-2.0). Evaluation only.

This pointwise response-ranking anchor uses the same qids, gold labels, and
nDCG@1 metric as the other baselines, for a direct comparison with a
purpose-built scalar reward model.

Inference-only pointwise reward-model anchor, reported only on
the response-quality axis.

Each ``(prompt, response)`` is
scored independently via a sequence-classification head (``num_labels=1``); the
scalar logit is the reward. With no cross-candidate attention, the scorer is
order-invariant by construction and is excluded from the τ-PSI axis.

Model-card contract (Skywork/Skywork-Reward-V2-*), pinned here:
  * ``AutoModelForSequenceClassification(..., num_labels=1)``: scalar reward.
  * Apply the model's **chat template** to ``[{user: prompt}, {assistant:
    response}]`` with ``tokenize=False``, **no system prompt**.
  * The card strips a leading duplicate BOS: the chat template already emits
    BOS, so a second one from the tokenizer would double it. We replicate the
    exact ``if tokenizer.bos_token and s.startswith(bos): s = s[len(bos):]``
    guard, then tokenize with ``add_special_tokens=True`` (BOS restored once).
  * Reward = ``model(**inputs).logits[0][0]``.

Required config keys under ``reranker:``:
  class: SkyworkBTReranker
  model_name: Skywork/Skywork-Reward-V2-Qwen3-4B
  device: "auto" | "cuda" | "cpu" | "cuda:<n>"   (default "auto")
  dtype: "auto" | "bfloat16" | "float16" | "float32"  (default "auto")
  batch_size: int                                 (default 8)
  max_length: int                                 (default 4096)

Optional:
  max_doc_chars: int   # pre-truncate very long responses (chars) before tokenizing
  attn_implementation: str  # e.g. "flash_attention_2" (card recommends on CUDA)
  revision: str        # pin the HF commit SHA before publishing a number
"""

from __future__ import annotations

import time

from presentation_dependence.rerankers._rank_result import scores_to_rank_result
from presentation_dependence.rerankers._torch_utils import resolve_device as _resolve_device
from presentation_dependence.rerankers._torch_utils import resolve_dtype as _resolve_dtype
from presentation_dependence.rerankers.base import Passage, Query, RankResult, Reranker
from presentation_dependence.utils.setup_logging import setup_logging


def build_bt_conversation(prompt: str, response: str, *, max_doc_chars: int | None = None) -> list[dict]:
    """Return the ``[user, assistant]`` conversation for one candidate.

    No system prompt (per the Skywork card). ``max_doc_chars`` optionally
    truncates the response text before tokenization so pathological long
    candidates don't blow the context budget; the prompt is left intact.
    """
    resp = str(response)
    if max_doc_chars is not None and max_doc_chars > 0:
        resp = resp[:max_doc_chars]
    return [
        {"role": "user", "content": str(prompt)},
        {"role": "assistant", "content": resp},
    ]


def strip_leading_bos(formatted: str, bos_token: str | None) -> str:
    """Drop a single leading BOS so tokenization doesn't double it.

    Mirrors the Skywork model-card guard exactly: the chat template already
    includes the BOS, so we remove it from the rendered string and let the
    tokenizer add it back once via ``add_special_tokens=True``.
    """
    if bos_token and formatted.startswith(bos_token):
        return formatted[len(bos_token) :]
    return formatted


class SkyworkBTReranker(Reranker):
    paradigm = "pointwise"

    def __init__(self, config: dict):
        super().__init__(config)
        self.logger = setup_logging(self.__class__.__name__, config)

        rc = self.reranker_config
        self.model_name = rc.get("model_name", "Skywork/Skywork-Reward-V2-Qwen3-4B")
        requested_device = rc.get("device", "auto")
        self.device = _resolve_device(requested_device)
        self.dtype = _resolve_dtype(rc.get("dtype", "auto"), self.device)
        self.batch_size = int(rc.get("batch_size", 8))
        if self.batch_size <= 0:
            raise ValueError("reranker.batch_size must be positive")
        self.max_length = int(rc.get("max_length", 4096))
        max_doc_chars = rc.get("max_doc_chars")
        self.max_doc_chars = int(max_doc_chars) if max_doc_chars is not None else None
        self.attn_implementation = rc.get("attn_implementation")
        self.revision = rc.get("revision")

        self.logger.info(
            "Loading %s on device=%s dtype=%s (BT scalar RM anchor)",
            self.model_name,
            self.device,
            self.dtype,
        )
        if self.revision:
            self.logger.info("Pinned HF revision=%s", self.revision)

        import torch  # noqa: F401  (imported for side-effect parity / clarity)
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tok_kwargs: dict = {"revision": self.revision} if self.revision else {}
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, **tok_kwargs)

        model_kwargs: dict = {
            "num_labels": 1,
            "torch_dtype": self.dtype,
        }
        if self.attn_implementation:
            model_kwargs["attn_implementation"] = self.attn_implementation
        if self.revision:
            model_kwargs["revision"] = self.revision
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, **model_kwargs)
        self.model.to(self.device)
        self.model.eval()

    def _format(self, prompt: str, response: str) -> str:
        conv = build_bt_conversation(prompt, response, max_doc_chars=self.max_doc_chars)
        formatted = self.tokenizer.apply_chat_template(conv, tokenize=False)
        return strip_leading_bos(formatted, getattr(self.tokenizer, "bos_token", None))

    def _score_batch(self, texts: list[str]) -> list[float]:
        """Score preformatted conversations and return one reward per item."""
        import torch

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits  # (B, 1)
        return [float(x) for x in logits.squeeze(-1).float().tolist()]

    def rank(self, query: Query, passages: list[Passage]) -> RankResult:
        if not passages:
            return {
                "top_k_psgs": [],
                "scores_init_order": [],
                "prompting_runtimes": [0.0],
                "paradigm": self.paradigm,
            }

        prompt = query["text"]
        formatted = [self._format(prompt, p["text"]) for p in passages]

        t0 = time.perf_counter()
        scores: list[float] = []
        for start in range(0, len(formatted), self.batch_size):
            scores.extend(self._score_batch(formatted[start : start + self.batch_size]))
        elapsed = time.perf_counter() - t0

        return scores_to_rank_result(
            scores,
            passages,
            elapsed,
            self.paradigm,
            model_name=self.model_name,
        )
