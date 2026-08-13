"""mxbai-rerank-large-v2 as a pointwise cross-encoder.

Model: `mixedbread-ai/mxbai-rerank-large-v2` (1.5B, Apache-2.0).

Paradigm: pointwise.
Each (query, passage) pair gets a single scalar relevance
score; the model does not attend across candidates. Consequence: no
within-list position bias to debias, so this model is an R-series reference
rather than a retrofitting target.

The `mxbai-rerank` library exposes a high-level helper; we use it as a
dependency so we don't reimplement the model-specific prompt
formatting. If we later want full control of tokenisation (e.g. for
custom batching at RL-rollout time), we switch to loading the underlying
HF model directly. Training rollouts don't need this wrapper; it is
eval-only.

Required config keys under `reranker:`:
  class: MxbaiPointwise
  model_name: mixedbread-ai/mxbai-rerank-large-v2
  device: "auto" | "cuda" | "cpu" | "cuda:<n>"   (default "auto")
  batch_size: int                                 (default 32)
  max_length: int                                 (default 8192)

Optional:
  revision: str       # HF commit SHA / branch. Pin before publishing a
                      # number; otherwise a silent repo update moves it.
  max_doc_chars: int  # pre-truncate passage text (chars) before scoring.
                      # 0 = disabled (legacy off-shelf). Set per surface
                      # (500 BEIR/legal, 1200 MS MARCO DL) for fair vs ours.

Apache-2.0 source and modification notice:

``_patch_mxbai_rerank_for_tf5`` replaces ``MxbaiRerankV2.prepare_inputs`` from
``mxbai-rerank==0.1.6`` / revision
``c27224ad2cb2622fc7a4260778b8ad9d2a6b6f0a``. Copyright 2025 mixedbread ai
inc. The local method retains upstream query/document tokenization, prompting,
concatenation, and padding, but replaces the removed
``tokenizer.prepare_for_model`` call with explicit Transformers-5-compatible
pair truncation and adds patch-detection guards. Licence text is at
``third_party_licenses/Apache-2.0.txt``; see ``THIRD_PARTY_NOTICES.md``.
The upstream repository has no root ``NOTICE``.
"""

from __future__ import annotations

import time

from presentation_dependence.rerankers._torch_utils import resolve_device as _resolve_device
from presentation_dependence.rerankers.base import Passage, Query, RankResult, Reranker
from presentation_dependence.rerankers.grade_rubrics import normalize_doc_text
from presentation_dependence.utils.setup_logging import setup_logging


def _patch_mxbai_rerank_for_tf5() -> bool:
    """Replace mxbai-rerank's `prepare_inputs` with a transformers-5 build.

    Upstream mxbai-rerank 0.1.6 calls
    `self.tokenizer.prepare_for_model(ids_a, ids_b, truncation="only_second",
    add_special_tokens=False, ...)` inside `MxbaiRerankV2.prepare_inputs`.
    The `prepare_for_model` method was removed in transformers v5 as part
    of the slow-tokenizer consolidation (PR huggingface/transformers#40936),
    so on tf5 the model load works but every `.rank()` call raises
    `AttributeError: Qwen2Tokenizer has no attribute prepare_for_model`.

    The specific call is benign: `add_special_tokens=False` means no
    CLS/SEP insertion and `truncation="only_second"` only controls which
    sequence gets trimmed when the combined length exceeds `max_length`.
    So the upstream method is equivalent to:

        combined = ids_a + ids_b
        if len(combined) > max_length: truncate from the tail of ids_b

    We rewrite `prepare_inputs` on the `MxbaiRerankV2` class at import time
    to use that equivalent, which keeps us on modern transformers and on
    the stock mxbai-rerank wheel (no fork). If upstream ships a fix we
    can drop this shim — `return False` signals "nothing to patch". Side
    effects are scoped to `MxbaiRerankV2`; no tokenizer API is restored.

    Returns True if the shim was applied, False if the installed version
    no longer needs it.
    """
    try:
        from mxbai_rerank import mxbai_rerank_v2 as _mx_v2
    except ImportError:
        return False

    cls = _mx_v2.MxbaiRerankV2
    if getattr(cls, "_tf5_patch_applied", False):
        return True

    src = getattr(cls.prepare_inputs, "__code__", None)
    if src is None or "prepare_for_model" not in (src.co_names or ()):
        return False

    def prepare_inputs(self, queries, documents, *, instruction=None):
        inputs = []
        instruction_prompt = self.instruction_prompt.format(instruction=instruction) if instruction else None
        for query, document in zip(queries, documents, strict=True):
            query_prompt = self.query_prompt.format(query=query)
            if instruction_prompt:
                query_prompt = "".join([instruction_prompt, self.sep, query_prompt])

            query_inputs = self.tokenizer(
                query_prompt,
                return_tensors=None,
                add_special_tokens=False,
                max_length=self.max_length * 3 // 4,
                truncation=True,
            )
            available_tokens = self.model_max_length - len(query_inputs["input_ids"]) - self.predefined_length
            doc_maxlen = min(available_tokens, self.max_length)
            document_inputs = self.tokenizer(
                self.doc_prompt.format(document=document),
                return_tensors=None,
                add_special_tokens=False,
                max_length=doc_maxlen,
                truncation=True,
            )

            # Emulate tf4 `prepare_for_model(ids_a, ids_b,
            # truncation="only_second", max_length=..., add_special_tokens=False,
            # return_attention_mask=False, return_token_type_ids=False)`:
            #   1. total = len(a) + len(b)
            #   2. if total > max_length and len(b) > overflow -> chop the
            #      tail of b (truncation_side="right", Qwen2 default)
            #   3. edge case: if overflow >= len(b), tf4 logs an error and
            #      leaves BOTH sequences unchanged (so the final input ids
            #      still overflow max_length); we mirror that exactly so the
            #      number-matches-upstream invariant holds.
            first = list(query_inputs["input_ids"])
            second = list(self.sep_inputs) + list(document_inputs["input_ids"])
            overflow = len(first) + len(second) - self.max_length
            if overflow > 0 and len(second) > overflow:
                second = second[: len(second) - overflow]
            combined = first + second
            item = {"input_ids": combined, "attention_mask": [1] * len(combined)}

            item["input_ids"] = self.concat_input_ids(item["input_ids"])
            item["attention_mask"] = [1] * len(item["input_ids"])
            inputs.append(item)

        return self.tokenizer.pad(
            inputs,
            padding="longest",
            max_length=self.max_length_padding,
            pad_to_multiple_of=8,
            return_tensors="pt",
        )

    cls.prepare_inputs = prepare_inputs
    cls._tf5_patch_applied = True
    return True


class MxbaiPointwise(Reranker):
    paradigm = "pointwise"

    def __init__(self, config: dict):
        super().__init__(config)
        self.logger = setup_logging(self.__class__.__name__, config)

        model_name = self.reranker_config.get("model_name", "mixedbread-ai/mxbai-rerank-large-v2")
        requested_device = self.reranker_config.get("device", "auto")
        device = _resolve_device(requested_device)
        self.batch_size = int(self.reranker_config.get("batch_size", 32))
        self.max_length = int(self.reranker_config.get("max_length", 8192))
        self.max_doc_chars = int(self.reranker_config.get("max_doc_chars") or 0)
        revision = self.reranker_config.get("revision")

        if requested_device == "auto" and device != requested_device:
            self.logger.info("Loading %s on device=%s (resolved from 'auto')", model_name, device)
        else:
            self.logger.info("Loading %s on device=%s", model_name, device)
        if revision:
            self.logger.info("Pinned HF revision=%s", revision)

        try:
            from mxbai_rerank import MxbaiRerankV2  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "mxbai_rerank is required for MxbaiPointwise. Install with `uv sync --extra rerankers`."
            ) from e

        # Must run BEFORE MxbaiRerankV2(...) so the patched `prepare_inputs`
        # is on the class when the first `.rank()` call hits it.
        _patch_mxbai_rerank_for_tf5()

        # `revision` threads through MxbaiRerankV2(**kwargs) → HF
        # from_pretrained. Absent → no kwarg passed (unambiguous "not
        # pinned" at the HF call boundary).
        hf_kwargs: dict = {"revision": revision} if revision else {}
        self.model = MxbaiRerankV2(
            model_name_or_path=model_name,
            device=device,
            max_length=self.max_length,
            **hf_kwargs,
        )

    def rank(self, query: Query, passages: list[Passage]) -> RankResult:
        if not passages:
            return {
                "top_k_psgs": [],
                "scores_init_order": [],
                "prompting_runtimes": [0.0],
                "paradigm": self.paradigm,
            }

        if self.max_doc_chars > 0:
            texts = [normalize_doc_text(p["text"], max_doc_chars=self.max_doc_chars) for p in passages]
        else:
            texts = [p["text"] for p in passages]

        t0 = time.perf_counter()
        results = self.model.rank(
            query=query["text"],
            documents=texts,
            return_documents=False,
            top_k=len(texts),
            batch_size=self.batch_size,
        )
        elapsed = time.perf_counter() - t0

        # `results` is a list of RankResult dataclasses with .index + .score;
        # we need (pid, score) in *original* order for scores_init_order and
        # in ranked order for top_k_psgs.
        scores_by_idx = [0.0] * len(texts)
        top_k: list[dict] = []
        for r in results:
            idx = int(r.index)
            score = float(r.score)
            scores_by_idx[idx] = score
            top_k.append(
                {
                    "pid": passages[idx]["pid"],
                    "text": passages[idx]["text"],
                    "score": score,
                }
            )

        return {
            "top_k_psgs": top_k,
            "scores_init_order": scores_by_idx,
            "prompting_runtimes": [elapsed],
            "paradigm": self.paradigm,
        }
