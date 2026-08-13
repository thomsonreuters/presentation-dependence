"""jina-reranker-v3 scoring-listwise baseline and retrofit target.

Model: ``jinaai/jina-reranker-v3`` (0.6 B, Qwen3-0.6B backbone with a
1024 → 512 → 256 MLP projector; described by Wang, Li, and Xiao in
*jina-reranker-v3: Last but Not Late Interaction for Document Reranking*).

The model emits one scalar per document from shared causal-attention context.
It reads each document representation at ``<|embed_token|>``, the query
representation at ``<|rerank_token|>``, projects both, and scores their cosine
similarity. Jina describes this as "last but not late" interaction. The model
produces native per-document scores rather than a synthetic score derived from
rank position.

Upstream ``rerank()`` packs candidates into blocks of at most 125 documents,
subject to the tokenizer budget. At ``k_input=100``, typical inputs require one
GPU forward. The 0.6 B model is about 10 times cheaper end to end than the 7 B
RankZephyr baseline, which uses several sliding windows. Wang et al.'s Table 1
reports a BEIR mean nDCG@10 of 61.94. We use this result as our public
scoring-listwise reference and retrofit target.

The wrapper calls ``JinaForRanking.rerank()`` directly to preserve its
multi-block aggregation. For lists longer than ``block_size``, each block
produces a query embedding. Upstream combines those embeddings with
``((1 + max_score) / 2).max()`` block weights before computing the final
document scores.

The wrapper adapts this output to ``RankResult``, sets
``paradigm = "scoring_listwise"``, and records raw cosine scores in input order
for PSI. It also applies the shared device, dtype, and revision handling.
The checkpoint is CC-BY-NC-4.0 and is evaluation-only in this repository; see
``THIRD_PARTY_NOTICES.md``.

Required config keys under ``reranker:``::

    class: JinaListwiseReranker
    model_name: jinaai/jina-reranker-v3
    revision: str                                           # 40-char HF commit SHA
    device: "auto" | "cuda" | "cpu" | "mps" | "cuda:<n>"   (default "auto")
    dtype: "auto" | "bfloat16" | "float16" | "float32"      (default "auto")
    max_doc_length: int                                      (default 2048)
    max_query_length: int                                    (default 512)
    max_doc_chars: int                                       (default 0 = disabled)

Optional::

    docs_per_score_forward: int  # If set (e.g. 20), score ``k_input`` candidates
                        # in disjoint chunks: one ``rerank()`` per chunk (OC-SFT /
                        # expected-grade readout B=20 protocol). Default 0 = single ``rerank()`` over
                        # the full list (native listwise, τ-PSI@B=k_input).
    doc_block_cap: int  # τ-PSI geometry / metadata only — must match HF
                        # ``rerank()`` doc-count cap if you pin it elsewhere
                        # (default 125 mirrors ``modeling.py`` ``block_size``).
"""

from __future__ import annotations

import time

from presentation_dependence.rerankers._torch_utils import resolve_device as _resolve_device
from presentation_dependence.rerankers._torch_utils import resolve_dtype as _resolve_dtype
from presentation_dependence.rerankers.base import Passage, Query, RankResult, Reranker
from presentation_dependence.rerankers.grade_rubrics import normalize_doc_text
from presentation_dependence.utils.setup_logging import setup_logging

# -----------------------------------------------------------------------------
# pure helpers (unit-testable without loading the model)
# -----------------------------------------------------------------------------


def _require_immutable_revision(value: object) -> str:
    """Return a reviewed Hugging Face commit SHA or fail before remote-code loading."""
    revision = str(value or "").strip().lower()
    if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
        raise ValueError(
            "JinaListwiseReranker requires reranker.revision to be an immutable "
            "40-character Hugging Face commit SHA because loading the model executes "
            "repository code via trust_remote_code=True."
        )
    return revision


def _rerank_results_to_rank_result(
    rerank_outputs: list[dict],
    passages: list[Passage],
    elapsed: float,
    paradigm: str,
) -> RankResult:
    """Map ``JinaForRanking.rerank()`` output → project ``RankResult``.

    Upstream returns a list of dicts ``{document, relevance_score, index,
    embedding}`` already sorted by score (highest first). We:

    - Use ``index`` to look up ``pid`` from the original ``Passage``;
      the upstream ``document`` field contains only text.
    - Build ``top_k_psgs`` in upstream order (i.e. best-first).
    - Build ``scores_init_order`` indexed by the **input** position so
      score-variance PSI can read scalars in the same frame as the input
      run file.

    This helper is split out as a pure function so the parsing contract
    can be unit-tested without loading 0.6 B parameters or hitting HF.
    """
    if not rerank_outputs:
        return {
            "top_k_psgs": [],
            "scores_init_order": [],
            "prompting_runtimes": [elapsed],
            "paradigm": paradigm,
        }

    n = len(passages)
    scores_by_idx: list[float] = [0.0] * n
    top_k: list[dict] = []
    seen_indexes: set[int] = set()
    for r in rerank_outputs:
        idx = int(r["index"])
        if idx < 0 or idx >= n:
            raise ValueError(f"jina-reranker-v3 returned out-of-range index={idx} for {n} passages")
        if idx in seen_indexes:
            raise ValueError(f"jina-reranker-v3 returned duplicate index={idx}")
        seen_indexes.add(idx)
        score = float(r["relevance_score"])
        scores_by_idx[idx] = score
        top_k.append(
            {
                "pid": passages[idx]["pid"],
                "text": passages[idx]["text"],
                "score": score,
            }
        )

    if len(top_k) != n:
        # Upstream contract is "all documents returned unless top_n is set";
        # we always pass top_n=None, so a length mismatch here is a real
        # protocol break and we want the eval to fail loudly rather than
        # silently dropping passages from PSI accounting.
        raise ValueError(
            f"jina-reranker-v3 returned {len(top_k)} documents for {n} input passages "
            f"(expected all). Upstream rerank() likely changed its top_n default."
        )

    return {
        "top_k_psgs": top_k,
        "scores_init_order": scores_by_idx,
        "prompting_runtimes": [elapsed],
        "paradigm": paradigm,
    }


def _scores_init_order_to_rank_result(
    scores_by_idx: list[float],
    passages: list[Passage],
    elapsed: float,
    paradigm: str,
) -> RankResult:
    """Build ``RankResult`` from per-input-position scores (best-first ``top_k``)."""
    indexed = sorted(range(len(passages)), key=lambda i: scores_by_idx[i], reverse=True)
    top_k = [
        {
            "pid": passages[i]["pid"],
            "text": passages[i]["text"],
            "score": scores_by_idx[i],
        }
        for i in indexed
    ]
    return {
        "top_k_psgs": top_k,
        "scores_init_order": list(scores_by_idx),
        "prompting_runtimes": [elapsed],
        "paradigm": paradigm,
    }


# -----------------------------------------------------------------------------
# Reranker class
# -----------------------------------------------------------------------------


class JinaListwiseReranker(Reranker):
    """jina-reranker-v3 wrapper using the official ``rerank()`` method.

    Single forward per query (or a small handful of forwards for
    ``len(passages) > block_size``, with upstream block-aggregation). No
    sliding window: scoring-head listwise doesn't need bubble-sort because
    the model already attends across all docs in one context.
    """

    paradigm = "scoring_listwise"

    def __init__(self, config: dict):
        super().__init__(config)
        self.logger = setup_logging(self.__class__.__name__, config)

        rc = self.reranker_config
        model_name = rc.get("model_name", "jinaai/jina-reranker-v3")
        requested_device = rc.get("device", "auto")
        device = _resolve_device(requested_device)
        dtype = _resolve_dtype(rc.get("dtype", "auto"), device)
        revision = _require_immutable_revision(rc.get("revision"))
        self.max_doc_length = int(rc.get("max_doc_length", 2048))
        self.max_query_length = int(rc.get("max_query_length", 512))
        self.max_doc_chars = int(rc.get("max_doc_chars") or 0)
        raw_dpf = rc.get("docs_per_score_forward")
        self.docs_per_score_forward = int(raw_dpf) if raw_dpf is not None else 0
        if self.docs_per_score_forward < 0:
            raise ValueError("reranker.docs_per_score_forward must be >= 0")

        if requested_device == "auto" and device != requested_device:
            self.logger.info("Loading %s on device=%s (resolved from 'auto'), dtype=%s", model_name, device, dtype)
        else:
            self.logger.info("Loading %s on device=%s, dtype=%s", model_name, device, dtype)
        self.logger.info("Pinned HF revision=%s", revision)
        if self.docs_per_score_forward > 0:
            self.logger.info(
                "Chunked listwise scoring: docs_per_score_forward=%d (τ-PSI@B=%d)",
                self.docs_per_score_forward,
                self.docs_per_score_forward,
            )

        from transformers import AutoModel  # local import: heavy

        # ``trust_remote_code=True`` is mandatory: the HF repo ships a
        # ``modeling.py`` with the custom ``JinaForRanking`` class, the
        # projector head, and the ``rerank()`` method we delegate to.
        # ``transformersInfo.custom_class`` in the model card is
        # ``modeling.JinaForRanking``: see
        # https://huggingface.co/jinaai/jina-reranker-v3/blob/main/modeling.py
        # The ``dtype=...`` kwarg is the transformers-v5 spelling
        # (formerly ``torch_dtype=``); we already pin tf5 elsewhere in
        # this project (see mxbai_pointwise._patch_mxbai_rerank_for_tf5
        # for a longer note on the version).
        hf_kwargs: dict = {"trust_remote_code": True, "dtype": dtype}
        hf_kwargs["revision"] = revision
        self.model = AutoModel.from_pretrained(model_name, **hf_kwargs)
        self.model.to(device)
        self.model.eval()
        self.device = device

    def rank(self, query: Query, passages: list[Passage]) -> RankResult:
        # Empty-input fast path: matches the IdentityReranker contract and
        # avoids triggering upstream's tokeniser-on-empty-list edge case.
        if not passages:
            return {
                "top_k_psgs": [],
                "scores_init_order": [],
                "prompting_runtimes": [0.0],
                "paradigm": self.paradigm,
            }

        if self.max_doc_chars > 0:
            documents = [normalize_doc_text(p["text"], max_doc_chars=self.max_doc_chars) for p in passages]
        else:
            documents = [p["text"] for p in passages]

        t0 = time.perf_counter()
        if self.docs_per_score_forward > 0:
            n = len(passages)
            scores_by_idx: list[float] = [0.0] * n
            for start in range(0, n, self.docs_per_score_forward):
                chunk_docs = documents[start : start + self.docs_per_score_forward]
                chunk_passages = passages[start : start + self.docs_per_score_forward]
                rerank_outputs = self.model.rerank(
                    query=query["text"],
                    documents=chunk_docs,
                    top_n=None,
                    return_embeddings=False,
                    max_doc_length=self.max_doc_length,
                    max_query_length=self.max_query_length,
                )
                chunk_result = _rerank_results_to_rank_result(
                    rerank_outputs=rerank_outputs,
                    passages=chunk_passages,
                    elapsed=0.0,
                    paradigm=self.paradigm,
                )
                for offset, score in enumerate(chunk_result["scores_init_order"]):
                    scores_by_idx[start + offset] = float(score)
            elapsed = time.perf_counter() - t0
            return _scores_init_order_to_rank_result(
                scores_by_idx=scores_by_idx,
                passages=passages,
                elapsed=elapsed,
                paradigm=self.paradigm,
            )

        # Native listwise: one ``rerank()`` over the full candidate list.
        rerank_outputs = self.model.rerank(
            query=query["text"],
            documents=documents,
            top_n=None,
            return_embeddings=False,
            max_doc_length=self.max_doc_length,
            max_query_length=self.max_query_length,
        )
        elapsed = time.perf_counter() - t0

        return _rerank_results_to_rank_result(
            rerank_outputs=rerank_outputs,
            passages=passages,
            elapsed=elapsed,
            paradigm=self.paradigm,
        )
