"""PairRM pairwise anchor for the response-ranking arm.

PairRM (``llm-blender/PairRM-hf``, DeBERTa-v3-large, about 0.4 B, MIT) is the
pairwise comparison between pointwise Skywork-V2 and the B>1 listwise students.
It uses the same qids, gold labels, and nDCG@1 metric as the other
response-ranking baselines, with particular attention to the
verifiable-correctness surfaces.

Inference-only pairwise anchor. Reported on the response-quality axis; not
treated as a batched scorer for stability evaluation.

The implementation loads the HF-compatible checkpoint with the vendored
:class:`~presentation_dependence.rerankers._pairrm_model.DebertaV2PairRM` head and the
published ``tokenize_pair`` format (``<|source|>``, ``<|candidate1|>``, and
``<|candidate2|>``). It does not depend on the ``llm_blender`` package, whose
transformers pin conflicts with this repository's tf5 pin, so PairRM runs in
the shared evaluation container. ``sentencepiece`` is its only additional
runtime dependency and is already included.

Apache-2.0 source and modification notice:

* The encoding algorithm is adapted from LLM-Blender's README at revision
  ``a79dd4572937c755b2f1c7fa59aca586e50dc93c``. The same function also
  appears in the MIT-tagged PairRM-hf model card; this source file derives
  from the Apache publication.
* Local changes inject the tokenizer, use strict zipped iteration and module
  constants, avoid mutating the candidate-length argument, and return the
  padded encoding directly. Ordered-pair evaluation, order debiasing, and
  Copeland aggregation are local.

The checkpoint's model-repository terms remain separate; no model weights are
redistributed here. Apache licence text is at
``third_party_licenses/Apache-2.0.txt``; see ``THIRD_PARTY_NOTICES.md``.

The pairwise mechanism emits one order-averaged aggregate score per candidate.
The evaluation interface records ``paradigm = "pointwise"`` because it consumes
those scalar scores. The resulting ranking is order-invariant and is reported
only on the quality axis.

To rank N candidates, the model scores all ordered pairs. A positive
``(prompt, A, B)`` logit favors A. Copeland wins determine the primary ranking,
with net pairwise margin as the tie-break.

PairRM has pairwise position bias: swapping A and B can reverse the verdict.
Each unordered pair ``{i, j}`` is scored in both orders::

    net(i>j) = ( logit([cand_i, cand_j])  -  logit([cand_j, cand_i]) ) / 2

The antisymmetric result, ``net(i>j) = -net(j>i)``, makes the aggregate ranking
order-invariant and keeps PairRM off the τ-PSI axis.

Required config keys under ``reranker:``:
  class: PairRMReranker
  model_name: llm-blender/PairRM-hf   (default)
  device: "auto" | "cuda" | "cpu"     (default "auto")
  dtype: "auto" | "float32" | ...     (default "auto")
  batch_size: int                     (default 16; pairwise forwards per batch)

Optional:
  max_doc_chars: int   # pre-truncate very long responses before tokenizing
  source_max_length: int   (default 1224)   # PairRM card defaults (total 2048)
  candidate_max_length: int (default 412)
  revision: str        # pin the HF commit SHA before publishing a number
"""

from __future__ import annotations

import time

from presentation_dependence.rerankers._rank_result import scores_to_rank_result
from presentation_dependence.rerankers._torch_utils import resolve_device as _resolve_device
from presentation_dependence.rerankers._torch_utils import resolve_dtype as _resolve_dtype
from presentation_dependence.rerankers.base import Passage, Query, RankResult, Reranker
from presentation_dependence.utils.setup_logging import setup_logging

SOURCE_PREFIX = "<|source|>"
CAND1_PREFIX = "<|candidate1|>"
CAND2_PREFIX = "<|candidate2|>"


def ordered_pairs(n: int) -> list[tuple[int, int]]:
    """All ordered index pairs ``(i, j)`` with ``i != j`` (both directions)."""
    return [(i, j) for i in range(n) for j in range(n) if i != j]


def copeland_scores_from_margins(n: int, net_margin: dict[tuple[int, int], float]) -> list[float]:
    """Aggregate order-debiased pairwise margins into per-candidate scores.

    ``net_margin[(i, j)]`` is the net (order-averaged) preference of candidate
    ``i`` over ``j`` (>0 ⇒ ``i`` better). It is treated as antisymmetric: if
    ``(i, j)`` is absent, ``-net_margin[(j, i)]`` is used.

    Score = **Copeland wins** (1 per opponent beaten, 0.5 per exact tie) plus a
    tiny margin term that breaks ties *within* an equal-win group without ever
    crossing a whole Copeland point — so the primary order is Copeland, refined
    by total net margin (a Borda-like signal). Returns per-candidate scalars in
    input order suitable for ``scores_to_rank_result``.
    """
    if n <= 0:
        return []

    wins = [0.0] * n
    total_margin = [0.0] * n
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            m = net_margin.get((i, j))
            if m is None:
                back = net_margin.get((j, i))
                m = -back if back is not None else 0.0
            total_margin[i] += m
            if m > 0:
                wins[i] += 1.0
            elif m == 0:
                wins[i] += 0.5

    # Tie-break weight: keep max |margin term difference| strictly below the
    # smallest Copeland gap (0.5, since ties add 0.5) so the margin only ever
    # orders candidates that already have equal Copeland wins.
    max_abs = max((abs(t) for t in total_margin), default=0.0)
    eps = 0.0 if max_abs == 0 else 0.25 / (max_abs * n)
    return [wins[i] + eps * total_margin[i] for i in range(n)]


def tokenize_pair(
    tokenizer,
    sources: list[str],
    candidate1s: list[str],
    candidate2s: list[str],
    *,
    source_max_length: int = 1224,
    candidate_max_length: int = 412,
):
    """Build PairRM ``<|source|>/<|candidate1|>/<|candidate2|>`` encodings.

    Adapted from the ``tokenize_pair`` function published in LLM-Blender's
    Apache-2.0 README: encode ``source_prefix + source`` (truncated), then split
    the remaining budget evenly between the two candidate prefixes, concatenate
    the three id blocks, and right-pad the batch to a fixed ``max_length``.
    Isolated as a pure function (tokenizer injected) so the prompt format is
    unit-testable without the 0.4B weights.
    """
    assert len(sources) == len(candidate1s) == len(candidate2s)
    max_length = source_max_length + 2 * candidate_max_length
    ids = []
    for src, c1, c2 in zip(sources, candidate1s, candidate2s, strict=True):
        source_ids = tokenizer.encode(SOURCE_PREFIX + src, max_length=source_max_length, truncation=True)
        cand_len = (max_length - len(source_ids)) // 2
        cand1_ids = tokenizer.encode(CAND1_PREFIX + c1, max_length=cand_len, truncation=True)
        cand2_ids = tokenizer.encode(CAND2_PREFIX + c2, max_length=cand_len, truncation=True)
        ids.append(source_ids + cand1_ids + cand2_ids)
    return tokenizer.pad({"input_ids": ids}, return_tensors="pt", padding="max_length", max_length=max_length)


class PairRMReranker(Reranker):
    paradigm = "pointwise"

    def __init__(self, config: dict):
        super().__init__(config)
        self.logger = setup_logging(self.__class__.__name__, config)

        rc = self.reranker_config
        self.model_name = rc.get("model_name", "llm-blender/PairRM-hf")
        requested_device = rc.get("device", "auto")
        self.device = _resolve_device(requested_device)
        self.dtype = _resolve_dtype(rc.get("dtype", "auto"), self.device)
        self.batch_size = int(rc.get("batch_size", 16))
        if self.batch_size <= 0:
            raise ValueError("reranker.batch_size must be positive")
        max_doc_chars = rc.get("max_doc_chars")
        self.max_doc_chars = int(max_doc_chars) if max_doc_chars is not None else None
        self.source_max_length = int(rc.get("source_max_length", 1224))
        self.candidate_max_length = int(rc.get("candidate_max_length", 412))
        self.revision = rc.get("revision")

        self.logger.info(
            "Loading %s on device=%s dtype=%s (pairwise anchor, order-averaged)",
            self.model_name,
            self.device,
            self.dtype,
        )
        if self.revision:
            self.logger.info("Pinned HF revision=%s", self.revision)

        from transformers import AutoTokenizer

        from presentation_dependence.rerankers._pairrm_model import DebertaV2PairRM

        hf_kwargs: dict = {"revision": self.revision} if self.revision else {}
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, **hf_kwargs)
        self.model = DebertaV2PairRM.from_pretrained(self.model_name, torch_dtype=self.dtype, **hf_kwargs)
        self.model.to(self.device)
        self.model.eval()

    def _truncate(self, text: str) -> str:
        if self.max_doc_chars is not None and self.max_doc_chars > 0:
            return str(text)[: self.max_doc_chars]
        return str(text)

    def _compare_logits(self, inputs: list[str], cands_a: list[str], cands_b: list[str]) -> list[float]:
        """Batched pairwise logits (>0 ⇒ A better). Isolated for test stubbing."""
        import torch

        out: list[float] = []
        for start in range(0, len(inputs), self.batch_size):
            enc = tokenize_pair(
                self.tokenizer,
                inputs[start : start + self.batch_size],
                cands_a[start : start + self.batch_size],
                cands_b[start : start + self.batch_size],
                source_max_length=self.source_max_length,
                candidate_max_length=self.candidate_max_length,
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                logits = self.model(**enc).logits
            out.extend(float(x) for x in logits.float().reshape(-1).tolist())
        return out

    def rank(self, query: Query, passages: list[Passage]) -> RankResult:
        n = len(passages)
        if n == 0:
            return {
                "top_k_psgs": [],
                "scores_init_order": [],
                "prompting_runtimes": [0.0],
                "paradigm": self.paradigm,
            }
        if n == 1:
            return scores_to_rank_result([0.0], passages, 0.0, self.paradigm, model_name=self.model_name)

        prompt = query["text"]
        cands = [self._truncate(p["text"]) for p in passages]
        pairs = ordered_pairs(n)

        inputs = [prompt] * len(pairs)
        cands_a = [cands[i] for (i, _j) in pairs]
        cands_b = [cands[j] for (_i, j) in pairs]

        t0 = time.perf_counter()
        logits = self._compare_logits(inputs, cands_a, cands_b)
        elapsed = time.perf_counter() - t0

        # logit for ordered pair (i, j) = preference of i over j in THAT order.
        # Order-averaged net(i>j) = (logit(i,j) - logit(j,i)) / 2.
        logit_by_pair = {pair: logits[k] for k, pair in enumerate(pairs)}
        net_margin: dict[tuple[int, int], float] = {}
        for i in range(n):
            for j in range(i + 1, n):
                net_margin[(i, j)] = (logit_by_pair[(i, j)] - logit_by_pair[(j, i)]) / 2.0

        scores = copeland_scores_from_margins(n, net_margin)
        return scores_to_rank_result(scores, passages, elapsed, self.paradigm, model_name=self.model_name)
