"""RankZephyr generative-listwise baseline and retrofit base.

Model: ``castorini/rank_zephyr_7b_v1_full`` (7B, MIT, Zephyr-Beta backbone
distilled from GPT-4 listwise rankings; described by Pradeep et al.). Used as an
off-the-shelf baseline only. Because its supervision came from GPT-4, do not
describe it as independently supervised or distil from it further.

Paradigm: **generative listwise**.
The model emits a textual permutation of identifier tokens like
``[3] > [1] > [2] > ...`` decoded autoregressively. There are no native per-doc
scalar scores — downstream ``scores_init_order`` is ``None`` and the TREC run
writer fills in ``score = N - rank_idx`` (see ``presentation_dependence.utils.trec``).

Apache-2.0 source and modification notice
-----------------------------------------

This file reproduces the RankZephyr prompt text and adapts prompt construction
from ``castorini/rank_llm`` revision
``e2ceebe68126430c0960f7282e14c709865d66cb``. It also adapts permutation
parsing, document truncation, and sliding-window behavior from
``sunnweiwei/RankGPT`` revision
``0d62bc3855c7c118048a7c47c18e719b938e291a``. Both repositories are
Apache-2.0 and have no root ``NOTICE`` file; the reviewed source portions carry
no pertinent copyright or attribution header.

Local modifications include the project reranker interface, bracket-only
identifier parsing, validated padding of omitted identifiers, query/document
identifier sanitization, dynamic token-budget shrinking, multiple execution
protocols, and optional log-prob collection. Licence text is at
``third_party_licenses/Apache-2.0.txt``; see ``THIRD_PARTY_NOTICES.md``.

Sliding window
--------------

For ``N > window_size`` candidates the published protocol uses a sliding-window
bubble-sort (window=20, stride=10). Windows progress **end → start**: the
first window reranks ``[N - W, N)``, the next reranks ``[N - W - S, N - S)``,
etc., until the last window covers ``[0, W)``. Each window rewrites its slice
of the global list in place, so good docs "bubble" toward the top across
~ceil((N - W) / S) + 1 sequential forwards. At ``N=100, W=20, S=10`` that's
9 forwards per query.

Design notes:

- ``window_size`` and ``stride`` are instance attributes, not buried locals,
  so the training loop can use single-window rollouts at small N, avoiding the
  sequential cost of overlapping windows when the candidate pool already fits.
- The HF ``generate()`` call is plumbed with ``return_dict_in_generate=True``
  + ``output_scores=True`` under an opt-in flag so later we can read per-step
  identifier-token log-probs without a second forward pass. Off by default
  to keep eval-time memory flat. The knob is for the **log-prob ablation only**:
  main-path RankZephyr stays rank-only, while the ablation derives
  trie-constrained first-identifier pseudo-scores from per-step token log-probs.
- The local sliding-window implementation avoids a dependency on
  ``castorini/rank_llm``, whose wrapper is coupled to its evaluation stack and
  does not expose the gradient hooks required for RL rollouts. Prompt template
  and passage-handling behaviour are still reproduced from ``rank_llm`` and
  ``sunnweiwei/RankGPT`` (both Apache-2.0); see ``THIRD_PARTY_NOTICES.md``.

Prompt-overflow handling
------------------------

Mistral-7B (Zephyr-β's backbone) uses **sliding-window attention with
window=4096**; the model can generate from ``max_position_embeddings=32768``
but its attention only "sees" the last 4096 tokens. BEIR corpora with long
abstracts (TREC-COVID mean ≈ 228 words vs. MS MARCO ≈ 63) easily push a
20-passage listwise prompt to 7–10 k tokens, at which point the model can't
attend to the early identifiers and emits garbage that the parser drops
back to identity, producing a no-op rerank equal to BM25.

The fix, adopted from ``castorini/rank_llm``'s ``--variable-passages`` flag
(:func:`RankListwiseOSLLM.create_prompt`), is an iterative shrink loop: we
start at ``max_doc_words`` and, if the tokenised chat-templated prompt
exceeds ``max_total_tokens - reserved_output_tokens``, shrink per-doc word
cap using ``rank_llm``'s heuristic and rebuild. DL19/DL20 prompts are already
~1.9 k tokens so the loop is a no-op there; BEIR settles around 100–120
words/passage.

Bracket-number leak from queries
--------------------------------

A separate but related failure mode appears on **BEIR ArguAna** (and any
future task with cited queries): 57 % of ArguAna queries carry ``[N]``
citation markers like ``[1]``, ``[7]``, ``[13]``. We strip ``[N] → (N)``
from passage text in :func:`_prepare_doc_content`, but the **same
replacement must also be applied to the query** before it is templated
into the prompt — the official template repeats ``{query}`` twice (prefix
+ suffix), so a single un-sanitised ``[7]`` inside the query body
duplicates as a phantom identifier the model can copy-quote in its output.
``rank_llm``'s ``singleturn_listwise_inference_handler.generate_prompt``
calls ``query = self._replace_number(query)`` before templating; we do the
same in :func:`_build_user_prompt`. Diagnosed against the off-shelf
RankZephyr BEIR ArguAna run (gap
of ~0.18 nDCG@10 vs. published, while every other measured BEIR task —
0/N queries with ``[N]`` patterns — was within ~0.04).

Inference protocols
-------------------

**Tier 1 (literature default):** ``rankzephyr_protocol: sliding`` (or omit —
same thing). Uses :func:`_window_schedule` with overlapping windows when
``stride < window_size`` (published defaults: ``W=20``, ``S=10``).

**Tier 2 (overlap ablation):** still ``sliding``; set ``stride == window_size``
so windows tile (for ``k_input`` a multiple of ``W``, e.g. 100/20) with fewer
forwards — same in-place bubble **mechanism** as tier 1.

**Tier 3 (``block_local_rrf``):** one listwise forward per disjoint block,
followed by RRF over the block-local rankings. This matches a ``⌈N/B⌉``
decode-and-merge budget, while tier 2 applies dependent forwards to one evolving
list. Tier 3 separates cross-block merge effects from overlap and stride
effects and a controlled comparison with batched-pointwise scalar
merging. It does not reproduce the published RankZephyr protocol. Results must
be labelled *RankZephyr-7B + block-local listwise + RRF* rather than the
published RankZephyr baseline.

Config (under ``reranker:`` in ``configs/experiments/<ID>.yaml`` — sliding):

    class: RankZephyrReranker
    model_name: castorini/rank_zephyr_7b_v1_full
    device: auto              # auto | cuda | cuda:<n> | mps | cpu
    dtype:  auto              # auto | bfloat16 | float16 | float32
    # rankzephyr_protocol: sliding   # default; tier 1–2
    window_size: 20
    stride:      10
    max_new_tokens: 200
    max_doc_words:  300       # per-passage word-truncation UPPER bound
    max_total_tokens: 4096    # hard budget for templated prompt; Mistral SWA
    return_logprobs: false    # opt-in for log-prob reconstruction
    revision:        null     # optional HF commit SHA (pin before publishing)

Tier 3 additionally (``stride`` ignored):

    rankzephyr_protocol: block_local_rrf
    block_size: 20            # docs per block (= listwise context width B); defaults to window_size
    rrf_k: 60                  # RRF constant (positive); standard IR default
"""

from __future__ import annotations

import re
import time

from presentation_dependence.rerankers._torch_utils import resolve_device as _resolve_device
from presentation_dependence.rerankers._torch_utils import resolve_dtype as _resolve_dtype
from presentation_dependence.rerankers._windowing import apply_window_permutation as _apply_window_permutation
from presentation_dependence.rerankers._windowing import sliding_window_bubble_schedule
from presentation_dependence.rerankers.base import Passage, Query, RankResult, Reranker
from presentation_dependence.utils.setup_logging import setup_logging


# Inference protocol names (``reranker.rankzephyr_protocol`` in YAML).
RANKZEPHYR_PROTOCOL_SLIDING = "sliding"
RANKZEPHYR_PROTOCOL_BLOCK_LOCAL_RRF = "block_local_rrf"

# -----------------------------------------------------------------------------
# pure helpers (unit-testable without loading the model)
# -----------------------------------------------------------------------------


_RANK_ID_RE = re.compile(r"\[(\d+)\]")


def _parse_permutation(generation: str, num_docs: int) -> list[int]:
    """Parse a RankGPT-style permutation string into 0-indexed local positions.

    The model is trained to emit ``[4] > [2] > [1]``; we accept anything the
    tokenizer happens to produce by scanning for ``[N]`` identifiers left-to-
    right (the ``>`` separator is a convention the model follows but we don't
    require). Returns a list of 0-indexed window positions with duplicates
    and out-of-range ids dropped, and missing ids appended in their original
    window order (the standard "padding" fallback from RankGPT). The result
    always has length
    ``num_docs`` and is a valid permutation of ``range(num_docs)``.
    """
    seen: set[int] = set()
    parsed: list[int] = []
    for m in _RANK_ID_RE.finditer(generation):
        raw = int(m.group(1))
        # Model is prompted with 1-indexed identifiers; translate back.
        idx = raw - 1
        if 0 <= idx < num_docs and idx not in seen:
            seen.add(idx)
            parsed.append(idx)
    # Pad with any missing positions, preserving the input order among them.
    # Matches the published protocol's "missing doc" fallback.
    for i in range(num_docs):
        if i not in seen:
            parsed.append(i)
    return parsed


def _window_schedule(n: int, window_size: int, stride: int) -> list[tuple[int, int]]:
    """Compute the list of ``(start, end)`` window bounds in the order
    RankZephyr visits them: from the end of the list toward the beginning.

    - If ``n <= window_size``: a single window ``(0, n)``.
    - Otherwise: start at ``(n - W, n)``, step ``-S``, until the next start
      would be negative. The final window is forced to ``(0, W)`` so position
      0 is always covered exactly once; this matches the RankZephyr protocol
      description ("iteratively reranks [...] until the whole list is
      covered").

    Example (n=100, W=20, S=10):
        (80, 100), (70, 90), (60, 80), (50, 70), (40, 60), (30, 50),
        (20, 40), (10, 30), (0, 20)   # 9 windows
    """
    return sliding_window_bubble_schedule(n, window_size, stride)


def _contiguous_passage_blocks(passages: list[Passage], block_size: int) -> list[list[Passage]]:
    """Partition ``passages`` into disjoint contiguous slices of length
    ``block_size`` (last slice may be shorter).
    """
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if not passages:
        return []
    return [passages[i : i + block_size] for i in range(0, len(passages), block_size)]


def _merge_rrf_block_rankings(
    blocks_ranked: list[list[Passage]],
    rrf_k: float,
    orig_index_by_pid: dict[str, int],
) -> list[Passage]:
    """Fuse per-block best-first lists into one global ordering.

    Standard RRF with one ranked list per document:
    ``score(d) = 1 / (rrf_k + rank_block(d))`` with **1-based** within-block
    ranks. Documents from different blocks are **never** compared inside the
    LM; only this merge rule orders across blocks.

    Tie-break when scores are equal: lower ``block_idx``, then lower within-
    block rank, then lower ``orig_index`` (input position before reranking),
    then ``pid`` — deterministic and reviewer-stable.
    """
    if rrf_k <= 0:
        raise ValueError(f"rrf_k must be positive, got {rrf_k}")
    scored: list[tuple[Passage, float, int, int, int, str]] = []
    for b_idx, ranked in enumerate(blocks_ranked):
        for rank_1b, p in enumerate(ranked, start=1):
            pid = str(p["pid"])
            if pid not in orig_index_by_pid:
                raise KeyError(f"RRF merge: unknown pid {pid!r}")
            orig = orig_index_by_pid[pid]
            rrf = 1.0 / (rrf_k + rank_1b)
            scored.append((p, rrf, b_idx, rank_1b, orig, pid))
    scored.sort(key=lambda t: (-t[1], t[2], t[3], t[4], t[5]))
    return [t[0] for t in scored]


_BRACKET_NUM_RE = re.compile(r"\[(\d+)\]")


def _replace_bracket_numbers(text: str) -> str:
    """Rewrite any ``[N]`` in ``text`` to ``(N)`` so passage content cannot
    be confused for prompt identifiers. Mirrors
    ``rank_llm.rerank.inference_handler.InferenceHandler._replace_number``.
    This is defensive: we decode only the newly generated tail for parsing
    (see :meth:`RankZephyrReranker._generate`), so a ``[N]`` that lives
    inside a passage can't leak into our parse pass — but it would leak into
    the model's *input*, where it might trigger the model to copy-quote a
    wrong identifier. Matching
    ``rank_llm.rerank.inference_handler.InferenceHandler._replace_number``
    removes that source of drift.
    """
    return _BRACKET_NUM_RE.sub(r"(\1)", text)


def _truncate_doc_by_words(text: str, max_words: int) -> str:
    """RankGPT convention: per-passage word-level truncation. Cheaper than
    tokenising each doc twice (once for counting, once in the model forward)
    and matches the published implementation.
    """
    if max_words <= 0:
        return text
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def _prepare_doc_content(text: str, max_words: int) -> str:
    """Per-doc content preparation that exactly matches
    ``rank_llm.rerank.inference_handler.InferenceHandler._convert_doc_to_prompt_content``:
    strip → ftfy normalise → word-truncate → rewrite bracket numbers.

    ``ftfy`` is imported lazily — the module is in the ``[cuda]`` extra;
    if absent we skip normalisation (correctness-equivalent on ASCII inputs
    like MS MARCO / most BEIR corpora).
    """
    try:
        from ftfy import fix_text  # type: ignore
    except ImportError:
        fix_text = None

    content = (text or "").strip()
    if fix_text is not None:
        content = fix_text(content)
    words = content.split()
    if max_words > 0 and len(words) > max_words:
        content = " ".join(words[:max_words])
    else:
        content = " ".join(words)
    return _replace_bracket_numbers(content)


def _apply_optional_char_cap(text: str, max_doc_chars: int | None) -> str:
    """Apply an exact character cap before RankZephyr's word/token shrinking."""
    return str(text)[:max_doc_chars] if max_doc_chars is not None else str(text)


# -----------------------------------------------------------------------------
# prompt construction
# -----------------------------------------------------------------------------


# System + prefix + body + suffix text below mirrors
# ``rank_llm/src/rank_llm/rerank/prompt_templates/rank_zephyr_template.yaml``
# byte-for-byte. RankZephyr was trained with this exact
# wording so drift here silently degrades quality: see INVESTIGATION doc.
_SYSTEM_PROMPT = (
    "You are RankLLM, an intelligent assistant that can rank passages based on their relevancy to the query"
)


def _build_user_prompt(query_text: str, window_docs: list[str]) -> str:
    r"""RankGPT / RankZephyr single-turn user prompt. Format reproduces the
    official ``rank_zephyr_template.yaml`` exactly:

    - prefix: ``"I will provide you with {num} passages, ...: {query}.\\n"``
    - body (repeated): ``"[{rank}] {candidate}\\n"``
    - suffix: ``"Search Query: {query}.\\nRank the {num} ... explain."``

    i.e. passages are joined by single ``\\n`` (not ``\\n\\n``), and the
    trailing terminator reads "Answer concisely and directly and only respond
    with the ranking results, do not say any word or explain." — shorter
    variants cause mild quality drift.

    The query is sanitised through :func:`_replace_bracket_numbers` before
    being templated in. This mirrors ``rank_llm``'s
    ``singleturn_listwise_inference_handler.generate_prompt`` (line
    ``query = self._replace_number(query)``). Without this, queries that
    contain ``[N]`` citation markers — e.g. **57 % of BEIR ArguAna
    queries** carry constructions like ``[1]``, ``[7]``, ``[13]`` — leak
    spurious identifier tokens into the prompt twice (prefix + suffix),
    confusing the model's listwise output and degrading the rerank toward
    identity. The failure was measured on the off-shelf RankZephyr
    BEIR ArguAna run.
    """
    safe_query = _replace_bracket_numbers(query_text or "")
    num = len(window_docs)
    prefix = (
        f"I will provide you with {num} passages, each indicated by a numerical "
        f"identifier []. Rank the passages based on their relevance to the search "
        f"query: {safe_query}.\n"
    )
    body = "".join(f"[{i + 1}] {doc}\n" for i, doc in enumerate(window_docs))
    suffix = (
        f"Search Query: {safe_query}.\n"
        f"Rank the {num} passages above based on their relevance to the search "
        f"query. All the passages should be included and listed using "
        f"identifiers, in descending order of relevance. The output format "
        f"should be [] > [], e.g., [2] > [1], Answer concisely and directly and "
        f"only respond with the ranking results, do not say any word or explain."
    )
    return prefix + body + suffix


# -----------------------------------------------------------------------------
# reranker
# -----------------------------------------------------------------------------


class RankZephyrReranker(Reranker):
    paradigm = "generative_listwise"

    def __init__(self, config: dict):
        super().__init__(config)
        self.logger = setup_logging(self.__class__.__name__, config)

        rc = self.reranker_config
        model_name = rc.get("model_name", "castorini/rank_zephyr_7b_v1_full")
        requested_device = rc.get("device", "auto")
        device = _resolve_device(requested_device)
        requested_dtype = rc.get("dtype", "auto")

        self.window_size = int(rc.get("window_size", 20))
        self.stride = int(rc.get("stride", 10))

        raw_proto = str(rc.get("rankzephyr_protocol") or RANKZEPHYR_PROTOCOL_SLIDING).strip().lower()
        self.rankzephyr_protocol = raw_proto
        if self.rankzephyr_protocol not in (
            RANKZEPHYR_PROTOCOL_SLIDING,
            RANKZEPHYR_PROTOCOL_BLOCK_LOCAL_RRF,
        ):
            raise ValueError(
                f"reranker.rankzephyr_protocol must be {RANKZEPHYR_PROTOCOL_SLIDING!r} or "
                f"{RANKZEPHYR_PROTOCOL_BLOCK_LOCAL_RRF!r}, got {raw_proto!r}"
            )
        if self.rankzephyr_protocol == RANKZEPHYR_PROTOCOL_BLOCK_LOCAL_RRF:
            self.block_size = int(rc.get("block_size") or self.window_size)
            if self.block_size <= 0:
                raise ValueError(
                    f"block_size must be positive for {RANKZEPHYR_PROTOCOL_BLOCK_LOCAL_RRF}, got {self.block_size}"
                )
            self.rrf_k = float(rc.get("rrf_k", 60))
            if self.rrf_k <= 0:
                raise ValueError(f"rrf_k must be positive, got {self.rrf_k}")
        else:
            self.block_size = self.window_size
            self.rrf_k = 60.0

        self.max_new_tokens = int(rc.get("max_new_tokens", 200))
        self.max_doc_words = int(rc.get("max_doc_words", 300))
        raw_max_doc_chars = rc.get("max_doc_chars")
        self.max_doc_chars = None if raw_max_doc_chars is None else int(raw_max_doc_chars)
        if self.max_doc_chars is not None and self.max_doc_chars <= 0:
            raise ValueError("reranker.max_doc_chars must be positive when set")
        # Mistral-v0.1 (Zephyr-β's backbone) has sliding_window=4096: prompts
        # longer than this overflow the attention window and the model starts
        # emitting garbage the parser falls back to identity for. Matches
        # rank_llm's ``--context-size 4096`` default. See module docstring
        # "Prompt-overflow handling".
        self.max_total_tokens = int(rc.get("max_total_tokens", 4096))
        # Minimum per-doc word budget the shrink loop will bottom out at.
        # Floor at 8 words so we always emit *some* content for every doc —
        # dropping a passage to 0 words would leave the identifier naked and
        # the model would refuse to rank it. 8 is what rank_llm effectively
        # converges to for extreme cases.
        self.min_doc_words = int(rc.get("min_doc_words", 8))
        self.return_logprobs = bool(rc.get("return_logprobs", False))
        revision = rc.get("revision")

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = _resolve_dtype(requested_dtype, device)
        self.device = device
        self.dtype = dtype
        self.torch = torch

        self.logger.info(
            "Loading %s on device=%s dtype=%s protocol=%s window=%d stride=%d",
            model_name,
            device,
            dtype,
            self.rankzephyr_protocol,
            self.window_size,
            self.stride,
        )
        if self.rankzephyr_protocol == RANKZEPHYR_PROTOCOL_BLOCK_LOCAL_RRF:
            self.logger.info(
                "RankZephyr tier 3: block_local_rrf block_size=%d rrf_k=%s (not literature L1)",
                self.block_size,
                self.rrf_k,
            )
        if revision:
            self.logger.info("Pinned HF revision=%s", revision)

        hf_kwargs: dict = {"revision": revision} if revision else {}

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **hf_kwargs)
        # Decoder-only models need left padding for correct generation when
        # we ever batch; we don't batch today but setting it now costs
        # nothing and removes a future footgun.
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            **hf_kwargs,
        )
        self.model.to(device)
        self.model.eval()

        # Cache the per-window reserved-output-token estimate. Mirrors
        # ``rank_llm.RankListwiseOSLLM.num_output_tokens(window_size)``:
        # tokenise the identifier permutation string the model is supposed
        # to emit ("[1] > [2] > ... > [W]") and use its length as the
        # generation budget. For W=20 on this tokenizer this is ~90 tokens.
        self._reserved_output_tokens_cache: dict[int, int] = {}

        # Track how often the shrink loop fires: on BEIR corpora we expect
        # it to trigger every window; on DL19/DL20 never. Exposed via
        # ``get_prompt_stats()`` for introspection + test assertions.
        self._shrink_iter_total = 0
        self._overflow_window_count = 0
        self._window_count = 0

    # -- top-level API ---------------------------------------------------------

    def rank(self, query: Query, passages: list[Passage]) -> RankResult:
        if not passages:
            return {
                "top_k_psgs": [],
                "scores_init_order": None,
                "prompting_runtimes": [],
                "paradigm": self.paradigm,
            }

        current = list(passages)
        runtimes: list[float] = []
        n = len(current)

        if self.rankzephyr_protocol == RANKZEPHYR_PROTOCOL_BLOCK_LOCAL_RRF:
            return self._rank_block_local_rrf(query, current, runtimes)

        for start, end in _window_schedule(n, self.window_size, self.stride):
            t0 = time.perf_counter()
            local_perm = self._rerank_window(query["text"], current[start:end])
            runtimes.append(time.perf_counter() - t0)
            current = _apply_window_permutation(current, start, end, local_perm)

        top_k = [{"pid": p["pid"], "text": p["text"], "score": None} for p in current]
        return {
            "top_k_psgs": top_k,
            # Pure generative-listwise: no per-doc scalar. TREC writer fills
            # in score = N - rank_idx on None; see utils.trec.write_trec_run.
            "scores_init_order": None,
            "prompting_runtimes": runtimes,
            "paradigm": self.paradigm,
        }

    def _rank_block_local_rrf(
        self,
        query: Query,
        passages: list[Passage],
        runtimes: list[float],
    ) -> RankResult:
        """Tier 3: one listwise decode per disjoint block, then RRF merge.

        Does **not** update a shared global list between blocks — see module
        docstring *Inference protocols*.
        """
        orig_index_by_pid = {str(p["pid"]): i for i, p in enumerate(passages)}
        blocks = _contiguous_passage_blocks(passages, self.block_size)
        ranked_blocks: list[list[Passage]] = []
        for block in blocks:
            t0 = time.perf_counter()
            local_perm = self._rerank_window(query["text"], block)
            runtimes.append(time.perf_counter() - t0)
            reordered = _apply_window_permutation(list(block), 0, len(block), local_perm)
            ranked_blocks.append(reordered)
        merged = _merge_rrf_block_rankings(ranked_blocks, self.rrf_k, orig_index_by_pid)
        top_k = [{"pid": p["pid"], "text": p["text"], "score": None} for p in merged]
        return {
            "top_k_psgs": top_k,
            "scores_init_order": None,
            "prompting_runtimes": runtimes,
            "paradigm": self.paradigm,
        }

    # -- inner steps -----------------------------------------------------------

    def get_prompt_stats(self) -> dict:
        """Return aggregate counters on prompt construction since instance
        creation. Used in tests + by the unified entrypoint to surface
        overflow rates in the run log. Keys:
        - ``n_windows``: total number of ``_rerank_window`` calls.
        - ``n_overflow_windows``: calls where shrink loop fired ≥ once.
        - ``total_shrink_iters``: cumulative shrink iterations.
        """
        return {
            "n_windows": self._window_count,
            "n_overflow_windows": self._overflow_window_count,
            "total_shrink_iters": self._shrink_iter_total,
        }

    def _reserved_output_tokens(self, window_size: int) -> int:
        """Estimate how many tokens ``generate()`` will need to emit for a
        window of size ``W``. Mirrors rank_llm's ``num_output_tokens``:
        tokenise the literal ``"[1] > [2] > ... > [W]"`` string and count.
        Cached per-``W`` (typically called with self.window_size once).
        """
        cached = self._reserved_output_tokens_cache.get(window_size)
        if cached is not None:
            return cached
        ids_str = " > ".join(f"[{i + 1}]" for i in range(max(window_size, 1)))
        n = len(self.tokenizer.encode(ids_str, add_special_tokens=False))
        self._reserved_output_tokens_cache[window_size] = n
        return n

    def _rerank_window(self, query_text: str, window: list[Passage]) -> list[int]:
        """Run a single listwise decoding pass over ``window`` and return
        a 0-indexed permutation of ``range(len(window))``.

        Builds the chat-templated prompt once at ``self.max_doc_words``. If
        the result exceeds ``max_total_tokens - reserved_output_tokens``,
        iteratively shrinks the per-doc word cap using the same step
        formula as ``rank_llm.RankListwiseOSLLM.create_prompt``:

            max_length -= max(1, (excess_tokens) // (W * 4))

        until either the prompt fits or ``max_length`` hits ``min_doc_words``.
        In the latter case we give up on shrinking and ship the overfull
        prompt anyway — better to have some attention signal than to pass
        the model an empty list.

        The loop is bounded by the monotonic word cap itself, not by a small
        fixed iteration limit: each iteration lowers ``max_length`` by at
        least one word, so it must terminate after at most
        ``max_doc_words - min_doc_words + 1`` shrink attempts.
        """
        self._window_count += 1
        W = len(window)
        budget = self.max_total_tokens - self._reserved_output_tokens(W)
        max_length = self.max_doc_words
        fired = False

        # Bound on shrink attempts. This should never be reached before
        # fitting/min_doc_words because step >= 1, but keeping the guard makes
        # the termination argument explicit for future tokenizer changes.
        max_shrink_attempts = max(self.max_doc_words - self.min_doc_words + 1, 1)

        for _ in range(max_shrink_attempts):
            docs = [
                _prepare_doc_content(
                    _apply_optional_char_cap(p["text"], getattr(self, "max_doc_chars", None)),
                    max_length,
                )
                for p in window
            ]
            prompt_str = self._render_prompt(query_text, docs)
            n_tokens = len(self.tokenizer(prompt_str, add_special_tokens=False)["input_ids"])
            if n_tokens <= budget:
                break
            fired = True
            self._shrink_iter_total += 1
            excess = n_tokens - budget
            # rank_llm uses W*4 in the denominator ("4 tokens per word" is a
            # deliberate overestimate so the loop under-shrinks and tolerates
            # multiple passes: ~1.3 tokens/word empirically). We keep the
            # formula identical so effective ``max_doc_words`` matches the
            # rank_llm shrink formula.
            step = max(1, excess // max(W * 4, 1))
            new_length = max_length - step
            if new_length <= self.min_doc_words:
                max_length = self.min_doc_words
                # Rebuild one final time at the floor and ship, even if still
                # over budget: see docstring.
                docs = [_prepare_doc_content(p["text"], max_length) for p in window]
                prompt_str = self._render_prompt(query_text, docs)
                self.logger.warning(
                    "Shrink loop hit min_doc_words=%d but prompt still %d tokens "
                    "(> budget=%d). Proceeding with overfull prompt; attention may "
                    "be degraded on early identifiers.",
                    max_length,
                    len(self.tokenizer(prompt_str, add_special_tokens=False)["input_ids"]),
                    budget,
                )
                break
            max_length = new_length
        else:
            # Defensive only: with step >= 1 and the min_doc_words check above,
            # this means a future edit broke the monotonicity invariant.
            docs = [_prepare_doc_content(p["text"], max_length) for p in window]
            prompt_str = self._render_prompt(query_text, docs)
            n_tokens = len(self.tokenizer(prompt_str, add_special_tokens=False)["input_ids"])
            self.logger.warning(
                "Shrink loop exhausted %d monotonic attempts at max_length=%d "
                "(prompt=%d tokens, budget=%d). Proceeding with current prompt; "
                "this indicates a shrink-loop invariant regression.",
                max_shrink_attempts,
                max_length,
                n_tokens,
                budget,
            )

        if fired:
            self._overflow_window_count += 1

        generation = self._generate(prompt_str)
        return _parse_permutation(generation, W)

    def _render_prompt(self, query_text: str, docs: list[str]) -> str:
        user = _build_user_prompt(query_text, docs)
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]
        # Zephyr-Beta's chat template produces
        # `<|system|>\n...</s>\n<|user|>\n...</s>\n<|assistant|>\n` —
        # apply_chat_template handles the special tokens, so we never
        # hand-write them.
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def _generate(self, prompt_str: str) -> str:
        inputs = self.tokenizer(
            prompt_str,
            return_tensors="pt",
            add_special_tokens=False,  # chat template already injected them
        ).to(self.device)

        gen_kwargs: dict = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": False,
            "num_beams": 1,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if self.return_logprobs:
            # Opt-in wiring for the log-prob ablation; see module docstring.
            gen_kwargs["return_dict_in_generate"] = True
            gen_kwargs["output_scores"] = True

        with self.torch.inference_mode():
            output = self.model.generate(**inputs, **gen_kwargs)

        if self.return_logprobs:
            sequences = output.sequences
        else:
            sequences = output

        # Strip prompt tokens; decode only the newly generated tail so the
        # parser doesn't re-match any ``[N]`` identifiers that appeared in
        # the passages themselves.
        prompt_len = inputs["input_ids"].shape[1]
        new_tokens = sequences[0, prompt_len:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)
