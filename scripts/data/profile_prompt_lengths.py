"""Profile per-chunk prompt token-length distribution for self-distill SFT.

Self-distill uses a setwise prompt: each chunk packs *all* docs (chunk_size,
default 20) into a single prompt with a `Grades:` block. The trainer then
forwards `slot_forward_batch_size` slot-prefixes at a time, where each
prefix is the full prompt up to that slot's grade-readout position. So
``max_length`` is the budget for a 20-doc prompt, not a single (q, d) pair.

Samples queries from a fixture, builds the actual prompt with the production
prefix-builder + tokenizer, and reports:
  - Distribution of pre-truncation token lengths (min/median/P90/P95/P99/max).
  - Fraction of chunks that exceed each candidate ``max_length`` cap.
  - Wallclock cost estimate per max_length (attention is quadratic).

Run::

    uv run python scripts/data/profile_prompt_lengths.py \
        --fixture data/msmarco-train-selfdistill-seed42/fixture.jsonl \
        --tokenizer Qwen/Qwen3-Reranker-4B \
        --reranker-class Qwen3Reranker \
        --chunk-size 20 \
        --max-doc-chars 1200 \
        --n-queries 300
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

# Make the in-repo package importable when run directly.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def _percentile(values: list[int], p: float) -> int:
    """Inclusive percentile via nearest-rank (integer-valued samples)."""
    if not values:
        return 0
    s = sorted(values)
    if p <= 0:
        return s[0]
    if p >= 100:
        return s[-1]
    k = max(0, min(len(s) - 1, int(round(p / 100 * (len(s) - 1)))))
    return s[k]


def main() -> int:  # noqa: C901
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-Reranker-4B")
    ap.add_argument("--revision", default=None, help="Optional immutable Hugging Face commit SHA.")
    ap.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow tokenizer repository code. Remote Hub IDs also require --revision with a 40-character commit SHA.",
    )
    ap.add_argument(
        "--reranker-class",
        default="Qwen3Reranker",
        choices=["Qwen3Reranker", "Qwen3InstructGradeReranker"],
    )
    ap.add_argument(
        "--instruction", default="Given a web search query, retrieve relevant passages that answer the query"
    )
    ap.add_argument("--chunk-size", type=int, default=20)
    ap.add_argument("--max-doc-chars", type=int, default=1200)
    ap.add_argument("--n-queries", type=int, default=200, help="Number of queries to sample.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--candidate-max-lengths",
        type=int,
        nargs="+",
        default=[1024, 1536, 2048, 3072, 4096, 6144, 8192],
        help="max_length caps to evaluate truncation fraction against.",
    )
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from presentation_dependence.self_distill.student import build_prefixes_for_chunk

    if args.trust_remote_code and not Path(args.tokenizer).exists():
        revision = str(args.revision or "").strip().lower()
        if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
            ap.error("--trust-remote-code for a Hub tokenizer requires --revision with a 40-character commit SHA")
    print(f"Loading tokenizer: {args.tokenizer}")
    t0 = time.time()
    tokenizer_kwargs = {"trust_remote_code": bool(args.trust_remote_code)}
    if args.revision:
        tokenizer_kwargs["revision"] = args.revision
    tok = AutoTokenizer.from_pretrained(args.tokenizer, **tokenizer_kwargs)
    print(f"  loaded in {time.time() - t0:.1f}s")

    import random

    rng = random.Random(args.seed)

    fixture = Path(args.fixture)
    print(f"Sampling {args.n_queries} queries from {fixture}")
    queries: list[dict] = []
    with fixture.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            queries.append(json.loads(line))
            if len(queries) >= max(args.n_queries * 10, 1000):
                break
    rng.shuffle(queries)
    queries = queries[: args.n_queries]
    print(f"  loaded {len(queries)} queries")

    longest_prefix_lens: list[int] = []
    first_prefix_lens: list[int] = []
    pre_grades_lens: list[int] = []
    skipped = 0
    checked_query_present = False

    for q in queries:
        passages_all = q.get("passages") or []
        if len(passages_all) < args.chunk_size:
            skipped += 1
            continue
        passages = passages_all[: args.chunk_size]
        passages_for_prefix = [{"text": p.get("text", ""), "pid": str(p.get("pid", ""))} for p in passages]
        query_text = str(q.get("query", ""))
        prefixes = build_prefixes_for_chunk(
            reranker_class=args.reranker_class,
            tokenizer=tok,
            instruction=args.instruction,
            query_text=query_text,
            passages=passages_for_prefix,
            max_doc_chars=args.max_doc_chars,
        )
        if len(prefixes) != len(passages):
            skipped += 1
            continue
        if not checked_query_present:
            # Guard against a real footgun: a tokenizer whose OWN baked-in
            # chat_template.jinja expects a different message schema (e.g.
            # Qwen/Qwen3-Reranker-4B's <Instruct>/<Query>/<Document> slots)
            # will silently swallow `query_text` -- apply_chat_template()
            # doesn't error, it just renders an empty/garbled prompt, and
            # every downstream length stat comes out identically wrong
            # (invariant across queries -- a strong tell if you see it).
            # Use --tokenizer matching the *base model* being scored
            # (e.g. Qwen/Qwen3-4B), not a specialized reranker checkpoint's
            # tokenizer, unless you are specifically profiling that model.
            probe = " ".join(query_text.split())[:40]
            if probe and probe not in prefixes[0]:
                print(
                    f"WARNING: query text not found verbatim in the built prompt "
                    f"(checked first {len(probe)} chars). This usually means "
                    f"--tokenizer {args.tokenizer!r} has its own chat template that "
                    f"doesn't match --reranker-class {args.reranker_class!r}'s expected "
                    f"message schema, silently dropping the query. Results below are "
                    f"likely meaningless (e.g. identical across all queries) -- try "
                    f"the tokenizer of the actual model being scored instead.",
                    file=sys.stderr,
                )
            checked_query_present = True
        first_ids = tok(prefixes[0], add_special_tokens=False)["input_ids"]
        last_ids = tok(prefixes[-1], add_special_tokens=False)["input_ids"]
        first_prefix_lens.append(len(first_ids))
        longest_prefix_lens.append(len(last_ids))
        chat_prefix_end = prefixes[0].rfind("Grades:")
        if chat_prefix_end >= 0:
            pre_grades_text = prefixes[0][: chat_prefix_end + len("Grades:\n")]
            pre_grades_lens.append(len(tok(pre_grades_text, add_special_tokens=False)["input_ids"]))

    n = len(longest_prefix_lens)
    if n == 0:
        print("ERROR: no chunks profiled")
        return 1
    print(f"\nProfiled {n} chunks (skipped {skipped})")
    print(f"  reranker_class : {args.reranker_class}")
    print(f"  chunk_size     : {args.chunk_size}")
    print(f"  max_doc_chars  : {args.max_doc_chars}")
    print(f"  tokenizer      : {args.tokenizer}\n")

    def _stats(name: str, values: list[int]) -> None:
        mn = min(values)
        med = statistics.median(values)
        p75 = _percentile(values, 75)
        p90 = _percentile(values, 90)
        p95 = _percentile(values, 95)
        p99 = _percentile(values, 99)
        mx = max(values)
        mean = statistics.mean(values)
        print(f"{name}: min={mn} mean={mean:.0f} med={med} p75={p75} p90={p90} p95={p95} p99={p99} max={mx}")

    print("Per-chunk pre-truncation prompt token length:")
    _stats("  longest-slot prefix (slot N)", longest_prefix_lens)
    _stats("  shortest-slot prefix (slot 1)", first_prefix_lens)
    if pre_grades_lens:
        _stats("  prompt up to 'Grades:' header", pre_grades_lens)

    print("\nFraction of chunks truncated at each candidate max_length:")
    print(f"  {'max_length':>10} | {'pct_truncated':>13} | {'avg_overflow':>12} | {'compute_idx':>11}")
    base_len = statistics.mean(longest_prefix_lens)
    for cap in args.candidate_max_lengths:
        truncated = [v for v in longest_prefix_lens if v > cap]
        pct = 100.0 * len(truncated) / n
        overflow = statistics.mean([v - cap for v in truncated]) if truncated else 0.0
        eff_len = min(cap, base_len)
        compute_idx = (eff_len / base_len) ** 1.6
        print(f"  {cap:>10} | {pct:>12.1f}% | {overflow:>11.0f}t | {compute_idx:>10.2f}x")

    print("\nNote: compute_idx ≈ (eff_len / current_avg_len) ** 1.6 — a rough mix of")
    print("quadratic (attention) and linear (MLP/feedforward) cost on long context.")
    print("Numbers <1.0 mean faster than the current geometry, >1.0 slower.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
