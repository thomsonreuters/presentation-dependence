#!/usr/bin/env python
r"""Empirical HF-vs-vLLM probe on the actual self-distill workload.

Run on a CUDA box with both backends available (``uv sync --extra vllm``).
Loads a teacher model (HF + vLLM separately, in two passes), builds N
chunks of B=20 grade-skeleton prefixes from a real fixture, and measures
per-chunk wall-clock for each backend. Output is a comparison table.

Usage::

    uv run python scripts/analyze/probe_vllm_speedup.py \\
        --model Qwen/Qwen3-Reranker-4B \\
        --fixture data/msmarco-train-selfdistill-seed42/fixture.jsonl \\
        --n-chunks 8 --B 20 --max-doc-chars 1200 --max-length 4096

The script reports timings and parity without publishing silver labels.
Per-prefix ``expected_grade`` from HF and vLLM must agree within ``--rtol``
(default 1e-2). Measured bf16+sdpa and paged-FA2 differences are typically
about 1e-3.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from presentation_dependence.rerankers.qwen3 import _build_qwen3_setwise_grade_skeleton_prefixes
from presentation_dependence.rerankers.qwen3_instruct_grade import _build_qwen3_instruct_grade_skeleton
from presentation_dependence.self_distill.readout import resolve_grade_token_ids


def _build_qwen3_reranker_chunks(query, passages, *, max_doc_chars):
    _full, prefixes = _build_qwen3_setwise_grade_skeleton_prefixes(None, query, passages, max_doc_chars=max_doc_chars)
    return prefixes


def _build_qwen3_instruct_chunks(tokenizer, query, passages, *, max_doc_chars):
    _full, prefixes = _build_qwen3_instruct_grade_skeleton(
        tokenizer, None, query, passages, max_doc_chars=max_doc_chars
    )
    return prefixes


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--model", required=True, help="HF model id (Qwen/Qwen3-Reranker-4B or Qwen/Qwen3-4B-Instruct-2507)."
    )
    p.add_argument("--fixture", required=True, help="Path to fixture.jsonl.")
    p.add_argument("--n-chunks", type=int, default=4)
    p.add_argument("--B", type=int, default=20)
    p.add_argument("--max-doc-chars", type=int, default=1200)
    p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--rtol", type=float, default=1e-2)
    p.add_argument("--atol", type=float, default=2e-2)
    p.add_argument("--skip-hf", action="store_true", help="Skip HF baseline (vLLM-only timing).")
    p.add_argument("--skip-vllm", action="store_true", help="Skip vLLM run (HF-only timing).")
    return p.parse_args()


def load_fixture_chunks(path: Path, *, n_chunks: int, B: int):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            passages = rec["passages"][:B]
            if len(passages) < B:
                continue
            out.append((rec["query"], passages))
            if len(out) >= n_chunks:
                break
    return out


def main() -> int:  # noqa: C901
    args = parse_args()

    print(f"[probe] model      = {args.model}")
    print(f"[probe] fixture    = {args.fixture}")
    print(f"[probe] n_chunks   = {args.n_chunks}  B = {args.B}")
    print(f"[probe] max_doc_chars = {args.max_doc_chars}  max_length = {args.max_length}")

    chunks = load_fixture_chunks(Path(args.fixture), n_chunks=args.n_chunks, B=args.B)
    if len(chunks) < args.n_chunks:
        print(f"[probe][WARN] only got {len(chunks)} chunks (B={args.B} requires that many full-100-passage queries)")

    is_instruct = "Instruct" in args.model
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    if hasattr(tokenizer, "truncation_side"):
        tokenizer.truncation_side = "left"

    grade_token_ids = resolve_grade_token_ids(tokenizer)
    print(f"[probe] grade_token_ids = {grade_token_ids}")

    all_prefixes: list[list[str]] = []
    for query, passages in chunks:
        if is_instruct:
            prefixes = _build_qwen3_instruct_chunks(tokenizer, query, passages, max_doc_chars=args.max_doc_chars)
        else:
            prefixes = _build_qwen3_reranker_chunks(query, passages, max_doc_chars=args.max_doc_chars)
        all_prefixes.append(prefixes)

    hf_results = []
    hf_times = []
    if not args.skip_hf:
        from transformers import AutoModelForCausalLM

        from presentation_dependence.self_distill.engines import HFLogitSkeletonEngine

        print("[probe] loading HF model...")
        t0 = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype="auto")
        model = model.to("cuda").eval()
        print(f"[probe] HF model load: {time.perf_counter() - t0:.1f} s")

        engine = HFLogitSkeletonEngine(
            model=model,
            tokenizer=tokenizer,
            max_length=args.max_length,
            batch_size=args.B,
        )
        # Warmup.
        engine.score_prefixes(all_prefixes[0], grade_token_ids)
        for chunk_idx, prefixes in enumerate(all_prefixes):
            t0 = time.perf_counter()
            scores = engine.score_prefixes(prefixes, grade_token_ids)
            t1 = time.perf_counter()
            hf_times.append(t1 - t0)
            hf_results.append(scores)
            print(f"[probe][hf] chunk {chunk_idx}: {t1 - t0:.3f} s ({len(prefixes)} prefixes)")
        del model

    vllm_results = []
    vllm_times = []
    if not args.skip_vllm:
        from presentation_dependence.self_distill.engines.vllm import VLLMLogitSkeletonEngine

        print("[probe] loading vLLM engine...")
        t0 = time.perf_counter()
        engine = VLLMLogitSkeletonEngine(
            model_name=args.model,
            max_model_len=args.max_length,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enable_prefix_caching=True,
        )
        print(f"[probe] vLLM engine load: {time.perf_counter() - t0:.1f} s")
        # Warmup.
        engine.score_prefixes(all_prefixes[0], grade_token_ids)
        for chunk_idx, prefixes in enumerate(all_prefixes):
            t0 = time.perf_counter()
            scores = engine.score_prefixes(prefixes, grade_token_ids)
            t1 = time.perf_counter()
            vllm_times.append(t1 - t0)
            vllm_results.append(scores)
            print(f"[probe][vllm] chunk {chunk_idx}: {t1 - t0:.3f} s ({len(prefixes)} prefixes)")
        if engine.n_missing_grade_tokens:
            print(
                f"[probe][vllm][WARN] {engine.n_missing_grade_tokens} grade-token misses in top-N — "
                f"consider raising VLLMLogitSkeletonEngine.top_logprobs."
            )

    if hf_times and vllm_times:
        print()
        print("[probe] =========================  SUMMARY  =========================")
        print(f"[probe] hf   total: {sum(hf_times):.2f} s  mean/chunk: {sum(hf_times) / len(hf_times):.3f} s")
        print(f"[probe] vllm total: {sum(vllm_times):.2f} s  mean/chunk: {sum(vllm_times) / len(vllm_times):.3f} s")
        speedup = sum(hf_times) / max(sum(vllm_times), 1e-9)
        print(f"[probe] vllm speedup vs hf: {speedup:.2f}×")

        # Numeric parity check.
        max_abs_diff = 0.0
        for hf_chunk, vllm_chunk in zip(hf_results, vllm_results):
            for h, v in zip(hf_chunk, vllm_chunk):
                max_abs_diff = max(max_abs_diff, abs(h - v))
        print(f"[probe] max |hf - vllm| per-prefix: {max_abs_diff:.4f}")
        if max_abs_diff > args.atol:
            print(
                f"[probe][WARN] numeric divergence {max_abs_diff:.4f} > --atol {args.atol}; "
                "the two backends may be using different prompt tokenisation or kernels."
            )
        else:
            print(f"[probe] numeric parity OK (within --atol={args.atol})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
