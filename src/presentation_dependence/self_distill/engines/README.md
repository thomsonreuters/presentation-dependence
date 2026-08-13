# `self_distill/engines/`

Inference backends for logit-skeleton scoring. Each engine takes a batch of
prompt prefixes, each ending immediately before a grade slot, and returns the
next-token expected value over the `{0, 1, 2, 3}` grade tokens. Reranker wrappers
own prompt construction; the engine only runs the per-prefix forward pass, so
callers stay backend-agnostic.

## Files

- [`base.py`](base.py): the `LogitSkeletonEngine` ABC and its `score_prefixes`
  contract (one expected-grade scalar per prefix).
- [`hf.py`](hf.py): `HFLogitSkeletonEngine`, a HuggingFace `transformers`
  forward pass. Default backend; supports CUDA and CPU, plus best-effort MPS
  where the selected wrapper/model operations are compatible. ROCm is not a
  validated release target. Recomputes the shared prefix body once per row (no
  prefix cache).
- [`vllm.py`](vllm.py): `VLLMLogitSkeletonEngine`, using
  `SamplingParams(max_tokens=1, logprobs=N)`. With `enable_prefix_caching` it
  computes the shared chunk body once, giving a ~3-5× per-chunk speedup for the
  B=20 prompt shape. Opt-in via the `[vllm]` extra; importing the module does not
  require vLLM, only constructing the engine does.
- [`_tf5_compat.py`](_tf5_compat.py): restores
  `tokenizer.all_special_tokens_extended` (removed in transformers 5.0) for
  vLLM 0.10.x. Retained only for the Qwen3-4B-Instruct-2507 and
  Qwen3-Reranker-4B silver configs. Applied at import of `vllm.py`; the worker
  hook is gated to installed vLLM versions older than 0.12. Remove the shim
  once those two configs migrate.

## Public API

[`__init__.py`](__init__.py) exports `LogitSkeletonEngine`,
`HFLogitSkeletonEngine`, and `make_engine(kind, **kwargs)`. `make_engine`
dispatches on `"hf"` or `"vllm"` and imports the vLLM engine lazily so the HF
stack works without the optional extra.
