# The expected-grade readout

How a score is extracted from a scorer's logits.

Read this before using a released adapter. Loading the same weights with a
different readout produces different scores, and every reported result uses the
readout defined here.

`Path-C`, the readout ID `pathc_expected_grade_logits_v1`, and the template ID
`pathc_grade_v1` also exist in code and configs as compatibility identifiers;
see the README [Legacy identifiers](../README.md#legacy-identifiers). They all
refer to this readout.

## Identity

```yaml
readout_id: pathc_expected_grade_logits_v1
paradigm: batched_pointwise
raw_grade_range: [0, 3]
reranker_score_range: [0, 1]
score_alignment: input_order
```



## Input

```json
{
  "query": "text",
  "passages": [{"pid": "p1", "text": "text"}]
}
```

Passage IDs must be unique. The implementation must retain input order and
return scores in that order. Empty input returns an empty result.

## Rubric and prompt

The config identifies:

- instruction;
- grade rubric;
- chat-template kwargs;
- document cap;
- optional render variant.

Each item is assigned a numbered slot. The prompt contains the task rubric,
query/prompt, numbered items, and a fixed answer skeleton.

The default scale is:

```text
0 = no useful relevance/support/quality
1 = weak relevance/support/quality
2 = strong but incomplete relevance/support/quality
3 = direct/complete relevance/support/quality
```

Task-specific wording is defined by the selected rubric.

## Chunk geometry

Geometry fields:

```yaml
docs_per_score_forward: 20
chunk_assignment: sorted
presentation_order: forward
```

`sorted` uses contiguous input chunks. `interleaved` is the round-robin control.
Each chunk shares one model context. Scores are restored to request order before
ranking.

## Fixed grade skeleton

Rendered skeleton:

```text
[1] Grade: 0
[2] Grade: 0
...
[B] Grade: 0
```

The dummy grade fixes the readout position; it is not the predicted score.

Prefix modes:

- `cumulative`: later slot prefixes include earlier skeleton lines;
- `independent`: each slot uses an independent neutral prefix.

Prefix mode and dummy grade are configuration fields, and both are part of a
result's identity.

## Grade tokens

The default grade strings are `("0", "1", "2", "3")`.

For the pinned tokenizer:

1. encode without special tokens;
2. each grade must be exactly one token;
3. grade token IDs must be distinct;
4. leading-space tokenization is used only when declared.

Tokenization failure is an error rather than a multi-token approximation.

## Score calculation

At each slot, extract the four grade-token logits/logprobs.

```text
p = softmax([logit_0, logit_1, logit_2, logit_3])
raw_expected_grade = 0*p_0 + 1*p_1 + 2*p_2 + 3*p_3
reranker_score = raw_expected_grade / 3
```

The raw expected grade is in `[0,3]`; the normalized reranker output is in
`[0,1]`. Ranking uses the continuous score, not the most likely integer grade.

Silver-label code may convert the normalized score back to `[0,3]` when storing
training targets. That is a storage convention, not the reranker API.

## Output

Logical `RankResult`:

```json
{
  "top_k_psgs": [{"pid": "p1", "text": "text"}],
  "scores_init_order": [0.91],
  "grade_probabilities_init_order": [[0.01, 0.03, 0.21, 0.75]],
  "prompting_runtimes": [0.12],
  "paradigm": "batched_pointwise"
}
```

`top_k_psgs` is best-first. Scores and optional probability vectors align with
the original input order.

## HF implementation

HF reads full-vocabulary logits at fixed grade positions.

Supported strategies:

- `per_prefix`: one causal prefix per slot;
- `single_forward`: one skeleton forward with indexed slot positions.

The config selects the strategy. Slot positions are recomputed after
tokenization and truncation.

## vLLM implementation

The default vLLM path is `per_prefix`:

1. submit each slot prefix;
2. request one next-token position at temperature 0;
3. request top logprobs;
4. extract grade-token logprobs;
5. apply the same expected-grade calculation.

The research default is `top_logprobs=50`. Missing grade-token lookups are
counted and must be zero for validated runs.

The vLLM interior-position `prompt_logprobs` path is not treated as equivalent
to the default per-prefix path in this repository.

## Truncation

- document text is normalized and capped by `max_doc_chars`;
- model input is capped by `max_length`/`max_model_len`;
- wrappers use the configured truncation direction;
- grade positions must remain resolvable after truncation.



## LoRA and merged weights

The readout is the same for:

- base weights;
- base plus LoRA adapter;
- merged base/adapter weights.

Weight-loading mode is recorded separately from the readout.

## Determinism and parity

Research validation fixes model/tokenizer revision, prompt settings, chunk
assignment, temperature, and runtime image.

To compare two implementations, run both over a fixed fixture and check:

- rendered prompt or prompt hash;
- grade token IDs;
- probability vectors;
- input-aligned scores;
- final ranking;
- missing-token and truncation behavior.

Choose the numerical tolerance to suit the comparison; this document does not
fix one.

## Versioning

Create a new readout version when changing:

- grade values or normalization;
- prompt/skeleton semantics;
- prefix semantics;
- score alignment;
- missing-token score behavior.

Model-specific token IDs, revisions, and whitespace flags are manifest/config
parameters when the readout semantics remain unchanged.

## What does not use this readout

The hosted closed-model path in `[TRAINING.md](TRAINING.md#hosted-teachers)`
decodes integer grades rather than reading logits at fixed positions, so it does
not implement this readout. Its results are labelled with a different protocol
for that reason.