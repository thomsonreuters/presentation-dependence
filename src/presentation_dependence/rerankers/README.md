# Reranker Development Checklist

Each reranker wraps one model family. Keep model-specific prompt templates,
parsing, and loading quirks local to that wrapper; share only generic contract
helpers.

## File map

### Contract and registry

- [`base.py`](base.py): the abstract `Reranker` interface, `RankResult`, and the
  paradigm/support flags.
- [`registry.py`](registry.py): the `class name -> class` map
  (`RERANKER_CLASSES`, `get_reranker_class`); one line per wrapper.

### Shared helpers

- [`_rank_result.py`](_rank_result.py): scalar-score to `RankResult` conversion.
- [`_torch_utils.py`](_torch_utils.py): `device: auto` / `dtype: auto`
  resolution.
- [`_windowing.py`](_windowing.py): pure helpers for generative-listwise window
  reranking.
- [`_validation.py`](_validation.py): `validate_rank_result` contract guard for
  tests and debug.
- [`_grade_skeleton.py`](_grade_skeleton.py): fixed-grade prompt helpers for
  generic chat SLM grade scorers.
- [`_slot_tokenization.py`](_slot_tokenization.py): locates each slot's readout
  position in a shared prompt, with left truncation. Shared by the grade
  skeleton and the Qwen3 setwise yes/no mode.
- [`_pairrm_model.py`](_pairrm_model.py): vendored `DebertaV2PairRM` head.
- [`grade_rubrics.py`](grade_rubrics.py): shared grade rubrics and the
  expected-grade readout user-prompt body for open and closed models.

### Input-symmetry variants

- [`render_variants.py`](render_variants.py): render-variant operators for
  batched-pointwise scoring (`supports_render_variants`).
- [`scale_variants.py`](scale_variants.py): scoring-scale variants
  (`supports_scale_variants`).

### Grade scorers (trained targets)

- [`qwen3_instruct_grade.py`](qwen3_instruct_grade.py): Qwen3-4B-Instruct as a
  shared-context batched-pointwise grade scorer (the primary trained target).
- [`granite_41.py`](granite_41.py): Granite 4.1 8B continuous expected-grade
  scorer.
- [`gemma4.py`](gemma4.py): Gemma-4-E4B-it grade scorer (`Gemma4GradeReranker`).
- [`qwen3.py`](qwen3.py): Qwen3-Reranker-4B official yes/no-logit scorer.

### Baselines and references

- [`rankzephyr.py`](rankzephyr.py): RankZephyr generative-listwise baseline.
- [`jina_listwise.py`](jina_listwise.py): jina-reranker-v3 scoring-listwise.
- [`mxbai_pointwise.py`](mxbai_pointwise.py): mxbai-rerank-large-v2 pointwise
  cross-encoder.
- [`skywork_bt.py`](skywork_bt.py): Skywork-Reward-V2 Bradley-Terry scalar
  reward (response ranking).
- [`pairrm.py`](pairrm.py): PairRM pairwise anchor (response ranking).
- [`closed_model_genbsc.py`](closed_model_genbsc.py): closed-model (GPT-5.x)
  generated-BSC baseline.

### Wrappers and controls

- [`capcal.py`](capcal.py): content-free calibration wrapper around another
  scorer.
- [`pine.py`](pine.py): PINE HF/eager attention-invariance reranker.
- [`identity.py`](identity.py): passes the first-stage order through unchanged.

## Add a New Reranker

1. Create `src/presentation_dependence/rerankers/<model_name>.py`.
2. Subclass `Reranker` from `presentation_dependence.rerankers.base`.
3. Set `paradigm` to one of:
   - `pointwise`
   - `scoring_listwise`
   - `generative_listwise`
   - `batched_pointwise`
4. Implement `rank(query, passages)` without mutating inputs.
5. Return a `RankResult` with:
   - `top_k_psgs` ordered best-first.
   - `scores_init_order` aligned to the input passages when scalar scores exist.
   - `scores_init_order=None` only for true rank-only generative listwise models.
   - `prompting_runtimes` as one value per model forward/generation.
   - `paradigm` matching the class attribute.
6. Handle `passages == []` explicitly and return an empty result.
7. Register the class in `registry.py`.
8. Update `configs/experiments/_schema.md` if the config keys are reusable.
9. Update `eval/tau_psi_geometry.py` if τ-PSI@B needs a dedicated branch, a
   batched-pointwise inclusion, or an exclusion.
10. Verify the wrapper against `validate_rank_result` and a smoke config.
11. Record the checkpoint's licence in the module docstring and in the licence
    table in `docs/MODELS.md`. If the wrapper adapts upstream source, or
    reproduces a prompt template or scoring procedure, add an entry to
    `THIRD_PARTY_NOTICES.md` naming the upstream project, its licence, and how
    this copy differs. A code repository's licence can differ from the
    licence on the corresponding weights.

## Shared Helpers

- Use `_torch_utils.py` for `device: auto` / `dtype: auto` behavior.
- Use `_windowing.py` only for generic window-list operations.
- Use `_rank_result.py` for generic scalar-score to `RankResult` conversion.
- Use `_validation.py::validate_rank_result()` in tests or debug code.
- Use `capcal.py::CapCalReranker` when a baseline is a pure probability-space
  calibration around an existing reranker. Do not fold CapCal knobs into the
  base scorer; the wrapped base should remain runnable uncalibrated.

## Contract Validation

`validate_rank_result(result, passages)` is a lightweight guard for tests and
new-model debugging. It checks that output pids come from the input, there are
no duplicate pids, score vectors align with the input length, empty inputs
produce empty outputs, and required metadata fields are present.

Use it in pure helper tests and smoke tests:

```python
from presentation_dependence.rerankers import validate_rank_result

result = reranker.rank(query, passages)
validate_rank_result(result, passages)
```

Do not add it unconditionally to hot inference paths. If a new model is under
active development and you need runtime checking, gate it behind a debug config
flag so benchmark runs do not pay the extra validation overhead.

```yaml
debug:
  validate_rank_result: true
```

With this flag enabled, `ExperimentManager` and `PsiExperimentRunner` validate
each reranker output before writing artifacts.

## Do Not Generalize

Do not merge model prompts, output parsers, chat templates, or document
preparation logic unless there is strong evidence that they are truly the same.
These details are often training-distribution sensitive and should stay close
to the model wrapper.

Architectural baselines such as PINE are not score post-processors. If a PINE
config is used, it must route through the dedicated HF/eager PINE path in
`pine.py`; do not publish permutation ensembling or calibration under an
architectural-invariance label.

## Checking a wrapper

The identity smoke validates only shared pipeline plumbing; changing its class
to an arbitrary wrapper does not create a valid model-specific config and may
require network/GPU access. First create a complete tracked example with every
required key, then validate its plan and run it against a fixed fixture on the
supported backend:

```bash
uv run python scripts/run_experiment.py -e <new-example.yaml> --dry-run
uv run python scripts/run_experiment.py -e <new-example.yaml>
```

Add a public unit/contract test for empty input, pid/order preservation, score
length/finite values, and backend-specific parsing; do not treat the unchanged
identity smoke as wrapper validation.
