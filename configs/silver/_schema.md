# Silver-generation config schema

`configs/silver/` is the single namespace for every silver-label generation
job. A config is one member of the discriminated union below. The discriminator
is the required `teacher.protocol` value; consumers must not infer the variant
from a filename or model name.

Short IDs are resolved only from this directory. Explicit YAML paths remain
supported. `scripts/run_silver_generation.py` is the canonical local consumer,
and validates the selected variant before constructing either backend.

## Open-weight K-shot BSC

Discriminator: `teacher.protocol: k_shot_bsc`.

Required keys:

- `id`: stable experiment and artifact ID; production files use the same ID as
  their filename stem.
- `reranker`: the normal experiment reranker mapping, including `class` and/or
  `model_name`. Add a reviewed 40-character `revision` for publication; many
  retained standalone configs currently use `null`.
- `data`: the normal experiment data-loader mapping.
- `teacher.protocol: k_shot_bsc`.
- `teacher.k_perms`: optional positive integer (default `15`); when `seeds` is
  supplied its length must equal `k_perms`.

These configs are consumed by `KShotBSCTeacher`, and write to
`runs/self-distill/<id>/<timestamp>/silver/`.
When `reranker.inference_engine: vllm`, `./setup.sh --full` is insufficient;
run through `uv run --extra vllm ...` or the matching container.

```yaml
id: qwen3-4b-nonthink-k10-bsc-msmarco-30k
reranker:
  class: Qwen3InstructGradeReranker
  model_name: Qwen/Qwen3-4B
data:
  dataloader_class: FixtureLoader
  run_path: data/msmarco-train-selfdistill-seed42/fixture.jsonl
teacher:
  protocol: k_shot_bsc
  k_perms: 10
  seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
```

Short IDs here resolve under `configs/silver/` via
`scripts/run_silver_generation.py`; student training configs resolve under
`configs/self-distill/` via `scripts/run_self_distill_sft.py`.

### Spreading a teacher pass over several GPUs

`teacher.local_data_parallel_workers: N` runs N worker processes, each owning
one GPU via `CUDA_VISIBLE_DEVICES` and holding a full model replica at
`tensor_parallel_size: 1`. The qids are split round-robin, and the parent
merges the per-qid outputs back into the ordinary `silver_labels.jsonl` /
`manifest.json` layout, so the result is identical to a single-process run.

This is replication, not sharding: the model must fit on one device. The
tracked configs declare `8` because that is the node the study ran on. Fewer
visible GPUs is not an error: the worker count is capped at what is visible,
and with one GPU the runner says so and takes the single-process path.
`teacher.local_data_parallel_worker_stagger_s` delays each worker's start,
which helps when several engines would otherwise compile at once.
[`../../docs/HARDWARE.md`](../../docs/HARDWARE.md#using-more-than-one-gpu) puts
this beside the other multi-GPU mechanisms.

## Hosted/closed-model generated BSC

Discriminator: `teacher.protocol: closed_model_generated_bsc`.

Required keys:

- `experiment_id` or `id`; if both are present they must match.
- `input`: a SilverGenerator input mapping. `run_path` may be directly under
  `input` or under `input.data`.
- `teacher.model_id`: hosted model routing identifier.
- `teacher.protocol: closed_model_generated_bsc`.
- `prompts`: a non-empty list of registered prompt-template IDs.

Optional `cost` and `output` mappings configure budget enforcement and the run
root. These configs are consumed by `SilverGenerator`, currently through the
canonical local CLI. Output remains
`runs/silver/<experiment_id>/<timestamp>/` unless `output.base_dir` overrides
that root.

Schematic only; does not restore an unavailable production closed-model config:

```yaml
experiment_id: <stable-id>
input:
  data:
    dataloader_class: FixtureLoader
    run_path: <fixture.jsonl>
teacher:
  protocol: closed_model_generated_bsc
  model_id: <configured-hosted-model-id>
prompts:
  - <registered-prompt-template-id>
cost:
  max_budget_usd: <approved-cap>
output:
  base_dir: runs/silver/<stable-id>
```

Hosted configs must not contain open-weight `reranker` or top-level `data`
blocks. Open-weight configs must not contain a `student` block. Student
training belongs exclusively in `configs/self-distill/`.

Hosted resume keys cover query ID, document ID, prompt-template ID, and
teacher-model ID, but not every scoring parameter, endpoint, input-text hash, or
candidate-pool hash. Auto-resume is therefore safe only with byte-equivalent
config and inputs; use `--new-run` after any change. For generated BSC, validate
that every successful document retains exactly `teacher.scoring.runs` raw
scores before using the mean as silver.

## Shared optional blocks

Beyond the required keys above, a silver config may carry the same optional
blocks as an experiment config, and most tracked ones do: `eval`, `logging`,
`qids_to_run_path`, and the `execution` block.
[`../experiments/_schema.md`](../experiments/_schema.md) defines each. The
loader validates the required keys for the selected variant and ignores the
rest, so a typo in an optional block fails silently rather than loudly.
