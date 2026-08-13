# Student training config schema (`configs/self-distill/<ID>.yaml`)

One YAML per student training run. These are the configs that turn silver labels
into a LoRA adapter. The teacher pass that produces those labels is separate and
lives in [`../silver/_schema.md`](../silver/_schema.md).

Run one with:

```bash
uv run python scripts/run_self_distill_sft.py -e <ID>
```

`scripts/run_self_distill_sft.py` resolves the config, applies `--override`
flags, and hands it to `train_student_sft_from_config` in
`src/presentation_dependence/self_distill/student.py`. Add `--dry-run` to print the
resolved plan and check that the declared inputs exist without loading a model.
`scripts/train/entrypoint.py` is the container-side path and reads the same
schema.

## Filename conventions

The filename stem must equal `id`; this holds for every tracked config and run
directories key off it. Names read left to right as base model, label regime,
objective, and training dataset:

- Base: `qwen3-4b-nonthink`, `gemma4`, `granite-41-8b`.
- Labels: `k1` (one presentation) or `k10` (order-averaged over ten).
- Objective: `sft` for plain regression, `supervised-consistency-lambda<code>-warmup<steps>` for OC-SFT, or a named ablation such as `debias-first` or `pos-aug-only`.
- Dataset: `msmarco-30k`, `hotpotqa-support-30k`, `ultrafeedback-rq-30k`.

Prefer adding a new YAML over mutating a tracked one.

## Top-level fields

All six are present in every tracked config.

| field       | type   | required | description                                                                                     |
| ----------- | ------ | -------- | ------------------------------------------------------------------------------------------------- |
| `id`        | string | yes      | Stable ID, equal to the filename stem.                                                          |
| `reranker`  | object | yes      | How the student is served during in-training evaluation, not how it is trained. Same schema as `reranker:` in [`../experiments/_schema.md`](../experiments/_schema.md) and dispatched through the same registry. |
| `student`   | object | yes      | Everything about training: see below.                                                           |
| `eval`      | object | yes      | `measures` (a list, `[ndcg_cut_10]` throughout) and `strict_qrels_filter`.                      |
| `execution` | object | yes      | `distributed: ddp` and `fixture_channel`, the data channel the fixture is read from.            |
| `logging`   | object | yes      | `level`, `INFO` throughout.                                                                     |

`execution.distributed: ddp` takes effect only when more than one GPU is
visible. Both `scripts/run_self_distill_sft.py` and the container entrypoint
honour it, re-execing under torchrun with one worker per GPU; they read
`SLM_NUM_GPUS` (falling back to the legacy `SM_NUM_GPUS` alias), and warn and
run single-process when they find one GPU. See
[`../../docs/HARDWARE.md`](../../docs/HARDWARE.md#using-more-than-one-gpu).

## `student:` block

### Model and paths

| field                     | description                                                                                     |
| ------------------------- | ------------------------------------------------------------------------------------------------- |
| `base_model`              | Hugging Face id or immutable local snapshot path of the model LoRA adapts. The student loader has no separate revision field; use a local snapshot for citable runs. |
| `reranker_class`          | Wrapper used to score during in-training evaluation. Matches `reranker.class`.                  |
| `silver_labels_path`      | Training labels. Required for every objective that regresses on silver.                         |
| `eval_silver_labels_path` | Labels for the in-training evaluation subset, which must be a subset of the fixture.            |
| `fixture_path`            | `fixture.jsonl`, the candidate pool the labels index into.                                      |
| `qrels_path`              | `qrels.txt`, needed whenever a qrels-based selection metric is used.                            |
| `output_dir`              | Where training artifacts and, by default, checkpoints are written. Source templates are trial-less; use explicit trial-shaped overrides for reproduction collection. |
| `device`, `dtype`         | `cuda` and `bfloat16` throughout.                                                               |
| `attn_implementation`     | Tracked Qwen and Granite student configs set `flash_attention_2`; Gemma configs set `sdpa`. `flash-attn` is **not** in `uv.lock` and is an optional CUDA install when you want FA2. Without it, override to `sdpa` (or use a Gemma recipe as written). There is no package-level default: omit the key and Transformers picks its own backend. |
| `instruction`, `grade_rubric_id`, `max_length`, `max_doc_chars` | Prompt-side settings; they must match the teacher that wrote the labels, or the student trains against prompts it will never see. |

### `lora:`

`r: 16`, `alpha: 32`, `dropout: 0.05`, and a `target_modules` regex covering the
attention and MLP projections. Identical across all tracked configs; the
regex is written to match the `language_model.layers.*` path shared by the
Qwen, Gemma, and Granite wrappers.

### `data:`

| field                      | description                                                                                   |
| -------------------------- | ------------------------------------------------------------------------------------------------ |
| `candidate_set_id`         | Names the pool the fixture came from, e.g. `msmarco_seed42_bm25_top100`.                      |
| `chunk_size`               | Documents per training chunk. Match it to the serving width you intend to evaluate at.        |
| `include_doc_ids_in_prompt`| `false` throughout.                                                                            |
| `preserve_group_metadata`  | Keeps per-query grouping so permutation views and the PSI proxy can be built.                 |
| `token_cache`              | `enabled`, `max_entries`, `cache_prefix_strings`, `cache_token_ids`. Reuses tokenization across the shared prompt prefix. |
| `permutation_augmentation` | Optional. `enabled`, `seeds`, `views_per_epoch`, `chunk_sizes`, `shuffle_rate`. Trains on shuffled presentations of the same chunk. |
| `view`                     | Descriptive only. Nothing reads it; the chunk builder is selected from `objective.type`. Keep it mirroring that field, or delete both the key and this row. |

### `objective:`

Absent means `supervised_mse` with `loss: mse`.

| field    | description                                                                                          |
| -------- | -------------------------------------------------------------------------------------------------- |
| `type`   | One of `supervised_mse`, `supervised_consistency` (OC-SFT), `mean_teacher`, `kl_to_base`. Validated; anything else raises. |
| `loss`   | One of `mse`, `kl_vector`, `combined`. Only the two supervised types accept anything but `mse`.      |
| `lambda` | OC-SFT only. Weight on the two-view consistency penalty. The tracked grid is 0.5 to 5.0.             |
| `lambda_warmup` | OC-SFT only. `steps`, `init`, `schedule`, e.g. 500 steps linear from 0.                       |
| `view_seeds`    | OC-SFT only. The two shuffled views compared each step, `[0, 1]` throughout.                  |
| `ips`    | DebiasFirst only. `enabled`, `relevance_threshold`, `smoothing_eps`, `clip`. Requires `loss: mse`. Uniform weights reduce the objective to the augmentation-only arm exactly. |

Pick λ with `scripts/select_lambda.py` rather than by hand or by a global
argmax; see [`../../docs/EVAL-PROTOCOL.md`](../../docs/EVAL-PROTOCOL.md).

### `training:`

Optimization is `lr: 2e-4` with `lr_scheduler: cosine` and
`warmup_ratio: 0.1`, `epochs: 1`, `weight_decay: 0.01`, `adam_beta1: 0.9`,
`adam_beta2: 0.95`, and `seed: 42`. `max_steps: null` means run the epoch out.

Throughput is `batch_size_per_device: 1` with `grad_accumulation_steps: 8` and
`gradient_checkpointing: true`, which is what fits a 4B student with a
twenty-document chunk on one 48 GB card. `slot_forward_batch_size` bounds the
grade-slot forward pass, `single_forward_readout` computes all slot logits in
one pass, and `ddp_no_sync_during_grad_accum` skips gradient sync on
accumulation steps.

Prefer these tracked values over ad-hoc overrides when a run does not fit.
Changing serving width, rubric, truncation, or the readout to avoid an OOM
changes the method and needs a new config.

Progress reporting is `log_every_n_steps`, `progress_jsonl`, and
`heartbeat_every_s` (a liveness line so a silent run is distinguishable from a
hung one). The current progress logger recreates its JSONL file when a training
process starts, including resume, so checkpoint state, not `progress.jsonl`, is
the durable resume authority.

### `checkpoint:`

`save_every_n_steps` sets the cadence; `keep_last_n`, `keep_n_best`,
`keep_n_best_metric`, and `keep_n_best_mode` bound what survives on disk, and
`keep_every_n_steps` pins a coarse trajectory beyond those. Selection reads the
retained set, so keeping too few checkpoints can discard the one
`select_lambda.py` would have chosen. `resume_from_checkpoint: null` starts
fresh. Optional `dir` moves resumable checkpoints outside `output_dir`; the
collector-compatible convention is
`checkpoints/<generated-id>/<trial>/student/`, parallel to
`runs/<generated-id>/<trial>/student/`.

### `evaluation:` and `observability:`

`evaluation` runs the in-training quality probe: `max_queries`,
`every_n_steps`, `at_start`, `at_end`. Its optional `psi` block
(`enabled`, `max_queries`, `seeds`) adds the order-stability proxy on a smaller
subset. These are diagnostics for watching a run, not the reported numbers; the
paper's figures come from the evaluation stage in
[`../reproduction/_schema.md`](../reproduction/_schema.md).

`observability.tensorboard.enabled` writes TensorBoard events into the output
directory. Do not select checkpoints from those curves.

## Validation

The loader validates `objective.type`, `objective.loss`, and their
compatibility, and requires `silver_labels_path` for objectives that regress on
silver. It does not reject unknown keys, so a typo in an optional block is
silently ignored rather than reported.
