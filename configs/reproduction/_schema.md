# Reproduction pipeline schema (`configs/reproduction/`)

This directory holds the pipeline definitions that `scripts/study.py` executes,
plus five registries they share. A pipeline file is the declaration of one
task's full path from silver labels to downstream evaluation; it is not a job
config. Stages read the blocks they own and generate the per-job YAMLs under
`configs/experiments/` and `configs/sweeps/`.

Three files are task pipelines and share the structure below:

- `passage-reranking.yaml`
- `multi-document-qa.yaml`
- `response-ranking.yaml`

Run them with `uv run python scripts/study.py <task> <stage> <verb>`, where
stage is `silver`, `training`, `direct-eval`, or `downstream-eval`, and verb is
`plan`, `materialize`, `validate`, `execute`, `collect`, or `summarize`.

Stages are ordered, and each one reads what the previous stage wrote. In a
source-only checkout with no `build/`, `silver` and `training` plan and
materialize offline, while `direct-eval` and `downstream-eval` stop at
`Missing checkpoint catalog` until training has run. That is the intended
behaviour, not a misconfiguration.

## Top-level fields

| field             | type   | required | description                                                                                     |
| ----------------- | ------ | -------- | ----------------------------------------------------------------------------------------------- |
| `schema_version`  | int    | yes      | Format version. Currently `1` for every file in this directory.                                 |
| `task`            | object | yes      | `id` and `description`. The `id` must match the filename stem and the `study.py` task argument. |
| `shared`          | string | yes      | Path to the shared registry, always `configs/reproduction/shared.yaml`.                         |
| `silver`          | object | yes      | Teacher pass that produces silver labels: see below.                                            |
| `training`        | object | yes      | Student LoRA training and checkpoint selection: see below.                                      |
| `direct_eval`     | object | yes      | Ranking-quality and τ-PSI evaluation of trained and reference scorers: see below.               |
| `downstream_eval` | object | yes      | Decision-level evaluation built on the direct-eval score log: see below.                        |
| `controls`        | object | yes      | Per-stage legacy control lists. Current executable controls live beside the stage that owns them. |
| `analyses`        | object | yes      | Carries `implementation: existing-analysis-scripts`. A marker, not a dispatch table.            |

Every stage block ends with an `outputs:` block, and every stage except
`downstream_eval` in the passage and response pipelines carries an `execution:`
block.

## `silver:` block

Produces the teacher labels the student trains on.

| field            | type   | description                                                                                          |
| ---------------- | ------ | ------------------------------------------------------------------------------------------------------ |
| `id`             | string | Stage identifier, e.g. `passage-reranking-silver`.                                                   |
| `candidate_pool` | object | How the pool is built: `query_source` and `query_source_key` (an `ir_datasets` name), `index`, `retriever`, `sampling_seed`, `query_count`, `candidates_per_query`, `output_dir`, and the filenames written into it (`topics`, `qrels`, `run`, `fixture`, `selected_qids`, `sample_order`). |
| `teacher`        | object | Passage pipeline only. One teacher: `model` (a key in `shared.yaml`), `protocol`, `presentations`, `serving_width`, `cross_query_batch`, `prompt_template`, `grade_max`, `output_subdir`. |
| `teachers`       | list   | QA and response pipelines. Batched and pointwise teacher passes, keyed by `id`; training and heldout cohorts may each have both roles. |
| `scoring`        | object | QA and response pipelines. Prompt-side settings shared by the teachers: `prompt_template`, `grade_rubric`, `instruction`, `max_length`, `max_doc_chars`, `topics_id`. |
| `split`          | object | Passage pipeline only. `training_queries`, `heldout_queries`, and the `policy` that separates them.   |
| `products`       | list   | The label files this stage writes. `labels` distinguishes order-averaged, presentation-seed, and pointwise-single products. |
| `consumers`      | object | Maps each product id to the training variants allowed to read it. Keeps a heldout product from reaching a training arm. |
| `validation`     | object | Assertions checked after generation: query count, candidate coverage, raw vector length, score range, and the duplicate/exhaustive/disjoint split checks. |

## `training:` block

| field                  | type   | description                                                                                     |
| ---------------------- | ------ | ------------------------------------------------------------------------------------------------- |
| `silver_manifest`      | string | Path to the manifest the silver stage wrote. This is the stage's input.                         |
| `seeds`                | object | Maps each training seed to its `view_seeds`, the two shuffled views OC-SFT compares.             |
| `variants`             | list   | The arms to train. Each has `id`, `silver` (which product to read), and either a `template` or a `template_pattern` with a `lambda_grid`. `view_seeded: true` marks arms whose views vary with the training seed. |
| `checkpoint_selection` | object | `metric`, `rule`, and `exact_tie_break` define the rule; `selector_cli` names the script that executes it and `decisions` the record it writes. `selection_protocol` names the reference implementation, which this pipeline does not import. |
| `optional_ablations`   | object | `enabled_by_default` plus a `variants` list: see the note on `runnable` below.                     |

## `direct_eval:` block

| field                | type   | description                                                                                        |
| -------------------- | ------ | ---------------------------------------------------------------------------------------------------- |
| `population`         | string | Named dataset population, resolved against the task's population registry.                         |
| `checkpoint_catalog` | string | Path to the catalog the training stage wrote. Required; the stage refuses to plan without it.      |
| `variants`           | object | `trained` lists training-variant ids to evaluate. `references` declares untrained comparators, each with `id`, `kind`, an inline `model` block, and optionally `order_invariant: true`. |
| `protocol`           | object | `serving_width`, `pool_depth`, `perturbations`, `presentation_seeds`, and `derive_self_consistency`. |
| `evaluation`         | object | `measures`, plus `headline_quality` and `headline_robustness` naming which of them the summary reports. |
| `optional_controls`  | object | `enabled_by_default` plus a `variants` list: see the note on `runnable` below.                       |

Reference `model` blocks follow the same schema as `reranker:` in
[`../experiments/_schema.md`](../experiments/_schema.md) and dispatch through
the same registry.

Use `--include-controls` consistently for direct-eval materialization,
validation, execution, and collection. Omitting it at collection deliberately
produces primary-only aggregates.

## `downstream_eval:` block

Turns scores into decisions and measures how those decisions move. It reads the
stored score log from `source_stage` rather than re-scoring documents.

| field                        | type   | description                                                                              |
| ---------------------------- | ------ | -------------------------------------------------------------------------------------------- |
| `source_stage`               | string | Which stage's score log to read, `direct-eval` in all three pipelines.                   |
| `population`, `variants`     |        | As in `direct_eval`, except `references` is a plain id list.                              |
| `consumer`                   | object | Passage pipeline. The decision rule: `id`, `rule`, `threshold_selection`, `relevance_cutoff`. |
| `reader`, `verdict_reader`   | object | QA pipeline. The frozen reader models, with `engine`, `model_name`, `revision`, and generation settings. `verdict_checkpoint_catalog` points at the scorer catalog the verdict bridge reads. |
| `protocol`                   | object | Split and presentation counts: `development_fraction`, `split_seed`, `threshold_grid_size`, `canonical_presentation`, `scorer_presentations`, `order_invariant_presentations`. |
| `evaluation`                 | object | `measures`, with `headline_quality` and `headline_stability`.                             |

## `runnable:` in ablation and control lists

`optional_ablations.variants` and `optional_controls.variants` are registries of
arms that were considered. Both consumers filter with `.get("runnable")`, so an
entry runs only if it says `runnable: true`. A missing key means the same thing
as `false`; write it out anyway, so a reader does not have to infer it.

An entry that is not runnable generates nothing, so each one carries a `note:`
saying why. The note has to point somewhere a reader can go: the arm is realized
by another stage, by a study under `configs/studies/`, or by a named script, or
it is incompatible with how this stage serves models. "No template is declared"
is not a reason, it restates `runnable: false`; an arm that cannot be explained
in these terms was dropped from these files rather than documented. `purpose`
and `note` are documentation, and nothing reads them.

Runnable ablations need a `template` and may carry `seeds`, a silver-product
override, and a `patch` that is
merged into the template. Runnable controls may carry a `source_variant` and
`training_seed` to select a trained checkpoint, and their own
`checkpoint_catalog` when they read one the primary stage did not write.

## `outputs:` blocks

Each stage declares where under `build/reproduction/<task>/<stage>/` it writes.
`root` is the directory; the other keys are filenames within it. Stages that
generate job configs use `configs_dir` and `sweep`; stages that record what ran
use `run_manifest`; the training stage writes `checkpoint_catalog`, which
`direct_eval` then reads. Evaluation stages add their reduction outputs
(`per_dataset`, `per_seed`, `aggregate`, `per_consumer`, `reduction_plan`).

These paths are the contract between stages, which is why a stage fails with
`Missing checkpoint catalog` rather than proceeding when an earlier stage has
not run.

## Registries

Five files in this directory are shared registries rather than pipelines.

| file                           | contents                                                                                             |
| ------------------------------ | -------------------------------------------------------------------------------------------------- |
| `shared.yaml`                  | `models` (named reranker blocks the pipelines reference by key), `seeds` (the `training` and `presentations` lists), and `variants` (canonical variant order for reporting). Referenced by every pipeline through its `shared:` field. |
| `appendix.yaml`                | The appendix entries currently represented in the release. Each names a `program`; canonical entries may name an `analyzer` and `preferred_input`, while static entries name committed sources. It is not a one-to-one registry of every numbered paper table/figure. |
| `appendix-programs.yaml`       | From-scratch execution programs shared by appendix entries, with their studies/tasks, inputs, and reducers. |
| `representative-analyses.yaml` | 7 analysis families, each with an offline `fixture`, an `operation`, its `checks`, and the `paper` anchor it reproduces. Drives the offline validation receipt. |
| `source-lock.yaml`             | Expected `sha256` and `size_bytes` for materialized non-internal population files, immutable dataset revisions needed to reconstruct them, plus model revisions found in the four high-level pipelines. The local data files are ignored, not tracked. Verified by `scripts/source_lock.py`; structural audit checks declarations only. |

Paths under `build/` in these files are outputs the pipeline writes when you run
it, not artifacts shipped with the repository.

## Subdirectories

| directory         | contents                                                                                                  |
| ----------------- | --------------------------------------------------------------------------------------------------------- |
| `populations/`    | The four named dataset populations a pipeline's `population:` field resolves against, loaded by `eval/dataset_catalog.py` as `populations/<id>.yaml`. Each declares a `pool_depth` and a `datasets` list giving every collection its loader, topics, index, first-stage run path, qrels, and prompt settings. |
| `evidence/`       | Decision records that stages write and later stages read, one directory per decision. `lambda-selection/` holds the `decisions.json` that `training.checkpoint_selection.decisions` points at, alongside the heldout trajectories behind it; `retained-set-threshold/` holds the cross-base retained-set summaries, one of which `representative/jaccard.json` anchors to. |
| `fixtures/`       | `taupsi-qids/`, the 16 capped query-id lists that pin which queries a τ-PSI run scores.                    |
| `representative/` | The seven offline fixtures `representative-analyses.yaml` names, one per analysis family, used to validate the analysis code without a GPU. |

Unlike the `build/` paths above, everything here is tracked and ships with the
repository.
