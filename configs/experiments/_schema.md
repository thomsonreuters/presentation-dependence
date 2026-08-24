# Experiment config schema (`configs/experiments/<ID>-<slug>.yaml`)

One YAML per experiment.

## Filename conventions

Config filenames are part of the experiment interface. The filename stem should
match `id`, and the directory stays flat: run directories, the results index,
and evidence mappings all key off that ID.

Common tokens:

- `R*`, `L*`, `A*`: reference, listwise/baseline, and adaptation series.
- `dl19`, `dl20`, `beir-<task>`, `legal-a`: dataset.
- `offshelf`, `baseonly`, `psi`, `psi-topup`, `smoke`: run mode.
- `w20-s20`, `block20-rrf`, `b20`, `b50`, `grade`: protocol/model knobs.

## Top-level fields


| field              | type   | required | description                                                                                                             |
| ------------------ | ------ | -------- | ----------------------------------------------------------------------------------------------------------------------- |
| `id`               | string | yes      | Stable ID (e.g. `example-passage-b20-psi`); run directories and result-index entries key off it.                   |
| `reranker`         | object | yes      | Reranker instantiation: see below.                                                                                     |
| `data`             | object | yes      | Dataset + first-stage retrieval config: see below.                                                                     |
| `eval`             | object | yes      | Evaluation config: see below.                                                                                          |
| `robustness`       | object | no       | K-permutation robustness config: see below.                                                                            |
| `pool_perturbation`| object | no       | Fixed-order top-N pool replace/drop control; runs instead of ordinary eval/PSI in the unified entrypoint.              |
| `context_decomposition` | object | no  | Category-H A1-A4 fixed-weight width decomposition; runs instead of ordinary eval/PSI in the unified entrypoint.         |
| `matched_variance_control` | object | no | Ten-read B=1 estimator control with varied cross-query batch positions and compositions.                           |
| `self_consistency` | object | no       | Optional repeated shuffled scoring for Qwen3-style setwise grading.                                                     |
| `execution`        | object | no       | Execution settings: channel name, distribution, environment: see below.                                                |
| `bundle`           | object | no       | Multi-collection bundle: run N collections for one model in one job: see below. Mutually exclusive with `reranker`/`data`/`eval`. |
| `debug`            | object | no       | Optional runtime checks for development; keep disabled for recorded runs unless diagnosing a model wrapper.             |
| `logging`          | object | no       | Standard logging config (`level`, `log_file`).                                                                          |
| `qids_to_run`      | list   | no       | Optional qid allowlist for partial reruns.                                                                              |
| `qids_to_run_path` | string | no       | Optional newline-delimited qid allowlist for larger diagnostic subsets. Shared by `run_experiment.py` and `run_psi.py`. |


## `reranker:` block

Dispatched through `src/presentation_dependence/rerankers/registry.py`. Required keys
depend on `class`.

```yaml
reranker:
  class: MxbaiPointwise         # must match a key in rerankers.registry.RERANKER_CLASSES
  model_name: mixedbread-ai/mxbai-rerank-large-v2
  device: auto                  # "auto" | "cuda" | "cuda:<n>" | "mps" | "cpu"
  batch_size: 32
  max_length: 8192
```

For Hugging Face rerankers, `auto` resolves CUDA, then MPS, then CPU. MPS and
CPU are development/inference alternatives only where the selected wrapper and
model support them. `dtype: auto` selects float32 away from CUDA and is the
portable choice for those checks; paper configs use bfloat16. vLLM configs
require the CUDA profiles documented in
[`docs/HARDWARE.md`](../../docs/HARDWARE.md#supported-hardware).

Optional keys (only where the reranker implementation reads them):


| field                     | applies to                                                                                                                                                                                                                            | purpose                                                                                                                                                                                                                                                                                                                                                                                         |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `model_size`              | Model families with multiple released sizes (currently `Qwen3Reranker`)                                                                                                                                                               | Human-/tool-readable size label (`0.6b`, `4b`, `8b`). Keep in sync with `model_name`; used for result tables and size sweeps.                                                                                                                                                                                                                                                    |
| `docs_per_score_forward`  | **Batched pointwise** passage rerankers using native Qwen-style APIs or prompted multi-document scoring. Must be **> 1** to distinguish geometry from classical pointwise CE (`docs_per_score_forward = 1`, implicit for `MxbaiPointwise`). | Drives `infer_tau_psi_geometry()`: reported **τ-PSI@B** equals this width.                                                                                                                                                                                                                                                                                                   |
| `scoring_mode`            | `Qwen3Reranker`                                                                                                                                                                                                                       | `official_pairwise` (default) mirrors the Qwen3 model-card yes/no scorer, one pair per score. `setwise_prompt` is the shared-context batched-pointwise experimental path: one query + B numbered documents in a single decoder context, parsed to B scalar scores. `setwise_grade_prompt` is the same shared-context path but asks for fixed integer relevance grades `0..3`, then maps them to scalar scores. |
| `scoring_method`          | `Granite41GradeReranker`                                                                                                                                                                                                              | `continuous_grade` (default): expected-grade readout. `generate_grades`: free-form decode + line parser (`[<id>] Grade: <0..3>`).                                                                                                                                                                                                                                                               |
| `grade_with_space_prefix` | `Granite41GradeReranker`                                                                                                                                                                                                              | When `true`, pass `with_space_prefix=True` into `resolve_grade_token_ids` so BPE digit tokens match prompt whitespace context.                                                                                                                                                                                                                                                                  |
| `rankzephyr_protocol`     | **RankZephyrReranker only**                                                                                                                                                                                                           | `sliding` (default): Pradeep+2023-style `_window_schedule` loop. `block_local_rrf`: tier 3; disjoint blocks, one listwise forward each, **RRF** merge.                                                                                                                                                                                              |
| `block_size`              | RankZephyr with `rankzephyr_protocol: block_local_rrf`                                                                                                                                                                                | Docs per block **B**; defaults to `window_size` if omitted. `**stride` is ignored** in this mode.                                                                                                                                                                                                                                                                                               |
| `rrf_k`                   | RankZephyr `block_local_rrf`                                                                                                                                                                                                          | Positive RRF constant (default **60**).                                                                                                                                                                                                                                                                                                                                                         |
| `setwise_missing_grade`   | `Qwen3Reranker` in `setwise_grade_prompt`                                                                                                                                                                                             | Optional fallback grade used only when the model omits one or more document ids. Use `0` for smoke/benchmark robustness; omit while debugging malformed outputs.                                                                                                                                                                                                                                |
| `grade_rubric_id`         | `Qwen3InstructGradeReranker`                                                                                                                                                                                                          | Prompt/rubric selector. `relevance_v1` is the default web-search relevance rubric; `answer_support_v1` scores passages for answer support in the multi-doc QA arm.                                                                                                                                                                                                                              |
| `base_reranker`           | `CapCalReranker`                                                                                                                                                                                                                      | Nested reranker config to calibrate. Use a scalar-emitting batched-PW base, preferably an expected-grade scorer that also emits `grade_probabilities_init_order`, so CapCal can subtract excess content-free prior mass in probability space.                                                                                                                                                       |
| `capcal`                  | `CapCalReranker`                                                                                                                                                                                                                      | Content-free calibration settings: `strength` (default `1.0`; beta in Lv+2026), `entropy_adaptive` (default `true`), `probe_query_texts` (default `null`, meaning use each real query with empty candidate contents), `probe_passage_text` (default `""`), and `require_grade_probabilities` (default `true`). Set `require_grade_probabilities=false` only for the weaker centered-score fallback. |
| `attn_implementation`     | `PineReranker`                                                                                                                                                                                                                        | Must be `eager`. PINE changes attention internals and cannot use FlashAttention, SDPA, or vLLM in this repo.                                                                                                                                                                                                                                                                                    |
| `pine_vllm_mode`          | `PineReranker`                                                                                                                                                                                                                        | Only read when `inference_engine: vllm`. Default `reject` fails loud because stock vLLM cannot run PINE's custom attention. Set `hf_fallback` only to make vLLM-oriented config generation explicitly run the faithful HF/eager PINE path instead.                                                                                                                                                 |


`Reranker.rank` must populate `scores_init_order` on every call whenever the model emits per-doc scalars; required for `**mean_score_variance`** in `psi_metrics.json` for batched-PW / scoring-head / pointwise stacks; omit only for pure generative-listwise text rankings (`scores_init_order: null`).

During wrapper development, enable the optional contract validator:

```yaml
debug:
  validate_rank_result: true
```

With this flag enabled, `ExperimentManager` and `PsiExperimentRunner` call
`validate_rank_result(result, passages)` after each reranker call. The check
catches unknown output pids, duplicate output pids, score-vector length
mismatches, and non-empty outputs for empty inputs. Leave it off for normal
benchmark runs.

Available `class` values (current):

- `IdentityReranker`: passes the first-stage order through unchanged.
Exists as a smoke-test path that exercises the full eval pipeline without
requiring a GPU.
- `MxbaiPointwise`: `mixedbread-ai/mxbai-rerank-large-v2`. Classical pointwise
cross-encoder; returns per-doc scores. Classical pointwise reference
in the experiment matrix.
- `Qwen3Reranker`: `Qwen/Qwen3-Reranker-4B`. Use
`scoring_mode: official_pairwise` + `docs_per_score_forward: 1` for the Qwen3
official pointwise reference. Use `scoring_mode: setwise_prompt` or
`setwise_grade_prompt` + `docs_per_score_forward: B` for shared-context
batched-PW pre-RL paths.
- `Qwen3InstructGradeReranker`: expected-grade Qwen3 scorer used by the primary
  off-the-shelf and OC-SFT paths; supports HF and vLLM backends and LoRA adapters.
- `Gemma4GradeReranker`: Gemma-4 expected-grade scorer; backend/image support
  differs from Qwen and some LoRA combinations require merged weights.
- `RankZephyrReranker`: `castorini/rank_zephyr_7b_v1_full`. Generative
listwise: default `sliding` `_window_schedule`, or `block_local_rrf` (tier 3)
with `block_size` + `rrf_k`; see the `rerankers/rankzephyr.py` module docstring.
- `Granite41GradeReranker`: `ibm-granite/granite-4.1-8b` expected-grade
scorer (cumulative skeleton by default; `grade_prefix_mode: independent`
optional). No Mamba kwargs. Defaults: `trust_remote_code=false`; set
`reranker.grade_with_space_prefix: true` if digit tokens need a leading-space
context for `resolve_grade_token_ids`.
- `CapCalReranker`: training-free content-free calibration wrapper (Lv+2026)
  around an existing scalar scorer. Primary use is a nested expected-grade
  batched-PW base
with grade probability vectors; the wrapper estimates per-slot no-content grade
priors and recalibrates real grade distributions before sorting.
- `PineReranker`: Wang+2025 PINE for Qwen-family expected-grade scoring.
HF/eager is the only faithful in-repo execution path (`attn_implementation:
eager`). Stock vLLM, FlashAttention, and SDPA are rejected because they cannot
run PINE's custom document-level attention. For vLLM-oriented config generators,
`inference_engine: vllm` plus `pine_vllm_mode: hf_fallback` explicitly runs the
HF/eager path rather than pretending stock vLLM is PINE. It emits per-doc scalar
scores and `grade_probabilities_init_order` like other expected-grade batched-PW
scorers.
- `JinaListwiseReranker`: see registry and per-class docs.
- `ClosedModelGenBscReranker`: generated-grade BSC over an
  OpenAI-compatible API endpoint; it is not the decode-free expected-grade
  readout.
- `SkyworkBTReranker`: `Skywork/Skywork-Reward-V2-Qwen3-*` in-family
Bradley-Terry scalar RM, the response-ranking external **pointwise** anchor. Loads
`AutoModelForSequenceClassification(num_labels=1)`, applies the model chat
template to `[user: prompt, assistant: response]` (no system prompt), strips a
duplicate BOS, reads the scalar reward logit. Order-invariant → quality axis
only (no `robustness:` block). Keys: `model_name`, `device`, `dtype`,
`batch_size`, `max_length`, optional `max_doc_chars` / `attn_implementation` /
`revision`. Runs in the shared tf5 container. Implementation:
`rerankers/skywork_bt.py`.
- `PairRMReranker`: `llm-blender/PairRM-hf` (~0.4B DeBERTa-v3), the
response-ranking external **pairwise** anchor. **HF-native**: loads the checkpoint with the vendored
`DebertaV2PairRM` head (`rerankers/_pairrm_model.py`) + the model-card
`tokenize_pair` format: **no `llm_blender` package**, so it runs on the shared
tf5 container like Skywork (only needs `sentencepiece`). Ranks N candidates from
all-ordered-pairs comparisons aggregated by Copeland, with **mandatory
order-averaging** (both A/B orders per pair) to cancel PairRM's pairwise position
bias → order-invariant → quality axis only. Keys: `model_name`, `device`,
`dtype`, `batch_size`, optional `max_doc_chars` / `source_max_length` /
`candidate_max_length` / `revision`. Implementation: `rerankers/pairrm.py`.

When a new base model is added, it lands as a file under `rerankers/` and a
single-line registry entry. Each reranker documents its own required config
keys in its docstring.

## `data:` block

First-stage retrieval is an external run file (BM25 / SPLADE / etc.), not
re-computed at run time.

```yaml
data:
  dataloader_class: PyseriniLoader
  topics: dl19-passage                  # pyserini topics key
  index: msmarco-v1-passage             # pyserini prebuilt index for passage text lookup
  run_path: data/dl19-passage/run.msmarco-v1-passage.bm25-default.dl19_sorted.txt
  k_input: 100                          # truncate first-stage list to top-k before reranking
```

## `eval:` block

Post-hoc evaluation config; read by `scripts/run_eval.py`. Evaluation is
deliberately separated from inference so a single inference run can be
re-evaluated under different qrels / measure sets.

```yaml
eval:
  qrels_path: data/trec-dl19-passage/qrels.txt
  measures:
    - ndcg_cut_10
    - ndcg_cut_5
    - map
    - recip_rank
  query_batch_size: 16        # engine-call batching width (default 16 for both passes; see below)
```

`query_batch_size` is read by **both** drivers and defaults to **16** in each,
but they batch different units:

- **PSI (`run_psi.py`): cross-permutation.** Packs a query's batchable
  presentations (`random_shuffle` plus the three `middle_injection` placements,
  about 13 at K=10) into one engine call instead of one `rank()` per
  presentation. Cuts rerank compute by roughly 1.1–1.2× with no quality change,
  verified metric-equivalent on Qwen3-0.6B and Granite-4.1-8B over DL19: nDCG
  within 1e-3, τ-PSI and Kendall τ identical. Render variants (`id_relabel`,
  `separator`, `rubric_paraphrase`, `scale`) always stay per-presentation.
- **Base pass (`run_experiment.py`): cross-query.** Packs N *different*
  queries' single B=20 passes into one engine call. On vLLM this rides the
  normal scheduled `generate(max_tokens=1, logprobs)` path: vLLM admits
  sequences up to `max_num_seqs` and the `gpu_memory_utilization` KV budget and
  queues the rest, so a longer prompt list lengthens the queue, **not the GPU
  peak**. Any model that runs a B=20 vLLM eval, 8B and 32B included, will not
  OOM from this. Measured against the per-query path on 97 queries at B=20 it
  is score-exact (nDCG Δ 0.0010, within noise) and 1.21× faster on base-pass
  compute; the realised speedup grows with query count as the fixed model-load
  cost amortizes, from about 1.04× at 97 queries toward the 1.21× asymptote.
  Caveat: the **HF** backend pads to longest, so cross-query batching there can
  cost more or inflate the batch tensor. `_run_query_batched` falls back to
  per-query `rank()` if the batched call raises.

Both paths only engage for batching-capable rerankers
(`supports_query_batching=True`, currently the Qwen3/Granite logit-skeleton
path); others silently fall back to the per-unit loop. Set
`query_batch_size: 1` to force the bit-identical per-unit path.

## `robustness:` block (optional : enables `scripts/run_psi.py`)

Activates K-permutation evaluation. When
absent, `run_psi.py` falls back to defaults (K=10, random_shuffle, seeds
0..K−1). When present, the block is self-contained; the existing
`reranker` / `data` / `eval` blocks are reused as-is, so a single config
can be run with BOTH `run_experiment.py` (quality) and `run_psi.py`
(robustness).

```yaml
robustness:
  K: 10                                      # number of permutations per query
  seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]      # len(seeds) == K (enforced)
  perturbations:                             # order and/or render strategies:
    - random_shuffle                         #   seeded Fisher-Yates; natural bucket mix
    - middle_injection                       #   forces relevant doc to top/mid/bot
    - id_relabel                             #   render variant: document-ID scheme (order fixed)
    - separator                              #   render variant: doc-block separator (order fixed)
    - rubric_paraphrase                      #   render variant: faithful rubric paraphrase (order fixed)
    - scale                                  #   scoring-scale variant: prompt + readout (order fixed)
  k_cutoff_for_ndcg: 10                      # nDCG cutoff used for Δ-nDCG
  derive_self_consistency: false             # optional; disable SC for render-only runs
  beta_gamma:                                # optional; row-level random-shuffle score log
    enabled: true
```

Order perturbations (`random_shuffle`, `middle_injection`) reorder the passage
list. Render perturbations (`id_relabel`, `separator`, `rubric_paraphrase`)
hold order fixed and vary expected-grade prompt rendering via
`presentation_dependence.rerankers.render_variants` (requires a reranker with
`supports_render_variants=True`, e.g. `Qwen3InstructGradeReranker`). The
`scale` perturbation varies scoring-scale wording and logit readout via
`presentation_dependence.rerankers.scale_variants` (`supports_scale_variants=True`).
Each strategy emits `K` variants (index 0 = canonical/seen). Stage-0 gate
configs live under the `IS0-*` experiment IDs.

Perturbations `partial_shuffle`, `block_swap`, and `retriever_noise` are not
implemented; add
them in `PsiExperimentRunner._generate_permutations()` when needed.
Aggregate artifact: `runs/<ID>/<ts>/psi/psi_metrics.json`.

`beta_gamma.enabled` additionally writes `psi/beta_gamma_scores.parquet`: one
row per (query, random-shuffle presentation, document) with the document's
input position and score. It requires `random_shuffle` in `perturbations` and a
reranker that populates `scores_init_order`, and it rejects the run unless
every query, presentation, document, and position is covered. Nothing in the
PSI metrics depends on it; it exists so `scripts/run_reader.py` can replay the
scorer's presentations against a frozen reader, so enable it on any run a
reader will later consume. `recipe` and `checkpoint` may also be set to label
the rows, and default to the experiment ID and `unknown`.

`qids_to_run` / `qids_to_run_path` may also be set **inside** the `robustness:`
block to cap PSI to a fixed qid subset while nDCG (`ExperimentManager`) still runs
the full collection (capped-τ-PSI). `psi_manager` prefers the
robustness-scoped allowlist, falling back to the top-level one.
The capped qid lists under `configs/reproduction/fixtures/taupsi-qids/` fix this subset in the runtime
estimate (`--m-queries M` for a capped nDCG pass).

## `pool_perturbation:` block (optional)

Activates the fixed-order companion-coupling control. The input fixture must contain
rank `pool_size + 1`; set `data.k_input: 101` for that control on a top-100 pool.
The runner scores three pools per selected query: canonical top-100, replace one
sampled rank and append rank 101, and drop the same sampled rank. It never
shuffles the surviving candidates.

```yaml
data:
  k_input: 101

pool_perturbation:
  pool_size: 100
  source_depth: 101
  perturbations: [replace, drop]
  seed: 0
  query_sample_seed: 0
  max_queries: 100
  k_cutoff_for_ndcg: 10
```

The query sample is deterministic over qids that have at least `source_depth`
candidates. The drop rank is uniform in 1..`pool_size` and independently seeded
per qid. Optional `qids_to_run` / `qids_to_run_path` fields restrict the eligible
pool before sampling.

Run locally with:

```bash
uv run python scripts/run_pool_perturbation.py -e <ID>
```

The unified entrypoint detects this block and runs only the pool
control, writing
`runs/<ID>/<timestamp>/pool_perturbation/{pool_metrics.json,pool_per_query.json}`.
Every metric is restricted to the `pool_size - 1` retained documents:
pool-PSI `(1 - Kendall tau) / 2`, top-k set-flip rate, and signed/absolute
nDCG@k change. A B=1 reference of exactly zero is recorded without inference.

## `context_decomposition:` block (optional)

Activates the Category-H A1-A4 control. One fixed checkpoint scores
each target alone, with its real companions, with length-matched unrelated
companions, and alone under a padded cumulative skeleton. A2-A4 keep the target
in the same slot.

```yaml
context_decomposition:
  pool_size: 100
  variable_pool_size: false
  width: 20
  dataset_key: dl19
  donor_seed_namespace: context-decomposition-dl19-ocsft-seed42
  qid_shard_count: 1
  qid_shard_index: 0
  screen_only: false
  seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  k_cutoff_for_ndcg: 10
  request_batch_size: 128
  bootstrap_samples: 10000
  bootstrap_seed: 0
```

The unified entrypoint detects this block and writes
`runs/<ID>/<timestamp>/context_decomposition/`. The canonical result uses
first-stage order. The K=10 result averages each target's score over the seeded
Fisher-Yates presentations before ranking. `A2-A1`, `A2-A3`, `A3-A4`, and
`A4-A1` are emitted at both ranking and score level.

Set `screen_only: true` for the context-decomposition screening stage. That mode scores
only A1 and A4, emits paired A4-A1 intervals, skips all real- and
unrelated-companion inference, and does not require a donor pool.

`pool_size` need not be divisible by `width`. When the published serving
protocol ends in a smaller final chunk, A3 uses that chunk's actual companion
count and A4 uses the same smaller skeleton width. This preserves A2 geometry
instead of padding a neighboring estimand.

Set `variable_pool_size: true` when the published fixture contains a variable
number of candidates per query. The runner then uses all available candidates
up to `pool_size` instead of dropping queries shorter than that maximum.

`qid_shard_count` and `qid_shard_index` split whole queries by deterministic
round-robin assignment. In full mode every shard still builds A3 donors from
the complete dataset pool. Keep `dataset_key` and `donor_seed_namespace`
identical across shards, then merge compact per-query outputs with
the package-owned context-decomposition merger. It rejects overlaps, gaps, and
protocol mismatches before recomputing the dataset aggregate.

## `matched_variance_control:` block (optional)

Activates the matched request-batching control. The reranker must support
cross-query batching, and `docs_per_score_forward` must equal `serving_width`
(which defaults to 1 for the matched-variance control). Every candidate is
scored ten times. Before
each pass, all selected fixed-order candidate chunks are globally shuffled and
then split into `request_batch_size` engine calls. This changes request position
and the other chunks sharing its vLLM call while keeping candidate order and
prompt bytes fixed.

```yaml
matched_variance_control:
  seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  serving_width: 1               # defaults to 1; set 20 for the matched wide control
  request_batch_size: 128        # candidate chunks per rank_query_batch call
  pool_size: 100                # optional; defaults to data.k_input
  k_cutoff_for_ndcg: 10         # use 1 for response ranking
  qid_shard_count: 1
  qid_shard_index: 0
  required_unique_positions: 2
  required_unique_compositions: 2
```

The unified entrypoint runs this block instead of ordinary eval/PSI and writes
`runs/<ID>/<timestamp>/matched_variance/`. Each seed has an auditable score and
batch-layout file. The aggregate is accepted only if every candidate occupies
at least the configured number of distinct within-call positions and batch
composition digests. The reported quality metric is nDCG after averaging the
ten fixed-width scores per candidate before ranking. The per-seed score files
can also be reduced through the threshold protocol to measure retained-set
Jaccard under request batching alone.

## `self_consistency:` block (optional)

Currently read by `Qwen3Reranker` when `self_consistency.enabled: true`.
It repeatedly shuffles the input list, scores each shuffled list, maps scores
back to the original input positions, and averages them before sorting.

```yaml
self_consistency:
  enabled: true
  K: 5
  seeds: [0, 1, 2, 3, 4]    # len(seeds) == K
  shuffle_docs: true        # currently required
  aggregate: mean_grade     # mean_grade | mean_score
```

Use this only for experiments that intentionally evaluate stochastic
setwise-grade aggregation. Leave absent for ordinary off-the-shelf baselines.

## `execution:` block (optional)

Execution settings: where a job reads its data, how it is parallelised, and what
environment it runs under. These five fields are what running code reads.

```yaml
execution:
  fixture_channel: dl19-passage     # optional; defaults from data.run_path parent
  environment:
    PYTORCH_CUDA_ALLOC_CONF: expandable_segments:True
```

| field | where it appears | purpose |
| --- | --- | --- |
| `fixture_channel` | here | Overrides the resolved channel name. Defaults to the parent directory of `data.run_path`, then `data.topics`. |
| `environment` | here | Trusted, non-secret environment defaults applied by runtime entrypoints after config/CLI overrides and before accelerator setup or child-process launch. Existing process variables (shell, `.env`, container, or scheduler) win. Dry runs and materializers do not mutate the environment. The images already set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. |
| `distributed` | `configs/self-distill/` | `ddp` makes the runner re-exec under single-node torchrun, one worker per GPU, sizing from `SLM_NUM_GPUS` or the visible device count. Honoured by both `scripts/run_self_distill_sft.py` and `scripts/train/entrypoint.py`; on one GPU it warns and runs a single process. |
| `extra_input_channels` | generated configs | Map `{<channel-name>: <path>}` for LoRA adapters (`lora-adapter`) and a bundle's shared artifacts. Written by the study materializer, not by hand. A channel name must not collide with a data channel. |
| `max_parallel` | `configs/sweeps/`, `configs/studies/` | Cells `run_sweep.py` keeps in flight at once, as concurrent processes on one machine rather than a fan-out across nodes. Tracked configs declare `1`; see [`../sweeps/_schema.md`](../sweeps/_schema.md) for when raising it is safe. |

Environment values must be scalars and keys must be POSIX-style names such as
`VLLM_WORKER_MULTIPROC_METHOD`. Do not put credentials in tracked YAML; pass
tokens and API keys through the shell, `.env`, container, or scheduler instead.
Entrypoints never log environment values.

Scheduler keys that older configs carried, `instance_type`, `instance_count`,
`max_run_s`, `framework_version`, `py_version`, `image_uri`, `tensorboard` and
`checkpoint`, are removed, and the block itself was renamed from `aws:`, which
named a service this distribution does not talk to. Nothing reads them; GPU
sizing is in
[`../../docs/HARDWARE.md`](../../docs/HARDWARE.md).

The `config_hygiene` layer of `study.py structural audit` walks every mapping
entry at any depth in `configs/**/*.yaml` and fails on a retired key, or on a
value that looks like an account, a bucket URI, a container registry or a
private index. Copying a stanza from an older tree therefore fails that audit
rather than silently reintroducing a key that does nothing.


### Which PSI field to report

`psi_metrics.json` emits **two** distinct PSI scalars; they measure
different things; don't mix them up:

- `zeng_psi_corpus` (in `aggregate`): **leaderboard-comparable** with
Zeng+2025 Table 1. Formula: `PSI = 1 − min(s) / max(s)` where `s` is
mean nDCG@10 per position bucket, pooled across queries. Requires
`middle_injection` or mixed-bucket `random_shuffle` data. Reports `None`
when no stratified buckets were produced.
- `mean_tau_based_psi` (in `aggregate`): `(1 − mean_τ) / 2`, a τ-remap
from the self-consistency literature (Tang+2024 family). **NOT
Zeng+2025 PSI**. Available for any run; including pure
`random_shuffle` without buckets.

When recording robustness numbers, fill the
`actual_zeng_psi_corpus` and `actual_tau_based_psi` slots separately;
always annotate with the perturbation protocol that produced the number
(Zeng+2025's own PSI numbers are on intra-passage buckets, not listwise
positions; the same formula can be applied to listwise positions, but that is
not Zeng+2025's measurement setting).

### `score_variance` vs `rank_variance` in `psi_metrics.json`

- `**mean_score_variance`** is computed whenever the PSI driver collected
`scores_init_order` across permutations: **batched pointwise**, architectural
scoring-head (e.g. Jina), and the mxbai-rerank-large-v2 classical pointwise
reference. For pointwise CE it
should track numerical noise (~determinism); for batched-PW τ-PSI experiments
it is a **co-primary** scalar robustness
signal with `**mean_tau_based_psi`** (interpret both; scales are within-model
only).
- `**mean_rank_variance**` is always populated when rankings overlap across `K`;
it is the primary ranking-level supplement when `**mean_score_variance**` is
absent (`None`), e.g. generative listwise without pseudo-scores.

Reported τ-PSI **B** is recorded separately via `tau_psi_context_batch_size` /
`tau_psi_inference_note` (`tau_psi_geometry.py` schema `v2`).

## `bundle:` block (optional : multi-collection eval in one job)

Runs **N collections for one model config in one process, loading the model once**
and reusing it across collections, which amortizes the model-load cost. A bundle
config has no `reranker`, `data`, `eval` or `robustness` of its own; it
**references existing per-collection configs** plus an `execution:` block. Consumed by
`scripts/train/entrypoint.py` → `presentation_dependence.eval.bundle.run_bundle`; planned +
emitted by the study materializer. Use that output, the tracked
runtime cap, and the sizing guidance in `docs/HARDWARE.md`; dated timing snapshots are
not executable inputs.

```yaml
id: <bundle-id>
bundle:
  member_configs:                 # existing configs/experiments/<id>.yaml, one per collection
    - example-passage-b20-psi
    - example-passage-granite-b20-psi
  shared_base: false              # optional (default false). true = Case B: one base load
                                  #   serves the base + N adapters (members may vary lora_path)
  reranker_overrides:             # optional; deep-merged into EVERY member's reranker
    vllm_settings:                #   e.g. retarget tensor_parallel_size to the bundle instance
      tensor_parallel_size: 4
execution:                        # shared settings for every collection in the bundle
  environment: {VLLM_WORKER_MULTIPROC_METHOD: spawn}
  extra_input_channels:           # shared model artifacts (identical across members)
    lora-adapter: checkpoints/<exp_id>/<trial>/student/checkpoint-step-00XXXX/
```

Fields:

- `bundle.member_configs` (list[str], required): per-collection experiment config
  ids resolved next to the bundle config. Each member carries its own
  `reranker` / `data` / `eval` / `robustness`.
- `bundle.reranker_overrides` (object, optional): deep-merged into every
  member's `reranker` before running, e.g. retarget
  `vllm_settings.tensor_parallel_size` to the bundle's instance. Top-level
  `reranker:` is rejected on bundle configs so these overrides cannot be
  silently ignored.

Constraints + behavior:

- **Members must share one model.** Only the per-collection prompt knobs
  (`reranker.instruction`, `reranker.max_doc_chars`) may differ; those are
  re-applied to the reused reranker per collection; every other `reranker` key must
  match across members (enforced by `presentation_dependence.eval.bundle.assert_shared_model`).
  Each prompt knob must be **all-or-none**: set on every member or on none. (If a
  knob were set on some and omitted on others, the omitting collection would inherit
  the previous collection's value on the reused reranker instead of the model default; also enforced by `assert_shared_model`.)
- **`bundle.shared_base: true` (Case B, vLLM only):** relaxes the above so members
  may also differ on `reranker.lora_path`: a single base is loaded **once** and
  serves the off-the-shelf base (members with no `lora_path`) plus N LoRA adapters,
  switching the active adapter per collection (`set_active_adapter`). The base loads
  with the **union** of member adapters (`enable_lora`, `max_loras=N`, a
  `LoRARequest` per adapter); base family/size/class/engine config must still match
  (`assert_shared_model(..., allow_varying=SHARED_BASE_VARYING_KEYS)`). Each distinct
  adapter must mount at a **distinct channel**: the generator names it
  `lora-<run-id>`. Shared-base channel unions are emitted by the tracked study
  materializers; there is no standalone `plan_eval.py` launcher in this release.
  Use it to collapse e.g. {off the shelf + OC-SFT}×collections from 2 jobs into 1. Distinct
  *base* models still cannot share a bundle.
- **Per-collection output:** each collection writes a standard `runs/<member-id>/<ts>/`
  tree (`metrics.json` + optional `psi/`), so one bundle produces **all** the
  collection trees and each collection records like a normal run.
- **Logging:** each collection gets its own on-disk logs
  (`experiment_*.log` / `evaluation_*.log` / `psi/psi_run_*.log`) in its run dir; the bundle driver resets the (name-cached) run-scoped loggers between collections so
  collections 2..N don't fall into collection 1's files. The combined log stream is
  sequential (collections run serially) and bracketed by `[bundle] collection i/N: <id>`
  markers. The entrypoint emits `METRIC <measure>=<value>` lines per collection, so
  each measure recurs once per collection: a scraper sees a **time series across
  collections**, not a per-collection breakdown.
  `runs/<member-id>/<ts>/metrics.json` holds the per-collection measurements.
- **Channels:** a container run mounts each member's data channel (from its
  `data.run_path` / `execution.fixture_channel`) plus the bundle's shared
  `execution.extra_input_channels`. Staging is bundle-aware: it fans out across
  members and revisits shared channels for member-specific
  side files (qid caps, permutation manifests), so members do not need separate
  staging calls.
- **Failure isolation:** a crash on one collection (e.g. a transient OOM on a long
  pole) is recorded and the bundle continues, so collections already done in the job
  aren't lost. `run_bundle` always writes `bundle_manifest.json` (per-collection
  `status: ok|failed`, `n_ok`, `n_failed`) and the run **succeeds** unless *every*
  collection failed, so the completed collections are kept. Because collections share one
  reused reranker, transient hooks a
  prior (possibly crash-interrupted) collection may have left set; currently
  `render_variant`: are reset before each collection so isolation can't silently
  corrupt a later collection's scores.
- **Timeout/failure is recoverable per collection:** completed collections persist; re-run
  the `failed`/unfinished members (or qids) and merge
  (`scripts/data/finalize_psi_topup.py`).

## Output layout

```text
runs/<ID>/<timestamp>/
    resolved_config.yaml              # snapshot of this YAML + git SHA at run time
    experiment.log
    per_query_results/<qid>/
        trec_results_raw.txt
        detailed_results.json
    metrics.json                      # written by run_eval.py
```
