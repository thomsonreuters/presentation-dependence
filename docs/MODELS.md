# Model usage

Choose, load, and evaluate rerankers and readers.
The source-code extension checklist remains in
`src/presentation_dependence/rerankers/README.md`.

## Quick choice

- No GPU / plumbing: `IdentityReranker` or BM25 import.
- Local pointwise baseline: `MxbaiPointwise`.
- Default expected-grade scorer: `Qwen3InstructGradeReranker`.
- Alternative expected-grade family: `Granite41GradeReranker` or
`Gemma4GradeReranker`.
- Published external reranker: `JinaListwiseReranker`,
`RankZephyrReranker`, `Qwen3Reranker`, `PairRMReranker`, or
`SkyworkBTReranker`.
- Calibration/control: `CapCalReranker`, `PineReranker`, matched variance,
or round-robin examples.
- Closed teacher: `ClosedModelGenBscReranker`.



## Common contract

Every experiment selects a class in YAML:

```yaml
reranker:
  class: Qwen3InstructGradeReranker
  model_name: Qwen/Qwen3-4B
  revision: 1cfa9a7208912126459214e8b04321603b3df60c
  device: auto
  dtype: bfloat16
```

The class is resolved through `rerankers/registry.py`. A `revision` is enforced
only when the runtime loader receives it. Canonical reproduction-pipeline models
must match `source-lock.yaml`; many standalone examples and breadth configs
still omit the field and therefore follow the Hub default. Expected-grade
models must implement the
[expected-grade readout contract](SCORING.md).

## Backends



### HF

Use when:

- running best-effort development inference on CPU/MPS;
- eager attention is required;
- training LoRA;
- validating token/readout parity.



### vLLM

Use when:

- running expected-grade inference on CUDA;
- scoring B-document contexts with continuous batching;
- serving a supported LoRA adapter;
- executing large sweeps.

The vLLM and SFT Dockerfiles are separate dependency-image recipes. Their
`BASE_IMAGE` is operator-supplied and is not pinned by digest in this release;
record and pin the chosen base digest for a citable run.
For supported expected-grade wrappers, select the backend with
`reranker.inference_engine: hf | vllm`. vLLM is CUDA-only and lazy-imported;
unsupported readout/backend combinations fail during config loading. A vLLM
wrapper does not also load the HF model on the same device. Teacher batching and
resume semantics are in
[TRAINING.md](TRAINING.md#open-weight-teachers).

Before changing a default backend, verify the frozen prompt/readout on the target
image with `scripts/analyze/probe_vllm_speedup.py`. Keep the CUDA stack,
tensor-parallel settings, and parity tolerance with the resulting artifact.
Wrapper expansion beyond the shipped set of wrappers is not an operator contract.

## Model catalog


| Class                        | Role                                            | Typical backend | LoRA                                | Start here                                                             |
| ---------------------------- | ----------------------------------------------- | --------------- | ----------------------------------- | ---------------------------------------------------------------------- |
| `IdentityReranker`           | plumbing smoke                                  | none            | no                                  | `configs/experiments/_smoke-identity.yaml`                             |
| `MxbaiPointwise`             | order-free baseline                             | HF              | no                                  | `configs/experiments/example-passage-mxbai-large-v2.yaml`              |
| `Qwen3InstructGradeReranker` | expected-grade / OC-SFT                         | HF, vLLM        | yes                                 | `configs/experiments/example-passage-b20-psi.yaml`                     |
| `Granite41GradeReranker`     | expected-grade family                           | HF, vLLM        | yes                                 | `configs/experiments/example-passage-granite-b20-psi.yaml`             |
| `Gemma4GradeReranker`        | expected-grade MoE/dense                        | HF, vLLM        | merge for some vLLM paths           | [MODELS.md](MODELS.md#trained-students)                              |
| `Qwen3Reranker`              | pointwise reference; specialized-base control   | HF, vLLM        | paper trains it; recipe not shipped | `configs/silver/a2-qwen3-reranker-k10-bsc-msmarco-30k.yaml`            |
| `JinaListwiseReranker`       | scoring-listwise reference                      | HF              | no                                  | `configs/experiments/example-passage-jina-v3-b20.yaml`                 |
| `RankZephyrReranker`         | generative listwise reference                   | HF              | no                                  | `configs/experiments/example-passage-rankzephyr-sliding-window.yaml`   |
| `PairRMReranker`             | response preference reference                   | HF              | no                                  | `configs/experiments/example-response-ranking-pairrm.yaml`             |
| `SkyworkBTReranker`          | response reward-model reference                 | HF              | no                                  | `configs/experiments/example-response-ranking-skywork-v2.yaml`         |
| `CapCalReranker`             | probability calibration                         | wraps base      | inherited                           | `configs/experiments/example-passage-capcal-b20-psi.yaml`              |
| `PineReranker`               | expected-grade compatibility / negative control | HF/eager        | no                                  | `configs/experiments/example-passage-pine-b20-psi.yaml`                |
| `ClosedModelGenBscReranker`  | closed/API BSC                                  | API             | no                                  | `configs/experiments/example-passage-closed-model-genbsc-b20-psi.yaml` |


Each reference reranker has a standalone example config, and is also declared as
a reference arm inside its task pipeline for the full matrix.

The manuscript additionally trains order-averaged distillation and OC-SFT from
`Qwen3-Reranker-4B` as a specialized-base control. This release retains the
teacher config but not the corresponding student-training recipe, so those
trained control rows are not materializable from the current public pipeline.

`JinaListwiseReranker` requires `trust_remote_code=True`, so the wrapper rejects
anything other than an immutable 40-character commit SHA. The standalone example
and all three reproduction pipelines use the reviewed revision recorded in
`configs/reproduction/source-lock.yaml`.

`rerankers/registry.py` lists additional experimental wrappers. Wrappers not
listed here have no maintained operator walkthrough.

## Checkpoint licences

See [`../THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md#models). The
structural audit rejects non-commercial checkpoints in teacher, student,
fine-tuning-base, and training-candidate roles.

## Base revisions

A few base revisions, and what each one is evidence of:


| Base                          | Revision                                   | Recorded in                                                         | Pins                                                          |
| ----------------------------- | ------------------------------------------ | ------------------------------------------------------------------- | ------------------------------------------------------------- |
| `Qwen/Qwen3-4B`               | `1cfa9a7208912126459214e8b04321603b3df60c` | `configs/reproduction/shared.yaml`, `passage-reranking.yaml`        | reproduction pipeline                                         |
| `ibm-granite/granite-4.1-8b`  | `1504002f650e656a0a3789d99574df12e3e94ed0` | `configs/reproduction/multi-document-qa.yaml`                       | reproduction pipeline                                         |
| `ibm-granite/granite-4.1-30b` | `4fae6278f7132abf5e971f9de49ebbad09c54cce` | `configs/silver/granite-41-30b-grade-cont-k10-bsc-msmarco-30k.yaml` | the 30B teacher                                               |
| `google/gemma-4-E4B-it`       | `d6436b3d62967e1af08bbb046c6300b2a9ae8e85` | pinned by revision                                                  | the snapshot the Gemma-4 LoRA adapters are **served** against |


`declared_model_pins` reads only the four high-level reproduction pipeline
files. The Qwen3-4B and Granite-8B rows above flow through those files; pinned
external references in the same pipelines do too. Standalone silver/student
configs and the breadth-model templates are outside that scan, so a passing
structural source-lock check does not pin the complete paper grid.

If a future run needs a citable base revision, set `revision:` in its config
before launching. That is the only mechanism in the pipeline today that makes the
base identity part of the run record.

## Tracked examples

Expected-grade students and protocols, under `configs/experiments/`:

- `example-passage-b20-psi.yaml`
- `example-passage-granite-b20-psi.yaml`
- `example-passage-round-robin-b20-psi.yaml`
- `example-passage-dense-first-stage.yaml`
- `example-passage-matched-variance-b1.yaml`
- `example-passage-pool-perturbation.yaml`
- `example-multi-document-qa-b10-psi.yaml`
- `example-response-ranking-b4-psi.yaml`

Reference rerankers, one per class:

- `example-passage-mxbai-large-v2.yaml`
- `example-passage-jina-v3-b20.yaml`
- `example-passage-rankzephyr-sliding-window.yaml`
- `example-passage-capcal-b20-psi.yaml`
- `example-passage-pine-b20-psi.yaml`
- `example-passage-closed-model-genbsc-b20-psi.yaml`
- `example-response-ranking-pairrm.yaml`
- `example-response-ranking-skywork-v2.yaml`

Smoke configs: `_smoke-fixture.yaml` is the zero-dependency install check
(FixtureLoader, synthetic data; see `poe smoke`). `_smoke-identity.yaml` is the
first real-data check on DL19 (pyserini; needs materialized data).

Related:

- `configs/self-distill/example-teacher-transfer-qwen3-1p7b-to-qwen3-4b.yaml`
- `configs/reader/example-answer-reader-hotpotqa.yaml`
- `configs/reader/example-verdict-reader-climate-fever.yaml`

Examples intentionally omit account-specific credentials and checkpoint URIs.

## Off-shelf local examples



### Identity

```bash
uv run python scripts/run_experiment.py \
  -e configs/experiments/_smoke-identity.yaml
uv run python scripts/run_eval.py -e _smoke-identity
```



### mxbai

```bash
uv run python scripts/run_experiment.py \
  -e configs/experiments/example-passage-mxbai-large-v2.yaml
uv run python scripts/run_eval.py -e example-passage-mxbai-large-v2
```

For the reviewed DL19 fixture, nDCG@10 should be in `[0.72, 0.74]`.

### Qwen3 expected-grade readout

```bash
RUN_DIR="runs/example-passage-b20-psi/$(date +%Y%m%d_%H%M%S)"
uv run python scripts/run_experiment.py \
  -e configs/experiments/example-passage-b20-psi.yaml --run-dir "$RUN_DIR"
uv run python scripts/run_psi.py \
  -e configs/experiments/example-passage-b20-psi.yaml --run-dir "$RUN_DIR"
```

Both phases share `--run-dir` so the identity-order pass and the permutation
pass describe one run; omitting it puts them in two timestamped directories.

Requires the DL19 fixture, model access, and a suitable GPU for practical
throughput. Build the fixture after Pyserini setup; the command is in
[DATA-SETUP.md](DATA-SETUP.md). It takes explicit `--topics` / `--run` /
`--index` flags rather than `-e`, because this config is a `FixtureLoader` one
and so names no first stage to read them from.

### Granite

```bash
uv run python scripts/run_experiment.py \
  -e configs/experiments/example-passage-granite-b20-psi.yaml
```



## Reference rerankers

Use tracked declarations that have passed review rather than reconstructing
model-specific prompts.

Each has a standalone example config, listed in the catalog above:

```bash
uv run python scripts/run_experiment.py \
  -e configs/experiments/example-passage-rankzephyr-sliding-window.yaml
uv run python scripts/run_eval.py -e example-passage-rankzephyr-sliding-window
```

The same rerankers are also reference arms in their task pipelines, which is how
the full matrix generates them after training collection has written
`build/reproduction/<task>/training/checkpoints.json`:

```bash
uv run python scripts/study.py passage-reranking direct-eval materialize
uv run python scripts/study.py passage-reranking direct-eval validate
```

Generated configs land under
`build/reproduction/passage-reranking/direct-eval/configs/` and can then be run
with `scripts/run_experiment.py -e <generated-config>`.
On a source-only checkout without that checkpoint catalog, use the standalone
example configs instead; direct-eval materialization intentionally stops.

- mxbai is the local pointwise reference.
- RankZephyr must use its native window/list protocol.
- PINE must use HF/eager attention; stock vLLM is not faithful.
- CapCal wraps a scalar/probability-producing base reranker.
- Skywork and PairRM are response-ranking anchors, not passage expected-grade
models.



## Trained students

Trained students are LoRA adapters that must be paired with the exact base used
for training. Inference wrappers enforce `reranker.revision` when present.
Student training currently has no separate revision field, so a reproducible
training run must set `student.base_model` to a prefetched immutable snapshot
path. Never substitute a different base, grade rubric, free-form decoding, or
the latest checkpoint in place of the selected catalog row.
[`EVAL-PROTOCOL.md`](EVAL-PROTOCOL.md) defines that catalog and selection.

The eleven primary OC-SFT students use these dense bases, one per base, each
distilled from its same-family open teacher:

| Family | Bases |
| --- | --- |
| Qwen3 | 1.7B, 4B, 8B, 14B, 32B |
| Gemma 4 | E2B, E4B, 31B |
| Granite 4.1 | 3B, 8B, 30B |

This source-only distribution contains no adapters or checkpoints, including
the separately reported GPT-5.4 GenBSC-taught Qwen3-4B student.

To evaluate one locally, point a tracked config at an adapter directory:

```bash
uv run python scripts/run_experiment.py \
  -e configs/experiments/example-passage-b20-psi.yaml \
  --override reranker.lora_path=/absolute/path/to/checkpoint-step-001000
uv run python scripts/run_psi.py \
  -e <config-id> \
  --override reranker.lora_path=/absolute/path/to/checkpoint
```

The adapter must match the config's base revision and LoRA target modules.
Check `resolved_config.yaml` afterwards to confirm which adapter and revision
were used. Generate evaluation configs with
`study.py <task> direct-eval materialize` rather than hand-editing this
override for a matrix.

HF loads the base revision and applies the adapter with PEFT, which is the path
for validating a new adapter, running small models on CPU or MPS, using eager
attention, or checking readout parity against vLLM. vLLM loads the pinned base,
enables LoRA, and creates a `LoRARequest` for the adapter, applying the same
prompt and grade-token readout and preserving input-order alignment:

```yaml
reranker:
  inference_engine: vllm
  lora_path: /path/to/checkpoint-step-001000
  max_lora_rank: 16
  vllm_settings:
    max_model_len: 4096
    tensor_parallel_size: 1
```

An adapter must not change tokenizer or prompt identity unless the training
contract declares it, and expected-grade logit readout must not become
generated-grade decoding. [SCORING.md](SCORING.md) is the contract.

A bundle may serve several adapters over one base. Member configs must pass
`assert_shared_model` on base model, revision, tokenizer, engine, and reranker
settings, every adapter must be present, and the LoRA rank limit must cover all
of them. Split a bundle that fails validation rather than relaxing the
validator.

When an adapter and base disagree you get missing modules, PEFT shape errors, or
scores inconsistent with the reference fixture; compare base revision and target
modules against the training config. When vLLM rejects an adapter, check the
architecture, rank, target modules, and vLLM version, and whether a merge is
required. When the adapter is right but the scores are wrong, check the readout,
rubric, grade scale, prompt, chat-template kwargs, document caps, serving width,
dtype, and tokenizer revision.

## Large models

For 30B/32B or large MoE checkpoints:

1. warm the exact HF revision into the Hub cache once;
2. ensure the model fits one visible device under a tracked TP=1 config;
3. avoid re-downloading the base for every run.

The release does not ship a validated TP>1 profile. See [HARDWARE.md](HARDWARE.md) for the distinction
between replication and tensor parallelism.

## Gemma-4 adapters

Some Gemma-4 and vLLM combinations cannot apply the research LoRA at runtime.
Merge instead: select the adapter checkpoint, load the pinned base and adapter
with `scripts/train/merge_lora_entrypoint.py`, write the merged weights, and
evaluate those. Record the base revision, the adapter URI and hash, and the
merged snapshot hash, and pin the base by revision so the merge is reproducible.
Evaluating the unadapted base is not a substitute.

```bash
uv run --extra training python scripts/train/merge_lora_entrypoint.py \
  --base-dir /path/to/pinned-base-snapshot \
  --adapter-dir checkpoints/<experiment>/<trial>/student/checkpoint-step-<step> \
  --output-dir checkpoints/merged/<semantic-id>
```

The manifest is written to `<output-dir>/merge_manifest.json` locally. Use
`--manifest-dir` to separate it. Containers may instead provide
`SLM_CHANNEL_BASE_MODEL`, `SLM_CHANNEL_LORA_ADAPTER`, and `SLM_OUTPUT_DIR`; the
legacy `SM_CHANNEL_*` / `SM_OUTPUT_DATA_DIR` aliases remain compatible with
older launchers.

## Readers

Readers are frozen downstream consumers of rankings. They load scorer PSI and
aligned-score artifacts plus answer and verdict sidecars, and do not load a
scorer LoRA checkpoint, so run and collect the scorer's artifacts first.

```bash
uv run --extra vllm python scripts/run_reader.py \
  --config configs/reader/example-answer-reader-hotpotqa.yaml \
  --scorer-run runs/example-multi-document-qa-b10-psi/<trial>
```

The scorer run must contain `psi/beta_gamma_scores.parquet`. `run_psi.py`
writes it only when the scorer config sets `robustness.beta_gamma.enabled: true`, so set that before the scorer run rather than after; the example configs
above already do. Use `--reader-engine echo` for plumbing. Reader configs do
not train models.

## Closed/API teachers

Closed models are scored through an OpenAI-compatible chat endpoint.
Required credentials and resume/finalize behavior are documented in
[TRAINING.md](TRAINING.md) and
[TROUBLESHOOTING.md](TROUBLESHOOTING.md).

## Validate a new model

1. Identify its protocol: pointwise, scoring-listwise, generative-listwise, or
  batched-pointwise.
2. Add the wrapper and registry entry.
3. Add contract/unit tests.
4. Verify empty input and input-order score alignment.
5. Run one fixed-fixture HF probe.
6. If adding vLLM, verify numerical and readout parity.
7. Add a tracked example config with a descriptive ID.
8. Pin model/image identity.
9. Match population, candidate depth, document cap, and metric to the intended
  comparison, or record the difference explicitly.
10. Record selected-run metadata through `RUN-ARTIFACTS.md`.

```bash
uv run --no-sync poe smoke
uv run python scripts/run_experiment.py -e <new-example.yaml> --dry-run
uv run python scripts/run_experiment.py -e <new-example.yaml>
```

The first command checks shared identity/evaluation plumbing only. The latter
two exercise the actual wrapper and may require model access and a GPU.

## Scoring readout

The Qwen3/Granite/Gemma batched-pointwise scorers use the expected-grade readout
defined in [SCORING.md](SCORING.md). Pointwise cross-encoders, reward models,
PairRM, Jina, and RankZephyr use their own published interfaces. Loading an
expected-grade checkpoint with a different readout produces different scores.