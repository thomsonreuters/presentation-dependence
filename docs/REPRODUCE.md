# Reproducing the results

Three depths, in increasing cost: the zero-dependency smoke that the install
works, one paired base-versus-student result end to end, and the full task
matrix. Start at the top even if you intend to run the whole thing, because each
depth validates the setup the next one assumes.

Commands assume the working directory is the package root.

## Reproduction fidelity

The reproduction layer was built after the original experiments ran. It
reconstructs the declared experiment matrix from tracked declarations, so it is a
best-attempt reproduction rather than a guarantee of bit-identical agreement with
the published numbers.

The audits check that tracked source is sufficient to rebuild the plans and that
identities declared by the high-level pipelines/source lock are pinned. They do
not pin the complete breadth grid or verify that a fresh execution re-derives
the reported metrics. A `status: complete` from an audit
means the declarations are complete, not that numbers were recomputed;
`study.py structural audit` repeats this as a `disclaimer` key in its JSON output
so the caveat travels with the verdict.

Expect close agreement rather than identity. Known sources of numeric divergence
are vLLM and GPU nondeterminism, differences in accelerator type, BM25 regime
differences that move nDCG@10 by 1 to 3 points when the `dataset_meta.yaml`
convention is not followed, additive rerun behaviour that reuses existing
per-query results, teacher sampling variation, and LoRA and DDP nondeterminism.

Legal-A and Legal-B are not distributed. The sixteen non-internal
per-collection rows do not depend on them. Six of those sixteen rows are not automatically
provisioned: DL21–DL23 are manual and Signal-1M, TREC-News, and Robust04 are
access-gated. Recomputing the exact paper values also requires selected
checkpoints and run artifacts that this source distribution does not embed.

## Prerequisites

```bash
./setup.sh --full
uv run --no-sync poe smoke
export HF_TOKEN=...
```

Setup downloads dependencies on a cold cache; only the subsequent
`uv run --no-sync` gate is network-free. `HF_TOKEN` is needed for model and
dataset access. `--full` does not install vLLM. The teacher commands below use
`uv run --extra vllm`; use the Hugging Face path on macOS or CPU. Data setup is
task-specific and starts in [Materializing the data](#materializing-the-data);
full population validation and source-lock verification are intentionally not
installation prerequisites.

## Zero-dependency smoke

The install check needs no data, network, GPU, Java, or Hugging Face cache.
`poe smoke` builds a synthetic fixture, runs it through
`ExperimentManager` → `EvalManager` → `pytrec_eval` with the identity reranker,
and asserts a fixed nDCG@10 of about 0.6805:

```bash
uv run --no-sync poe smoke
```

Expanded form (same path as the `poe` task):

```bash
uv run python scripts/data/build_smoke_fixture.py
uv run python scripts/run_experiment.py -e configs/experiments/_smoke-fixture.yaml
uv run python scripts/run_eval.py -e _smoke-fixture
uv run python scripts/check_smoke_result.py
```

This is plumbing only. Do not treat it as a substitute for a
first-stage or model check on real collections.

## First real-data check

After DL19 is materialized (next section), score the BM25 first stage with no
model. That exercises data loading, TREC output, and `pytrec_eval` on real
inputs:

```bash
uv run python scripts/import_run_as_baseline.py \
    -r data/dl19-passage/run.msmarco-v1-passage.bm25-default.dl19_sorted.txt \
    -q data/dl19-passage/qrels.txt \
    -i bm25-dl19-baseline
uv run python scripts/run_eval.py -e bm25-dl19-baseline
```

Expect mean `ndcg_cut_10` near 0.506. To exercise the reranker path on the same
data still without a model:

```bash
uv run python scripts/run_experiment.py -e configs/experiments/_smoke-identity.yaml
uv run python scripts/run_eval.py -e _smoke-identity
```

If both succeed, the data and evaluation path is ready. Neither replaces
[`poe smoke`](#zero-dependency-smoke).

## Materializing the data

`data/` is untracked, so rebuild it on a new machine. This needs Java 21 and a
Hugging Face token. The first index fetch downloads about 1.5 GB into
`~/.cache/pyserini/`; the full BEIR set adds several GB more.

```bash
export JAVA_HOME=/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home  # macOS example
uv run python scripts/setup_reproduction_data.py plan --task passage-reranking
uv run python scripts/setup_reproduction_data.py run --task passage-reranking
```

`plan` previews all 18 canonical rows including the manual and restricted ones;
`run` materializes each missing public automatable member. Signal-1M, TREC-News,
and Robust04 need licences plus `--include-gated`. DL21 through DL23, Legal-A,
and Legal-B stay manual rows. The per-dataset commands target the exact run and
qrels paths declared by
`configs/reproduction/populations/reranking-primary-18.yaml`.
Collection terms are listed in
[`../THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md#datasets).
`[DATA-SETUP.md](DATA-SETUP.md)` covers the other two tasks.

`setup_reproduction_data.py validate` is a full-population gate and therefore
still reports every missing gated, manual, and internal row after the default
automated run. `source_lock.py verify` excludes Legal-A/B but still requires all
other public, gated, and manual files. Use neither as an automated-subset
success check; their exact scopes are in `[DATA-SETUP.md](DATA-SETUP.md)`.

## One result end to end

This walkthrough measures robustness amortization: the difference between an
off-shelf base and a trained student on the same dataset. The off-shelf base at
K=1 has high τ-PSI. A student trained on K=10 self-consistency silver serves at
K=1 and reduces τ-PSI without losing nDCG@10.

It uses OG Qwen3-4B (non-thinking), an MS MARCO 30K K=10 batched-self-consistency
teacher, and a single-pass K=1 student. The pipeline is identical up to training,
where three recipes share the base, teacher silver, LoRA, AdamW, cosine schedule,
DDP settings, and checkpoint policy, and differ only in the target and objective:


| Recipe   | Config id under `configs/self-distill/`                                          | Silver target    | Objective                      |
| -------- | -------------------------------------------------------------------------------- | ---------------- | ------------------------------ |
| K=1 SFT  | `qwen3-4b-nonthink-sft-msmarco-30k-k1-labels`                                    | single-order K=1 | MSE                            |
| K=10 SFT | `qwen3-4b-nonthink-sft-msmarco-30k-k10-labels`                                   | K=10 BSC average | MSE                            |
| OC-SFT   | `qwen3-4b-nonthink-k1-supervised-consistency-lambda<code>-warmup500-msmarco-30k` | single-order K=1 | MSE plus consistency penalty λ |


K=1 SFT isolates the gain from teaching the expected-grade task and output
format. K=10 SFT isolates the gain from amortizing the ensemble into the labels.
OC-SFT distills the same K=1 silver and adds a penalty tying two shuffled views
together, reaching the ensemble's stability from a single teacher pass. The
commands below default to K=10 SFT.

### Stage 1: build the training candidates

```bash
uv run python -m scripts.data.setup_msmarco_self_distill --query-count 30000
```

Writes `data/msmarco-train-selfdistill-seed42/` with `fixture.jsonl`,
`qrels.txt`, `topics.tsv`, and `qids_30k.txt`. The 1K, 30K, and 100K sets are
nested prefixes. Needs `JAVA_HOME` and pyserini; topics and qrels come from
ir_datasets (`msmarco-passage/train`) and download on first run.

### Stage 2: generate silver labels

The teacher scores each query under K=10 shuffled permutations using the
expected-grade readout on vLLM, then averages them into continuous `[0,3]`
labels. The config is
`[configs/silver/qwen3-4b-nonthink-k10-bsc-msmarco-30k.yaml](../configs/silver/qwen3-4b-nonthink-k10-bsc-msmarco-30k.yaml)`
(`class: Qwen3InstructGradeReranker`, `model_name: Qwen/Qwen3-4B`,
`enable_thinking: false`, `inference_engine: vllm`, `k_perms: 10`,
`cross_query_batch: 4`).

Run the reduced check before committing GPU hours:

```bash
uv run --extra vllm python scripts/run_silver_generation.py \
    --config qwen3-4b-nonthink-k10-bsc-msmarco-30k \
    --override 'qids_to_run=[1037798]' \
    --override teacher.k_perms=2 \
    --override 'teacher.seeds=[0, 1]'

uv run --extra vllm python scripts/run_silver_generation.py \
    --config qwen3-4b-nonthink-k10-bsc-msmarco-30k
```

For a functional HF check without vLLM (slow on CPU and not a substitute for
the tracked CUDA/vLLM teacher path), override the tracked CUDA/vLLM settings
explicitly:

```bash
uv run python scripts/run_silver_generation.py \
    --config qwen3-4b-nonthink-k10-bsc-msmarco-30k \
    --override 'qids_to_run=[1037798]' \
    --override teacher.k_perms=2 \
    --override 'teacher.seeds=[0, 1]' \
    --override teacher.local_data_parallel_workers=1 \
    --override teacher.cross_query_batch=1 \
    --override reranker.inference_engine=hf \
    --override reranker.device=auto
```

A completed pass writes `silver/silver_labels.jsonl` and `manifest.json`,
containing 30,000 queries and 2,999,860 records, with train-qrels silver nDCG@10
near 0.311 for OG 4B (Pearson near 0.898 against the 4B-Instruct-2507 silver).

Verify coverage and schema:

```bash
python3 - <<'EOF'
import json
p="runs/self-distill/qwen3-4b-nonthink-k10-bsc-msmarco-30k/<ts>/silver/silver_labels.jsonl"
n=0; qids=set(); klens=set(); bad=0
for line in open(p):
    r=json.loads(line); n+=1; qids.add(r["query_id"])
    klens.add(len(r.get("score_raw_vector") or []))
    if not 0.0<=r["score_continuous"]<=3.0: bad+=1
print("records",n,"queries",len(qids),"k-vector lengths",klens,"out-of-range",bad)
EOF
# expect: records 2999860  queries 30000  k-vector lengths {10}  out-of-range 0
```

Check the silver as a prediction, which reranks the fixture by silver score
against the train qrels. The silver row should read near 0.311, above BM25 and
below the NIST oracle:

```bash
uv run python scripts/data/eval_silver_ndcg.py \
    --run-dir runs/self-distill/qwen3-4b-nonthink-k10-bsc-msmarco-30k/<ts>/silver \
    --fixture data/msmarco-train-selfdistill-seed42/fixture.jsonl \
    --qrels   data/msmarco-train-selfdistill-seed42/qrels.txt
```

`--run-dir` is the directory that directly holds `silver_labels.jsonl`. The
script prints three rows: `bm25_baseline`, `silver`, and `oracle_nist`.

### Stage 2 to Stage 3: split and derive

The student config does not read `silver/silver_labels.jsonl`. It reads the
filenames named in `student.silver_labels_path` and `eval_silver_labels_path`:
the K=10 split for K=10 SFT, and the derived K=1 split for K=1 SFT and OC-SFT.

Place the teacher output under the name the split step expects, then split it
into 29.5K train and 500 held-out shards. The held-out qids are a deterministic
suffix of the seed-42 order.

```bash
cp runs/self-distill/qwen3-4b-nonthink-k10-bsc-msmarco-30k/<ts>/silver/silver_labels.jsonl \
   data/msmarco-train-selfdistill-seed42/silver_labels_qwen3_4b_k10.jsonl

uv run python scripts/data/prepare_heldout_eval_cohort.py \
    --silver-in data/msmarco-train-selfdistill-seed42/silver_labels_qwen3_4b_k10.jsonl \
    --n 500
```

For K=1 SFT and OC-SFT only, derive the K=1 labels from the K=10 vector on both
shards. With `--k-out 1` the script renames `_k10` to `_k1_seed0`, matching those
configs:

```bash
for shard in train_29500 heldout_500; do
  uv run python scripts/derive_k1_silver_from_bsc.py \
      --silver-in data/msmarco-train-selfdistill-seed42/silver_labels_qwen3_4b_k10_${shard}.jsonl \
      --k-out 1
done
```



### Stage 3: train the student

This stage needs a GPU; there is no CPU alternative.

In an OC-SFT config id, `<code>` is λ times 100, zero-padded to three digits:
`050` is 0.5, `100` is 1.0, `200` is 2.0, `300` is 3.0, `400` is 4.0, and `500`
is 5.0. `warmup500` is separate and sets the 500-step linear ramp of λ from zero
to target (`objective.lambda_warmup.steps`). The SFT recipes set
`student.data.view: regression` with no `objective` block; OC-SFT sets
`student.data.view: supervised_consistency` plus:

```yaml
objective:
  type: supervised_consistency
  loss: mse
  lambda: 5.0                # from the filename: <code> 500 -> λ 5.0
  view_seeds: [0, 1]         # the two shuffled views the penalty ties together
  lambda_warmup: {steps: 500, init: 0.0, schedule: linear}
```

Do not hand-pick λ. It is selected per model on the held-out split by
`scripts/select_lambda.py` under the rules in
`[EVAL-PROTOCOL.md](EVAL-PROTOCOL.md)`. To reproduce one result, train the
selected λ config directly, or train the λ ∈ {0.5, 1, 2, 3, 4, 5} sweep and
select afterwards.

```bash
uv run python scripts/study.py passage-reranking training materialize

ID=passage-reranking-training--k10-sft--seed42
CONFIG=build/reproduction/passage-reranking/training/configs/$ID.yaml
DATA=data/msmarco-train-selfdistill-seed42

# Functional check: isolated from the trial that collection will consume.
SMOKE_TRIAL=smoke-$(date -u +%Y%m%d_%H%M%S)
uv run python scripts/run_self_distill_sft.py -e "$CONFIG" \
    --override student.training.max_steps=5 \
    --override student.data.max_train_queries=64 \
    --override student.silver_labels_path="$DATA/silver_labels_qwen3_4b_k10_train_29500.jsonl" \
    --override student.eval_silver_labels_path="$DATA/silver_labels_qwen3_4b_k10_heldout_500.jsonl" \
    --override student.fixture_path="$DATA/fixture.jsonl" \
    --override student.qrels_path="$DATA/qrels.txt" \
    --override student.output_dir="runs/$ID/$SMOKE_TRIAL/student" \
    --override student.checkpoint.dir="checkpoints/$ID/$SMOKE_TRIAL/student"

# Collector-compatible full trial.
TRIAL=$(date -u +%Y%m%d_%H%M%S)
uv run python scripts/run_self_distill_sft.py -e "$CONFIG" \
    --override student.silver_labels_path="$DATA/silver_labels_qwen3_4b_k10_train_29500.jsonl" \
    --override student.eval_silver_labels_path="$DATA/silver_labels_qwen3_4b_k10_heldout_500.jsonl" \
    --override student.fixture_path="$DATA/fixture.jsonl" \
    --override student.qrels_path="$DATA/qrels.txt" \
    --override student.output_dir="runs/$ID/$TRIAL/student" \
    --override student.checkpoint.dir="checkpoints/$ID/$TRIAL/student"
```

Run the five-step version first as a functional check. Training is single-node
DDP when the config declares `execution.distributed: ddp` and more than one GPU
is visible: the runner re-execs itself under torchrun with one worker per GPU,
sizing from `SLM_NUM_GPUS` or the detected device count. On one GPU it says so
and continues in a single process.

The explicit output overrides above are required for local collection. Source
templates otherwise write directly to `runs/self-distill/<id>/student`, with no
trial level, while the reproduction collector consumes
`runs/<generated-id>/<trial>/student/training_summary.json` and constructs the
parallel checkpoint path
`checkpoints/<generated-id>/<trial>/student/checkpoint-step-XXXXXX/`. Container
entrypoints mint the trial-shaped run tree themselves; the plain local runner
does not. Live curves come from
`uv run --extra training tensorboard --logdir runs/`.

Confirm the trial carries the selection input before moving on:

```bash
test -f "runs/$ID/$TRIAL/student/training_summary.json"
```



### Stage 4: evaluate both arms

Evaluate the off-shelf base with no adapter and the trained student with the
Stage-3 adapter, on the same dataset. Each writes `metrics.json` for nDCG, MAP,
and MRR, and `psi/psi_metrics.json` for τ-PSI, Kendall τ, and ΔnDCG.

Select the checkpoint first, since Stage 3 emits several. The training loop
retains checkpoints ranked by held-out `qrels_ndcg_cut_10`; K=10 SFT selects step
1200 in the recorded run. Plain SFT takes the highest-ranked retained checkpoint,
and OC-SFT applies `scripts/select_lambda.py` across the λ grid.

```bash
uv run python scripts/study.py passage-reranking training collect \
  --runs-root runs --discover-latest
uv run python scripts/study.py passage-reranking direct-eval materialize
uv run python scripts/study.py passage-reranking direct-eval validate
```

For one DL19 comparison, run the generated configs:

```bash
BASE=build/reproduction/passage-reranking/direct-eval/configs/passage-reranking-direct-eval--off-shelf--dl19.yaml
STUDENT=build/reproduction/passage-reranking/direct-eval/configs/passage-reranking-direct-eval--k10-sft--seed42--dl19.yaml

uv run python scripts/run_psi.py -e "$BASE"
uv run python scripts/run_psi.py -e "$STUDENT"
```

On DL19 the selected OG Qwen3-4B K=10 SFT student reaches nDCG@10 near 0.73 with
τ-PSI near 0.12, against an off-shelf base with similar or lower nDCG and higher
τ-PSI.

### Stage 5: record the result

```bash
uv run python scripts/record_result.py runs/<ID>/<timestamp> \
    --status complete --role primary --summary "one sentence"
uv run python scripts/record_result.py --validate-index
uv run python scripts/query_results.py --collection dl19 --model-family qwen3
```

`docs/results_index.yaml` ships empty. It is where you record your own selected
runs, not a record of ours.

## The full task matrix

```text
scripts/study.py <task> <stage> <command>
```

Tasks are `passage-reranking`, `multi-document-qa`, and `response-ranking`.
Stages are `silver`, `training`, `direct-eval`, and `downstream-eval`. Commands
are `plan` to inspect without writing, `materialize` to generate configs under
`build/`, `validate` to verify declarations and inputs, `execute --dry-run` to
print the commands, `execute --run` to run them, `collect` to convert runs into
task outputs, and `summarize` to print the aggregate.

Task declarations live in `configs/reproduction/`, one YAML per task plus
`shared.yaml`.


| Task              | Population    | Primary training jobs | Reported direct metric | Downstream consumer                                                                |
| ----------------- | ------------- | --------------------- | ---------------------- | ---------------------------------------------------------------------------------- |
| Passage reranking | Primary 18    | 24                    | nDCG@10 and τ-PSI      | Retained-set reduction                                                             |
| Multi-document QA | 3 collections | 18                    | nDCG@10 and τ-PSI      | Frozen QA answer reader; Climate-FEVER verdict is a separate passage-scorer bridge |
| Response ranking  | 5 collections | 30                    | nDCG@1 and τ-PSI       | Response selection and pair flip                                                   |


Counts come from the tracked task YAML and are frozen by the structural audit.
Optional ablations and controls are excluded from them; add `--include-ablations`
to a `training` stage or `--include-controls` to a `direct-eval` stage to plan,
materialize, validate, run, or collect those branches. Collection must use the
same flag as materialization so control runs enter the canonical outputs.

The current public pointwise-control materializers cover the declared Qwen3-4B
task pipelines. They do not cover every additional paper replication, including
the Gemma-E4B response comparison and the extra 1.7B/32B response checks. The
Nectar setup is likewise not the manuscript's deduplicated 434-prompt cohort; see
`[DATA-SETUP.md](DATA-SETUP.md#response-ranking)`. Do not describe those cells
as clean-clone reproducible from this pipeline.

Reference scope is also task-specific. Jina rows retained in the QA/response
pipeline are repository-only diagnostics: the manuscript's external stability
comparison is passage-reranking only. Skywork and PairRM are response direct
quality references, not pair-flip/stability baselines, and must be excluded from
manuscript downstream aggregates.

Run stages in order, because each consumes the preceding declared output. Within
a stage, `plan`, `materialize` and `validate` are free and local; run them before
anything expensive.

```bash
uv run python scripts/study.py <task> <stage> plan
uv run python scripts/study.py <task> <stage> materialize
uv run python scripts/study.py <task> <stage> validate
uv run python scripts/study.py <task> <stage> execute --dry-run
```

For `training`, the generated sweep currently preserves each template's
trial-less `student.output_dir`. That is suitable for direct local experiments
but not for the collector, which requires
`runs/<generated-id>/<trial>/student/training_summary.json`. Until the local
runner and collector share one layout, run each generated training config with
the explicit output and checkpoint overrides shown in
[Stage 3](#stage-3-train-the-student). Do not use `training execute --run` for
runs that you intend to collect.

Silver collection takes one explicit teacher run for passage reranking. QA and
response ranking additionally collect width-1 teacher runs for the separately
trained pointwise controls:

```bash
uv run python scripts/study.py passage-reranking silver collect \
  --teacher-run runs/<teacher>/<trial>
uv run python scripts/study.py <task> silver collect \
  --training-teacher-run runs/<teacher-train>/<trial> \
  --heldout-teacher-run runs/<teacher-heldout>/<trial> \
  --pointwise-training-teacher-run runs/<pointwise-teacher-train>/<trial> \
  --pointwise-heldout-teacher-run runs/<pointwise-teacher-heldout>/<trial>
```

Training collection applies the declared λ and checkpoint selection rule and
writes `checkpoints.json`. Direct-evaluation collection writes per-run,
per-dataset, per-seed, and aggregate files. Passage and response downstream
stages are inference-free reductions over direct-evaluation outputs; QA
requires frozen readers.

`multi-document-qa downstream-eval execute --run` dispatches both generated
reader phases locally through `scripts/run_reader.py --config <generated.yaml>`:
R3 uses the answer sidecar, and V3 uses Climate-FEVER verdict labels and the
paired-qid manifest. Each generated config sets `phase` and `scorer.exp_id`;
when `--scorer-run` is omitted, the reader resolves that id under `runs/` to the
newest `psi/beta_gamma_scores.parquet` (see
[`configs/reader/README.md`](../configs/reader/README.md#scorer-run-resolution)).
Pin a trial with an explicit `--scorer-run` when latest-by-mtime is not the
intended scorer. The phase-aware container equivalent is in the same reader
README.

Collectors resolve explicit bindings from
`build/reproduction/evidence/source-bindings/<task>/<stage>.json`. Write that
file to bind a stage to exact run directories, or pass `--discover-latest` to
scan for the newest matching run.

### Readiness gates

```bash
uv run python scripts/study.py structural audit
uv run python scripts/study.py structural snapshot
```

The structural audit checks required tracked sources, expected job counts,
appendix declarations, and immutable model declarations. Its source-lock layer
uses `files_checked: false`: it verifies the declared file set, not the presence
or hashes of local data. Run `source_lock.py verify` separately only after the
complete non-internal population has been provisioned.

After setup has completed, the code-only checks below need no GPU, credentials,
network, or licensed data:

```bash
uv run --no-sync poe lint
uv run --no-sync poe smoke
uv run python scripts/setup_reproduction_data.py plan
uv run python scripts/study.py structural audit
uv run python scripts/study.py appendix audit
uv run python scripts/study.py representative validate
uv run python scripts/study.py passage-reranking silver plan
uv run python scripts/study.py multi-document-qa silver plan
uv run python scripts/study.py response-ranking silver plan
```

For a fully provisioned checkout, including gated/manual public inputs, add:

```bash
uv run python scripts/setup_reproduction_data.py validate
uv run python scripts/source_lock.py verify
```

`validate` additionally expects Legal-A/B and therefore remains nonzero in the
public artifact. The source lock excludes those two internal collections.

## Appendix artifacts

The appendix entries currently represented in the release are declared in
`configs/reproduction/appendix.yaml` and share execution programs from
`configs/reproduction/appendix-programs.yaml`. This is not yet a one-to-one
registry of all 31 numbered tables and 10 numbered figures in the manuscript.

```bash
uv run python scripts/study.py appendix audit
uv run python scripts/study.py appendix plan <artifact-id>
uv run python scripts/study.py appendix run <artifact-id>
```

`plan` prints the program workflow that produces an entry's inputs, plus the
analyzer command when one is declared. Entries without an analyzer stop at
their collected program outputs under `build/reproduction/`; the removed
reducer/renderer layer cannot be inferred from the declaration. Being declared
is not the same as being runnable or executed. The audit's `runnable`,
`blockers`, and `analyzer_inputs_present` fields are authoritative.
Eleven of the thirty-one declared artifacts currently name an analyzer.

Analyzers under `scripts/analyze/` read collected outputs from
`build/reproduction/<program>/` and write tables to
`build/reproduction/analysis/<analysis>/`. Sixteen of the twenty-two scripts are
registered in `appendix-programs.yaml`; the other six are operational tools for
λ screening, geometry backfill, readout validity, variance decomposition,
self-consistency derivation, and an HF-versus-vLLM speed probe. There is no
manuscript figure pipeline; analyzers write the numeric tables the manuscript
figures were drawn from. The optional `analysis` extra installs matplotlib for
ad-hoc plots only.

### Studies

Study definitions live under `configs/studies/` and expand from tracked
populations, task declarations, templates, and checkpoint catalogs. Generated
configs and sweeps are disposable.

```bash
uv run python scripts/materialize_appendix_study.py <condition>
uv run python scripts/collect_study_results.py <study> <condition>
```

Conditions include `placeholder-controls`, `instrument-width`, `multiseed-width`,
`fixed-weight-multiseed`, `context-decomposition`, `trained-channel-cross`,
`matched-variance`, `grade-one-control`, `round-robin-eval-grid`,
`complete-readout-grid`, `first-stage-panel`, and `first-stage-newsurf`.

Each materializer writes `materialization.json` and `run_manifest.json` under
`build/reproduction/studies/<study>/<condition>/`. Fill every `run_dir`
explicitly before collecting; collection rejects missing, extra, duplicate, or
blank mappings, hashes every consumed artifact, and writes `results.json`. It
never scans `runs/`.

Conditions that emit a sweep are materialized first and run through
`run_sweep.py`:

```bash
uv run python scripts/materialize_appendix_study.py round-robin-eval-grid
uv run python scripts/run_sweep.py \
  configs/sweeps/round-robin-eval-grid--canonical.yaml --dry-run
```

The fixed-weight context-decomposition condition regenerates all 18 passage
datasets from `reranking-primary-18` and the selected OC-SFT checkpoint catalog
rather than consuming historical runs. After its shards exist, the
context-decomposition merger validates overlap, coverage, and protocol
compatibility before the run manifest is pointed at the merged directories.

First-stage transfer needs six selected local adapters. Bind them without
editing the tracked study:

```bash
mkdir -p build/reproduction/first-stage-transfer
cp configs/studies/first-stage-transfer-checkpoints.example.yaml \
  build/reproduction/first-stage-transfer/checkpoints.yaml
# Edit only the copied file: replace each value with its selected adapter path.

uv run python scripts/materialize_appendix_study.py first-stage-panel
uv run python scripts/materialize_appendix_study.py first-stage-newsurf
```

Materialization verifies every adapter before writing generated configs and
records hashes in its provenance. It does not discover a latest trial. Fill both
generated manifests, since the newsurf manifest declares each bundle member and
every member needs its own `run_dir`, then merge the two conditions:

```bash
uv run python scripts/collect_study_results.py first-stage-transfer
```



### Representative validation

Seven analysis families are validated offline against one recorded value from a
tracked fixture. This checks the analysis implementations; it does not establish
that every appendix cell was re-executed.

```bash
uv run python scripts/study.py representative plan
uv run python scripts/study.py representative validate
uv run python scripts/study.py representative validate --family tau-psi
```

The families are `bootstrap-ci` (hierarchical percentile interval with an
excludes-zero gate), `tau-psi` and `verdict-flip` and `pair-flip` (means over
tracked dataset-level values), `jaccard` (mean pairwise Jaccard over retained
sets), `gain-histogram` (per-query gain bins, counting `delta >= 0` as improved),
and `calibration` (OLS endpoint calibration with CR1 model-clustered errors).
The declaration is `configs/reproduction/representative-analyses.yaml` and the
fixtures are the seven JSON files under `configs/reproduction/representative/`.

Results land under `build/reproduction/representative/`, each recording the
fixture SHA-256, source reference, paper label, operation, observed value,
expected value, tolerance, and content hash. Fixtures are reduced examples and
declare their evidence level, distinguishing values derived from a retained
analysis from constructed protocol examples. The same operations accept full
inputs without code changes. Do not loosen a tolerance to absorb an unexplained
mismatch.

## Run artifact layout

A base evaluation run:

```text
runs/<ID>/<timestamp>/
├── resolved_config.yaml
├── experiment.log
├── evaluation.log
├── per_query_results/<qid>/
│   ├── detailed_results.json
│   ├── trec_results_raw.txt
│   ├── trec_results_deduplicated.txt
│   └── eval_results.jsonl
├── all_queries_eval_results.jsonl
└── metrics.json
```

`run_psi.py` can share that directory, writing robustness output alongside the
identity-order base:

```text
runs/<ID>/<timestamp>/
├── metrics.json                 # identity-order base
├── per_query_results/<qid>/
└── psi/
    ├── psi_run.log
    ├── psi_metrics.json
    ├── psi_per_query.json
    └── per_query_results/<qid>/
        ├── input_positions.json
        └── permutation_<idx>_<label>/
            ├── detailed_results.json
            └── trec_results_raw.txt
```

`metrics.json` is the aggregate quality artifact and `psi/psi_metrics.json` the
aggregate robustness artifact, including protocol metadata and τ-PSI@B geometry
fields. A teacher run writes `runs/self-distill/<ID>/<timestamp>/silver/` with
`manifest.json` and `silver_labels.jsonl`, one continuous label per query and
document pair plus the raw K-shot vector behind it.

Generated output is rooted at `build/reproduction/`, with one directory per task
and stage, `studies/<study>/<condition>/`, and
`evidence/source-bindings/<task>/<stage>.json`.
`[RUN-ARTIFACTS.md](RUN-ARTIFACTS.md)` documents the pool-perturbation and
recovery layouts.

## Runtime expectations

Planning estimates from the recorded OG Qwen3-4B runs, on the hardware described
in `[HARDWARE.md](HARDWARE.md)`.


| Stage                                      | Wall-clock                                                                    |
| ------------------------------------------ | ----------------------------------------------------------------------------- |
| Data, 30K fixture                          | tens of minutes, one time                                                     |
| Silver, K=10 teacher                       | 10 to 17 h on one 8×A100 node; sharding roughly 30 ways reduces this to hours |
| Student SFT                                | about 7 h on one 8×A100 node                                                  |
| Evaluation and PSI, per dataset, both arms | DL19 0.5 to 1 h; the public battery 60 to 90 h                                |
| Record                                     | seconds                                                                       |


