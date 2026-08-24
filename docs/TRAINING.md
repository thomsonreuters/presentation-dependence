# Generating silver labels and training a student

One pipeline, 2 halves: A teacher scores a candidate pool under several presentations and writes continuous silver labels; a student then fits those labels with LoRA. For a reproducible student run, `student.base_model` must resolve
to the reviewed immutable snapshot; the current loader does not enforce a
separate revision field.

```text
materialize the candidate fixture
  -> run the teacher
  -> validate and publish the silver JSONL
  -> derive train, held-out, and single-order (`k1`) products
  -> train the LoRA student
  -> select lambda and checkpoint
  -> direct and downstream evaluation
```

Teacher configs live under `configs/silver/` and student configs under
`configs/self-distill/`; their field schemas are
`[configs/silver/_schema.md](../configs/silver/_schema.md)` and
`[configs/self-distill/_schema.md](../configs/self-distill/_schema.md)`. Both
run through the same two entry points:

```text
scripts/run_silver_generation.py
scripts/run_self_distill_sft.py
scripts/data/prepare_heldout_eval_cohort.py
scripts/derive_k1_silver_from_bsc.py
scripts/select_lambda.py
```



## Prerequisites

```bash
./setup.sh --full
uv run --no-sync poe smoke
uv run python -m scripts.data.setup_msmarco_self_distill --query-count 30000
```

The MS MARCO teacher fixture is `data/msmarco-train-selfdistill-seed42/`, and it
must contain the declared qid file, topics, qrels, first-stage run, and fixture.
`[DATA-SETUP.md](DATA-SETUP.md)` builds it.

`--full` does not install vLLM. For a tracked teacher that declares
`inference_engine: vllm`, add `--extra vllm` to its `uv run` command. Full
population validation and source-lock verification are evaluation-data gates,
not prerequisites for building the MS MARCO training pool.

## Teachers

Two protocols produce silver, and they differ in how a grade is read rather than
in what they emit.

### Open-weight teachers

Protocol `k_shot_bsc`.

The teacher reads fixed grade-token logits at known answer slots and computes
`E[g] = Σ g·P(g)`. It never decodes a grade string. The exact prompt, slot,
token, probability, scaling, and alignment behaviour is defined in
[`SCORING.md`](SCORING.md); `Path-C` is a legacy identifier listed in the README
[Legacy identifiers](../README.md#legacy-identifiers). The reranker emits `[0,1]` and
`KShotBSCTeacher` scales to `[0, grade_max]`.

For each query and candidate set:

1. Create `T` Fisher-Yates teacher permutations from the declared seeds.
2. Score each order through the reranker's B-document expected-grade path.
3. Realign scores to the original document IDs.
4. Persist each document's `T`-vector and its mean.
5. Write one per-query shard atomically.
6. Publish the aggregate JSONL and manifest only after every query succeeds.

```yaml
teacher:
  protocol: k_shot_bsc
  k_perms: 10
  seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  prompt_template_id: grade_int_v1
  grade_max: 3
  cross_query_batch: 4
  output_subdir: silver
```

The MS MARCO production geometry is paper `T=10` (stored as
`teacher.k_perms: 10`), `docs_per_score_forward=20`, `k_input=100`,
`max_doc_chars=1200`, `max_length=4096`, and teacher-permutation seeds `0..9`.
Changing any of these changes the scientific protocol; record an intentional
change as a new semantic config rather than an override.

Tracked examples are `a1-qwen3-instruct-k10-bsc-msmarco-30k.yaml`,
`a2-qwen3-reranker-k10-bsc-msmarco-30k.yaml`, and
`qwen3-4b-nonthink-k10-bsc-msmarco-30k.yaml` under `configs/silver/`.

```bash
uv run --extra vllm python scripts/run_silver_generation.py --config <teacher-id>
```

Output lands under `runs/self-distill/<teacher-id>/<trial>/silver/`.

`reranker.inference_engine` selects `hf` or `vllm` for the
supported Qwen3, Gemma-4, and Granite wrappers. The class default is `hf` for
best-effort CPU and MPS development; CUDA teacher configs may select `vllm`.
vLLM is valid only for supported continuous expected-grade paths, and config
loading rejects an unsupported combination rather than falling back quietly.
The engines under
`src/presentation_dependence/self_distill/engines/` lazy-import vLLM, keep HF and vLLM model
ownership mutually exclusive, and set `VLLM_WORKER_MULTIPROC_METHOD=spawn`
before the import to avoid CUDA fork failures. Model families do not necessarily
share a viable image; see `[HARDWARE.md](HARDWARE.md)` for the CUDA and vLLM
compatibility matrix. Measure before changing a default:

```bash
uv run python scripts/analyze/probe_vllm_speedup.py --help
```

With `vllm` and `teacher.cross_query_batch: N`, the driver batches N queries per presentation seed, shuffles every candidate set,
calls `rank_query_batch`, and writes the completed per-query records atomically.
Start at 1 and raise to 4 or 8 only after a smoke run on the target hardware. A
mid-batch failure loses only the in-flight batch. The selected value is recorded
in the silver manifest.

### Hosted teachers

Protocol `closed_model_generated_bsc`.

`SilverGenerator` builds pointwise requests and hands them to a `SilverClient`,
a one-method contract that scores a list of requests and persists the raw
provider replies. Grouping by query happens before submission, so a batched
client sees one query's whole candidate set at once.

`OpenAICompatibleClient` talks to any endpoint speaking the OpenAI
chat-completions API: a hosted service, a gateway, or a local vLLM server. It
uses the standard library only, so no provider SDK enters the dependency set.

```bash
export OPENAI_API_KEY=...            # required
export OPENAI_BASE_URL=...           # only for a non-default endpoint
```

It builds the grade prompt and answer skeleton from
`presentation_dependence.rerankers.grade_rubrics` rather than a private template, which is
what keeps this row comparable to the open-weight scorers: same instruction,
same rubric, same `[<id>] Grade: <0|1|2|3>` readout.

```yaml
teacher:
  protocol: closed_model_generated_bsc
  model_id: gpt-5.4
  client: openai
  endpoint:
    base_url: null          # null uses the provider default
  scoring:
    subset_size: 20
    runs: 10
    run_seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    score_min: 0
    score_max: 3
```

One prompt carries `subset_size` candidates, the pool is scored under `runs`
shuffled presentations, and the per-candidate grades are averaged. That is the
closed-model analogue of `T=10` teacher permutations, and `subset_size: 20` matches the
B=20 chunk size of the open-weight configs. Each `(query_id, doc_id)` gets one
row keeping the per-presentation vector alongside the mean.

Set `runs: 1` when the evaluation runner owns presentation diversity, which is
what the evaluation configs do: a single run scores the identity order without
shuffling. Raising `runs` there reshuffles inside the client and breaks parity
with open-weight PSI.

A generated response can omit one candidate's grade. The hosted client retains
the scores it did parse, so publication runs must explicitly require every
document's raw vector length to equal `teacher.scoring.runs`; do not treat a
nonempty shorter vector as complete. The open-weight expected-grade path has a
different contract and requires one aligned score/vector per document and
presentation.

This closed-model path is a generated-score baseline, not the expected-grade
readout. A chat endpoint cannot reproduce `E[grade]` unless it exposes token
logprobs at the grade positions, so the pipeline persists the generated per-run
vector instead and labels the scoring path accordingly.

## The silver-label contract

One JSONL row per query and document pair:

```json
{
  "query_id": "q1",
  "doc_id": "d1",
  "score_continuous": 2.4,
  "score_raw_vector": [3.0, 2.0, 2.2],
  "teacher_model_id": "Qwen/Qwen3-4B",
  "teacher_protocol": "k_shot_bsc",
  "prompt_template_id": "grade_int_v1",
  "k_perms": 3,
  "timestamp": "2026-05-07T20:16:34Z"
}
```

`score_continuous` lies in `[0, grade_max]`, `score_raw_vector` retains the K
presentation scores, and the query and document IDs match the declared fixture.
Model, protocol, prompt, K, seeds, and counts are recorded in the manifest.

## Validate a teacher before scaling

Reduce a tracked config rather than writing a synthetic one:

```bash
uv run python scripts/run_silver_generation.py \
  --config qwen3-4b-nonthink-k10-bsc-msmarco-30k \
  --override 'qids_to_run=[1037798]' \
  --override teacher.k_perms=2 \
  --override 'teacher.seeds=[0, 1]' \
  --override reranker.docs_per_score_forward=4 \
  --override reranker.batch_size=4 \
  --override reranker.max_length=1024
```

Confirm that continuous scores lie within `[0, grade_max]`, every vector has
length K, query and document IDs are unique, manifest counts equal the JSONL
record count, and at least some documents vary across permutations.

Then check ranking quality against a judged set. Build a fixture from BM25
top-100 over the 43 NIST-judged DL19 queries, retarget the tracked open-weight
teacher with overrides (there is no separate DL19 silver YAML), generate silver,
and score it:

```bash
uv run python scripts/data/build_silver_fixture.py \
  --out data/silver-fixtures/trec-dl19-bm25-top100.jsonl \
  --candidates-per-query 100 \
  --trec-run data/dl19-passage/run.msmarco-v1-passage.bm25-default.dl19_sorted.txt
uv run --extra vllm python scripts/run_silver_generation.py \
  --config qwen3-4b-nonthink-k10-bsc-msmarco-30k \
  --override data.run_path=data/silver-fixtures/trec-dl19-bm25-top100.jsonl \
  --override 'qids_to_run_path=null' \
  --override eval.qrels_path=data/dl19-passage/qrels.txt
uv run python scripts/data/eval_silver_ndcg.py \
  --run-dir runs/self-distill/qwen3-4b-nonthink-k10-bsc-msmarco-30k/<trial>/silver \
  --fixture data/silver-fixtures/trec-dl19-bm25-top100.jsonl \
  --qrels data/dl19-passage/qrels.txt
```

Clearing `qids_to_run_path` is required: the tracked config otherwise filters to
the MS MARCO 30K allowlist, which does not intersect DL19. With an empty
allowlist the teacher scores every query in the fixture.

`eval_silver_ndcg.py --run-dir` names the directory that directly contains
`silver_labels.jsonl`. Open-weight runs use
`runs/self-distill/<id>/<trial>/silver/`; hosted runs use
`runs/silver/<id>/<trial>/`.

Acceptance requires nDCG@10 above the BM25 baseline of about 0.506 on DL19. The
NIST oracle rerank at about 0.892 is the upper bound. Do not infer production
time from a one-query smoke.

## Resume and durability

Teacher output is atomic per query, so resume into the same run directory:

```bash
uv run python scripts/run_silver_generation.py \
  --config <teacher-id> \
  --run-dir runs/self-distill/<teacher-id>/<trial>/
```

Completed qids are skipped and the aggregate is regenerated deterministically. A
partial run does not publish aggregate files, and its per-query shards stay
resumable.

Resume does not compare a complete config/input fingerprint. The open-weight
path skips an existing qid shard by presence; the hosted key covers query,
document, prompt-template ID, and teacher-model ID, but not every scoring
parameter, endpoint identity, source-text hash, or candidate-pool hash. Resume
only with the byte-equivalent resolved config and inputs. If any scientific
setting or source changed, start a new run; do not rely on the existing key to
detect the mismatch.

As a partial backstop, the open-weight path compares the teacher model, protocol,
prompt template, K, seeds, and grade range against the previous run's
`silver/manifest.json` before reusing shards, and warns when they differ. It is
a warning, not a refusal, and it cannot help after an interrupted run, because
the manifest is only written on completion. Treat it as a way of catching an
obvious mistake, not as a guarantee.

The hosted generator adds four recovery properties:

1. Per-record durability. Each row is written with `f.flush()` and
  `os.fsync(f.fileno())` before the next request is submitted.
2. A resume key of `(query_id, doc_id, prompt_template_id, teacher_model_id)`.
  On startup the generator reads the labels file into a set of completed keys
   and skips those work items.
3. A run-directory lock. An advisory `fcntl.flock` on `<run_dir>/.silver.lock`
  prevents two concurrent writers and holds the writer's PID, so a later
   process can recover stale metadata once the holder is gone.
4. Cost state rebuilt from the labels rather than a sidecar. A crash mid-flush
  can leave `cost_report.json` out of sync, so resume reconstructs
   `n_calls_total`, `per_model_cost_usd`, and `total_cost_usd` by re-reading
   `silver_labels.jsonl`.

`run_silver_generation.py` auto-detects the most recent run directory for an
experiment id; pass `--new-run` to force a fresh one.

For a multi-hour run that must survive terminal closure, wrap the same entry
point rather than adding a script:

```bash
mkdir -p runs/silver/_launcher_logs
config=<config_id>
log="runs/silver/_launcher_logs/${config}_$(date +%Y%m%d_%H%M%S).log"

# Portable Linux/other Unix:
nohup uv run python scripts/run_silver_generation.py --config "$config" \
  >"$log" 2>&1 &

# On macOS, use this instead when preventing idle sleep is required:
# nohup caffeinate -dimsu uv run python scripts/run_silver_generation.py \
#   --config "$config" >"$log" 2>&1 &

echo "$!" >"runs/silver/_launcher_logs/${config}.pid"
disown
```

`caffeinate` is macOS-only; it is normally absent on Linux. Neither command
restarts Python after a crash; re-run the same entry point and let
the generator resume. Do not start a second writer while the recorded PID is
alive.

## Building the candidate pool

The canonical passage-training pool keeps the full BM25 top 100:

```bash
uv run python -m scripts.data.setup_msmarco_self_distill --query-count 30000
```

It deterministically samples MS MARCO train queries with seed 42 and writes
`data/msmarco-train-selfdistill-seed42/{fixture.jsonl,qrels.txt,topics.tsv, qids_30k.txt}` plus the top-100 run. The 1K/30K/100K cohorts are prefix-stable.

`scripts/data/build_msmarco_train_silver_pool.py` is a separate stratified
20-document exploratory builder. Its `fixture.<n>x<kept>.jsonl` output is not
the primary paper pool and is incompatible with configs expecting the canonical
100-candidate `fixture.jsonl`.

`FixtureLoader` consumes one JSON object per line:

```json
{
  "qid": "467040",
  "query": "number one girl scout cookie",
  "passages": [
    {"pid": "8752317", "text": "...", "bm25_rank": 1},
    {"pid": "8200339", "text": "...", "bm25_rank": 2}
  ]
}
```

`bm25_rank` is informational. NIST grades are attached only when a fixture is
built from a judged set, so MS MARCO train fixtures have none. The teacher
refuses to publish when a declared qid is absent from the fixture.

## Prompt templates

`src/presentation_dependence/silver_data/prompts/templates.py` registers three templates.
Current hosted configs use `grade_int_v1`, a 0 to 3 pointwise rubric with
integer output. `pathc_grade_v1` renders the expected-grade readout for the
closed-model reranker, taking its system prompt and rubric from the reranker
config. `grade_float_v1` emits a continuous 0.000 to 1.000 score and is not
enabled in any config, because the parser validates integers and the raw
completion mode it needs is not implemented.

In a DL19 dry run the integer prompt produced about 63% grade 1, 20% grade 3,
10% grade 0, and 7% grade 2, a distribution bimodal toward 1 and 3.

## Prepare the student products

Passage reproduction derives the held-out split and the single-order (`k1`) labels
deterministically:

```bash
uv run python scripts/data/prepare_heldout_eval_cohort.py \
  --silver-in data/msmarco-train-selfdistill-seed42/silver_labels_qwen3_4b_k10.jsonl \
  --n 500

uv run python scripts/derive_k1_silver_from_bsc.py \
  --silver-in data/msmarco-train-selfdistill-seed42/silver_labels_qwen3_4b_k10_train_29500.jsonl \
  --k-out 1
uv run python scripts/derive_k1_silver_from_bsc.py \
  --silver-in data/msmarco-train-selfdistill-seed42/silver_labels_qwen3_4b_k10_heldout_500.jsonl \
  --k-out 1
```

For the declared three-task workflow, collect through the study interface
instead. Passage takes one teacher run. QA and response ranking take separate
training/held-out batched-teacher runs plus the corresponding B=1 pointwise
teacher runs:

```bash
uv run python scripts/study.py passage-reranking silver collect \
  --teacher-run runs/<teacher>/<trial>

uv run python scripts/study.py <multi-document-qa|response-ranking> silver collect \
  --training-teacher-run runs/<teacher-train>/<trial> \
  --heldout-teacher-run runs/<teacher-heldout>/<trial> \
  --pointwise-training-teacher-run runs/<pointwise-train>/<trial> \
  --pointwise-heldout-teacher-run runs/<pointwise-heldout>/<trial>
```

To freeze a partial hosted run into a clean corpus,
`scripts/data/finalize_silver_run.py` writes `silver_labels.clean.jsonl`
alongside `silver_labels.partial.jsonl`, coverage and score-distribution
metrics, and a `.frozen` marker. It does not mutate `silver_labels.jsonl`, which
remains the resume input. Only fully covered queries reach the clean file, and
that is the one training should consume.

## Train the student

```bash
uv run python scripts/run_self_distill_sft.py -e <student-id>
```

Students consume continuous silver labels and train a LoRA adapter. The current
student loader accepts `student.base_model` but has no separate `revision`
field; a Hub model name therefore follows that repository's current default.
For a citable run, prefetch the reviewed commit and set `student.base_model` to
that immutable local snapshot path. Recording a revision elsewhere in the YAML
does not make the student loader enforce it. The base snapshot, rubric,
tokenizer, checkpoint selection rule, and data split are all part of the
experiment identity.

The trainer materializes deterministic views under `student/data_views/`: scalar
regression examples, grouped preference or listwise examples when requested, and
presentation views for consistency or augmentation objectives. Declare the view
seeds; do not regenerate views under an unrecorded seed.

The default objective is scalar expected-grade MSE:

```yaml
student:
  objective:
    type: supervised_mse
```

OC-SFT adds consistency across supervised presentation views while keeping the
regression target:

```yaml
student:
  objective:
    type: supervised_consistency
    lambda: 5.0
    view_seeds: [0, 1]
    lambda_warmup: {steps: 500, init: 0.0, schedule: linear}
```

Select lambda on the held-out split under the rule in
`[EVAL-PROTOCOL.md](EVAL-PROTOCOL.md)`, never from final test quality. Position
augmentation and DebiasFirst controls are also available as declared objectives.

Control memory through tracked config values rather than ad-hoc overrides:
`slot_forward_batch_size`, `gradient_checkpointing`, the attention
implementation, the LoRA target regex, precision, and the DDP settings.
Tracked Qwen and Granite SFT recipes set
`attn_implementation=flash_attention_2`, but `flash-attn` is not a locked
dependency: install it separately for CUDA if you want that path, or override
to `sdpa` (Gemma recipes already do). Omitting the key leaves the Transformers
default. Changing serving width, rubric, truncation, or the expected-grade
readout to avoid an OOM changes the method and needs a new config.

Single-node DDP runs one process per GPU and shards student examples by rank,
with rank 0 as the only writer. It is requested by `execution.distributed: ddp`
and takes effect whenever more than one GPU is visible; the runner re-execs
itself under torchrun, so there is nothing to launch by hand. See
`[HARDWARE.md](HARDWARE.md#using-more-than-one-gpu)`. Gradient accumulation may
use `no_sync` between synchronization steps, and resume requires a compatible
model, optimizer, data, objective, and geometry.

The loop writes `student/progress.jsonl`, TensorBoard summaries,
`student/training_summary.json`, and synchronized checkpoints, recording
optimizer step, train and eval loss, held-out quality, learning rate, throughput,
GPU memory, and checkpoint path.

The plain local runner uses `student.output_dir` exactly as written. Most source
templates therefore write to a trial-less
`runs/self-distill/<student-id>/student/`, with checkpoints below that directory.
The reproduction collector instead requires a generated ID and trial:

```bash
ID=<generated-training-config-id>
TRIAL=$(date -u +%Y%m%d_%H%M%S)
uv run python scripts/run_self_distill_sft.py -e <generated-config.yaml> \
  --override student.output_dir="runs/$ID/$TRIAL/student" \
  --override student.checkpoint.dir="checkpoints/$ID/$TRIAL/student"

uv run --extra training tensorboard --logdir runs/
test -f "runs/$ID/$TRIAL/student/training_summary.json"
```

Container entrypoints create a trial-shaped run tree automatically. For local
collection, the explicit overrides above are required until the local runner
and collector share one layout.

## Select and hand off

```bash
uv run python scripts/study.py <task> training collect \
  --runs-root runs --discover-latest
uv run python scripts/study.py <task> direct-eval materialize
uv run python scripts/study.py <task> direct-eval validate
uv run python scripts/study.py <task> direct-eval execute --dry-run
```

Selection applies the declared held-out metric and cadence, the task's lambda
rule first for OC-SFT, and the declared tie-break, then writes the checkpoint
catalog that direct evaluation materializes from.
`[MODELS.md](MODELS.md#trained-students)` covers evaluating the selected adapter, and
`scripts/record_result.py` records the run you select into
`docs/results_index.yaml`, which ships empty.

## Output layout

```text
runs/self-distill/<teacher-id>/<trial>/
├── resolved_config.yaml
├── teacher_run.log
└── silver/
    ├── per_qid/
    ├── silver_labels.jsonl
    └── manifest.json

runs/<generated-student-id>/<trial>/student/
├── checkpoint-final/
├── data_views/
├── progress.jsonl
├── resolved_student_config.json
└── training_summary.json

checkpoints/<generated-student-id>/<trial>/student/
├── checkpoint-step-XXXXXX/
└── latest_checkpoint.json
```

`[RUN-ARTIFACTS.md](RUN-ARTIFACTS.md)` covers the shared artifact rules.

## Common failures

A missing fixture or qid means the declared data setup did not complete; rerun
it rather than weakening validation. A partial teacher resumes into the same run
directory. Constant scores point at the expected-grade tokens, the prompt, the
scale, or backend parity; for a student, check lambda warmup, silver scale, view
alignment, and the checkpoint curves, and do not compensate by selecting against
test data. A missing candidate during collection means the training run is not
where the collector is looking, and a partial checkpoint catalog must not pass
as complete. An adapter and base mismatch shows up as differing base revision,
LoRA target modules, rank, tokenizer, or architecture. vLLM may reject an
adapter on architecture or rank grounds, or require merged weights.

`[TROUBLESHOOTING.md](TROUBLESHOOTING.md)` has a few recovery procedures.

