# Data setup for manuscript reproduction

Map each declared reproduction population to the script that
materializes its ignored local `data/` directory.

Commands assume the working directory is the package root. The generated files are not committed. Dataset identity and paths are defined
by:

- `configs/reproduction/populations/reranking-primary-18.yaml`
- `configs/reproduction/populations/reranking-frozen-18.yaml`
- `configs/reproduction/populations/multi-document-qa-3.yaml`
- `configs/reproduction/populations/response-ranking-5.yaml`

## Prerequisites

- Python 3.11 and the project `uv` environment.
- `HF_TOKEN` for Hugging Face datasets/models where required.
- Java 21 and `JAVA_HOME` for Pyserini downloads.

Per-dataset terms are tabulated in
[../THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md). Confirm that your
planned use is permitted before downloading or processing third-party data.

Verify the local prerequisites:

```bash
uv run python --version
java -version
uv run python -c "import datasets, ir_datasets, pyserini; print('data imports ok')"
```



## Unified setup command

Preview every setup row. The plan always shows all 18 passage-reranking
population members, including manual, internal, and access-gated rows:

```bash
uv run python scripts/setup_reproduction_data.py plan
```

Run missing public automated steps:

```bash
uv run python scripts/setup_reproduction_data.py run
```

Access-gated Signal-1M, TREC-News, and Robust04 require an explicit opt-in after
the operator has obtained the applicable licenses:

```bash
uv run python scripts/setup_reproduction_data.py run \
  --task passage-reranking --include-gated
```

`--include-internal` marks internal rows as enabled in the plan, but Legal-A and
Legal-B remain manual: they are not distributed with this repository.

The `validate` command is a full-population gate, not a check of only the rows
that `run` can automate. It reports missing manual, gated, and internal rows and
returns nonzero until all of them are present:

```bash
uv run python scripts/setup_reproduction_data.py validate
```

The source lock is slightly narrower: it excludes Legal-A and Legal-B, but
requires every other declared public file, including manually supplied
DL21–DL23 and access-gated Signal-1M, TREC-News, and Robust04:

```bash
uv run python scripts/source_lock.py verify
```

Consequently neither command is the completion check for the default automated
setup. Use `plan` to inspect per-row readiness after an automated run; use
`validate` only for a fully provisioned population and `source_lock.py verify`
only for a fully provisioned non-internal population.

Use `--task passage-reranking`, `multi-document-qa`, or `response-ranking` to
limit the operation. The command reports access-gated and manual steps
explicitly. The sections below document the underlying commands.

## Passage-reranking training pool

Build the deterministic MS MARCO 30K BM25 top-100 candidate pool:

```bash
uv run python -m scripts.data.setup_msmarco_self_distill --query-count 30000
```

Expected root:

```text
data/msmarco-train-selfdistill-seed42/
```

The silver stage later writes or derives the K=10/K=1 labels and held-out split;
those are lifecycle outputs, not raw-data setup.

## Passage-reranking evaluation population



### Public Pyserini datasets

The unified planner derives one fetch command per automatable dataset from
`reranking-primary-18.yaml`, including each exact run filename:

```bash
uv run python scripts/setup_reproduction_data.py plan \
  --task passage-reranking
uv run python scripts/setup_reproduction_data.py run \
  --task passage-reranking
```

This covers DL19, DL20, and the public BEIR/TREC members of the canonical
population. DL20's Pyserini topic key, DBPedia's filename slug, and ArguAna's
`--remove-query` requirement are encoded in the adapter registry. The reusable
individual command is `scripts/data/fetch_pyserini_dataset.py`; pass
`--run-name` when an exact population filename is required.

The canonical population uses `PyseriniLoader` directly. Several standalone
examples instead use `FixtureLoader`, so after fetching DL19 build their
self-contained input explicitly:

```bash
uv run python scripts/data/build_fixture_pyserini.py \
  --topics data/dl19-passage/topics.tsv \
  --run data/dl19-passage/run.msmarco-v1-passage.bm25-default.dl19_sorted.txt \
  --index msmarco-v1-passage \
  --k 100 \
  --out data/dl19-passage/fixture.jsonl
```

`-e <config>` is the shorter form, but it reads `data.topics_tsv`,
`data.run_path` and `data.index` out of that config, so it works only for a
`PyseriniLoader` config such as `example-passage-mxbai-large-v2`. Pointing it
at a `FixtureLoader` config is refused: the fixture is what it would build.

That writes `data/dl19-passage/fixture.jsonl`. Fetching the Pyserini run alone
does not create it.

Evaluate prepared BEIR runs with the BM25-specific batch evaluator:

```bash
bash scripts/data/eval_beir_bm25.sh
```

It writes baseline runs under `bm25-beir-<slug>-baseline`.

### DL21, DL22, and DL23

These use index-free fixtures rather than a local Pyserini corpus:

```bash
uv run python scripts/data/build_fixture_irds.py --help
```

Each row remains manual because the official scoreddocs run, judged topics, and
qrels source paths are not declared safely. After obtaining those inputs, call
the builder with `msmarco-passage-v2/trec-dl-2021`, `...-2022`, or `...-2023`
and write the fixture/qrels paths declared in `reranking-primary-18.yaml`.

### Legal-A and Legal-B

These datasets are internal and must be materialized from authorized source
bundles at the roots declared by `reranking-primary-18.yaml`:

```text
data/legal-a/
data/legal-b/
```

Legal-A and Legal-B are internal collections: the canonical plan records precise manual rows and never invokes an automated setup command. Supply both `fixture.jsonl` and `qrels.txt` at each declared root. These fixtures embed
licensed passage text; do not redistribute them.

## Multi-document QA



### QA training and held-out silver pool

```bash
uv run python -m scripts.data.setup_hotpotqa_support_dataset \
  --split train --max-queries 30500
```

Expected root:

```text
data/hotpotqa-distractor-support-train/
```



### QA direct-evaluation datasets

```bash
uv run python -m scripts.data.setup_hotpotqa_support_dataset \
  --split dev --max-queries 200

uv run python -m scripts.data.setup_multihopqa_support_dataset \
  --dataset 2wiki --max-queries 200

uv run python -m scripts.data.setup_multihopqa_support_dataset \
  --dataset musique --max-queries 200
```

Expected roots:

```text
data/hotpotqa-distractor-support-dev/
data/2wiki-distractor-support-dev/
data/musique-support-dev/
```

Build answer sidecars for frozen-reader evaluation:

```bash
uv run python -m scripts.data.setup_qa_answers --dataset hotpotqa
uv run python -m scripts.data.setup_qa_answers --dataset 2wiki
uv run python -m scripts.data.setup_qa_answers --dataset musique
```

Climate-FEVER verdict evaluation reuses the passage-reranking Climate-FEVER
fixture and requires the sidecars produced by:

```bash
uv run python -m scripts.data.setup_verdict_data --dataset climate-fever
```

This writes `verdicts.jsonl` plus sorted `verdict_qids.txt`; the latter excludes
`DISPUTED` labels and is the exact cohort consumed by V3 validation.

## Response ranking



### Response-ranking training and held-out silver pool

```bash
uv run python -m scripts.data.setup_ultrafeedback_response_quality_dataset \
  --max-queries 30700
```

Expected root:

```text
data/ultrafeedback-response-quality-train/
```



### Response-ranking direct-evaluation datasets

```bash
uv run python -m scripts.data.setup_rewardbench2_response_quality_dataset
uv run python -m scripts.data.setup_nectar_response_quality_dataset \
  --revision 3c6b4c47fa1cc38869f9f32dce1699f7abad8b06 \
  --max-queries 500

uv run python -m scripts.data.setup_ppe_response_quality_dataset \
  --dataset lmarena-ai/PPE-MATH-Best-of-K \
  --out data/ppe-math-response-quality

uv run python -m scripts.data.setup_ppe_response_quality_dataset \
  --dataset lmarena-ai/PPE-MMLU-Pro-Best-of-K \
  --out data/ppe-mmlu-pro-response-quality

uv run python -m scripts.data.setup_rmbench_response_quality_dataset
```

The Nectar command materializes the 498-query source pool at the immutable revision pinned by the script. The used response-ranking population then selects the manuscript's 434 prompts after exact-prompt overlap removal against UltraFeedback, using the tracked `configs/reproduction/fixtures/nectar-dedup-clean-qids.txt` manifest.

Expected roots are the five paths in
`configs/reproduction/populations/response-ranking-5.yaml`.

## Validate local coverage

After setup, validate without launching jobs:

```bash
uv run python scripts/study.py passage-reranking direct-eval materialize
uv run python scripts/study.py passage-reranking direct-eval validate

uv run python scripts/study.py multi-document-qa direct-eval materialize
uv run python scripts/study.py multi-document-qa direct-eval validate

uv run python scripts/study.py response-ranking direct-eval materialize
uv run python scripts/study.py response-ranking direct-eval validate
```

Direct-evaluation materialization requires the preceding training checkpoint
catalog. Before training completes, use the silver-stage validation and inspect
the population YAML paths directly.

## Baseline sanity checks

These assume the relevant `data/` trees already exist. For the
zero-dependency install check (no data), use `uv run --no-sync poe smoke` as in
[`REPRODUCE.md`](REPRODUCE.md#zero-dependency-smoke).

### First real-data check (DL19, no model)

DL19 first-stage evaluation:

```bash
uv run python scripts/import_run_as_baseline.py \
  -r data/dl19-passage/run.msmarco-v1-passage.bm25-default.dl19_sorted.txt \
  -q data/dl19-passage/qrels.txt \
  -i bm25-dl19-baseline
uv run python scripts/run_eval.py -e bm25-dl19-baseline
```

Reference nDCG@10 is approximately `0.5058` over 43 judged queries.

Pipeline identity smoke:

```bash
uv run python scripts/run_experiment.py \
  -e configs/experiments/_smoke-identity.yaml
uv run python scripts/run_eval.py -e _smoke-identity
```

Identity should preserve DL19 top-100 nDCG@10/MRR. MAP may differ from a
top-1000 run because of candidate truncation.

Legal-A first-stage import, after authorized manual materialization:

```bash
uv run python scripts/import_run_as_baseline.py \
  -r data/legal-a/run.firststage.legal-a_sorted.txt \
  -q data/legal-a/qrels.txt \
  -i firststage-legal-a-baseline
uv run python scripts/run_eval.py -e firststage-legal-a-baseline
```

Licensed legal fixtures must not be redistributed.

## Fixtures for the container path

`PyseriniLoader` reads a prebuilt index directly, so a local canonical-population
run needs nothing further. A `FixtureLoader` example or a container job needs a
self-contained fixture built beforehand with
`scripts/data/build_fixture_pyserini.py`; the container images do not install
Pyserini.

## Operator tools

Most of `scripts/data/` is reached through the setup command above or through a
pipeline stage. Six scripts are not reachable that way and are run by hand when
you need them:


| script                                   | what it does                                                                                                                         |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `derive_label_pool_subset_silver.py`     | Cut a query-level subset out of a silver JSONL, which is how the label-pool sizes in the sample-efficiency study are made.           |
| `derive_negative_control_silver.py`      | Build random-label and shuffled-label negative controls that keep the student input pipeline and label schema intact.                |
| `build_recalibrated_silver.py`           | Quantile-match pointwise silver to the batched-teacher shape, per query.                                                             |
| `check_iterative_self_distill_silver.py` | Sanity-check silver from iterative self-distillation.                                                                                |
| `profile_prompt_lengths.py`              | Report the per-chunk prompt token-length distribution before committing to a `max_length`.                                           |
| `score_math_accuracy.py`                 | Exact-match accuracy for generated responses on PPE-MATH; the only consumer of the `math-verify` dependency in the `analysis` extra. |


Each takes `--help`, and each writes only to paths you name on the command line;
`profile_prompt_lengths.py` writes nothing at all.

## Related

- [MODELS.md](MODELS.md): model and reference examples.
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md): diagnosis and acceptance checks.
- [REPRODUCE.md](REPRODUCE.md#one-result-end-to-end): single-result
walkthrough.
- [REPRODUCE.md](REPRODUCE.md#the-full-task-matrix): reproduction stages after
data setup.

