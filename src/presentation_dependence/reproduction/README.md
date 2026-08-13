# `reproduction/`

Turns tracked declarations into runnable experiment configs, launches them,
reads results back, and audits that the declarations are complete. Metrics,
rerankers, and training loops live elsewhere; this package imports them.

Operator CLI: `scripts/study.py` (wrapper around [`runner.py`](runner.py)).
Walkthrough: [`docs/REPRODUCE.md`](../../../docs/REPRODUCE.md).
The declarations this package reads are the task pipelines in
`configs/reproduction/`, documented in
[`_schema.md`](../../../configs/reproduction/_schema.md).

## The CLI

```text
study.py <task> <stage> <command> [options]
study.py <family> <command> [options]
```

- Tasks: `passage-reranking` | `multi-document-qa` | `response-ranking`
- Stages: `silver` → `training` → `direct-eval` → `downstream-eval`
- Families (non-stage): `structural` | `appendix` | `representative`

## What each command costs

Only `execute --run` uses a GPU. Everything else is offline.

| Command | Reads | Writes | Cost |
|---|---|---|---|
| `plan` | declarations | nothing | offline |
| `materialize` | declarations | generated configs, sweeps, and receipts under `build/reproduction/` | offline |
| `validate` | generated configs, declared data | nothing | offline |
| `execute --dry-run` | generated sweep | nothing | offline; prints the commands |
| `execute --run` | generated sweep | `runs/` | runs every cell locally |
| `collect` | `runs/` | `build/reproduction/<task>/<stage>/` | offline |
| `summarize` | collected output | nothing | offline |

`execute` requires exactly one of `--dry-run` or `--run`; there is no default.

## Stage prerequisites

Stages are ordered. Each needs the previous stage's collected output:

- `training materialize` needs silver products in the fixture channel.
- `direct-eval plan` needs `build/reproduction/<task>/training/checkpoints.json`,
  written by `training collect`. Without it, every direct-eval and
  downstream-eval command fails with `Missing checkpoint catalog`.
- `downstream-eval` needs collected direct-eval output. For passage and
  response ranking that is an offline reduction over existing score logs, so
  `execute` launches nothing. Multi-document QA launches frozen readers:
  `execute --run` materializes the generated R3/V3 configs and runs
  `scripts/run_reader.py --config <generated.yaml>` for each. Scorer resolution
  (latest `psi/beta_gamma_scores.parquet` under `scorer.exp_id`, or an explicit
  `--scorer-run`) is documented in
  [`configs/reader/README.md`](../../../configs/reader/README.md#scorer-run-resolution).
  Default `validate` does not check that those score logs exist; set
  `check_scorer_runs` in code or confirm the direct-eval PSI runs wrote
  `beta_gamma` before execute.

Optional direct-eval controls use `--include-controls` for all four lifecycle
verbs: materialize, validate, execute, and collect. The collection flag is what
places their runs in `per_dataset.json`, `per_seed.json`, and `aggregate.json`.

In a source-only checkout with no `runs/` and no `build/`, `silver` and
`training` plan and materialize; `direct-eval` and `downstream-eval` stop at
`Missing checkpoint catalog` until training has run, and `collect` cannot run at
all. That is expected.

## A worked path

Full stage-by-stage walkthrough (runtimes, expected metrics):
[`docs/REPRODUCE.md`](../../../docs/REPRODUCE.md).
Below is the same shape, with real output from this checkout.

1. Inspect the CLI.

```bash
uv run python scripts/study.py --help
```

2. Inspect a stage without writing. `plan` answers what the stage would do
and is the cheapest check after a declaration edit:

```bash
uv run python scripts/study.py passage-reranking training plan
```

```json
{
  "job_count": 24,
  "variants": ["k1-sft", "k10-sft", "oc-sft"],
  "training_seeds": [42, 43, 44],
  "checkpoint_selection": {
    "rule": "heldout-argmax",
    "decision_cell": {"family": "Qwen3-4B-OG (non-thinking)", "k": "K=1"},
    "decisions": "configs/reproduction/evidence/lambda-selection/decisions.json"
  },
  "jobs": [
    {
      "id": "passage-reranking-training--k1-sft--seed42",
      "seed": 42,
      "silver": "k1",
      "template": "configs/self-distill/qwen3-4b-nonthink-sft-msmarco-30k-k1-labels.yaml",
      "variant": "k1-sft"
    }
  ]
}
```

The 24 jobs are 3 for `k1-sft` and 3 for `k10-sft` (one per training seed), plus
18 for `oc-sft` (three seeds across the six-point lambda grid). Lambda is
selected on held-out data, not assumed. `structural audit` freezes that count;
a declaration edit that changes it fails the audit.

3. Generate the configs. `materialize` writes experiment YAMLs, the sweep, and
its receipt under `build/reproduction/<task>/training/`; generated files are
disposable and do not enter the tracked `configs/experiments/` namespace:

```bash
uv run python scripts/study.py passage-reranking training materialize
uv run python scripts/study.py passage-reranking training validate
```

`validate` recomputes the expected job count, compares it to disk, and checks
that every silver file named by a config exists.

The generated training sweep preserves each template's trial-less local
`student.output_dir`. As a result, `training execute --run` does not currently
write the trial layout consumed by `training collect`. Run generated configs
individually with the output/checkpoint overrides in
[`docs/REPRODUCE.md`](../../../docs/REPRODUCE.md#stage-3-train-the-student) when
the outputs will be collected.

4. Preview the paid step before spending.

```bash
uv run python scripts/study.py passage-reranking training execute --dry-run
```

Prints the command it would run. Swap `--dry-run` for `--run` only when you
mean it.

5. Collect after the runs land.

```bash
uv run python scripts/study.py passage-reranking training collect \
  --runs-root runs --discover-latest
```

Writes `build/reproduction/passage-reranking/training/checkpoints.json`, which
later stages read. Resolves lambda from the recorded decision and fails if it
cannot.

6. Appendix artifacts report missing inputs. With no collected outputs,
`plan` names the missing input instead of failing silently:

```bash
uv run python scripts/study.py appendix plan table-readout
```

```json
{
  "runnable": false,
  "blockers": [
    "declared reproduction program has not produced analyzer input: build/reproduction/studies/setup-readout-silver/readout/series.json"
  ],
  "workflow": ["materialize", "materialize", "analyze", "analyze", "analyze"]
}
```

`workflow` is the program that produces that input, so the blocker says what to
run next. `appendix plan` exits 2 when an artifact is not runnable (usable as a
script check). Stage `plan` commands always exit 0.

7. Check the checkout itself.

```bash
uv run python scripts/study.py structural audit
```

```json
{
  "status": "complete",
  "layers": {
    "git_tracking": {"status": "pass"},
    "compact_recipes": {"status": "pass"},
    "source_closure": {"status": "pass"},
    "source_lock": {"status": "pass"}
  }
}
```

Exit 2 means incomplete. The most common cause is a new file that a plan reaches
but git does not track yet (`untracked_required`).

The structural source-lock layer checks declarations with `files_checked:
false`; it does not hash local data. Run `scripts/source_lock.py verify`
separately after every non-internal population file has been provisioned.

## Collection binds to exact runs

Collectors do not guess which run backs a job. They read explicit bindings from
`build/reproduction/evidence/source-bindings/<task>/<stage>.json`, or you pass
`--discover-latest` to scan for the newest matching run.

Silver `collect` arguments differ by task: passage reranking needs
`--teacher-run`; QA and response ranking need `--training-teacher-run` and
`--heldout-teacher-run`, plus the two corresponding
`--pointwise-*-teacher-run` inputs when their declared B=1 controls are
collected.

## OC-SFT lambda comes from recorded decisions

`training collect` resolves lambda in this order: an explicit
`lambda_selection.json` receipt; the pipeline's `decision_cell` against the
recorded decisions; then a declared `expected_lambda`. With none of those, it
fails.

It does not recompute. The one-standard-error band and collapse rules in
[`docs/EVAL-PROTOCOL.md`](../../../docs/EVAL-PROTOCOL.md) need per-lambda score
deviations that live in run artifacts. A plain held-out argmax is not a safe
substitute: historical replay found repeated disagreements, with the plain
argmax choosing a lower lambda because it lacks the 1-SE tie-break.

## Editing this package

### Generated output matched a byte-for-byte baseline

This distribution ships without the development snapshot suite. Historical
development tests hashed generated study outputs against a committed baseline,
but that baseline does not ship and its old condition/file counts are not a
release contract. Materialization reads a checkpoint catalog a source-only
checkout does not have, so verify a full current expansion before and after any
generator change.

If you change generation here, compare a full materialization before and after
by hand; there is no digest in this tree to catch a drift for you.

### Some counts are frozen deliberately

The structural audit freezes the primary training counts declared by the task
pipelines. Study condition/cell counts come from the current tracked
declarations and may change when the manuscript design changes; record such a
change as a scientific decision and inspect the regenerated materialization
receipt rather than copying a historical number from prose.

### Adding things

- **A study condition**: declare it under `configs/studies/<study>.yaml`, add it
  to `MATERIALIZERS` in `study_materialization.py`.
  Study ownership comes from the declarations; nothing needs a hardcoded study
  id. Reuse `_emit_cell` unless the condition needs bespoke bundling.
- **A study file**: add it to `STUDY_FILES` (what `_study_for_condition` scans).
- **A dataset population**: add `configs/reproduction/populations/<name>.yaml`
  and reference it from a pipeline's `direct_eval.population` or a condition.
- **A stage or verb**: dispatch lives in `runner.py`. Handlers differ on
  purpose (distinct error types, per-task collect arguments, QA's execute path,
  downstream's offline reduction). Preserve those differences; do not unify them
  away.

## Module map

**Entry point and plumbing**

- [`runner.py`](runner.py): task/stage CLI and command families.
- [`pipelines.py`](pipelines.py): load a task pipeline and apply shared
  infrastructure (`load_pipeline` and the three task loaders).
- [`errors.py`](errors.py): `ReproductionError` plus per-stage subclasses.
  `runner` catches these, prints one stderr line, and exits 2.
- [`common.py`](common.py): shared stage I/O (`sha256_file`, `tracked_files`,
  `prepare_generated_directory`, `write_silver_receipts`, `silver_block`).

**Data and stages**

- [`data_setup.py`](data_setup.py): canonical population data materialization.
- [`silver.py`](silver.py): passage-reranking silver, all four verbs.
- [`cohort_silver.py`](cohort_silver.py): QA and response-ranking silver via
  two-cohort derivation.
- [`training.py`](training.py): training jobs; `collect` writes the checkpoint
  catalog and resolves lambda.
- [`direct_eval.py`](direct_eval.py): direct-eval configs (no collect).
- [`direct_results.py`](direct_results.py): collect and aggregate direct-eval
  metrics.

**Downstream evaluation**

- [`downstream_common.py`](downstream_common.py): shared aggregation and output
  writers.
- [`passage_downstream.py`](passage_downstream.py): retained-set reduction.
- [`qa_downstream.py`](qa_downstream.py): frozen-reader downstream; its sweep
  schema differs from the others (`launcher` and `configs`). Local
  `execute --run` is wired in [`runner.py`](runner.py) to
  `scripts/run_reader.py --config …` for both R3 and V3.
- [`response_downstream.py`](response_downstream.py): response-selection
  reductions.
- [`reduction.py`](reduction.py): inference-free reduction plumbing.

**Studies and analysis inputs**

- [`study_materialization.py`](study_materialization.py): expands declared study
  conditions into configs, bundles, and sweeps. Study ownership and protocol
  settings come from `configs/studies/*.yaml`; bespoke bundling remains
  explicit where required.
- [`study_results.py`](study_results.py): collect explicit `run_manifest.json`
  mappings. Never scans `runs/`.
- [`mechanism_inputs.py`](mechanism_inputs.py): mechanism inputs from exact
  run-manifest trials.
- [`analysis_inputs.py`](analysis_inputs.py): helpers for analyzer scripts
  (`load_per_consumer`, `summarize_seeded_metric`).

**Provenance and audits**

- [`source_lock.py`](source_lock.py): build and verify the model revisions and
  local dataset hashes declared by the four high-level pipelines. It does not
  pin container images or the complete breadth-model/training-input set.
- [`source_bindings.py`](source_bindings.py): resolve explicit run bindings.
- [`structural.py`](structural.py): check that tracked source can rebuild the
  plans. Four layers: `git_tracking`, `compact_recipes`, `source_closure`,
  `source_lock`. Exits 2 when incomplete. Does not verify that anything ran, or
  that a rerun reproduces published numbers.
- [`appendix.py`](appendix.py): plan and run the currently registered appendix
  entries. An entry without an analyzer stops at collected program output, and
  declaration alone does not make it runnable. Inspect the audit's `blockers`
  and `runnable` fields instead of relying on a static count in prose.

[`__init__.py`](__init__.py) re-exports plan/materialize/validate/collect for
the three tasks. Only tests use it; modules import each other directly.
