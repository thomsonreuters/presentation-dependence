# Run artifacts

On-disk layout of everything a run writes. Base evaluation and PSI can share one
run directory even though `ExperimentManager` and `PsiExperimentRunner` are
separate drivers.

A flattened `permutation_<k>/` layout is not implemented.

## Base evaluation run

Local command:

```bash
uv run python scripts/run_experiment.py -e configs/experiments/<ID>.yaml
uv run python scripts/run_eval.py -e <ID>
```

Current layout:

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

`metrics.json` is the aggregate quality artifact for nDCG, MAP, and MRR.
`EvalManager` skips unreadable, missing, empty, or metric-less query runs and
aggregates over the rows that remain; `n_queries: 0` is also a valid output
shape. File presence is therefore not a completeness receipt, so every
`metrics.json` carries a `coverage` block:

```json
"coverage": {
  "complete": false,
  "query_dirs_found": 18,
  "queries_evaluated": 16,
  "queries_skipped": 2,
  "malformed_lines_dropped": 3,
  "skipped": [{"qid": "1037798", "reason": "missing_or_empty_trec_run"}]
}
```

`complete` is true only when nothing was skipped and no malformed TREC row was
dropped. Dropped rows matter on their own: they truncate that query's ranking,
so it is still scored but on fewer documents. Reject a publication run whose
coverage is not complete, and check `queries_evaluated` against the declared
population or allowlist, which coverage cannot know.

## PSI run



### Standalone PSI

```bash
uv run python scripts/run_psi.py -e configs/experiments/<ID>.yaml
```



### Co-located with base evaluation

Use the same YAML and run directory for the identity-order base followed by PSI:

```bash
uv run python scripts/run_experiment.py -e <ID>
uv run python scripts/run_eval.py -e runs/<ID>/<timestamp>/
uv run python scripts/run_psi.py -e <ID> --run-dir runs/<ID>/<timestamp>/
```

Closed-model laptop launcher (both phases, resume on base):

```bash
scripts/run_eval_robust_psi.sh example-passage-closed-model-genbsc-b20-psi
```

The launcher resumes completed base queries, then writes robustness artifacts  
into the same run directory.

Resume is safe only when the resolved config and inputs are unchanged. Existing
base shards are recognized by file presence and PSI shards by usable permutation
count, so neither path can tell what produced them. Start a new run directory
after any model, prompt, data, qid, serving-width, K, seed, or perturbation
change.

To make a violation visible rather than silent, `resolved_config.yaml` carries a
`_run_fingerprint` over the fields that change what a scored query means: the
reranker class, model, revision and adapter, the prompt and rubric, the readout
and serving width, the first-stage run and `k_input`, the qrels, and the git SHA.
Resuming into a directory whose fingerprint differs logs a warning naming the
differing fields and appends them to `resume_history.json` beside the run. It is
a warning, not a refusal: the run proceeds and the directory ends up holding
shards from both configurations, which is what the warning is telling you.
Reject any publication run with a `resume_history.json`. Nothing is reported for
directories written before the fingerprint existed, since there is nothing to
compare against.

Co-located layout:

```text
runs/<ID>/<timestamp>/
├── resolved_config.yaml
├── metrics.json                 # identity-order base (phase 1)
├── per_query_results/<qid>/     # base rerank outputs
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

`psi/psi_metrics.json` is the aggregate robustness artifact. It includes the
aggregate PSI metrics, protocol metadata, and τ-PSI@B geometry fields.
For publication, require the expected qid set and exactly `M` random evaluation
permutations per qid (stored under the historical `K_permutations` field). A
query with an empty passage list or no permutations is skipped, and an
individual permutation that produces no output is skipped, without any of them
entering `failed_qids`, so none of these fail the run. Its `coverage` block
records them:

```json
"coverage": {
  "complete": false,
  "queries_requested": 18,
  "queries_aggregated": 17,
  "queries_skipped": 1,
  "skipped": [{"qid": "264014", "reason": "empty_passage_list"}],
  "expected_presentations_per_query": 10,
  "queries_below_expected_presentations": [{"qid": "1037798", "presentations": 8}]
}
```

`queries_below_expected_presentations` is the one to watch: those queries are in
the aggregate but averaged over fewer than K presentations, which biases τ-PSI
rather than only reducing sample size. The check applies when `random_shuffle` is
configured; otherwise `expected_presentations_per_query` is null.

## Fixed-order pool perturbation

```bash
uv run python scripts/run_pool_perturbation.py -e <ID>
```

```text
runs/<ID>/<timestamp>/
├── resolved_config.yaml
├── pool_perturbation.log
└── pool_perturbation/
    ├── pool_metrics.json
    ├── pool_per_query.json
    ├── selected_qids.txt
    ├── progress.jsonl
    └── per_query_results/<qid>/
        ├── metrics.json
        ├── canonical/{detailed_results.json,trec_results_raw.txt}
        ├── replace/{detailed_results.json,trec_results_raw.txt}
        └── drop/{detailed_results.json,trec_results_raw.txt}
```

`pool_metrics.json` is the aggregate artifact. Every reported ranking,
top-k set, and nDCG metric is restricted to retained documents.

## Self-distill teacher run

Local command:

```bash
uv run python scripts/run_silver_generation.py --config configs/silver/<ID>.yaml
```

Local layout:

```text
runs/self-distill/<ID>/<timestamp>/
├── resolved_config.yaml
├── teacher_run.log
└── silver/
    ├── manifest.json
    └── silver_labels.jsonl
```

`silver/silver_labels.jsonl` is the teacher-side training artifact: one
continuous silver label per `(query, document)` pair, plus the raw
teacher-permutation score
vector used to compute it.

During a run, `silver/per_qid/<qid>.jsonl` holds the per-query resume shards
that merge into `silver_labels.jsonl` on success.

## Self-distill student run

The plain local runner uses the template's `student.output_dir` verbatim. Source
templates normally produce:

```text
runs/self-distill/<ID>/student/
├── checkpoints/checkpoint-step-XXXXXX/
├── checkpoint-final/
├── progress.jsonl
├── resolved_student_config.json
└── training_summary.json
```

That layout has no trial level and is not discoverable by the reproduction
collector. For a collector-compatible local run, override both roots:

```bash
ID=<generated-training-config-id>
TRIAL=$(date -u +%Y%m%d_%H%M%S)
uv run python scripts/run_self_distill_sft.py -e <generated-config.yaml> \
  --override student.output_dir="runs/$ID/$TRIAL/student" \
  --override student.checkpoint.dir="checkpoints/$ID/$TRIAL/student"
```

Container entrypoints create the trial-shaped run tree automatically.

## Base evaluation and PSI in one run

When a YAML carries a `robustness:` block, `scripts/train/entrypoint.py` runs:

```text
ExperimentManager -> EvalManager -> PsiExperimentRunner
```

The reranker instance is reused, so the model loads once and both outputs land
under the same run directory:

```text
runs/<ID>/<timestamp>/
├── metrics.json
└── psi/
    └── psi_metrics.json
```

`psi_metrics.json` is always under `psi/`, never at the run root.

## Offline PSI recovery

If a PSI job terminates after writing per-query permutation artifacts but before
the top-level aggregate, rebuild it through the maintained package API:

```bash
uv run python -c "from pathlib import Path; from presentation_dependence.eval.partial_psi import rebuild_partial_psi_metrics; rebuild_partial_psi_metrics(Path('runs/<ID>/<timestamp>'))"
```

This writes or overwrites:

```text
runs/<ID>/<timestamp>/psi/psi_metrics.json
runs/<ID>/<timestamp>/psi/psi_per_query.json
```



## Geometry backfill

To refresh τ-PSI@B geometry fields on an existing PSI metrics file:

```bash
uv run python scripts/analyze/backfill_tau_psi_geometry.py runs/<ID>/<timestamp>
```

This updates:

```text
runs/<ID>/<timestamp>/psi/psi_metrics.json
```



## Pruning redundant run detail

`scripts/data/prune_eval_detail.py` reclaims disk by deleting regenerable or
redundant detail across three independently guarded categories. Required
aggregates and merged artifacts are retained. The default is a dry run; pass
`--apply` to delete.

```bash
# Preview across every run:
uv run python scripts/data/prune_eval_detail.py --all --no-measure

# Apply to every run:
uv run python scripts/data/prune_eval_detail.py --apply --all

# Apply to one trial:
uv run python scripts/data/prune_eval_detail.py --apply \
    runs/<ID>/<timestamp>
```



### 1. Eval detail (PSI / quality trials)

Removed: `psi/per_query_results/`, `per_query_results/` (top-level),
`all_queries_eval_results.jsonl`. Kept: `metrics.json`,
`psi/psi_metrics.json`, `psi/psi_per_query.json`, `psi/sc_*.json`.
Refused when `psi/per_query_results/` is present but `psi/psi_metrics.json`
is missing because deletion would block partial-PSI aggregate recovery. Pass
`--allow-unaggregated` to override. Skipped when there is no aggregate
(`metrics.json` / `psi/psi_metrics.json`) to fall back on.

### 2. Training `data_views/` (SFT / self-distill trials)

Removed: `student/data_views/`, which contains regression, pairwise, listwise,
and permutation JSONL inspection sidecars written by `student.py`. Training and
evaluation do not consume these files; they regenerate deterministically from
the silver labels. Kept: `student/checkpoint-final/`, `student/progress.jsonl`,
`student/training_summary.json`, `student/resolved_student_config.json`.
Refused when the run is not complete (no `checkpoint-final/` and no
`training_summary.json`).

### 3. Silver resume state (`*-bsc-*` teacher trials)

Removed: `silver/per_qid/` (per-query resume shards) and `workers/`
(per-worker shards). Both are merged into `silver/silver_labels.jsonl`, which
`student_data.py` reads. Kept:
`silver/silver_labels.jsonl`, `silver/manifest.json`. Refused unless the
merge is verified complete: `silver_labels.jsonl` line count must equal
`manifest.json`'s `n_records` (the teacher writes the manifest last, only
on full success).

### Common behavior

Discovery (`--all`, `--exp-id`, `--sweep`, positional) recognises eval trials
by `resolved_config.yaml`, training trials by `student/`, and teacher trials by
`silver/`. Each pruned trial gets a
`.pruned.json` marker so re-runs are idempotent (`noop`). Generation continues
to produce these directories; rerun the pruner to reclaim them.

## Recording and querying selected runs



### Record a selected run

```bash
uv run python scripts/record_result.py runs/<ID>/<timestamp> \
  --status complete \
  --role primary \
  --summary "one sentence"
```

Other roles and statuses are appropriate for diagnostics, negative controls,
partial runs, failures, and superseded attempts:

```bash
uv run python scripts/record_result.py runs/<ID>/<timestamp> \
  --status diagnostic \
  --role negative-control \
  --summary "why this run matters"

uv run python scripts/record_result.py runs/<ID>/<timestamp> --dry-run
```

Validate after every change:

```bash
uv run python scripts/record_result.py --validate-index
```

Validation checks schema, fields, and canonical ordering without requiring the
local `runs/` cache. Add `--check-paths` when the referenced runs are expected
to be present on the current machine.

### Query selected runs

```bash
uv run python scripts/query_results.py
uv run python scripts/query_results.py --collection dl19 --status complete
uv run python scripts/query_results.py --model-family qwen3 --has-psi
uv run python scripts/query_results.py --format json
```

Queries also work from selected metadata alone; add `--check-paths` to audit the
optional local run cache.

Markdown output is a drafting aid. Use artifact JSON for exact values.

### Sort the index

The deterministic scan order is:

1. collection;
2. dataset;
3. series;
4. model family;
5. protocol;
6. status;
7. role;
8. experiment ID;
9. run directory.

Repair ordering with:

```bash
uv run python scripts/record_result.py --sort-index
uv run python scripts/record_result.py --validate-index
```



### Interpretation guardrails

- Record the selected run, not every retry or top-up.
- Do not promote smoke, diagnostic, failed, partial, or negative-control runs
into primary comparisons.
- Keep PSI protocol and τ-PSI geometry with every robustness measurement.
- Do not compare `mean_tau_based_psi` across different context batch sizes.
- Dashboard and Markdown values may be rounded; use artifact JSON.
- `docs/results_index.yaml` is the sole selected-run registry. Put durable
caveats in its `notes`/`caveats` fields and keep full interpretation with the
hash-bearing reproduction artifact or paper source; do not recreate a
Markdown ledger or scoreboard.

