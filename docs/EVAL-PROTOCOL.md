# Evaluation and checkpoint-selection protocol

Select checkpoints and lambda values for trained recipes. Applies to
K=10 SFT, the K-label and label-count sweeps, OC-SFT, defensive baselines,
multi-seed verification, and cross-family runs.

## Selection invariants

All selection uses held-out development data. Reported test sets and downstream
evaluation datasets do not participate in checkpoint or lambda selection.

Within one training config, select the checkpoint with the highest declared
held-out metric. An exact tie selects the earliest step. Passage reranking uses
held-out MS MARCO nDCG@10; QA uses its disjoint 500-question HotpotQA held-out
split; response ranking uses held-out UltraFeedback for checkpoint quality.

Lambda selection depends on the task pipeline:

- Passage reranking and multi-document QA apply
`select_heldout_lambda`: exclude collapsed or incomplete candidates, select
the largest lambda within one standard error of the best converged held-out
nDCG@10, and select that lambda's earliest maximum checkpoint.
- Reported response-ranking λ uses 0.5 at 1.7B and 1.0 at 4B. `select_amortization_matching` remains an optional implementation for new experiments that want the smallest λ matching a K-shot stability target.

The task declarations are under `training.checkpoint_selection` in
`configs/reproduction/<task>.yaml`, where `selection_protocol` names the
implementation of record.

No pipeline stage executes `select_amortization_matching`; it needs per-lambda  
development stability rows that this release does not produce.  
`scripts/select_lambda.py` executes  
`select_heldout_lambda` over discovered runs and writes the decision record at  
`configs/reproduction/evidence/lambda-selection/decisions.json`; collection reads that record. Response ranking anchors on its declared `expected_lambda`unless an explicit receipt is supplied. Collection fails rather than guessing when none of a decision cell, an anchor, or a receipt resolves.

## Configuration that enforces the rule

Training configs implement the rule with:

```yaml
student:
  evaluation:
    max_queries: 100
    every_n_steps: 200
    at_start: true
    at_end: true
  checkpoint:
    enabled: true
    save_every_n_steps: 200
    keep_last_n: 4
    keep_every_n_steps: 400
    keep_n_best: 3
    keep_n_best_metric: qrels_ndcg_cut_10
    keep_n_best_mode: max
```

`keep_n_best_metric: qrels_ndcg_cut_10` with `keep_n_best_mode: max` retains the
top three checkpoints ranked by held-out qrels nDCG@10. Selection reports the
highest-ranked retained checkpoint unless the task-level lambda selector
specifies a different rule.

## One-standard-error lambda and checkpoint selection

Plain SFT selects the highest-ranked held-out checkpoint within one training
config. Passage-reranking and multi-document-QA OC-SFT additionally select
across the declared lambda grid with `scripts/select_lambda.py`.

The selector checks candidate completeness, converged regions, collapse,
one-standard-error bands, and tie-breaking. Its summary includes
`confidence=HIGH|LOW|ABSTAIN`.

- default exit `0`: successful HIGH or LOW decision;
- exit `2`: no trustworthy candidate; decision is `ABSTAIN`;
- with `--strict`, LOW exits `3`.

The command requires the task config template and run source:

```bash
uv run python scripts/select_lambda.py \
  --config-template "<cfg>-supervised-consistency-lambda{code}-warmup500-msmarco-30k" \
  --family "<Family>" --k "K=1" --recipe "<recipe>" \
  --emit-cell
```



### Collection reads the recorded selection; it does not recompute it

`study.py <task> training collect` does not re-derive lambda. It resolves the
selection in this order: an explicit `lambda_selection.json` receipt, then the
`decision_cell` its pipeline declares against
`configs/reproduction/evidence/lambda-selection/decisions.json`, then a declared
`expected_lambda`. With none of those it fails rather than guessing.

The collapse and one-standard-error rules need per-lambda score deviations from
run artifacts, which tracked evidence does not carry. A plain held-out argmax is
not a safe substitute: historical replay disagreed with `select_lambda.py`
because that selector breaks 1-SE ties toward higher lambda, with gaps as large
as 5.0 versus 2.0. Rerunning a grid means rerunning `select_lambda.py` and
updating the decision record, not letting collection infer a value.

For the legacy passage-reranking sweep, the implemented rule is:

1. use the latest job for each lambda so superseded restarts cannot win;
2. exclude pre-convergence checkpoints (steps below 1000);
3. disqualify a lambda when minimum `student_stdev` is below `0.02`;
4. retain candidates within `0.015` of the converged argmax;
5. choose the highest retained lambda and that trajectory's converged argmax
  checkpoint.

`HIGH` is suitable for final use. `LOW` is provisional because of collapse,
incomplete coverage, a boundary choice, a flat/non-contiguous band, a spike,
thin samples, low quality, or missing dispersion. `ABSTAIN` means no candidate
is trustworthy. A checkpoint already consumed downstream remains fixed only
when its exception is recorded in the selected-run/evidence workflow.

Machine-readable historical decisions and trajectory assets under
`configs/reproduction/evidence/lambda-selection/` remain non-Markdown evidence, not operator
documentation. The selected lambda and checkpoint are recorded in
`build/reproduction/<task>/training/checkpoints.json`.

## Passage-reranking defaults


| Knob                            | Value                  | Rationale                              |
| ------------------------------- | ---------------------- | -------------------------------------- |
| Held-out set                    | `qids_heldout_500.txt` | 500 reserved queries; eval samples 100 |
| `evaluation.max_queries`        | `100`                  | Fixed across compared runs             |
| `evaluation.every_n_steps`      | `200`                  | Default evaluation cadence             |
| `checkpoint.save_every_n_steps` | `200`                  | Default checkpoint cadence             |
| `checkpoint.keep_n_best_metric` | `qrels_ndcg_cut_10`    | The selection metric                   |
| `checkpoint.keep_n_best_mode`   | `max`                  | Argmax over checkpoints                |
| `checkpoint.keep_n_best`        | `3`                    | Top-3 retained, top-1 reported         |


Short-horizon passage-reranking recipes (≤500 steps) may use a cadence of `100`.
Smoke configs that use `save_every_n_steps: 1` are not used for reporting.
Other tasks use the cadence declared in their reproduction pipeline; for
example, multi-document QA declares `cadence_steps: 50`.

## Adding a new training config: checklist

Before launching a new SFT or SFT-with-consistency run:

- [ ] `evaluation.max_queries` matches the task declaration
- [ ] Evaluation and checkpoint cadence match the task's
  ```
  `training.checkpoint_selection` declaration
  ```
- [ ] `checkpoint.keep_n_best_metric` matches the declared selection metric
- [ ] `checkpoint.keep_n_best_mode` matches the declared selection direction
- [ ] `checkpoint.keep_n_best: 3`
- [ ] Within a recipe family, cadence and held-out data are
  ```
  consistent across all variants
  ```
- [ ] Passage reranking uses `qids_heldout_500.txt`; a different held-out set
  ```
  requires a new task declaration
  ```
- [ ] Reported test and downstream datasets are excluded from selection

For a multi-config sweep, use the same cadence and held-out data across all
variants.

## Recording a run

When a training run finishes, maintained records follow
`[RUN-ARTIFACTS.md](RUN-ARTIFACTS.md#recording-and-querying-selected-runs)` and use the selected checkpoint:

- `runs/<ID>/<timestamp>/metrics.json` reports selected-checkpoint metrics.
- `psi/psi_metrics.json` reports PSI on the same selected checkpoint.
- the `results_index.yaml` row uses the selected-checkpoint numbers and includes the
selected step in the run record.

The selected step is documented in `results_index.yaml` under the run record but
is not added to historical scoreboard/results cells.

## The checkpoint catalog

Training collection writes the selected checkpoints to
`build/reproduction/<task>/training/checkpoints.json`, and appendix reporting
reads that catalog. One row per selected checkpoint:

```json
{
  "variant": "oc-sft",
  "training_seed": 42,
  "selected_step": 1000,
  "checkpoint_uri": "checkpoints/.../student/checkpoint-step-001000/",
  "trial": "...",
  "selection_rule": "heldout-argmax"
}
```

Direct-evaluation and study materializers resolve `(variant, training_seed)`
against this catalog and fail when the requested row is missing, rather than
falling back to a recent run. Never select a checkpoint by recency.

A trained result is identified by its base model and immutable revision, the
adapter checkpoint URI and hash, the training variant and seed, the selected
step and the rule that chose it, the prompt, rubric and readout, and the serving
width and truncation. Two runs that differ in any of those are different
results.

After evaluating a selected checkpoint, verify that the resolved base revision,
adapter path, checkpoint URI and step, metric and PSI artifacts, and source-lock
status all match what was intended, then record it:

```bash
uv run python scripts/record_result.py runs/<id>/<trial> \
  --status complete --role primary --summary "selected checkpoint evaluation"
```

If direct materialization picks up an old checkpoint, re-run training
collection, inspect `checkpoints.json`, remove the stale generated configs, and
rematerialize. Do not patch adapter URIs into generated configs by hand.

Query selected runs with:

```bash
uv run python scripts/query_results.py --status complete --format markdown
```



## What this protocol does not cover

- Inference-time K (BSC K=1 vs K=10 at eval): orthogonal. PSI artifact geometry
and inference-K choices are documented elsewhere.
- PSI protocol (B=20, ten random-shuffle presentations): independent of
checkpoint selection. Targeted injections are reported separately.
- Off-the-shelf baselines (mxbai, Jina, RankZephyr-7B, and Qwen3-Reranker  
pointwise): no checkpoint to select, so the checkpoint-selection rule does  
not apply.

