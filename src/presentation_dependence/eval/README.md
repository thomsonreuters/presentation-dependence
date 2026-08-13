# `eval/`

Evaluates a reranker (`presentation_dependence.rerankers`) against a
first-stage run. Produces per-query scores, ranking-quality metrics (nDCG),
and order-stability metrics (PSI / τ-PSI) with their channel decompositions.
Execution code reads `runs/`; reporting reducers read `build/reproduction/`.

## Job orchestration

- [`experiment_manager.py`](experiment_manager.py): experiment runner: loads a
  reranker, iterates queries, writes per-query outputs. Reruns are additive
  (existing per-query results are reused; only missing qids are reranked).
- [`eval_manager.py`](eval_manager.py): scores per-query TREC runs with
  `pytrec_eval` and writes `metrics.json`.
- [`bundle.py`](bundle.py): runs several evaluation collections in one job on a
  single model load (base plus N LoRA adapters).
- [`runner_setup.py`](runner_setup.py): shared setup for the experiment and PSI
  runners: reranker construction, run-directory provenance, logging.
- [`loaders.py`](loaders.py): dataloaders for first-stage retrieved runs:
  `PyseriniLoader`, `FixtureLoader`, `BaseLoader`, `LOADER_CLASSES`.
- [`dataset_catalog.py`](dataset_catalog.py): loads dataset populations for
  config generators and studies.

## PSI (permutation / presentation sensitivity)

- [`psi.py`](psi.py): core metrics from aligned rankings and scores
  (`evaluate_psi`, `zeng_psi`), including the Δ-nDCG position readout.
- [`psi_manager.py`](psi_manager.py): PSI run driver: generates presentations
  (random shuffles, middle injection, render/scale variants), scores each, and
  aggregates τ-PSI.
- [`psi_artifacts.py`](psi_artifacts.py): writes the PSI result envelope
  (`psi_metrics.json`).
- [`partial_psi.py`](partial_psi.py): rebuilds PSI aggregates from completed
  per-query artifacts, for recovery after a truncated run.
- [`psi_topup.py`](psi_topup.py): merge helpers for focused PSI top-up runs onto
  a partial result.
- [`tau_psi_geometry.py`](tau_psi_geometry.py): infers τ-PSI@B reporting fields
  (window/stride/batch geometry) from a config and its reranker. Registering a
  new reranker may need a branch here.

## Channel decompositions

- [`pool_perturbation.py`](pool_perturbation.py): fixed-order candidate-pool
  perturbation (replace/drop top-N); runs in place of ordinary eval/PSI.
- [`context_decomposition.py`](context_decomposition.py): decomposition of the
  fixed-weight serving-width effect.
- [`matched_variance_control.py`](matched_variance_control.py): repeated
  fixed-order (B=1) scoring with varied cross-query batch positions and
  compositions.
- [`set_sensitivity.py`](set_sensitivity.py): split a permutation run's
  within-document score variance into slot, chunk-index, and companion shares.
  Driven by `scripts/analyze/analyze_variance_decomposition.py` through
  `analysis/set_sensitivity.py`. Offline: re-reads a finished PSI run; no GPU.
- [`copresence_scores.py`](copresence_scores.py): score-level diagnostics for the
  fixed-weights co-presence estimand. Helper only; no in-tree importer yet.

## Self-consistency

- [`self_consistency.py`](self_consistency.py): K-shot self-consistency metrics
  from per-permutation reranker scores (batched-pointwise scorers).
- [`rank_self_consistency.py`](rank_self_consistency.py): rank-space variant for
  generative-listwise rerankers, which emit no per-candidate score.

## Aligned scores and logs

- [`aligned_scores.py`](aligned_scores.py): loads per-document scores aligned
  across scorer presentations; the common input to PSI and the reductions below.
- [`score_log.py`](score_log.py): Parquet I/O for presentation-aligned
  per-document score logs.
- [`score_distribution.py`](score_distribution.py): score-scale diagnostics for
  aligned presentations.

## Decision reductions

- [`threshold_stability.py`](threshold_stability.py): stability of thresholded
  document sets under reshuffling.
- [`retained_set_reduction.py`](retained_set_reduction.py): end-to-end
  retained-set threshold reduction over aligned scorer outputs.
- [`topk_stability.py`](topk_stability.py): top-k ranking stability across
  repeated presentations.
- [`response_selection.py`](response_selection.py): response-selection
  reductions over scorer presentations.

## Public API

[`__init__.py`](__init__.py) re-exports the common entry points:
`ExperimentManager`, `EvalManager`, the loaders (`LOADER_CLASSES`, `BaseLoader`,
`FixtureLoader`, `PyseriniLoader`), and `evaluate_psi` / `zeng_psi`.

## Notes

- `experiment_manager.py`, `eval_manager.py`, and `loaders.py` define the run
  and TREC file formats. Keep them byte-stable: run files are compared across
  experiments, so a formatting change silently invalidates comparisons.
- `pool_perturbation`, `context_decomposition`, `matched_variance_control`, and
  `self_consistency` are optional config blocks; their schema is in
  `configs/experiments/_schema.md`.
