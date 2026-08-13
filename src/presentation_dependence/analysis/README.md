# `analysis/`

Post-hoc analysis primitives and the representative-check path. Computes
the numbers that figure and table scripts consume; off the training/eval hot
path.

## Metrics and representative checks

- [`metrics.py`](metrics.py): analysis primitives aligned with the reported
  appendix implementations: bootstrap CIs (`paired_bootstrap_ci`,
  `hierarchical_bootstrap_ci`), `tau_psi_from_aligned_scores`,
  `verdict_flip_rate`, `mean_pairwise_jaccard`, `gain_histogram`,
  `clustered_linear_calibration`. `compute_analysis` is the
  operation dispatcher for representative checks (precomputed payloads only;
  not run directories).
- [`representative.py`](representative.py): YAML-driven representative appendix
  analysis: runs one recorded value per family from a tracked fixture
  (`configs/reproduction/representative-analyses.yaml`) and records SHA-256
  provenance. Dispatches operations through `metrics.compute_analysis`.
- [`cli.py`](cli.py): `study.py representative` parser (plan / run / validate),
  wired into `reproduction.runner`.

## Harvesting and decompositions

- [`amortization.py`](amortization.py): run classification and scalar harvesting
  from `runs/` for the amortization analyses: family/size/recipe parsing
  (`parse_run`), collection detection, `harvest`, and `discover`, which ranks run
  stems by collection coverage only. Lambda is never inferred from a run name;
  callers read the recorded decision instead.
- [`set_sensitivity.py`](set_sensitivity.py): per-cell driver for the
  score-variance decomposition. Locates each cell's retained aligned-score
  artifacts, reduces them through `eval.set_sensitivity`, and reports slot /
  chunk-index / companion shares. Entry point:
  `scripts/analyze/analyze_variance_decomposition.py`.

[`__init__.py`](__init__.py) re-exports the `metrics` primitives and the
`representative` plan/run/validate functions.
