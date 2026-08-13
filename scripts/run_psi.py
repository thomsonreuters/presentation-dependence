#!/usr/bin/env python
r"""Run K-permutation PSI scoring for one experiment config.

The command uses the same experiment config as ``scripts/run_experiment.py``.
It runs K permuted passes per query (default 10) and aggregates the robustness
metrics implemented by ``presentation_dependence.eval.psi.evaluate_psi``.

The experiment YAML needs a sibling ``robustness:`` block on top of the
usual ``reranker`` / ``data`` / ``eval`` blocks:

    robustness:
      K: 10
      seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]     # len must equal K
      perturbations: [random_shuffle]            # or: [random_shuffle, middle_injection]
      k_cutoff_for_ndcg: 10

When the block is absent, defaults apply (K=10, random_shuffle,
seeds 0..K-1). Override from the CLI with the standard dotted-key syntax,
e.g. ``--override robustness.K=20``.

Usage::

    uv run python scripts/run_psi.py -e example-passage-b20-psi
    uv run python scripts/run_psi.py -e example-passage-b20-psi \\
        --override robustness.K=20 \\
        --override 'robustness.perturbations=[random_shuffle, middle_injection]'

Outputs land under ``runs/<ID>/<ts>/psi/``. For open-model parity, run
identity-order base scoring first and pass ``--run-dir`` so ``metrics.json`` and
``psi/`` share one timestamped dir.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from dotenv import load_dotenv

from presentation_dependence.utils.dry_run import emit_plan
from presentation_dependence.utils.config import (
    apply_execution_environment,
    apply_overrides,
    load_experiment_config,
    write_resolved_config,
)
from presentation_dependence.utils.gpu import log_cuda_diagnostics
from presentation_dependence.utils.run_paths import default_runs_root


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-e",
        "--exp-config",
        type=str,
        required=True,
        help="Exp-id (e.g. example-passage-b20-psi) or path to configs/experiments/<ID>.yaml.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Dotted-key override, e.g. robustness.K=20. Repeatable.",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=default_runs_root(),
        help="Root directory for run artefacts (default: SLM_RUNS_ROOT, else runs/).",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help=(
            "Reuse this exact run dir (same timestamped dir as a preceding "
            "run_experiment.py pass). Writes PSI artefacts into <run-dir>/psi/ "
            "and does not overwrite the identity-order per_query_results/ or "
            "metrics.json from step 1."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve the config, check declared inputs exist, print the plan, and exit. No GPU, no model load.",
    )
    args = parser.parse_args()

    load_dotenv()

    config_path, cfg = load_experiment_config(args.exp_config)
    if args.override:
        cfg = apply_overrides(cfg, args.override)
        tmp_dir = Path(tempfile.mkdtemp(prefix="slm_psi_resolved_"))
        config_path = write_resolved_config(cfg, tmp_dir / config_path.name)
        print(f"[psi] applied {len(args.override)} override(s); resolved -> {config_path}")

    if args.dry_run:
        raise SystemExit(
            emit_plan(
                "psi",
                config_path,
                cfg,
                project_root=_project_root(),
                overrides=args.override,
                extra={"runs_root": args.runs_root, "run_dir": args.run_dir, "robustness": cfg.get("robustness")},
            )
        )

    apply_execution_environment(cfg)
    log_cuda_diagnostics("[psi][diagnostics]")

    # Imported here so --dry-run needs neither pytrec_eval nor torch.
    from presentation_dependence.eval.psi_manager import PsiExperimentRunner

    runner = PsiExperimentRunner(
        config_path=config_path,
        runs_root=args.runs_root,
        run_dir=args.run_dir,
    )
    metrics = runner.run()

    print(f"[OK] psi run_dir = {runner.psi_dir}")
    agg = metrics.get("aggregate", {})
    print(f"[OK] n_queries={agg.get('n_queries')}  K={agg.get('K_permutations')}")
    headline = (
        "mean_score_variance",
        "mean_rank_variance",
        "mean_kendall_tau",
        "mean_tau_based_psi",
        "mean_zeng_psi",  # per-query Zeng PSI averaged
        "zeng_psi_corpus",  # matches Zeng et al. Table 1
        "mean_delta_ndcg",
        "mean_per_perm_ndcg",  # E_q[ mean_k nDCG ] — bracketed by SC from above
        "mean_worst_perm_ndcg",
        "mean_best_perm_ndcg",
    )
    for m in headline:
        val = agg.get(m)
        print(f"     {m}: {val}")

    # Also surface the K-shot self-consistency K-curve when the live run
    # produced ``sc_metrics.json`` (batched-PW / scoring-listwise rerankers).
    sc_path = runner.psi_dir / "sc_metrics.json"
    if sc_path.exists():
        sc = json.loads(sc_path.read_text())
        first_measure = (sc.get("measures") or ["ndcg_cut_10"])[0]
        print(f"[OK] SC K-curve ({first_measure}, n={sc.get('n_queries_with_scores')}):")
        for K, block in sorted(sc.get("by_K", {}).items(), key=lambda kv: int(kv[0])):
            m = block.get("metrics", {}).get(first_measure, {})
            print(f"     K={int(K):>3}  mean={m.get('mean')}  std={m.get('std')}")


if __name__ == "__main__":
    main()
