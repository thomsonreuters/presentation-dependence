"""Backfill τ-PSI@B fields on existing ``psi/psi_metrics.json`` from ``resolved_config.yaml``.

Requires ``resolved_config.yaml`` and ``psi/psi_metrics.json`` under the run
directory. Geometry is inferred from YAML without loading a reranker.

Usage::

  uv run python scripts/analyze/backfill_tau_psi_geometry.py runs/<ID>/<ts>/
  uv run python scripts/analyze/backfill_tau_psi_geometry.py --exp-id <ID>
  uv run python scripts/analyze/backfill_tau_psi_geometry.py <run_dir> --dry-run
  uv run python scripts/analyze/backfill_tau_psi_geometry.py <run_dir> --force
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from presentation_dependence.eval.tau_psi_geometry import merge_tau_psi_geometry_into_psi_metrics
from presentation_dependence.utils.run_paths import default_runs_root, resolve_psi_cli_run_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dir",
        nargs="?",
        type=str,
        help="Path to runs/<ID>/<ts>/ (or omit if using --exp-id).",
    )
    parser.add_argument(
        "--exp-id",
        type=str,
        default=None,
        help="Resolve to the latest timestamp under runs/<exp-id>/.",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=default_runs_root(),
        help="Runs root (default: SLM_RUNS_ROOT, else runs/).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be written; do not modify psi_metrics.json.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing tau_psi_* keys (default: skip when both are present).",
    )
    args = parser.parse_args()

    try:
        run_dir = resolve_psi_cli_run_dir(run_dir=args.run_dir, exp_id=args.exp_id, runs_root=args.runs_root)
    except ValueError as e:
        parser.error(str(e).replace("exp_id", "--exp-id"))

    cfg_path = run_dir / "resolved_config.yaml"
    if not cfg_path.is_file():
        raise SystemExit(f"[backfill] Missing {cfg_path}")

    psi_path = run_dir / "psi" / "psi_metrics.json"
    if not psi_path.is_file():
        raise SystemExit(f"[backfill] Missing {psi_path}")

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    with open(psi_path) as f:
        existing = json.load(f)

    action, merged = merge_tau_psi_geometry_into_psi_metrics(
        existing,
        cfg,
        reranker=None,
        overwrite=args.force,
    )

    print(f"[backfill] run_dir = {run_dir}")
    print(f"[backfill] action   = {action}")
    print(
        f"[backfill] tau_psi_context_batch_size = {merged.get('tau_psi_context_batch_size')!r}",
    )
    note = merged.get("tau_psi_inference_note")
    if note:
        snippet = note[:200] + ("…" if len(note) > 200 else "")
        print(f"[backfill] tau_psi_inference_note     = {snippet!r}")

    if action == "skipped":
        print("[backfill] nothing to do (use --force to refresh).")
        return 0

    if args.dry_run:
        print("[backfill] dry-run: not writing.")
        return 0

    with open(psi_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    print(f"[backfill] wrote {psi_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
