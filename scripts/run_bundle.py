#!/usr/bin/env python
r"""Run a multi-surface eval bundle locally: one model load, N datasets.

A bundle YAML carries ``bundle.member_configs: [<exp_id>, ...]`` naming
per-surface configs that already exist under ``configs/experiments/``. Every
member is resolved, then all of them run serially against one loaded reranker,
which is what makes bundling worth doing: for a 30B base, loading the model
dominates the cost of evaluating a small surface.

Usage::

    python scripts/run_bundle.py -e <bundle-id>
    python scripts/run_bundle.py -e configs/experiments/<bundle-id>.yaml --dry-run

Per-surface output lands under ``<runs-root>/<member-id>/<ts>/``, the same tree
a single-surface run produces, so scoring and recording need no special case. A
failure is isolated to its surface: the others still complete and remain on
disk.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv  # noqa: E402

from presentation_dependence.utils.config import (  # noqa: E402
    apply_bundle_member_overrides,
    apply_execution_environment,
    apply_overrides,
    load_experiment_config,
)
from presentation_dependence.utils.dry_run import emit_plan  # noqa: E402
from presentation_dependence.utils.run_paths import default_runs_root  # noqa: E402


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_output_roots(args: argparse.Namespace, project_root: Path) -> tuple[Path, Path]:
    """Resolve manifest and member-run roots without introducing ``runs/runs``."""
    default_root = default_runs_root()
    if not default_root.is_absolute():
        default_root = project_root / default_root
    output_dir = args.output_dir or default_root
    if args.runs_root is not None:
        runs_root = args.runs_root
    elif args.output_dir is not None:
        runs_root = output_dir / "runs"
    else:
        runs_root = default_root
    return output_dir, runs_root


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-e", "--exp-config", required=True, help="Bundle exp-id or path to its YAML.")
    p.add_argument("--override", action="append", default=[], help="Dotted-key override. Repeatable.")
    p.add_argument("--runs-root", type=Path, default=None, help="Run-artifact root. Default: <output-dir>/runs.")
    p.add_argument("--output-dir", type=Path, default=None, help="Where the bundle manifest lands. Default: runs/.")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve the bundle and its members, check declared inputs exist, print the plan, and exit.",
    )
    return p.parse_args()


def main() -> int:  # noqa: C901
    """Resolve a bundle's members and run them against one reranker."""
    args = parse_args()
    load_dotenv()
    root = _project_root()

    config_path, cfg = load_experiment_config(args.exp_config)
    if args.override:
        cfg = apply_overrides(cfg, args.override)

    members = (cfg.get("bundle") or {}).get("member_configs") or []
    if not members:
        print(f"[bundle][FATAL] {config_path} has no bundle.member_configs", file=sys.stderr)
        return 2
    if cfg.get("reranker"):
        print(
            "[bundle][FATAL] a top-level reranker block is ignored for bundles; "
            "use bundle.reranker_overrides.* instead.",
            file=sys.stderr,
        )
        return 2

    bundle_cfg = cfg.get("bundle") or {}
    member_cfgs: list[dict] = []
    for member in members:
        _, member_cfg = load_experiment_config(str(member))
        member_cfgs.append(apply_bundle_member_overrides(member_cfg, bundle_cfg))

    if args.dry_run:
        worst = 0
        for member_cfg in member_cfgs:
            worst = max(
                worst,
                emit_plan(
                    "bundle-member",
                    config_path,
                    member_cfg,
                    project_root=root,
                    overrides=args.override,
                    extra={"bundle": cfg.get("id"), "shared_base": (cfg.get("bundle") or {}).get("shared_base", False)},
                ),
            )
        return worst

    apply_execution_environment(cfg)

    # Imported here so --dry-run needs neither torch nor pytrec_eval.
    from presentation_dependence.eval.bundle import run_bundle

    output_dir, runs_root = _resolve_output_roots(args, root)
    summaries = run_bundle(
        member_cfgs,
        output_dir=output_dir,
        runs_root=runs_root,
        shared_base=bool((cfg.get("bundle") or {}).get("shared_base", False)),
    )
    failed = [s for s in summaries if s.get("status") != "ok"]
    for summary in failed:
        print(f"[bundle] FAILED surface {summary.get('id')}: {summary.get('error')}", file=sys.stderr)
    print(f"[bundle] {len(summaries) - len(failed)}/{len(summaries)} surface(s) complete", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
