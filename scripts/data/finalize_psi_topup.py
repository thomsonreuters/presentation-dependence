#!/usr/bin/env python
"""Merge a PSI top-up run into its parent and re-aggregate headline metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

from presentation_dependence.eval.partial_psi import rebuild_partial_psi_metrics
from presentation_dependence.eval.psi_topup import merge_psi_topup


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True, help="Parent (partial) run directory.")
    parser.add_argument("--topup", type=Path, required=True, help="Top-up run directory.")
    parser.add_argument("--overwrite", action="store_true", help="Replace complete parent qids.")
    parser.add_argument(
        "--min-perms",
        type=int,
        default=None,
        help="Minimum complete presentations required for PSI aggregation.",
    )
    args = parser.parse_args()

    parent = args.parent.resolve()
    topup = args.topup.resolve()

    print("[finalize] merge …")
    merge_psi_topup(parent, topup, overwrite=args.overwrite)
    print("[finalize] aggregate …")
    rebuild_partial_psi_metrics(parent, min_perms=args.min_perms)
    print(f"[finalize] done -> {parent / 'psi/psi_metrics.json'}")


if __name__ == "__main__":
    main()
