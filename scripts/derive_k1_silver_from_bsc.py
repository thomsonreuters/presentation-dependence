#!/usr/bin/env python
"""Derive lower-K silver labels from a K-shot BSC silver JSONL."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from presentation_dependence.silver_data.transforms import derive_silver_shard


def parse_args() -> argparse.Namespace:
    """Parse lower-K derivation CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--silver-in", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--perm-index", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=None)
    parser.add_argument("--k-out", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Derive and report one lower-K silver shard."""
    args = parse_args()
    start_index = args.perm_index if args.start_index is None else args.start_index
    try:
        stats = derive_silver_shard(
            args.silver_in,
            args.out,
            k_out=args.k_out,
            start_index=start_index,
            force=args.force,
        )
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        print(f"[derive-k][FATAL] {exc}", file=sys.stderr)
        return 2
    print(f"[derive-k] input       = {stats.input_path}")
    print(f"[derive-k] output      = {stats.output_path}")
    print(f"[derive-k] k_out       = {stats.k_out}")
    print(f"[derive-k] start_index = {stats.start_index}")
    print(f"[derive-k] records     = {stats.records}")
    print(f"[derive-k] qids        = {stats.qids}")
    print(f"[derive-k] score stats = mean {stats.score_mean:.6f} min {stats.score_min:.6f} max {stats.score_max:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
