#!/usr/bin/env python
"""Split self-distillation silver into deterministic train and held-out cohorts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from presentation_dependence.silver_data.transforms import split_silver_cohort


DEFAULT_DATA_DIR = Path("data/msmarco-train-selfdistill-seed42")


def parse_args() -> argparse.Namespace:
    """Parse held-out split CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--silver-in", type=Path, required=True)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--out-stem", type=str, default=None)
    parser.add_argument("--reuse-qid-list", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Split the requested silver shard."""
    args = parse_args()
    out_dir = args.out_dir or args.data_dir
    try:
        result = split_silver_cohort(
            args.silver_in,
            n_heldout=args.n,
            sample_order_path=args.data_dir / "sample_qids_seed42.txt",
            training_qids_path=args.data_dir / "qids_30k.txt",
            out_dir=out_dir,
            out_stem=args.out_stem,
            reuse_qid_list=args.reuse_qid_list,
            force=args.force,
        )
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        print(f"[heldout][FATAL] {exc}", file=sys.stderr)
        return 2
    print(
        f"[heldout] split {result.input_records} records: "
        f"{result.training_records} -> {result.training_path.name} "
        f"({len(result.training_qids)} qids), "
        f"{result.heldout_records} -> {result.heldout_path.name} "
        f"({len(result.heldout_qids)} qids)"
    )
    if result.reused_qid_list:
        print(f"[heldout] reusing existing held-out qid list at {result.heldout_qids_path}")
    print(f"[heldout] qids:          {result.heldout_qids_path}")
    print(f"[heldout] train silver:  {result.training_path}")
    print(f"[heldout] eval silver:   {result.heldout_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
