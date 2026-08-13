#!/usr/bin/env python
"""Create a non-overlapping delta between nested deterministic QID prefixes."""

from __future__ import annotations

import argparse
from pathlib import Path

from presentation_dependence.silver_data.transforms import materialize_qid_delta


def parse_args() -> argparse.Namespace:
    """Parse QID delta CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--sample-order", type=Path, default=None)
    parser.add_argument("--target-count", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    """Materialize and report one QID delta."""
    args = parse_args()
    delta = materialize_qid_delta(
        args.base,
        args.target,
        args.out,
        sample_order=args.sample_order,
        target_count=args.target_count,
    )
    print(f"[make_qid_delta] base:   {args.base}")
    print(f"[make_qid_delta] target: {args.target}")
    print(f"[make_qid_delta] delta:  {args.out} ({len(delta)} qids)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
