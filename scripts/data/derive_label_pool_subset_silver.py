#!/usr/bin/env python
r"""Derive a query-level subset from a self-distill silver JSONL.

Label-pool sample-efficiency runs keep the optimizer-step budget fixed while
varying the number of silver-labeled training queries. This helper selects
complete query groups so each selected query retains all candidate labels.

Modes:
  random (default): deterministic random sample of N qids (--seed required).
  prefix: first N qids in silver source order (nested frozen prefixes).

Examples:
    # Random subset (Qwen3-4B-Instruct label-pool sweep)
    uv run python scripts/data/derive_label_pool_subset_silver.py \\
        --silver-in data/msmarco-train-selfdistill-seed42/silver_labels_a1_k10_train_29500.jsonl \\
        --n-qids 1000 --seed 0 \\
        --qid-out data/msmarco-train-selfdistill-seed42/qids_a1_k10_train_1000_seed0.txt

    # Nested prefix subset (OG Qwen3-4B sample-efficiency sweep)
    uv run python scripts/data/derive_label_pool_subset_silver.py \\
        --mode prefix \\
        --silver-in data/msmarco-train-selfdistill-seed42/silver_labels_qwen3_4b_k1_seed0_train_29500.jsonl \\
        --n-qids 1000 \\
        --qid-out data/msmarco-train-selfdistill-seed42/qids_qwen3_4b_k1_train_1000_prefix.txt
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any


_TRAIN_COUNT_RE = re.compile(r"_train_\d+$")


def _default_out_path(silver_in: Path, n_qids: int, *, mode: str, seed: int | None) -> Path:
    """Return the conventional subset output path."""
    stem = silver_in.stem
    if mode == "prefix":
        replacement = f"_train_{n_qids}_prefix"
    else:
        replacement = f"_train_{n_qids}_seed{seed}"
    if _TRAIN_COUNT_RE.search(stem):
        stem = _TRAIN_COUNT_RE.sub(replacement, stem)
    elif mode == "prefix":
        stem = f"{stem}_train_{n_qids}_prefix"
    else:
        stem = f"{stem}_subset{n_qids}_seed{seed}"
    return silver_in.with_name(f"{stem}{silver_in.suffix}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--silver-in", required=True, type=Path, help="Input train silver JSONL.")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSONL. Defaults to replacing _train_<N> with _train_<n-qids>_seed<seed>.",
    )
    p.add_argument("--n-qids", required=True, type=int, help="Number of complete query groups to select.")
    p.add_argument(
        "--mode",
        choices=("random", "prefix"),
        default="random",
        help="Selection mode: random sample (default) or first-N prefix in source order.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Deterministic query-subset seed for --mode random. Ignored for prefix.",
    )
    p.add_argument(
        "--qid-out",
        type=Path,
        default=None,
        help="Optional path for the sampled qids, one per line in source-order.",
    )
    p.add_argument("--force", action="store_true", help="Overwrite --out / --qid-out if they already exist.")
    return p.parse_args()


def _read_grouped(path: Path) -> OrderedDict[str, list[dict[str, Any]]]:
    grouped: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"line {line_no}: invalid JSON: {e}") from e
            if not isinstance(rec, dict):
                raise ValueError(f"line {line_no}: expected JSON object, got {type(rec).__name__}")
            qid = rec.get("query_id")
            if qid is None:
                raise ValueError(f"line {line_no}: missing query_id")
            grouped.setdefault(str(qid), []).append(rec)
    return grouped


def _with_subset_provenance(
    rec: dict[str, Any],
    *,
    silver_in: Path,
    n_qids: int,
    mode: str,
    seed: int | None,
) -> dict[str, Any]:
    out = dict(rec)
    extra = dict(out.get("extra") or {})
    meta: dict[str, Any] = {
        "source": str(silver_in),
        "n_qids": int(n_qids),
        "mode": mode,
    }
    if mode == "random":
        meta["seed"] = int(seed if seed is not None else 0)
    extra["label_pool_subset"] = meta
    out["extra"] = extra
    return out


def main() -> int:  # noqa: C901
    args = parse_args()
    if args.n_qids <= 0:
        print("[subset-silver][FATAL] --n-qids must be positive", file=sys.stderr)
        return 2
    if not args.silver_in.is_file():
        print(f"[subset-silver][FATAL] input not found: {args.silver_in}", file=sys.stderr)
        return 2

    if args.mode == "prefix" and args.seed not in (0, None):
        print("[subset-silver][WARN] --seed is ignored for --mode prefix", flush=True)

    out_path = args.out or _default_out_path(args.silver_in, args.n_qids, mode=args.mode, seed=args.seed)
    for path in (out_path, args.qid_out):
        if path is not None and path.exists() and not args.force:
            print(f"[subset-silver][FATAL] output exists, pass --force to overwrite: {path}", file=sys.stderr)
            return 2

    try:
        grouped = _read_grouped(args.silver_in)
    except ValueError as e:
        print(f"[subset-silver][FATAL] {e}", file=sys.stderr)
        return 2

    qids = list(grouped.keys())
    if args.n_qids > len(qids):
        print(
            f"[subset-silver][FATAL] requested {args.n_qids} qids but input only has {len(qids)}",
            file=sys.stderr,
        )
        return 2

    if args.mode == "prefix":
        selected_source_order = qids[: args.n_qids]
    else:
        selected = set(random.Random(args.seed).sample(qids, args.n_qids))
        selected_source_order = [qid for qid in qids if qid in selected]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_records = 0
    with open(out_path, "w", encoding="utf-8") as out_f:
        for qid in selected_source_order:
            for rec in grouped[qid]:
                out_f.write(
                    json.dumps(
                        _with_subset_provenance(
                            rec,
                            silver_in=args.silver_in,
                            n_qids=args.n_qids,
                            mode=args.mode,
                            seed=args.seed,
                        ),
                        ensure_ascii=False,
                    )
                )
                out_f.write("\n")
                n_records += 1

    if args.qid_out is not None:
        args.qid_out.parent.mkdir(parents=True, exist_ok=True)
        args.qid_out.write_text("".join(f"{qid}\n" for qid in selected_source_order), encoding="utf-8")

    print(f"[subset-silver] input   = {args.silver_in}", flush=True)
    print(f"[subset-silver] output  = {out_path}", flush=True)
    print(f"[subset-silver] qids    = {len(selected_source_order)} / {len(qids)}", flush=True)
    print(f"[subset-silver] records = {n_records}", flush=True)
    print(f"[subset-silver] mode    = {args.mode}", flush=True)
    if args.mode == "random":
        print(f"[subset-silver] seed    = {args.seed}", flush=True)
    if args.qid_out is not None:
        print(f"[subset-silver] qid_out = {args.qid_out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
