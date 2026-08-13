#!/usr/bin/env python
r"""Derive random-label and shuffled-label negative-control silver JSONL.

The controls keep the exact student SFT input pipeline and silver-label schema,
but break the document -> relevance target mapping:

* ``random-valid`` samples each target from the empirical label distribution of
  the input JSONL.
* ``shuffled-within-query`` permutes real labels across docs within each query,
  preserving the query-local label histogram.

Example:
    uv run python scripts/data/derive_negative_control_silver.py \\
        --silver-in data/msmarco-train-selfdistill-seed42/silver_labels_a1_k1_seed0_train_29500.jsonl \\
        --mode shuffled-within-query
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


VALID_MODES = ("random-valid", "shuffled-within-query")


@dataclass(frozen=True, slots=True)
class LabelPayload:
    score: float
    raw_vector: tuple[float, ...]


def _default_out_path(silver_in: Path, mode: str, seed: int) -> Path:
    token = {
        "random-valid": "random_labels",
        "shuffled-within-query": "shuffled_labels",
    }[mode]
    insert = f"_{token}_seed{seed}"
    stem = silver_in.stem
    for split_token in ("_train_", "_heldout_"):
        if split_token in stem:
            return silver_in.with_name(f"{stem.replace(split_token, insert + split_token, 1)}{silver_in.suffix}")
    return silver_in.with_name(f"{stem}{insert}{silver_in.suffix}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--silver-in", required=True, type=Path, help="Input silver JSONL.")
    p.add_argument("--out", type=Path, default=None, help="Output JSONL. Defaults to inserting the control token.")
    p.add_argument("--mode", required=True, choices=VALID_MODES, help="Negative-control label assignment mode.")
    p.add_argument("--seed", type=int, default=42, help="Deterministic RNG seed. Default: 42.")
    p.add_argument("--force", action="store_true", help="Overwrite --out if it already exists.")
    return p.parse_args()


def _iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
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
            yield line_no, rec


def _payload_from_record(rec: dict[str, Any], *, line_no: int) -> LabelPayload:
    try:
        score = float(rec["score_continuous"])
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"line {line_no}: score_continuous must be numeric") from e
    if not math.isfinite(score):
        raise ValueError(f"line {line_no}: score_continuous must be finite")

    raw = rec.get("score_raw_vector")
    if raw is None:
        raw_values = (score,)
    elif isinstance(raw, list) and raw:
        try:
            raw_values = tuple(float(value) for value in raw)
        except (TypeError, ValueError) as e:
            raise ValueError(f"line {line_no}: score_raw_vector contains a non-numeric value") from e
        if any(not math.isfinite(value) for value in raw_values):
            raise ValueError(f"line {line_no}: score_raw_vector contains a non-finite value")
    else:
        raise ValueError(f"line {line_no}: score_raw_vector must be null or a non-empty list")
    return LabelPayload(score=score, raw_vector=raw_values)


def _with_payload(
    rec: dict[str, Any],
    payload: LabelPayload,
    *,
    mode: str,
    seed: int,
) -> dict[str, Any]:
    out = dict(rec)
    out["score_continuous"] = payload.score
    out["score_raw_vector"] = list(payload.raw_vector)
    out["k_perms"] = len(payload.raw_vector)

    extra_raw = out.get("extra") or {}
    extra = dict(extra_raw) if isinstance(extra_raw, dict) else {}
    extra["negative_control"] = {
        "mode": mode,
        "seed": seed,
        "label_source": "empirical_input_distribution",
    }
    out["extra"] = extra
    return out


def _write_random_valid(silver_in: Path, out_path: Path, *, seed: int) -> tuple[int, int, float, float, float]:
    payloads = [_payload_from_record(rec, line_no=line_no) for line_no, rec in _iter_jsonl(silver_in)]
    if not payloads:
        raise ValueError(f"no records found in {silver_in}")

    rng = random.Random(seed)
    n_records = 0
    qids: set[str] = set()
    score_sum = 0.0
    score_min = float("inf")
    score_max = float("-inf")
    with open(out_path, "w", encoding="utf-8") as out_f:
        for _line_no, rec in _iter_jsonl(silver_in):
            payload = rng.choice(payloads)
            converted = _with_payload(rec, payload, mode="random-valid", seed=seed)
            score = float(converted["score_continuous"])
            qids.add(str(converted["query_id"]))
            score_sum += score
            score_min = min(score_min, score)
            score_max = max(score_max, score)
            out_f.write(json.dumps(converted, ensure_ascii=False))
            out_f.write("\n")
            n_records += 1
    return n_records, len(qids), score_sum / n_records, score_min, score_max


def _deranged_indices(n: int, rng: random.Random) -> list[int]:
    if n <= 1:
        return list(range(n))
    order = list(range(n))
    for _attempt in range(100):
        rng.shuffle(order)
        if all(new_idx != old_idx for new_idx, old_idx in enumerate(order)):
            return order
    # Deterministic fallback that guarantees no fixed point for n > 1.
    return list(range(1, n)) + [0]


def _write_group(
    out_f,
    group: list[tuple[int, dict[str, Any]]],
    *,
    rng: random.Random,
    seed: int,
) -> tuple[int, float, float, float]:
    order = _deranged_indices(len(group), rng)
    payloads = [_payload_from_record(rec, line_no=line_no) for line_no, rec in group]
    score_sum = 0.0
    score_min = float("inf")
    score_max = float("-inf")
    for rec_idx, (_line_no, rec) in enumerate(group):
        converted = _with_payload(rec, payloads[order[rec_idx]], mode="shuffled-within-query", seed=seed)
        score = float(converted["score_continuous"])
        score_sum += score
        score_min = min(score_min, score)
        score_max = max(score_max, score)
        out_f.write(json.dumps(converted, ensure_ascii=False))
        out_f.write("\n")
    return len(group), score_sum, score_min, score_max


def _write_shuffled_within_query(silver_in: Path, out_path: Path, *, seed: int) -> tuple[int, int, float, float, float]:
    rng = random.Random(seed)
    n_records = 0
    qids: set[str] = set()
    closed_qids: set[str] = set()
    score_sum = 0.0
    score_min = float("inf")
    score_max = float("-inf")
    current_qid: str | None = None
    group: list[tuple[int, dict[str, Any]]] = []

    with open(out_path, "w", encoding="utf-8") as out_f:
        for line_no, rec in _iter_jsonl(silver_in):
            qid = str(rec.get("query_id"))
            if current_qid is None:
                current_qid = qid
            if qid != current_qid:
                closed_qids.add(current_qid)
                if qid in closed_qids:
                    raise ValueError(
                        f"line {line_no}: query_id {qid!r} reappeared after another query; "
                        "input must be grouped by query_id for shuffled-within-query"
                    )
                count, subtotal, group_min, group_max = _write_group(out_f, group, rng=rng, seed=seed)
                n_records += count
                score_sum += subtotal
                score_min = min(score_min, group_min)
                score_max = max(score_max, group_max)
                qids.add(current_qid)
                current_qid = qid
                group = []
            group.append((line_no, rec))

        if group and current_qid is not None:
            count, subtotal, group_min, group_max = _write_group(out_f, group, rng=rng, seed=seed)
            n_records += count
            score_sum += subtotal
            score_min = min(score_min, group_min)
            score_max = max(score_max, group_max)
            qids.add(current_qid)

    if n_records == 0:
        raise ValueError(f"no records found in {silver_in}")
    return n_records, len(qids), score_sum / n_records, score_min, score_max


def main() -> int:
    args = parse_args()
    if not args.silver_in.is_file():
        print(f"[neg-silver][FATAL] input not found: {args.silver_in}", file=sys.stderr)
        return 2
    out_path = args.out or _default_out_path(args.silver_in, args.mode, args.seed)
    if out_path.exists() and not args.force:
        print(f"[neg-silver][FATAL] output exists, pass --force to overwrite: {out_path}", file=sys.stderr)
        return 2
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if args.mode == "random-valid":
            stats = _write_random_valid(args.silver_in, out_path, seed=args.seed)
        elif args.mode == "shuffled-within-query":
            stats = _write_shuffled_within_query(args.silver_in, out_path, seed=args.seed)
        else:  # argparse enforces choices; keep this for type-checker clarity.
            raise ValueError(f"unknown mode: {args.mode}")
    except ValueError as e:
        print(f"[neg-silver][FATAL] {e}", file=sys.stderr)
        return 2

    n_records, n_qids, score_mean, score_min, score_max = stats
    print(f"[neg-silver] input       = {args.silver_in}", flush=True)
    print(f"[neg-silver] output      = {out_path}", flush=True)
    print(f"[neg-silver] mode        = {args.mode}", flush=True)
    print(f"[neg-silver] seed        = {args.seed}", flush=True)
    print(f"[neg-silver] records     = {n_records}", flush=True)
    print(f"[neg-silver] qids        = {n_qids}", flush=True)
    print(
        f"[neg-silver] score stats = mean {score_mean:.6f} min {score_min:.6f} max {score_max:.6f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
