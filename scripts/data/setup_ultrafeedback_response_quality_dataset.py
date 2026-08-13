#!/usr/bin/env python
"""Materialize UltraFeedback as a response-quality-scoring FixtureLoader dataset.

The response-ranking / reward-modeling cross-task arm treats each
UltraFeedback prompt as the query and each of its N candidate completions as one
scored "passage". The per-response GPT-4 ``overall_score`` (1-10) is the gold
quality signal. It is exposed two ways so the existing eval/training contract
runs untouched:

    - graded qrels in {0, 1, 2, 3} via fixed score thresholds (for graded nDCG),
      with explicit rel=0 rows kept (so ``strict_qrels_filter`` sees judged-but-
      zero responses);
    - the raw ``overall_score`` stored per response in the fixture record
      (``gold_score``) for Kendall-tau-to-gold agreement metrics.

Mirrors ``scripts/data/setup_hotpotqa_support_dataset.py``: prompt -> query,
N responses -> passages, quality label -> qrels. The candidate order preserves
the source completion order (a neutral, non-gold order) so input-order
perturbations measure a real position effect.

Outputs match the Java-free eval/training contract:

    data/ultrafeedback-response-quality-<split>/
    ├── fixture.jsonl
    ├── topics.tsv
    ├── qrels.txt
    ├── run.ultrafeedback-response-quality-<split>_sorted.txt
    ├── qids_*.txt                # head prefixes (train)
    ├── qids_heldout_<H>.txt      # disjoint tail slice (lambda/checkpoint select)
    ├── qids_dev_<D>.txt          # disjoint tail slice (existence/eval)
    └── dataset_meta.yaml

UltraFeedback ships a single HF split, so train / heldout / dev are carved as
disjoint slices of one materialization (heldout + dev taken from the tail, train
prefixes from the head), the same convention the QA arm used for heldout500.

Usage:

    uv run python -m scripts.data.setup_ultrafeedback_response_quality_dataset --max-queries 30700
    uv run python -m scripts.data.setup_ultrafeedback_response_quality_dataset --src-jsonl sample.jsonl --out /tmp/uf

Tests pass ``--src-jsonl`` with synthetic UltraFeedback-shaped records so this
script stays covered offline; normal runs load ``openbmb/UltraFeedback`` via
``datasets``.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import yaml


DEFAULT_DATASET = "openbmb/UltraFeedback"
DEFAULT_CONFIG: str | None = None
DEFAULT_PREFIX_COUNTS = (100, 200, 1000, 3000, 10000, 30000)
DEFAULT_GRADE_THRESHOLDS = (4.0, 6.0, 8.0)  # <4->0, [4,6)->1, [6,8)->2, >=8->3
RETRIEVER_TOKEN = "ultrafeedback-completion-order"
REPO_ROOT = Path(__file__).resolve().parents[3]
SLM_ROOT = Path(__file__).resolve().parents[2]


def normalize_split(split: str) -> tuple[str, str]:
    """Return ``(hf_split, slug_split)`` for CLI split aliases."""
    raw = split.strip().lower()
    if raw in {"train", "all"}:
        return "train", "train"
    raise ValueError("split must be 'train' (UltraFeedback ships a single split)")


def default_out_dir(slug_split: str) -> Path:
    return Path("data") / f"ultrafeedback-response-quality-{slug_split}"


def _slug_token(raw: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "-", raw.strip())
    token = token.strip("-")
    return token or "qid"


def _prompt_text(record: dict[str, Any]) -> str:
    raw = record.get("instruction") or record.get("prompt") or record.get("question") or ""
    return " ".join(str(raw).split())


def _completion_pairs(record: dict[str, Any]) -> list[tuple[str, float]]:
    """Return ``[(response_text, overall_score), ...]`` from a UltraFeedback record."""
    completions = record.get("completions") or record.get("responses") or record.get("candidates") or []
    if not isinstance(completions, list):
        raise ValueError(f"Unsupported UltraFeedback completions shape: {type(completions).__name__}")
    out: list[tuple[str, float]] = []
    for item in completions:
        if not isinstance(item, dict):
            raise ValueError(f"Unsupported completion entry shape: {type(item).__name__}")
        text = item.get("response")
        if text is None:
            text = item.get("text") or item.get("completion") or ""
        score = item.get("overall_score")
        if score is None:
            score = item.get("score")
        if score is None:
            annotations = item.get("annotations")
            if isinstance(annotations, dict):
                score = annotations.get("overall_score")
        out.append((str(text), _coerce_score(score)))
    return out


def _coerce_score(score: Any) -> float:
    try:
        value = float(score)
    except (TypeError, ValueError):
        return float("nan")
    return value


def _record_qid(record: dict[str, Any], index: int) -> str:
    # NB: openbmb/UltraFeedback has no unique per-row id; its ``source`` field is
    # the subset name (evol_instruct, sharegpt, ...) and is NOT unique, so it must
    # not be used as a qid. Fall back to a stable source-position index.
    raw = record.get("id") or record.get("_id") or record.get("qid")
    if raw:
        return _slug_token(str(raw))
    return f"uf-{index:06d}"


def grade_from_score(score: float, thresholds: tuple[float, ...]) -> int:
    """Map a continuous quality score to an integer grade via ascending thresholds.

    ``grade`` is the count of thresholds the score meets or exceeds, so
    ``thresholds=(4, 6, 8)`` yields {0,1,2,3}.
    """
    grade = 0
    for cut in thresholds:
        if score >= cut:
            grade += 1
    return grade


def load_records_from_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def load_records_from_hf(dataset: str, config: str | None, split: str) -> list[dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset(dataset, config, split=split) if config else load_dataset(dataset, split=split)
    return [dict(rec) for rec in ds]


def deterministic_prefix_order(qids: Iterable[str], *, seed: int | None) -> list[str]:
    out = sorted(map(str, qids))
    if seed is not None:
        random.Random(int(seed)).shuffle(out)
    return out


def write_topics(topics: dict[str, str], path: Path, qids: list[str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for qid in qids:
            f.write(f"{qid}\t{topics[qid]}\n")


def write_qrels(qrels: dict[str, dict[str, int]], path: Path, qids: list[str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for qid in qids:
            for pid in sorted(qrels[qid]):
                f.write(f"{qid} Q0 {pid} {qrels[qid][pid]}\n")


def write_run(passages_by_qid: dict[str, list[dict[str, Any]]], path: Path, qids: list[str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for qid in qids:
            for passage in passages_by_qid[qid]:
                rank = int(passage["rank"])
                score = float(passage["score"])
                f.write(f"{qid} Q0 {passage['pid']} {rank} {score:.6f} {RETRIEVER_TOKEN}\n")


def write_fixture(
    topics: dict[str, str], passages_by_qid: dict[str, list[dict[str, Any]]], path: Path, qids: list[str]
) -> None:
    with path.open("w", encoding="utf-8") as f:
        for qid in qids:
            rec = {"qid": qid, "query": topics[qid], "passages": passages_by_qid[qid]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def qids_prefix_filename(n: int) -> str:
    return f"qids_{n // 1000}k.txt" if n >= 1000 and n % 1000 == 0 else f"qids_{n}.txt"


def write_qid_files(
    out_dir: Path,
    order: list[str],
    *,
    prefix_counts: Iterable[int],
    heldout_size: int,
    dev_size: int,
) -> dict[str, int]:
    """Write qids_all + head prefixes (train) + disjoint tail heldout/dev slices."""
    written: dict[str, int] = {}
    all_path = out_dir / "qids_all.txt"
    all_path.write_text("".join(f"{qid}\n" for qid in order), encoding="utf-8")
    written[all_path.name] = len(order)

    total = len(order)
    reserve = max(int(heldout_size), 0) + max(int(dev_size), 0)
    head_limit = total - reserve  # train prefixes must stay within the head

    for raw_count in prefix_counts:
        count = int(raw_count)
        if count <= 0 or count > head_limit:
            continue
        path = out_dir / qids_prefix_filename(count)
        path.write_text("".join(f"{qid}\n" for qid in order[:count]), encoding="utf-8")
        written[path.name] = count

    cursor = head_limit
    if heldout_size > 0 and total >= reserve:
        slice_qids = order[cursor : cursor + heldout_size]
        path = out_dir / f"qids_heldout_{heldout_size}.txt"
        path.write_text("".join(f"{qid}\n" for qid in slice_qids), encoding="utf-8")
        written[path.name] = len(slice_qids)
        cursor += heldout_size
    if dev_size > 0 and total >= reserve:
        slice_qids = order[cursor : cursor + dev_size]
        path = out_dir / f"qids_dev_{dev_size}.txt"
        path.write_text("".join(f"{qid}\n" for qid in slice_qids), encoding="utf-8")
        written[path.name] = len(slice_qids)
    return written


def materialize_records(  # noqa: C901
    records: list[dict[str, Any]],
    *,
    out_dir: Path,
    slug_split: str,
    hf_split: str,
    dataset: str,
    config: str | None,
    max_queries: int | None,
    required_responses: int | None,
    grade_thresholds: tuple[float, ...],
    prefix_seed: int | None,
    prefix_counts: Iterable[int],
    heldout_size: int,
    dev_size: int,
    force: bool,
    dry_run: bool,
) -> dict[str, Any]:
    if out_dir.exists() and any(out_dir.iterdir()) and not force and not dry_run:
        raise FileExistsError(f"{out_dir} already exists and is non-empty; pass --force to overwrite")

    topics: dict[str, str] = {}
    qrels: dict[str, dict[str, int]] = {}
    passages_by_qid: dict[str, list[dict[str, Any]]] = {}
    grade_counts: Counter[int] = Counter()
    response_counts: Counter[int] = Counter()
    skipped_non_required_response_count = 0
    skipped_bad_score_count = 0
    skipped_empty_prompt_count = 0

    for idx, record in enumerate(records):
        if max_queries is not None and len(topics) >= int(max_queries):
            break
        qid = _record_qid(record, idx)
        if qid in topics:
            raise ValueError(f"Duplicate qid after normalization: {qid!r}")
        prompt = _prompt_text(record)
        if not prompt:
            skipped_empty_prompt_count += 1
            continue
        completions = _completion_pairs(record)
        if required_responses is not None and len(completions) != int(required_responses):
            skipped_non_required_response_count += 1
            continue
        if any((not text.strip()) or (not math.isfinite(score)) for text, score in completions):
            skipped_bad_score_count += 1
            continue

        passages: list[dict[str, Any]] = []
        qrels[qid] = {}
        n = len(completions)
        for pos, (text, score) in enumerate(completions, start=1):
            pid = f"{_slug_token(qid)}-r{pos:02d}"
            body = " ".join(str(text).split()).strip()
            grade = grade_from_score(score, grade_thresholds)
            qrels[qid][pid] = grade
            grade_counts[grade] += 1
            passages.append(
                {
                    "pid": pid,
                    "text": body,
                    "gold_score": float(score),
                    "gold_grade": grade,
                    "score": float(n - pos + 1),
                    "rank": pos,
                }
            )
        topics[qid] = prompt
        passages_by_qid[qid] = passages
        response_counts[n] += 1

    if not topics:
        raise ValueError("No UltraFeedback records selected after filtering")
    if max_queries is not None and len(topics) < int(max_queries):
        raise ValueError(
            f"Only selected {len(topics)} records after filtering, fewer than requested max_queries={max_queries}"
        )

    order = deterministic_prefix_order(topics.keys(), seed=prefix_seed)
    qids = sorted(topics)

    run_name = f"run.ultrafeedback-response-quality-{slug_split}_sorted.txt"
    qid_files: dict[str, int] = {}
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        write_topics(topics, out_dir / "topics.tsv", qids)
        write_qrels(qrels, out_dir / "qrels.txt", qids)
        write_run(passages_by_qid, out_dir / run_name, qids)
        write_fixture(topics, passages_by_qid, out_dir / "fixture.jsonl", qids)
        qid_files = write_qid_files(
            out_dir,
            order,
            prefix_counts=prefix_counts,
            heldout_size=heldout_size,
            dev_size=dev_size,
        )
        write_dataset_meta(
            out_dir / "dataset_meta.yaml",
            dataset=dataset,
            config=config,
            hf_split=hf_split,
            slug_split=slug_split,
            n_queries=len(qids),
            run_name=run_name,
            grade_counts=dict(sorted(grade_counts.items())),
            response_counts=dict(sorted(response_counts.items())),
            required_responses=required_responses,
            grade_thresholds=grade_thresholds,
            skipped_non_required_response_count=skipped_non_required_response_count,
            skipped_bad_score_count=skipped_bad_score_count,
            skipped_empty_prompt_count=skipped_empty_prompt_count,
            prefix_seed=prefix_seed,
            heldout_size=heldout_size,
            dev_size=dev_size,
            qid_files=qid_files,
        )

    return {
        "out_dir": str(out_dir),
        "n_queries": len(qids),
        "grade_counts": dict(sorted(grade_counts.items())),
        "response_counts": dict(sorted(response_counts.items())),
        "skipped_non_required_response_count": skipped_non_required_response_count,
        "skipped_bad_score_count": skipped_bad_score_count,
        "skipped_empty_prompt_count": skipped_empty_prompt_count,
        "qid_files": qid_files,
        "dry_run": dry_run,
    }


def write_dataset_meta(
    path: Path,
    *,
    dataset: str,
    config: str | None,
    hf_split: str,
    slug_split: str,
    n_queries: int,
    run_name: str,
    grade_counts: dict[int, int],
    response_counts: dict[int, int],
    required_responses: int | None,
    grade_thresholds: tuple[float, ...],
    skipped_non_required_response_count: int,
    skipped_bad_score_count: int,
    skipped_empty_prompt_count: int,
    prefix_seed: int | None,
    heldout_size: int,
    dev_size: int,
    qid_files: dict[str, int],
) -> None:
    meta = {
        "dataset": f"ultrafeedback-response-quality-{slug_split}",
        "source": {
            "dataset": dataset,
            "config": config,
            "split": hf_split,
            "loader": "datasets.load_dataset",
        },
        "created_at": date.today().isoformat(),
        "access": "public",
        "task": "response-quality scoring (reward modeling)",
        "label_type": "graded",
        "qrels": {
            "semantics": (
                "integer quality grade in {0,1,2,3} from the per-response GPT-4 overall_score "
                "via ascending thresholds; explicit rel=0 rows kept"
            ),
            "grade_thresholds": list(grade_thresholds),
            "grade_passages_per_grade": grade_counts,
            "explicit_zero_rows": True,
            "gold_score_field": "passages[].gold_score (raw overall_score for Kendall-tau-to-gold)",
        },
        "candidate_set": {
            "source_order": "UltraFeedback completion order (neutral, non-gold)",
            "responses_per_query": response_counts,
            "run_file": run_name,
            "retriever": RETRIEVER_TOKEN,
        },
        "counts": {"queries": n_queries},
        "filtering": {
            "required_responses_per_query": required_responses,
            "skipped_non_required_response_count": skipped_non_required_response_count,
            "skipped_bad_score_count": skipped_bad_score_count,
            "skipped_empty_prompt_count": skipped_empty_prompt_count,
        },
        "splits": {
            "note": ("UltraFeedback ships one HF split; train = head prefixes, heldout + dev = disjoint tail slices."),
            "heldout_size": heldout_size,
            "dev_size": dev_size,
        },
        "qid_prefixes": {
            "seed": prefix_seed,
            "files": qid_files,
            "note": "Prefix files are deterministic over sorted qids; shuffled when seed is not null.",
        },
        "files": {
            "fixture": "fixture.jsonl",
            "topics": "topics.tsv",
            "qrels": "qrels.txt",
            "run": run_name,
        },
    }
    path.write_text(yaml.safe_dump(meta, sort_keys=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="train", help="UltraFeedback split (only 'train' is shipped).")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="HF dataset name.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="HF dataset config (default: none).")
    parser.add_argument("--src-jsonl", type=Path, default=None, help="Offline UltraFeedback-shaped JSONL source.")
    parser.add_argument("--out", type=Path, default=None, help="Output directory. Defaults under data/.")
    parser.add_argument("--max-queries", type=int, default=None, help="Optional prefix limit before writing outputs.")
    parser.add_argument(
        "--required-responses",
        type=int,
        default=4,
        help="Require exactly this many candidate responses per prompt; use 0 to disable.",
    )
    parser.add_argument(
        "--grade-thresholds",
        type=str,
        default=",".join(str(t) for t in DEFAULT_GRADE_THRESHOLDS),
        help="Ascending overall_score cut points mapping to grades {0,1,2,3}.",
    )
    parser.add_argument("--heldout-size", type=int, default=500, help="Disjoint tail slice for lambda/ckpt selection.")
    parser.add_argument("--dev-size", type=int, default=200, help="Disjoint tail slice for the existence/eval set.")
    parser.add_argument("--prefix-seed", type=int, default=42, help="Seed for qid order; use -1 to keep sorted.")
    parser.add_argument(
        "--prefix-counts",
        type=str,
        default=",".join(str(n) for n in DEFAULT_PREFIX_COUNTS),
        help="Comma-separated head qid prefix sizes to write when available.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite a non-empty output directory.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and summarize without writing files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        hf_split, slug_split = normalize_split(args.split)
        out_dir = args.out or default_out_dir(slug_split)
        if not out_dir.is_absolute():
            out_dir = SLM_ROOT / out_dir
        prefix_seed = None if int(args.prefix_seed) < 0 else int(args.prefix_seed)
        prefix_counts = [int(tok) for tok in str(args.prefix_counts).split(",") if tok.strip()]
        grade_thresholds = tuple(float(tok) for tok in str(args.grade_thresholds).split(",") if tok.strip())
        if not grade_thresholds or list(grade_thresholds) != sorted(grade_thresholds):
            raise ValueError("--grade-thresholds must be a non-empty ascending list")
        if args.src_jsonl:
            records = load_records_from_jsonl(args.src_jsonl)
            source_dataset = str(args.src_jsonl)
            source_config = "jsonl"
        else:
            records = load_records_from_hf(args.dataset, args.config, hf_split)
            source_dataset = args.dataset
            source_config = args.config
        summary = materialize_records(
            records,
            out_dir=out_dir,
            slug_split=slug_split,
            hf_split=hf_split,
            dataset=source_dataset,
            config=source_config,
            max_queries=args.max_queries,
            required_responses=(None if int(args.required_responses) <= 0 else int(args.required_responses)),
            grade_thresholds=grade_thresholds,
            prefix_seed=prefix_seed,
            prefix_counts=prefix_counts,
            heldout_size=int(args.heldout_size),
            dev_size=int(args.dev_size),
            force=bool(args.force),
            dry_run=bool(args.dry_run),
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
