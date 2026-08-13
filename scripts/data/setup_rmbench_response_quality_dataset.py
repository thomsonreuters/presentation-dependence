#!/usr/bin/env python
"""Materialize RM-Bench as a response-quality FixtureLoader eval dataset.

RM-Bench (`THU-KEG/RM-Bench`, arXiv 2410.16184) stresses reward models on subtle
content differences and style robustness. Each sample has a `prompt`, three
`chosen` responses and three `rejected` responses at three style levels
(concise / detailed-plain / detailed-markdown), plus a `domain`
(chat / code / math / safety-refuse / safety-response). Stage-2 eval surface
for the response-ranking arm: the style-robustness surface.

Mapping onto the existing (query, candidate_docs, graded_labels) contract:

    prompt     -> query
    chosen[]   -> responses graded 3  (correct, 3 style variants)
    rejected[] -> responses graded 0  (incorrect, 3 style variants)

So N=6 per prompt. Keeping all six style variants co-present in one listwise set
makes the standard readout a style-robustness test: a style-biased RM would
rank a markdown-formatted *rejected* response above a concise *chosen* one, which
nDCG@1 / full-nDCG penalizes. Each passage also carries a `style` tag
(concise/detailed_plain/detailed_markdown) so the RM-Bench 3x3 hard/normal/easy
style-matrix analysis can be layered on later without re-materializing.

As for RewardBench-2, the chosen responses are listed first in the source, so
deterministically shuffle the response order per query (seeded by qid) for a
neutral input order: needed for honest single-pass nDCG and a clean
position/flip-rate read (the PSI harness re-shuffles across K perms separately).

Per-domain qid files (`qids_subset_<domain>.txt`) slice a single eval via
`qids_to_run_path`. Eval-only: one materialization, no train/heldout slices.

Usage:

    uv run python -m scripts.data.setup_rmbench_response_quality_dataset
    uv run python -m scripts.data.setup_rmbench_response_quality_dataset --src-jsonl sample.jsonl --out /tmp/rmb

Tests pass ``--src-jsonl`` with synthetic RM-Bench-shaped records so this script
stays covered offline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any

import yaml


DEFAULT_DATASET = "THU-KEG/RM-Bench"
DEFAULT_SPLIT = "train"
CHOSEN_GRADE = 3
REJECTED_GRADE = 0
STYLE_LEVELS = ("concise", "detailed_plain", "detailed_markdown")
RETRIEVER_TOKEN = "rmbench-shuffled-order"
SLM_ROOT = Path(__file__).resolve().parents[2]


def default_out_dir() -> Path:
    return Path("data") / "rmbench-response-quality"


def _slug_token(raw: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(raw).strip())
    return token.strip("-") or "qid"


def _prompt_text(record: dict[str, Any]) -> str:
    return " ".join(str(record.get("prompt") or record.get("instruction") or "").split())


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(v) for v in value]
    raise ValueError(f"Unsupported chosen/rejected shape: {type(value).__name__}")


def _record_qid(domain: str, domain_index: int) -> str:
    # RM-Bench ids are not guaranteed globally unique/stable; use a domain-scoped
    # counter, which is unique and human-readable (mirrors the RewardBench-2 adapter).
    return f"rmb-{_slug_token(domain)}-{domain_index:05d}"


def _qid_seed(qid: str) -> int:
    return int(hashlib.sha1(qid.encode("utf-8")).hexdigest()[:8], 16)


def _style_for(index_in_side: int) -> str:
    return STYLE_LEVELS[index_in_side] if index_in_side < len(STYLE_LEVELS) else f"variant_{index_in_side}"


def load_records_from_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_records_from_hf(dataset: str, split: str) -> list[dict[str, Any]]:
    from datasets import load_dataset

    return [dict(rec) for rec in load_dataset(dataset, split=split)]


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
                f.write(
                    f"{qid} Q0 {passage['pid']} {int(passage['rank'])} {float(passage['score']):.6f} {RETRIEVER_TOKEN}\n"
                )


def write_fixture(
    topics: dict[str, str], passages_by_qid: dict[str, list[dict[str, Any]]], path: Path, qids: list[str]
) -> None:
    with path.open("w", encoding="utf-8") as f:
        for qid in qids:
            rec = {"qid": qid, "query": topics[qid], "passages": passages_by_qid[qid]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def write_subset_qid_files(out_dir: Path, subset_to_qids: dict[str, list[str]]) -> dict[str, int]:
    written: dict[str, int] = {}
    for subset, qids in sorted(subset_to_qids.items()):
        path = out_dir / f"qids_subset_{_slug_token(subset)}.txt"
        path.write_text("".join(f"{qid}\n" for qid in qids), encoding="utf-8")
        written[path.name] = len(qids)
    return written


def materialize_records(  # noqa: C901
    records: list[dict[str, Any]],
    *,
    out_dir: Path,
    dataset: str,
    split: str,
    max_queries: int | None,
    required_responses: int | None,
    shuffle_responses: bool,
    force: bool,
    dry_run: bool,
) -> dict[str, Any]:
    if out_dir.exists() and any(out_dir.iterdir()) and not force and not dry_run:
        raise FileExistsError(f"{out_dir} already exists and is non-empty; pass --force to overwrite")

    topics: dict[str, str] = {}
    qrels: dict[str, dict[str, int]] = {}
    passages_by_qid: dict[str, list[dict[str, Any]]] = {}
    subset_to_qids: dict[str, list[str]] = defaultdict(list)
    grade_counts: Counter[int] = Counter()
    response_counts: Counter[int] = Counter()
    skipped_non_required = 0
    skipped_empty = 0
    per_domain_index: Counter[str] = Counter()

    for record in records:
        if max_queries is not None and len(topics) >= int(max_queries):
            break
        domain = str(record.get("domain") or record.get("subset") or "all")
        prompt = _prompt_text(record)
        chosen = _as_list(record.get("chosen"))
        rejected = _as_list(record.get("rejected"))
        if not prompt or not chosen:
            skipped_empty += 1
            continue
        per_domain_index[domain] += 1
        qid = _record_qid(domain, per_domain_index[domain])
        if qid in topics:
            raise ValueError(f"Duplicate qid after normalization: {qid!r}")
        # (text, grade, style): style level derived from position within each side
        graded = [(c, CHOSEN_GRADE, _style_for(i)) for i, c in enumerate(chosen)]
        graded += [(r, REJECTED_GRADE, _style_for(i)) for i, r in enumerate(rejected)]
        if required_responses is not None and len(graded) != int(required_responses):
            skipped_non_required += 1
            continue
        if any(not str(text).strip() for text, _, _ in graded):
            skipped_empty += 1
            continue
        if shuffle_responses:
            random.Random(_qid_seed(qid)).shuffle(graded)

        passages: list[dict[str, Any]] = []
        qrels[qid] = {}
        n = len(graded)
        for pos, (text, grade, style) in enumerate(graded, start=1):
            pid = f"{qid}-r{pos:02d}"
            body = " ".join(str(text).split()).strip()
            qrels[qid][pid] = grade
            grade_counts[grade] += 1
            passages.append(
                {
                    "pid": pid,
                    "text": body,
                    "gold_grade": grade,
                    "style": style,
                    "score": float(n - pos + 1),
                    "rank": pos,
                }
            )
        topics[qid] = prompt
        passages_by_qid[qid] = passages
        response_counts[n] += 1
        subset_to_qids[domain].append(qid)

    if not topics:
        raise ValueError("No RM-Bench records selected after filtering")

    qids = sorted(topics)
    run_name = "run.rmbench-response-quality_sorted.txt"
    qid_files: dict[str, int] = {}
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        write_topics(topics, out_dir / "topics.tsv", qids)
        write_qrels(qrels, out_dir / "qrels.txt", qids)
        write_run(passages_by_qid, out_dir / run_name, qids)
        write_fixture(topics, passages_by_qid, out_dir / "fixture.jsonl", qids)
        (out_dir / "qids_all.txt").write_text("".join(f"{q}\n" for q in qids), encoding="utf-8")
        qid_files["qids_all.txt"] = len(qids)
        qid_files.update(write_subset_qid_files(out_dir, subset_to_qids))
        meta = {
            "dataset": "rmbench-response-quality",
            "source": {"dataset": dataset, "split": split, "loader": "datasets.load_dataset"},
            "created_at": date.today().isoformat(),
            "access": "public",
            "task": "response-quality scoring (reward modeling) — RM-Bench robustness eval surface",
            "label_type": "graded",
            "qrels": {
                "semantics": f"chosen -> {CHOSEN_GRADE}, rejected -> {REJECTED_GRADE}; explicit zero rows kept",
                "grade_passages_per_grade": dict(sorted(grade_counts.items())),
                "explicit_zero_rows": True,
            },
            "candidate_set": {
                "source_order": "per-qid deterministic shuffle (gold not slot-1)"
                if shuffle_responses
                else "source order",
                "responses_per_query": dict(sorted(response_counts.items())),
                "style_levels": list(STYLE_LEVELS),
                "style_note": "each passage carries a `style` tag; all 6 style variants co-present -> "
                "listwise readout is a style-robustness test; RM-Bench 3x3 hard/normal/easy "
                "matrix can be layered on later from the style tags.",
                "run_file": run_name,
                "retriever": RETRIEVER_TOKEN,
            },
            "counts": {"queries": len(qids)},
            "subsets": {k: len(v) for k, v in sorted(subset_to_qids.items())},
            "filtering": {
                "required_responses_per_query": required_responses,
                "skipped_non_required_response_count": skipped_non_required,
                "skipped_empty_count": skipped_empty,
            },
            "qid_files": qid_files,
            "files": {"fixture": "fixture.jsonl", "topics": "topics.tsv", "qrels": "qrels.txt", "run": run_name},
        }
        (out_dir / "dataset_meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False), encoding="utf-8")

    return {
        "out_dir": str(out_dir),
        "n_queries": len(qids),
        "grade_counts": dict(sorted(grade_counts.items())),
        "response_counts": dict(sorted(response_counts.items())),
        "subsets": {k: len(v) for k, v in sorted(subset_to_qids.items())},
        "skipped_non_required_response_count": skipped_non_required,
        "skipped_empty_count": skipped_empty,
        "qid_files": qid_files,
        "dry_run": dry_run,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=DEFAULT_DATASET, help="HF dataset name.")
    p.add_argument("--split", default=DEFAULT_SPLIT, help="HF split.")
    p.add_argument("--src-jsonl", type=Path, default=None, help="Offline RM-Bench-shaped JSONL source.")
    p.add_argument("--out", type=Path, default=None, help="Output directory. Defaults under data/.")
    p.add_argument("--max-queries", type=int, default=None, help="Optional cap.")
    p.add_argument(
        "--required-responses",
        type=int,
        default=6,
        help="Require exactly this many responses (chosen+rejected) per prompt; 0 to disable.",
    )
    p.add_argument(
        "--no-shuffle-responses",
        dest="shuffle_responses",
        action="store_false",
        help="Keep source order (gold first). Default shuffles per qid for a neutral order.",
    )
    p.set_defaults(shuffle_responses=True)
    p.add_argument("--force", action="store_true", help="Overwrite a non-empty output directory.")
    p.add_argument("--dry-run", action="store_true", help="Validate and summarize without writing.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    try:
        out_dir = args.out or default_out_dir()
        if not out_dir.is_absolute():
            out_dir = SLM_ROOT / out_dir
        if args.src_jsonl:
            records = load_records_from_jsonl(args.src_jsonl)
            source_dataset = str(args.src_jsonl)
        else:
            records = load_records_from_hf(args.dataset, args.split)
            source_dataset = args.dataset
        summary = materialize_records(
            records,
            out_dir=out_dir,
            dataset=source_dataset,
            split=args.split,
            max_queries=args.max_queries,
            required_responses=(None if int(args.required_responses) <= 0 else int(args.required_responses)),
            shuffle_responses=bool(args.shuffle_responses),
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
