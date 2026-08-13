#!/usr/bin/env python
"""Materialize HotpotQA distractor as a support-scoring FixtureLoader dataset.

Each HotpotQA question is the query, and each of the 10 distractor-setting
paragraphs is one candidate passage. Sentence-level
supporting facts are aggregated to binary passage-level qrels:

    rel = 1  if the paragraph title has at least one supporting fact
    rel = 0  otherwise

Outputs match the existing Java-free eval/training contract:

    data/hotpotqa-distractor-support-<split>/
    ├── fixture.jsonl
    ├── topics.tsv
    ├── qrels.txt
    ├── run.hotpotqa-distractor-support-<split>_sorted.txt
    ├── qids_*.txt
    └── dataset_meta.yaml

Usage:

    uv run python -m scripts.data.setup_hotpotqa_support_dataset --split dev --max-queries 200
    uv run python -m scripts.data.setup_hotpotqa_support_dataset --split train --max-queries 3000

Tests pass ``--src-jsonl`` with synthetic HotpotQA-shaped records so this script
stays covered offline; normal runs load ``hotpot_qa/distractor`` via
``datasets``.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import yaml


DEFAULT_DATASET = "hotpot_qa"
DEFAULT_CONFIG = "distractor"
DEFAULT_PREFIX_COUNTS = (100, 200, 1000, 3000, 10000, 30000)
RETRIEVER_TOKEN = "hotpotqa-distractor-order"
REPO_ROOT = Path(__file__).resolve().parents[3]
SLM_ROOT = Path(__file__).resolve().parents[2]


def normalize_split(split: str) -> tuple[str, str]:
    """Return ``(hf_split, slug_split)`` for CLI split aliases."""
    raw = split.strip().lower()
    if raw in {"dev", "valid", "validation"}:
        return "validation", "dev"
    if raw == "train":
        return "train", "train"
    raise ValueError("split must be one of: train, dev, validation")


def default_out_dir(slug_split: str) -> Path:
    return Path("data") / f"hotpotqa-distractor-support-{slug_split}"


def _norm_title(title: Any) -> str:
    return " ".join(str(title).split()).casefold()


def _slug_token(raw: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "-", raw.strip())
    token = token.strip("-")
    return token or "qid"


def _context_pairs(context: Any) -> list[tuple[str, list[str]]]:
    """Return ``[(title, sentences), ...]`` from HF or JSON-native context."""
    if isinstance(context, dict):
        titles = context.get("title") or context.get("titles") or []
        sentences = context.get("sentences") or []
        return [(str(t), [str(s) for s in ss]) for t, ss in zip(titles, sentences, strict=False)]
    if isinstance(context, list):
        out: list[tuple[str, list[str]]] = []
        for item in context:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                out.append((str(item[0]), [str(s) for s in item[1]]))
            elif isinstance(item, dict):
                title = item.get("title") or item.get("name") or ""
                sentences = item.get("sentences") or item.get("text") or []
                if isinstance(sentences, str):
                    sentences = [sentences]
                out.append((str(title), [str(s) for s in sentences]))
        return out
    raise ValueError(f"Unsupported HotpotQA context shape: {type(context).__name__}")


def _support_titles(record: dict[str, Any]) -> set[str]:
    sf = record.get("supporting_facts") or record.get("supporting_facts_by_title") or {}
    if isinstance(sf, dict):
        titles = sf.get("title") or sf.get("titles") or []
        return {_norm_title(title) for title in titles}
    if isinstance(sf, list):
        titles: set[str] = set()
        for item in sf:
            if isinstance(item, (list, tuple)) and item:
                titles.add(_norm_title(item[0]))
            elif isinstance(item, dict) and "title" in item:
                titles.add(_norm_title(item["title"]))
        return titles
    raise ValueError(f"Unsupported HotpotQA supporting_facts shape: {type(sf).__name__}")


def _record_qid(record: dict[str, Any], index: int) -> str:
    return str(record.get("id") or record.get("_id") or record.get("qid") or f"hotpot-{index:06d}")


def load_records_from_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def load_records_from_hf(dataset: str, config: str, split: str) -> list[dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset(dataset, config, split=split)
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


def write_qid_files(out_dir: Path, qids: list[str], prefix_counts: Iterable[int]) -> dict[str, int]:
    written: dict[str, int] = {}
    all_path = out_dir / "qids_all.txt"
    all_path.write_text("".join(f"{qid}\n" for qid in qids), encoding="utf-8")
    written[all_path.name] = len(qids)
    for raw_count in prefix_counts:
        count = int(raw_count)
        if count <= 0 or count > len(qids):
            continue
        path = out_dir / qids_prefix_filename(count)
        path.write_text("".join(f"{qid}\n" for qid in qids[:count]), encoding="utf-8")
        written[path.name] = count
    return written


def materialize_records(  # noqa: C901
    records: list[dict[str, Any]],
    *,
    out_dir: Path,
    slug_split: str,
    hf_split: str,
    dataset: str,
    config: str,
    max_queries: int | None,
    required_passages: int | None,
    required_supports: int | None,
    prefix_seed: int | None,
    prefix_counts: Iterable[int],
    force: bool,
    dry_run: bool,
) -> dict[str, Any]:
    if out_dir.exists() and any(out_dir.iterdir()) and not force and not dry_run:
        raise FileExistsError(f"{out_dir} already exists and is non-empty; pass --force to overwrite")

    topics: dict[str, str] = {}
    qrels: dict[str, dict[str, int]] = {}
    passages_by_qid: dict[str, list[dict[str, Any]]] = {}
    support_counts: Counter[int] = Counter()
    paragraph_counts: Counter[int] = Counter()
    skipped_non_required_passage_count = 0
    skipped_non_required_support_count = 0

    for idx, record in enumerate(records):
        if max_queries is not None and len(topics) >= int(max_queries):
            break
        qid = _record_qid(record, idx)
        if qid in topics:
            raise ValueError(f"Duplicate qid after normalization: {qid!r}")
        question = " ".join(str(record.get("question") or "").split())
        if not question:
            raise ValueError(f"qid={qid}: empty question")
        context = _context_pairs(record.get("context"))
        if not context:
            raise ValueError(f"qid={qid}: empty context")
        if required_passages is not None and len(context) != int(required_passages):
            skipped_non_required_passage_count += 1
            continue
        supports = _support_titles(record)

        passages: list[dict[str, Any]] = []
        qrels[qid] = {}
        for pos, (title, sentences) in enumerate(context, start=1):
            pid = f"{_slug_token(qid)}-p{pos:02d}"
            body = " ".join(" ".join(str(s).split()) for s in sentences).strip()
            text = f"{title}\n{body}".strip()
            rel = 1 if _norm_title(title) in supports else 0
            qrels[qid][pid] = rel
            passages.append(
                {
                    "pid": pid,
                    "title": title,
                    "text": text,
                    "score": float(len(context) - pos + 1),
                    "rank": pos,
                }
            )
        topics[qid] = question
        passages_by_qid[qid] = passages
        n_support = sum(qrels[qid].values())
        if required_supports is not None and n_support != int(required_supports):
            topics.pop(qid, None)
            passages_by_qid.pop(qid, None)
            qrels.pop(qid, None)
            skipped_non_required_support_count += 1
            continue
        support_counts[n_support] += 1
        paragraph_counts[len(passages)] += 1

    if not topics:
        raise ValueError("No HotpotQA records selected after filtering")
    if max_queries is not None and len(topics) < int(max_queries):
        raise ValueError(
            f"Only selected {len(topics)} records after filtering, fewer than requested max_queries={max_queries}"
        )

    prefix_qids = deterministic_prefix_order(topics.keys(), seed=prefix_seed)
    qids = sorted(topics)

    run_name = f"run.hotpotqa-distractor-support-{slug_split}_sorted.txt"
    qid_files: dict[str, int] = {}
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        write_topics(topics, out_dir / "topics.tsv", qids)
        write_qrels(qrels, out_dir / "qrels.txt", qids)
        write_run(passages_by_qid, out_dir / run_name, qids)
        write_fixture(topics, passages_by_qid, out_dir / "fixture.jsonl", qids)
        qid_files = write_qid_files(out_dir, prefix_qids, prefix_counts)
        write_dataset_meta(
            out_dir / "dataset_meta.yaml",
            dataset=dataset,
            config=config,
            hf_split=hf_split,
            slug_split=slug_split,
            n_queries=len(qids),
            run_name=run_name,
            support_counts=dict(sorted(support_counts.items())),
            paragraph_counts=dict(sorted(paragraph_counts.items())),
            required_passages=required_passages,
            required_supports=required_supports,
            skipped_non_required_passage_count=skipped_non_required_passage_count,
            skipped_non_required_support_count=skipped_non_required_support_count,
            prefix_seed=prefix_seed,
            qid_files=qid_files,
        )

    return {
        "out_dir": str(out_dir),
        "n_queries": len(qids),
        "support_counts": dict(sorted(support_counts.items())),
        "paragraph_counts": dict(sorted(paragraph_counts.items())),
        "skipped_non_required_passage_count": skipped_non_required_passage_count,
        "skipped_non_required_support_count": skipped_non_required_support_count,
        "qid_files": qid_files,
        "dry_run": dry_run,
    }


def write_dataset_meta(
    path: Path,
    *,
    dataset: str,
    config: str,
    hf_split: str,
    slug_split: str,
    n_queries: int,
    run_name: str,
    support_counts: dict[int, int],
    paragraph_counts: dict[int, int],
    required_passages: int | None,
    required_supports: int | None,
    skipped_non_required_passage_count: int,
    skipped_non_required_support_count: int,
    prefix_seed: int | None,
    qid_files: dict[str, int],
) -> None:
    meta = {
        "dataset": f"hotpotqa-distractor-support-{slug_split}",
        "source": {
            "dataset": dataset,
            "config": config,
            "split": hf_split,
            "loader": "datasets.load_dataset",
        },
        "created_at": date.today().isoformat(),
        "access": "public",
        "task": "multi-doc QA passage support scoring",
        "label_type": "binary",
        "qrels": {
            "semantics": "rel=1 iff the paragraph title has at least one HotpotQA supporting fact; rel=0 otherwise",
            "support_passages_per_query": support_counts,
            "explicit_zero_rows": True,
        },
        "candidate_set": {
            "source_order": "HotpotQA distractor context order",
            "expected_passages_per_query": 10,
            "paragraphs_per_query": paragraph_counts,
            "run_file": run_name,
            "retriever": RETRIEVER_TOKEN,
        },
        "counts": {"queries": n_queries},
        "filtering": {
            "required_passages_per_query": required_passages,
            "required_support_passages_per_query": required_supports,
            "skipped_non_required_passage_count": skipped_non_required_passage_count,
            "skipped_non_required_support_count": skipped_non_required_support_count,
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="dev", help="HotpotQA split: train, dev, or validation.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="HF dataset name.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="HF dataset config.")
    parser.add_argument("--src-jsonl", type=Path, default=None, help="Offline HotpotQA-shaped JSONL source.")
    parser.add_argument("--out", type=Path, default=None, help="Output directory. Defaults under data/.")
    parser.add_argument("--max-queries", type=int, default=None, help="Optional prefix limit before writing outputs.")
    parser.add_argument(
        "--required-passages",
        type=int,
        default=10,
        help="Require exactly this many context passages per question; use 0 to disable.",
    )
    parser.add_argument(
        "--required-supports",
        type=int,
        default=2,
        help="Require exactly this many positive support passages per question; use 0 to disable.",
    )
    parser.add_argument("--prefix-seed", type=int, default=42, help="Seed for qid prefix files; use -1 to keep sorted.")
    parser.add_argument(
        "--prefix-counts",
        type=str,
        default=",".join(str(n) for n in DEFAULT_PREFIX_COUNTS),
        help="Comma-separated qid prefix sizes to write when available.",
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
            required_passages=(None if int(args.required_passages) <= 0 else int(args.required_passages)),
            required_supports=(None if int(args.required_supports) <= 0 else int(args.required_supports)),
            prefix_seed=prefix_seed,
            prefix_counts=prefix_counts,
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
