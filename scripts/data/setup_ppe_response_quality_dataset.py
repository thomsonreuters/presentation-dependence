#!/usr/bin/env python
"""Materialize PPE Best-of-K (correctness) as an N-way response-quality eval set.

PPE (Preference Proxy Evaluations, lmarena-ai) best-of-K correctness sets (e.g.
`lmarena-ai/PPE-MMLU-Pro-Best-of-K`) give, per prompt, 32 same-model sampled
responses each with a verifiable binary correctness label (`scores`) and a
domain `category`. Second external N>=4 quality surface for the response-ranking
arm: tests whether the RM can pick a correct response among same-distribution
samples. Never trained on; fully held-out.

To get a non-degenerate, non-saturated, fixed-ish N per prompt we take up to
`--max-correct` correct + up to `--max-incorrect` incorrect responses (requiring
>=1 of each, dropping all-correct / all-incorrect prompts), graded:

    correct   -> grade 3
    incorrect -> grade 0

Per-qid deterministic shuffle neutralizes order. `category` -> subdomain slice
(via the qid prefix). Eval-only.

Usage:
    uv run python -m scripts.data.setup_ppe_response_quality_dataset
    uv run python -m scripts.data.setup_ppe_response_quality_dataset --src-jsonl sample.jsonl --out /tmp/ppe
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

DEFAULT_DATASET = "lmarena-ai/PPE-MMLU-Pro-Best-of-K"
DEFAULT_SPLIT = "train"
RETRIEVER_TOKEN = "ppe-bestofk-shuffled"
SLM_ROOT = Path(__file__).resolve().parents[2]


def default_out_dir() -> Path:
    return Path("data") / "ppe-mmlu-pro-response-quality"


def _slug(raw: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", str(raw)) or "all"


def _qid_seed(qid: str) -> int:
    return int(hashlib.sha1(qid.encode("utf-8")).hexdigest()[:8], 16)


def _responses_and_scores(record: dict[str, Any]) -> tuple[list[str], list[bool]]:
    scores = record.get("scores")
    if not isinstance(scores, list):
        raise ValueError("record missing `scores` list")
    # response_1..response_N columns
    resp = []
    for i in range(1, len(scores) + 1):
        resp.append(str(record.get(f"response_{i}", "")))
    return resp, [bool(s) for s in scores]


def load_records_from_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_records_from_hf(dataset: str, split: str) -> list[dict[str, Any]]:
    from datasets import load_dataset

    return [dict(r) for r in load_dataset(dataset, split=split)]


def materialize(records, *, out_dir, dataset, split, max_correct, max_incorrect, shuffle, force, dry_run):  # noqa: C901
    if out_dir.exists() and any(out_dir.iterdir()) and not force and not dry_run:
        raise FileExistsError(f"{out_dir} exists and is non-empty; pass --force")
    topics, qrels, passages_by_qid = {}, {}, {}
    subset_to_qids: dict[str, list[str]] = defaultdict(list)
    grade_counts: Counter[int] = Counter()
    n_resp_counts: Counter[int] = Counter()
    per_cat_index: Counter[str] = Counter()
    skipped = 0
    for idx, rec in enumerate(records):
        # MMLU-Pro: question/category; MATH: problem/type(+level)
        cat = _slug(rec.get("category") or rec.get("type") or "all")
        question = " ".join(str(rec.get("question") or rec.get("problem") or rec.get("prompt") or "").split())
        try:
            resp, scores = _responses_and_scores(rec)
        except ValueError:
            skipped += 1
            continue
        correct = [i for i, s in enumerate(scores) if s and resp[i].strip()]
        incorrect = [i for i, s in enumerate(scores) if not s and resp[i].strip()]
        if not question or not correct or not incorrect:
            skipped += 1
            continue
        rng = random.Random(_qid_seed(f"ppe-{idx}"))
        sel_c = rng.sample(correct, min(max_correct, len(correct)))
        sel_i = rng.sample(incorrect, min(max_incorrect, len(incorrect)))
        graded = [(resp[i], 3) for i in sel_c] + [(resp[i], 0) for i in sel_i]
        per_cat_index[cat] += 1
        qid = f"ppe-{cat}-{per_cat_index[cat]:05d}"
        if shuffle:
            rng.shuffle(graded)
        passages, qrels[qid] = [], {}
        n = len(graded)
        for pos, (text, grade) in enumerate(graded, start=1):
            pid = f"{qid}-r{pos:02d}"
            qrels[qid][pid] = grade
            grade_counts[grade] += 1
            passages.append(
                {
                    "pid": pid,
                    "text": " ".join(text.split()).strip(),
                    "gold_grade": grade,
                    "score": float(n - pos + 1),
                    "rank": pos,
                }
            )
        topics[qid] = question
        passages_by_qid[qid] = passages
        n_resp_counts[n] += 1
        subset_to_qids[cat].append(qid)
    if not topics:
        raise ValueError("No PPE records selected after filtering")

    qids = sorted(topics)
    run_name = "run.ppe-mmlu-pro-response-quality_sorted.txt"
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "topics.tsv").write_text("".join(f"{q}\t{topics[q]}\n" for q in qids), encoding="utf-8")
        with (out_dir / "qrels.txt").open("w", encoding="utf-8") as f:
            for q in qids:
                for pid in sorted(qrels[q]):
                    f.write(f"{q} Q0 {pid} {qrels[q][pid]}\n")
        with (out_dir / run_name).open("w", encoding="utf-8") as f:
            for q in qids:
                for p in passages_by_qid[q]:
                    f.write(f"{q} Q0 {p['pid']} {int(p['rank'])} {float(p['score']):.6f} {RETRIEVER_TOKEN}\n")
        with (out_dir / "fixture.jsonl").open("w", encoding="utf-8") as f:
            for q in qids:
                f.write(
                    json.dumps({"qid": q, "query": topics[q], "passages": passages_by_qid[q]}, ensure_ascii=False)
                    + "\n"
                )
        (out_dir / "qids_all.txt").write_text("".join(f"{q}\n" for q in qids), encoding="utf-8")
        for sub, sq in sorted(subset_to_qids.items()):
            (out_dir / f"qids_subset_{sub}.txt").write_text("".join(f"{q}\n" for q in sq), encoding="utf-8")
        meta = {
            "dataset": "ppe-mmlu-pro-response-quality",
            "source": {"dataset": dataset, "split": split, "loader": "datasets.load_dataset"},
            "created_at": date.today().isoformat(),
            "access": "public",
            "task": "response-quality scoring (reward modeling) — PPE best-of-K correctness eval surface",
            "label_type": "graded",
            "qrels": {
                "semantics": "verifiable correct -> 3, incorrect -> 0; balanced subset per prompt",
                "grade_passages_per_grade": dict(sorted(grade_counts.items())),
                "explicit_zero_rows": True,
            },
            "candidate_set": {
                "selection": f"up to {max_correct} correct + {max_incorrect} incorrect per prompt",
                "n_responses": dict(sorted(n_resp_counts.items())),
                "source_order": "per-qid deterministic shuffle" if shuffle else "source order",
                "run_file": run_name,
                "retriever": RETRIEVER_TOKEN,
            },
            "counts": {"queries": len(qids)},
            "subsets": {k: len(v) for k, v in sorted(subset_to_qids.items())},
            "filtering": {"skipped": skipped},
            "files": {"fixture": "fixture.jsonl", "topics": "topics.tsv", "qrels": "qrels.txt", "run": run_name},
        }
        (out_dir / "dataset_meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False), encoding="utf-8")
    return {
        "out_dir": str(out_dir),
        "n_queries": len(qids),
        "grade_counts": dict(sorted(grade_counts.items())),
        "n_responses": dict(sorted(n_resp_counts.items())),
        "subsets": {k: len(v) for k, v in sorted(subset_to_qids.items())},
        "skipped": skipped,
        "dry_run": dry_run,
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--split", default=DEFAULT_SPLIT)
    p.add_argument("--src-jsonl", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--max-correct", type=int, default=4)
    p.add_argument("--max-incorrect", type=int, default=4)
    p.add_argument("--no-shuffle-responses", dest="shuffle_responses", action="store_false")
    p.set_defaults(shuffle_responses=True)
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main():
    a = parse_args()
    try:
        out_dir = a.out or default_out_dir()
        if not out_dir.is_absolute():
            out_dir = SLM_ROOT / out_dir
        if a.src_jsonl:
            records = load_records_from_jsonl(a.src_jsonl)
            ds = str(a.src_jsonl)
        else:
            records = load_records_from_hf(a.dataset, a.split)
            ds = a.dataset
        summary = materialize(
            records,
            out_dir=out_dir,
            dataset=ds,
            split=a.split,
            max_correct=int(a.max_correct),
            max_incorrect=int(a.max_incorrect),
            shuffle=bool(a.shuffle_responses),
            force=bool(a.force),
            dry_run=bool(a.dry_run),
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
