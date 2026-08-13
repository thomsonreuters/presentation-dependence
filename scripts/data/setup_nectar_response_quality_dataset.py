#!/usr/bin/env python
r"""Materialize Nectar as an N=7 response-quality FixtureLoader eval dataset.

Nectar (`berkeley-nest/Nectar`) is the 7-wise listwise preference set: each prompt
has 7 model responses with a GPT-4-assigned `rank` (1=best .. 7=worst). Larger-N
(N=7) listwise stress surface for the response-ranking arm. Never trained on;
held-out vs the UltraFeedback-trained students.

Mapping onto the (query, candidate_docs, graded_labels) contract:

    prompt (\\n\\nHuman: ... \\n\\nAssistant:) -> query (de-framed)
    answers[].answer -> responses
    rank (1..7)      -> graded qrels {0,1,2,3} (rank 1 -> 3, 2-3 -> 2, 4-5 -> 1, 6-7 -> 0)

The raw rank is stored per response (`gold_rank` / `gold_score = 8-rank`) for
Kendall-tau-to-gold. Per-qid deterministic shuffle neutralizes any source
ordering. Eval-only.

Usage:
    uv run python -m scripts.data.setup_nectar_response_quality_dataset \
        --revision 3c6b4c47fa1cc38869f9f32dce1699f7abad8b06 --max-queries 500
    uv run python -m scripts.data.setup_nectar_response_quality_dataset --src-jsonl sample.jsonl --out /tmp/nectar
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from presentation_dependence.reproduction.data_setup import NECTAR_HF_REVISION

DEFAULT_DATASET = "berkeley-nest/Nectar"
DEFAULT_SPLIT = "train"
RETRIEVER_TOKEN = "nectar-shuffled-order"
SLM_ROOT = Path(__file__).resolve().parents[2]


def default_out_dir() -> Path:
    return Path("data") / "nectar-response-quality"


def _qid_seed(qid: str) -> int:
    return int(hashlib.sha1(qid.encode("utf-8")).hexdigest()[:8], 16)


def grade_from_rank(rank: int) -> int:
    # rank 1 -> 3, 2-3 -> 2, 4-5 -> 1, 6-7 -> 0
    if rank <= 1:
        return 3
    if rank <= 3:
        return 2
    if rank <= 5:
        return 1
    return 0


def deframe_prompt(raw: str) -> str:
    r"""Extract the user query from Nectar's '\\n\\nHuman: .. \\n\\nAssistant:' format."""
    text = str(raw)
    # last Human turn before the trailing Assistant cue
    m = re.findall(r"Human:\s*(.*?)\s*(?:\n\nAssistant:|\Z)", text, flags=re.DOTALL)
    body = m[-1] if m else text
    return " ".join(body.split())


def _answer_pairs(record: dict[str, Any]) -> list[tuple[str, int]]:
    answers = record.get("answers") or []
    out: list[tuple[str, int]] = []
    for a in answers:
        if not isinstance(a, dict):
            continue
        out.append((str(a.get("answer", "")), int(a.get("rank", 0))))
    return out


def load_records_from_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_records_from_hf(
    dataset: str,
    split: str,
    revision: str,
    max_queries: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset(dataset, split=split, revision=revision)
    n = len(ds)
    idx = list(range(n))
    if max_queries is not None and max_queries < n:
        idx = random.Random(seed).sample(idx, max_queries)
        idx.sort()
    return [dict(ds[i]) for i in idx]


def materialize(records, *, out_dir, dataset, split, revision, required_responses, shuffle, force, dry_run):  # noqa: C901
    if out_dir.exists() and any(out_dir.iterdir()) and not force and not dry_run:
        raise FileExistsError(f"{out_dir} exists and is non-empty; pass --force")
    topics, qrels, passages_by_qid = {}, {}, {}
    grade_counts: Counter[int] = Counter()
    n_resp_counts: Counter[int] = Counter()
    skipped = 0
    for idx, rec in enumerate(records):
        qid = f"nectar-{idx:06d}"
        prompt = deframe_prompt(rec.get("prompt", ""))
        pairs = _answer_pairs(rec)
        if required_responses is not None and len(pairs) != int(required_responses):
            skipped += 1
            continue
        if not prompt or any((not t.strip()) or r <= 0 for t, r in pairs):
            skipped += 1
            continue
        graded = [(t, grade_from_rank(r), r) for t, r in pairs]
        if shuffle:
            random.Random(_qid_seed(qid)).shuffle(graded)
        passages, qrels[qid] = [], {}
        n = len(graded)
        for pos, (text, grade, rank) in enumerate(graded, start=1):
            pid = f"{qid}-r{pos:02d}"
            qrels[qid][pid] = grade
            grade_counts[grade] += 1
            passages.append(
                {
                    "pid": pid,
                    "text": " ".join(text.split()).strip(),
                    "gold_rank": rank,
                    "gold_score": 8 - rank,
                    "gold_grade": grade,
                    "score": float(n - pos + 1),
                    "rank": pos,
                }
            )
        topics[qid] = prompt
        passages_by_qid[qid] = passages
        n_resp_counts[n] += 1
    if not topics:
        raise ValueError("No Nectar records selected after filtering")

    qids = sorted(topics)
    run_name = "run.nectar-response-quality_sorted.txt"
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
        meta = {
            "dataset": "nectar-response-quality",
            "source": {
                "dataset": dataset,
                "split": split,
                "revision": revision,
                "loader": "datasets.load_dataset",
            },
            "created_at": date.today().isoformat(),
            "access": "public",
            "task": "response-quality scoring (reward modeling) — Nectar N=7 listwise eval surface",
            "label_type": "graded",
            "qrels": {
                "semantics": "GPT-4 rank 1..7 -> grade {3,2,2,1,1,0,0}; gold_score=8-rank stored per response",
                "grade_passages_per_grade": dict(sorted(grade_counts.items())),
                "explicit_zero_rows": True,
            },
            "candidate_set": {
                "n_responses": dict(sorted(n_resp_counts.items())),
                "source_order": "per-qid deterministic shuffle" if shuffle else "source order",
                "run_file": run_name,
                "retriever": RETRIEVER_TOKEN,
            },
            "counts": {"queries": len(qids)},
            "filtering": {"required_responses": required_responses, "skipped": skipped},
            "files": {"fixture": "fixture.jsonl", "topics": "topics.tsv", "qrels": "qrels.txt", "run": run_name},
        }
        (out_dir / "dataset_meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False), encoding="utf-8")
    return {
        "out_dir": str(out_dir),
        "n_queries": len(qids),
        "grade_counts": dict(sorted(grade_counts.items())),
        "n_responses": dict(sorted(n_resp_counts.items())),
        "skipped": skipped,
        "dry_run": dry_run,
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--split", default=DEFAULT_SPLIT)
    p.add_argument("--revision", default=NECTAR_HF_REVISION, help="Immutable Hugging Face dataset commit.")
    p.add_argument("--src-jsonl", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--max-queries", type=int, default=500, help="Held-out eval slice size (seeded sample).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--required-responses", type=int, default=7, help="Require exactly this many; 0 to disable.")
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
            revision = None
        else:
            revision = str(a.revision).strip().lower()
            if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
                raise ValueError("--revision must be a 40-character Hugging Face commit SHA")
            records = load_records_from_hf(a.dataset, a.split, revision, a.max_queries, a.seed)
            ds = a.dataset
        summary = materialize(
            records,
            out_dir=out_dir,
            dataset=ds,
            split=a.split,
            revision=revision,
            required_responses=(None if int(a.required_responses) <= 0 else int(a.required_responses)),
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
