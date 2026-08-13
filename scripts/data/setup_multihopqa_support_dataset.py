#!/usr/bin/env python
"""Materialize 2WikiMultiHopQA / MuSiQue as B=10 support-scoring fixtures.

The fixtures support downstream QA evaluation without retrieval or silver
generation. Their schema matches the HotpotQA support-scoring object:

    question -> query
    10 passages -> candidate passages
    exactly 2 support passages -> binary qrels

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
from typing import Any

import yaml


SLM_ROOT = Path(__file__).resolve().parents[2]
RETRIEVER_TOKEN = "fixed-distractor-order"
DEFAULT_PREFIX_COUNTS = (100, 200)


DATASETS: dict[str, dict[str, str]] = {
    "2wiki": {
        "hf_dataset": "voidful/2WikiMultihopQA",
        "hf_config": "default",
        "split": "validation",
        "slug": "2wiki-distractor-support-dev",
        "name": "2WikiMultiHopQA distractor support dev",
    },
    "musique": {
        "hf_dataset": "dgslibisey/MuSiQue",
        "hf_config": "default",
        "split": "validation",
        "slug": "musique-support-dev",
        "name": "MuSiQue support dev",
    },
}


def _slug_token(raw: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "-", raw.strip()).strip("-")
    return token or "qid"


def _norm_title(raw: Any) -> str:
    return " ".join(str(raw).split()).casefold()


def _text(sentences: Any) -> str:
    if isinstance(sentences, str):
        return " ".join(sentences.split())
    return " ".join(" ".join(str(s).split()) for s in sentences)


def _parse_2wiki(record: dict[str, Any]) -> tuple[str, str, list[dict[str, Any]]]:
    qid = str(record.get("_id") or record.get("id"))
    question = " ".join(str(record["question"]).split())
    support_titles = {_norm_title(item[0]) for item in record.get("supporting_facts") or [] if item}
    passages = []
    for idx, item in enumerate(record.get("context") or [], start=1):
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        title = str(item[0])
        text = f"{title}\n{_text(item[1])}".strip()
        passages.append(
            {
                "pid": f"{_slug_token(qid)}-p{idx:02d}",
                "title": title,
                "text": text,
                "rank": idx,
                "score": float(100 - idx),
                "is_support": _norm_title(title) in support_titles,
            }
        )
    return qid, question, passages


def _parse_musique(record: dict[str, Any]) -> tuple[str, str, list[dict[str, Any]]]:
    qid = str(record.get("id"))
    question = " ".join(str(record["question"]).split())
    raw_passages = []
    for item in record.get("paragraphs") or []:
        idx = int(item["idx"])
        title = str(item.get("title") or f"paragraph-{idx}")
        text = f"{title}\n{' '.join(str(item.get('paragraph_text') or '').split())}".strip()
        raw_passages.append(
            {
                "pid": f"{_slug_token(qid)}-p{idx:02d}",
                "title": title,
                "text": text,
                "rank": idx + 1,
                "score": float(100 - idx),
                "is_support": bool(item.get("is_supporting")),
            }
        )

    supports = [p for p in raw_passages if p["is_support"]]
    distractors = [p for p in raw_passages if not p["is_support"]]
    selected = sorted(supports + distractors[: max(0, 10 - len(supports))], key=lambda p: int(p["rank"]))
    # Renumber rank after selection so the fixed candidate order is contiguous.
    for rank, passage in enumerate(selected, start=1):
        passage["rank"] = rank
        passage["score"] = float(10 - rank + 1)
    return qid, question, selected


def load_records(dataset_key: str) -> list[dict[str, Any]]:
    from datasets import load_dataset

    spec = DATASETS[dataset_key]
    ds = load_dataset(spec["hf_dataset"], spec["hf_config"], split=spec["split"])
    return [dict(rec) for rec in ds]


def parse_record(dataset_key: str, record: dict[str, Any]) -> tuple[str, str, list[dict[str, Any]]]:
    if dataset_key == "2wiki":
        return _parse_2wiki(record)
    if dataset_key == "musique":
        return _parse_musique(record)
    raise ValueError(f"Unknown dataset {dataset_key!r}")


def deterministic_qids(qids: list[str], seed: int | None) -> list[str]:
    out = sorted(qids)
    if seed is not None:
        random.Random(seed).shuffle(out)
    return out


def write_outputs(  # noqa: C901
    *,
    dataset_key: str,
    out_dir: Path,
    max_queries: int,
    seed: int | None,
    force: bool,
) -> dict[str, Any]:
    if out_dir.exists() and any(out_dir.iterdir()) and not force:
        raise FileExistsError(f"{out_dir} already exists and is non-empty; pass --force")

    topics: dict[str, str] = {}
    passages_by_qid: dict[str, list[dict[str, Any]]] = {}
    qrels: dict[str, dict[str, int]] = {}
    skipped = Counter()

    for record in load_records(dataset_key):
        if len(topics) >= max_queries:
            break
        qid, question, passages = parse_record(dataset_key, record)
        n_support = sum(1 for p in passages if p["is_support"])
        if len(passages) != 10:
            skipped[f"passages_{len(passages)}"] += 1
            continue
        if n_support != 2:
            skipped[f"support_{n_support}"] += 1
            continue
        topics[qid] = question
        passages_by_qid[qid] = [{k: v for k, v in passage.items() if k != "is_support"} for passage in passages]
        qrels[qid] = {passage["pid"]: int(bool(passage["is_support"])) for passage in passages}

    if len(topics) < max_queries:
        raise ValueError(f"Selected only {len(topics)} records, fewer than requested {max_queries}")

    out_dir.mkdir(parents=True, exist_ok=True)
    qids_sorted = sorted(topics)
    qids_sample = deterministic_qids(qids_sorted, seed)
    slug = DATASETS[dataset_key]["slug"]
    run_name = f"run.{slug}_sorted.txt"

    with (out_dir / "topics.tsv").open("w", encoding="utf-8") as f:
        for qid in qids_sorted:
            f.write(f"{qid}\t{topics[qid]}\n")
    with (out_dir / "qrels.txt").open("w", encoding="utf-8") as f:
        for qid in qids_sorted:
            for pid in sorted(qrels[qid]):
                f.write(f"{qid} Q0 {pid} {qrels[qid][pid]}\n")
    with (out_dir / run_name).open("w", encoding="utf-8") as f:
        for qid in qids_sorted:
            for passage in passages_by_qid[qid]:
                f.write(f"{qid} Q0 {passage['pid']} {passage['rank']} {passage['score']:.6f} {RETRIEVER_TOKEN}\n")
    with (out_dir / "fixture.jsonl").open("w", encoding="utf-8") as f:
        for qid in qids_sorted:
            f.write(
                json.dumps({"qid": qid, "query": topics[qid], "passages": passages_by_qid[qid]}, ensure_ascii=False)
                + "\n"
            )
    (out_dir / "qids_all.txt").write_text("".join(f"{qid}\n" for qid in qids_sample), encoding="utf-8")
    for count in DEFAULT_PREFIX_COUNTS:
        if count <= len(qids_sample):
            (out_dir / f"qids_{count}.txt").write_text(
                "".join(f"{qid}\n" for qid in qids_sample[:count]),
                encoding="utf-8",
            )

    meta = {
        "dataset": slug,
        "name": DATASETS[dataset_key]["name"],
        "source": DATASETS[dataset_key],
        "created_at": date.today().isoformat(),
        "task": "multi-doc QA passage support scoring",
        "access": "public",
        "candidate_set": {
            "passages_per_query": 10,
            "support_passages_per_query": 2,
            "retriever": RETRIEVER_TOKEN,
            "run_file": run_name,
        },
        "counts": {"queries": len(topics), "qrels_rows": len(topics) * 10},
        "filtering": dict(skipped),
        "qid_prefixes": {"seed": seed, "files": {"qids_200.txt": 200, "qids_all.txt": len(topics)}},
        "files": {"fixture": "fixture.jsonl", "topics": "topics.tsv", "qrels": "qrels.txt", "run": run_name},
    }
    (out_dir / "dataset_meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False), encoding="utf-8")
    return {"dataset": dataset_key, "out_dir": str(out_dir), "n_queries": len(topics), "skipped": dict(skipped)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--max-queries", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42, help="Seed for qid prefix files; use -1 to keep sorted.")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        seed = None if int(args.seed) < 0 else int(args.seed)
        out = args.out or Path("data") / DATASETS[args.dataset]["slug"]
        if not out.is_absolute():
            out = SLM_ROOT / out
        summary = write_outputs(
            dataset_key=args.dataset, out_dir=out, max_queries=args.max_queries, seed=seed, force=args.force
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
