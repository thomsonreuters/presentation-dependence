#!/usr/bin/env python
"""Emit a gold-answer sidecar (``answers.jsonl``) for the multi-doc QA fixtures.

The support-scoring fixtures built by ``setup_hotpotqa_support_dataset.py`` and
``setup_multihopqa_support_dataset.py`` drop the free-text gold
answer (the QA arm only needs binary support qrels). The reader bridge
needs that gold answer to score EM/F1, so write a sidecar keyed by
the exact fixture qid without touching the existing fixture / qrels / topics
files.

    data/<slug>/answers.jsonl   # one JSON object per line:
        {"qid": "...", "answers": ["primary", "alias", ...]}

EM/F1 take the max over the accepted-answer list (SQuAD multi-reference
convention; see ``presentation_dependence.reader.answer_eval``). The qid extraction mirrors
the support-fixture builders exactly so the sidecar joins 1:1 with the fixture.

Usage:

    uv run python -m scripts.data.setup_qa_answers --dataset hotpotqa
    uv run python -m scripts.data.setup_qa_answers --dataset 2wiki
    uv run python -m scripts.data.setup_qa_answers --dataset musique

``--src-jsonl`` passes HF-shaped records so the script stays covered offline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

SLM_ROOT = Path(__file__).resolve().parents[2]


# dataset key -> (default fixture slug, HF dataset, HF config, split)
DATASETS: dict[str, dict[str, str]] = {
    "hotpotqa": {
        "slug": "hotpotqa-distractor-support-dev",
        "hf_dataset": "hotpot_qa",
        "hf_config": "distractor",
        "split": "validation",
    },
    "2wiki": {
        "slug": "2wiki-distractor-support-dev",
        "hf_dataset": "voidful/2WikiMultihopQA",
        "hf_config": "default",
        "split": "validation",
    },
    "musique": {
        "slug": "musique-support-dev",
        "hf_dataset": "dgslibisey/MuSiQue",
        "hf_config": "default",
        "split": "validation",
    },
}


def _clean(text: Any) -> str:
    return " ".join(str(text).split())


def _aliases(record: dict[str, Any]) -> list[str]:
    raw = record.get("answer_aliases") or record.get("answer_alias") or []
    if isinstance(raw, str):
        raw = [raw]
    return [_clean(a) for a in raw if a is not None and _clean(a)]


def _answers_hotpotqa(record: dict[str, Any]) -> tuple[str, list[str]]:
    qid = str(record.get("id") or record.get("_id") or record.get("qid"))
    answers = [_clean(record.get("answer"))] + _aliases(record)
    return qid, answers


def _answers_2wiki(record: dict[str, Any]) -> tuple[str, list[str]]:
    qid = str(record.get("_id") or record.get("id"))
    answers = [_clean(record.get("answer"))] + _aliases(record)
    return qid, answers


def _answers_musique(record: dict[str, Any]) -> tuple[str, list[str]]:
    qid = str(record.get("id"))
    answers = [_clean(record.get("answer"))] + _aliases(record)
    return qid, answers


ANSWER_PARSERS: dict[str, Callable[[dict[str, Any]], tuple[str, list[str]]]] = {
    "hotpotqa": _answers_hotpotqa,
    "2wiki": _answers_2wiki,
    "musique": _answers_musique,
}


def _dedup_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def read_fixture_qids(fixture_path: Path) -> list[str]:
    qids: list[str] = []
    with fixture_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                qids.append(str(json.loads(line)["qid"]))
    return qids


def load_records_from_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_records_from_hf(dataset: str, config: str, split: str) -> list[dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset(dataset, config, split=split)
    return [dict(rec) for rec in ds]


def build_answers(
    dataset_key: str,
    records: list[dict[str, Any]],
    fixture_qids: list[str],
) -> dict[str, list[str]]:
    parser = ANSWER_PARSERS[dataset_key]
    by_qid: dict[str, list[str]] = {}
    for record in records:
        qid, answers = parser(record)
        answers = _dedup_keep_order(answers)
        if not answers:
            raise ValueError(f"{dataset_key}: record qid={qid!r} has no gold answer")
        by_qid[qid] = answers

    wanted = set(fixture_qids)
    missing = sorted(wanted - by_qid.keys())
    if missing:
        raise ValueError(
            f"{dataset_key}: {len(missing)} fixture qids have no gold answer in the source "
            f"(qid scheme mismatch?). First few: {missing[:5]}"
        )
    return {qid: by_qid[qid] for qid in fixture_qids}


def write_answers(answers: dict[str, list[str]], path: Path, qids: list[str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for qid in qids:
            f.write(json.dumps({"qid": qid, "answers": answers[qid]}, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--data-dir", type=Path, default=None, help="Fixture dir; defaults to data/<slug>.")
    parser.add_argument("--src-jsonl", type=Path, default=None, help="Offline HF-shaped JSONL source (tests).")
    parser.add_argument("--out", type=Path, default=None, help="Sidecar path; defaults to <data-dir>/answers.jsonl.")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing answers.jsonl.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        spec = DATASETS[args.dataset]
        data_dir = args.data_dir or (Path("data") / spec["slug"])
        if not data_dir.is_absolute():
            data_dir = SLM_ROOT / data_dir
        fixture_path = data_dir / "fixture.jsonl"
        if not fixture_path.exists():
            raise FileNotFoundError(f"fixture not found: {fixture_path}; build the support fixture first")
        out_path = args.out or (data_dir / "answers.jsonl")
        if out_path.exists() and not args.force:
            raise FileExistsError(f"{out_path} already exists; pass --force to overwrite")

        fixture_qids = read_fixture_qids(fixture_path)
        if args.src_jsonl:
            records = load_records_from_jsonl(args.src_jsonl)
        else:
            records = load_records_from_hf(spec["hf_dataset"], spec["hf_config"], spec["split"])
        answers = build_answers(args.dataset, records, fixture_qids)
        write_answers(answers, out_path, fixture_qids)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3

    print(
        json.dumps(
            {
                "dataset": args.dataset,
                "out": str(out_path),
                "n_queries": len(fixture_qids),
                "example": {fixture_qids[0]: answers[fixture_qids[0]]} if fixture_qids else {},
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
