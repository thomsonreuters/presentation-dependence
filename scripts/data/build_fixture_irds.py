#!/usr/bin/env python
"""Build a FixtureLoader ``fixture.jsonl`` from an ir_datasets doc corpus.

The index-free analogue of ``build_fixture_pyserini.py``: instead of a
pyserini prebuilt Lucene index (which for MS MARCO v2 is a 38.8 GB, throttled
download), it pulls candidate passage *text* from ir_datasets' random-access
docstore. Used for the TREC-DL 2021/22/23 surfaces (MS MARCO v2 passage), whose
first-stage runs come from the official ``scoreddocs``.

Output schema (one JSON object per line), matching ``FixtureLoader``::

    {"qid": "...", "query": "...",
     "passages": [{"pid": "...", "text": "..."}, ...]}   # rank order, top-k

Only qids present in BOTH the run and the topics TSV are written (the topics
TSV is the judged subset for DL2x), mirroring ``stage.build_fixture``.

Usage::

    uv run python scripts/data/build_fixture_irds.py \
        --dataset msmarco-passage-v2/trec-dl-2021 \
        --topics  data/dl21-passage/topics.tsv \
        --run     data/dl21-passage/run.bm25-official.dl21.txt \
        --out     data/dl21-passage/fixture.jsonl --k 100
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path


def _load_topics(path: Path) -> dict[str, str]:
    topics: dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) == 2:
                topics[parts[0]] = parts[1]
    return topics


def _load_run_topk(path: Path, k: int) -> "collections.OrderedDict[str, list[str]]":
    """Qid -> [pid, ...] in ascending-rank order, truncated to top-k."""
    per: dict[str, list[tuple[int, str]]] = collections.defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            qid, _q0, pid, rank = parts[0], parts[1], parts[2], parts[3]
            try:
                rk = int(rank)
            except ValueError:
                continue
            per[qid].append((rk, pid))
    out: "collections.OrderedDict[str, list[str]]" = collections.OrderedDict()
    for qid, entries in per.items():
        entries.sort()
        out[qid] = [pid for _rk, pid in entries[:k]]
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, help="ir_datasets id (e.g. msmarco-passage-v2/trec-dl-2021).")
    p.add_argument("--topics", required=True, type=Path, help="qid<TAB>query TSV (judged subset).")
    p.add_argument("--run", required=True, type=Path, help="TREC run file (qid Q0 pid rank score tag).")
    p.add_argument("--out", required=True, type=Path, help="Output fixture.jsonl path.")
    p.add_argument("--k", type=int, default=100, help="Candidates per query (default 100).")
    args = p.parse_args()

    import ir_datasets

    topics = _load_topics(args.topics)
    run = _load_run_topk(args.run, args.k)
    qids = [q for q in run if q in topics]
    if not qids:
        print("[fixture][FATAL] no overlap between run qids and topics.tsv", file=sys.stderr)
        return 2

    # Unique pids across the kept queries -> one batched docstore lookup.
    need_pids: set[str] = set()
    for q in qids:
        need_pids.update(run[q])
    print(f"[fixture] dataset={args.dataset} queries={len(qids)} unique_pids={len(need_pids)} k={args.k}", flush=True)

    store = ir_datasets.load(args.dataset).docs_store()
    docs = store.get_many(sorted(need_pids))  # {pid: Doc(.text)}
    print(f"[fixture] fetched {len(docs)}/{len(need_pids)} passages from docstore", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    n_missing = 0
    with open(args.out, "w", encoding="utf-8") as outf:
        for qid in qids:
            passages = []
            for pid in run[qid]:
                doc = docs.get(pid)
                if doc is None:
                    n_missing += 1
                    continue
                text = getattr(doc, "text", "") or ""
                if not text:
                    continue
                passages.append({"pid": pid, "text": text})
            rec = {"qid": qid, "query": topics[qid], "passages": passages}
            outf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_written += 1

    size_mb = args.out.stat().st_size / 1e6
    print(
        f"[fixture] wrote {n_written} queries ({n_missing} missing pids) -> {args.out} ({size_mb:.1f} MB)", flush=True
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
