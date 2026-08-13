#!/usr/bin/env python
r"""End-to-end builder for the MS MARCO Passage Train silver-data pool.

Pipeline (deterministic; same seed -> same outputs):

  1. Sample N queries from `ir_datasets:msmarco-passage/train` with `--seed`.
  2. Run BM25 (Pyserini prebuilt `msmarco-v1-passage` index) to fetch top-K
     candidates per query.
  3. Apply stratified rank-bucket subsampling:
        * keep `--keep-top` candidates from rank 1..keep-top
        * keep `--keep-mid`  candidates uniformly from the middle band
        * keep `--keep-deep` candidates uniformly from the deep band
     Default split = 3 top + 10 mid + 7 deep = 20 per query.
  4. Resolve passage texts via the same Pyserini index (no extra download).
  5. Write three artifacts under `--out-dir`:
        topics.<n>.tsv               qid \\t text
        run.bm25.top<k>.txt          TREC run, full BM25 top-K
        fixture.<n>x<kept>.jsonl     SilverGenerator FixtureLoader file

Each step writes atomically with write-then-rename. A non-empty existing output
is skipped unless ``--force`` is set.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import ir_datasets
from presentation_dependence.utils.pyserini_index import pyserini_import, require_prebuilt_index


LOGGER = logging.getLogger("msmarco_train_silver_pool")


# --- helpers --------------------------------------------------------------


def _atomic_write_text(path: Path, lines_iter) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("w", encoding="utf-8") as f:
        for line in lines_iter:
            f.write(line)
            if not line.endswith("\n"):
                f.write("\n")
    os.replace(tmp, path)


def _step_skip_if_exists(path: Path, force: bool, label: str) -> bool:
    if force or not path.exists() or path.stat().st_size == 0:
        return False
    LOGGER.info("[skip] %s already exists at %s (use --force to rebuild)", label, path)
    return True


# --- step 1: sample queries -----------------------------------------------


def sample_queries(seed: int, n: int) -> list[tuple[str, str]]:
    LOGGER.info("Loading ir_datasets msmarco-passage/train queries ...")
    ds = ir_datasets.load("msmarco-passage/train")
    rng = random.Random(seed)
    all_queries: list[tuple[str, str]] = [(q.query_id, q.text) for q in ds.queries_iter()]
    LOGGER.info("Loaded %d total queries", len(all_queries))
    rng.shuffle(all_queries)
    sampled = all_queries[:n]
    LOGGER.info("Sampled %d queries (seed=%d).", len(sampled), seed)
    return sampled


def write_topics(topics_path: Path, queries: list[tuple[str, str]]) -> None:
    def lines():
        for qid, text in queries:
            text_safe = text.replace("\t", " ").replace("\n", " ").replace("\r", " ")
            yield f"{qid}\t{text_safe}"

    _atomic_write_text(topics_path, lines())


def read_topics(topics_path: Path) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    with topics_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            qid, _, text = line.partition("\t")
            out.append((qid, text))
    return out


# --- step 2: BM25 top-K ---------------------------------------------------


def build_bm25_run(
    queries: list[tuple[str, str]],
    run_path: Path,
    *,
    k: int,
    threads: int,
    batch_size: int,
) -> None:
    """Runs Pyserini BM25 and writes a TREC run file (atomic).

    Imports `pyserini` lazily so this script can still print --help on hosts
    without Java.
    """
    with pyserini_import():
        from pyserini.search.lucene import LuceneSearcher  # type: ignore

    LOGGER.info("Loading prebuilt index msmarco-v1-passage (downloads on first use) ...")
    searcher = require_prebuilt_index(LuceneSearcher.from_prebuilt_index("msmarco-v1-passage"), "msmarco-v1-passage")
    searcher.set_bm25(k1=0.82, b=0.68)  # Anserini's MS MARCO defaults

    tmp = run_path.with_suffix(run_path.suffix + ".tmp")
    run_path.parent.mkdir(parents=True, exist_ok=True)
    n_queries = len(queries)
    started = time.monotonic()
    written_lines = 0
    LOGGER.info("Running BM25 over %d queries (batch_size=%d, threads=%d) ...", n_queries, batch_size, threads)
    with tmp.open("w", encoding="utf-8") as f:
        for batch_start in range(0, n_queries, batch_size):
            batch = queries[batch_start : batch_start + batch_size]
            qids = [q[0] for q in batch]
            qtexts = [q[1] for q in batch]
            results = searcher.batch_search(qtexts, qids, k=k, threads=threads)
            for qid in qids:
                hits = results.get(qid) or []
                for rank, hit in enumerate(hits, start=1):
                    score = float(hit.score)
                    docid = str(hit.docid)
                    f.write(f"{qid} Q0 {docid} {rank} {score:.6f} pyserini-bm25\n")
                    written_lines += 1
            elapsed = time.monotonic() - started
            done = batch_start + len(batch)
            qps = done / max(1e-9, elapsed)
            remain = (n_queries - done) / qps if qps > 0 else float("inf")
            LOGGER.info(
                "  bm25 progress: %d / %d queries (%.1f q/s, ETA %.1f min, lines=%d)",
                done,
                n_queries,
                qps,
                remain / 60.0,
                written_lines,
            )
    os.replace(tmp, run_path)
    LOGGER.info("Wrote %s (%d lines)", run_path, written_lines)


def parse_bm25_run(run_path: Path) -> dict[str, list[tuple[int, str, float]]]:
    out: dict[str, list[tuple[int, str, float]]] = {}
    with run_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            qid, _q0, docid, rank, score = parts[:5]
            try:
                out.setdefault(qid, []).append((int(rank), docid, float(score)))
            except ValueError:
                continue
    for qid in out:
        out[qid].sort(key=lambda r: r[0])
    return out


# --- step 3: stratified subsampling --------------------------------------


def stratified_sample_indices(
    n_candidates: int,
    *,
    keep_top: int,
    keep_mid: int,
    keep_deep: int,
    mid_end_rank: int,
    rng: random.Random,
) -> list[int]:
    """Returns 0-indexed positions within the ranked candidate list."""
    top_band = list(range(0, min(keep_top, n_candidates)))
    mid_pool = list(range(len(top_band), min(mid_end_rank, n_candidates)))
    deep_pool = list(range(min(mid_end_rank, n_candidates), n_candidates))

    def take(pool: list[int], k: int) -> list[int]:
        if k <= 0 or not pool:
            return []
        if k >= len(pool):
            return list(pool)
        return rng.sample(pool, k)

    return sorted(top_band + take(mid_pool, keep_mid) + take(deep_pool, keep_deep))


# --- step 4: fixture build ------------------------------------------------


def build_fixture(
    queries: list[tuple[str, str]],
    run_by_qid: dict[str, list[tuple[int, str, float]]],
    fixture_path: Path,
    *,
    keep_top: int,
    keep_mid: int,
    keep_deep: int,
    mid_end_rank: int,
    seed: int,
    docs_resolver,
) -> int:
    rng = random.Random(seed)
    tmp = fixture_path.with_suffix(fixture_path.suffix + ".tmp")
    fixture_path.parent.mkdir(parents=True, exist_ok=True)

    needed_doc_ids: set[str] = set()
    selected: list[tuple[str, str, list[tuple[str, int]]]] = []
    for qid, qtext in queries:
        ranks = run_by_qid.get(qid)
        if not ranks:
            LOGGER.warning("query %s has no BM25 results; skipping", qid)
            continue
        n_cand = len(ranks)
        idxs = stratified_sample_indices(
            n_cand,
            keep_top=keep_top,
            keep_mid=keep_mid,
            keep_deep=keep_deep,
            mid_end_rank=mid_end_rank,
            rng=rng,
        )
        kept: list[tuple[str, int]] = []
        for i in idxs:
            rank, doc_id, _score = ranks[i]
            kept.append((doc_id, rank))
            needed_doc_ids.add(doc_id)
        selected.append((qid, qtext, kept))

    LOGGER.info(
        "Stratified sampling: %d queries, %d unique docs to fetch.",
        len(selected),
        len(needed_doc_ids),
    )

    doc_text = docs_resolver(needed_doc_ids)
    LOGGER.info("Resolved %d / %d passages.", len(doc_text), len(needed_doc_ids))

    written = 0
    with tmp.open("w", encoding="utf-8") as f:
        for qid, qtext, kept in selected:
            passages = []
            for doc_id, bm25_rank in kept:
                text = doc_text.get(doc_id)
                if text is None:
                    continue
                passages.append({"pid": doc_id, "text": text, "bm25_rank": int(bm25_rank)})
            if not passages:
                continue
            row = {"qid": qid, "query": qtext, "passages": passages}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
    os.replace(tmp, fixture_path)
    LOGGER.info("Wrote fixture %s (%d queries).", fixture_path, written)
    return written


def make_pyserini_docs_resolver():
    """Resolve doc texts via the same prebuilt MS MARCO Pyserini index."""
    with pyserini_import():
        from pyserini.search.lucene import LuceneSearcher  # type: ignore
    import json as _json

    searcher = require_prebuilt_index(LuceneSearcher.from_prebuilt_index("msmarco-v1-passage"), "msmarco-v1-passage")

    def resolve(doc_ids: set[str]) -> dict[str, str]:
        out: dict[str, str] = {}
        for doc_id in doc_ids:
            doc = searcher.doc(doc_id)
            if doc is None:
                continue
            raw = doc.raw()
            if raw is None:
                continue
            try:
                rec = _json.loads(raw)
                contents = rec.get("contents")
                if isinstance(contents, str) and contents:
                    out[doc_id] = contents
                    continue
            except (ValueError, TypeError):
                pass
            out[doc_id] = raw
        return out

    return resolve


# --- main -----------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-queries", type=int, default=30_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bm25-top-k", type=int, default=100)
    parser.add_argument("--bm25-threads", type=int, default=8)
    parser.add_argument("--bm25-batch", type=int, default=128)
    parser.add_argument("--keep-top", type=int, default=3)
    parser.add_argument("--keep-mid", type=int, default=10)
    parser.add_argument("--keep-deep", type=int, default=7)
    parser.add_argument("--mid-end-rank", type=int, default=40)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    out = args.out_dir
    topics_path = out / f"topics.{args.n_queries}.tsv"
    run_path = out / f"run.bm25.top{args.bm25_top_k}.txt"
    kept = args.keep_top + args.keep_mid + args.keep_deep
    fixture_path = out / f"fixture.{args.n_queries}x{kept}.jsonl"

    # 1) topics
    if not _step_skip_if_exists(topics_path, args.force, "topics"):
        queries = sample_queries(args.seed, args.n_queries)
        write_topics(topics_path, queries)
    queries = read_topics(topics_path)
    LOGGER.info("Topics: %d queries (%s)", len(queries), topics_path)

    # 2) bm25 top-K
    if not _step_skip_if_exists(run_path, args.force, "bm25 run"):
        build_bm25_run(
            queries,
            run_path,
            k=args.bm25_top_k,
            threads=args.bm25_threads,
            batch_size=args.bm25_batch,
        )
    run_by_qid = parse_bm25_run(run_path)
    LOGGER.info("Loaded BM25 run for %d queries from %s", len(run_by_qid), run_path)

    # 3 + 4) stratified fixture
    if not _step_skip_if_exists(fixture_path, args.force, "fixture"):
        docs_resolver = make_pyserini_docs_resolver()
        build_fixture(
            queries,
            run_by_qid,
            fixture_path,
            keep_top=args.keep_top,
            keep_mid=args.keep_mid,
            keep_deep=args.keep_deep,
            mid_end_rank=args.mid_end_rank,
            seed=args.seed,
            docs_resolver=docs_resolver,
        )
    LOGGER.info("Done. Fixture: %s", fixture_path)


if __name__ == "__main__":
    sys.exit(main())
