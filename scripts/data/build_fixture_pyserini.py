#!/usr/bin/env python
r"""Flatten a prebuilt pyserini index plus a first-stage run into a fixture.

`PyseriniLoader` reads passages straight out of a prebuilt Lucene index, so
local evaluation needs nothing from this script. It exists for the container
path: the images never install pyserini, so a job that evaluates a
pyserini-backed collection has to be handed a self-contained `fixture.jsonl`
built beforehand.

`scripts/data/build_fixture_irds.py` covers collections served by ir_datasets
(DL21/22/23) and `build_smoke_fixture.py` writes a synthetic one. This is the
prebuilt-index path.

``-e`` derives every input from a config, which requires that config to name a
first stage: ``data.topics_tsv``, ``data.run_path`` pointing at a TREC run, and
``data.index``. A ``FixtureLoader`` config has none of those, only the fixture
this script would write, so ``-e`` rejects one; build for those with the
explicit flags below.

Usage::

    python scripts/data/build_fixture_pyserini.py -e example-passage-mxbai-large-v2
    python scripts/data/build_fixture_pyserini.py -e example-passage-mxbai-large-v2 --out-dir data/dl19-passage
    python scripts/data/build_fixture_pyserini.py \
        --topics data/dl19-passage/topics.tsv \
        --run data/dl19-passage/run.msmarco-v1-passage.bm25-default.dl19_sorted.txt \
        --index msmarco-v1-passage --k 100 --out data/dl19-passage/fixture.jsonl

Writes `fixture.jsonl` (one JSON record per query: qid, query, passages) plus a
`_meta.json` sidecar recording the sha256, k, and index the fixture was built
from, so a downstream consumer can tell two fixtures apart.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from presentation_dependence.utils.config import load_experiment_config, resolve_channel  # noqa: E402
from presentation_dependence.utils.pyserini_index import pyserini_import, require_prebuilt_index  # noqa: E402


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """Return the sha256 of a file, read in chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def build_fixture(  # noqa: C901
    topics_tsv: Path,
    run_path: Path,
    index_name: str,
    k: int,
    out_path: Path,
) -> int:
    """Materialise fixture.jsonl using pyserini + the prebuilt index.

    Returns the number of queries written.
    """
    try:
        with pyserini_import():
            from pyserini.search.lucene import LuceneSearcher  # type: ignore
    except Exception as e:  # pragma: no cover - env-specific
        print(f"[fixture][FATAL] pyserini import failed: {e}", file=sys.stderr)
        print("Hint: `uv sync` and set JAVA_HOME; pyserini needs a JVM.", file=sys.stderr)
        raise SystemExit(2) from e

    print(f"[fixture] building: index={index_name} k={k}", flush=True)
    searcher = require_prebuilt_index(LuceneSearcher.from_prebuilt_index(index_name), index_name)

    topics: dict[str, str] = {}
    with open(topics_tsv, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            topics[parts[0]] = parts[1]

    per_qid: dict[str, list[tuple[int, str]]] = {}
    with open(run_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            qid, _q0, pid, rank = parts[0], parts[1], parts[2], parts[3]
            try:
                rk = int(rank)
            except ValueError:
                continue
            per_qid.setdefault(qid, []).append((rk, pid))

    n_written = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as outf:
        for qid, entries in per_qid.items():
            if qid not in topics:
                print(f"[fixture][WARN] qid={qid} missing from topics.tsv", flush=True)
                continue
            entries.sort()
            passages: list[dict] = []
            for _rk, pid in entries[:k]:
                doc = searcher.doc(pid)
                if doc is None:
                    print(f"[fixture][WARN] pid={pid} missing from {index_name}", flush=True)
                    continue
                raw = doc.raw()
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError:
                    parsed = {"contents": raw}
                text = parsed.get("contents") or parsed.get("text") or ""
                if not text:
                    continue
                passages.append({"pid": pid, "text": text})
            outf.write(json.dumps({"qid": qid, "query": topics[qid], "passages": passages}, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"[fixture] wrote {n_written} queries to {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)", flush=True)
    return n_written


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-e", "--exp-config", default=None, help="Experiment id or path; derives every input below.")
    p.add_argument("--topics", type=Path, default=None, help="topics.tsv (qid<TAB>query).")
    p.add_argument("--run", type=Path, default=None, help="First-stage TREC run file.")
    p.add_argument("--index", default=None, help="Prebuilt pyserini index name.")
    p.add_argument("--k", type=int, default=None, help="Top-k passages per query. Default: data.k_input or 100.")
    p.add_argument("--out", type=Path, default=None, help="Output fixture.jsonl path.")
    p.add_argument("--out-dir", type=Path, default=None, help="Directory to write fixture.jsonl into.")
    return p.parse_args()


def main() -> int:  # noqa: C901
    """Build one fixture from a config or from explicit inputs."""
    args = parse_args()
    root = _project_root()
    topics, run_path, index_name, k, out = args.topics, args.run, args.index, args.k, args.out

    if args.exp_config:
        config_path, cfg = load_experiment_config(args.exp_config)
        data = cfg.get("data") or {}
        if (data.get("dataloader_class") == "FixtureLoader") or cfg.get("student"):
            print(
                f"[fixture][FATAL] {config_path.name} already uses a prebuilt fixture "
                f"({data.get('run_path') or 'student.fixture_path'}); nothing to build.",
                file=sys.stderr,
            )
            return 2
        missing = [
            name
            for name, val in (
                ("data.topics_tsv", data.get("topics_tsv")),
                ("data.run_path", data.get("run_path")),
                ("data.index", data.get("index")),
            )
            if not val
        ]
        if missing:
            print(f"[fixture][FATAL] {config_path.name} is missing {', '.join(missing)}", file=sys.stderr)
            return 2
        topics = topics or root / str(data["topics_tsv"])
        run_path = run_path or root / str(data["run_path"])
        index_name = index_name or str(data["index"])
        k = k if k is not None else int(data.get("k_input", 100))
        out = out or (args.out_dir or root / "data" / resolve_channel(cfg)) / "fixture.jsonl"

    if not all([topics, run_path, index_name, out]):
        print("[fixture][FATAL] need -e, or all of --topics --run --index --out", file=sys.stderr)
        return 2

    n = build_fixture(Path(topics), Path(run_path), str(index_name), int(k or 100), Path(out))
    meta = {
        "sha256": sha256_file(Path(out)),
        "k_input": int(k or 100),
        "index": index_name,
        "n_queries": n,
        "build_ts": datetime.now(timezone.utc).isoformat(),
    }
    meta_path = Path(out).with_name("_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[fixture] meta -> {meta_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
