#!/usr/bin/env python
"""Container entry point: build BGE MS MARCO train self-distill candidates (30K qids).

Materialises ``data/msmarco-train-selfdistill-seed42-bge/`` from the BM25
channel beside it. The retrieval itself is
``scripts/data/setup_msmarco_self_distill_dense.py``, which runs standalone;
this wrapper adds the Java and pyserini bootstrap a bare container needs, and a
size guard that catches an empty retrieval before anything consumes it.

Arguments:
  --query-count <int>      Queries to materialise (default 30000).
  --backend <faiss|hnsw>   Dense backend (default hnsw).
  --data-root <path>       Where the channels live (default: <project>/data).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Shared Java/pyserini bootstrap. The launcher ships `scripts/train/` as the
# container source dir, so in the container this module sits flat next to this
# file; the package path is the fallback for running it locally.
try:
    from _container_bootstrap import _build_env, _code_root, _install_java_and_deps, _run
except ModuleNotFoundError:  # pragma: no cover - local invocation only
    from scripts.train._container_bootstrap import (
        _build_env,
        _code_root,
        _install_java_and_deps,
        _run,
    )

BM25_CHANNEL = "msmarco-train-selfdistill-seed42"
BGE_CHANNEL = "msmarco-train-selfdistill-seed42-bge"


def _require_bm25_inputs(data_root: Path, query_count: int) -> None:
    """Fail early if the BM25 channel this build reads from is incomplete."""
    local = data_root / BM25_CHANNEL
    qids = f"qids_{query_count // 1000}k.txt" if query_count % 1000 == 0 else "qids_30k.txt"
    missing = [name for name in ("topics.tsv", "qrels.txt", qids) if not (local / name).is_file()]
    if missing:
        raise SystemExit(
            f"[msmarco-train-dense][FATAL] {local} is missing {missing}. "
            "Build the BM25 channel first with scripts/data/setup_msmarco_self_distill.py."
        )


def _check_channel(data_root: Path, channel: str) -> None:
    """Reject an empty retrieval before it is written into a training config."""
    data_dir = data_root / channel
    fixture = data_dir / "fixture.jsonl"
    if not fixture.is_file():
        raise SystemExit(f"[msmarco-train-dense][FATAL] missing {fixture}")
    size_mb = fixture.stat().st_size / 1e6
    if size_mb < 50:
        raise SystemExit(
            f"[msmarco-train-dense][FATAL] fixture.jsonl only {size_mb:.1f} MB — expected ~1+ GB for 30K x 100; "
            "retrieval likely empty."
        )
    meta = {"channel": channel, "built_by": "build_msmarco_train_dense_entrypoint.py"}
    (data_dir / "_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[msmarco-train-dense] {fixture} ok ({size_mb:.0f} MB)", flush=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--query-count", type=int, default=30000)
    p.add_argument("--backend", default="hnsw", choices=("faiss", "hnsw"))
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--pip-extra", default="")
    args = p.parse_args()

    root = _code_root()
    os.chdir(root)
    data_root = args.data_root or (root / "data")
    pip_extra = [s.strip() for s in args.pip_extra.split(",") if s.strip()]
    java_home, libjvm_dir = _install_java_and_deps(pip_extra)
    env = _build_env(java_home, libjvm_dir)

    _require_bm25_inputs(data_root, args.query_count)

    cmd = [
        sys.executable,
        "scripts/data/setup_msmarco_self_distill_dense.py",
        "--query-count",
        str(args.query_count),
        "--backend",
        args.backend,
        "--out-dir",
        str(data_root / BGE_CHANNEL),
        "--bm25-dir",
        str(data_root / BM25_CHANNEL),
        "--force",
    ]
    _run(cmd, env=env, check=True)
    _check_channel(data_root, BGE_CHANNEL)
    print("[msmarco-train-dense] DONE.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
