#!/usr/bin/env python
"""Robustly pre-download a pyserini prebuilt index tar into the local cache.

pyserini's built-in downloader (``pyserini.util.download_url``) does a single
non-resumable GET and verifies md5/size only at the end. On flaky networks /
proxies that cut long transfers this fails repeatedly on large index tars (observed
md5 mismatches at 1.8/8 GB and a truncation at 969 MB of 1.5 GB). This helper
pre-fetches the exact tar pyserini expects, using a resumable retrying download
(``curl -C - --retry``), then verifies md5. With a valid tar already in the
cache, pyserini skips its own download and just extracts.

Used by ``scripts/data/setup_dense_sparse.sh`` (set ``ROBUST_DOWNLOAD=1``) before the
fetcher runs, so the large BGE dense indexes (8-26 GB) download reliably.

Usage::

    uv run python scripts/data/prefetch_pyserini_index.py beir-v1.0.0-climate-fever.bge-base-en-v1.5.flat
    uv run python scripts/data/prefetch_pyserini_index.py msmarco-v1-passage.bge-base-en-v1.5.hnsw --retries 50
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import os
import subprocess
import sys
from pathlib import Path


def _index_info(name: str) -> dict | None:
    import pyserini.prebuilt_index_info as info  # type: ignore

    for attr in dir(info):
        v = getattr(info, attr)
        if isinstance(v, dict) and name in v and isinstance(v[name], dict):
            return v[name]
    return None


def _cache_dir() -> Path:
    # Mirror pyserini's default index cache location.
    root = os.environ.get("PYSERINI_CACHE") or str(Path.home() / ".cache" / "pyserini")
    d = Path(root) / "indexes"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def prefetch(name: str, *, retries: int = 30, force: bool = False) -> int:
    info = _index_info(name)
    if info is None:
        print(f"[prefetch][FATAL] unknown prebuilt index {name!r}", file=sys.stderr)
        return 2
    filename = info.get("filename")
    urls = info.get("urls") or []
    md5 = info.get("md5")
    if not filename or not urls:
        print(f"[prefetch][FATAL] no filename/urls for {name!r}", file=sys.stderr)
        return 2

    cache = _cache_dir()
    stem = filename[: -len(".tar.gz")] if filename.endswith(".tar.gz") else filename
    # Already extracted? pyserini extracts to "<stem>.<hash>/" -> nothing to do.
    extracted = [p for p in glob.glob(str(cache / f"{stem}.*")) if Path(p).is_dir()]
    if extracted and not force:
        print(f"[prefetch] {name}: already extracted at {extracted[0]} — skip.")
        return 0

    tar = cache / filename
    if tar.exists() and not force:
        if md5 and _md5(tar) == md5:
            print(f"[prefetch] {name}: cached tar md5 OK — skip download.")
            return 0
        print(f"[prefetch] {name}: cached tar md5 mismatch/unknown — re-downloading.")
        tar.unlink()

    last_err = ""
    for url in urls:
        print(f"[prefetch] {name}: resumable download <- {url}", flush=True)
        cmd = [
            "curl",
            "-L",
            "--fail",
            "--retry",
            str(retries),
            "--retry-all-errors",
            "--retry-delay",
            "3",
            "-C",
            "-",
            "-o",
            str(tar),
            url,
        ]
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            last_err = f"curl rc={rc} for {url}"
            print(f"[prefetch][WARN] {last_err}; trying next URL...", flush=True)
            continue
        if md5 and _md5(tar) != md5:
            last_err = f"md5 mismatch after download from {url}"
            print(f"[prefetch][WARN] {last_err}; removing + trying next URL...", flush=True)
            tar.unlink(missing_ok=True)
            continue
        size = tar.stat().st_size
        print(f"[prefetch] {name}: OK ({size / 1e9:.2f} GB, md5 verified).", flush=True)
        return 0
    print(f"[prefetch][FATAL] {name}: all URLs failed ({last_err})", file=sys.stderr)
    return 1


def purge(name: str) -> int:
    """Delete the cached tar + extracted dir for a prebuilt index (reclaim disk).

    Safe once the candidate-set fixture is built: the Stage-1 eval path uses the
    self-contained ``fixture.jsonl`` (FixtureLoader), not the pyserini index.
    """
    info = _index_info(name)
    if info is None:
        print(f"[purge][WARN] unknown prebuilt index {name!r}; nothing to purge.")
        return 0
    filename = info.get("filename") or ""
    stem = filename[: -len(".tar.gz")] if filename.endswith(".tar.gz") else filename
    cache = _cache_dir()
    removed = 0
    for path in glob.glob(str(cache / f"{stem}*")):
        p = Path(path)
        try:
            if p.is_dir():
                import shutil

                shutil.rmtree(p)
            else:
                p.unlink()
            removed += 1
        except OSError as e:
            print(f"[purge][WARN] could not remove {p}: {e}")
    print(f"[purge] {name}: removed {removed} cache item(s).")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("index", help="pyserini prebuilt index name (e.g. beir-v1.0.0-climate-fever.bge-base-en-v1.5.flat).")
    p.add_argument("--retries", type=int, default=30, help="curl --retry count (default 30).")
    p.add_argument("--force", action="store_true", help="Re-download even if a valid tar/extract exists.")
    p.add_argument(
        "--purge", action="store_true", help="Delete this index's cached tar + extracted dir instead of downloading."
    )
    args = p.parse_args()
    if args.purge:
        return purge(args.index)
    return prefetch(args.index, retries=args.retries, force=args.force)


if __name__ == "__main__":
    raise SystemExit(main())
