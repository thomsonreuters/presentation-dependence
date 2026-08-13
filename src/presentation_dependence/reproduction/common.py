"""Shared reproduction-stage I/O helpers."""

from __future__ import annotations

import hashlib
import json
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from .errors import SilverStageError


@lru_cache(maxsize=4)
def tracked_files(project_root: Path) -> frozenset[str]:
    """Return every git-tracked path under ``project_root``.

    Cached because the tracked-source audits ask per artifact. Raises rather
    than returning an empty set, so a git failure cannot be misread as "nothing
    is tracked".
    """
    completed = subprocess.run(
        ["git", "ls-files", "--", "."],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(completed.stderr.strip() or "git ls-files failed")
    return frozenset(completed.stdout.splitlines())


def silver_block(pipeline: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return a validated silver block."""
    value = pipeline.get("silver")
    if not isinstance(value, Mapping):
        raise SilverStageError("silver must be a mapping")
    return value


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_generated_directory(path: Path, *, suffixes: tuple[str, ...] = (".yaml",)) -> Path:
    """Create a build directory after removing stale generated files."""
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_file() and child.suffix in suffixes:
            child.unlink()
    return path


def write_silver_receipts(
    pipeline: Mapping[str, Any],
    destination: Path,
    manifest: Mapping[str, Any],
    products: Mapping[str, Mapping[str, str]],
    *,
    title: str,
    details: tuple[str, ...] = (),
) -> None:
    """Write a silver manifest and Markdown product table."""
    outputs = silver_block(pipeline)["outputs"]
    (destination / str(outputs["manifest"])).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        f"# {title}",
        "",
        *details,
        *(("",) if details else ()),
        "| Product | Path | SHA-256 |",
        "|---|---|---|",
        *[
            f"| {product_id} | `{product['path']}` | `{product['sha256']}` |"
            for product_id, product in products.items()
        ],
    ]
    (destination / str(outputs["summary"])).write_text("\n".join(lines) + "\n", encoding="utf-8")
