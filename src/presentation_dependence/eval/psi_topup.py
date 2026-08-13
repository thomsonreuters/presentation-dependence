"""Safe merge helpers for focused PSI top-up runs."""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

_PERM_DIR_RE = re.compile(r"^permutation_(\d+)_(.+)$")


def _count_perm_dirs(qdir: Path) -> int:
    if not qdir.is_dir() or qdir.is_symlink():
        return 0
    return sum(
        1 for child in qdir.iterdir() if child.is_dir() and not child.is_symlink() and _PERM_DIR_RE.match(child.name)
    )


def _reject_symlinks(path: Path) -> None:
    if path.is_symlink() or any(child.is_symlink() for child in path.rglob("*")):
        raise ValueError(f"Refusing to merge symlinked PSI artifacts: {path}")


def merge_psi_topup(  # noqa: C901
    parent_run_dir: Path,
    topup_run_dir: Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Copy usable top-up query artifacts into a parent run.

    Existing complete parent queries are retained unless ``overwrite`` is true.
    Symlinked artifacts are rejected so a run tree cannot copy data from or
    delete data outside the two explicitly supplied run directories.
    """
    parent_run_dir = parent_run_dir.resolve()
    topup_run_dir = topup_run_dir.resolve()
    if parent_run_dir == topup_run_dir:
        raise ValueError("Parent and top-up run directories must differ")

    parent_pq = parent_run_dir / "psi" / "per_query_results"
    topup_pq = topup_run_dir / "psi" / "per_query_results"
    if not topup_pq.is_dir():
        raise FileNotFoundError(f"topup psi/per_query_results missing: {topup_pq}")
    _reject_symlinks(topup_pq)
    if parent_pq.exists():
        _reject_symlinks(parent_pq)
    parent_pq.mkdir(parents=True, exist_ok=True)

    n_copied = 0
    n_overwritten = 0
    n_skipped_complete = 0
    n_topup_qids = 0

    for src_qdir in sorted(topup_pq.iterdir()):
        if not src_qdir.is_dir() or src_qdir.is_symlink():
            continue
        qid = src_qdir.name
        n_topup_qids += 1
        src_n = _count_perm_dirs(src_qdir)
        src_has_aligned = (src_qdir / "aligned_scores.json").is_file()
        if src_n == 0 and not src_has_aligned:
            print(f"[merge] qid={qid}: topup has no usable artifacts; skipping")
            continue

        dst_qdir = parent_pq / qid
        dst_n = _count_perm_dirs(dst_qdir)
        dst_has_aligned = (dst_qdir / "aligned_scores.json").is_file()
        if ((src_n > 0 and dst_n >= src_n) or (src_n == 0 and src_has_aligned and dst_has_aligned)) and not overwrite:
            print(f"[merge] qid={qid}: parent has {dst_n} perms (>= topup's {src_n}); skipping")
            n_skipped_complete += 1
            continue

        if dst_qdir.exists():
            for child in list(dst_qdir.iterdir()):
                if child.is_dir() and not child.is_symlink() and _PERM_DIR_RE.match(child.name):
                    shutil.rmtree(child)
                elif child.name == "input_positions.json":
                    child.unlink()
            n_overwritten += int(dst_n > 0)
        else:
            dst_qdir.mkdir(parents=True)

        for child in src_qdir.iterdir():
            destination = dst_qdir / child.name
            if child.is_dir():
                shutil.copytree(child, destination)
            else:
                shutil.copy2(child, destination)
        n_copied += 1
        print(f"[merge] qid={qid}: copied {src_n} permutation dirs -> {dst_qdir}")

    return {
        "n_topup_qids": n_topup_qids,
        "n_copied_or_overwritten": n_copied,
        "n_overwritten_existing": n_overwritten,
        "n_skipped_already_complete": n_skipped_complete,
    }
