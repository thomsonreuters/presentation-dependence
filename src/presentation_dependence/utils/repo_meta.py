"""Repository metadata helpers for run provenance."""

from __future__ import annotations

import subprocess
from pathlib import Path


def git_head_sha(cwd: str | Path | None = None) -> str | None:
    """Return the current Git HEAD SHA, or ``None`` outside a Git checkout."""
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(cwd) if cwd is not None else None,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return None
