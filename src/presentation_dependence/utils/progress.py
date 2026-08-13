"""Small append-only progress reporting helpers for eval-style loops.

Log aggregators are easiest to read when progress is plain text on normal log
lines, so that is the primary output, with JSONL kept as the richer trace.
"""

from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "warming"
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _gpu_memory_gb() -> dict[str, float]:
    """Best-effort CUDA memory snapshot; returns empty off GPU or without torch."""
    try:
        import torch

        if not torch.cuda.is_available():
            return {}
        index = torch.cuda.current_device()
        free_b, total_b = torch.cuda.mem_get_info(index)
        return {
            "gpu_mem_alloc_gb": torch.cuda.memory_allocated(index) / (1024**3),
            "gpu_mem_reserved_gb": torch.cuda.memory_reserved(index) / (1024**3),
            "gpu_mem_free_gb": free_b / (1024**3),
            "gpu_mem_total_gb": total_b / (1024**3),
        }
    except Exception:
        return {}


def progress_config(config: dict[str, Any], *, default_heartbeat_s: float = 300.0) -> tuple[bool, float]:
    """Return ``(write_jsonl, heartbeat_every_s)`` from the config's logging block."""
    logging_cfg = config.get("logging") or {}
    progress_cfg = logging_cfg.get("progress") or {}
    write_jsonl = bool(progress_cfg.get("jsonl", logging_cfg.get("progress_jsonl", True)))
    heartbeat_every_s = float(
        progress_cfg.get(
            "heartbeat_every_s",
            logging_cfg.get("progress_heartbeat_every_s", default_heartbeat_s),
        )
    )
    return write_jsonl, heartbeat_every_s


class ProgressTracker:
    """Track loop progress for enriched log lines and JSONL traces."""

    def __init__(
        self,
        *,
        phase: str,
        total_work: int,
        total_queries: int | None,
        jsonl_path: Path,
        write_jsonl: bool = True,
        heartbeat_every_s: float = 300.0,
    ) -> None:
        """Create a tracker for one append-only eval loop."""
        self.phase = phase
        self.total_work = max(int(total_work), 0)
        self.total_queries = total_queries if total_queries is None else max(int(total_queries), 0)
        self.jsonl_path = jsonl_path
        self.write_jsonl = write_jsonl
        self.heartbeat_every_s = max(float(heartbeat_every_s), 0.0)
        self.started_work = 0
        self.completed_work = 0
        self.failed_work = 0
        self.skipped_work = 0
        self.start_monotonic = time.monotonic()
        self.last_heartbeat_monotonic = self.start_monotonic
        self.last_label: str | None = None

        if self.write_jsonl:
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            self.jsonl_path.write_text("", encoding="utf-8")

    @property
    def accounted_work(self) -> int:
        """Return work units that are no longer active or pending."""
        return self.completed_work + self.failed_work + self.skipped_work

    def start_unit(
        self,
        *,
        label: str,
        q_index: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> tuple[int, str]:
        """Register the next unit and return ``(work_index, progress_suffix)``."""
        next_index = max(self.started_work, self.accounted_work) + 1
        if self.total_work:
            next_index = min(next_index, self.total_work)
        self.started_work = next_index
        self.last_label = label
        suffix = self.format_suffix(work_index=next_index, q_index=q_index)
        self._write_record(
            {
                "event": "start",
                "label": label,
                "work_index": next_index,
                "q_index": q_index,
                **(extra or {}),
            }
        )
        return next_index, suffix

    def finish_unit(
        self,
        *,
        label: str,
        work_index: int,
        q_index: int | None = None,
        duration_s: float | None = None,
        success: bool = True,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Record one completed or failed work unit."""
        if success:
            self.completed_work += 1
        else:
            self.failed_work += 1
        self.last_label = label
        self._write_record(
            {
                "event": "finish",
                "label": label,
                "work_index": work_index,
                "q_index": q_index,
                "success": success,
                "duration_s": duration_s,
                **(extra or {}),
            }
        )

    def skip_units(
        self,
        *,
        count: int,
        label: str,
        q_index: int | None = None,
        reason: str,
    ) -> None:
        """Account for planned work units that the driver skipped."""
        count = max(int(count), 0)
        if count == 0:
            return
        self.skipped_work += count
        self.last_label = label
        self._write_record(
            {
                "event": "skip",
                "label": label,
                "q_index": q_index,
                "count": count,
                "reason": reason,
            }
        )

    def format_suffix(self, *, work_index: int | None = None, q_index: int | None = None) -> str:
        """Compact suffix for normal per-qid/per-permutation log lines."""
        display_work = self.accounted_work if work_index is None else work_index
        parts: list[str] = []
        if q_index is not None and self.total_queries is not None:
            parts.append(f"q={q_index}/{self.total_queries}")
        if self.total_work:
            pct = 100.0 * min(display_work, self.total_work) / self.total_work
            parts.append(f"work={display_work}/{self.total_work}")
            parts.append(f"{pct:.1f}%")
        else:
            parts.append("work=0/0")
        parts.append(f"eta={_format_duration(self._eta_s())}")
        return " ".join(parts)

    def maybe_heartbeat(self, logger: Any, *, force: bool = False) -> None:
        """Emit a sparse meta line; normal progress lines remain the main UX."""
        if self.heartbeat_every_s <= 0:
            return
        now = time.monotonic()
        if not force and (now - self.last_heartbeat_monotonic) < self.heartbeat_every_s:
            return
        self.last_heartbeat_monotonic = now

        elapsed = now - self.start_monotonic
        rate = self.accounted_work / elapsed if elapsed > 0 else 0.0
        avg = elapsed / self.accounted_work if self.accounted_work else None
        pieces = [
            f"[progress][{self.phase}]",
            self.format_suffix(),
            f"elapsed={_format_duration(elapsed)}",
            f"rate={rate:.3f}/s",
            f"avg={_format_duration(avg)}",
            f"failures={self.failed_work}",
            f"skipped={self.skipped_work}",
        ]
        if self.last_label:
            pieces.append(f"last={self.last_label}")
        gpu = _gpu_memory_gb()
        if gpu:
            pieces.append(
                "gpu_mem="
                f"{gpu['gpu_mem_alloc_gb']:.1f}GB_alloc/"
                f"{gpu['gpu_mem_reserved_gb']:.1f}GB_reserved "
                f"free={gpu['gpu_mem_free_gb']:.1f}/{gpu['gpu_mem_total_gb']:.1f}GB"
            )
        logger.info(" ".join(pieces))

    def _eta_s(self) -> float | None:
        elapsed = time.monotonic() - self.start_monotonic
        if self.accounted_work <= 0 or elapsed <= 0 or self.total_work <= 0:
            return None
        remaining = max(self.total_work - self.accounted_work, 0)
        return remaining / (self.accounted_work / elapsed)

    def _write_record(self, record: dict[str, Any]) -> None:
        if not self.write_jsonl:
            return
        now = time.monotonic()
        enriched = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "phase": self.phase,
            "total_work": self.total_work,
            "total_queries": self.total_queries,
            "completed_work": self.completed_work,
            "failed_work": self.failed_work,
            "skipped_work": self.skipped_work,
            "elapsed_s": now - self.start_monotonic,
            "eta_s": self._eta_s(),
            **record,
        }
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(enriched, sort_keys=True) + "\n")
