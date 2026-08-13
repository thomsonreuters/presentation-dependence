"""Inline config tokens and ``METRIC`` lines for reader container runs.

The container entrypoint is the only consumer. It accepts a config as one
shell-safe token instead of a path, and flattens results into
``METRIC name=value`` log lines.
"""

from __future__ import annotations

import base64
import json
from typing import Mapping

__all__ = [
    "SCORER_PARQUET_FILENAME",
    "reader_scalar_metrics",
    "encode_resolved_config",
    "decode_resolved_config",
]

SCORER_PARQUET_FILENAME = "beta_gamma_scores.parquet"

# Numeric R3 corpus fields reported for each k.
_R3_FIELDS = (
    "canonical_em",
    "canonical_f1",
    "mean_em",
    "mean_f1",
    "mean_answer_flip_rate",
    "mean_em_spread",
    "mean_f1_spread",
    "gold_in_topk_rate",
    "gold_slot_mean",
    "gold_slot_var",
)
_V3_FIELDS = (
    "canonical_accuracy",
    "mean_accuracy",
    "mean_verdict_flip_rate",
    "mean_accuracy_spread",
    "mean_invalid_rate",
    "evidence_in_topk_rate",
    "evidence_slot_mean",
    "evidence_slot_var",
)


def encode_resolved_config(config: Mapping) -> str:
    """Encode a config as one shell-safe token, for passing it inline."""
    raw = json.dumps(config, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii")


def decode_resolved_config(payload: str) -> dict:
    """Decode a config produced by :func:`encode_resolved_config`."""
    raw = base64.urlsafe_b64decode(payload.encode("ascii")).decode("utf-8")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("resolved reader config must decode to a mapping")
    return value


def reader_scalar_metrics(phase: str, payload: Mapping) -> dict[str, float]:
    """Flatten a reader result into ``{metric_name: value}`` for METRIC lines.

    ``payload`` is the ``corpus`` dict (``{k: {field: value}}``).
    """
    out: dict[str, float] = {}
    if phase == "R3":
        for k, cell in payload.items():
            for field in _R3_FIELDS:
                val = cell.get(field)
                if val is not None:
                    out[f"reader_{field}_k{k}"] = float(val)
    elif phase == "V3":
        for k, cell in payload.items():
            for field in _V3_FIELDS:
                value = cell.get(field)
                if value is not None:
                    out[f"verdict_{field}_k{k}"] = float(value)
    else:
        raise ValueError(f"unknown reader phase {phase!r}")
    return out
