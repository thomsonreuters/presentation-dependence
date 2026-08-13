"""Semantic and legacy identifiers for claim-verdict bridge runs."""

from __future__ import annotations

VERDICT_BRIDGE_PREFIX = "verdict-flip-bridge"
LEGACY_VERDICT_RUN_PREFIX_ALIASES = (
    "T2.1-verdict",
    "T2.5-verdict",
    "Table2-verdict",
)


def semantic_verdict_run_id(run_id: str) -> str:
    """Return the semantic identifier for a verdict-bridge run."""
    for prefix in LEGACY_VERDICT_RUN_PREFIX_ALIASES:
        if run_id == prefix:
            return VERDICT_BRIDGE_PREFIX
        if run_id.startswith(f"{prefix}-"):
            return f"{VERDICT_BRIDGE_PREFIX}{run_id[len(prefix) :]}"
    return run_id
