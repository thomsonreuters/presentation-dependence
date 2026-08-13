"""Pinned prompt and extraction for the claim-verification reader."""

from __future__ import annotations

from typing import Mapping, Sequence

from presentation_dependence.reader.prompt import render_passages
from presentation_dependence.reader.verdict_eval import normalize_verdict

SYSTEM_PROMPT = (
    "You are a precise claim-verification assistant. Use only the provided "
    "evidence passages. Classify the claim as SUPPORTED, REFUTED, or NEI "
    "(not enough information). Output exactly one label."
)


def build_verdict_messages(
    claim: str,
    passages: Sequence[Mapping[str, object]],
) -> list[dict[str, str]]:
    """Render the fixed three-way verdict conversation."""
    user = (
        "Return exactly one of: SUPPORTED, REFUTED, NEI.\n\n"
        f"Evidence:\n{render_passages(passages)}\n\n"
        f"Claim: {str(claim).strip()}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def extract_verdict(generated_text: str) -> str | None:
    """Extract and normalize one generated verdict."""
    return normalize_verdict(generated_text)
