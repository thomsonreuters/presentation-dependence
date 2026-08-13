"""Three-way claim-verification labels and stability metrics.

Uses the labels SUPPORTED, REFUTED, and NEI. Climate-FEVER's DISPUTED class
remains separate; callers exclude it and report the exclusion count.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Sequence

VERDICT_LABELS = ("SUPPORTED", "REFUTED", "NEI")

_ALIASES = {
    "SUPPORT": "SUPPORTED",
    "SUPPORTED": "SUPPORTED",
    "SUPPORTS": "SUPPORTED",
    "ENTAILMENT": "SUPPORTED",
    "ENTAILS": "SUPPORTED",
    "CONTRADICT": "REFUTED",
    "CONTRADICTED": "REFUTED",
    "CONTRADICTION": "REFUTED",
    "REFUTE": "REFUTED",
    "REFUTED": "REFUTED",
    "REFUTES": "REFUTED",
    "NEI": "NEI",
    "NOT ENOUGH INFO": "NEI",
    "NOT ENOUGH INFORMATION": "NEI",
    "INSUFFICIENT EVIDENCE": "NEI",
    "INSUFFICIENT INFORMATION": "NEI",
}


def normalize_verdict(text: str) -> str | None:
    """Normalize a generated label, returning ``None`` for invalid output."""
    value = re.sub(r"[_-]+", " ", str(text or "").strip().upper())
    value = re.sub(r"[^A-Z ]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    if value in _ALIASES:
        return _ALIASES[value]
    # Be tolerant of a model adding a short prefix/suffix, but reject ambiguous
    # generations containing more than one label.
    matches = {_ALIASES[alias] for alias in _ALIASES if re.search(rf"\b{re.escape(alias)}\b", value)}
    return next(iter(matches)) if len(matches) == 1 else None


def verdict_accuracy(prediction: str | None, gold: str) -> float:
    """Return exact three-way verdict accuracy."""
    return 1.0 if prediction is not None and prediction == gold else 0.0


def verdict_flip_rate(predictions: Sequence[str | None]) -> float:
    """Fraction differing from the modal normalized verdict.

    Invalid generations are a separate category rather than silently mapped to
    NEI. This keeps malformed-output instability visible in both flip and
    ``invalid_rate``.
    """
    values = [p if p is not None else "<INVALID>" for p in predictions]
    if len(values) <= 1:
        return 0.0
    majority = Counter(values).most_common(1)[0][1]
    return (len(values) - majority) / len(values)


def accuracy_spread(predictions: Sequence[str | None], gold: str) -> float:
    """Return max-minus-min correctness over presentations."""
    values = [verdict_accuracy(p, gold) for p in predictions]
    return max(values) - min(values) if values else 0.0
