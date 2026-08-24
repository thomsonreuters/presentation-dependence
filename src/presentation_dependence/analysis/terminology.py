"""Canonical paper terminology for user-facing analysis output."""

from __future__ import annotations


PAPER_METHOD_LABELS = {
    "off-the-shelf": "Off the shelf",
    "capcal": "CapCal",
    "round-robin": "Round-robin",
    "bsc": "BSC",
    "jina": "jina-reranker-v3",
    "gpt-5.4": "GPT-5.4",
    "single-order": "Single-order",
    "order-averaged": "Order-averaged",
    "debias-first": "DebiasFirst",
    "permutation-augmentation": "Permutation augmentation",
    "oc-sft": "OC-SFT",
}

_METHOD_ALIASES = {
    "base": "off-the-shelf",
    "off-shelf": "off-the-shelf",
    "off-the-shelf": "off-the-shelf",
    "capcal": "capcal",
    "round-robin": "round-robin",
    "bsc": "bsc",
    "batched-self-consistency": "bsc",
    "jina": "jina",
    "jina-reranker-v3": "jina",
    "gpt54": "gpt-5.4",
    "gpt-5.4": "gpt-5.4",
    "k1-sft": "single-order",
    "k1sft": "single-order",
    "k=1-sft": "single-order",
    "single-order": "single-order",
    "single-order-distillation": "single-order",
    "k10-sft": "order-averaged",
    "k10sft": "order-averaged",
    "k=10-sft": "order-averaged",
    "order-averaged": "order-averaged",
    "order-averaged-distillation": "order-averaged",
    "debias-first": "debias-first",
    "debiasfirst": "debias-first",
    "position-augmentation": "permutation-augmentation",
    "posaug": "permutation-augmentation",
    "pos-aug-only": "permutation-augmentation",
    "shuffled-view-augmentation": "permutation-augmentation",
    "permutation-augmentation": "permutation-augmentation",
    "oc-sft": "oc-sft",
    "ocsft": "oc-sft",
    "supcon": "oc-sft",
    "supervised-consistency": "oc-sft",
}


def canonical_method_key(value: str) -> str:
    """Resolve an internal or historical method token to its paper key."""
    normalized = str(value).strip().lower().replace("_", "-").replace(" ", "-")
    try:
        return _METHOD_ALIASES[normalized]
    except KeyError as exc:
        raise KeyError(f"Unknown method terminology token: {value!r}") from exc


def paper_method_label(value: str) -> str:
    """Return the canonical display label used in the paper."""
    return PAPER_METHOD_LABELS[canonical_method_key(value)]


__all__ = ["PAPER_METHOD_LABELS", "canonical_method_key", "paper_method_label"]
