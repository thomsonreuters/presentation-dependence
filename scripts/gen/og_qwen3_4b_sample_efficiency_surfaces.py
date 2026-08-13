"""Shared surface registry for OG Qwen3-4B label-pool (N) sample-efficiency.

Used by eval-config generation, analyze, plot, and sweep bundling.
"""

from __future__ import annotations

from typing import Any

# Original 11-surface battery.
SAMPLE_EFF_SURFACES_ORIGINAL: tuple[str, ...] = (
    "dl19",
    "dl20",
    "trec-covid",
    "nfcorpus",
    "touche2020",
    "dbpedia-entity",
    "scifact",
    "signal1m",
    "trec-news",
    "robust04",
    "legal-a",
)

# Extension to the full 18-dataset reporting grid.
SAMPLE_EFF_SURFACES_EXTENSION: tuple[str, ...] = (
    "dl21",
    "dl22",
    "dl23",
    "fiqa",
    "arguana",
    "climate-fever",
    "legal-b",
)

SAMPLE_EFF_SURFACES: tuple[str, ...] = SAMPLE_EFF_SURFACES_ORIGINAL + SAMPLE_EFF_SURFACES_EXTENSION

# Metadata for analyze / plot (n_queries = full judged pool; tau_cap => N=100 cap file).
SURFACE_META: tuple[dict[str, Any], ...] = (
    {"id": "dl19", "label": "DL19", "group": "reducible", "n_queries": 43, "tau_cap": False},
    {"id": "dl20", "label": "DL20", "group": "reducible", "n_queries": 54, "tau_cap": False},
    {"id": "dl21", "label": "DL21", "group": "dl_track", "n_queries": 53, "tau_cap": False},
    {"id": "dl22", "label": "DL22", "group": "dl_track", "n_queries": 76, "tau_cap": False},
    {"id": "dl23", "label": "DL23", "group": "dl_track", "n_queries": 82, "tau_cap": False},
    {"id": "trec-covid", "label": "TREC-COVID", "group": "reducible", "n_queries": 50, "tau_cap": True},
    {"id": "nfcorpus", "label": "NFCorpus", "group": "reducible", "n_queries": 308, "tau_cap": True},
    {"id": "touche2020", "label": "Touche-2020", "group": "reducible", "n_queries": 49, "tau_cap": False},
    {"id": "dbpedia-entity", "label": "DBPedia", "group": "reducible", "n_queries": 400, "tau_cap": True},
    {"id": "scifact", "label": "SciFact", "group": "reducible", "n_queries": 300, "tau_cap": True},
    {"id": "fiqa", "label": "FiQA", "group": "reducible", "n_queries": 648, "tau_cap": True},
    {"id": "arguana", "label": "ArguAna", "group": "reducible", "n_queries": 1406, "tau_cap": True},
    {"id": "climate-fever", "label": "Climate-FEVER", "group": "reducible", "n_queries": 1535, "tau_cap": True},
    {"id": "signal1m", "label": "Signal-1M", "group": "gated", "n_queries": 97, "tau_cap": True},
    {"id": "trec-news", "label": "TREC-NEWS", "group": "gated", "n_queries": 57, "tau_cap": True},
    {"id": "robust04", "label": "Robust04", "group": "gated", "n_queries": 249, "tau_cap": True},
    {"id": "legal-a", "label": "Legal-A", "group": "irreducible", "n_queries": 97, "tau_cap": False},
    {"id": "legal-b", "label": "Legal-B", "group": "irreducible", "n_queries": 175, "tau_cap": True},
)

SURFACE_BY_ID = {s["id"]: s for s in SURFACE_META}

# Plot groupings: headline = all non-legal (16); legal = Legal-A + Legal-B.
NON_LEGAL_SURFACES: tuple[tuple[str, str], ...] = (
    ("dl19", "DL19"),
    ("dl20", "DL20"),
    ("dl21", "DL21"),
    ("dl22", "DL22"),
    ("dl23", "DL23"),
    ("trec-covid", "TREC-COVID"),
    ("nfcorpus", "NFCorpus"),
    ("touche2020", "Touche-2020"),
    ("dbpedia-entity", "DBPedia"),
    ("scifact", "SciFact"),
    ("fiqa", "FiQA"),
    ("arguana", "ArguAna"),
    ("climate-fever", "Climate-FEVER"),
    ("signal1m", "Signal-1M"),
    ("trec-news", "TREC-NEWS"),
    ("robust04", "Robust04"),
)
LEGAL_SURFACES: tuple[tuple[str, str], ...] = (
    ("legal-a", "Legal-A"),
    ("legal-b", "Legal-B"),
)

# Back-compat alias (headline figure uses NON_LEGAL_SURFACES).
MAIN_SURFACES = NON_LEGAL_SURFACES

DL_TRACK_SURFACES: tuple[tuple[str, str], ...] = (
    ("dl21", "DL21"),
    ("dl22", "DL22"),
    ("dl23", "DL23"),
)
REDUCIBLE_SURFACES: tuple[tuple[str, str], ...] = (
    ("trec-covid", "TREC-COVID"),
    ("dbpedia-entity", "DBPedia"),
    ("scifact", "SciFact"),
    ("climate-fever", "Climate-FEVER"),
)
LEGAL_B_SURFACES: tuple[tuple[str, str], ...] = (("legal-b", "Legal-B"),)
GATED_SURFACES: tuple[tuple[str, str], ...] = (
    ("signal1m", "Signal-1M"),
    ("trec-news", "TREC-NEWS"),
    ("robust04", "Robust04"),
)

ALL_PLOT_GROUPS: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = (
    ("Non-legal surfaces (headline)", NON_LEGAL_SURFACES),
    ("Legal (irreducible)", LEGAL_SURFACES),
)
