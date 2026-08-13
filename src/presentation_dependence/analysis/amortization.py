"""Run classification and scalar harvesting for amortization analyses."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RUNS = PROJECT_ROOT / "runs"
SURFACES_CANONICAL_10 = [
    "dl19",
    "dl20",
    "nfcorpus",
    "fiqa",
    "touche2020",
    "arguana",
    "climate-fever",
    "legal-a",
    "legal-b",
]
SURFACES_EXTRA_9 = [
    "dl21",
    "dl22",
    "dl23",
    "trec-covid",
    "dbpedia-entity",
    "scifact",
    "signal1m",
    "trec-news",
    "robust04",
]
SURFACES = SURFACES_CANONICAL_10 + SURFACES_EXTRA_9
_SURFACE_MATCH = sorted(SURFACES, key=len, reverse=True)
SKIP_TOKENS = (
    "dummy",
    "neutral",
    "smoke",
    "readout",
    "shard",
    "qmark",
    "trunc",
    "yesno",
    "-pw-",
    "pointwise",
    "norope",
    "wheel",
    "baseonly",
)
EXCLUDE_SIZE_TOKENS = ("12b", "26b", "27b")
EXCLUDE_FAMILY_TOKENS = (
    "qwen35",
    "instruct",
    "reranker",
    "gpt54",
    "claude",
    "mxbai",
)
OC_SFT_TOKEN = "oc_sft"
K10_OC_SFT_TOKEN = "k10_oc_sft"
LEGACY_OC_SFT_TOKEN = "supcon"
LEGACY_K10_OC_SFT_TOKENS = ("k10supcon", "k10_supcon")


def normalize_recipe_token(recipe: str) -> str:
    """Normalize legacy OC-SFT recipe tokens used by historical artifacts."""
    if recipe == LEGACY_OC_SFT_TOKEN:
        return OC_SFT_TOKEN
    if recipe in LEGACY_K10_OC_SFT_TOKENS:
        return K10_OC_SFT_TOKEN
    return recipe


@dataclass(frozen=True)
class Parsed:
    """Canonical identity parsed from one historical run ID."""

    family: str
    size: str
    recipe: str
    surface: str
    stem: str
    run_id: str


@dataclass(frozen=True)
class Resolved:
    """One exact complete trial selected for a canonical cell."""

    family: str
    size: str
    recipe: str
    surface: str
    stem: str
    run_id: str
    ts_dir: Path
    harvested: dict[str, Any]


def detect_surface(name: str) -> str | None:
    """Return the longest canonical surface token in a run ID."""
    return next(
        (surface for surface in _SURFACE_MATCH if surface in name),
        None,
    )


def stem_of(name: str, surface: str) -> str:
    """Return the run prefix before a surface token."""
    prefix = name[: name.find(surface)]
    return re.sub(r"beir-$", "", prefix).rstrip("-")


def _qwen_size(name: str) -> str | None:
    return next(
        (size for size in ("1p7b", "4b", "8b", "14b", "32b") if f"qwen3-{size}-" in name),
        None,
    )


def parse_run(run_id: str) -> Parsed | None:  # noqa: C901
    """Classify canonical family, size, recipe, and surface from a run ID."""
    name = run_id.lower()
    if any(token in name for token in SKIP_TOKENS) or name.startswith(("first-stage-panel-", "se-")) or "-ts-" in name:
        return None
    surface = detect_surface(name)
    if surface is None:
        return None
    stem = stem_of(run_id, surface)

    if not any(token in name for token in EXCLUDE_FAMILY_TOKENS):
        remaps = (
            ("gemma4-lora-step1800", "gemma4", "e4b", "k10sft"),
            ("granite41-lora-step1800", "granite", "8b", "k10sft"),
            (
                "gemma4-e2b-supcon-k10-l500-step2305",
                "gemma4",
                "e2b",
                K10_OC_SFT_TOKEN,
            ),
            (
                "gemma4-e2b-supcon-l500-step1200",
                "gemma4",
                "e2b",
                OC_SFT_TOKEN,
            ),
            (
                "gemma4-e4b-supcon-k10-l500-step1000",
                "gemma4",
                "e4b",
                K10_OC_SFT_TOKEN,
            ),
        )
        for token, family, size, recipe in remaps:
            if token in name:
                return Parsed(family, size, recipe, surface, stem, run_id)
        if "gemma4-31b-" in name and "supcon-k10-l500-step2200" in name:
            return Parsed("gemma4", "31b", K10_OC_SFT_TOKEN, surface, stem, run_id)
        if "gemma4-31b-" in name and "supcon-k1-l200-step1200" in name:
            return Parsed("gemma4", "31b", OC_SFT_TOKEN, surface, stem, run_id)

    family = None
    size = None
    if "granite" in name:
        family = "granite"
        size = next(
            (
                value
                for value in ("30b", "3b", "8b")
                if any(
                    token in name
                    for token in (
                        f"granite41-{value}",
                        f"granite-41-{value}",
                        f"granite-{value}",
                    )
                )
            ),
            None,
        )
    elif "gemma4" in name:
        if any(token in name for token in EXCLUDE_SIZE_TOKENS):
            return None
        family = "gemma4"
        size = next(
            (value for value in ("e2b", "e4b", "31b") if value in name),
            "e4b",
        )
    elif "qwen3" in name:
        if any(token in name for token in EXCLUDE_SIZE_TOKENS + EXCLUDE_FAMILY_TOKENS):
            return None
        if "think" in name and "nonthink" not in name:
            return None
        family, size = "qwen3", _qwen_size(name)
    if family is None or size is None:
        return None

    recipe = None
    if LEGACY_OC_SFT_TOKEN in name:
        if not ("warmup" in name or "-wu-" in name):
            return None
        recipe = K10_OC_SFT_TOKEN if "k10-supcon" in name or "k10supcon" in name else OC_SFT_TOKEN
    elif "hyb" in name:
        recipe = "k10hybrid"
    elif re.search(r"k10[-_](sft|step|lora)", name) or "nt-k10-step" in name or "k10-lora" in name:
        recipe = "k10sft"
    elif re.search(r"k1[-_](sft|step|lora)", name) or "nt-k1-step" in name or "k1-lora" in name:
        recipe = "k1sft"
    elif "offshelf" in name or "cont-b20" in name:
        recipe = "base"
    if recipe is None:
        return None
    return Parsed(family, size, recipe, surface, stem, run_id)


def read_json(path: Path) -> dict[str, Any] | None:
    """Read a JSON mapping, returning None for missing or invalid input."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def harvest(trial: Path, *, require_self_consistency: bool = True) -> dict[str, Any] | None:
    """Harvest quality and robustness scalars from one exact trial."""
    metrics = read_json(trial / "metrics.json")
    psi = read_json(trial / "psi" / "psi_metrics.json")
    sc = read_json(trial / "psi" / "sc_metrics.json")
    if metrics is None or psi is None:
        return None
    if require_self_consistency and sc is None:
        return None
    if psi.get("tau_psi_context_batch_size") != 20:
        return None
    aggregate = psi.get("aggregate", {})
    tau_psi = aggregate.get("mean_tau_based_psi")
    if tau_psi is None:
        return None
    by_k = (sc or {}).get("by_K", {})
    curve = {}
    for key, node in by_k.items():
        value = node.get("metrics", {}).get("ndcg_cut_10")
        if value is not None:
            curve[int(key)] = float(value["mean"])
    if require_self_consistency and not {1, 10}.issubset(curve):
        return None
    perm_avg = aggregate.get("mean_per_perm_ndcg") if aggregate.get("k_cutoff_for_ndcg") == 10 else None
    return {
        "ndcg_k1_metrics": _optional_float(metrics.get("mean_ndcg_cut_10")),
        "ndcg_k1_sc": curve.get(1),
        "ndcg_k1_permavg": _optional_float(perm_avg),
        "ndcg_k10_sc": curve.get(10),
        "tau_psi": float(tau_psi),
        "score_var": _optional_float(aggregate.get("mean_score_variance")),
        "sc_curve": curve,
        "n_queries": metrics.get("n_queries"),
        "per_perm_ndcg": _optional_float(aggregate.get("mean_per_perm_ndcg")),
        "worst_perm_ndcg": _optional_float(aggregate.get("mean_worst_perm_ndcg")),
        "zeng_psi_corpus": _optional_float(aggregate.get("zeng_psi_corpus")),
    }


def latest_complete(run_root: Path) -> tuple[Path, dict[str, Any]] | None:
    """Return the newest complete trial under an experiment directory."""
    for trial in sorted(
        (path for path in run_root.iterdir() if path.is_dir()),
        key=lambda path: path.name,
        reverse=True,
    ):
        harvested = harvest(trial)
        if harvested is not None:
            return trial, harvested
    return None


def _negative_lexical(value: str) -> tuple[int, ...]:
    return tuple(-ord(character) for character in value)


def discover(
    *,
    runs_root: Path = RUNS,
) -> tuple[
    dict[tuple[str, str, str, str], Resolved],
    dict[str, dict[str, Any]],
]:
    """Resolve canonical cells from local runs for audit tooling.

    Stems are ranked by surface coverage only. Which lambda a trained cell used
    is never inferred from a run name: callers needing it read the recorded
    decision in
    ``configs/reproduction/evidence/lambda-selection/decisions.json``.
    """
    if not runs_root.is_dir():
        raise FileNotFoundError(f"runs dir not found: {runs_root}")
    candidates: dict[
        tuple[str, str, str],
        dict[str, dict[str, Resolved]],
    ] = defaultdict(lambda: defaultdict(dict))
    for run_root in sorted(runs_root.iterdir(), key=lambda path: path.name):
        if not run_root.is_dir():
            continue
        parsed = parse_run(run_root.name)
        if parsed is None:
            continue
        found = latest_complete(run_root)
        if found is None:
            continue
        trial, harvested = found
        candidates[(parsed.family, parsed.size, parsed.recipe)][parsed.stem][parsed.surface] = Resolved(
            parsed.family,
            parsed.size,
            parsed.recipe,
            parsed.surface,
            parsed.stem,
            parsed.run_id,
            trial,
            harvested,
        )

    resolved = {}
    debug = {}
    for (family, size, recipe), by_stem in candidates.items():
        ordered = sorted(
            by_stem,
            key=lambda stem: (
                -len(by_stem[stem]),
                len(stem),
                _negative_lexical(stem),
            ),
        )
        chosen = ordered[0]
        merged = {}
        for stem in ordered:
            for surface, row in by_stem[stem].items():
                key = (family, size, recipe, surface)
                if key not in resolved:
                    resolved[key] = row
                    merged[surface] = stem
        debug[f"{family}/{size}/{recipe}"] = {
            "chosen_stem": chosen,
            "n_surfaces": sum(stem == chosen for stem in merged.values()),
            "merged_stems": sorted({stem for stem in merged.values() if stem != chosen}),
            "n_surfaces_total": len(merged),
            "alternatives": {stem: sorted(by_stem[stem]) for stem in sorted(by_stem) if stem != chosen},
        }
    return resolved, debug
