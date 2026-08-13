#!/usr/bin/env python
"""Per-dataset scaling grid for base / K=1 SFT / K=10 SFT / OC-SFT, from runs/.

Emits the 18-dataset (Table T2) per-dataset nDCG@10 / tau-PSI@B=20 grid for the four
recipes, for all 11 cross-family scaling students.

Ground truth = ``runs/`` (metrics.json + psi/psi_metrics.json). We reuse the
amortization-law harvester's ``parse_run`` (canonical family/size/recipe
classification, legacy remaps, ablation skip-list) via import, and its
``discover`` to read the canonical checkpoint stem per
(family, size, recipe). Because the ladder eval and the extension eval of the
same checkpoint use different job stems, we UNION across stems at the canonical
training step (and lambda for OC-SFT); ``base`` (off-shelf) has no step so we
union all off-shelf runs for the model.

Gemma-31B is excluded by ``parse_run`` (EXCLUDE_SIZE_TOKENS) so it is harvested
with explicit stems. Ladder-9 columns are overlaid from ``rows.json`` (base,
k10sft, oc_sft) so the already-published columns stay byte-consistent; the 9
extension columns come from runs/. Read-only apart from artifacts under
``build/reproduction/analysis/scaling_per_dataset/`` and reporting
fragments under ``build/reproduction/analysis/scaling_per_dataset-report/``.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

from presentation_dependence.analysis import amortization as azl

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUNS = ROOT / "runs"
ROWS = ROOT / "build" / "reproduction" / "analysis" / "amortization_law" / "rows.json"
OUT_DIR = ROOT / "build" / "reproduction" / "analysis" / "scaling_per_dataset"
REPORT_DIR = ROOT / "build" / "reproduction" / "analysis" / "scaling_per_dataset-report"

PUBLIC = [
    "DL19",
    "DL20",
    "DL21",
    "DL22",
    "DL23",
    "Touche",
    "FiQA",
    "NFCorpus",
    "ArguAna",
    "Climate",
    "T-COVID",
    "DBPedia",
    "SciFact",
    "Signal1M",
    "T-NEWS",
    "Robust04",
]
LEGAL = ["Legal-A", "Legal-B"]
DS18 = PUBLIC + LEGAL
STANDARD_IR = ["DL19", "DL20", "Touche", "FiQA", "NFCorpus", "ArguAna", "Climate"]
NINE = STANDARD_IR + LEGAL

# Canonical amortization surface token -> report column.
TOK2COL = {
    "dl19": "DL19",
    "dl20": "DL20",
    "dl21": "DL21",
    "dl22": "DL22",
    "dl23": "DL23",
    "touche2020": "Touche",
    "fiqa": "FiQA",
    "nfcorpus": "NFCorpus",
    "arguana": "ArguAna",
    "climate-fever": "Climate",
    "trec-covid": "T-COVID",
    "dbpedia-entity": "DBPedia",
    "scifact": "SciFact",
    "signal1m": "Signal1M",
    "trec-news": "T-NEWS",
    "robust04": "Robust04",
    "legal-a": "Legal-A",
    "legal-b": "Legal-B",
}
# All 18 tokens for parse_run surface detection (longest first).
ALL_TOKENS = sorted(TOK2COL, key=len, reverse=True)

MODELS = [
    ("qwen3", "1p7b"),
    ("qwen3", "4b"),
    ("qwen3", "8b"),
    ("qwen3", "14b"),
    ("qwen3", "32b"),
    ("gemma4", "e2b"),
    ("gemma4", "e4b"),
    ("gemma4", "31b"),
    ("granite", "3b"),
    ("granite", "8b"),
    ("granite", "30b"),
]
LABEL = {
    ("qwen3", "1p7b"): "Qwen3-1.7B",
    ("qwen3", "4b"): "Qwen3-4B",
    ("qwen3", "8b"): "Qwen3-8B",
    ("qwen3", "14b"): "Qwen3-14B",
    ("qwen3", "32b"): "Qwen3-32B",
    ("gemma4", "e2b"): "Gemma-E2B",
    ("gemma4", "e4b"): "Gemma-E4B",
    ("gemma4", "31b"): "Gemma-31B",
    ("granite", "3b"): "Granite-3B",
    ("granite", "8b"): "Granite-8B",
    ("granite", "30b"): "Granite-30B",
}
RECIPES = ["base", "k1sft", "k10sft", "oc_sft"]
RECIPE_LABEL = {"base": "off-shelf", "k1sft": "K=1 SFT", "k10sft": "K=10 SFT", "oc_sft": "OC-SFT"}

STEP_RE = re.compile(r"step(\d+)")
LAM_RE = re.compile(r"-l(\d+)\b")


# OC-SFT lambda and step come from the recorded selection, never from a run
# name. Auto-discovery ranks stems by surface coverage, which can land on a
# lambda the protocol did not select, so the cell is looked up here instead.
DECISIONS = ROOT / "configs/reproduction/evidence/lambda-selection/decisions.json"
OC_SFT_DECISION_CELL = {
    ("qwen3", "1p7b"): ("Qwen3-1.7B-OG (non-thinking)", "K=1"),
    ("qwen3", "4b"): ("Qwen3-4B-OG (non-thinking)", "K=1"),
    ("qwen3", "8b"): ("Qwen3-8B-OG (non-thinking)", "K=1"),
    ("qwen3", "14b"): ("Qwen3-14B-OG (non-thinking)", "K=1"),
    ("qwen3", "32b"): ("Qwen3-32B-OG (non-thinking)", "K=1"),
    ("gemma4", "e2b"): ("Gemma-4 (E2B-it)", "K=1"),
    ("gemma4", "e4b"): ("Gemma-4 (E4B-it)", "K=1"),
    ("gemma4", "31b"): ("Gemma-4 31B", "K=1"),
    ("granite", "3b"): ("Granite-4.1-3B", "K=1"),
    ("granite", "8b"): ("Granite-4.1-8B", "K=1"),
    ("granite", "30b"): ("Granite-4.1-30B", "K=1"),
}


CANONICAL_OC_SFT_RECIPE = "OC-SFT warmup-500"


def recorded_oc_sft_selection() -> dict[tuple[str, str], dict]:
    """Return the recorded OC-SFT lambda and step for each declared cell.

    A family and K can appear several times: iterative continuations inherit
    lambda rather than selecting it (``is_lambda_selection: false``), and some
    families also record cross-family-silver or hybrid variants. Only genuine
    selections count, and where several remain the canonical OC-SFT recipe wins.
    """
    rows = json.loads(DECISIONS.read_text(encoding="utf-8"))
    by_cell: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        if row.get("lambda_star") is None or row.get("is_lambda_selection") is False:
            continue
        by_cell.setdefault((str(row["family"]), str(row["k"])), []).append(row)

    out: dict[tuple[str, str], dict] = {}
    for model, cell in OC_SFT_DECISION_CELL.items():
        found = by_cell.get(cell, [])
        if len(found) > 1:
            preferred = [r for r in found if r.get("recipe") == CANONICAL_OC_SFT_RECIPE]
            found = preferred or found
        picks = {(float(r["lambda_star"]), r.get("lambda_star_step")) for r in found}
        if not picks:
            raise SystemExit(f"no recorded lambda decision for {cell} in {DECISIONS}")
        if len(picks) > 1:
            raise SystemExit(
                f"ambiguous recorded decision for {cell}: {sorted(picks)}; "
                "disambiguate by recipe in the decision record"
            )
        lam, step = picks.pop()
        out[model] = {"lam": int(round(lam * 100)), "step": step}
    return out


def canonical_checkpoints() -> dict[tuple[str, str, str], dict]:
    """Map (family, size, recipe) to its selected checkpoint."""
    _resolved, debug = azl.discover()
    out: dict[tuple[str, str, str], dict] = {}
    for key, d in debug.items():
        parts = key.split("/")
        if len(parts) != 3:
            continue
        fam, size, recipe = parts
        stem = d.get("chosen_stem", "") or ""
        st = STEP_RE.search(stem)
        lm = LAM_RE.search(stem)
        out[(fam, size, recipe)] = {
            "step": int(st.group(1)) if st else None,
            "lam": int(lm.group(1)) if lm else None,
        }
    for model, selection in recorded_oc_sft_selection().items():
        out[(model[0], model[1], "oc_sft")] = dict(selection)
    # Gemma-E4B K=1 SFT only exists as a merged ckpt (sftk1-merged-step1600),
    # which discover() over the 9 ladder surfaces never resolved.
    out.setdefault(("gemma4", "e4b", "k1sft"), {"step": 1600, "lam": None})
    return out


def _gemma_merged_sft_fallback(name: str):
    """Catch Gemma E2B/E4B ``sftk1``/``sftk10`` merged-checkpoint eval names that
    ``parse_run`` misses (its k-sft regex expects ``k1-{sft,step,lora}``).
    """
    if "gemma4" not in name or azl.LEGACY_OC_SFT_TOKEN in name or "-ts-" in name or "26b" in name:
        return None
    m = re.search(r"gemma4-(e2b|e4b)-(sftk1|sftk10)-merged", name)
    if not m:
        return None
    size = m.group(1)
    recipe = "k1sft" if m.group(2) == "sftk1" else "k10sft"
    surface = azl.detect_surface(name.lower())
    if surface is None:
        return None
    return azl.Parsed("gemma4", size, recipe, surface, azl.stem_of(name, surface), name)


def relaxed_harvest(ts_dir: Path) -> dict | None:
    metrics = azl.read_json(ts_dir / "metrics.json")
    if not metrics or metrics.get("mean_ndcg_cut_10") is None:
        return None
    out = {"ndcg": float(metrics["mean_ndcg_cut_10"]), "psi": None}
    psi = azl.read_json(ts_dir / "psi" / "psi_metrics.json")
    if psi and psi.get("tau_psi_context_batch_size") == 20:
        agg = psi.get("aggregate", {})
        if agg.get("mean_tau_based_psi") is not None:
            out["psi"] = float(agg["mean_tau_based_psi"])
    return out


def latest(run_dir: Path) -> dict | None:
    ts_dirs = sorted((d for d in run_dir.iterdir() if d.is_dir()), key=lambda p: p.name, reverse=True)
    fallback = None
    for ts in ts_dirs:
        h = relaxed_harvest(ts)
        if h is None:
            continue
        if h["psi"] is not None:
            return {"ts": ts.name, **h}
        if fallback is None:
            fallback = {"ts": ts.name, **h}
    return fallback


def collect_via_parse_run(canon) -> dict:  # noqa: C901
    """base/k1sft/k10sft/oc_sft for the 10 non-31B models, unioned across stems."""
    azl._SURFACE_MATCH[:] = ALL_TOKENS  # patch surface detection to all 19
    azl.SURFACES[:] = ALL_TOKENS
    cand: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for run_dir in sorted(RUNS.iterdir(), key=lambda p: p.name):
        n = run_dir.name
        # Accept "-vllm-psi" plus common suffixed variants ("-rerun", "-offshelf").
        if not run_dir.is_dir() or "-vllm-psi" not in n:
            continue
        # Exclude non-canonical experiments that parse_run does NOT skip:
        #  - First-stage dense/learned-sparse runs
        #    (not the BM25 ladder). Mixing those with the BM25 ladder confounds
        #    the comparison.
        #  - Sample-efficiency subsampled-N sweep runs (non-30k pools).
        #  - CapCal calibration-wrapper runs.
        if n.startswith(("first-stage-panel-", "SE-")) or "-bge-" in n or "-splade-" in n:
            continue
        if any(t in n for t in ("-qb1", "-qb16")):
            continue
        parsed = azl.parse_run(run_dir.name)
        if parsed is None:
            parsed = _gemma_merged_sft_fallback(run_dir.name)
        if parsed is None:
            continue
        key = (parsed.family, parsed.size)
        if key not in LABEL or parsed.recipe not in RECIPES:
            continue
        ck = canon.get((parsed.family, parsed.size, parsed.recipe))
        if ck is None:
            continue
        if parsed.recipe != "base":
            st = STEP_RE.search(parsed.stem)
            if not st or (ck["step"] is not None and int(st.group(1)) != ck["step"]):
                continue
            if parsed.recipe == "oc_sft" and ck["lam"] is not None:
                lm = LAM_RE.search(parsed.stem)
                allowed = {ck["lam"]}
                # Qwen3-14B OC-SFT: ladder-9 is λ=300 (canonical); the extension
                # surfaces only exist on a λ=500 checkpoint (same step1400). Accept
                # both so the 14B row is complete.
                if (parsed.family, parsed.size) == ("qwen3", "14b"):
                    allowed = {300, 500}
                if not lm or int(lm.group(1)) not in allowed:
                    continue
        col = TOK2COL.get(parsed.surface)
        if col is None:
            continue
        got = latest(run_dir)
        if got:
            cand[(parsed.family, parsed.size)][parsed.recipe][col].append({"run_id": run_dir.name, **got})
    return cand


_OC_SFT_EXCLUDE = (
    "k10",
    "hybrid",
    "sft",
    "-bge-",
    "-splade-",
    "genbsc",
    "gpt54",
    "-ts-",
    "merged",
    "26b-a4b",
    "-b1-",
    "-b10-",
    "debias",
    "capcal",
    "baseonly",
    "-qb1",
    "-qb16",
    "dummy",
    "neutral",
    "yesno",
)


def _model_of_oc_sft(name: str):
    """Return family/size for a legacy-named OC-SFT run, including Gemma-31B."""
    if azl.LEGACY_OC_SFT_TOKEN not in name:
        return None
    checks = [
        (("qwen3", "1p7b"), r"qwen3-1p7b"),
        (("qwen3", "4b"), r"qwen3-4b"),
        (("qwen3", "8b"), r"qwen3-8b"),
        (("qwen3", "14b"), r"qwen3-14b"),
        (("qwen3", "32b"), r"qwen3-32b"),
        (("gemma4", "e2b"), r"gemma4-e2b"),
        (("gemma4", "31b"), r"gemma4-31b"),
        (("gemma4", "e4b"), r"gemma4-e4b"),
        (("granite", "3b"), r"granite-?41-3b"),
        (("granite", "8b"), r"granite-?41-8b"),
        (("granite", "30b"), r"granite-?41-30b"),
    ]
    for key, pat in checks:
        if re.search(pat, name):
            return key
    return None


def collect_oc_sft(canon) -> dict:  # noqa: C901
    """OC-SFT via (family,size,λ,step) matching (no `warmup`-token requirement,
    unlike parse_run) so extension stems like ``gemma4-e2b-supcon-l500-step1200``
    are not dropped. Qwen3-14B accepts λ∈{300,500}.
    """
    cls = {(f, s): (v["lam"], v["step"]) for (f, s, r), v in canon.items() if r == "oc_sft"}
    cls[("gemma4", "31b")] = (200, 1200)
    per: dict = defaultdict(lambda: defaultdict(list))
    for run_dir in sorted(RUNS.iterdir(), key=lambda p: p.name):
        n = run_dir.name
        if not run_dir.is_dir() or "-vllm-psi" not in n or azl.LEGACY_OC_SFT_TOKEN not in n:
            continue
        if n.startswith(("first-stage-panel-", "SE-")) or any(x in n for x in _OC_SFT_EXCLUDE):
            continue
        mk = _model_of_oc_sft(n)
        if mk is None or mk not in cls:
            continue
        lam, step = cls[mk]
        st = STEP_RE.search(n)
        lm = LAM_RE.search(n)
        if not st or int(st.group(1)) != step:
            continue
        allowed = {lam} if lam is not None else set()
        if mk == ("qwen3", "14b"):
            allowed = {300, 500}
        if lam is not None and (not lm or int(lm.group(1)) not in allowed):
            continue
        col = None
        for t in ALL_TOKENS:
            if t in n:
                col = TOK2COL[t]
                break
        if col is None:
            continue
        got = latest(run_dir)
        if got:
            per[mk][col].append({"run_id": n, **got})
    return per


def collect_gemma31b() -> dict:  # noqa: C901
    """Gemma-31B (excluded by parse_run): explicit per-recipe stems."""
    cand: dict = defaultdict(lambda: defaultdict(list))
    for run_dir in sorted(RUNS.iterdir(), key=lambda p: p.name):
        n = run_dir.name
        if not run_dir.is_dir() or "-vllm-psi" not in n:
            continue
        if (
            "gemma4-31b" not in n
            or "ts-" in n
            or "-bge-" in n
            or "-splade-" in n
            or any(t in n for t in ("-qb1", "-qb16"))
        ):
            continue
        recipe = None
        if "sft-k1-step1600" in n:
            recipe = "k1sft"
        elif "sft-k10-step1200" in n:
            recipe = "k10sft"
        elif "supcon-k1-l200-step1200" in n:
            recipe = "oc_sft"
        elif (
            ("setwise-grade-cont-b20" in n or "newsurf-offshelf" in n)
            and azl.LEGACY_OC_SFT_TOKEN not in n
            and "sft" not in n
        ):
            recipe = "base"
        if recipe is None:
            continue
        col = None
        for t in ALL_TOKENS:
            if t in n:
                col = TOK2COL[t]
                break
        if col is None:
            continue
        got = latest(run_dir)
        if got:
            cand[recipe][col].append({"run_id": n, **got})
    return cand


def dedup(cands: list[dict]) -> dict | None:
    if not cands:
        return None
    withpsi = [c for c in cands if c["psi"] is not None]
    pool = withpsi or cands
    return sorted(pool, key=lambda c: c["ts"], reverse=True)[0]


def load_rows_json():
    """Ladder-9 authoritative overlays: base (base_* fields), k10sft, oc_sft."""
    rows = json.loads(ROWS.read_text())
    base: dict = defaultdict(dict)
    ksft: dict = defaultdict(dict)
    sup: dict = defaultdict(dict)
    for r in rows:
        key = (r["family"], r["size"])
        if key not in LABEL:
            continue
        col = TOK2COL.get(r["dataset"])
        if col is None:
            continue
        base[key].setdefault(col, {"ndcg": r.get("ndcg_base_k1_metrics"), "psi": r.get("psi_base")})
        if r.get("recipe") == "k10sft":
            ksft[key][col] = {"ndcg": r.get("ndcg_y_k1_metrics"), "psi": r.get("psi_y")}
        if azl.normalize_recipe_token(str(r.get("recipe"))) == "oc_sft":
            sup[key][col] = {"ndcg": r.get("ndcg_y_k1_metrics"), "psi": r.get("psi_y")}
    return {"base": base, "k10sft": ksft, "oc_sft": sup}


def build() -> dict:
    canon = canonical_checkpoints()
    raw = collect_via_parse_run(canon)
    g31 = collect_gemma31b()
    # Dedicated OC-SFT collector.
    oc_sft = collect_oc_sft(canon)
    rowsj = load_rows_json()

    data: dict = {r: defaultdict(dict) for r in RECIPES}
    for mk in MODELS:
        for recipe in RECIPES:
            if recipe == "oc_sft":
                per_surf = oc_sft.get(mk, {})
            elif mk == ("gemma4", "31b"):
                per_surf = g31.get(recipe, {})
            else:
                per_surf = raw.get(mk, {}).get(recipe, {})
            for col in DS18:
                pick = dedup(per_surf.get(col, []))
                if pick:
                    data[recipe][mk][col] = {
                        "ndcg": pick["ndcg"],
                        "psi": pick["psi"],
                        "run_id": pick["run_id"],
                        "ts": pick["ts"],
                    }
            # overlay rows.json on the 9 ladder surfaces
            overlay = rowsj.get(recipe, {}).get(mk, {})
            for col, rc in overlay.items():
                if col in NINE and rc["ndcg"] is not None:
                    data[recipe][mk][col] = {"ndcg": rc["ndcg"], "psi": rc["psi"], "run_id": "rows.json", "ts": ""}
    return data


DOC_PATH = REPORT_DIR / "summary.md"


def mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def _cell(c) -> str:
    if not c:
        return "—"
    psi = f"{c['psi']:.3f}" if c["psi"] is not None else "—"
    return f"{c['ndcg']:.3f}/{psi}"


def write_doc(data) -> None:
    """Compose one self-contained doc with all four recipe grids + summary."""
    L: list[str] = []
    L.append("# Cross-family scaling: per-dataset results (all recipes)")
    L.append("")
    L.append(
        "Generated by `scripts/analyze/scaling_recipes_audit.py` (read-only harvest of "
        "`runs/`). Each cell is **nDCG@10 / τ-PSI@B=20**; `—` = no run, "
        "`x/—` = nDCG only (no B=20 τ-PSI). Single seed."
    )
    L.append("")
    L.append(
        "- **Ground truth**: per-run `metrics.json` + `psi/psi_metrics.json` under "
        "`runs/`. Ladder-9 columns (7 standard-IR + 2 legal) are taken from "
        "`build/reproduction/analysis/amortization_law/rows.json`; "
        "the other 9 columns are "
        "harvested from `runs/`."
    )
    L.append(
        "- **Checkpoint selection**: the canonical OC-SFT lambda and step "
        "are read from the recorded decision in "
        "`configs/reproduction/evidence/lambda-selection/decisions.json`, not "
        "inferred from run names; SFT checkpoints come from the run stem. "
        "Extension surfaces are unioned across job stems at that same (step, λ)."
    )
    L.append(
        "- **Datasets (Table T2, 18)**: 16 public = 5 TREC-DL (DL19–23) + 11 "
        "BEIR/TREC; 2 legal = Legal-A/Legal-B. `nine` = 7 standard-IR + 2 legal "
        "(the legacy comparable mean)."
    )
    L.append(
        "- **Machine-readable**: `build/reproduction/analysis/"
        "scaling_per_dataset/"
        "scaling_recipes_per_dataset.csv` (with per-cell run provenance)."
    )
    L.append("")
    # §1 comparison
    L.append("## 1. Cross-recipe summary — nine-dataset mean (comparable across all models)")
    L.append("")
    L.append("nDCG@10 / τ-PSI. The only mean available for every model × recipe.")
    L.append("")
    L.append("| Model | off-shelf | K=1 SFT | K=10 SFT | OC-SFT |")
    L.append("| --- | --- | --- | --- | --- |")
    for mk in MODELS:
        cols = []
        for recipe in RECIPES:
            d = data[recipe][mk]
            nine = mean([d[c]["ndcg"] for c in NINE if c in d])
            tps = mean([d[c]["psi"] for c in NINE if c in d and d[c]["psi"] is not None])
            cols.append(f"{_f(nine)}/{_f(tps)}" if nine is not None else "—")
        L.append(f"| {LABEL[mk]} | {cols[0]} | {cols[1]} | {cols[2]} | {cols[3]} |")
    L.append("")
    # §2-5 per-recipe grids
    for i, recipe in enumerate(RECIPES, start=2):
        d = data[recipe]
        L.append(f"## {i}. {RECIPE_LABEL[recipe]} — per-dataset (nDCG@10 / τ-PSI)")
        L.append("")
        L.append("| Model | " + " | ".join(DS18) + " | avg16pub | avg18 | nine | ninePSI |")
        L.append("| --- | " + " | ".join("---" for _ in range(len(DS18) + 4)) + " |")
        for mk in MODELS:
            cells = [_cell(d[mk].get(c)) for c in DS18]
            a16 = mean([d[mk][c]["ndcg"] for c in PUBLIC if c in d[mk]])
            a18 = mean([d[mk][c]["ndcg"] for c in DS18 if c in d[mk]])
            nine = mean([d[mk][c]["ndcg"] for c in NINE if c in d[mk]])
            tps = mean([d[mk][c]["psi"] for c in NINE if c in d[mk] and d[mk][c]["psi"] is not None])
            L.append(f"| {LABEL[mk]} | " + " | ".join(cells) + f" | {_f(a16)} | {_f(a18)} | {_f(nine)} | {_f(tps)} |")
        if recipe == "oc_sft":
            L.append("")
            L.append(
                "> **Qwen3-14B row**: the 9 ladder columns (DL19/DL20, Touche, "
                "FiQA, NFCorpus, ArguAna, Climate, Legal-A, Legal-B) are the "
                "canonical **λ=300** checkpoint; the extension columns **DL21, "
                "DL22, DL23, T-COVID, DBPedia, SciFact, Signal1M** are from a "
                "**λ=500** checkpoint (same step 1400, different λ — the λ=300 "
                "checkpoint was never evaluated on those surfaces). T-NEWS/Robust04 "
                "were not run at either λ. `avg`/`nine` for 14B mix the two "
                "λ; read with care."
            )
        L.append("")
    # §6 caveats
    L.append("## 6. Coverage caveats")
    L.append("")
    L.append(
        "Under the canonical **BM25** first stage, the only genuinely-missing cells "
        "are the two access-gated NIST surfaces (T-NEWS, Robust04) and two "
        "model-specific holes:"
    )
    L.append("")
    L.append(
        "- **T-NEWS + Robust04 τ-PSI**: computed (BM25) for **Qwen3-4B** (all recipes "
        "with cells) and **Gemma-E4B K=1 SFT** (merged s1600, `trec2-gemma4-e4b-sftk1`). "
        "For Qwen3-32B / Gemma-E2B/E4B (K=10 SFT + OC-SFT) / Gemma-31B / Granite-3B/8B/30B "
        "the T-NEWS/Robust04 evals are quality-only (nDCG present, the B=20 PSI harness was "
        "not run). No self-distill run at all for Qwen3-1.7B/8B/14B on these two. (Note: "
        "under a **BGE dense** first stage — the Stage-1 experiment — T-NEWS/Robust04 τ-PSI "
        "does exist for Qwen3-4B and Granite-8B, but that is a different first stage and is "
        "deliberately excluded here.)"
    )
    L.append(
        "- **Qwen3-14B OC-SFT**: row combines two checkpoints — ladder-9 at "
        "λ=300 (canonical) and the 7 extension surfaces at λ=500 (the λ=300 "
        "checkpoint was never evaluated there). Its `avg`/`nine` mix λ; "
        "see the note under the OC-SFT table."
    )
    L.append(
        "- **`avg16pub` / `avg18` are NOT comparable across rows** with different "
        "coverage (n varies); use `nine` for cross-model/recipe comparison."
    )
    L.append(
        "- Ladder-9 numbers match `rows.json`; the runs/ harvest agrees "
        "to ≤0.003 nDCG on the overlap. First-stage and sample-efficiency "
        "(subsampled-N) runs are excluded so only the canonical BM25 ladder is used."
    )
    L.append("")
    DOC_PATH.write_text("\n".join(L) + "\n", encoding="utf-8")


def _f(x, nd=3):
    return f"{x:.{nd}f}" if x is not None else "—"


def main() -> int:  # noqa: C901
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    missing = [path for path in (RUNS, ROWS) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Scaling legacy analysis requires fetched runs and the generated "
            "amortization rows input; missing: " + ", ".join(str(path) for path in missing)
        )
    data = build()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    for recipe in RECIPES:
        d = data[recipe]
        print(f"\n########## {RECIPE_LABEL[recipe]} ##########")
        print(
            f"{'Model':12s} "
            + " ".join(f"{c:>9s}" for c in DS18)
            + f" {'16pub':>6s} {'18':>6s} {'nine':>6s} {'ninePSI':>7s}"
        )
        for mk in MODELS:
            cells = []
            for c in DS18:
                cell = d[mk].get(c)
                if cell:
                    psi = f"{cell['psi']:.2f}" if cell["psi"] is not None else "—"
                    cells.append(f"{cell['ndcg']:.2f}/{psi}")
                else:
                    cells.append("—")
            a16 = mean([d[mk][c]["ndcg"] for c in PUBLIC if c in d[mk]])
            a18 = mean([d[mk][c]["ndcg"] for c in DS18 if c in d[mk]])
            nine = mean([d[mk][c]["ndcg"] for c in NINE if c in d[mk]])
            tps = mean([d[mk][c]["psi"] for c in NINE if c in d[mk] and d[mk][c]["psi"] is not None])
            print(
                f"{LABEL[mk]:12s} "
                + " ".join(f"{c:>9s}" for c in cells)
                + f" {_f(a16):>6s} {_f(a18):>6s} {_f(nine):>6s} {_f(tps):>7s}"
            )

    # CSV: long form across recipes
    with open(OUT_DIR / "scaling_recipes_per_dataset.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["recipe", "model", "dataset", "ndcg_cut_10", "tau_psi_b20", "run_id", "timestamp"])
        for recipe in RECIPES:
            for mk in MODELS:
                for c in DS18:
                    cell = data[recipe][mk].get(c)
                    if cell:
                        w.writerow(
                            [
                                recipe,
                                LABEL[mk],
                                c,
                                f"{cell['ndcg']:.4f}",
                                "" if cell["psi"] is None else f"{cell['psi']:.4f}",
                                cell["run_id"],
                                cell["ts"],
                            ]
                        )
    # per-recipe markdown grids
    for recipe in RECIPES:
        d = data[recipe]
        lines = [
            "| Model | " + " | ".join(DS18) + " | avg16pub | avg18 | nine | ninePSI |",
            "| --- | " + " | ".join("---" for _ in range(len(DS18) + 4)) + " |",
        ]
        for mk in MODELS:
            cells = []
            for c in DS18:
                cell = d[mk].get(c)
                if cell:
                    psi = f"{cell['psi']:.3f}" if cell["psi"] is not None else "—"
                    cells.append(f"{cell['ndcg']:.3f}/{psi}")
                else:
                    cells.append("—")
            a16 = mean([d[mk][c]["ndcg"] for c in PUBLIC if c in d[mk]])
            a18 = mean([d[mk][c]["ndcg"] for c in DS18 if c in d[mk]])
            nine = mean([d[mk][c]["ndcg"] for c in NINE if c in d[mk]])
            tps = mean([d[mk][c]["psi"] for c in NINE if c in d[mk] and d[mk][c]["psi"] is not None])
            lines.append(
                f"| {LABEL[mk]} | " + " | ".join(cells) + f" | {_f(a16)} | {_f(a18)} | {_f(nine)} | {_f(tps)} |"
            )
        (REPORT_DIR / f"scaling_{recipe}_per_dataset.md").write_text("\n".join(lines) + "\n")

    write_doc(data)
    print(f"\nwrote: {OUT_DIR / 'scaling_recipes_per_dataset.csv'}")
    print("       per-recipe md: " + ", ".join(f"scaling_{recipe}_per_dataset.md" for recipe in RECIPES))
    print(f"       doc: {DOC_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
