#!/usr/bin/env python
"""Harvest nDCG@10 + tau-PSI@B=20 for the OG Qwen3-4B sample-efficiency grid.

Reads PSI eval artifacts (local ``runs/`` after fetch) and writes per-cell JSON
records. Reuses legacy passage-reranking run directories for 30K endpoints when
sample-efficiency configs have not been run yet.

Example::

    uv run python -m scripts.analyze.analyze_og_qwen3_4b_sample_efficiency
    uv run python -m scripts.analyze.analyze_og_qwen3_4b_sample_efficiency --write-provenance
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
# Importable as `scripts.*` whether this runs as a module or as a file path.
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.gen.og_qwen3_4b_sample_efficiency_surfaces import (
    SURFACE_BY_ID,
    SURFACE_META,
)

OUT_DIR = REPO / "build" / "reproduction" / "analysis" / "ogqwen3_4b_sample_efficiency"
MANIFEST_PATH = OUT_DIR / "cells_manifest.json"
TAU_CAP = 100
PSI_K = 10

RANDOM_DIR_RE = re.compile(r"^permutation_\d+_random_s(?P<seed>\d+)$")
INJECT_DIR_RE = re.compile(r"^permutation_\d+_inject_(?P<bucket>top|middle|bottom)$")
INJECTION_ORDER = {"top": 0, "middle": 1, "bottom": 2}
LEGACY_OC_SFT_ARM_ALIASES = {
    "k1_supcon": "k1_oc_sft",
    "k10_supcon_hybrid": "k10_oc_sft_hybrid",
}
LEGACY_OC_SFT_TOKEN = "supcon"


def _normalize_arm(arm: str) -> str:
    """Normalize legacy arm tokens while preserving historical IDs and paths."""
    return LEGACY_OC_SFT_ARM_ALIASES.get(arm, arm)


# 30K reuse: the sample-efficiency harvest runs from the grid evaluation.
LEGACY_PREFIXES: dict[str, list[str]] = {
    "k1_sft_30k": ["SE-qwen3-4b-k1-lora-step1400"],
    "k10_sft_30k": ["SE-qwen3-4b-k10-lora-step1200"],
    "k1_supcon_30k": ["SE-qwen3-4b-supcon-l500-warmup-lora-step1000"],
    "k10_supcon_hybrid_30k": ["SE-qwen3-4b-k10-hybrid-l500-warmup-lora-step2000"],
}
LEGACY_PREFIXES["k1_oc_sft_30k"] = LEGACY_PREFIXES["k1_supcon_30k"]
LEGACY_PREFIXES["k10_oc_sft_hybrid_30k"] = LEGACY_PREFIXES["k10_supcon_hybrid_30k"]

SURFACES: tuple[dict[str, Any], ...] = SURFACE_META


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _round_metric(value: float | None) -> float | None:
    v = _finite_float(value)
    return None if v is None else round(v, 4)


def _per_query_tau_values(pq: dict[str, Any], *, cap: int | None) -> list[float]:
    qids = sorted(pq.keys())
    if cap is not None:
        qids = qids[:cap]
    vals: list[float] = []
    for qid in qids:
        v = _finite_float(pq.get(qid, {}).get("tau_based_psi"))
        if v is not None:
            vals.append(v)
    return vals


def _run_glob(prefix: str, surface: str) -> str:
    if prefix.endswith("*"):
        return f"runs/{prefix[:-1]}*{surface}-vllm-psi/*"
    return f"runs/{prefix}-{surface}-vllm-psi/*"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_run(prefixes: list[str], surface: str) -> Path | None:
    for prefix in prefixes:
        matches = [p for p in sorted(REPO.glob(_run_glob(prefix, surface))) if p.is_dir()]
        matches = [p for p in matches if (p / "metrics.json").is_file() and (p / "psi").is_dir()]
        if matches:
            return matches[-1]
    return None


def _prefixes_for_cell(cell: dict) -> list[str]:
    cell_id = cell["cell_id"]
    if cell_id in LEGACY_PREFIXES:
        return LEGACY_PREFIXES[cell_id]
    run_id = cell.get("run_id") or f"se-{cell_id}"
    step = cell.get("selected_step")
    if step is None:
        return []
    return [f"SE-{run_id}-lora-step{int(step)}"]


def resolve_run_for_cell(cell: dict, surface: str) -> Path | None:
    """Return local run dir with metrics + psi for (cell, surface), or None."""
    prefixes = _prefixes_for_cell(cell)
    return _resolve_run(prefixes, surface) if prefixes else None


def member_needs_launch(cell: dict, surface: str) -> bool:
    """True when no harvestable local artifact exists for this cell×surface."""
    return resolve_run_for_cell(cell, surface) is None


def member_exp_id(cell: dict, surface: str) -> str:
    run_id = cell.get("run_id") or f"qwen3-4b-{cell['cell_id'].replace('_', '-')}"
    step = int(cell["selected_step"])
    return f"SE-{run_id}-lora-step{step}-{surface}-vllm-psi"


def _read_trec_ranking(path: Path) -> list[str]:
    ranking: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 3:
                ranking.append(str(parts[2]))
    return ranking


def _count_inversions(values: list[int]) -> int:
    n = len(values)
    if n < 2:
        return 0
    mid = n // 2
    left, right = values[:mid], values[mid:]
    inv = _count_inversions(left) + _count_inversions(right)
    i = j = k = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            values[k] = left[i]
            i += 1
        else:
            values[k] = right[j]
            inv += len(left) - i
            j += 1
        k += 1
    while i < len(left):
        values[k] = left[i]
        i += 1
        k += 1
    while j < len(right):
        values[k] = right[j]
        j += 1
        k += 1
    return inv


def _kendall_tau_fast(a: list[str], b: list[str]) -> float | None:
    b_set = set(b)
    common = [doc for doc in a if doc in b_set]
    if len(common) < 2:
        return None
    common_set = set(common)
    b_pos = {doc: idx for idx, doc in enumerate(b) if doc in common_set}
    b_positions_in_a_order = [b_pos[doc] for doc in common]
    discordant = _count_inversions(b_positions_in_a_order)
    denom = len(common) * (len(common) - 1) // 2
    return float(1.0 - (2.0 * discordant / denom))


def _tau_psi_from_rankings(rankings: list[list[str]]) -> float | None:
    if len(rankings) < 2:
        return None
    taus: list[float] = []
    for i in range(len(rankings)):
        for j in range(i + 1, len(rankings)):
            tau = _kendall_tau_fast(rankings[i], rankings[j])
            if tau is not None:
                taus.append(tau)
    if not taus:
        return None
    mean_tau = sum(taus) / len(taus)
    return max(0.0, min(1.0, (1.0 - mean_tau) / 2.0))


def _rankings_by_query(run_dir: Path) -> dict[str, dict[str, Any]]:
    per_query_dir = run_dir / "psi" / "per_query_results"
    if not per_query_dir.is_dir():
        return {}
    out: dict[str, dict[str, Any]] = {}
    for qdir in sorted(p for p in per_query_dir.iterdir() if p.is_dir()):
        random_rankings: dict[int, list[str]] = {}
        injection_rankings: list[tuple[int, list[str]]] = []
        for pdir in sorted(p for p in qdir.iterdir() if p.is_dir()):
            trec_path = pdir / "trec_results_raw.txt"
            if not trec_path.is_file():
                continue
            m = RANDOM_DIR_RE.match(pdir.name)
            if m:
                random_rankings[int(m.group("seed"))] = _read_trec_ranking(trec_path)
                continue
            m = INJECT_DIR_RE.match(pdir.name)
            if m:
                injection_rankings.append((INJECTION_ORDER[m.group("bucket")], _read_trec_ranking(trec_path)))
        if random_rankings:
            out[qdir.name] = {
                "random": random_rankings,
                "injections": [r for _, r in sorted(injection_rankings)],
            }
    return out


def _tau_psi(run_dir: Path, *, cap: int | None) -> tuple[float | None, int, bool]:
    rankings_by_qid = _rankings_by_query(run_dir)
    qids = sorted(rankings_by_qid)
    capped = cap is not None and len(qids) > cap
    if capped:
        qids = qids[:cap]
    per_query: list[float] = []
    for qid in qids:
        payload = rankings_by_qid[qid]
        random_rankings: dict[int, list[str]] = payload["random"]
        selected_seeds = sorted(random_rankings)[:PSI_K]
        if len(selected_seeds) < PSI_K:
            continue
        rankings = [random_rankings[s] for s in selected_seeds]
        rankings.extend(payload.get("injections") or [])
        tau_psi = _tau_psi_from_rankings(rankings)
        if tau_psi is not None:
            per_query.append(tau_psi)
    if not per_query:
        return _tau_psi_from_aggregates(run_dir, cap=cap)
    mean = sum(per_query) / len(per_query)
    return (_finite_float(mean), len(per_query), capped)


def _tau_psi_from_aggregates(run_dir: Path, *, cap: int | None) -> tuple[float | None, int, bool]:
    """Fallback when fetch pruned ``psi/per_query_results`` but left aggregate JSON."""
    psi_dir = run_dir / "psi"
    per_query_path = psi_dir / "psi_per_query.json"
    if per_query_path.is_file():
        pq = _read_json(per_query_path)
        vals = _per_query_tau_values(pq, cap=cap)
        if vals:
            capped = cap is not None and len(pq) > cap
            return sum(vals) / len(vals), len(vals), capped
    metrics_path = psi_dir / "psi_metrics.json"
    if metrics_path.is_file():
        agg = _read_json(metrics_path).get("aggregate") or {}
        v = _finite_float(agg.get("mean_tau_based_psi"))
        n = agg.get("n_tau_based_psi") or agg.get("n_queries")
        if v is not None:
            return v, int(n or 0), False
    return None, 0, False


def _ndcg10(run_dir: Path) -> float | None:
    metrics = _read_json(run_dir / "metrics.json")
    return _finite_float(metrics.get("mean_ndcg_cut_10"))


def _classify_tau_shape(records: list[dict], *, surface: str, arm: str) -> str:
    """Classify the tau-PSI-vs-N curve shape for one arm."""
    by_n = sorted(
        ((r["N"], r.get("tau_psi")) for r in records if r["surface"] == surface and r["arm"] == arm),
        key=lambda x: x[0],
    )
    vals = [f for _, v in by_n if (f := _finite_float(v)) is not None]
    if len(vals) < 2:
        return "insufficient_data"
    if max(vals) - min(vals) < 0.02:
        return "flat"
    # Early plateau: 80% of peak by 1K (or smallest N above 100).
    peak = max(vals)
    early_n = vals[0]
    for n, v in by_n:
        if n >= 1000 and v is not None:
            early_n = v
            break
    last = _finite_float(by_n[-1][1])
    if early_n >= 0.8 * peak and last is not None and last - early_n < 0.03:
        return "early_plateau"
    if all(vals[i] <= vals[i + 1] + 1e-6 for i in range(len(vals) - 1)):
        return "monotone_gain"
    return "mixed"


def harvest(manifest: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for cell in manifest:
        prefixes = _prefixes_for_cell(cell)
        for surf in SURFACES:
            run_dir = resolve_run_for_cell(cell, surf["id"]) if prefixes else None
            cap = TAU_CAP if surf["tau_cap"] else None
            tau_psi = ndcg = None
            n_queries = 0
            psi_capped = False
            run_path = None
            if run_dir is not None:
                ndcg = _ndcg10(run_dir)
                tau_psi, n_queries, psi_capped = _tau_psi(run_dir, cap=cap)
                run_path = str(run_dir.relative_to(REPO))
            rows.append(
                {
                    "arm": _normalize_arm(str(cell["arm"])),
                    "k_silver": cell["k_silver"],
                    "loss": "oc_sft" if cell["loss"] == LEGACY_OC_SFT_TOKEN else cell["loss"],
                    "N": cell["n_qids"],
                    "n_label": cell["n_label"],
                    "surface": surf["id"],
                    "surface_group": surf["group"],
                    "ndcg10": _round_metric(ndcg),
                    "tau_psi": _round_metric(tau_psi),
                    "selected_step": cell.get("selected_step"),
                    "eff_epochs": cell.get("eff_epochs"),
                    "n_queries": n_queries or surf["n_queries"],
                    "psi_capped": psi_capped,
                    "lambda": cell.get("lambda"),
                    "source": cell.get("source"),
                    "run_dir": run_path,
                    "status": "complete" if run_dir is not None else "pending",
                }
            )
    return rows


def provenance_verdict(rows: list[dict], manifest: list[dict]) -> dict[str, Any]:
    """Step-matched shape check: {K10 SFT, K1 OC-SFT} x {1K, 30K} x DL19."""
    eff_epochs = {c["cell_id"]: c.get("eff_epochs") for c in manifest if c.get("eff_epochs") is not None}
    dl19 = [r for r in rows if r["surface"] == "dl19" and r["tau_psi"] is not None]
    pairs = []
    for arm in ("k10_sft", "k1_oc_sft"):
        for n_label in ("1k", "30k"):
            match = next((r for r in dl19 if r["arm"] == arm and r["n_label"] == n_label), None)
            if match:
                pairs.append(
                    {"arm": arm, "n_label": n_label, "tau_psi": match["tau_psi"], "step": match["selected_step"]}
                )
    verdict = "insufficient_data"
    if len(pairs) == 4:
        k10_gap = abs(
            next(p["tau_psi"] for p in pairs if p["arm"] == "k10_sft" and p["n_label"] == "30k")
            - next(p["tau_psi"] for p in pairs if p["arm"] == "k10_sft" and p["n_label"] == "1k")
        )
        k1_gap = abs(
            next(p["tau_psi"] for p in pairs if p["arm"] == "k1_oc_sft" and p["n_label"] == "30k")
            - next(p["tau_psi"] for p in pairs if p["arm"] == "k1_oc_sft" and p["n_label"] == "1k")
        )
        if k10_gap < 0.03 and k1_gap >= 0.05:
            verdict = "K10-SFT-saturates-early-K1-OC-SFT-gains-with-N"
        elif k1_gap < 0.03 and k10_gap >= 0.05:
            verdict = "K1-OC-SFT-saturates-early-K10-SFT-gains-with-N"
        else:
            verdict = "both-arms-show-N-sensitivity-on-DL19"
    return {
        "eff_epochs_by_cell": eff_epochs,
        "step_matched_dl19": pairs,
        "one_sentence_verdict": verdict,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--write-provenance", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.manifest.is_file():
        print(f"[sample-efficiency-analysis][FATAL] missing manifest: {args.manifest}", file=sys.stderr)
        return 2

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = harvest(manifest)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    json_path = args.out_dir / "sample_efficiency_results.json"
    json_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")

    csv_path = args.out_dir / "sample_efficiency_results.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    shapes = {
        arm: {sid: _classify_tau_shape(rows, surface=sid, arm=arm) for sid in SURFACE_BY_ID}
        for arm in ("k1_sft", "k10_sft", "k1_oc_sft", "k10_oc_sft_hybrid")
    }
    (args.out_dir / "tau_shape_classification.json").write_text(json.dumps(shapes, indent=2) + "\n", encoding="utf-8")

    n_complete = sum(1 for r in rows if r["status"] == "complete")
    missing = [
        r
        for r in rows
        if r["status"] == "complete"
        and (_round_metric(r.get("tau_psi")) is None or _round_metric(r.get("ndcg10")) is None)
    ]
    if missing:
        print(
            f"[sample-efficiency-analysis][WARN] {len(missing)} complete rows missing finite metrics",
            file=sys.stderr,
        )
        for r in missing[:5]:
            print(f"  {r['surface']} N={r['N']} {r['arm']}", file=sys.stderr)
    print(f"[sample-efficiency-analysis] {n_complete}/{len(rows)} cell×surface records complete")
    print(f"[sample-efficiency-analysis] wrote {json_path}")

    if args.write_provenance:
        prov = provenance_verdict(rows, manifest)
        prov_path = args.out_dir / "provenance.json"
        prov_path.write_text(json.dumps(prov, indent=2) + "\n", encoding="utf-8")
        print(f"[sample-efficiency-analysis] verdict: {prov['one_sentence_verdict']}")
        print(f"[sample-efficiency-analysis] wrote {prov_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
