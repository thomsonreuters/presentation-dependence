#!/usr/bin/env python
"""Aggregate Category B multi-seed quality and stability with paired uncertainty.

Input is the Category B run manifest after its ``evaluations`` mapping has been
filled. Every evaluation record must contain ``arm``, ``seed``, ``dataset``, and
``run_dir``. Run directories must retain:

* ``metrics.json``;
* ``all_queries_eval_results.jsonl``;
* ``psi/psi_metrics.json``;
* ``psi/psi_per_query.json``.

The reducer reports all seeds, sample SD across seed-level dataset-balanced
means, within-seed paired query/dataset intervals, a nested
seed/dataset/query bootstrap, equivalence against +/-0.005 nDCG, and the
predeclared seed-45/46 escalation gate.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
BUILD_DIR = ROOT / "build/reproduction/analysis/category_b_multiseed"
# This manifest is an execution input, not tracked evidence. Materialize it from
# retained runs (or pass --manifest) before invoking this legacy reducer.
MANIFEST_PATH = BUILD_DIR / "run_manifest.json"
DEFAULT_OUT = BUILD_DIR / "analysis.json"
PRIMARY_SEEDS = (42, 43, 44)
EQUIVALENCE_MARGIN = 0.005
PRACTICAL_THRESHOLD = 0.01
BOOTSTRAP_SAMPLES = 10_000

CONTRASTS = {
    "generic_ocsft_minus_k10sft": ("R-F1", "R-B1"),
    "generic_ocsft_minus_posaug": ("R-F1", "R-B2"),
    "generic_ocsft_minus_debiasfirst": ("R-F1", "R-B3"),
    "generic_ocsft_minus_k1sft": ("R-F1", "R-B4"),
    "specialist_ocsft_minus_k10sft": ("R-S1", "R-S2"),
}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_quality_per_query(path: Path) -> dict[str, float]:
    """Load canonical per-query nDCG@10 from one eval run."""
    values: dict[str, float] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            qid = str(row["qid"])
            if qid in values:
                raise ValueError(f"{path}: duplicate qid {qid}")
            values[qid] = float(row["ndcg_cut_10"])
    if not values:
        raise ValueError(f"{path}: no query metrics")
    return values


def load_tau_per_query(path: Path) -> dict[str, float]:
    """Load per-query tau-PSI from one PSI run."""
    rows = _read_json(path)
    values = {
        str(qid): float(row["tau_based_psi"]) for qid, row in rows.items() if row.get("tau_based_psi") is not None
    }
    if not values:
        raise ValueError(f"{path}: no tau_based_psi values")
    return values


def paired_difference(left: Mapping[str, float], right: Mapping[str, float]) -> np.ndarray:
    """Return aligned ``left-right`` values over identical qids."""
    if set(left) != set(right):
        missing_left = sorted(set(right) - set(left))[:5]
        missing_right = sorted(set(left) - set(right))[:5]
        raise ValueError(f"paired qid mismatch: missing_left={missing_left}, missing_right={missing_right}")
    qids = sorted(left)
    left_values = np.asarray([left[qid] for qid in qids], dtype=float)
    right_values = np.asarray([right[qid] for qid in qids], dtype=float)
    finite = np.isfinite(left_values) & np.isfinite(right_values)
    if not np.any(finite):
        raise ValueError("paired vectors contain no jointly finite values")
    return left_values[finite] - right_values[finite]


def _ci(samples: np.ndarray, level: float) -> list[float]:
    alpha = (1.0 - level) / 2.0
    low, high = np.quantile(samples, [alpha, 1.0 - alpha])
    return [float(low), float(high)]


def hierarchical_bootstrap(
    per_seed: Mapping[int, Mapping[str, np.ndarray]],
    *,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = 0,
) -> np.ndarray:
    """Resample seed, then dataset, then paired query differences."""
    seed_ids = sorted(per_seed)
    if not seed_ids:
        raise ValueError("no seeds to bootstrap")
    datasets = sorted(set.intersection(*(set(per_seed[s]) for s in seed_ids)))
    if not datasets:
        raise ValueError("no common datasets to bootstrap")
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=float)
    for index in range(samples):
        selected_seeds = rng.choice(seed_ids, size=len(seed_ids), replace=True)
        seed_means: list[float] = []
        for selected_seed in selected_seeds:
            selected_datasets = rng.choice(datasets, size=len(datasets), replace=True)
            dataset_means: list[float] = []
            for dataset in selected_datasets:
                values = per_seed[int(selected_seed)][str(dataset)]
                draw = rng.choice(values, size=values.size, replace=True)
                dataset_means.append(float(draw.mean()))
            seed_means.append(float(np.mean(dataset_means)))
        estimates[index] = float(np.mean(seed_means))
    return estimates


def observed_seed_means(per_seed: Mapping[int, Mapping[str, np.ndarray]]) -> dict[int, float]:
    """Compute equal-dataset means for every seed."""
    return {
        seed: float(np.mean([values.mean() for values in datasets.values()])) for seed, datasets in per_seed.items()
    }


def normal_tost(estimate: float, bootstrap: np.ndarray, margin: float = EQUIVALENCE_MARGIN) -> dict[str, Any]:
    """Normal-approximation TOST using the nested-bootstrap standard error."""
    se = float(bootstrap.std(ddof=1))
    if se == 0:
        equivalent = abs(estimate) < margin
        p_tost = 0.0 if equivalent else 1.0
    else:

        def normal_cdf(z):
            return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

        p_lower = 1.0 - normal_cdf((estimate + margin) / se)
        p_upper = normal_cdf((estimate - margin) / se)
        p_tost = max(p_lower, p_upper)
        equivalent = p_tost < 0.05
    return {
        "margin": margin,
        "estimate": estimate,
        "bootstrap_se": se,
        "ci90": _ci(bootstrap, 0.90),
        "p_tost": float(p_tost),
        "equivalent": bool(equivalent),
    }


def summarize_contrast(
    per_seed: Mapping[int, Mapping[str, np.ndarray]],
    *,
    samples: int = BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    """Summarize one paired contrast and its seed-extension gate."""
    seed_means = observed_seed_means(per_seed)
    nested = hierarchical_bootstrap(per_seed, samples=samples, seed=bootstrap_seed)
    estimate = float(np.mean(list(seed_means.values())))
    values = np.asarray(list(seed_means.values()), dtype=float)
    within_seed = {
        str(seed): {
            "estimate": value,
            "ci95": _ci(
                hierarchical_bootstrap(
                    {seed: per_seed[seed]},
                    samples=samples,
                    seed=bootstrap_seed + seed,
                ),
                0.95,
            ),
        }
        for seed, value in seed_means.items()
    }
    ci95 = _ci(nested, 0.95)
    reasons: list[str] = []
    if abs(estimate) <= PRACTICAL_THRESHOLD:
        reasons.append("absolute_quality_contrast_within_0.01")
    if bool(np.any(values > 0) and np.any(values < 0)):
        reasons.append("seed_sign_change")
    if ci95[0] <= 0.0 <= ci95[1]:
        reasons.append("nested_interval_overlaps_zero")
    return {
        "estimate": estimate,
        "seed_values": {str(seed): value for seed, value in seed_means.items()},
        "seed_sd": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "ci95": ci95,
        "within_seed": within_seed,
        "equivalence": normal_tost(estimate, nested),
        "extend_to_seeds_45_46": bool(reasons),
        "extension_reasons": reasons,
    }


def load_evaluations(
    manifest: dict[str, Any],
) -> tuple[
    dict[tuple[str, int, str], dict[str, float]],
    dict[tuple[str, int, str], dict[str, float]],
    dict[str, Any],
]:
    """Load per-query vectors and aggregate metrics from manifest run dirs."""
    quality: dict[tuple[str, int, str], dict[str, float]] = {}
    tau: dict[tuple[str, int, str], dict[str, float]] = {}
    aggregates: dict[str, Any] = {}
    for key, record in manifest.get("evaluations", {}).items():
        run_dir = ROOT / record["run_dir"]
        identity = (
            str(record["arm"]),
            int(record["seed"]),
            str(record["dataset"]),
        )
        quality[identity] = load_quality_per_query(run_dir / "all_queries_eval_results.jsonl")
        tau[identity] = load_tau_per_query(run_dir / "psi" / "psi_per_query.json")
        metrics = _read_json(run_dir / "metrics.json")
        psi_metrics = _read_json(run_dir / "psi" / "psi_metrics.json")
        aggregates[key] = {
            "n_queries": int(metrics["n_queries"]),
            "ndcg_cut_10": float(metrics["mean_ndcg_cut_10"]),
            "n_tau_queries": int(psi_metrics["aggregate"]["n_tau_based_psi"]),
            "tau_psi": float(psi_metrics["aggregate"]["mean_tau_based_psi"]),
        }
    return quality, tau, aggregates


def build_contrast_vectors(
    vectors: Mapping[tuple[str, int, str], Mapping[str, float]],
    left: str,
    right: str,
    seeds: tuple[int, ...],
) -> dict[int, dict[str, np.ndarray]]:
    """Build seed/dataset paired vectors for one arm contrast."""
    dataset_sets = [
        {dataset for arm, vector_seed, dataset in vectors if arm == target_arm and vector_seed == seed}
        for seed in seeds
        for target_arm in (left, right)
    ]
    datasets = sorted(set.intersection(*dataset_sets))
    if not datasets:
        raise ValueError(f"{left} vs {right}: no common datasets over seeds {seeds}")
    out: dict[int, dict[str, np.ndarray]] = {}
    for seed in seeds:
        out[seed] = {}
        for dataset in datasets:
            left_values = vectors[(left, seed, dataset)]
            right_values = vectors[(right, seed, dataset)]
            out[seed][dataset] = paired_difference(left_values, right_values)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in PRIMARY_SEEDS),
        help="Comma-separated student seeds to include.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = _read_json(args.manifest)
    quality, tau, aggregates = load_evaluations(manifest)
    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    result: dict[str, Any] = {
        "protocol": {
            "seeds": list(seeds),
            "dataset_weighting": "equal",
            "bootstrap": "seed, then dataset, then paired query",
            "samples": args.samples,
            "equivalence_margin": EQUIVALENCE_MARGIN,
            "practical_threshold": PRACTICAL_THRESHOLD,
        },
        "aggregates": aggregates,
        "quality_contrasts": {},
        "tau_contrasts": {},
    }
    for index, (name, (left, right)) in enumerate(CONTRASTS.items()):
        quality_vectors = build_contrast_vectors(quality, left, right, seeds)
        tau_vectors = build_contrast_vectors(tau, left, right, seeds)
        result["quality_contrasts"][name] = summarize_contrast(
            quality_vectors,
            samples=args.samples,
            bootstrap_seed=100 + index,
        )
        result["tau_contrasts"][name] = summarize_contrast(
            tau_vectors,
            samples=args.samples,
            bootstrap_seed=200 + index,
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"[category-b-analysis] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
