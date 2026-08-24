#!/usr/bin/env python
"""Reduce the Qwen3-4B fixed-order pool-perturbation battery.

Discovers the latest fetched run per (dataset, arm) under ``runs/``, verifies
paired-query identity across arms, and writes a machine-readable summary plus
a compact Markdown sidecar.

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from presentation_dependence.analysis.terminology import paper_method_label


ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = ROOT / "runs"
DEFAULT_OUT = ROOT / "build" / "reproduction" / "analysis" / "pool_perturbation"
DEFAULT_REPORT = ROOT / "build" / "reproduction" / "reporting" / "pool_perturbation"

# Ordered so the original three-collection cells stay first in every table.
DATASETS: tuple[str, ...] = (
    "arguana",
    "fiqa",
    "nfcorpus",
    "dl19",
    "dl20",
    "climate-fever",
    "trec-covid",
    "robust04",
    "dbpedia-entity",
    "scifact",
    "signal1m",
    "trec-news",
    "touche2020",
    "legal-a",
    "legal-b",
)
ARMS = ("base", "k1sft", "ocsft")
PERTURBATIONS = ("replace", "drop")
METRICS = {
    "pool_psi": "pool_psi",
    "top10_flip": "top_k_set_changed",
    "delta_ndcg": "delta_ndcg_at_k_retained",
    "abs_delta_ndcg": "delta_ndcg_at_k_retained",
}

# Expected paired-query count per dataset: the number of qids that are both
# judged (qrels) and reach source_depth=101 in the first-stage run, capped at
# pool_perturbation.max_queries in the matching config. Six of the twelve new
# collections have fewer than 100 total eligible queries; see the generator
# (`scripts/gen/gen_pool_perturbation_extension.py`) and the handoff addendum for
# the derivation of each.
EXPECTED_N: dict[str, int] = {
    "arguana": 100,
    "fiqa": 100,
    "nfcorpus": 100,
    "dl19": 43,
    "dl20": 54,
    "climate-fever": 100,
    "trec-covid": 37,
    "robust04": 100,
    "dbpedia-entity": 100,
    "scifact": 100,
    "signal1m": 97,
    "trec-news": 57,
    "touche2020": 49,
    "legal-a": 96,
    "legal-b": 100,
}

LABELS = {arm: paper_method_label(arm) for arm in ARMS}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _exp_id(dataset: str, arm: str) -> str:
    return f"PP-qwen3-4b-{arm}-{dataset}"


def _latest_run_dir(exp_id: str) -> Path:
    """Return the most recently timestamped run dir with pool-perturbation output."""
    exp_root = RUNS_ROOT / exp_id
    if not exp_root.is_dir():
        raise FileNotFoundError(f"no runs found for {exp_id} under {exp_root}")
    candidates = sorted(
        p for p in exp_root.iterdir() if p.is_dir() and (p / "pool_perturbation" / "pool_metrics.json").is_file()
    )
    if not candidates:
        raise FileNotFoundError(f"no completed pool_perturbation run under {exp_root}")
    return candidates[-1]


def _load(datasets: tuple[str, ...]) -> tuple[dict, dict, dict]:
    aggregate: dict[str, dict] = {}
    per_query: dict[str, dict] = {}
    runs: dict[str, dict] = {}
    for dataset in datasets:
        aggregate[dataset] = {}
        per_query[dataset] = {}
        runs[dataset] = {}
        for arm in ARMS:
            exp_id = _exp_id(dataset, arm)
            run_dir = _latest_run_dir(exp_id)
            runs[dataset][arm] = str(run_dir.relative_to(ROOT))
            aggregate[dataset][arm] = _read_json(run_dir / "pool_perturbation" / "pool_metrics.json")
            per_query[dataset][arm] = _read_json(run_dir / "pool_perturbation" / "pool_per_query.json")
    return aggregate, per_query, runs


def _validate(datasets: tuple[str, ...], aggregate: dict, per_query: dict) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for dataset in datasets:
        qid_lists = [list(per_query[dataset][arm]) for arm in ARMS]
        if not all(qids == qid_lists[0] for qids in qid_lists[1:]):
            raise ValueError(f"{dataset}: qid order differs across arms")
        n = len(qid_lists[0])
        expected = EXPECTED_N.get(dataset)
        if expected is not None and n != expected:
            raise ValueError(f"{dataset}: expected {expected} paired queries, got {n}")
        for qid in qid_lists[0]:
            anchors = [
                (
                    per_query[dataset][arm][qid]["drop_rank_1based"],
                    per_query[dataset][arm][qid]["dropped_pid"],
                    per_query[dataset][arm][qid]["replacement_pid"],
                )
                for arm in ARMS
            ]
            if not all(anchor == anchors[0] for anchor in anchors[1:]):
                raise ValueError(f"{dataset}/{qid}: perturbation differs across arms")
        for arm in ARMS:
            for perturbation in PERTURBATIONS:
                arm_n = aggregate[dataset][arm]["aggregate"][perturbation]["n_queries"]
                if arm_n != n:
                    raise ValueError(f"{dataset}/{arm}/{perturbation}: expected n={n}, got {arm_n}")
        checks[dataset] = {
            "n_queries": n,
            "same_qids_across_arms": True,
            "same_drop_and_replacement_across_arms": True,
        }
    return checks


def _cell_rows(datasets: tuple[str, ...], aggregate: dict) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        for arm in ARMS:
            for perturbation in PERTURBATIONS:
                block = aggregate[dataset][arm]["aggregate"][perturbation]
                rows.append(
                    {
                        "dataset": dataset,
                        "arm": arm,
                        "perturbation": perturbation,
                        "n_queries": block["n_queries"],
                        "pool_psi": block["mean_pool_psi"],
                        "top10_flip_rate": block["top_k_set_flip_rate"],
                        "delta_ndcg": block["mean_delta_ndcg_at_k_retained"],
                        "abs_delta_ndcg": block["mean_abs_delta_ndcg_at_k_retained"],
                    }
                )
    return rows


def _macro(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for perturbation in PERTURBATIONS:
        out[perturbation] = {}
        for arm in ARMS:
            cells = [row for row in rows if row["arm"] == arm and row["perturbation"] == perturbation]
            out[perturbation][arm] = {
                key: float(np.mean([float(row[key]) for row in cells]))
                for key in (
                    "pool_psi",
                    "top10_flip_rate",
                    "delta_ndcg",
                    "abs_delta_ndcg",
                )
            }
    return out


def _query_value(row: dict[str, Any], perturbation: str, metric: str) -> float:
    value = row[perturbation][METRICS[metric]]
    value_f = float(value)
    return abs(value_f) if metric == "abs_delta_ndcg" else value_f


def _hierarchical_paired_bootstrap(
    datasets: tuple[str, ...],
    per_query: dict,
    *,
    perturbation: str,
    metric: str,
    left: str,
    right: str,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    observed_by_dataset: list[float] = []
    paired: dict[str, np.ndarray] = {}
    for dataset in datasets:
        qids = list(per_query[dataset][left])
        diffs = np.asarray(
            [
                _query_value(per_query[dataset][left][qid], perturbation, metric)
                - _query_value(per_query[dataset][right][qid], perturbation, metric)
                for qid in qids
            ],
            dtype=float,
        )
        paired[dataset] = diffs
        observed_by_dataset.append(float(np.mean(diffs)))

    boots = np.empty(samples, dtype=float)
    for i in range(samples):
        sampled_datasets = rng.choice(datasets, size=len(datasets), replace=True)
        dataset_means = []
        for dataset in sampled_datasets:
            diffs = paired[str(dataset)]
            idx = rng.integers(0, len(diffs), size=len(diffs))
            dataset_means.append(float(np.mean(diffs[idx])))
        boots[i] = float(np.mean(dataset_means))
    return {
        "contrast": f"{left}_minus_{right}",
        "metric": metric,
        "perturbation": perturbation,
        "estimate": float(np.mean(observed_by_dataset)),
        "ci95": [float(x) for x in np.quantile(boots, [0.025, 0.975])],
        "bootstrap": {
            "method": "paired hierarchical percentile bootstrap; datasets then queries",
            "samples": samples,
            "seed": seed,
            "n_datasets": len(datasets),
        },
    }


def _comparisons(datasets: tuple[str, ...], per_query: dict, samples: int, seed: int) -> list[dict[str, Any]]:
    rows = []
    contrasts = (("base", "ocsft"), ("k1sft", "ocsft"), ("base", "k1sft"))
    for perturbation in PERTURBATIONS:
        for metric in ("pool_psi", "top10_flip", "delta_ndcg", "abs_delta_ndcg"):
            for left, right in contrasts:
                rows.append(
                    _hierarchical_paired_bootstrap(
                        datasets,
                        per_query,
                        perturbation=perturbation,
                        metric=metric,
                        left=left,
                        right=right,
                        samples=samples,
                        seed=seed,
                    )
                )
    return rows


def _find_comparison(
    comparisons: list[dict[str, Any]],
    *,
    perturbation: str,
    metric: str,
    contrast: str,
) -> dict[str, Any]:
    return next(
        row
        for row in comparisons
        if row["perturbation"] == perturbation and row["metric"] == metric and row["contrast"] == contrast
    )


def _markdown(datasets: tuple[str, ...], payload: dict[str, Any]) -> str:
    lines = [
        "# Fixed-order pool perturbation: Qwen3-4B results",
        "",
        "Generated by `scripts/analyze/analyze_pool_perturbation.py` from the fetched runs "
        f"for {len(datasets)} collections ({', '.join(datasets)}).",
        "All metrics are restricted to the retained documents (pool_size - 1); "
        "values are dataset-macro means unless noted. Per-collection n varies -- "
        "see the per-dataset table below and `validation` in the JSON sidecar.",
        "",
        "## Headline (replace one top-100 document with BM25 rank 101)",
        "",
        "| Arm | pool-PSI | retained top-10 flip | signed delta nDCG@10 | absolute delta nDCG@10 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    macro = payload["macro"]["replace"]
    for arm in ARMS:
        row = macro[arm]
        lines.append(
            f"| {LABELS[arm]} | {row['pool_psi']:.4f} | "
            f"{100 * row['top10_flip_rate']:.1f}% | {row['delta_ndcg']:+.4f} | "
            f"{row['abs_delta_ndcg']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Per-dataset replace results",
            "",
            "| Dataset | n | Arm | pool-PSI | top-10 flip | signed delta nDCG@10 | absolute delta nDCG@10 |",
            "| --- | ---: | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for dataset in datasets:
        n = payload["validation"][dataset]["n_queries"]
        for arm in ARMS:
            row = next(
                item
                for item in payload["cells"]
                if item["dataset"] == dataset and item["arm"] == arm and item["perturbation"] == "replace"
            )
            lines.append(
                f"| {dataset} | {n} | {LABELS[arm]} | {row['pool_psi']:.4f} | "
                f"{100 * row['top10_flip_rate']:.0f}% | {row['delta_ndcg']:+.4f} | "
                f"{row['abs_delta_ndcg']:.4f} |"
            )

    lines.extend(["", "## Paired macro contrasts (replace)", ""])
    for metric, label in (
        ("pool_psi", "Off the shelf minus OC-SFT pool-PSI"),
        ("top10_flip", "Off the shelf minus OC-SFT top-10 flip probability"),
        ("pool_psi", "Single-order minus OC-SFT pool-PSI"),
    ):
        contrast = "base_minus_ocsft" if label.startswith("Off the shelf") else "k1sft_minus_ocsft"
        comp = _find_comparison(
            payload["comparisons"],
            perturbation="replace",
            metric=metric,
            contrast=contrast,
        )
        scale = 100.0 if metric == "top10_flip" else 1.0
        digits = 1 if metric == "top10_flip" else 4
        unit = " percentage points" if metric == "top10_flip" else ""
        lines.append(
            f"- {label}: {scale * comp['estimate']:.{digits}f}{unit}, "
            f"95% CI [{scale * comp['ci95'][0]:.{digits}f}, "
            f"{scale * comp['ci95'][1]:.{digits}f}]{unit} "
            f"(n_datasets={comp['bootstrap']['n_datasets']})."
        )

    base = macro["base"]["pool_psi"]
    oc = macro["ocsft"]["pool_psi"]
    lines.extend(
        [
            "",
            "## Drop-only control (replace minus drop, macro pool-PSI)",
            "",
        ]
    )
    for arm in ARMS:
        gap = payload["replace_minus_drop_macro_pool_psi"][arm]
        lines.append(f"- {LABELS[arm]}: {gap:+.4f}")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- Fixed order does not make the off-the-shelf scorer stable: pool-PSI is {base:.4f} "
            f"and the retained top-10 changes on {100 * macro['base']['top10_flip_rate']:.1f}% of queries.",
            f"- OC-SFT lowers macro pool-PSI by {100 * (1 - oc / base):.1f}% "
            f"({base:.4f} to {oc:.4f}); single-order distillation barely moves it.",
            "- Replace and drop results track closely (see the drop-only control above), "
            "so the result is not driven by the ragged final chunk.",
            "- Signed retained-set nDCG changes stay near zero while ranking and top-10 membership move, "
            "supporting an instability rather than quality-loss interpretation.",
            "",
        ]
    )
    return "\n".join(lines)


def build(datasets: tuple[str, ...] = DATASETS, samples: int = 20_000, seed: int = 0) -> dict[str, Any]:
    aggregate, per_query, runs = _load(datasets)
    checks = _validate(datasets, aggregate, per_query)
    cells = _cell_rows(datasets, aggregate)
    macro = _macro(cells)
    comparisons = _comparisons(datasets, per_query, samples, seed)
    replace_drop_gap = {arm: macro["replace"][arm]["pool_psi"] - macro["drop"][arm]["pool_psi"] for arm in ARMS}
    return {
        "schema_version": 2,
        "datasets": list(datasets),
        "runs": runs,
        "validation": checks,
        "cells": cells,
        "macro": macro,
        "comparisons": comparisons,
        "replace_minus_drop_macro_pool_psi": replace_drop_gap,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Override the dataset list (default: all 15).",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    datasets = tuple(args.datasets) if args.datasets else DATASETS
    payload = build(datasets=datasets, samples=args.bootstrap_samples, seed=args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "pool_perturbation_results.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.report_dir / "pool_perturbation_results.md").write_text(
        _markdown(datasets, payload),
        encoding="utf-8",
    )
    print(args.out_dir / "pool_perturbation_results.json")
    print(args.report_dir / "pool_perturbation_results.md")


if __name__ == "__main__":
    main()
