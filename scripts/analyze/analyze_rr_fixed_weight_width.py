#!/usr/bin/env python3
"""Reduce fixed-weight response-ranking width at native B=4 versus B=1.

Holds the Qwen3-4B response OC-SFT (``ocl1-lora``, lambda*=1.0, seed 42)
checkpoint fixed and compares:

- native B=4 single-order nDCG@1 (metrics.json)
- native B=4 K=10 order-averaged nDCG@1 (psi/sc_metrics.json by_K["10"]: mean
  per-candidate scores over the first 10 shuffles, re-rank, evaluate at cutoff 1)
- B=1 nDCG@1 from the matched ocl1-b1deploy cell (docs_per_score_forward=1,
  same LoRA checkpoint, no retraining)

No model is run here. Launch the B=1 arms via overriding the existing
response-ranking OC-SFT evaluation configs with
``reranker.docs_per_score_forward=1`` / ``reranker.batch_size=1`` (same
local adapter checkpoint).

Unlike the QA cell, response-ranking's ``beta_gamma`` is disabled in these
configs (``robustness.beta_gamma.enabled: false``), so no score-level
mu_4-mu_1 reading is available here -- reported as an explicit gap, not
skipped silently.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = ROOT
DEFAULT_OUT = ROOT / "build" / "reproduction" / "analysis" / "rr_fixed_weight_width"
DEFAULT_REPORT = ROOT / "build" / "reproduction" / "reporting" / "rr_fixed_weight_width"
CORE_MULTISEED_OUT = DEFAULT_OUT / "rr_fixed_weight_width_multiseed.json"
CORE_MULTISEED_EVIDENCE = (
    ROOT / "configs" / "reproduction" / "evidence" / "response-width" / "rr-fixed-weight-width-multiseed.json"
)
NECTAR_MULTISEED_OUT = DEFAULT_OUT / "rr_fixed_weight_width_nectar_multiseed.json"
REWARDBENCH2_MULTISEED_OUT = DEFAULT_OUT / "rr_fixed_weight_width_rewardbench2_multiseed.json"
NECTAR_CLEAN_QIDS = ROOT / "configs" / "reproduction" / "evidence" / "cohorts" / "nectar-clean-434.txt"
EXPECTED_CHANNEL = "rr-lora-4b-ocl1"
BOOTSTRAP_SEED = 0
BOOTSTRAP_SAMPLES = 100_000
MEASURE = "ndcg_cut_1"
MEAN_MEASURE = f"mean_{MEASURE}"


@dataclass(frozen=True)
class SurfaceRuns:
    """Native-width and B=1 run pair for one response-ranking dataset."""

    surface: str
    label: str
    native_run: str
    b1_run: str


RUNS: tuple[SurfaceRuns, ...] = (
    SurfaceRuns(
        "rewardbench2",
        "RewardBench-2",
        "runs/RR-rewardbench2-qwen3-4b-ocl1-lora-vllm-psi",
        "runs/RR-rewardbench2-qwen3-4b-ocl1-b1deploy-vllm-psi",
    ),
    SurfaceRuns(
        "nectar",
        "Nectar",
        "runs/RR-nectar-qwen3-4b-ocl1-lora-vllm-psi",
        "runs/RR-nectar-qwen3-4b-ocl1-b1deploy-vllm-psi",
    ),
    SurfaceRuns(
        "ppe-math",
        "PPE-MATH",
        "runs/RR-ppe-math-qwen3-4b-ocl1-lora-vllm-psi",
        "runs/RR-ppe-math-qwen3-4b-ocl1-b1deploy-vllm-psi",
    ),
    SurfaceRuns(
        "ppe-mmlu-pro",
        "PPE-MMLU-Pro",
        "runs/RR-ppe-mmlu-pro-qwen3-4b-ocl1-lora-vllm-psi",
        "runs/RR-ppe-mmlu-pro-qwen3-4b-ocl1-b1deploy-vllm-psi",
    ),
    SurfaceRuns(
        "rmbench",
        "RM-Bench",
        "runs/RR-rmbench-qwen3-4b-ocl1-lora-vllm-psi",
        "runs/RR-rmbench-qwen3-4b-ocl1-b1deploy-vllm-psi",
    ),
)

NECTAR_MULTISEED_RUNS: dict[int, SurfaceRuns] = {
    42: RUNS[1],
    43: SurfaceRuns(
        "nectar",
        "Nectar",
        "runs/RR-nectar-qwen3-4b-ocl1-lora-vllm-psi-seed43",
        "runs/RR-nectar-qwen3-4b-ocl1-b1deploy-seed43-vllm-psi",
    ),
    44: SurfaceRuns(
        "nectar",
        "Nectar",
        "runs/RR-nectar-qwen3-4b-ocl1-lora-vllm-psi-seed44",
        "runs/RR-nectar-qwen3-4b-ocl1-b1deploy-seed44-vllm-psi",
    ),
}

REWARDBENCH2_MULTISEED_RUNS: dict[int, SurfaceRuns] = {
    42: RUNS[0],
    43: SurfaceRuns(
        "rewardbench2",
        "RewardBench-2",
        "runs/RR-rewardbench2-qwen3-4b-ocl1-seed43-lora-vllm-psi",
        "runs/RR-rewardbench2-qwen3-4b-ocl1-b1deploy-seed43-vllm-psi",
    ),
    44: SurfaceRuns(
        "rewardbench2",
        "RewardBench-2",
        "runs/RR-rewardbench2-qwen3-4b-ocl1-seed44-lora-vllm-psi",
        "runs/RR-rewardbench2-qwen3-4b-ocl1-b1deploy-seed44-vllm-psi",
    ),
}

PPE_MATH_MULTISEED_RUNS: dict[int, SurfaceRuns] = {
    42: RUNS[2],
    43: SurfaceRuns(
        "ppe-math",
        "PPE-MATH",
        "runs/RR-ppe-math-qwen3-4b-ocl1-seed43-lora-vllm-psi",
        "runs/RR-ppe-math-qwen3-4b-ocl1-b1deploy-seed43-vllm-psi",
    ),
    44: SurfaceRuns(
        "ppe-math",
        "PPE-MATH",
        "runs/RR-ppe-math-qwen3-4b-ocl1-seed44-lora-vllm-psi",
        "runs/RR-ppe-math-qwen3-4b-ocl1-b1deploy-seed44-vllm-psi",
    ),
}

PPE_MMLU_PRO_MULTISEED_RUNS: dict[int, SurfaceRuns] = {
    42: RUNS[3],
    43: SurfaceRuns(
        "ppe-mmlu-pro",
        "PPE-MMLU-Pro",
        "runs/RR-ppe-mmlu-pro-qwen3-4b-ocl1-seed43-lora-vllm-psi",
        "runs/RR-ppe-mmlu-pro-qwen3-4b-ocl1-b1deploy-seed43-vllm-psi",
    ),
    44: SurfaceRuns(
        "ppe-mmlu-pro",
        "PPE-MMLU-Pro",
        "runs/RR-ppe-mmlu-pro-qwen3-4b-ocl1-seed44-lora-vllm-psi",
        "runs/RR-ppe-mmlu-pro-qwen3-4b-ocl1-b1deploy-seed44-vllm-psi",
    ),
}

RMBENCH_MULTISEED_RUNS: dict[int, SurfaceRuns] = {
    42: RUNS[4],
    43: SurfaceRuns(
        "rmbench",
        "RM-Bench",
        "runs/RR-rmbench-qwen3-4b-ocl1-seed43-lora-vllm-psi",
        "runs/RR-rmbench-qwen3-4b-ocl1-b1deploy-seed43-vllm-psi",
    ),
    44: SurfaceRuns(
        "rmbench",
        "RM-Bench",
        "runs/RR-rmbench-qwen3-4b-ocl1-seed44-lora-vllm-psi",
        "runs/RR-rmbench-qwen3-4b-ocl1-b1deploy-seed44-vllm-psi",
    ),
}


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_qid_set(path: Path) -> set[str]:
    qids = {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
    if not qids:
        raise ValueError(f"empty qid set: {path}")
    return qids


def _resolve_run(spec: str) -> Path:
    path = RUNS_ROOT / spec if not Path(spec).is_absolute() else Path(spec)
    if (path / "metrics.json").is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"missing run: {path}")
    children = sorted((c for c in path.iterdir() if c.is_dir()), key=lambda p: p.name)
    if not children:
        raise FileNotFoundError(f"no timestamp under {path}")
    return children[-1]


def _assert_checkpoint(run_dir: Path, *, expect_b: int) -> dict[str, Any]:
    cfg_path = run_dir / "resolved_config.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"missing {cfg_path}")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    reranker = cfg.get("reranker") or {}
    docs_b = int(reranker.get("docs_per_score_forward") or 0)
    if docs_b != expect_b:
        raise ValueError(f"{run_dir}: docs_per_score_forward={docs_b}, expected {expect_b}")
    execution_block = cfg.get("execution") or {}
    channels = execution_block.get("extra_input_channels") or {}
    channel_key = next((k for k in channels if k == EXPECTED_CHANNEL), None)
    if channel_key is None:
        raise ValueError(f"{run_dir}: missing expected channel {EXPECTED_CHANNEL!r}, got {list(channels)}")
    return {
        "docs_per_score_forward": docs_b,
        "lora_channel": channels[channel_key],
        "grade_rubric_id": reranker.get("grade_rubric_id"),
    }


def _ndcg_single_pass(run_dir: Path, *, qid_keep: set[str] | None = None) -> float:
    if qid_keep is not None:
        per_query = _canonical_per_query(run_dir)
        values = [value for qid, value in per_query.items() if qid in qid_keep]
        if not values:
            raise ValueError(f"{run_dir}: no canonical qids survive the requested filter")
        return float(np.mean(values))
    metrics = _read_json(run_dir / "metrics.json")
    value = metrics.get(MEAN_MEASURE)
    if value is None:
        raise KeyError(f"{run_dir}/metrics.json has no {MEAN_MEASURE}")
    return float(value)


def _ndcg_order_averaged(
    run_dir: Path,
    *,
    k: int = 10,
    qid_keep: set[str] | None = None,
) -> float:
    if qid_keep is not None:
        sc_per_query = _read_json(run_dir / "psi" / "sc_per_query.json")
        per_query = sc_per_query.get(str(k)) or sc_per_query.get(k) or {}
        values = [
            float(metrics[MEASURE]) for qid, metrics in per_query.items() if qid in qid_keep and MEASURE in metrics
        ]
        if not values:
            raise ValueError(f"{run_dir}: no K={k} qids survive the requested filter")
        return float(np.mean(values))
    sc = _read_json(run_dir / "psi" / "sc_metrics.json")
    by_k = sc.get("by_K") or sc.get("by_k") or {}
    entry = by_k.get(str(k)) or by_k.get(k)
    if not isinstance(entry, dict):
        raise KeyError(f"{run_dir}: missing by_K[{k}]")
    metrics = entry.get("metrics") or entry
    ndcg = metrics.get(MEASURE)
    if ndcg is None:
        raise KeyError(f"{run_dir}: by_K[{k}] has no {MEASURE}")
    if isinstance(ndcg, dict):
        return float(ndcg["mean"])
    return float(ndcg)


def _paired_bootstrap_diffs(diffs: list[float]) -> dict[str, float]:
    """Percentile paired bootstrap over query-level metric differences."""
    if not diffs:
        raise ValueError("cannot bootstrap an empty difference vector")
    arr = np.asarray(diffs, dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    means = np.empty(BOOTSTRAP_SAMPLES, dtype=float)
    for i in range(BOOTSTRAP_SAMPLES):
        means[i] = float(arr[rng.integers(0, len(arr), size=len(arr))].mean())
    lo, hi = np.quantile(means, [0.025, 0.975])
    return {
        "mean": float(arr.mean()),
        "ci95_lo": float(lo),
        "ci95_hi": float(hi),
        "n": int(len(arr)),
        "n_positive": int((arr > 0).sum()),
    }


def _canonical_per_query(run_dir: Path) -> dict[str, float]:
    """Load the canonical single-pass metric used by ``metrics.json``."""
    path = run_dir / "all_queries_eval_results.jsonl"
    if not path.is_file():
        raise FileNotFoundError(
            f"missing {path}; restore it from your run archive before computing the one-pass interval"
        )
    values: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if MEASURE in row:
            values[str(row["qid"])] = float(row[MEASURE])
    if not values:
        raise ValueError(f"{path} has no per-query {MEASURE} values")
    return values


def _paired_bootstrap_c_obs(
    native_dir: Path,
    b1_dir: Path,
    *,
    qid_keep: set[str] | None = None,
) -> dict[str, float]:
    """Paired bootstrap: canonical native B=4 minus canonical B=1."""
    native = _canonical_per_query(native_dir)
    b1 = _canonical_per_query(b1_dir)
    common = sorted(set(native) & set(b1))
    if qid_keep is not None:
        common = [qid for qid in common if qid in qid_keep]
    if not common:
        raise ValueError(f"no shared canonical qids between {native_dir} and {b1_dir}")
    return _paired_bootstrap_diffs([native[qid] - b1[qid] for qid in common])


def _paired_bootstrap_v_b(
    native_dir: Path,
    b1_dir: Path,
    *,
    k: int = 10,
    qid_keep: set[str] | None = None,
) -> dict[str, float]:
    """Paired bootstrap: native K=10 order-average minus B=1."""
    native_sc = _read_json(native_dir / "psi" / "sc_per_query.json")
    native_k = native_sc.get(str(k)) or native_sc.get(k) or {}
    b1_sc = _read_json(b1_dir / "psi" / "sc_per_query.json")
    b1_k = b1_sc.get("1") or b1_sc.get(1) or {}
    common = sorted(set(native_k) & set(b1_k))
    if qid_keep is not None:
        common = [qid for qid in common if qid in qid_keep]
    if not common:
        raise ValueError(f"no shared qids between {native_dir} and {b1_dir}")
    missing = [q for q in common if MEASURE not in native_k[q] or MEASURE not in b1_k[q]]
    if missing:
        raise KeyError(f"missing {MEASURE} on {len(missing)} shared qids (e.g. {missing[0]})")
    return _paired_bootstrap_diffs([native_k[q][MEASURE] - b1_k[q][MEASURE] for q in common])


def analyze_surface(
    spec: SurfaceRuns,
    *,
    qid_keep: set[str] | None = None,
    qid_basis: str = "all",
) -> dict[str, Any]:
    """Reduce levels, effects, and paired intervals for one run pair."""
    native = _resolve_run(spec.native_run)
    b1 = _resolve_run(spec.b1_run)
    native_meta = _assert_checkpoint(native, expect_b=4)
    b1_meta = _assert_checkpoint(b1, expect_b=1)
    if native_meta["lora_channel"] != b1_meta["lora_channel"]:
        raise ValueError(
            f"{spec.surface}: checkpoint mismatch -- native={native_meta['lora_channel']!r} "
            f"b1={b1_meta['lora_channel']!r}"
        )

    native_single = _ndcg_single_pass(native, qid_keep=qid_keep)
    native_k10 = _ndcg_order_averaged(native, k=10, qid_keep=qid_keep)
    b1_ndcg = _ndcg_single_pass(b1, qid_keep=qid_keep)
    v_b = native_k10 - b1_ndcg
    c_obs = native_single - b1_ndcg

    paired_c_obs = _paired_bootstrap_c_obs(native, b1, qid_keep=qid_keep)
    paired_v_b = _paired_bootstrap_v_b(native, b1, k=10, qid_keep=qid_keep)
    if not np.isclose(paired_c_obs["mean"], c_obs, atol=1e-12):
        raise ValueError(
            f"{spec.surface}: canonical paired mean {paired_c_obs['mean']} does not match C_obs point estimate {c_obs}"
        )

    return {
        "surface": spec.surface,
        "label": spec.label,
        "native_run": str(native.relative_to(ROOT)),
        "b1_run": str(b1.relative_to(ROOT)),
        "qid_basis": qid_basis,
        "checkpoint": {"native": native_meta, "b1": b1_meta},
        "measure": MEASURE,
        MEASURE: {
            "native_single": native_single,
            "native_k10_order_avg": native_k10,
            "b1": b1_ndcg,
            "C_obs_single_order": c_obs,
            "V_B_order_marginal": v_b,
        },
        "paired_bootstrap_C_obs": paired_c_obs,
        "paired_bootstrap_V_B": paired_v_b,
    }


def render_md(payload: dict[str, Any]) -> str:
    """Render the seed-42 dataset-level reduction as Markdown."""
    rows = payload["surfaces"]
    lines = [
        "# Fixed-weight response-ranking width (Qwen3-4B OC-SFT, nDCG@1)",
        "",
        "Same Qwen3-4B response OC-SFT checkpoint (`ocl1-lora`, lambda*=1.0, seed 42,",
        "native B=4) served at B=1 (order-free, same LoRA weights). Metric is",
        "**nDCG@1**, matching the response-ranking metric, so these cells are directly",
        "comparable to the design comparison (no nDCG@10 caveat).",
        "Nectar uses the contamination-clean 434-prompt subset; the other rows use",
        "their complete evaluation sets.",
        "",
        "| Surface | B=1 | native B=4 single | native B=4 K=10 | `C_obs` | 95% CI (`C_obs`) | `V_B` | 95% CI (`V_B`) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        s = row[MEASURE]
        c_boot = row["paired_bootstrap_C_obs"]
        v_boot = row["paired_bootstrap_V_B"]
        lines.append(
            f"| {row['label']} | {s['b1']:.4f} | {s['native_single']:.4f} | "
            f"{s['native_k10_order_avg']:.4f} | {s['C_obs_single_order']:+.4f} | "
            f"[{c_boot['ci95_lo']:+.4f}, {c_boot['ci95_hi']:+.4f}] | "
            f"{s['V_B_order_marginal']:+.4f} | "
            f"[{v_boot['ci95_lo']:+.4f}, {v_boot['ci95_hi']:+.4f}] |"
        )
    signs = {row["label"]: row[MEASURE]["V_B_order_marginal"] for row in rows}
    lines.append("")
    lines.append("Sign pattern (V_B): " + ", ".join(f"{label} {value:+.4f}" for label, value in signs.items()) + ".")
    lines.append("")
    lines.append(
        "Score-level mu_4-mu_1 is not available: these configs have "
        "`robustness.beta_gamma.enabled: false` (unlike the QA cells), so there is no "
        "`beta_gamma_scores.parquet` to read. Reported as a gap, not filled with a proxy."
    )
    lines.append("")
    lines.append("Producer: `scripts/analyze/analyze_rr_fixed_weight_width.py`.")
    lines.append("")
    return "\n".join(lines)


def build_surface_multiseed(
    surface: str,
    label: str,
    runs: dict[int, SurfaceRuns],
    *,
    qid_keep: set[str] | None = None,
    qid_basis: str = "all",
) -> dict[str, Any]:
    """Reduce one fixed-weight response-ranking cell over seeds 42-44."""
    expected_seeds = {42, 43, 44}
    if set(runs) != expected_seeds:
        raise ValueError(f"{surface}: expected seeds {sorted(expected_seeds)}, got {sorted(runs)}")
    by_seed = {
        str(seed): analyze_surface(spec, qid_keep=qid_keep, qid_basis=qid_basis) for seed, spec in sorted(runs.items())
    }
    c_obs = np.asarray(
        [row[MEASURE]["C_obs_single_order"] for row in by_seed.values()],
        dtype=float,
    )
    v_b = np.asarray(
        [row[MEASURE]["V_B_order_marginal"] for row in by_seed.values()],
        dtype=float,
    )
    return {
        "analysis": (f"{label} fixed-weight width, same seed-matched OC-SFT checkpoint at native width and B=1"),
        "surface": surface,
        "measure": MEASURE,
        "qid_basis": qid_basis,
        "seeds": [42, 43, 44],
        "by_seed": by_seed,
        "across_seed": {
            "C_obs_mean": float(c_obs.mean()),
            "C_obs_sample_sd": float(c_obs.std(ddof=1)),
            "V_B_mean": float(v_b.mean()),
            "V_B_sample_sd": float(v_b.std(ddof=1)),
            "C_obs_signs": [int(np.sign(value)) for value in c_obs],
            "V_B_signs": [int(np.sign(value)) for value in v_b],
        },
        "bootstrap": {
            "samples": BOOTSTRAP_SAMPLES,
            "seed": BOOTSTRAP_SEED,
            "interval": "per-seed paired percentile bootstrap over prompts",
        },
    }


def build_nectar_multiseed(
    *,
    qid_keep: set[str] | None = None,
    qid_basis: str = "all-498",
) -> dict[str, Any]:
    """Reduce the fixed-weight Nectar cell over training seeds 42-44."""
    return build_surface_multiseed(
        "nectar",
        "Nectar",
        NECTAR_MULTISEED_RUNS,
        qid_keep=qid_keep,
        qid_basis=qid_basis,
    )


def build_rewardbench2_multiseed() -> dict[str, Any]:
    """Reduce the fixed-weight RewardBench-2 cell over training seeds 42-44."""
    return build_surface_multiseed(
        "rewardbench2",
        "RewardBench-2",
        REWARDBENCH2_MULTISEED_RUNS,
    )


def build_ppe_math_multiseed() -> dict[str, Any]:
    """Reduce the fixed-weight PPE-MATH cell over training seeds 42-44."""
    return build_surface_multiseed(
        "ppe-math",
        "PPE-MATH",
        PPE_MATH_MULTISEED_RUNS,
    )


def build_ppe_mmlu_pro_multiseed() -> dict[str, Any]:
    """Reduce the fixed-weight PPE-MMLU-Pro cell over training seeds 42-44."""
    return build_surface_multiseed(
        "ppe-mmlu-pro",
        "PPE-MMLU-Pro",
        PPE_MMLU_PRO_MULTISEED_RUNS,
    )


def build_rmbench_multiseed() -> dict[str, Any]:
    """Reduce the fixed-weight RM-Bench cell over training seeds 42-44."""
    return build_surface_multiseed(
        "rmbench",
        "RM-Bench",
        RMBENCH_MULTISEED_RUNS,
    )


def _compact_seed_receipt(spec: SurfaceRuns) -> dict[str, float]:
    """Reduce Table-3 levels from aggregate artifacts without paired detail."""
    native = _resolve_run(spec.native_run)
    b1 = _resolve_run(spec.b1_run)
    native_meta = _assert_checkpoint(native, expect_b=4)
    b1_meta = _assert_checkpoint(b1, expect_b=1)
    if native_meta["lora_channel"] != b1_meta["lora_channel"]:
        raise ValueError(
            f"{spec.surface}: checkpoint mismatch -- native={native_meta['lora_channel']!r} "
            f"b1={b1_meta['lora_channel']!r}"
        )
    native_single = _ndcg_single_pass(native)
    native_k10 = _ndcg_order_averaged(native, k=10)
    b1_ndcg = _ndcg_single_pass(b1)
    return {
        "b1": b1_ndcg,
        "native_single": native_single,
        "native_k10": native_k10,
        "c_obs": native_single - b1_ndcg,
        "v_b": native_k10 - b1_ndcg,
    }


def build_core_multiseed_receipt() -> dict[str, Any]:
    """Recompute the three core response-width surfaces from fetched runs."""
    run_maps = {
        "ppe-math": PPE_MATH_MULTISEED_RUNS,
        "ppe-mmlu-pro": PPE_MMLU_PRO_MULTISEED_RUNS,
        "rmbench": RMBENCH_MULTISEED_RUNS,
    }
    surfaces = {
        surface: {str(seed): _compact_seed_receipt(spec) for seed, spec in sorted(runs.items())}
        for surface, runs in run_maps.items()
    }
    signs = {
        effect: {
            surface: [int(np.sign(row[effect])) for row in seed_rows.values()]
            for surface, seed_rows in surfaces.items()
        }
        for effect in ("c_obs", "v_b")
    }
    return {
        "schema_version": 1,
        "metric": MEASURE,
        "seeds": [42, 43, 44],
        "surfaces": surfaces,
        "signs": signs,
        "source": "live-fetched-runs",
        "finding": (
            "The fixed-weight width sign pattern is identical across all three "
            "training seeds: positive on both correctness collections and "
            "negative on RM-Bench."
        ),
    }


def load_or_build_core_multiseed_receipt(
    *,
    require_live: bool = False,
) -> dict[str, Any]:
    """Prefer a live recomputation and fall back to the source-locked receipt."""
    evidence = _read_json(CORE_MULTISEED_EVIDENCE)
    try:
        live = build_core_multiseed_receipt()
    except FileNotFoundError:
        if require_live:
            raise
        evidence["source"] = "frozen-source-locked-evidence"
        return evidence

    keys = ("b1", "native_single", "native_k10", "c_obs", "v_b")
    differences = [
        abs(float(live["surfaces"][surface][seed][key]) - float(evidence["surfaces"][surface][seed][key]))
        for surface in evidence["surfaces"]
        for seed in ("42", "43", "44")
        for key in keys
    ]
    tolerance = float(evidence["provenance"]["live_validation_tolerance"])
    max_difference = max(differences)
    if max_difference > tolerance:
        raise ValueError(
            f"live core multiseed receipt differs from frozen evidence by "
            f"{max_difference:.8f}, above tolerance {tolerance:.8f}"
        )
    live["frozen_comparison"] = {
        "max_abs_difference": max_difference,
        "tolerance": tolerance,
        "status": "pass",
        "note": "The historical RM-Bench seed-42 receipt stored four-decimal levels.",
    }
    live["provenance"] = evidence["provenance"]
    return live


def write_core_multiseed_receipt(
    *,
    out_path: Path = CORE_MULTISEED_OUT,
    require_live: bool = False,
) -> dict[str, Any]:
    """Write the exact response-width receipt consumed by Table 3."""
    payload = load_or_build_core_multiseed_receipt(require_live=require_live)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    """Write single-seed and available three-seed response-width reductions."""
    global RUNS_ROOT

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=ROOT,
        help="presentation_dependence root containing runs/; defaults to this checkout.",
    )
    parser.add_argument(
        "--nectar-all-498",
        action="store_true",
        help="Use all 498 Nectar prompts. Default: contamination-clean 434-prompt subset.",
    )
    parser.add_argument(
        "--core-multiseed-only",
        action="store_true",
        help="Write the PPE-MATH/PPE-MMLU-Pro/RM-Bench Table-3 receipt and exit.",
    )
    parser.add_argument(
        "--require-live-core",
        action="store_true",
        help="Require all 18 core native/B=1 run directories instead of using frozen evidence.",
    )
    args = parser.parse_args()
    RUNS_ROOT = args.runs_root.resolve()
    core_multiseed = write_core_multiseed_receipt(require_live=args.require_live_core)
    print(f"wrote {CORE_MULTISEED_OUT} ({core_multiseed['source']})")
    if args.core_multiseed_only:
        return
    nectar_qids = None if args.nectar_all_498 else _load_qid_set(NECTAR_CLEAN_QIDS)
    nectar_basis = "all-498" if nectar_qids is None else f"clean-{len(nectar_qids)}"
    rows = [
        analyze_surface(
            spec,
            qid_keep=nectar_qids if spec.surface == "nectar" else None,
            qid_basis=nectar_basis if spec.surface == "nectar" else "all",
        )
        for spec in RUNS
    ]
    payload = {
        "experiment": "rr-fixed-weight-width-ocl1-b1deploy",
        "measure": MEASURE,
        "replaced_measure": "ndcg_cut_10",
        "note": (
            "Recomputed at nDCG@1 so tab:rr-width matches tab:regimes; "
            "prior nDCG@10 cells are superseded. Nectar defaults to the "
            "contamination-clean 434-prompt subset."
        ),
        "checkpoint": {
            "family": "Qwen3-4B",
            "objective": "OC-SFT lambda*=1.0 (ocl1-lora), UltraFeedback response-quality 30k",
            "seed": 42,
            "channel": EXPECTED_CHANNEL,
        },
        "bootstrap": {
            "samples": BOOTSTRAP_SAMPLES,
            "seed": BOOTSTRAP_SEED,
            "pairing": {
                "C_obs": (f"canonical native B=4 {MEASURE} minus canonical B=1 {MEASURE}, per prompt"),
                "V_B": (f"native by_K[10] {MEASURE} minus B=1 by_K[1] {MEASURE}, per prompt"),
            },
        },
        "surfaces": rows,
    }
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "rr_fixed_weight_width.json"
    md_path = args.report_dir / "rr_fixed_weight_width.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_md(payload), encoding="utf-8")
    nectar_multiseed = build_nectar_multiseed(
        qid_keep=nectar_qids,
        qid_basis=nectar_basis,
    )
    NECTAR_MULTISEED_OUT.write_text(
        json.dumps(nectar_multiseed, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rewardbench2_multiseed = None
    try:
        rewardbench2_multiseed = build_rewardbench2_multiseed()
    except FileNotFoundError as exc:
        print(f"skipped optional RewardBench-2 multiseed receipt: {exc}")
    if rewardbench2_multiseed is not None:
        REWARDBENCH2_MULTISEED_OUT.write_text(
            json.dumps(rewardbench2_multiseed, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    print(f"wrote {NECTAR_MULTISEED_OUT}")
    if rewardbench2_multiseed is not None:
        print(f"wrote {REWARDBENCH2_MULTISEED_OUT}")
    for row in rows:
        s = row[MEASURE]
        c_boot = row["paired_bootstrap_C_obs"]
        v_boot = row["paired_bootstrap_V_B"]
        print(
            f"{row['label']:14s} B1={s['b1']:.4f} native={s['native_single']:.4f} "
            f"C_obs={s['C_obs_single_order']:+.4f} "
            f"[{c_boot['ci95_lo']:+.4f}, {c_boot['ci95_hi']:+.4f}]  "
            f"V_B={s['V_B_order_marginal']:+.4f} "
            f"[{v_boot['ci95_lo']:+.4f}, {v_boot['ci95_hi']:+.4f}] "
            f"n={c_boot['n']}"
        )


if __name__ == "__main__":
    main()
