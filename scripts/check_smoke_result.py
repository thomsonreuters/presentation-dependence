#!/usr/bin/env python
"""Fail unless the latest synthetic smoke run matches its fixed contract."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from presentation_dependence.utils.run_paths import default_runs_root  # noqa: E402


EXPECTED_QUERIES = 6
EXPECTED_NDCG = 0.6804846774060783
ABS_TOLERANCE = 1e-12


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    runs_root = default_runs_root()
    if not runs_root.is_absolute():
        runs_root = root / runs_root
    exp_root = runs_root / "_smoke-fixture"
    candidates = sorted(trial for trial in exp_root.iterdir() if trial.is_dir() and (trial / "metrics.json").is_file())
    if not candidates:
        raise FileNotFoundError(f"No completed smoke metrics found under {exp_root}")

    metrics_path = candidates[-1] / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    failures: list[str] = []

    if int(metrics.get("n_queries", -1)) != EXPECTED_QUERIES:
        failures.append(f"n_queries={metrics.get('n_queries')!r}, expected {EXPECTED_QUERIES}")
    observed = float(metrics.get("mean_ndcg_cut_10", float("nan")))
    if not math.isclose(observed, EXPECTED_NDCG, rel_tol=0.0, abs_tol=ABS_TOLERANCE):
        failures.append(f"mean_ndcg_cut_10={observed!r}, expected {EXPECTED_NDCG!r}")
    if (metrics.get("coverage") or {}).get("complete") is not True:
        failures.append(f"coverage is incomplete: {metrics.get('coverage')!r}")

    if failures:
        raise RuntimeError(f"Smoke contract failed for {metrics_path}: {'; '.join(failures)}")
    print(f"Smoke contract passed: queries={EXPECTED_QUERIES}, mean_ndcg_cut_10={observed:.16f}, coverage=complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
