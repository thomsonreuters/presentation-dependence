#!/usr/bin/env python3
"""Screen a ported lambda against one training curve.

Teacher-strength cells start from the lambda candidate encoded by their tracked
study materializer. Checks that choice against the cell's
``progress.jsonl`` using the converged-region and collapse rules from
``scripts/select_lambda.py`` plus a relative stability check.

Train the cell once at the ported lambda, then screen the converged region
(``steps >= conv_step``) on three logged signals:

    collapse   min student_stdev < --collapse: LOWER_LAMBDA
               (penalty too strong; re-run one grid step down)
    quality    converged-argmax qrels_ndcg_cut_10 < --quality-floor (or NaN)
               : ABSTAIN (broken run; inspect, not a lambda move)
    stability  psi_permutation_score_variance_mean (converged median) materially
               above the student's self-distill baseline (x(1+--psivar-rel-tol))
               : RAISE_LAMBDA (consistency penalty too weak; one step up)
    else       PASS: accept ported lambda; checkpoint = converged nDCG argmax

The in-training PSI curve logs
``psi_permutation_score_variance_mean`` (per-doc permutation score variance),
not downstream tau-PSI@B. Screen stability relative to the
student's self-distillation baseline rather than an absolute tau band.

The tool reads a local fetched ``progress.jsonl`` and requires no GPU.

A ``LOWER_LAMBDA`` or ``RAISE_LAMBDA`` verdict requires a new training job at a
different lambda. The script only recommends that run; launch requires explicit
operator approval.

Usage:
    uv run python scripts/analyze/screen_lambda_from_curve.py \
        --config ts-qwen3-32b-to-qwen3-4b-k1-oc-sft-l500-warmup500-msmarco-30k \
        --baseline-config qwen3-4b-nonthink-k1-supervised-consistency-lambda500-warmup500-msmarco-30k
    uv run python scripts/analyze/screen_lambda_from_curve.py --selftest
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from select_lambda import find_latest_local_progress, parse_progress_jsonl  # noqa: E402
from presentation_dependence.utils.run_paths import default_runs_root  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def _default_runs_root() -> Path:
    """Anchor a relative runs root on the repo, so cwd does not move it."""
    root = default_runs_root()
    return root if root.is_absolute() else REPO / root


GRID = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
NDCG_METRIC = "student_eval/qrels_ndcg_cut_10"
PSIVAR_METRIC = "student_eval/psi_permutation_score_variance_mean"


def step_lambda(lam: float, direction: int) -> float | None:
    """Next grid lambda below (-1) / above (+1) lam, or None at the boundary."""
    try:
        i = GRID.index(lam)
    except ValueError:
        # snap to nearest grid point
        i = min(range(len(GRID)), key=lambda j: abs(GRID[j] - lam))
    j = i + direction
    return GRID[j] if 0 <= j < len(GRID) else None


def screen(
    ndcg: dict[int, float],
    min_stdev: float | None,
    psivar: dict[int, float] | None,
    *,
    lam: float,
    baseline_psivar: float | None = None,
    conv_step: int = 1000,
    collapse: float = 0.02,
    quality_floor: float = 0.40,
    psivar_rel_tol: float = 0.25,
) -> dict:
    """Pure single-arm verdict. Returns {decision, reason, step, ndcg, next_lambda}.

    decision in {PASS, LOWER_LAMBDA, RAISE_LAMBDA, ABSTAIN}. Offline-testable.
    """
    conv = {s: v for s, v in ndcg.items() if s >= conv_step and v is not None and math.isfinite(v)}
    if len(conv) < 2:
        return {
            "decision": "ABSTAIN",
            "reason": f"only {len(conv)} converged nDCG eval(s) (>= step {conv_step}) -- trajectory too short",
            "step": None,
            "ndcg": None,
            "next_lambda": None,
        }

    # 1. collapse guard (penalty too strong)
    if min_stdev is not None and min_stdev < collapse:
        return {
            "decision": "LOWER_LAMBDA",
            "reason": f"collapse: min student_stdev {min_stdev:.4f} < {collapse}",
            "step": None,
            "ndcg": None,
            "next_lambda": step_lambda(lam, -1),
        }

    # checkpoint = converged-region nDCG argmax
    best_step = max(conv, key=conv.get)
    best = conv[best_step]

    # 2. quality sanity (broken run, not a lambda move)
    if best < quality_floor:
        return {
            "decision": "ABSTAIN",
            "reason": f"converged-best nDCG {best:.4f} < sanity floor {quality_floor} -- likely broken run, inspect",
            "step": best_step,
            "ndcg": best,
            "next_lambda": None,
        }

    # 3. stability vs self-distill baseline (penalty too weak)
    if psivar and baseline_psivar is not None:
        cvar = [v for s, v in psivar.items() if s >= conv_step and v is not None and math.isfinite(v)]
        if cvar:
            cur = statistics.median(cvar)
            limit = baseline_psivar * (1.0 + psivar_rel_tol)
            if cur > limit:
                up = step_lambda(lam, +1)
                return {
                    "decision": "RAISE_LAMBDA" if up is not None else "ABSTAIN",
                    "reason": f"stability weak: perm-score-var {cur:.4g} > baseline "
                    f"{baseline_psivar:.4g} x(1+{psivar_rel_tol}) = {limit:.4g}"
                    + ("" if up is not None else " but already at top of grid"),
                    "step": best_step,
                    "ndcg": best,
                    "next_lambda": up,
                }
        stab = "ok"
    elif baseline_psivar is None:
        stab = "unchecked (no baseline psivar)"
    else:
        stab = "unchecked (no psivar curve)"

    return {
        "decision": "PASS",
        "reason": f"no collapse; nDCG {best:.4f} >= {quality_floor}; stability {stab}",
        "step": best_step,
        "ndcg": best,
        "next_lambda": None,
    }


# --------------------------------------------------------------------------- #
def _load(config: str, runs_root: Path):
    prog = find_latest_local_progress(runs_root, config)
    if prog is None:
        return None, None, None
    ndcg, min_stdev = parse_progress_jsonl(prog, metric=NDCG_METRIC)
    psivar, _ = parse_progress_jsonl(prog, metric=PSIVAR_METRIC)
    return ndcg, min_stdev, (psivar or None)


def _baseline_psivar(config: str | None, runs_root: Path, conv_step: int) -> float | None:
    if not config:
        return None
    _, _, psivar = _load(config, runs_root)
    if not psivar:
        return None
    cvar = [v for s, v in psivar.items() if s >= conv_step]
    return statistics.median(cvar) if cvar else None


def selftest() -> int:
    """Offline checks of the pure screen() decision logic on synthetic curves."""
    healthy_ndcg = {200: 0.41, 800: 0.46, 1000: 0.47, 1400: 0.475, 2000: 0.474}
    cases = [
        (
            "healthy port -> PASS",
            {
                "ndcg": healthy_ndcg,
                "min_stdev": 0.08,
                "psivar": {1000: 0.010, 1400: 0.009},
                "lam": 5.0,
                "baseline_psivar": 0.010,
            },
            "PASS",
        ),
        (
            "collapse -> LOWER_LAMBDA(4.0)",
            {"ndcg": healthy_ndcg, "min_stdev": 0.005, "psivar": {1400: 0.001}, "lam": 5.0, "baseline_psivar": 0.010},
            "LOWER_LAMBDA",
        ),
        (
            "weak stability -> RAISE_LAMBDA(3.0)",
            {
                "ndcg": healthy_ndcg,
                "min_stdev": 0.08,
                "psivar": {1000: 0.030, 1400: 0.031},
                "lam": 2.0,
                "baseline_psivar": 0.010,
            },
            "RAISE_LAMBDA",
        ),
        (
            "broken quality -> ABSTAIN",
            {"ndcg": {1000: 0.30, 1400: 0.31}, "min_stdev": 0.08, "psivar": None, "lam": 5.0, "baseline_psivar": None},
            "ABSTAIN",
        ),
        (
            "short trajectory -> ABSTAIN",
            {"ndcg": {200: 0.45}, "min_stdev": 0.08, "psivar": None, "lam": 5.0, "baseline_psivar": None},
            "ABSTAIN",
        ),
        (
            "no baseline -> PASS (stability unchecked)",
            {"ndcg": healthy_ndcg, "min_stdev": 0.08, "psivar": {1400: 0.5}, "lam": 5.0, "baseline_psivar": None},
            "PASS",
        ),
        (
            "collapse at boundary lam=0.5 -> next_lambda None",
            {"ndcg": healthy_ndcg, "min_stdev": 0.001, "psivar": None, "lam": 0.5, "baseline_psivar": None},
            "LOWER_LAMBDA",
        ),
    ]
    ok = True
    for name, kw, expect in cases:
        r = screen(**kw)
        passed = r["decision"] == expect
        ok &= passed
        nl = r.get("next_lambda")
        print(
            f"[{'ok ' if passed else 'FAIL'}] {name:48} -> {r['decision']}"
            f"{f' (next lambda={nl})' if nl is not None else ''}"
        )
        if not passed:
            print(f"        expected {expect}; reason: {r['reason']}")
    print("\nSELFTEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="cell config id (run dir under runs/)")
    ap.add_argument("--lambda", dest="lam", type=float, help="the ported lambda being screened")
    ap.add_argument("--baseline-config", help="self-distill config id for the stability baseline")
    ap.add_argument("--baseline-psivar", type=float, default=None)
    ap.add_argument("--runs-root", type=Path, default=_default_runs_root())
    ap.add_argument("--conv-step", type=int, default=1000)
    ap.add_argument("--collapse", type=float, default=0.02)
    ap.add_argument("--quality-floor", type=float, default=0.40)
    ap.add_argument("--psivar-rel-tol", type=float, default=0.25)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    if not args.config or args.lam is None:
        ap.error("--config and --lambda are required (or use --selftest)")

    ndcg, min_stdev, psivar = _load(args.config, args.runs_root)
    if ndcg is None:
        print(
            f"[screen] no local progress.jsonl for {args.config} under {args.runs_root} (train + fetch the cell first)"
        )
        return 2
    baseline = args.baseline_psivar
    if baseline is None and args.baseline_config:
        baseline = _baseline_psivar(args.baseline_config, args.runs_root, args.conv_step)

    r = screen(
        ndcg,
        min_stdev,
        psivar,
        lam=args.lam,
        baseline_psivar=baseline,
        conv_step=args.conv_step,
        collapse=args.collapse,
        quality_floor=args.quality_floor,
        psivar_rel_tol=args.psivar_rel_tol,
    )
    print(f"[screen] {args.config}")
    print(f"  decision : {r['decision']}")
    print(f"  reason   : {r['reason']}")
    if r["step"] is not None:
        print(f"  checkpoint: step {r['step']} (nDCG {r['ndcg']:.4f})")
    if r["next_lambda"] is not None:
        print(f"  re-run at lambda={r['next_lambda']} (requires operator approval), then re-screen")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
