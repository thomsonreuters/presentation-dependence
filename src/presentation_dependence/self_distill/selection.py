"""Checkpoint and OC-SFT lambda selection rules."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


_FLOAT_EPS = 1e-12


@dataclass(slots=True)
class LambdaResult:
    """Held-out quality trajectory for one lambda candidate."""

    lam: float
    code: str
    epoch: int | None = None
    n_runs: int = 0
    traj: dict[int, float] = field(default_factory=dict)
    min_stdev: float | None = None
    missing: bool = False

    @property
    def global_step(self) -> int:
        """Step with the globally best trajectory value."""
        return max(self.traj, key=self.traj.get)

    @property
    def global_best(self) -> float:
        """Globally best trajectory value."""
        return self.traj[self.global_step]

    @property
    def max_step(self) -> int:
        """Last observed trajectory step."""
        return max(self.traj) if self.traj else -1

    def converged(self, conv_step: int) -> dict[int, float]:
        """Return trajectory values after the warm-up boundary."""
        return {step: value for step, value in self.traj.items() if step >= conv_step}

    def conv_argmax(self, conv_step: int) -> tuple[int, float] | None:
        """Return earliest best converged checkpoint."""
        values = self.converged(conv_step)
        if not values:
            return None
        best = max(values.values())
        step = min(step for step, value in values.items() if value == best)
        return step, best


@dataclass(slots=True)
class LambdaDecision:
    """Auditable result of held-out OC-SFT lambda selection."""

    lambda_star: float
    step: int
    metric_value: float
    argmax_lambda: float
    argmax_value: float
    band: list[float]
    collapsed: list[float]
    missing: list[float]
    tie_invoked: bool
    boundary: bool
    warnings: list[str]
    errors: list[str]

    @property
    def confidence(self) -> str:
        """Selection confidence label."""
        if self.errors:
            return "ABSTAIN"
        return "LOW" if self.warnings else "HIGH"


def select_heldout_lambda(  # noqa: C901
    results: list[LambdaResult],
    *,
    standard_error: float,
    convergence_step: int,
    collapse_floor: float,
    min_final_step: int = 2000,
    quality_floor: float = 0.40,
    min_converged_evals: int = 3,
) -> LambdaDecision | None:
    """Apply the held-out 1-SE and collapse rules used by OC-SFT."""
    grid = sorted(result.lam for result in results)
    converged: dict[float, tuple[int, float]] = {}
    collapsed: list[float] = []
    missing: list[float] = []
    no_stdev: list[float] = []
    partial: list[float] = []
    for result in results:
        if result.missing or not result.traj:
            missing.append(result.lam)
            continue
        if result.min_stdev is None:
            no_stdev.append(result.lam)
        elif result.min_stdev < collapse_floor:
            collapsed.append(result.lam)
            continue
        selected = result.conv_argmax(convergence_step)
        if selected is None:
            missing.append(result.lam)
            continue
        if result.max_step < min_final_step:
            partial.append(result.lam)
        converged[result.lam] = selected
    if not converged:
        return None

    warnings: list[str] = []
    errors: list[str] = []
    if len(converged) < len(results) - len(results) // 2:
        errors.append(f"ABSTAIN: only {len(converged)}/{len(results)} lambda candidates are usable")
    best_value = max(value for _step, value in converged.values())
    argmax_lambda = max(lam for lam, (_step, value) in converged.items() if value == best_value)
    band = sorted(lam for lam, (_step, value) in converged.items() if value >= best_value - standard_error - _FLOAT_EPS)
    lambda_star = max(band)
    step, value = converged[lambda_star]
    winner = next(result for result in results if result.lam == lambda_star)
    n_converged = len(winner.converged(convergence_step))
    if n_converged < 2:
        errors.append(f"ABSTAIN: lambda={lambda_star} has only {n_converged} converged evaluations")
    if missing:
        warnings.append(f"incomplete lambda grid: missing={sorted(missing)}")
    if collapsed:
        warnings.append(f"collapsed candidates below score stdev {collapse_floor}: {sorted(collapsed)}")
    if no_stdev:
        warnings.append(f"collapse check unavailable for {sorted(no_stdev)}")
    if partial:
        warnings.append(f"partial training trajectories: {sorted(partial)}")
    spread = best_value - min(value for _step, value in converged.values())
    if spread < standard_error:
        warnings.append("lambda grid is flatter than the standard-error band")
    if len(band) == len(converged) and len(converged) > 2:
        warnings.append("1-SE band spans the full usable lambda grid")
    boundary = lambda_star == max(grid)
    if boundary:
        warnings.append("selected lambda is at the grid boundary")
    if 2 <= n_converged < min_converged_evals:
        warnings.append(f"selected lambda has only {n_converged} converged evaluations")
    if best_value < quality_floor:
        warnings.append(f"best held-out quality {best_value:.4f} is below {quality_floor}")
    return LambdaDecision(
        lambda_star=lambda_star,
        step=step,
        metric_value=round(value, 5),
        argmax_lambda=argmax_lambda,
        argmax_value=round(best_value, 5),
        band=band,
        collapsed=sorted(collapsed),
        missing=sorted(missing),
        tie_invoked=len(band) > 1,
        boundary=boundary,
        warnings=warnings,
        errors=errors,
    )


def parse_training_progress(path: Path, *, metric: str) -> tuple[dict[int, float], float | None]:
    """Build step-to-metric trajectory and minimum student score stdev."""
    field = metric.split("/")[-1]
    trajectory: dict[int, float] = {}
    stdevs: list[float] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("event") != "eval":
            continue
        step = int(row["step"])
        if row.get(field) is not None:
            trajectory[step] = float(row[field])
        if row.get("student_stdev") is not None:
            stdevs.append(float(row["student_stdev"]))
    return trajectory, min(stdevs) if stdevs else None


@dataclass(frozen=True)
class StabilityRow:
    """In-distribution stability evidence for one response-ranking recipe."""

    tau_psi: float
    score_variance: float
    quality: float | None = None


@dataclass(frozen=True)
class AmortizationDecision:
    """Smallest lambda matching the K-shot stability target."""

    lambda_star: float
    reference_tau_psi: float
    qualifiers: tuple[float, ...]


def select_amortization_matching(
    candidates: Mapping[float, StabilityRow],
    reference: StabilityRow,
    *,
    score_variance_floor: float = 0.0,
    tolerance: float = 1e-9,
) -> AmortizationDecision:
    """Select the smallest lambda matching reference dev stability."""
    qualifiers = sorted(
        lam
        for lam, row in candidates.items()
        if row.tau_psi <= reference.tau_psi + tolerance and row.score_variance >= score_variance_floor
    )
    if not qualifiers:
        raise ValueError("No lambda matches the amortization stability target")
    return AmortizationDecision(
        lambda_star=qualifiers[0],
        reference_tau_psi=reference.tau_psi,
        qualifiers=tuple(qualifiers),
    )


def best_checkpoint(
    summary: Mapping[str, Any],
    metric: str,
) -> tuple[int, float]:
    """Select maximum held-out metric with earliest-step exact tie break."""
    rows = [
        row for row in summary.get("eval_metrics") or [] if row.get(metric) is not None and row.get("step") is not None
    ]
    if not rows:
        raise ValueError(f"Training summary has no {metric} evaluations")
    value = max(float(row[metric]) for row in rows)
    step = min(int(row["step"]) for row in rows if float(row[metric]) == value)
    return step, value
