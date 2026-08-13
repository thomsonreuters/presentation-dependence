"""Write PSI result artefacts."""

from __future__ import annotations

from typing import Any

from presentation_dependence.eval.tau_psi_geometry import compact_tau_psi_geometry, infer_tau_psi_geometry


def build_psi_metrics_envelope(
    config: dict,
    metrics: dict,
    *,
    perturbations: list[str],
    K: int,
    seeds: list[int],
    reranker: object | None = None,
    aggregation_note: str | None = None,
    exp_id: str | None = None,
    render_strategies: list[str] | None = None,
) -> dict[str, Any]:
    """Build the top-level ``psi_metrics.json`` payload."""
    compact = {
        "exp_id": exp_id if exp_id is not None else config.get("id"),
        "aggregate": metrics["aggregate"],
        "protocol": metrics["protocol"],
        "perturbations": perturbations,
        "K": K,
        "seeds": seeds,
        **compact_tau_psi_geometry(infer_tau_psi_geometry(config, reranker)),
    }
    if render_strategies:
        compact["render_strategies"] = list(render_strategies)
        if render_strategies == ["scale"]:
            compact["symmetry_mode"] = "input_scale"
        else:
            compact["symmetry_mode"] = "input_render"
    if aggregation_note is not None:
        compact["_aggregation_note"] = aggregation_note
    return compact
