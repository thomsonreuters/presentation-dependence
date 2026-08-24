#!/usr/bin/env python3
"""Validate the completed six-arm contiguous-versus-round-robin study.

The tracked evidence receipt preserves the completed primary-18 result. When a
fresh collected result exists, the four generated trained arms are re-aggregated
from it and must reproduce the same claim gates.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from presentation_dependence.analysis.terminology import paper_method_label


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "configs/reproduction/evidence/round-robin/p04-six-arm-summary.json"
COLLECTED = ROOT / "build/reproduction/studies/mechanism-boundary/round-robin-eval-grid/results.json"
OUTPUT = ROOT / "build/reproduction/analysis/round-robin-six-arm"
REPORTING = ROOT / "build/reproduction/reporting/round-robin-six-arm"
GENERATED_VARIANTS = (
    "k1-sft",
    "k10-sft",
    "shuffled-view-augmentation",
    "debias-first",
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON mapping: {path}")
    return value


def _collected_round_robin(path: Path, expected_datasets: set[str]) -> dict[str, dict[str, float]]:
    payload = _load(path)
    cells: dict[str, dict[str, dict[str, float]]] = {variant: {} for variant in GENERATED_VARIANTS}
    for row in payload.get("rows") or []:
        variant = str(row["variant"])
        if variant not in cells:
            raise ValueError(f"unexpected collected round-robin variant: {variant}")
        dataset = str(row["dataset"])
        metrics = row["artifacts"]["metrics"]["payload"]
        robustness = row["artifacts"]["robustness"]["payload"]
        quality = metrics.get("mean_ndcg_cut_10", metrics.get("ndcg_cut_10"))
        aggregate = robustness.get("aggregate", robustness)
        tau_psi = aggregate.get("mean_tau_based_psi")
        if quality is None or tau_psi is None:
            raise ValueError(f"missing nDCG/tau-PSI in collected row {row['config']}")
        cells[variant][dataset] = {
            "ndcg_cut_10": float(quality),
            "tau_psi": float(tau_psi),
        }

    result = {}
    for variant, rows in cells.items():
        if set(rows) != expected_datasets:
            missing = sorted(expected_datasets - set(rows))
            extra = sorted(set(rows) - expected_datasets)
            raise ValueError(f"{variant}: collected datasets do not match primary-18; missing={missing}, extra={extra}")
        result[variant] = {
            "ndcg_cut_10": statistics.fmean(row["ndcg_cut_10"] for row in rows.values()),
            "tau_psi": statistics.fmean(row["tau_psi"] for row in rows.values()),
        }
    return result


def build_result(
    *,
    evidence_path: Path = EVIDENCE,
    collected_path: Path = COLLECTED,
    require_collected: bool = False,
) -> dict[str, Any]:
    """Build and gate the frozen result, optionally validating a fresh collection."""
    evidence = _load(evidence_path)
    variants = {
        variant: {
            assignment: {metric: float(value) for metric, value in cell.items()}
            for assignment, cell in assignments.items()
        }
        for variant, assignments in evidence["variants"].items()
    }
    collected_status = "not-present"
    if collected_path.is_file():
        collected = _collected_round_robin(collected_path, set(map(str, evidence["datasets"])))
        for variant, cell in collected.items():
            variants[variant]["round_robin"] = cell
        collected_status = "validated"
    elif require_collected:
        raise FileNotFoundError(f"missing fresh collected result: {collected_path}")

    rows = []
    for variant, assignments in variants.items():
        contiguous = assignments["contiguous"]
        round_robin = assignments["round_robin"]
        rows.append(
            {
                "variant": variant,
                "contiguous": contiguous,
                "round_robin": round_robin,
                "delta": {
                    "ndcg_cut_10": round_robin["ndcg_cut_10"] - contiguous["ndcg_cut_10"],
                    "tau_psi": round_robin["tau_psi"] - contiguous["tau_psi"],
                },
            }
        )

    max_tau_shift = max(abs(row["delta"]["tau_psi"]) for row in rows)
    by_variant = {row["variant"]: row for row in rows}
    contiguous_gain = (
        by_variant["oc-sft"]["contiguous"]["ndcg_cut_10"] - by_variant["off-shelf"]["contiguous"]["ndcg_cut_10"]
    )
    round_robin_gain = (
        by_variant["oc-sft"]["round_robin"]["ndcg_cut_10"] - by_variant["off-shelf"]["round_robin"]["ndcg_cut_10"]
    )
    max_allowed = float(evidence["claim_gates"]["max_abs_tau_psi_shift"])
    gates = {
        "six_variants": len(rows) == 6,
        "primary18": len(evidence["datasets"]) == 18,
        "max_abs_tau_psi_shift": max_tau_shift,
        "max_abs_tau_psi_shift_allowed": max_allowed,
        "tau_shift_gate_pass": max_tau_shift <= max_allowed,
        "contiguous_trained_gain": contiguous_gain,
        "round_robin_trained_gain": round_robin_gain,
        "round_robin_trained_gain_positive": round_robin_gain > 0,
        "round_robin_gain_smaller": round_robin_gain < contiguous_gain,
    }
    status = (
        "complete"
        if all(
            (
                gates["six_variants"],
                gates["primary18"],
                gates["tau_shift_gate_pass"],
                gates["round_robin_trained_gain_positive"],
                gates["round_robin_gain_smaller"],
            )
        )
        else "failed"
    )
    return {
        "schema_version": 1,
        "analysis": "six-arm contiguous versus round-robin primary-18",
        "status": status,
        "collected_status": collected_status,
        "protocol": evidence["protocol"],
        "datasets": evidence["datasets"],
        "rows": rows,
        "gates": gates,
        "provenance": {
            "evidence": str(evidence_path.relative_to(ROOT)),
            "source_receipts": evidence["provenance"],
            "fresh_collected": (
                str(collected_path.relative_to(ROOT)) if collected_path.is_relative_to(ROOT) else str(collected_path)
            ),
        },
    }


def render_markdown(result: dict[str, Any]) -> str:
    """Render a concise reporting handoff."""
    lines = [
        "# Six-arm round-robin study",
        "",
        f"Status: **{result['status']}**. Fresh collection: `{result['collected_status']}`.",
        "",
        "| Variant | Contiguous nDCG | Round-robin nDCG | Δ nDCG | Contiguous τ-PSI | Round-robin τ-PSI | Δ τ-PSI |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result["rows"]:
        lines.append(
            f"| {paper_method_label(row['variant'])} | "
            f"{row['contiguous']['ndcg_cut_10']:.4f} | "
            f"{row['round_robin']['ndcg_cut_10']:.4f} | "
            f"{row['delta']['ndcg_cut_10']:+.4f} | "
            f"{row['contiguous']['tau_psi']:.4f} | "
            f"{row['round_robin']['tau_psi']:.4f} | "
            f"{row['delta']['tau_psi']:+.4f} |"
        )
    gates = result["gates"]
    lines.extend(
        [
            "",
            f"Maximum absolute τ-PSI shift: **{gates['max_abs_tau_psi_shift']:.4f}** "
            f"(gate ≤ {gates['max_abs_tau_psi_shift_allowed']:.3f}).",
            f"Trained-over-off-the-shelf nDCG@10 gain: contiguous "
            f"**{gates['contiguous_trained_gain']:+.4f}**, round-robin "
            f"**{gates['round_robin_trained_gain']:+.4f}**.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    """Write the gated JSON and Markdown receipts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=EVIDENCE)
    parser.add_argument("--collected", type=Path, default=COLLECTED)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--report-dir", type=Path, default=REPORTING)
    parser.add_argument("--require-collected", action="store_true")
    args = parser.parse_args()

    result = build_result(
        evidence_path=args.evidence,
        collected_path=args.collected,
        require_collected=args.require_collected,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "analysis.json"
    markdown_path = args.report_dir / "report.md"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(json_path)}, indent=2))
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
