#!/usr/bin/env python3
"""Validate the paper's per-collection quality-versus-decision argmax claim."""

from __future__ import annotations

import argparse
import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any

from presentation_dependence.analysis.terminology import paper_method_label


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "configs" / "reproduction" / "evidence" / "metric-discordance" / "quality-argmax-docid.json"
OUTPUT = ROOT / "build" / "reproduction" / "analysis" / "metric-discordance"
REPORTING = ROOT / "build" / "reproduction" / "reporting" / "metric-discordance"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON mapping: {path}")
    return value


def _round_half_up(value: float, places: int = 3) -> float:
    quantum = Decimal(1).scaleb(-places)
    return float(Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP))


def build_result(evidence_path: Path = EVIDENCE) -> dict[str, Any]:
    """Validate collection coverage, argmax identities, and headline gates."""
    evidence = _load(evidence_path)
    systems = tuple(map(str, evidence["protocol"]["systems"]))
    system_labels = {system: paper_method_label(system) for system in systems}
    if evidence.get("system_labels") != system_labels:
        raise ValueError("metric-discordance system labels do not match canonical paper terminology")
    rows = [dict(row) for row in evidence["rows"]]
    datasets = [str(row["dataset"]) for row in rows]
    if len(datasets) != len(set(datasets)):
        raise ValueError("metric-discordance evidence contains duplicate datasets")

    invalid_systems = sorted(
        {
            str(row[key])
            for row in rows
            for key in ("best_ndcg_system", "best_jaccard_system")
            if str(row[key]) not in systems
        }
    )
    if invalid_systems:
        raise ValueError(f"unknown systems in metric-discordance evidence: {invalid_systems}")

    inconsistent = [
        str(row["dataset"])
        for row in rows
        if bool(row["differ"]) != (str(row["best_ndcg_system"]) != str(row["best_jaccard_system"]))
    ]
    if inconsistent:
        raise ValueError(f"inconsistent argmax-difference flags: {inconsistent}")

    n_differ = sum(bool(row["differ"]) for row in rows)
    ndcg_display = [_round_half_up(float(value)) for value in evidence["ndcg_means"].values()]
    ndcg_span = max(ndcg_display) - min(ndcg_display)
    expected = evidence["claim_gates"]
    gates = {
        "seven_systems": len(systems) == 7,
        "primary18": len(rows) == int(expected["expected_collections"]) == 18,
        "different_argmax_count": n_differ,
        "different_argmax_expected": int(expected["expected_different_argmax"]),
        "different_argmax_gate_pass": n_differ == int(expected["expected_different_argmax"]),
        "ndcg_mean_span_at_3dp": ndcg_span,
        "ndcg_mean_span_expected": float(expected["reported_ndcg_mean_span_at_3dp"]),
        "ndcg_span_gate_pass": abs(ndcg_span - float(expected["reported_ndcg_mean_span_at_3dp"])) < 1e-12,
    }
    status = (
        "complete"
        if all(
            (
                gates["seven_systems"],
                gates["primary18"],
                gates["different_argmax_gate_pass"],
                gates["ndcg_span_gate_pass"],
            )
        )
        else "failed"
    )
    return {
        "schema_version": 1,
        "analysis": "quality-versus-retained-set argmax discordance",
        "status": status,
        "protocol": evidence["protocol"],
        "system_labels": system_labels,
        "ndcg_means": evidence["ndcg_means"],
        "jaccard_means": evidence["jaccard_means"],
        "rows": rows,
        "gates": gates,
        "provenance": {
            "evidence": str(evidence_path.relative_to(ROOT)),
            "source_receipts": evidence["provenance"],
        },
    }


def render_markdown(result: dict[str, Any]) -> str:
    """Render the checkable per-collection matrix used by the body claim."""
    labels = result["system_labels"]
    lines = [
        "# Metric discordance: quality versus retained-set overlap",
        "",
        f"Status: **{result['status']}**.",
        "",
        "| Collection | Best nDCG@10 | Best retained-set Jaccard | Different? |",
        "| --- | --- | --- | --- |",
    ]
    for row in result["rows"]:
        lines.append(
            f"| {row['dataset']} | "
            f"{labels[row['best_ndcg_system']]} ({row['best_ndcg']:.3f}) | "
            f"{labels[row['best_jaccard_system']]} ({row['best_jaccard']:.3f}) | "
            f"{'yes' if row['differ'] else 'no'} |"
        )
    gates = result["gates"]
    lines.extend(
        [
            "",
            f"Different argmax systems: **{gates['different_argmax_count']} of 18**.",
            f"Seven-system 18-mean nDCG@10 span at paper precision: **{gates['ndcg_mean_span_at_3dp']:.3f}**.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    """Write machine-readable and human-readable claim receipts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=EVIDENCE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--report-dir", type=Path, default=REPORTING)
    args = parser.parse_args()

    result = build_result(args.evidence)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "analysis.json"
    report_path = args.report_dir / "report.md"
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(output_path)}, indent=2))
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
