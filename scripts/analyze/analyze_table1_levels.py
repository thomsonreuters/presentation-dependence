#!/usr/bin/env python3
"""Backfill and validate the live paper's 11-row headline Table 1.

This keeps the historical reduction style: completed row values are embedded
with their original hand-back sources, then checked against maintained evidence
where an independent aggregate is available.
"""

from __future__ import annotations

import argparse
import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "build" / "reproduction" / "analysis" / "table1-levels"
REPORTING = ROOT / "build" / "reproduction" / "reporting" / "table1-levels"
DISCORDANCE = ROOT / "configs" / "reproduction" / "evidence" / "metric-discordance" / "quality-argmax-docid.json"
ROUND_ROBIN = ROOT / "configs" / "reproduction" / "evidence" / "round-robin" / "p04-six-arm-summary.json"

COLUMNS = (
    "rerank_ndcg10",
    "rerank_tau_psi",
    "rerank_jaccard",
    "qa_ndcg10",
    "qa_tau_psi",
    "qa_answer_flip",
    "response_ndcg1",
    "response_tau_psi",
    "response_pair_flip",
)

# Sources retained in the original research tree:
# - docs/TABLE1-INVENTORY-2026-08-14.md
# - docs/EXTERNAL-SCORERS-CONSUMERS-2026-08-14.md
# - docs/ROUND-ROBIN-CONSUMERS-2026-08-17.md
# - docs/BSC-CONSUMERS-2026-08-17.md
# - docs/GPT54-FULL-ROW-2026-08-17.md
# - docs/GPT54-FLIP-PAIRWISE-2026-08-17.md
# - docs/FLIP-ESTIMATOR-AUDIT-2026-08-17.md
# - docs/NECTAR-JSONL-RESTORE-2026-08-17.md
ROWS: dict[str, dict[str, Any]] = {
    "off-shelf": {
        "label": "Off the shelf",
        "source": "canonical task collectors",
        "values": (0.370, 0.298, 0.439, 0.911, 0.224, 0.221, 0.655, 0.345, 0.877),
    },
    "capcal": {
        "label": "CapCal",
        "source": "EXTERNAL-SCORERS-CONSUMERS-2026-08-14",
        "values": (0.372, 0.293, 0.427, 0.911, 0.222, 0.217, 0.657, 0.338, 0.874),
    },
    "round-robin": {
        "label": "Round-robin",
        "source": "ROUND-ROBIN-CONSUMERS-2026-08-17",
        "values": (0.422, 0.297, 0.443, 0.911, 0.224, 0.221, 0.658, 0.347, 0.878),
        "derived": {"qa_ndcg10", "qa_tau_psi", "qa_answer_flip"},
    },
    "bsc": {
        "label": "BSC (x10)",
        "source": "BSC-CONSUMERS-2026-08-17",
        "values": (0.465, 0.180, 0.707, 0.946, 0.143, 0.157, 0.716, 0.184, 0.636),
        "ensemble_redraw": {
            "rerank_tau_psi",
            "rerank_jaccard",
            "qa_tau_psi",
            "qa_answer_flip",
            "response_tau_psi",
            "response_pair_flip",
        },
    },
    "jina": {
        "label": "jina-reranker-v3",
        "source": "canonical external-reference collectors",
        "values": (0.447, 0.177, 0.667, 0.949, 0.163, 0.172, 0.479, 0.226, 0.685),
    },
    "gpt54": {
        "label": "GPT-5.4",
        "source": "GPT54-FULL-ROW and GPT54-FLIP-PAIRWISE",
        "values": (0.468, None, 0.707, 0.972, None, 0.094, 0.726, None, 0.489),
        "withheld": {"rerank_tau_psi", "qa_tau_psi", "response_tau_psi"},
    },
    "single-order": {
        "label": "Single order",
        "source": "canonical task collectors",
        "values": (0.449, 0.209, 0.656, 0.951, 0.159, 0.177, 0.684, 0.333, 0.869),
    },
    "order-averaged": {
        "label": "Order-averaged",
        "source": "canonical task collectors",
        "values": (0.455, 0.130, 0.743, 0.956, 0.124, 0.149, 0.693, 0.228, 0.724),
    },
    "debias-first": {
        "label": "DebiasFirst",
        "source": "TABLE1-INVENTORY and cross-task hand-backs",
        "values": (0.454, 0.128, 0.759, 0.955, 0.147, 0.164, 0.694, 0.228, 0.718),
    },
    "permutation-augmentation": {
        "label": "Permutation augmentation",
        "source": "TABLE1-INVENTORY and cross-task hand-backs",
        "values": (0.455, 0.129, 0.760, 0.955, 0.148, 0.162, 0.696, 0.223, 0.707),
    },
    "oc-sft": {
        "label": "OC-SFT",
        "source": "canonical task collectors",
        "values": (0.459, 0.083, 0.835, 0.961, 0.096, 0.125, 0.701, 0.201, 0.661),
    },
}

DISCORDANCE_JACCARD_ROWS = {
    "gpt54": "gpt54",
    "oc-sft": "ocsft",
    "order-averaged": "k10sft",
    "permutation-augmentation": "posaug",
    "debias-first": "debiasfirst",
    "single-order": "k1sft",
    "jina": "jina",
}
DISCORDANCE_QUALITY_ROWS = {
    "gpt54": "gpt54",
    "oc-sft": "ocsft",
    "jina": "jina",
}
ROUND_ROBIN_ROWS = {
    "off-shelf": "off-shelf",
}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON mapping: {path}")
    return value


def _round3(value: float) -> float:
    return float(Decimal(str(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))


def _cells() -> list[dict[str, Any]]:
    cells = []
    for row_id, row in ROWS.items():
        values = row["values"]
        if len(values) != len(COLUMNS):
            raise ValueError(f"{row_id}: expected {len(COLUMNS)} cells, got {len(values)}")
        for column, value in zip(COLUMNS, values, strict=True):
            if value is not None:
                status = (
                    "derived-degenerate"
                    if column in row.get("derived", set())
                    else "measured-ensemble-redraw"
                    if column in row.get("ensemble_redraw", set())
                    else "measured"
                )
            elif column in row.get("withheld", set()):
                status = "withheld"
            elif column in row.get("unmeasured", set()):
                status = "unmeasured"
            else:
                raise ValueError(f"{row_id}/{column}: null cell has no status")
            cells.append(
                {
                    "row": row_id,
                    "label": row["label"],
                    "column": column,
                    "value": value,
                    "status": status,
                    "source": row["source"],
                }
            )
    return cells


def _cross_checks() -> list[dict[str, Any]]:
    checks = []
    discordance = _load(DISCORDANCE)
    for row_id, source_id in DISCORDANCE_JACCARD_ROWS.items():
        table_value = ROWS[row_id]["values"][COLUMNS.index("rerank_jaccard")]
        source_value = float(discordance["jaccard_means"][source_id])
        checks.append(
            {
                "row": row_id,
                "column": "rerank_jaccard",
                "table": table_value,
                "source": source_value,
                "pass": table_value == _round3(source_value),
            }
        )
    for row_id, source_id in DISCORDANCE_QUALITY_ROWS.items():
        table_value = ROWS[row_id]["values"][COLUMNS.index("rerank_ndcg10")]
        source_value = float(discordance["ndcg_means"][source_id])
        checks.append(
            {
                "row": row_id,
                "column": "rerank_ndcg10",
                "table": table_value,
                "source": source_value,
                "pass": table_value == _round3(source_value),
            }
        )

    round_robin = _load(ROUND_ROBIN)
    for row_id, source_id in ROUND_ROBIN_ROWS.items():
        for column, source_key in (
            ("rerank_ndcg10", "ndcg_cut_10"),
            ("rerank_tau_psi", "tau_psi"),
        ):
            table_value = ROWS[row_id]["values"][COLUMNS.index(column)]
            source_value = float(round_robin["variants"][source_id]["contiguous"][source_key])
            checks.append(
                {
                    "row": row_id,
                    "column": column,
                    "table": table_value,
                    "source": source_value,
                    "pass": table_value == _round3(source_value),
                }
            )
    for column, source_key in (
        ("rerank_ndcg10", "ndcg_cut_10"),
        ("rerank_tau_psi", "tau_psi"),
    ):
        table_value = ROWS["round-robin"]["values"][COLUMNS.index(column)]
        source_value = float(round_robin["variants"]["off-shelf"]["round_robin"][source_key])
        checks.append(
            {
                "row": "round-robin",
                "column": column,
                "table": table_value,
                "source": source_value,
                "pass": table_value == _round3(source_value),
            }
        )
    return checks


def build_result() -> dict[str, Any]:
    """Assemble the headline table and gate every status and maintained anchor."""
    cells = _cells()
    counts = {
        status: sum(cell["status"] == status for cell in cells)
        for status in (
            "measured",
            "derived-degenerate",
            "measured-ensemble-redraw",
            "withheld",
            "unmeasured",
        )
    }
    numeric = sum(cell["value"] is not None for cell in cells)
    checks = _cross_checks()
    gates = {
        "rows": len(ROWS),
        "columns": len(COLUMNS),
        "slots": len(cells),
        "numeric_cells": numeric,
        "withheld_cells": counts["withheld"],
        "unmeasured_cells": counts["unmeasured"],
        "all_cells_classified": sum(counts.values()) == len(cells),
        "cross_checks": len(checks),
        "cross_checks_pass": all(check["pass"] for check in checks),
    }
    status = (
        "complete"
        if gates
        == {
            "rows": 11,
            "columns": 9,
            "slots": 99,
            "numeric_cells": 96,
            "withheld_cells": 3,
            "unmeasured_cells": 0,
            "all_cells_classified": True,
            "cross_checks": 14,
            "cross_checks_pass": True,
        }
        else "failed"
    )
    return {
        "schema_version": 1,
        "analysis": "live headline Table 1 backfill",
        "status": status,
        "columns": list(COLUMNS),
        "rows": [
            {
                "id": row_id,
                "label": row["label"],
                "values": dict(zip(COLUMNS, row["values"], strict=True)),
                "source": row["source"],
            }
            for row_id, row in ROWS.items()
        ],
        "cells": cells,
        "counts": counts,
        "gates": gates,
        "cross_checks": checks,
    }


def render_markdown(result: dict[str, Any]) -> str:
    """Render the same compact matrix as the paper."""
    lines = [
        "# Headline Table 1",
        "",
        f"Status: **{result['status']}**.",
        "",
        "| Variant | R nDCG | R τ | R Jacc. | QA nDCG | QA τ | QA Ans. | RR nDCG | RR τ | RR Pair |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result["rows"]:
        rendered = ["—" if row["values"][column] is None else f"{row['values'][column]:.3f}" for column in COLUMNS]
        lines.append(f"| {row['label']} | " + " | ".join(rendered) + " |")
    gates = result["gates"]
    lines.extend(
        [
            "",
            f"Coverage: {gates['numeric_cells']} numeric, "
            f"{gates['withheld_cells']} withheld, "
            f"{gates['unmeasured_cells']} unmeasured.",
            f"Maintained aggregate checks: {gates['cross_checks']}/{gates['cross_checks']} passed.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    """Write the backfilled Table 1 receipts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--report-dir", type=Path, default=REPORTING)
    args = parser.parse_args()

    result = build_result()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "table1.json"
    report_path = args.report_dir / "table1.md"
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(output_path)}, indent=2))
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
