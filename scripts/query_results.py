#!/usr/bin/env python
"""Query docs/results_index.yaml for quick result overviews."""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Any

import yaml  # type: ignore[reportMissingModuleSource]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_record_result() -> ModuleType:
    path = Path(__file__).with_name("record_result.py")
    spec = importlib.util.spec_from_file_location("record_result", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _matches(record: dict[str, Any], field: str, allowed: list[str] | None) -> bool:
    if not allowed:
        return True
    return str(record.get(field) or "") in set(allowed)


def filter_records(
    rows: list[dict[str, Any]],
    *,
    status: list[str] | None = None,
    role: list[str] | None = None,
    surface: list[str] | None = None,
    series: list[str] | None = None,
    model_family: list[str] | None = None,
    benchmark_group: list[str] | None = None,
    has_psi: bool = False,
) -> list[dict[str, Any]]:
    """Filter result records by common dashboard fields."""
    out = []
    for row in rows:
        if has_psi and "psi" not in row:
            continue
        if not _matches(row, "status", status):
            continue
        if not _matches(row, "role", role):
            continue
        if not _matches(row, "surface", surface):
            continue
        if not _matches(row, "series", series):
            continue
        if not _matches(row, "model_family", model_family):
            continue
        if not _matches(row, "benchmark_group", benchmark_group):
            continue
        out.append(row)
    return out


def _counts(rows: list[dict[str, Any]], field: str) -> Counter[str]:
    return Counter(str(row.get(field) or "unknown") for row in rows)


def _format_counter(title: str, counts: Counter[str]) -> list[str]:
    lines = [f"{title}:"]
    for key, value in sorted(counts.items()):
        lines.append(f"  {key}: {value}")
    return lines


def format_summary(rows: list[dict[str, Any]]) -> str:
    """Return a compact text summary of selected result records."""
    lines = [f"{len(rows)} result record(s)"]
    for field, title in (
        ("status", "By status"),
        ("surface", "By collection"),
        ("model_family", "By model family"),
    ):
        lines.extend(_format_counter(title, _counts(rows, field)))
    return "\n".join(lines)


def format_markdown(rows: list[dict[str, Any]], *, record_result: ModuleType) -> str:
    """Return dashboard-ready markdown rows."""
    header = "| ID | Collection | Protocol | nDCG@10 | tau-PSI | Status | Summary |"
    sep = "| --- | --- | --- | ---: | ---: | --- | --- |"
    body = [record_result.emit_markdown_row(row) for row in rows]
    return "\n".join([header, sep, *body])


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index",
        type=Path,
        default=_project_root() / "docs" / "results_index.yaml",
        help="Machine-readable results index to query.",
    )
    parser.add_argument("--status", action="append", default=None)
    parser.add_argument("--role", action="append", default=None)
    parser.add_argument(
        "--collection",
        "--surface",
        dest="collection",
        action="append",
        default=None,
        help="Filter by collection (results_index field `surface`). --surface is a deprecated alias.",
    )
    parser.add_argument("--series", action="append", default=None)
    parser.add_argument("--model-family", action="append", default=None)
    parser.add_argument("--benchmark-group", action="append", default=None)
    parser.add_argument("--has-psi", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--format", choices=("summary", "markdown", "json"), default="summary")
    parser.add_argument(
        "--check-paths",
        action="store_true",
        help="Also require indexed local run directories to exist.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the results-query CLI."""
    args = parse_args()
    record_result = _load_record_result()
    index = _load_yaml(args.index)
    record_result.validate_results_index(
        index,
        index_root=_project_root() if args.check_paths else None,
    )
    rows = record_result.sort_result_records(list(index.get("results") or []))
    rows = filter_records(
        rows,
        status=args.status,
        role=args.role,
        surface=args.collection,
        series=args.series,
        model_family=args.model_family,
        benchmark_group=args.benchmark_group,
        has_psi=args.has_psi,
    )
    if args.limit is not None:
        rows = rows[: args.limit]

    if args.format == "json":
        print(json.dumps(rows, indent=2))
    elif args.format == "markdown":
        print(format_markdown(rows, record_result=record_result))
    else:
        print(format_summary(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
