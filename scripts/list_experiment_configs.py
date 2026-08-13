#!/usr/bin/env python
"""List experiment YAMLs with a few searchable fields."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _dataset_token(cfg: dict[str, Any]) -> str:
    data = cfg.get("data") or {}
    run_path = data.get("run_path")
    if run_path:
        parent = Path(str(run_path)).parent.name
        if parent:
            return parent
    topics = data.get("topics")
    return str(topics) if topics else "-"


def _row_for_config(path: Path) -> dict[str, str]:
    cfg = yaml.safe_load(path.read_text()) or {}
    rb = cfg.get("robustness")
    return {
        "id": str(cfg.get("id", path.stem)),
        "file": path.name,
        "class": str((cfg.get("reranker") or {}).get("class", "-")),
        "dataset": _dataset_token(cfg),
        "robustness": "yes" if rb else "no",
    }


def iter_config_rows(config_dir: Path) -> list[dict[str, str]]:
    """Return one summary row per experiment config."""
    return [_row_for_config(path) for path in sorted(config_dir.glob("*.yaml"))]


def _matches(row: dict[str, str], *, query: str | None, model: str | None, dataset: str | None) -> bool:
    haystack = " ".join(row.values()).lower()
    if query and query.lower() not in haystack:
        return False
    if model and model.lower() not in row["class"].lower() and model.lower() not in row["id"].lower():
        return False
    if dataset and dataset.lower() not in row["dataset"].lower() and dataset.lower() not in row["id"].lower():
        return False
    return True


def format_rows(rows: list[dict[str, str]]) -> str:
    """Format rows as a simple stable table."""
    headers = ("id", "class", "dataset", "robustness", "file")
    widths = {h: len(h) for h in headers}
    for row in rows:
        for h in headers:
            widths[h] = max(widths[h], len(row[h]))

    out = ["  ".join(h.ljust(widths[h]) for h in headers)]
    out.append("  ".join("-" * widths[h] for h in headers))
    for row in rows:
        out.append("  ".join(row[h].ljust(widths[h]) for h in headers))
    return "\n".join(out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--query",
        "-q",
        help="Substring search across id/class/dataset/file.",
    )
    parser.add_argument("--model", help="Filter by reranker class or id token, e.g. qwen3 or RankZephyr.")
    parser.add_argument("--dataset", help="Filter by dataset/run-path token, e.g. beir, dl20, legal-a.")
    parser.add_argument("--robustness", choices=["yes", "no"], help="Filter on presence of a robustness block.")
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=_project_root() / "configs" / "experiments",
        help="Directory containing experiment YAMLs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = [
        row
        for row in iter_config_rows(args.config_dir)
        if _matches(row, query=args.query, model=args.model, dataset=args.dataset)
        and (args.robustness is None or row["robustness"] == args.robustness)
    ]
    print(format_rows(rows))
    print(f"\n{len(rows)} config(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
