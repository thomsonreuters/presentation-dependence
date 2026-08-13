"""Shared `--dry-run` reporting for the local runners.

Every runner resolves a config, applies overrides, then loads a model and reads
data. A dry run stops after the first half: it prints what would run and checks
that the declared inputs exist, without importing torch, touching a GPU, or
calling a provider.

That makes the wiring testable on any machine, which matters because the
expensive failures here are configuration failures -- a missing fixture, a
mistyped override key, a dataset that was never materialized -- and none of
those need an accelerator to detect.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

# Config keys that name an input file the run will read.
_INPUT_KEYS = (
    ("data", "run_path"),
    ("data", "topics_tsv"),
    ("data", "qrels_path"),
    ("eval", "qrels_path"),
    ("student", "fixture_path"),
    ("student", "qrels_path"),
    ("input", "fixture_path"),
    ("input", "qrels_path"),
)


def _dig(cfg: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    node: Any = cfg
    for key in path:
        if not isinstance(node, Mapping):
            return None
        node = node.get(key)
    return node


def declared_inputs(cfg: Mapping[str, Any], project_root: Path) -> list[dict[str, Any]]:
    """Return the input files this config declares, and whether each is present."""
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for path in _INPUT_KEYS:
        value = _dig(cfg, path)
        if not isinstance(value, str) or value in seen:
            continue
        seen.add(value)
        resolved = Path(value) if Path(value).is_absolute() else project_root / value
        rows.append({"key": ".".join(path), "path": value, "exists": resolved.is_file()})
    return rows


def emit_plan(
    kind: str,
    config_path: Path,
    cfg: Mapping[str, Any],
    *,
    project_root: Path,
    overrides: Iterable[str] = (),
    extra: Mapping[str, Any] | None = None,
) -> int:
    """Print the plan for a run and report whether its inputs are present.

    Returns a process exit code: ``0`` when every declared input exists, ``2``
    when one is missing, so a dry run is usable as a check in a pipeline.
    """
    inputs = declared_inputs(cfg, project_root)
    missing = [row["path"] for row in inputs if not row["exists"]]
    reranker = cfg.get("reranker") if isinstance(cfg.get("reranker"), Mapping) else {}
    plan: dict[str, Any] = {
        "kind": kind,
        "mode": "dry-run",
        "config": str(config_path),
        "id": cfg.get("id") or cfg.get("experiment_id"),
        "overrides": list(overrides),
        "reranker_class": reranker.get("class"),
        "model": reranker.get("model_name") or reranker.get("model_id"),
        "inputs": inputs,
        "missing_inputs": missing,
        "executed": False,
    }
    if extra:
        plan.update(extra)
    print(json.dumps(plan, indent=2, sort_keys=True, default=str))
    if missing:
        print(f"[dry-run] {len(missing)} declared input(s) missing; materialize them before a real run.")
        return 2
    return 0
