"""Set-sensitivity cell decomposition over retained aligned-score artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from presentation_dependence.eval.set_sensitivity import (
    assemble_doc_rows,
    compute_query_metrics,
    per_query_to_dicts,
    summarize_per_query,
)
from presentation_dependence.utils.trec import dirname_to_qid


@dataclass
class CellResult:
    """One analyzed set-sensitivity cell."""

    label: str
    exp_id: str
    role: str
    family: str
    run_dir: Path
    k_input: int
    batch_size: int
    seeds: list[int]
    n_queries: int
    n_queries_skipped: int
    summary: dict[str, Any]
    per_query: list[dict[str, Any]]


def _latest_trial(run_root: Path) -> Path | None:
    if not run_root.is_dir():
        return None
    trials = sorted(path for path in run_root.iterdir() if path.is_dir())
    return trials[-1] if trials else None


def _fixture_order(path: Path) -> dict[str, list[str]]:
    rows: dict[str, list[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            rows[str(record["qid"])] = [str(passage["pid"]) for passage in record["passages"]]
    return rows


def _relevant_documents(path: Path) -> dict[str, set[str]]:
    rows: dict[str, set[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            relevant = int(parts[3]) > 0
        except ValueError:
            continue
        if relevant:
            rows.setdefault(parts[0], set()).add(parts[2])
    return rows


def analyze_cell(
    label: str,
    exp_id: str,
    role: str,
    family: str,
    *,
    runs_root: Path,
    max_queries: int | None,
) -> CellResult:
    """Compute fixed-effect variance components for one explicit experiment."""
    trial = _latest_trial(runs_root / exp_id)
    if trial is None:
        raise FileNotFoundError(f"{label}: no run dir under {runs_root / exp_id}")
    config = yaml.safe_load((trial / "resolved_config.yaml").read_text(encoding="utf-8"))
    data = config.get("data", {})
    reranker = config.get("reranker", {})
    robustness = config.get("robustness", {})
    k_input = int(data["k_input"])
    batch_size = int(reranker.get("docs_per_score_forward", 20))
    seeds = list(robustness.get("seeds") or range(int(robustness.get("K", 10))))

    data_dir = runs_root.parent / "data" / Path(str(data["run_path"])).parent.name
    fixture = _fixture_order(data_dir / "fixture.jsonl")
    qrels = _relevant_documents(data_dir / "qrels.txt")
    per_query_root = trial / "psi" / "per_query_results"
    if not per_query_root.is_dir():
        raise FileNotFoundError(f"{label}: {per_query_root} missing; fetch retained query details")

    metrics = []
    skipped = 0
    query_dirs = sorted(path for path in per_query_root.iterdir() if path.is_dir())
    if max_queries is not None:
        query_dirs = query_dirs[:max_queries]
    for query_dir in query_dirs:
        aligned_path = query_dir / "aligned_scores.json"
        if not aligned_path.is_file():
            skipped += 1
            continue
        query_id = dirname_to_qid(query_dir.name)
        first_stage = fixture.get(query_id)
        aligned = json.loads(aligned_path.read_text(encoding="utf-8")).get("scores", {})
        if not first_stage or not aligned:
            skipped += 1
            continue
        document_rows = assemble_doc_rows(
            first_stage[:k_input],
            seeds,
            aligned,
            batch_size,
        )
        row = compute_query_metrics(
            query_id,
            document_rows,
            rel_pids=qrels.get(query_id),
        )
        if row is None:
            skipped += 1
            continue
        metrics.append(row)
    if not metrics:
        raise RuntimeError(f"{label}: no usable queries (skipped {skipped})")
    return CellResult(
        label=label,
        exp_id=exp_id,
        role=role,
        family=family,
        run_dir=trial,
        k_input=k_input,
        batch_size=batch_size,
        seeds=seeds,
        n_queries=len(metrics),
        n_queries_skipped=skipped,
        summary=summarize_per_query(metrics),
        per_query=per_query_to_dicts(metrics),
    )
