"""Load per-document scores aligned across scorer presentations."""

from __future__ import annotations

import json
from pathlib import Path

from presentation_dependence.eval.score_log import read_score_log
from presentation_dependence.utils.trec import dirname_to_qid


def _from_json_tree(run_dir: Path) -> dict[str, dict[str, list[float]]]:
    per_query = run_dir / "psi/per_query_results"
    result = {}
    if not per_query.is_dir():
        return result
    for query_dir in sorted(path for path in per_query.iterdir() if path.is_dir()):
        path = query_dir / "aligned_scores.json"
        if not path.is_file():
            continue
        scores = json.loads(path.read_text(encoding="utf-8")).get("scores")
        if isinstance(scores, dict) and scores:
            result[dirname_to_qid(query_dir.name)] = {
                str(pid): [float(value) for value in values] for pid, values in scores.items()
            }
    return result


def _from_parquet(run_dir: Path) -> dict[str, dict[str, list[float]]]:
    score_log = run_dir / "psi/beta_gamma_scores.parquet"
    if not score_log.is_file():
        return {}
    sparse: dict[str, dict[str, dict[int, float]]] = {}
    for row in read_score_log(score_log):
        sparse.setdefault(str(row["query_id"]), {}).setdefault(str(row["doc_id"]), {})[int(row["perm_idx"])] = float(
            row["score"]
        )
    result = {}
    for qid, documents in sparse.items():
        presentations = sorted(set.intersection(*({*scores} for scores in documents.values())))
        if presentations:
            result[qid] = {
                pid: [scores[presentation] for presentation in presentations] for pid, scores in documents.items()
            }
    return result


def _from_trec_tree(run_dir: Path) -> dict[str, dict[str, list[float]]]:
    paths = sorted((run_dir / "per_query_results").glob("*/trec_results_deduplicated.txt"))
    if not paths:
        paths = sorted((run_dir / "per_query_results").glob("*/trec_results_raw.txt"))
    result: dict[str, dict[str, list[float]]] = {}
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) >= 6:
                result.setdefault(parts[0], {})[parts[2]] = [float(parts[4])]
    return result


def load_aligned_scores(
    run_dir: Path,
) -> dict[str, dict[str, list[float]]]:
    """Load aligned JSON, PSI parquet, or a TREC score tree."""
    return _from_json_tree(run_dir) or _from_parquet(run_dir) or _from_trec_tree(run_dir)
