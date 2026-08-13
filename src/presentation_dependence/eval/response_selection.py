"""Reusable response-selection reductions over scorer presentations."""

from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


def load_fixture_pids(path: Path) -> dict[str, list[str]]:
    """Load each response-quality query's candidate IDs."""
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            result[str(row["qid"])] = [str(passage["pid"]) for passage in row["passages"]]
    return result


def _modal_change(values: Sequence[Any]) -> float:
    modal_count = Counter(values).most_common(1)[0][1]
    return 1.0 - modal_count / len(values)


def response_selection_metrics(
    aligned: Mapping[str, Mapping[str, Sequence[float]]],
    fixture: Mapping[str, Sequence[str]],
    qrels: Mapping[str, Mapping[str, int]],
) -> dict[str, float | int]:
    """Compute selection, pair-flip, quality, and benchmark-range metrics."""
    per_query = []
    quality_by_presentation: list[list[float]] = []
    for qid, pids in fixture.items():
        scores = aligned.get(qid)
        if not scores:
            continue
        scored = {pid: scores[pid] for pid in pids if pid in scores and scores[pid]}
        if not scored:
            continue
        presentations = min(len(values) for values in scored.values())
        ordered = [
            sorted(
                ((float(values[presentation]), pid) for pid, values in scored.items()),
                key=lambda item: (-item[0], item[1]),
            )
            for presentation in range(presentations)
        ]
        picks = [ranking[0][1] for ranking in ordered]
        pairs = [(ranking[0][1], ranking[-1][1]) for ranking in ordered]
        rels = qrels.get(qid, {})
        max_rel = max(rels.values()) if rels else 0
        grades = [int(rels.get(pid, 0)) for pid in picks]
        normalized = [float(grade / max_rel) if max_rel > 0 else 0.0 for grade in grades]
        while len(quality_by_presentation) < len(normalized):
            quality_by_presentation.append([])
        for presentation, quality in enumerate(normalized):
            quality_by_presentation[presentation].append(quality)
        per_query.append(
            {
                "selection_flip_rate": _modal_change(picks),
                "selected_quality_flip_rate": _modal_change(grades),
                "preference_pair_flip_rate": _modal_change(pairs),
                "mean_selected_quality": statistics.fmean(normalized),
                "canonical_selected_quality": normalized[0],
            }
        )
    if not per_query:
        raise ValueError("No aligned response scores could be reduced")
    presentation_means = [statistics.fmean(values) for values in quality_by_presentation if values]
    return {
        "n_queries": len(per_query),
        "benchmark_score_range": (max(presentation_means) - min(presentation_means) if presentation_means else 0.0),
        **{metric: statistics.fmean(row[metric] for row in per_query) for metric in per_query[0]},
    }
