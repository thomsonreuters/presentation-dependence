"""Evaluation diagnostics for self-distill student checkpoints.

The diagnostics measure agreement with the silver teacher and score stability
under document permutations.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from typing import Callable

from presentation_dependence.self_distill.student_data import GroupedSilverExample, permutation_groups


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys):
        raise ValueError("pearson inputs must have same length")
    if len(xs) < 2:
        return float("nan")
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (dx * dy) if dx and dy else float("nan")


def rank_order(values: list[float]) -> list[int]:
    """Return 0-best ranks for ``values``."""
    order = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
    ranks = [0] * len(values)
    for rank, idx in enumerate(order):
        ranks[idx] = rank
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys):
        raise ValueError("spearman inputs must have same length")
    if len(xs) < 2:
        return float("nan")
    rx = rank_order(xs)
    ry = rank_order(ys)
    return pearson([float(x) for x in rx], [float(y) for y in ry])


def ndcg_at_k(doc_ids_ranked: list[str], qrels: dict[str, int], k: int = 10) -> float | None:
    if not qrels:
        return None
    rels = [qrels.get(d, 0) for d in doc_ids_ranked[:k]]
    dcg = sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(rels))
    ideal = sorted(qrels.values(), reverse=True)[:k]
    idcg = sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else None


def silver_prediction_diagnostics(
    examples: list[GroupedSilverExample],
    predicted_scores: dict[tuple[str, str], float],
) -> dict[str, float]:
    """Compare student predictions to teacher silver labels and qrels."""
    y_true: list[float] = []
    y_pred: list[float] = []
    ndcgs: list[float] = []
    by_q: dict[str, list[tuple[float, str]]] = defaultdict(list)

    for ex in examples:
        for doc_id, teacher_score in zip(ex.doc_ids, ex.teacher_scores_mean, strict=True):
            key = (ex.query_id, doc_id)
            if key not in predicted_scores:
                raise ValueError(f"missing prediction for qid={ex.query_id} doc_id={doc_id}")
            pred = float(predicted_scores[key])
            y_true.append(float(teacher_score))
            y_pred.append(pred)
            by_q[ex.query_id].append((pred, doc_id))
        if ex.qrels:
            ranked = [doc for _score, doc in sorted(by_q[ex.query_id], reverse=True)]
            ndcg = ndcg_at_k(ranked, ex.qrels, k=10)
            if ndcg is not None:
                ndcgs.append(ndcg)

    mse = sum((a - b) ** 2 for a, b in zip(y_true, y_pred, strict=True)) / len(y_true)
    return {
        "mse_to_silver": mse,
        "pearson_to_silver": pearson(y_true, y_pred),
        "spearman_to_silver": spearman(y_true, y_pred),
        "teacher_mean": statistics.mean(y_true),
        "student_mean": statistics.mean(y_pred),
        "teacher_stdev": statistics.stdev(y_true) if len(y_true) > 1 else 0.0,
        "student_stdev": statistics.stdev(y_pred) if len(y_pred) > 1 else 0.0,
        "qrels_ndcg_cut_10": statistics.mean(ndcgs) if ndcgs else float("nan"),
        "n_predictions": float(len(y_pred)),
    }


def permutation_invariance_diagnostics(
    examples: list[GroupedSilverExample],
    scorer: Callable[[GroupedSilverExample, list[str]], dict[str, float]],
    *,
    seeds: list[int],
) -> dict[str, float]:
    """Measure prediction stability under document permutations.

    ``scorer`` receives the original group plus a permuted doc-id order and
    returns ``{doc_id: score}`` in original doc-id space.
    """
    per_doc_scores: dict[tuple[str, str], list[float]] = defaultdict(list)
    per_q_spearman: list[float] = []
    worst_ndcgs: list[float] = []

    for ex in examples:
        teacher_order_scores = dict(zip(ex.doc_ids, ex.teacher_scores_mean, strict=True))
        q_ndcgs: list[float] = []
        for pg in permutation_groups([ex], seeds=seeds):
            pred = scorer(ex, pg.doc_ids_permuted)
            if set(pred) != set(ex.doc_ids):
                raise ValueError(f"scorer returned doc set mismatch for qid={ex.query_id}")
            for doc_id, score in pred.items():
                per_doc_scores[(ex.query_id, doc_id)].append(float(score))
            preds_in_doc_order = [pred[d] for d in ex.doc_ids]
            teacher_in_doc_order = [teacher_order_scores[d] for d in ex.doc_ids]
            per_q_spearman.append(spearman(teacher_in_doc_order, preds_in_doc_order))
            if ex.qrels:
                ranked = [doc for doc, _score in sorted(pred.items(), key=lambda kv: kv[1], reverse=True)]
                nd = ndcg_at_k(ranked, ex.qrels, k=10)
                if nd is not None:
                    q_ndcgs.append(nd)
        if q_ndcgs:
            worst_ndcgs.append(min(q_ndcgs))

    variances = [statistics.variance(v) for v in per_doc_scores.values() if len(v) > 1]
    return {
        "permutation_score_variance_mean": statistics.mean(variances) if variances else 0.0,
        "permutation_score_variance_median": statistics.median(variances) if variances else 0.0,
        "permutation_spearman_mean": statistics.mean(per_q_spearman) if per_q_spearman else float("nan"),
        "worst_permutation_ndcg_cut_10": statistics.mean(worst_ndcgs) if worst_ndcgs else float("nan"),
        "n_permutation_predictions": float(sum(len(v) for v in per_doc_scores.values())),
    }
