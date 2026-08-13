"""Evaluate reader answers from the stored scorer rankings.

The pipeline reconstructs each scorer ranking from
``psi/beta_gamma_scores.parquet``. It uses the ``query_id``, ``perm_idx``,
``doc_id``, ``position``, and ``score`` columns written for PSI
``random_shuffle`` presentations. It joins passage text from the QA
``fixture.jsonl``, passes the ranked top-k passages to a frozen reader in scorer
order, and evaluates the answer against ``answers.jsonl``.

Outputs include:

- an answer and EM/F1 for each query, scorer-input permutation, and k;
- canonical EM/F1, answer flip rate, and EM/F1 spread for each query and k; and
- the gold passage's reader slot for the R4 scorer-by-reader-position
  decomposition.

Answer flip rate is the downstream analogue of τ-PSI.
Data preparation is independent of generation and can be tested offline.
Generation requires a ``ReaderEngine``.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from presentation_dependence.eval.score_log import read_score_log
from presentation_dependence.reader.answer_eval import (
    answer_flip_rate,
    best_em,
    best_f1,
    em_spread,
    f1_spread,
)
from presentation_dependence.reader.engine import ReaderEngine
from presentation_dependence.reader.prompt import build_reader_messages, extract_answer

__all__ = [
    "load_fixture",
    "load_gold_answers",
    "load_gold_pids",
    "ranked_topk_by_perm",
    "ReaderItem",
    "build_reader_items",
    "score_items",
    "aggregate_per_query",
    "aggregate_corpus",
    "run_reader_bridge",
]


def load_fixture(fixture_path: str | Path) -> dict[str, dict]:
    """Load ``fixture.jsonl`` as ``{qid: {query, pid_to_text}}``."""
    out: dict[str, dict] = {}
    with Path(fixture_path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            qid = str(rec["qid"])
            pid_to_text = {str(p["pid"]): str(p.get("text", "")) for p in rec.get("passages", [])}
            out[qid] = {"query": str(rec.get("query", "")), "pid_to_text": pid_to_text}
    return out


def load_gold_answers(answers_path: str | Path) -> dict[str, list[str]]:
    """Load ``answers.jsonl`` as ``{qid: [answers]}``."""
    out: dict[str, list[str]] = {}
    with Path(answers_path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            out[str(rec["qid"])] = [str(a) for a in rec.get("answers", [])]
    return out


def load_gold_pids(qrels_path: str | Path) -> dict[str, set[str]]:
    """Load positive support qrels as ``{qid: {gold_pid, ...}}``."""
    gold: dict[str, set[str]] = defaultdict(set)
    for line in Path(qrels_path).read_text().splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[3] not in ("0", "-1"):
            gold[parts[0]].add(parts[2])
    return dict(gold)


def ranked_topk_by_perm(rows: Sequence[Mapping]) -> dict[str, dict[int, list[str]]]:
    """Reconstruct the scorer's ranked doc order per ``(query, perm_idx)``.

    For each ``(query_id, perm_idx)`` group, sort documents by descending score (ties
    broken by input ``position`` then ``doc_id`` for determinism) and return the
    ranked ``doc_id`` list. This is the scorer output passed to the reader.
    """
    grouped: dict[tuple[str, int], list[tuple[float, int, str]]] = defaultdict(list)
    for row in rows:
        qid = str(row["query_id"])
        perm = int(row["perm_idx"])
        grouped[(qid, perm)].append((float(row["score"]), int(row["position"]), str(row["doc_id"])))

    out: dict[str, dict[int, list[str]]] = defaultdict(dict)
    for (qid, perm), triples in grouped.items():
        triples.sort(key=lambda t: (-t[0], t[1], t[2]))
        out[qid][perm] = [doc_id for _, _, doc_id in triples]
    return {qid: dict(perms) for qid, perms in out.items()}


@dataclass
class ReaderItem:
    """One reader generation request for a ``(query, perm, k)`` triple."""

    qid: str
    perm_idx: int
    k: int
    ranked_pids: list[str]
    gold_slot: int | None  # 0-based reader slot of the best-placed gold pid in top-k, else None
    n_gold_in_topk: int
    conversation: list[dict] = field(default_factory=list)
    raw_output: str = ""
    prediction: str = ""
    em: float = 0.0
    f1: float = 0.0


def build_reader_items(
    rows: Sequence[Mapping],
    fixture: Mapping[str, dict],
    gold_pids: Mapping[str, set[str]],
    *,
    k_values: Sequence[int],
    cot: bool = False,
) -> list[ReaderItem]:
    """Build reader requests from each stored query, permutation, and k."""
    ranked = ranked_topk_by_perm(rows)
    items: list[ReaderItem] = []
    for qid in sorted(ranked):
        if qid not in fixture:
            raise KeyError(f"qid {qid!r} present in score log but missing from fixture")
        pid_to_text = fixture[qid]["pid_to_text"]
        query = fixture[qid]["query"]
        gold = set(gold_pids.get(qid, set()))
        for perm_idx in sorted(ranked[qid]):
            full_order = ranked[qid][perm_idx]
            for k in k_values:
                topk = full_order[:k]
                passages = [{"pid": pid, "text": pid_to_text.get(pid, "")} for pid in topk]
                gold_slots = [slot for slot, pid in enumerate(topk) if pid in gold]
                items.append(
                    ReaderItem(
                        qid=qid,
                        perm_idx=perm_idx,
                        k=k,
                        ranked_pids=list(topk),
                        gold_slot=min(gold_slots) if gold_slots else None,
                        n_gold_in_topk=len(gold_slots),
                        conversation=build_reader_messages(query, passages, cot=cot),
                    )
                )
    return items


def score_items(
    items: Sequence[ReaderItem],
    engine: ReaderEngine,
    gold_answers: Mapping[str, list[str]],
    *,
    cot: bool = False,
    batch_size: int = 0,
) -> list[ReaderItem]:
    """Generate answers for every item and fill EM/F1 in place."""
    conversations = [item.conversation for item in items]
    outputs: list[str] = []
    if batch_size and batch_size > 0:
        for start in range(0, len(conversations), batch_size):
            outputs.extend(engine.generate_batch(conversations[start : start + batch_size]))
    else:
        outputs = engine.generate_batch(conversations)
    if len(outputs) != len(items):
        raise RuntimeError(f"engine returned {len(outputs)} outputs for {len(items)} items")

    for item, raw in zip(items, outputs):
        golds = gold_answers.get(item.qid, [])
        pred = extract_answer(raw, cot=cot)
        item.raw_output = raw
        item.prediction = pred
        item.em = best_em(pred, golds)
        item.f1 = best_f1(pred, golds)
    return list(items)


def aggregate_per_query(
    items: Sequence[ReaderItem],
    gold_answers: Mapping[str, list[str]],
    *,
    canonical_perm: int = 0,
) -> list[dict]:
    """Return per-(query, k) downstream value and stability rows."""
    by_key: dict[tuple[str, int], list[ReaderItem]] = defaultdict(list)
    for item in items:
        by_key[(item.qid, item.k)].append(item)

    rows: list[dict] = []
    for (qid, k), group in sorted(by_key.items()):
        group_sorted = sorted(group, key=lambda it: it.perm_idx)
        preds = [it.prediction for it in group_sorted]
        golds = gold_answers.get(qid, [])
        canonical = next((it for it in group_sorted if it.perm_idx == canonical_perm), group_sorted[0])
        gold_slots = [it.gold_slot for it in group_sorted if it.gold_slot is not None]
        rows.append(
            {
                "qid": qid,
                "k": k,
                "n_perms": len(group_sorted),
                "canonical_perm": canonical.perm_idx,
                "canonical_em": canonical.em,
                "canonical_f1": canonical.f1,
                "mean_em": float(np.mean([it.em for it in group_sorted])),
                "mean_f1": float(np.mean([it.f1 for it in group_sorted])),
                "answer_flip_rate": answer_flip_rate(preds),
                "em_spread": em_spread(preds, golds),
                "f1_spread": f1_spread(preds, golds),
                "gold_in_topk_rate": float(np.mean([1.0 if it.n_gold_in_topk > 0 else 0.0 for it in group_sorted])),
                "gold_slot_mean": float(np.mean(gold_slots)) if gold_slots else None,
                "gold_slot_var": float(np.var(gold_slots)) if gold_slots else None,
            }
        )
    return rows


def aggregate_corpus(per_query: Sequence[Mapping]) -> dict[int, dict]:
    """Corpus-level means per k from the per-query rows."""
    by_k: dict[int, list[Mapping]] = defaultdict(list)
    for row in per_query:
        by_k[int(row["k"])].append(row)

    def _mean(rows: Sequence[Mapping], key: str) -> float | None:
        vals = [float(r[key]) for r in rows if r.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    out: dict[int, dict] = {}
    for k, rows in sorted(by_k.items()):
        out[k] = {
            "n_queries": len(rows),
            "canonical_em": _mean(rows, "canonical_em"),
            "canonical_f1": _mean(rows, "canonical_f1"),
            "mean_em": _mean(rows, "mean_em"),
            "mean_f1": _mean(rows, "mean_f1"),
            "mean_answer_flip_rate": _mean(rows, "answer_flip_rate"),
            "mean_em_spread": _mean(rows, "em_spread"),
            "mean_f1_spread": _mean(rows, "f1_spread"),
            "gold_in_topk_rate": _mean(rows, "gold_in_topk_rate"),
            "gold_slot_mean": _mean(rows, "gold_slot_mean"),
            "gold_slot_var": _mean(rows, "gold_slot_var"),
        }
    return out


def run_reader_bridge(
    *,
    score_log: str | Path,
    fixture_path: str | Path,
    answers_path: str | Path,
    qrels_path: str | Path,
    engine: ReaderEngine,
    k_values: Sequence[int],
    cot: bool = False,
    canonical_perm: int = 0,
    batch_size: int = 0,
) -> dict:
    """Read scorer rankings, generate answers, and aggregate reader metrics."""
    rows = read_score_log(score_log)
    fixture = load_fixture(fixture_path)
    gold_answers = load_gold_answers(answers_path)
    gold_pids = load_gold_pids(qrels_path)

    items = build_reader_items(rows, fixture, gold_pids, k_values=k_values, cot=cot)
    score_items(items, engine, gold_answers, cot=cot, batch_size=batch_size)
    per_query = aggregate_per_query(items, gold_answers, canonical_perm=canonical_perm)
    corpus = aggregate_corpus(per_query)

    scorer_recipes = sorted({str(r.get("recipe", "")) for r in rows})
    scorer_checkpoints = sorted({str(r.get("checkpoint", "")) for r in rows})
    n_perms = len({int(r["perm_idx"]) for r in rows})
    return {
        "items": items,
        "per_query": per_query,
        "corpus": corpus,
        "meta": {
            "reader": engine.name,
            "cot": cot,
            "k_values": list(k_values),
            "canonical_perm": canonical_perm,
            "scorer_recipes": scorer_recipes,
            "scorer_checkpoints": scorer_checkpoints,
            "n_queries": len(per_query) // max(1, len(k_values)),
            "n_scorer_input_perms": n_perms,
        },
    }
