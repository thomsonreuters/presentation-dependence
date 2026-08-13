"""Evaluate verdict stability from scorer rankings and a frozen reader."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from presentation_dependence.eval.score_log import read_score_log
from presentation_dependence.reader.engine import ReaderEngine
from presentation_dependence.reader.pipeline import load_fixture, load_gold_pids, ranked_topk_by_perm
from presentation_dependence.reader.verdict_eval import (
    VERDICT_LABELS,
    accuracy_spread,
    normalize_verdict,
    verdict_accuracy,
    verdict_flip_rate,
)
from presentation_dependence.reader.verdict_prompt import build_verdict_messages


def load_gold_verdicts(path: str | Path) -> dict[str, str]:
    """Load ``verdicts.jsonl`` as qid-to-label."""
    out: dict[str, str] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            label = str(row["label"]).strip().upper()
            if label in (*VERDICT_LABELS, "DISPUTED"):
                out[str(row["qid"])] = label
            else:
                raise ValueError(f"unknown verdict label {label!r} for qid={row.get('qid')!r}")
    return out


def load_qids(path: str | Path | None) -> set[str] | None:
    """Load an optional paired-qid manifest."""
    if path is None:
        return None
    return {line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()}


@dataclass
class VerdictItem:
    """One claim, scorer presentation, and reader top-k request."""

    qid: str
    perm_idx: int
    k: int
    ranked_pids: list[str]
    evidence_slot: int | None
    n_evidence_in_topk: int
    gold_verdict: str
    conversation: list[dict[str, str]] = field(default_factory=list)
    raw_output: str = ""
    prediction: str | None = None
    accuracy: float = 0.0


def build_verdict_items(
    rows: Sequence[Mapping],
    fixture: Mapping[str, dict],
    evidence_pids: Mapping[str, set[str]],
    gold_verdicts: Mapping[str, str],
    *,
    k_values: Sequence[int],
    allowed_qids: set[str] | None = None,
) -> list[VerdictItem]:
    """Build downstream verdict requests from scorer rankings."""
    ranked = ranked_topk_by_perm(rows)
    items: list[VerdictItem] = []
    for qid in sorted(ranked):
        if allowed_qids is not None and qid not in allowed_qids:
            continue
        gold = gold_verdicts.get(qid)
        if gold not in VERDICT_LABELS:
            continue
        if qid not in fixture:
            raise KeyError(f"qid {qid!r} present in rankings but missing from fixture")
        pid_to_text = fixture[qid]["pid_to_text"]
        evidence = set(evidence_pids.get(qid, set()))
        for perm_idx in sorted(ranked[qid]):
            for k in k_values:
                topk = ranked[qid][perm_idx][: int(k)]
                slots = [slot for slot, pid in enumerate(topk) if pid in evidence]
                passages = [{"pid": pid, "text": pid_to_text.get(pid, "")} for pid in topk]
                items.append(
                    VerdictItem(
                        qid=qid,
                        perm_idx=perm_idx,
                        k=int(k),
                        ranked_pids=list(topk),
                        evidence_slot=min(slots) if slots else None,
                        n_evidence_in_topk=len(slots),
                        gold_verdict=gold,
                        conversation=build_verdict_messages(fixture[qid]["query"], passages),
                    )
                )
    return items


def score_verdict_items(
    items: Sequence[VerdictItem],
    engine: ReaderEngine,
    *,
    batch_size: int = 0,
) -> list[VerdictItem]:
    """Generate and score a sequence of verdict requests."""
    conversations = [item.conversation for item in items]
    outputs: list[str] = []
    if batch_size > 0:
        for start in range(0, len(conversations), batch_size):
            outputs.extend(engine.generate_batch(conversations[start : start + batch_size]))
    else:
        outputs = engine.generate_batch(conversations)
    if len(outputs) != len(items):
        raise RuntimeError(f"engine returned {len(outputs)} outputs for {len(items)} items")
    for item, raw in zip(items, outputs, strict=True):
        item.raw_output = raw
        item.prediction = normalize_verdict(raw)
        item.accuracy = verdict_accuracy(item.prediction, item.gold_verdict)
    return list(items)


def aggregate_verdict_per_query(
    items: Sequence[VerdictItem],
    *,
    canonical_perm: int = 0,
) -> list[dict]:
    """Aggregate downstream stability and correctness per claim and k."""
    grouped: dict[tuple[str, int], list[VerdictItem]] = defaultdict(list)
    for item in items:
        grouped[(item.qid, item.k)].append(item)
    out: list[dict] = []
    for (qid, k), group in sorted(grouped.items()):
        ordered = sorted(group, key=lambda item: item.perm_idx)
        canonical = next((item for item in ordered if item.perm_idx == canonical_perm), ordered[0])
        predictions = [item.prediction for item in ordered]
        slots = [item.evidence_slot for item in ordered if item.evidence_slot is not None]
        out.append(
            {
                "qid": qid,
                "k": k,
                "gold_verdict": canonical.gold_verdict,
                "n_perms": len(ordered),
                "canonical_perm": canonical.perm_idx,
                "canonical_prediction": canonical.prediction,
                "canonical_accuracy": canonical.accuracy,
                "mean_accuracy": float(np.mean([item.accuracy for item in ordered])),
                "verdict_flip_rate": verdict_flip_rate(predictions),
                "accuracy_spread": accuracy_spread(predictions, canonical.gold_verdict),
                "invalid_rate": float(np.mean([prediction is None for prediction in predictions])),
                "evidence_in_topk_rate": float(np.mean([item.n_evidence_in_topk > 0 for item in ordered])),
                "evidence_slot_mean": float(np.mean(slots)) if slots else None,
                "evidence_slot_var": float(np.var(slots)) if slots else None,
            }
        )
    return out


def aggregate_verdict_corpus(per_query: Sequence[Mapping]) -> dict[int, dict]:
    """Average per-claim verdict metrics for each k."""
    grouped: dict[int, list[Mapping]] = defaultdict(list)
    for row in per_query:
        grouped[int(row["k"])].append(row)

    def mean(rows: Sequence[Mapping], key: str) -> float | None:
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        return float(np.mean(values)) if values else None

    out: dict[int, dict] = {}
    for k, rows in sorted(grouped.items()):
        out[k] = {
            "n_queries": len(rows),
            "label_counts": dict(sorted(Counter(str(row["gold_verdict"]) for row in rows).items())),
            "canonical_accuracy": mean(rows, "canonical_accuracy"),
            "mean_accuracy": mean(rows, "mean_accuracy"),
            "mean_verdict_flip_rate": mean(rows, "verdict_flip_rate"),
            "mean_accuracy_spread": mean(rows, "accuracy_spread"),
            "mean_invalid_rate": mean(rows, "invalid_rate"),
            "evidence_in_topk_rate": mean(rows, "evidence_in_topk_rate"),
            "evidence_slot_mean": mean(rows, "evidence_slot_mean"),
            "evidence_slot_var": mean(rows, "evidence_slot_var"),
        }
    return out


def run_verdict_bridge(
    *,
    score_log: str | Path,
    fixture_path: str | Path,
    verdicts_path: str | Path,
    qrels_path: str | Path,
    engine: ReaderEngine,
    k_values: Sequence[int],
    qids_path: str | Path | None = None,
    canonical_perm: int = 0,
    batch_size: int = 0,
) -> dict:
    """Run the scorer-to-verdict bridge end to end."""
    rows = read_score_log(score_log)
    fixture = load_fixture(fixture_path)
    verdicts = load_gold_verdicts(verdicts_path)
    qids = load_qids(qids_path)
    evidence = load_gold_pids(qrels_path)
    items = build_verdict_items(
        rows,
        fixture,
        evidence,
        verdicts,
        k_values=k_values,
        allowed_qids=qids,
    )
    score_verdict_items(items, engine, batch_size=batch_size)
    per_query = aggregate_verdict_per_query(items, canonical_perm=canonical_perm)
    included_qids = {row["qid"] for row in per_query}
    eligible = {qid for qid, label in verdicts.items() if label in VERDICT_LABELS}
    requested = eligible if qids is None else eligible & qids
    return {
        "items": items,
        "per_query": per_query,
        "corpus": aggregate_verdict_corpus(per_query),
        "meta": {
            "reader": engine.name,
            "k_values": [int(k) for k in k_values],
            "canonical_perm": int(canonical_perm),
            "n_scorer_input_perms": len({item.perm_idx for item in items}),
            "n_queries": len(included_qids),
            "n_requested_queries": len(requested),
            "n_missing_from_rankings": len(requested - included_qids),
            "excluded_label_counts": dict(
                sorted(Counter(label for label in verdicts.values() if label not in VERDICT_LABELS).items())
            ),
        },
    }
