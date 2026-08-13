"""Answer EM/F1 and stability metrics for the QA reader bridge.

Normalization and EM/F1 follow the official SQuAD and HotpotQA evaluation
(Rajpurkar et al.'s SQuAD ``evaluate-v2.0.py`` and Yang et al.'s HotpotQA
``hotpot_evaluate_v1.py``):
lowercase, strip punctuation, drop the articles {a, an, the}, and collapse
whitespace, then compute token-overlap F1 and exact string match. Every
reader-bridge result uses the normalization defined here.

Multi-gold handling: each question may carry a list of acceptable gold answers
(e.g. aliases). EM/F1 take the maximum over that list, matching SQuAD's
max-over-references convention.

The module performs no I/O and depends only on the standard library, so the
metrics run in laptop and CI tests without optional dependencies.

Upstream provenance and modification notice:

* SQuAD ``evaluate-v2.0.py`` at revision
  ``09eac9971f46889fa057ff2c870bf71092ba9d55`` is MIT, Copyright (c)
  2020 Pranav Rajpurkar.
* HotpotQA ``hotpot_evaluate_v1.py`` at revision
  ``fa3a36370899e1d85822de61e58c85ea19993154`` is Apache-2.0,
  Copyright 2018 Zhilin Yang, Peng Qi, Saizheng Zhang.
* This file is a modified, standard-library-only rewrite: normalization uses
  precompiled tables, F1 returns one scalar, multi-gold/empty handling is
  explicit, and the stability helpers are local.

Licence texts are at ``third_party_licenses/SQuAD-MIT.txt`` and
``third_party_licenses/Apache-2.0.txt``. See ``THIRD_PARTY_NOTICES.md``.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Iterable, Sequence

__all__ = [
    "normalize_answer",
    "exact_match",
    "f1_score",
    "metric_max_over_ground_truths",
    "best_em",
    "best_f1",
    "answer_flip_rate",
    "em_spread",
    "f1_spread",
    "majority_answer",
]

_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", flags=re.UNICODE)
_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_answer(text: str) -> str:
    """Apply SQuAD/HotpotQA lowercase, punctuation, article, and whitespace normalization."""
    s = str(text).lower()
    s = s.translate(_PUNCT_TABLE)
    s = _ARTICLES_RE.sub(" ", s)
    s = _WHITESPACE_RE.sub(" ", s).strip()
    return s


def _normalized_tokens(text: str) -> list[str]:
    norm = normalize_answer(text)
    return norm.split() if norm else []


def exact_match(prediction: str, ground_truth: str) -> float:
    """1.0 iff the normalized prediction equals the normalized gold, else 0.0."""
    return 1.0 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0.0


def f1_score(prediction: str, ground_truth: str) -> float:
    """Token-overlap F1 between normalized prediction and gold (HotpotQA v1 semantics).

    Mirrors the official scripts' special-casing of yes/no/empty answers: if either
    side is a single normalized special token ({yes, no, noanswer}) the two must
    match exactly to score, so F1 is never inflated by partial token overlap there.
    """
    pred_norm = normalize_answer(prediction)
    gold_norm = normalize_answer(ground_truth)

    special = {"yes", "no", "noanswer"}
    if pred_norm in special and pred_norm != gold_norm:
        return 0.0
    if gold_norm in special and pred_norm != gold_norm:
        return 0.0

    pred_tokens = pred_norm.split()
    gold_tokens = gold_norm.split()
    if not pred_tokens or not gold_tokens:
        # If either is empty after normalization, F1 is 1.0 only if both are empty.
        return 1.0 if pred_tokens == gold_tokens else 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)


def metric_max_over_ground_truths(
    metric_fn,
    prediction: str,
    ground_truths: Sequence[str],
) -> float:
    """Max of ``metric_fn(prediction, g)`` over the accepted gold answers."""
    golds = [g for g in ground_truths if g is not None]
    if not golds:
        return 0.0
    return max(metric_fn(prediction, g) for g in golds)


def best_em(prediction: str, ground_truths: Sequence[str]) -> float:
    """EM taking the max over multiple accepted gold answers."""
    return metric_max_over_ground_truths(exact_match, prediction, ground_truths)


def best_f1(prediction: str, ground_truths: Sequence[str]) -> float:
    """F1 taking the max over multiple accepted gold answers."""
    return metric_max_over_ground_truths(f1_score, prediction, ground_truths)


def majority_answer(predictions: Sequence[str]) -> str:
    """Most common normalized prediction (ties broken by first occurrence)."""
    preds = list(predictions)
    if not preds:
        return ""
    counts = Counter(normalize_answer(p) for p in preds)
    top_norm, _ = counts.most_common(1)[0]
    for p in preds:
        if normalize_answer(p) == top_norm:
            return p
    return preds[0]


def answer_flip_rate(predictions: Sequence[str]) -> float:
    """Fraction of predictions whose normalized form differs from the majority.

    This is the per-question downstream tau-PSI analogue: 0.0 means every
    permutation yielded the same answer, higher means the reader's answer is
    unstable across the scorer-input permutations.
    """
    preds = list(predictions)
    if len(preds) <= 1:
        return 0.0
    counts = Counter(normalize_answer(p) for p in preds)
    majority_count = counts.most_common(1)[0][1]
    return (len(preds) - majority_count) / len(preds)


def _spread(values: Sequence[float]) -> float:
    vals = [float(v) for v in values]
    if not vals:
        return 0.0
    return max(vals) - min(vals)


def em_spread(predictions: Sequence[str], ground_truths: Sequence[str]) -> float:
    """max-min of best-EM across a set of predictions for one question."""
    return _spread([best_em(p, ground_truths) for p in predictions])


def f1_spread(predictions: Sequence[str], ground_truths: Sequence[str]) -> float:
    """max-min of best-F1 across a set of predictions for one question."""
    return _spread([best_f1(p, ground_truths) for p in predictions])


def as_gold_list(answer: object) -> list[str]:
    """Coerce a fixture/sidecar gold-answer field into a list of accepted strings."""
    if answer is None:
        return []
    if isinstance(answer, str):
        return [answer]
    if isinstance(answer, Iterable):
        return [str(a) for a in answer if a is not None and str(a) != ""]
    return [str(answer)]
