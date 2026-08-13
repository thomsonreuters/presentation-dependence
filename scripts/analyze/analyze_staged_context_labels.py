#!/usr/bin/env python3
"""Compare batched-teacher and pointwise-teacher silver labels.

The two JSONL inputs are streamed in lockstep. The reducer fails if their
query/document universes or ordering differ, computes all metrics on paired
queries, including judgment-conditioned label separation and judged-positive
record share, and writes a machine-readable artifact plus a compact Markdown
view. No model is run.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
from scipy.stats import kendalltau, rankdata


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "msmarco-train-selfdistill-seed42"
DEFAULT_OUT = ROOT / "build" / "reproduction" / "analysis" / "staged_context"
DEFAULT_REPORT = ROOT / "build" / "reproduction" / "reporting" / "staged_context"
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 0
SCORE_MIN = 0.0
SCORE_MAX = 3.0
ENDPOINT_EPSILON = 0.02
FLAT_SD_THRESHOLD = 0.01
HISTOGRAM_EDGES = np.linspace(SCORE_MIN, SCORE_MAX, 31)


@dataclass(frozen=True)
class Cohort:
    """One paired silver-label cohort."""

    name: str
    qids_path: Path
    batched_path: Path
    pointwise_path: Path


COHORTS = (
    Cohort(
        "train_29500",
        DATA / "qids_train_29500.txt",
        DATA / "silver_labels_qwen3_4b_k1_seed0_train_29500.jsonl",
        DATA / "silver_labels_qwen3_4b_pointwise_b1_train_29500.jsonl",
    ),
    Cohort(
        "heldout_500",
        DATA / "qids_heldout_500.txt",
        DATA / "silver_labels_qwen3_4b_k1_seed0_heldout_500.jsonl",
        DATA / "silver_labels_qwen3_4b_pointwise_b1_heldout_500.jsonl",
    ),
)


def load_qrels(path: Path) -> dict[str, dict[str, int]]:
    """Load TREC qrels as qid -> docid -> grade."""
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            parts = line.split()
            if len(parts) < 4:
                raise ValueError(f"{path}: malformed qrels line {line_no}")
            qid, _, docid, grade = parts[:4]
            qrels[qid][docid] = int(grade)
    return dict(qrels)


def _grouped_records(path: Path) -> Iterator[tuple[str, list[tuple[str, float]]]]:
    """Yield one ordered document-score vector per contiguous qid."""
    current_qid: str | None = None
    rows: list[tuple[str, float]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            record = json.loads(line)
            qid = str(record["query_id"])
            docid = str(record["doc_id"])
            score = float(record["score_continuous"])
            if not math.isfinite(score) or not SCORE_MIN <= score <= SCORE_MAX:
                raise ValueError(f"{path}:{line_no}: score outside [0,3]: {score}")
            if current_qid is None:
                current_qid = qid
            if qid != current_qid:
                if qid in seen:
                    raise ValueError(f"{path}: qid {qid!r} is not contiguous")
                seen.add(current_qid)
                yield current_qid, rows
                current_qid, rows = qid, []
            rows.append((docid, score))
    if current_qid is not None:
        if current_qid in seen:
            raise ValueError(f"{path}: qid {current_qid!r} is not contiguous")
        yield current_qid, rows


def iter_paired_queries(
    batched_path: Path, pointwise_path: Path
) -> Iterator[tuple[str, list[str], np.ndarray, np.ndarray]]:
    """Stream paired vectors while asserting exact query/document parity."""
    batched = _grouped_records(batched_path)
    pointwise = _grouped_records(pointwise_path)
    sentinel = object()
    while True:
        left = next(batched, sentinel)
        right = next(pointwise, sentinel)
        if left is sentinel or right is sentinel:
            if left is not right:
                raise ValueError("silver files contain different query counts")
            return
        left_qid, left_rows = left
        right_qid, right_rows = right
        if left_qid != right_qid:
            raise ValueError(f"query mismatch: {left_qid!r} != {right_qid!r}")
        left_docs = [docid for docid, _ in left_rows]
        right_docs = [docid for docid, _ in right_rows]
        if left_docs != right_docs:
            raise ValueError(f"{left_qid}: candidate order/universe differs")
        yield (
            left_qid,
            left_docs,
            np.asarray([score for _, score in left_rows], dtype=float),
            np.asarray([score for _, score in right_rows], dtype=float),
        )


def _dcg(grades: Sequence[int], k: int = 10) -> float:
    return sum((2.0**grade - 1.0) / math.log2(rank + 2.0) for rank, grade in enumerate(grades[:k]))


def ndcg_at_10(docids: Sequence[str], scores: np.ndarray, judged: dict[str, int]) -> float:
    """Match the repo's silver evaluator, including doc-id tie-breaking."""
    ideal = _dcg(sorted((grade for grade in judged.values() if grade > 0), reverse=True))
    if ideal <= 0:
        return float("nan")
    ranked = sorted(zip(docids, scores, strict=True), key=lambda item: (-item[1], item[0]))
    return _dcg([judged.get(docid, 0) for docid, _ in ranked]) / ideal


def positive_vs_candidate_auc(docids: Sequence[str], scores: np.ndarray, judged: dict[str, int]) -> float:
    """Accuracy for positive judged docs versus candidate docs treated as grade zero.

    MS MARCO train qrels contain positives but no explicit negatives, so the
    estimable comparison treats candidate documents without a positive judgment
    as grade zero. Ties receive half credit.
    """
    positives = [i for i, docid in enumerate(docids) if judged.get(docid, 0) > 0]
    zero_grade = [i for i, docid in enumerate(docids) if judged.get(docid, 0) == 0]
    if not positives or not zero_grade:
        return float("nan")
    wins = 0.0
    total = 0
    for positive in positives:
        comparisons = scores[positive] - scores[zero_grade]
        wins += float(np.count_nonzero(comparisons > 0))
        wins += 0.5 * float(np.count_nonzero(comparisons == 0))
        total += len(zero_grade)
    return wins / total


def graded_pair_accuracy(docids: Sequence[str], scores: np.ndarray, judged: dict[str, int]) -> float:
    """Accuracy over explicitly judged candidate pairs with unequal grades."""
    judged_indices = [index for index, docid in enumerate(docids) if docid in judged]
    wins = 0.0
    total = 0
    for offset, left in enumerate(judged_indices):
        for right in judged_indices[offset + 1 :]:
            left_grade = judged[docids[left]]
            right_grade = judged[docids[right]]
            if left_grade == right_grade:
                continue
            expected = 1.0 if left_grade > right_grade else -1.0
            observed = scores[left] - scores[right]
            wins += float(observed * expected > 0)
            wins += 0.5 * float(observed == 0)
            total += 1
    return wins / total if total else float("nan")


def paired_bootstrap(values: Sequence[float]) -> dict[str, Any]:
    """Percentile bootstrap for a paired mean without a giant sample matrix."""
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        raise ValueError("paired bootstrap requires at least one difference")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    estimates = np.empty(BOOTSTRAP_SAMPLES, dtype=float)
    batch = 256
    for start in range(0, BOOTSTRAP_SAMPLES, batch):
        stop = min(start + batch, BOOTSTRAP_SAMPLES)
        indices = rng.integers(0, array.size, size=(stop - start, array.size))
        estimates[start:stop] = array[indices].mean(axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return {
        "n_queries": int(array.size),
        "mean_delta_batched_minus_pointwise": float(array.mean()),
        "ci95": [float(low), float(high)],
        "samples": BOOTSTRAP_SAMPLES,
        "seed": BOOTSTRAP_SEED,
    }


class LabelAccumulator:
    """Streaming univariate and per-query summaries for one label source."""

    def __init__(self) -> None:
        """Initialize empty score and per-query accumulators."""
        self.n_scores = 0
        self.score_sum = 0.0
        self.score_sq_sum = 0.0
        self.near_min = 0
        self.near_max = 0
        self.near_integer = 0
        self.histogram = np.zeros(len(HISTOGRAM_EDGES) - 1, dtype=np.int64)
        self.rounded_counts: Counter[float] = Counter()
        self.query_sds: list[float] = []
        self.query_ranges: list[float] = []

    def add(self, scores: np.ndarray) -> None:
        """Add one query's score vector."""
        self.n_scores += scores.size
        self.score_sum += float(scores.sum())
        self.score_sq_sum += float(np.square(scores).sum())
        self.near_min += int(np.count_nonzero(scores <= SCORE_MIN + ENDPOINT_EPSILON))
        self.near_max += int(np.count_nonzero(scores >= SCORE_MAX - ENDPOINT_EPSILON))
        distances = np.min(np.abs(scores[:, None] - np.arange(4)[None, :]), axis=1)
        self.near_integer += int(np.count_nonzero(distances <= ENDPOINT_EPSILON))
        self.histogram += np.histogram(scores, bins=HISTOGRAM_EDGES)[0]
        self.rounded_counts.update(float(value) for value in np.round(scores, 6))
        self.query_sds.append(float(scores.std(ddof=0)))
        self.query_ranges.append(float(scores.max() - scores.min()))

    def summary(self) -> dict[str, Any]:
        """Return serializable distribution and spread summaries."""
        modal, modal_count = self.rounded_counts.most_common(1)[0]
        mean = self.score_sum / self.n_scores
        variance = max(0.0, self.score_sq_sum / self.n_scores - mean * mean)
        return {
            "n_scores": self.n_scores,
            "score_mean": mean,
            "score_sd": math.sqrt(variance),
            "within_query_sd_mean": float(np.mean(self.query_sds)),
            "within_query_sd_median": float(np.median(self.query_sds)),
            "queries_sd_below_0_01": int(np.count_nonzero(np.asarray(self.query_sds) < FLAT_SD_THRESHOLD)),
            "queries_sd_below_0_01_fraction": float(np.mean(np.asarray(self.query_sds) < FLAT_SD_THRESHOLD)),
            "within_query_range_mean": float(np.mean(self.query_ranges)),
            "fraction_within_0_02_of_zero": self.near_min / self.n_scores,
            "fraction_within_0_02_of_three": self.near_max / self.n_scores,
            "fraction_within_0_02_of_integer_grade": self.near_integer / self.n_scores,
            "modal_score_rounded_6dp": modal,
            "modal_score_tie_rate_rounded_6dp": modal_count / self.n_scores,
            "histogram": {
                "range": [SCORE_MIN, SCORE_MAX],
                "bin_edges": HISTOGRAM_EDGES.tolist(),
                "counts": self.histogram.tolist(),
                "fractions": (self.histogram / self.n_scores).tolist(),
            },
        }


class JudgmentConditionedAccumulator:
    """Score summaries split into judged-positive and all-other candidates."""

    def __init__(self) -> None:
        """Initialize empty positive and other-candidate moments."""
        self.positive_count = 0
        self.positive_sum = 0.0
        self.positive_sq_sum = 0.0
        self.positive_near_zero = 0
        self.other_count = 0
        self.other_sum = 0.0
        self.other_sq_sum = 0.0

    def add(self, docids: Sequence[str], scores: np.ndarray, judged: dict[str, int]) -> None:
        """Add one query after splitting candidates by positive judgment."""
        positive_mask = np.asarray([judged.get(docid, 0) > 0 for docid in docids], dtype=bool)
        positives = scores[positive_mask]
        others = scores[~positive_mask]
        self.positive_count += int(positives.size)
        self.positive_sum += float(positives.sum())
        self.positive_sq_sum += float(np.square(positives).sum())
        self.positive_near_zero += int(np.count_nonzero(positives <= SCORE_MIN + ENDPOINT_EPSILON))
        self.other_count += int(others.size)
        self.other_sum += float(others.sum())
        self.other_sq_sum += float(np.square(others).sum())

    def summary(self, all_record_count: int) -> dict[str, Any]:
        """Return grade separation and endpoint diagnostics."""
        if self.positive_count == 0 or self.other_count == 0:
            raise ValueError("judgment-conditioned summary requires both groups")
        positive_mean = self.positive_sum / self.positive_count
        other_mean = self.other_sum / self.other_count
        raw_separation = positive_mean - other_mean
        conditioned_count = self.positive_count + self.other_count
        conditioned_sum = self.positive_sum + self.other_sum
        conditioned_mean = conditioned_sum / conditioned_count
        conditioned_variance = max(
            0.0,
            (self.positive_sq_sum + self.other_sq_sum) / conditioned_count - conditioned_mean * conditioned_mean,
        )
        conditioned_sd = math.sqrt(conditioned_variance)
        return {
            "judged_positive_records": self.positive_count,
            "all_other_candidate_records": self.other_count,
            "all_records": all_record_count,
            "judged_positive_record_fraction": self.positive_count / all_record_count,
            "mean_grade_judged_positives": positive_mean,
            "mean_grade_all_other_candidates": other_mean,
            "raw_separation": raw_separation,
            "standardization": ("raw separation / SD of all teacher grades on qids with a positive qrel"),
            "conditioned_all_grade_sd": conditioned_sd,
            "standardized_separation": raw_separation / conditioned_sd,
            "judged_positives_within_0_02_of_zero": self.positive_near_zero,
            "judged_positives_within_0_02_of_zero_fraction": (self.positive_near_zero / self.positive_count),
        }


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    return {
        "n_queries": int(array.size),
        "mean": float(array.mean()),
        "sd": float(array.std(ddof=1)),
        "min": float(array.min()),
        "q10": float(np.quantile(array, 0.10)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "q75": float(np.quantile(array, 0.75)),
        "q90": float(np.quantile(array, 0.90)),
        "max": float(array.max()),
    }


def analyze_cohort(  # noqa: C901
    cohort: Cohort, qrels: dict[str, dict[str, int]]
) -> dict[str, Any]:
    """Analyze one paired train or held-out cohort."""
    expected_qids = {line.strip() for line in cohort.qids_path.read_text(encoding="utf-8").splitlines() if line.strip()}
    seen_qids: set[str] = set()
    labels = {"batched": LabelAccumulator(), "pointwise": LabelAccumulator()}
    conditioned = {
        "batched": JudgmentConditionedAccumulator(),
        "pointwise": JudgmentConditionedAccumulator(),
    }
    ndcg = {"batched": {}, "pointwise": {}}
    recall_conditioned = {"batched": {}, "pointwise": {}}
    pair_accuracy = {"batched": {}, "pointwise": {}}
    spearman: list[float] = []
    kendall: list[float] = []
    candidate_positive_queries = 0

    for qid, docids, batched_scores, pointwise_scores in iter_paired_queries(
        cohort.batched_path, cohort.pointwise_path
    ):
        if qid not in expected_qids:
            raise ValueError(f"{cohort.name}: unexpected qid {qid}")
        seen_qids.add(qid)
        labels["batched"].add(batched_scores)
        labels["pointwise"].add(pointwise_scores)

        rho = float(np.corrcoef(rankdata(batched_scores), rankdata(pointwise_scores))[0, 1])
        tau = float(kendalltau(batched_scores, pointwise_scores, variant="b").statistic)
        if math.isfinite(rho):
            spearman.append(rho)
        if math.isfinite(tau):
            kendall.append(tau)

        judged = qrels.get(qid)
        if not judged or not any(grade > 0 for grade in judged.values()):
            continue
        conditioned["batched"].add(docids, batched_scores, judged)
        conditioned["pointwise"].add(docids, pointwise_scores, judged)
        has_candidate_positive = any(judged.get(docid, 0) > 0 for docid in docids)
        candidate_positive_queries += int(has_candidate_positive)
        for name, scores in (
            ("batched", batched_scores),
            ("pointwise", pointwise_scores),
        ):
            value = ndcg_at_10(docids, scores, judged)
            ndcg[name][qid] = value
            if has_candidate_positive:
                recall_conditioned[name][qid] = value
            auc = positive_vs_candidate_auc(docids, scores, judged)
            if math.isfinite(auc):
                pair_accuracy[name][qid] = auc

    if seen_qids != expected_qids:
        missing = sorted(expected_qids - seen_qids)[:10]
        raise ValueError(f"{cohort.name}: missing {len(expected_qids - seen_qids)} qids: {missing}")

    def paired_summary(metric: dict[str, dict[str, float]]) -> dict[str, Any]:
        shared = sorted(set(metric["batched"]) & set(metric["pointwise"]))
        left = np.asarray([metric["batched"][qid] for qid in shared])
        right = np.asarray([metric["pointwise"][qid] for qid in shared])
        return {
            "batched_mean": float(left.mean()),
            "pointwise_mean": float(right.mean()),
            "paired": paired_bootstrap(left - right),
        }

    label_summaries = {name: accumulator.summary() for name, accumulator in labels.items()}
    conditioned_summaries = {
        name: accumulator.summary(
            all_record_count=label_summaries[name]["n_scores"],
        )
        for name, accumulator in conditioned.items()
    }
    positive_counts = {item["judged_positive_records"] for item in conditioned_summaries.values()}
    if len(positive_counts) != 1:
        raise ValueError(f"{cohort.name}: conditioned positive counts differ by source")

    return {
        "cohort": cohort.name,
        "query_set": str(cohort.qids_path.relative_to(ROOT)),
        "n_queries": len(seen_qids),
        "n_qids_with_positive_qrel": len(ndcg["batched"]),
        "n_qids_with_positive_in_candidate_set": candidate_positive_queries,
        "candidate_documents_per_query": sorted({labels["batched"].n_scores // len(seen_qids)}),
        "score_contract": {
            "field": "score_continuous",
            "range": [SCORE_MIN, SCORE_MAX],
            "endpoint_epsilon": ENDPOINT_EPSILON,
            "prompt_template": "grade_int_v1",
            "readout": "expected grade from grade-token probabilities",
        },
        "labels": label_summaries,
        "judgment_conditioned_label_separation": {
            "definition": (
                "On qids with a positive qrel, judged positives are candidate documents "
                "with qrel grade > 0; all other candidates include unjudged documents "
                "and explicit grade-zero documents. Near zero means score <= 0.02. "
                "The record fraction uses the full cohort as denominator."
            ),
            "judged_positive_records": next(iter(positive_counts)),
            "judged_positive_record_fraction": conditioned_summaries["batched"]["judged_positive_record_fraction"],
            "sources": conditioned_summaries,
        },
        "teacher_ranking_quality_ndcg_at_10": paired_summary(ndcg),
        "teacher_ranking_quality_ndcg_at_10_recall_conditioned": paired_summary(recall_conditioned),
        "judged_pair_accuracy": {
            "estimand": ("positive judged candidate versus candidate treated as grade zero; ties receive half credit"),
            **paired_summary(pair_accuracy),
        },
        "agreement": {
            "spearman": _distribution(spearman),
            "kendall_tau_b": _distribution(kendall),
        },
    }


def select_decision(primary: dict[str, Any]) -> dict[str, Any]:
    """Apply the pre-declared rule, including the unlisted reverse outcome."""
    pair = primary["judged_pair_accuracy"]["paired"]
    low, high = pair["ci95"]
    if low > 0:
        row = "information"
        reason = "Batched labels win judged-pair accuracy."
    elif high < 0:
        row = "reverse_information_target_geometry"
        reason = (
            "Pointwise labels win judged-pair accuracy but train the worse "
            "student; ordering information cannot explain the downstream penalty."
        )
    else:
        row = "calibration_or_mixed"
        reason = (
            "Judged-pair accuracy is comparable; distribution and agreement "
            "diagnostics determine whether the outcome is calibration or mixed."
        )
    return {"row": row, "reason": reason, "primary_metric": pair}


def analyze() -> dict[str, Any]:
    """Analyze both the primary training and held-out cohorts."""
    qrels = load_qrels(DATA / "qrels.txt")
    cohorts = [analyze_cohort(cohort, qrels) for cohort in COHORTS]
    return {
        "schema_version": 2,
        "qrels": {
            "path": str((DATA / "qrels.txt").relative_to(ROOT)),
            "source": "ir_datasets msmarco-passage/train",
            "interpretation": (
                "Sparse positive-only train judgments; unjudged candidate documents "
                "are treated as grade zero by nDCG and pair accuracy."
            ),
        },
        "bootstrap": {
            "method": "paired query percentile bootstrap",
            "samples": BOOTSTRAP_SAMPLES,
            "seed": BOOTSTRAP_SEED,
        },
        "cohorts": cohorts,
        "decision": select_decision(cohorts[0]),
    }


def _fmt_metric(block: dict[str, Any]) -> str:
    paired = block["paired"]
    return (
        f"batched {block['batched_mean']:.4f}, pointwise "
        f"{block['pointwise_mean']:.4f}, delta "
        f"{paired['mean_delta_batched_minus_pointwise']:+.4f}, 95% CI "
        f"[{paired['ci95'][0]:+.4f}, {paired['ci95'][1]:+.4f}], "
        f"n={paired['n_queries']}"
    )


def markdown(payload: dict[str, Any]) -> str:
    """Render the JSON payload as a compact audit table."""
    lines = [
        "# Staged-context label diagnostics",
        "",
        "Generated by `scripts/analyze/analyze_staged_context_labels.py`. Scores are expected "
        "grades on [0, 3], not probabilities on [0, 1].",
        "",
    ]
    for cohort in payload["cohorts"]:
        lines.extend(
            [
                f"## {cohort['cohort']}",
                "",
                f"- Query set: `{cohort['query_set']}` ({cohort['n_queries']} paired queries).",
                f"- Positive qrels: {cohort['n_qids_with_positive_qrel']}; positive in "
                f"top-100: {cohort['n_qids_with_positive_in_candidate_set']}.",
                f"- Silver nDCG@10: {_fmt_metric(cohort['teacher_ranking_quality_ndcg_at_10'])}.",
                "- Recall-conditioned silver nDCG@10: "
                f"{_fmt_metric(cohort['teacher_ranking_quality_ndcg_at_10_recall_conditioned'])}.",
                f"- Positive-vs-candidate pair accuracy: {_fmt_metric(cohort['judged_pair_accuracy'])}.",
                "",
                "| Source | Mean score | Mean/median query SD | SD<0.01 | Mean range | "
                "Near 0 | Near 3 | Near integer | Modal tie rate |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for source in ("batched", "pointwise"):
            item = cohort["labels"][source]
            lines.append(
                f"| {source} | {item['score_mean']:.4f} | "
                f"{item['within_query_sd_mean']:.4f} / "
                f"{item['within_query_sd_median']:.4f} | "
                f"{item['queries_sd_below_0_01_fraction']:.3%} | "
                f"{item['within_query_range_mean']:.4f} | "
                f"{item['fraction_within_0_02_of_zero']:.3%} | "
                f"{item['fraction_within_0_02_of_three']:.3%} | "
                f"{item['fraction_within_0_02_of_integer_grade']:.3%} | "
                f"{item['modal_score_tie_rate_rounded_6dp']:.3%} |"
            )
        separation = cohort["judgment_conditioned_label_separation"]
        lines.extend(
            [
                "",
                "### Judgment-conditioned label separation",
                "",
                f"Judged positives account for {separation['judged_positive_records']:,} "
                f"of {cohort['labels']['batched']['n_scores']:,} records "
                f"({separation['judged_positive_record_fraction']:.3%}).",
                "",
                "| Source | Positive mean | Other mean | Raw separation | Standardized separation | Positive ≈0 |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for source in ("batched", "pointwise"):
            item = separation["sources"][source]
            lines.append(
                f"| {source} | {item['mean_grade_judged_positives']:.4f} | "
                f"{item['mean_grade_all_other_candidates']:.4f} | "
                f"{item['raw_separation']:+.4f} | "
                f"{item['standardized_separation']:+.4f} | "
                f"{item['judged_positives_within_0_02_of_zero_fraction']:.3%} |"
            )
        agreement = cohort["agreement"]
        lines.extend(
            [
                "",
                "- Agreement: per-query Spearman mean/median "
                f"{agreement['spearman']['mean']:.4f}/{agreement['spearman']['median']:.4f}; "
                "Kendall tau-b mean/median "
                f"{agreement['kendall_tau_b']['mean']:.4f}/"
                f"{agreement['kendall_tau_b']['median']:.4f}.",
                "",
            ]
        )
    lines.extend(
        [
            "## Decision",
            "",
            f"**{payload['decision']['row']}**: {payload['decision']['reason']}",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    """Parse output-path arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> int:
    """Run the paired label analysis and write both artifacts."""
    args = parse_args()
    payload = analyze()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "label_diagnostics.json"
    md_path = args.report_dir / "label_diagnostics.md"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(markdown(payload), encoding="utf-8")
    print(payload["decision"]["row"])
    print(f"Wrote {json_path.relative_to(ROOT)}")
    print(f"Wrote {md_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
