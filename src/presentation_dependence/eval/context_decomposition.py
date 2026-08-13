"""Decomposition of the fixed-weight serving-width effect.

The runner scores one fixed checkpoint under four prompt constructions:

* A1: target alone, one grade line;
* A2: target with its real B-1 companions;
* A3: target with length-matched documents from other queries;
* A4: target alone, but at the same slot in a B-line cumulative skeleton.

Canonical first-stage order and the published Fisher-Yates seeds 0..9 share
the same target slots in A2-A4. A3 intentionally makes the target the only
query-matched document; it estimates the value of informative alternatives,
not the value of any arbitrary alternatives.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

from presentation_dependence.eval.experiment_manager import _apply_prompt_overrides
from presentation_dependence.eval.psi import ndcg_at_k
from presentation_dependence.eval.runner_setup import (
    instantiate_dataloader,
    instantiate_reranker,
    load_qrels,
    partial_run_error,
    write_resolved_run_config,
)
from presentation_dependence.rerankers.grade_rubrics import normalize_doc_text
from presentation_dependence.utils.progress import ProgressTracker, progress_config
from presentation_dependence.utils.setup_logging import setup_logging
from presentation_dependence.utils.trec import qid_to_dirname


ARM_NAMES = ("A1", "A2", "A3", "A4")
SCREEN_ARM_NAMES = ("A1", "A4")
DECOMPOSITION_TERMS = {
    "published_width": ("A2", "A1"),
    "semantic_real_vs_unrelated": ("A2", "A3"),
    "companion_length_and_position": ("A3", "A4"),
    "cumulative_skeleton": ("A4", "A1"),
}


def stable_seed(*parts: object) -> int:
    """Return a process- and Python-version-independent 64-bit seed."""
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def published_presentations(
    passages: list[dict],
    seeds: Iterable[int],
) -> list[tuple[str, list[dict]]]:
    """Canonical order plus the PSI harness' seeded Fisher-Yates orders."""
    out = [("canonical", list(passages))]
    for seed in seeds:
        shuffled = list(passages)
        random.Random(int(seed)).shuffle(shuffled)
        out.append((f"random_s{int(seed)}", shuffled))
    return out


def shard_qids(
    qids: list[str],
    *,
    shard_count: int,
    shard_index: int,
) -> list[str]:
    """Deterministically partition whole queries across independent jobs."""
    if shard_count < 1:
        raise ValueError("shard_count must be positive")
    if not 0 <= shard_index < shard_count:
        raise ValueError(f"shard_index must be in [0, {shard_count}), got {shard_index}")
    return list(qids[shard_index::shard_count])


def rank_ids_from_scores(
    pids_first_stage: list[str],
    scores: dict[str, float],
) -> list[str]:
    """Rank by descending score with first-stage order as the tie break."""
    first_stage_rank = {pid: rank for rank, pid in enumerate(pids_first_stage)}
    return sorted(
        pids_first_stage,
        key=lambda pid: (-float(scores[pid]), first_stage_rank[pid]),
    )


def score_effect_summary(
    left: dict[str, float],
    right: dict[str, float],
) -> dict[str, float]:
    """Signed and absolute per-document score difference."""
    if set(left) != set(right):
        raise ValueError("score effects require identical document IDs")
    deltas = [float(left[pid]) - float(right[pid]) for pid in left]
    return {
        "mean": float(np.mean(deltas)),
        "mean_abs": float(np.mean(np.abs(deltas))),
    }


def decompose_values(values: dict[str, float]) -> dict[str, float]:
    """Apply the pre-registered four-term decomposition to scalar arm values."""
    out = {name: float(values[left]) - float(values[right]) for name, (left, right) in DECOMPOSITION_TERMS.items()}
    out["closure_residual"] = out["published_width"] - (
        out["semantic_real_vs_unrelated"] + out["companion_length_and_position"] + out["cumulative_skeleton"]
    )
    return out


def scaffold_reading_from_arms(values: dict[str, float]) -> dict[str, float | None]:
    """Compute the scaffold share and scaffold-corrected width effect."""
    decomposition = decompose_values(values)
    width = decomposition["published_width"]
    scaffold_total = decomposition["cumulative_skeleton"]
    return {
        **decomposition,
        "scaffold_total": scaffold_total,
        "scaffold_share": scaffold_total / width if width != 0.0 else None,
        # The pre-registered correction subtracts the cumulative skeleton.
        "scaffold_corrected_effect": (width - decomposition["cumulative_skeleton"]),
    }


@dataclass(frozen=True)
class Donor:
    """One candidate document available as an unrelated A3 companion."""

    query_id: str
    pid: str
    passage: dict
    matched_chars: int


class LengthMatchedDonorIndex:
    """Nearest-character-length donor lookup with deterministic tie-breaking."""

    def __init__(self, query_pools: dict[str, list[dict]], *, max_doc_chars: int):
        """Index every donor by normalized, truncated character length."""
        by_length: dict[int, list[Donor]] = {}
        for qid, passages in query_pools.items():
            for passage in passages:
                text = normalize_doc_text(
                    str(passage.get("text", "")),
                    max_doc_chars=max_doc_chars,
                )
                donor = Donor(
                    query_id=str(qid),
                    pid=str(passage["pid"]),
                    passage=passage,
                    matched_chars=len(text),
                )
                by_length.setdefault(donor.matched_chars, []).append(donor)
        if not by_length:
            raise ValueError("cannot build a donor index from an empty query pool")
        for length, donors in by_length.items():
            by_length[length] = sorted(
                donors,
                key=lambda donor: (donor.query_id, donor.pid),
            )
        self.by_length = by_length
        self.lengths = sorted(by_length)

    def choose(
        self,
        *,
        desired_chars: int,
        target_qid: str,
        excluded_pids: set[str],
        used_pids: set[str],
        namespace: str,
    ) -> Donor:
        """Choose the nearest eligible length, varying ties by stable hash."""
        insertion = bisect.bisect_left(self.lengths, int(desired_chars))
        left = insertion - 1
        right = insertion
        while left >= 0 or right < len(self.lengths):
            left_distance = abs(self.lengths[left] - desired_chars) if left >= 0 else float("inf")
            right_distance = abs(self.lengths[right] - desired_chars) if right < len(self.lengths) else float("inf")
            if left_distance < right_distance:
                candidate_lengths = [self.lengths[left]]
                left -= 1
            elif right_distance < left_distance:
                candidate_lengths = [self.lengths[right]]
                right += 1
            else:
                candidate_lengths = []
                if left >= 0:
                    candidate_lengths.append(self.lengths[left])
                    left -= 1
                if right < len(self.lengths):
                    candidate_lengths.append(self.lengths[right])
                    right += 1
                candidate_lengths.sort(key=lambda length: stable_seed(namespace, "length", length))

            for length in candidate_lengths:
                bucket = self.by_length[length]
                start = stable_seed(namespace, "bucket", length) % len(bucket)
                for offset in range(len(bucket)):
                    donor = bucket[(start + offset) % len(bucket)]
                    if donor.query_id == target_qid:
                        continue
                    if donor.pid in excluded_pids or donor.pid in used_pids:
                        continue
                    used_pids.add(donor.pid)
                    return donor
        raise ValueError(f"no unrelated donor available for qid={target_qid} desired_chars={desired_chars}")


def _normalized_chars(passage: dict, *, max_doc_chars: int) -> int:
    return len(
        normalize_doc_text(
            str(passage.get("text", "")),
            max_doc_chars=max_doc_chars,
        )
    )


def build_unrelated_context(
    real_context: list[dict],
    *,
    target_index: int,
    target_qid: str,
    target_pool_pids: set[str],
    donor_index: LengthMatchedDonorIndex,
    max_doc_chars: int,
    namespace: str,
) -> tuple[list[dict], list[int]]:
    """Replace every real companion with a nearest-length unrelated donor."""
    if not 0 <= target_index < len(real_context):
        raise ValueError("target_index is outside real_context")
    target = real_context[target_index]
    used_pids: set[str] = set()
    out: list[dict] = []
    errors: list[int] = []
    for slot, real_passage in enumerate(real_context):
        if slot == target_index:
            out.append(target)
            continue
        desired = _normalized_chars(
            real_passage,
            max_doc_chars=max_doc_chars,
        )
        donor = donor_index.choose(
            desired_chars=desired,
            target_qid=target_qid,
            excluded_pids=target_pool_pids,
            used_pids=used_pids,
            namespace=f"{namespace}:slot={slot}",
        )
        out.append(donor.passage)
        errors.append(abs(donor.matched_chars - desired))
    return out, errors


def _bootstrap_interval(
    values: list[float],
    *,
    samples: int,
    seed: int,
) -> list[float] | None:
    if not values or samples <= 0:
        return None
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(arr, size=(samples, len(arr)), replace=True).mean(axis=1)
    return [
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    ]


def _aggregate_query_rows(
    rows: dict[str, dict],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict:
    aggregate: dict[str, Any] = {}
    for reading in ("canonical", "order_averaged_k10"):
        arm_values = {arm: [float(row["ndcg"][reading]["arms"][arm]) for row in rows.values()] for arm in ARM_NAMES}
        effects = {
            name: [float(row["ndcg"][reading]["decomposition"][name]) for row in rows.values()]
            for name in (*DECOMPOSITION_TERMS, "closure_residual")
        }
        aggregate[reading] = {
            "arms": {
                arm: {
                    "mean_ndcg_cut_10": float(np.mean(values)),
                    "n_queries": len(values),
                }
                for arm, values in arm_values.items()
            },
            "decomposition": {
                name: {
                    "mean": float(np.mean(values)),
                    "query_bootstrap_95_ci": _bootstrap_interval(
                        values,
                        samples=bootstrap_samples,
                        seed=stable_seed(bootstrap_seed, reading, name),
                    ),
                }
                for name, values in effects.items()
            },
        }

    for reading in ("canonical", "order_averaged_k10"):
        aggregate.setdefault("score_level", {})[reading] = {}
        for name in DECOMPOSITION_TERMS:
            signed = [float(row["scores"][reading]["decomposition"][name]["mean"]) for row in rows.values()]
            absolute = [float(row["scores"][reading]["decomposition"][name]["mean_abs"]) for row in rows.values()]
            aggregate["score_level"][reading][name] = {
                "mean_signed": float(np.mean(signed)),
                "mean_absolute": float(np.mean(absolute)),
                "n_queries": len(signed),
            }

    match_histogram: dict[int, int] = {}
    for row in rows.values():
        histogram = row.get("length_matching", {}).get("absolute_char_error_histogram") or {}
        for error, count in histogram.items():
            error_i = int(error)
            match_histogram[error_i] = match_histogram.get(error_i, 0) + int(count)
    n_matches = sum(match_histogram.values())
    weighted_sum = sum(error * count for error, count in match_histogram.items())
    p95_error = None
    if n_matches:
        threshold = 0.95 * n_matches
        cumulative = 0
        for error in sorted(match_histogram):
            cumulative += match_histogram[error]
            if cumulative >= threshold:
                p95_error = float(error)
                break
    aggregate["length_matching"] = {
        "n_companions": n_matches,
        "mean_absolute_char_error": (float(weighted_sum / n_matches) if n_matches else None),
        "p95_absolute_char_error": p95_error,
        "exact_match_rate": (float(match_histogram.get(0, 0) / n_matches) if n_matches else None),
        "absolute_char_error_histogram": {str(error): match_histogram[error] for error in sorted(match_histogram)},
    }
    covariate_keys = (
        "mean_pool_doc_chars",
        "std_pool_doc_chars",
        "qrels_positive_fraction",
        "mean_qrel_grade_over_pool",
    )
    aggregate["collection_covariates"] = {}
    for key in covariate_keys:
        values = [
            float(row["collection_covariates"][key])
            for row in rows.values()
            if row.get("collection_covariates", {}).get(key) is not None
        ]
        aggregate["collection_covariates"][key] = float(np.mean(values)) if values else None
    return aggregate


def _aggregate_screen_rows(
    rows: dict[str, dict],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict:
    """Aggregate an A1/A4-only screen without requiring A2 or A3 inference."""
    aggregate: dict[str, Any] = {}
    for reading in ("canonical", "order_averaged_k10"):
        arm_values = {
            arm: [float(row["ndcg"][reading]["arms"][arm]) for row in rows.values()] for arm in SCREEN_ARM_NAMES
        }
        effects = [float(row["ndcg"][reading]["cumulative_skeleton"]) for row in rows.values()]
        aggregate[reading] = {
            "arms": {
                arm: {
                    "mean_ndcg_cut_10": float(np.mean(values)),
                    "n_queries": len(values),
                }
                for arm, values in arm_values.items()
            },
            "cumulative_skeleton": {
                "mean": float(np.mean(effects)),
                "query_bootstrap_95_ci": _bootstrap_interval(
                    effects,
                    samples=bootstrap_samples,
                    seed=stable_seed(
                        bootstrap_seed,
                        reading,
                        "cumulative_skeleton",
                    ),
                ),
            },
        }

        signed = [float(row["scores"][reading]["cumulative_skeleton"]["mean"]) for row in rows.values()]
        absolute = [float(row["scores"][reading]["cumulative_skeleton"]["mean_abs"]) for row in rows.values()]
        aggregate.setdefault("score_level", {})[reading] = {
            "cumulative_skeleton": {
                "mean_signed": float(np.mean(signed)),
                "mean_absolute": float(np.mean(absolute)),
                "n_queries": len(signed),
            }
        }

    covariate_keys = (
        "mean_pool_doc_chars",
        "std_pool_doc_chars",
        "qrels_positive_fraction",
        "mean_qrel_grade_over_pool",
    )
    aggregate["collection_covariates"] = {}
    for key in covariate_keys:
        values = [
            float(row["collection_covariates"][key])
            for row in rows.values()
            if row.get("collection_covariates", {}).get(key) is not None
        ]
        aggregate["collection_covariates"][key] = float(np.mean(values)) if values else None
    return aggregate


class ContextDecompositionRunner:
    """Run and reduce the Category-H A1-A4 control on one dataset."""

    def __init__(  # noqa: C901 - validates one YAML-driven runner contract
        self,
        config_path: str | Path,
        runs_root: str | Path = "runs",
        run_dir: str | Path | None = None,
        reranker: object | None = None,
    ):
        """Load the context-decomposition config, data, scorer, and resumable output tree."""
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")
        self.config: dict = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        self.exp_id = str(self.config["id"])
        self.data_config = self.config.get("data") or {}
        self.control_cfg = self.config.get("context_decomposition") or {}
        if not self.control_cfg:
            raise ValueError("config.context_decomposition is required")
        self.screen_only = bool(self.control_cfg.get("screen_only", False))

        self.pool_size = int(self.control_cfg.get("pool_size", 100))
        self.width = int(self.control_cfg.get("width", 20))
        self.variable_pool_size = bool(self.control_cfg.get("variable_pool_size", False))
        if self.pool_size < 1 or self.width < 1:
            raise ValueError("pool_size and width must be positive")
        self.seeds = [int(seed) for seed in self.control_cfg.get("seeds", list(range(10)))]
        if len(self.seeds) != 10:
            raise ValueError("Category H requires exactly ten published presentations")
        self.k_cutoff = int(self.control_cfg.get("k_cutoff_for_ndcg", 10))
        self.request_batch_size = int(self.control_cfg.get("request_batch_size", 128))
        if self.request_batch_size < 1:
            raise ValueError("context_decomposition.request_batch_size must be positive")
        max_queries = self.control_cfg.get("max_queries")
        self.max_queries = int(max_queries) if max_queries is not None else None
        self.bootstrap_samples = int(self.control_cfg.get("bootstrap_samples", 10_000))
        self.bootstrap_seed = int(self.control_cfg.get("bootstrap_seed", 0))
        self.slot0_score_tolerance = float(self.control_cfg.get("slot0_score_tolerance", 1e-4))
        if self.slot0_score_tolerance < 0:
            raise ValueError("context_decomposition.slot0_score_tolerance must be non-negative")
        self.qid_shard_count = int(self.control_cfg.get("qid_shard_count", 1))
        self.qid_shard_index = int(self.control_cfg.get("qid_shard_index", 0))
        shard_qids(
            [],
            shard_count=self.qid_shard_count,
            shard_index=self.qid_shard_index,
        )
        self.donor_seed_namespace = str(self.control_cfg.get("donor_seed_namespace", self.exp_id))
        self.dataset_key = str(self.control_cfg.get("dataset_key", self.exp_id))

        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.run_dir = Path(run_dir) if run_dir is not None else Path(runs_root) / self.exp_id / ts
        self.output_dir = self.run_dir / "context_decomposition"
        self.results_dir = self.output_dir / "per_query_results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logging(
            self.__class__.__name__,
            self.config,
            output_file=str(self.run_dir / "context_decomposition.log"),
        )

        if reranker is None:
            self.reranker = instantiate_reranker(self.config, self.logger)
        else:
            self.reranker = reranker
            _apply_prompt_overrides(
                self.reranker,
                self.config.get("reranker") or {},
                self.logger,
            )
        if not hasattr(self.reranker, "score_selected_slots"):
            raise TypeError("context decomposition requires a reranker with score_selected_slots()")
        configured_width = int(
            (self.config.get("reranker") or {}).get(
                "docs_per_score_forward",
                self.width,
            )
        )
        if configured_width != self.width:
            raise ValueError(
                f"reranker.docs_per_score_forward={configured_width} "
                f"does not match context_decomposition.width={self.width}"
            )
        self.max_doc_chars = int((self.config.get("reranker") or {}).get("max_doc_chars", 1200))

        self.dataloader = instantiate_dataloader(self.config, self.data_config)
        qrels_path = (self.config.get("eval") or {}).get("qrels_path")
        if not qrels_path:
            raise ValueError("context decomposition requires eval.qrels_path")
        self.qrels = load_qrels(qrels_path)
        if run_dir is None:
            write_resolved_run_config(
                self.run_dir,
                self.config,
                config_path=self.config_path,
                ts=ts,
                data_config=self.data_config,
            )

    def _query_dir(self, qid: str) -> Path:
        return self.results_dir / qid_to_dirname(qid)

    def _load_completed(self, qid: str) -> dict | None:
        path = self._query_dir(qid) / "metrics.json"
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def _score_selected(self, query: dict, requests: list[dict]) -> list[float]:
        scores: list[float] = []
        for start in range(0, len(requests), self.request_batch_size):
            chunk = requests[start : start + self.request_batch_size]
            scores.extend(self.reranker.score_selected_slots(query, chunk))
        if len(scores) != len(requests):
            raise RuntimeError(f"selected-slot scorer returned {len(scores)} scores for {len(requests)} requests")
        return [float(score) for score in scores]

    def _score_a2(self, query: dict, presented: list[dict]) -> dict[str, float]:
        chunks = [presented[start : start + self.width] for start in range(0, len(presented), self.width)]
        items = [(query, chunk) for chunk in chunks]
        if getattr(self.reranker, "supports_query_batching", False):
            results = self.reranker.rank_query_batch(items)
        else:
            results = [self.reranker.rank(query, chunk) for chunk in chunks]
        if len(results) != len(chunks):
            raise RuntimeError("A2 scorer did not return one result per chunk")
        out: dict[str, float] = {}
        for chunk, result in zip(chunks, results, strict=True):
            scores = result.get("scores_init_order")
            if not isinstance(scores, list) or len(scores) != len(chunk):
                raise ValueError("A2 result lacks aligned scores_init_order")
            for passage, score in zip(chunk, scores, strict=True):
                out[str(passage["pid"])] = float(score)
        if len(out) != len(presented):
            raise ValueError("A2 scoring did not cover the full target pool")
        return out

    def _score_query_screen(
        self,
        query: dict,
        passages: list[dict],
    ) -> dict:
        """Score only A1 and A4 for the cheap suite-wide scaffold screen."""
        qid = str(query["qid"])
        first_stage = list(passages[: self.pool_size])
        pids_first_stage = [str(passage["pid"]) for passage in first_stage]
        qrels_for_q = self.qrels.get(qid) or {}
        pool_doc_chars = [
            _normalized_chars(
                passage,
                max_doc_chars=self.max_doc_chars,
            )
            for passage in first_stage
        ]
        pool_rels = [int(qrels_for_q.get(pid, 0)) for pid in pids_first_stage]

        a1_requests = [
            {
                "passages": [passage],
                "target_skeleton_index": 0,
            }
            for passage in first_stage
        ]
        a1_values = self._score_selected(query, a1_requests)
        a1_scores = dict(zip(pids_first_stage, a1_values, strict=True))

        a4_by_presentation: dict[str, dict[str, float]] = {}
        n_a4_slot0_reused = 0
        for presentation_label, presented in published_presentations(
            first_stage,
            self.seeds,
        ):
            a4_requests: list[dict] = []
            a4_requested_pids: list[str] = []
            a4_scores: dict[str, float] = {}
            for position, target in enumerate(presented):
                chunk_start = (position // self.width) * self.width
                context_width = len(presented[chunk_start : chunk_start + self.width])
                target_index = position % self.width
                skeleton_labels = tuple(f"[{i}]" for i in range(1, context_width + 1))
                target_pid = str(target["pid"])
                if target_index == 0:
                    a4_scores[target_pid] = a1_scores[target_pid]
                    n_a4_slot0_reused += 1
                else:
                    a4_requests.append(
                        {
                            "passages": [target],
                            "document_slot_labels": (skeleton_labels[target_index],),
                            "skeleton_slot_labels": skeleton_labels,
                            "target_skeleton_index": target_index,
                        }
                    )
                    a4_requested_pids.append(target_pid)
            a4_values = self._score_selected(query, a4_requests)
            a4_scores.update(zip(a4_requested_pids, a4_values, strict=True))
            if set(a4_scores) != set(pids_first_stage):
                raise ValueError("A4 screen did not cover the full target pool")
            a4_by_presentation[presentation_label] = a4_scores

        random_labels = [f"random_s{seed}" for seed in self.seeds]
        canonical_scores = {
            "A1": a1_scores,
            "A4": a4_by_presentation["canonical"],
        }
        order_averaged_scores = {
            "A1": a1_scores,
            "A4": {
                pid: float(np.mean([a4_by_presentation[label][pid] for label in random_labels]))
                for pid in pids_first_stage
            },
        }

        def ndcg_arms(
            score_map: dict[str, dict[str, float]],
        ) -> dict[str, float]:
            return {
                arm: ndcg_at_k(
                    rank_ids_from_scores(pids_first_stage, scores),
                    qrels_for_q,
                    k=self.k_cutoff,
                )
                for arm, scores in score_map.items()
            }

        canonical_ndcg = ndcg_arms(canonical_scores)
        order_averaged_ndcg = ndcg_arms(order_averaged_scores)
        return {
            "qid": qid,
            "pids_first_stage": pids_first_stage,
            "scores_by_arm": {
                "A1": {"invariant": a1_scores},
                "A4": a4_by_presentation,
            },
            "ndcg": {
                "canonical": {
                    "arms": canonical_ndcg,
                    "cumulative_skeleton": (canonical_ndcg["A4"] - canonical_ndcg["A1"]),
                },
                "order_averaged_k10": {
                    "arms": order_averaged_ndcg,
                    "cumulative_skeleton": (order_averaged_ndcg["A4"] - order_averaged_ndcg["A1"]),
                },
            },
            "scores": {
                "canonical": {
                    "cumulative_skeleton": score_effect_summary(
                        canonical_scores["A4"],
                        canonical_scores["A1"],
                    )
                },
                "order_averaged_k10": {
                    "cumulative_skeleton": score_effect_summary(
                        order_averaged_scores["A4"],
                        order_averaged_scores["A1"],
                    )
                },
            },
            "collection_covariates": {
                "mean_pool_doc_chars": float(np.mean(pool_doc_chars)),
                "std_pool_doc_chars": float(np.std(pool_doc_chars, ddof=0)),
                "qrels_positive_fraction": float(np.mean(np.asarray(pool_rels) > 0)),
                "mean_qrel_grade_over_pool": float(np.mean(pool_rels)),
            },
            "cleanliness_checks": {
                "a4_slot0_max_abs_difference_from_a1": 0.0,
                "a4_slot0_reused_from_a1_count": n_a4_slot0_reused,
            },
        }

    def _score_query(
        self,
        query: dict,
        passages: list[dict],
        donor_index: LengthMatchedDonorIndex,
    ) -> dict:
        qid = str(query["qid"])
        first_stage = list(passages[: self.pool_size])
        pids_first_stage = [str(passage["pid"]) for passage in first_stage]
        target_pool_pids = set(pids_first_stage)
        qrels_for_q = self.qrels.get(qid) or {}
        pool_doc_chars = [
            _normalized_chars(
                passage,
                max_doc_chars=self.max_doc_chars,
            )
            for passage in first_stage
        ]
        pool_rels = [int(qrels_for_q.get(pid, 0)) for pid in pids_first_stage]
        a1_requests = [
            {
                "passages": [passage],
                "target_skeleton_index": 0,
            }
            for passage in first_stage
        ]
        a1_values = self._score_selected(query, a1_requests)
        a1_scores = dict(zip(pids_first_stage, a1_values, strict=True))

        scores_by_arm: dict[str, dict[str, dict[str, float]]] = {
            "A2": {},
            "A3": {},
            "A4": {},
        }
        length_errors: list[int] = []
        n_a4_slot0_reused = 0
        for presentation_label, presented in published_presentations(
            first_stage,
            self.seeds,
        ):
            scores_by_arm["A2"][presentation_label] = self._score_a2(
                query,
                presented,
            )

            a3_requests: list[dict] = []
            a4_requests: list[dict] = []
            a4_requested_pids: list[str] = []
            a4_reused_scores: dict[str, float] = {}
            target_pids_in_request_order: list[str] = []
            for position, target in enumerate(presented):
                chunk_start = (position // self.width) * self.width
                real_context = presented[chunk_start : chunk_start + self.width]
                target_index = position % self.width
                context_width = len(real_context)
                skeleton_labels = tuple(f"[{i}]" for i in range(1, context_width + 1))
                unrelated, errors = build_unrelated_context(
                    real_context,
                    target_index=target_index,
                    target_qid=qid,
                    target_pool_pids=target_pool_pids | set(qrels_for_q),
                    donor_index=donor_index,
                    max_doc_chars=self.max_doc_chars,
                    namespace=(
                        f"{self.donor_seed_namespace}:qid={qid}:"
                        f"presentation={presentation_label}:"
                        f"target={target['pid']}"
                    ),
                )
                length_errors.extend(errors)
                a3_requests.append(
                    {
                        "passages": unrelated,
                        "target_skeleton_index": target_index,
                    }
                )
                target_pid = str(target["pid"])
                if target_index == 0:
                    # The selected A4 prefix at slot zero is byte-identical to
                    # A1. Reuse A1 instead of rescoring the same prompt in a
                    # different vLLM batch, which can introduce ~1e-2 numeric
                    # drift despite identical tokens.
                    a4_reused_scores[target_pid] = a1_scores[target_pid]
                    n_a4_slot0_reused += 1
                else:
                    a4_requests.append(
                        {
                            "passages": [target],
                            "document_slot_labels": (skeleton_labels[target_index],),
                            "skeleton_slot_labels": skeleton_labels,
                            "target_skeleton_index": target_index,
                        }
                    )
                    a4_requested_pids.append(target_pid)
                target_pids_in_request_order.append(target_pid)

            a3_values = self._score_selected(query, a3_requests)
            a4_values = self._score_selected(query, a4_requests)
            scores_by_arm["A3"][presentation_label] = dict(zip(target_pids_in_request_order, a3_values, strict=True))
            a4_scores = dict(a4_reused_scores)
            a4_scores.update(zip(a4_requested_pids, a4_values, strict=True))
            if set(a4_scores) != set(target_pids_in_request_order):
                raise ValueError("A4 scoring did not cover the full target pool")
            scores_by_arm["A4"][presentation_label] = a4_scores

        canonical_scores = {
            "A1": a1_scores,
            **{arm: scores_by_arm[arm]["canonical"] for arm in ("A2", "A3", "A4")},
        }
        random_labels = [f"random_s{seed}" for seed in self.seeds]
        order_averaged_scores = {"A1": a1_scores}
        for arm in ("A2", "A3", "A4"):
            order_averaged_scores[arm] = {
                pid: float(np.mean([scores_by_arm[arm][label][pid] for label in random_labels]))
                for pid in pids_first_stage
            }

        def ndcg_arms(score_map: dict[str, dict[str, float]]) -> dict[str, float]:
            return {
                arm: ndcg_at_k(
                    rank_ids_from_scores(pids_first_stage, scores),
                    qrels_for_q,
                    k=self.k_cutoff,
                )
                for arm, scores in score_map.items()
            }

        canonical_ndcg = ndcg_arms(canonical_scores)
        order_averaged_ndcg = ndcg_arms(order_averaged_scores)

        def score_decomposition(
            score_map: dict[str, dict[str, float]],
        ) -> dict[str, dict[str, float]]:
            return {
                name: score_effect_summary(
                    score_map[left],
                    score_map[right],
                )
                for name, (left, right) in DECOMPOSITION_TERMS.items()
            }

        length_error_histogram: dict[str, int] = {}
        for error in length_errors:
            key = str(int(error))
            length_error_histogram[key] = length_error_histogram.get(key, 0) + 1
        row = {
            "qid": qid,
            "pids_first_stage": pids_first_stage,
            "scores_by_arm": {
                "A1": {"invariant": a1_scores},
                **scores_by_arm,
            },
            "ndcg": {
                "canonical": {
                    "arms": canonical_ndcg,
                    "decomposition": decompose_values(canonical_ndcg),
                },
                "order_averaged_k10": {
                    "arms": order_averaged_ndcg,
                    "decomposition": decompose_values(order_averaged_ndcg),
                },
            },
            "scores": {
                "canonical": {
                    "decomposition": score_decomposition(canonical_scores),
                },
                "order_averaged_k10": {
                    "decomposition": score_decomposition(order_averaged_scores),
                },
            },
            "length_matching": {
                "n_companions": len(length_errors),
                "absolute_char_error_histogram": length_error_histogram,
            },
            "collection_covariates": {
                "mean_pool_doc_chars": float(np.mean(pool_doc_chars)),
                "std_pool_doc_chars": float(np.std(pool_doc_chars, ddof=0)),
                "qrels_positive_fraction": float(np.mean(np.asarray(pool_rels) > 0)),
                "mean_qrel_grade_over_pool": float(np.mean(pool_rels)),
            },
            "cleanliness_checks": {
                "a4_slot0_max_abs_difference_from_a1": 0.0,
                "a4_slot0_reused_from_a1_count": n_a4_slot0_reused,
            },
        }
        max_slot0_diff = row["cleanliness_checks"]["a4_slot0_max_abs_difference_from_a1"]
        if isinstance(max_slot0_diff, (int, float)) and max_slot0_diff > self.slot0_score_tolerance:
            raise ValueError(
                "A4 slot-zero prefix is not score-identical to A1: "
                f"max_abs_diff={max_slot0_diff} "
                f"tolerance={self.slot0_score_tolerance}"
            )
        return row

    def run(self) -> dict:  # noqa: C901 - shared full/screen runner lifecycle
        """Score all eligible queries and write dataset-level A1-A4 metrics."""
        run_path = str(self.data_config["run_path"])
        queries = self.dataloader.get_qs_from_run(run_path)
        query_pools: dict[str, list[dict]] = {}
        query_by_qid: dict[str, dict] = {}
        for query in queries:
            qid = str(query["qid"])
            passages = self.dataloader.get_psgs_from_run(run_path, qid) or []
            if qid not in self.qrels:
                continue
            if not passages:
                continue
            if not self.variable_pool_size and len(passages) < self.pool_size:
                continue
            query_by_qid[qid] = query
            query_pools[qid] = list(passages[: self.pool_size])
        if not self.screen_only and len(query_pools) < 2:
            raise ValueError("context decomposition needs at least two eligible queries to draw unrelated companions")

        eligible_qids = list(query_pools)
        if self.max_queries is not None:
            eligible_qids = eligible_qids[: self.max_queries]
        eligible_index = {qid: index for index, qid in enumerate(eligible_qids)}
        selected_qids = shard_qids(
            eligible_qids,
            shard_count=self.qid_shard_count,
            shard_index=self.qid_shard_index,
        )
        if not selected_qids:
            raise ValueError(
                "context decomposition shard is empty: "
                f"index={self.qid_shard_index} count={self.qid_shard_count} "
                f"eligible={len(eligible_qids)}"
            )
        donor_index = (
            None
            if self.screen_only
            else LengthMatchedDonorIndex(
                query_pools,
                max_doc_chars=self.max_doc_chars,
            )
        )
        (self.output_dir / "selected_qids.txt").write_text(
            "".join(f"{qid}\n" for qid in selected_qids),
            encoding="utf-8",
        )

        per_query: dict[str, dict] = {}
        failed_qids: list[str] = []
        write_progress_jsonl, heartbeat_s = progress_config(self.config)
        progress = ProgressTracker(
            phase="context_decomposition",
            total_work=len(selected_qids),
            total_queries=len(selected_qids),
            jsonl_path=self.output_dir / "progress.jsonl",
            write_jsonl=write_progress_jsonl,
            heartbeat_every_s=heartbeat_s,
        )
        for q_index, qid in enumerate(selected_qids, start=1):
            cached = self._load_completed(qid)
            if cached is not None:
                per_query[qid] = cached
                progress.skip_units(
                    count=1,
                    label=f"qid={qid}",
                    q_index=q_index,
                    reason="query_resumed_from_disk",
                )
                continue

            label = f"qid={qid}"
            work_index, suffix = progress.start_unit(
                label=label,
                q_index=q_index,
                extra={"qid": qid},
            )
            self.logger.info(
                "Context decomposition qid=%s (%s)",
                qid,
                suffix,
            )
            started = time.monotonic()
            try:
                if self.screen_only:
                    row = self._score_query_screen(
                        query_by_qid[qid],
                        query_pools[qid],
                    )
                else:
                    if donor_index is None:
                        raise RuntimeError("A1-A4 mode requires a donor index")
                    row = self._score_query(
                        query_by_qid[qid],
                        query_pools[qid],
                        donor_index,
                    )
                row["eligible_query_index"] = eligible_index[qid]
                qdir = self._query_dir(qid)
                qdir.mkdir(parents=True, exist_ok=True)
                (qdir / "metrics.json").write_text(
                    json.dumps(row, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
                per_query[qid] = row
                progress.finish_unit(
                    label=label,
                    work_index=work_index,
                    q_index=q_index,
                    duration_s=time.monotonic() - started,
                    success=True,
                    extra={"qid": qid},
                )
                progress.maybe_heartbeat(self.logger)
            except Exception as exc:  # noqa: BLE001 - finish other qids, reject partial aggregate
                self.logger.error(
                    "Failed context decomposition qid=%s: %s",
                    qid,
                    exc,
                    exc_info=True,
                )
                failed_qids.append(qid)
                progress.finish_unit(
                    label=label,
                    work_index=work_index,
                    q_index=q_index,
                    duration_s=time.monotonic() - started,
                    success=False,
                    extra={"qid": qid, "reason": type(exc).__name__},
                )

        if failed_qids:
            raise partial_run_error(
                failed_qids,
                phase="context decomposition",
                aggregate_label="aggregate A1-A4 metrics",
            )

        aggregate = (
            _aggregate_screen_rows(
                per_query,
                bootstrap_samples=self.bootstrap_samples,
                bootstrap_seed=self.bootstrap_seed,
            )
            if self.screen_only
            else _aggregate_query_rows(
                per_query,
                bootstrap_samples=self.bootstrap_samples,
                bootstrap_seed=self.bootstrap_seed,
            )
        )
        metrics = {
            "schema_version": 1,
            "exp_id": self.exp_id,
            "aggregate": aggregate,
            "protocol": {
                "mode": "a4_screen" if self.screen_only else "a1_a2_a3_a4",
                "arms": (
                    {
                        "A1": "target alone; one grade line",
                        "A4": (
                            "target alone in the user prompt; numeric target marker "
                            "and cumulative grade slot match A2 within a B-line skeleton"
                        ),
                    }
                    if self.screen_only
                    else {
                        "A1": "target alone; one grade line",
                        "A2": "target plus its real pool companions",
                        "A3": (
                            "target plus nearest-character-length documents from "
                            "other queries; target is the only query-matched document"
                        ),
                        "A4": (
                            "target alone in the user prompt; numeric target marker "
                            "and cumulative grade slot match A2/A3 within a B-line skeleton"
                        ),
                    }
                ),
                "decomposition": (
                    {"cumulative_skeleton": "A4-A1"}
                    if self.screen_only
                    else {name: f"{left}-{right}" for name, (left, right) in DECOMPOSITION_TERMS.items()}
                ),
                "pool_size": self.pool_size,
                "variable_pool_size": self.variable_pool_size,
                "width": self.width,
                "canonical_order": "first-stage rank order",
                "random_presentations": ("random.Random(seed).shuffle(top-pool), matching the PSI harness"),
                "seeds": self.seeds,
                "target_slot": ("identical in A2, A3, and A4 for every target and presentation"),
                "partial_chunk_policy": (
                    "A3 companion count and A4 skeleton width match the target's "
                    "actual A2 chunk, including a final chunk smaller than width"
                ),
                "length_match_unit": (
                    "characters after whitespace normalization and the dataset's max_doc_chars truncation"
                ),
                "donor_exclusions": (
                    "same query, any pid in the target query's top pool, and "
                    "duplicate donor use within one target context"
                ),
                "a3_interpretation": (
                    None
                    if self.screen_only
                    else (
                        "A3 makes the target the only query-matched document, so A2-A3 "
                        "measures informative-real versus unrelated alternatives, not "
                        "presence versus absence of any alternatives"
                    )
                ),
                "k_cutoff": self.k_cutoff,
                "bootstrap_samples": self.bootstrap_samples,
                "bootstrap_seed": self.bootstrap_seed,
                "slot0_score_tolerance": self.slot0_score_tolerance,
                "dataset_key": self.dataset_key,
                "donor_seed_namespace": self.donor_seed_namespace,
                "qid_shard_count": self.qid_shard_count,
                "qid_shard_index": self.qid_shard_index,
                "n_eligible_queries_total": len(eligible_qids),
                "checkpoint_path": str((self.config.get("reranker") or {}).get("lora_path", "")),
                "temperature": 0,
                "n_queries": len(per_query),
            },
        }
        metrics_path = self.output_dir / "context_decomposition_metrics.json"
        metrics_path.write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        compact_per_query = {
            qid: {key: value for key, value in row.items() if key not in {"pids_first_stage", "scores_by_arm"}}
            for qid, row in per_query.items()
        }
        (self.output_dir / "context_decomposition_per_query.json").write_text(
            json.dumps(compact_per_query, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        self.logger.info("Wrote %s", metrics_path)
        return metrics


def _context_output_dir(path: str | Path) -> Path:
    candidate = Path(path)
    nested = candidate / "context_decomposition"
    return nested if nested.is_dir() else candidate


def merge_context_decomposition_shards(  # noqa: C901 - strict merge validation
    shard_paths: list[str | Path],
    *,
    output_dir: str | Path,
    merged_exp_id: str | None = None,
) -> dict:
    """Merge disjoint query shards and recompute the exact dataset aggregate."""
    if not shard_paths:
        raise ValueError("at least one shard path is required")

    loaded: list[tuple[Path, dict, dict]] = []
    for raw_path in shard_paths:
        context_dir = _context_output_dir(raw_path)
        metrics_path = context_dir / "context_decomposition_metrics.json"
        per_query_path = context_dir / "context_decomposition_per_query.json"
        if not metrics_path.is_file() or not per_query_path.is_file():
            raise FileNotFoundError(f"shard is missing context outputs: {context_dir}")
        loaded.append(
            (
                context_dir,
                json.loads(metrics_path.read_text(encoding="utf-8")),
                json.loads(per_query_path.read_text(encoding="utf-8")),
            )
        )

    signature_keys = (
        "mode",
        "arms",
        "decomposition",
        "pool_size",
        "variable_pool_size",
        "width",
        "canonical_order",
        "random_presentations",
        "seeds",
        "target_slot",
        "partial_chunk_policy",
        "length_match_unit",
        "donor_exclusions",
        "a3_interpretation",
        "k_cutoff",
        "bootstrap_samples",
        "bootstrap_seed",
        "slot0_score_tolerance",
        "dataset_key",
        "donor_seed_namespace",
        "qid_shard_count",
        "n_eligible_queries_total",
        "checkpoint_path",
        "temperature",
    )
    first_protocol = loaded[0][1].get("protocol") or {}
    expected_signature = {key: first_protocol.get(key) for key in signature_keys}
    shard_count = int(first_protocol.get("qid_shard_count", 0))
    expected_indices = set(range(shard_count))
    seen_indices: set[int] = set()
    rows_by_index: dict[int, tuple[str, dict]] = {}
    source_shards: list[dict] = []
    for context_dir, metrics, rows in loaded:
        protocol = metrics.get("protocol") or {}
        signature = {key: protocol.get(key) for key in signature_keys}
        if signature != expected_signature:
            raise ValueError(f"shard protocol mismatch: {context_dir}")
        shard_index = int(protocol.get("qid_shard_index", -1))
        if shard_index in seen_indices:
            raise ValueError(f"duplicate shard index: {shard_index}")
        seen_indices.add(shard_index)
        source_shards.append(
            {
                "path": str(context_dir),
                "exp_id": metrics.get("exp_id"),
                "qid_shard_index": shard_index,
                "n_queries": len(rows),
            }
        )
        for qid, row in rows.items():
            eligible_query_index = int(row["eligible_query_index"])
            if eligible_query_index in rows_by_index:
                prior_qid = rows_by_index[eligible_query_index][0]
                raise ValueError(f"overlapping eligible query index {eligible_query_index}: {prior_qid} and {qid}")
            rows_by_index[eligible_query_index] = (str(qid), row)

    if seen_indices != expected_indices:
        raise ValueError(f"incomplete shard set: got={sorted(seen_indices)} expected={sorted(expected_indices)}")
    n_expected = int(first_protocol.get("n_eligible_queries_total", 0))
    if set(rows_by_index) != set(range(n_expected)):
        missing = sorted(set(range(n_expected)) - set(rows_by_index))
        extra = sorted(set(rows_by_index) - set(range(n_expected)))
        raise ValueError(f"query coverage mismatch: missing={missing[:10]} extra={extra[:10]}")

    ordered_rows = {rows_by_index[index][0]: rows_by_index[index][1] for index in range(n_expected)}
    aggregate = (
        _aggregate_screen_rows(
            ordered_rows,
            bootstrap_samples=int(first_protocol["bootstrap_samples"]),
            bootstrap_seed=int(first_protocol["bootstrap_seed"]),
        )
        if first_protocol.get("mode") == "a4_screen"
        else _aggregate_query_rows(
            ordered_rows,
            bootstrap_samples=int(first_protocol["bootstrap_samples"]),
            bootstrap_seed=int(first_protocol["bootstrap_seed"]),
        )
    )
    merged_protocol = dict(first_protocol)
    merged_protocol.pop("qid_shard_index", None)
    merged_protocol["n_queries"] = len(ordered_rows)
    merged_protocol["merged_from_shards"] = source_shards
    metrics = {
        "schema_version": 1,
        "exp_id": merged_exp_id or f"{first_protocol.get('dataset_key', 'context')}-merged",
        "aggregate": aggregate,
        "protocol": merged_protocol,
    }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "context_decomposition_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_path / "context_decomposition_per_query.json").write_text(
        json.dumps(ordered_rows, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_path / "selected_qids.txt").write_text(
        "".join(f"{qid}\n" for qid in ordered_rows),
        encoding="utf-8",
    )
    return metrics
