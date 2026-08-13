"""Repeated fixed-order scoring with varied batch positions and compositions.

The ordinary B=1 width arm is read once while the wide K=10 arm averages ten
score vectors. Dispatches every B=1 candidate ten times. For each seed it
globally shuffles all selected (query, candidate) requests before chunking them
into vLLM calls, so a byte-identical candidate prefix moves through both request
positions and co-batched request compositions. Scores are averaged per candidate
before ranking.

The same machinery also supports a fixed serving width greater than one. In
that mode each request is one fixed-order candidate chunk; only the order and
composition of the chunks sharing an engine call change between reads.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from presentation_dependence.eval.context_decomposition import rank_ids_from_scores, shard_qids
from presentation_dependence.eval.experiment_manager import _apply_prompt_overrides
from presentation_dependence.eval.psi import ndcg_at_k
from presentation_dependence.eval.runner_setup import (
    instantiate_dataloader,
    instantiate_reranker,
    load_qrels,
    write_resolved_run_config,
)
from presentation_dependence.utils.progress import ProgressTracker, progress_config
from presentation_dependence.utils.setup_logging import setup_logging


RequestRecord = tuple[str, str, dict, list[dict]]


def _composition_digest(records: list[RequestRecord]) -> str:
    """Return an order-independent digest of one engine-call composition."""
    members = sorted(
        f"{qid}\0{passage['pid']}" for qid, _request_id, _query, passages in records for passage in passages
    )
    return hashlib.sha256("\n".join(members).encode("utf-8")).hexdigest()[:16]


def _load_qids_allowlist(config: dict[str, Any]) -> set[str]:
    """Read top-level inline and file-backed qid allowlists."""
    out = set(map(str, config.get("qids_to_run", []) or []))
    path_raw = config.get("qids_to_run_path")
    if path_raw:
        with Path(path_raw).open(encoding="utf-8") as handle:
            out.update(line.strip() for line in handle if line.strip())
    return out


def _mean(values: list[float]) -> float:
    """Return a float mean and reject empty inputs."""
    if not values:
        raise ValueError("cannot average an empty list")
    return float(np.mean(np.asarray(values, dtype=float)))


class MatchedVarianceControlRunner:
    """Re-read one fixed-width checkpoint under ten batch arrangements."""

    def __init__(  # noqa: C901 - one YAML contract is validated together
        self,
        config_path: str | Path,
        runs_root: str | Path = "runs",
        run_dir: str | Path | None = None,
        reranker: object | None = None,
    ):
        """Load the scorer, dataset, and matched-variance batching protocol."""
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")
        self.config: dict[str, Any] = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        self.exp_id = str(self.config["id"])
        self.data_config = self.config.get("data") or {}
        self.control_cfg = self.config.get("matched_variance_control") or {}
        if not self.control_cfg:
            raise ValueError("config.matched_variance_control is required")

        self.seeds = [int(seed) for seed in self.control_cfg.get("seeds", list(range(10)))]
        if self.seeds != list(range(10)):
            raise ValueError("Matched-variance control requires the ten seeds 0..9")
        self.request_batch_size = int(self.control_cfg.get("request_batch_size", 128))
        if self.request_batch_size < 2:
            raise ValueError("matched_variance_control.request_batch_size must be at least 2")
        self.serving_width = int(self.control_cfg.get("serving_width", 1))
        if self.serving_width < 1:
            raise ValueError("matched_variance_control.serving_width must be positive")
        self.k_cutoff = int(self.control_cfg.get("k_cutoff_for_ndcg", 10))
        if self.k_cutoff < 1:
            raise ValueError("k_cutoff_for_ndcg must be positive")
        pool_size_raw = self.control_cfg.get(
            "pool_size",
            self.data_config.get("k_input"),
        )
        self.pool_size = int(pool_size_raw) if pool_size_raw is not None else None
        if self.pool_size is not None and self.pool_size < 1:
            raise ValueError("pool_size must be positive when set")
        self.qid_shard_count = int(self.control_cfg.get("qid_shard_count", 1))
        self.qid_shard_index = int(self.control_cfg.get("qid_shard_index", 0))
        shard_qids(
            [],
            shard_count=self.qid_shard_count,
            shard_index=self.qid_shard_index,
        )
        self.required_unique_positions = int(self.control_cfg.get("required_unique_positions", 2))
        self.required_unique_compositions = int(self.control_cfg.get("required_unique_compositions", 2))
        if (
            min(
                self.required_unique_positions,
                self.required_unique_compositions,
            )
            < 2
        ):
            raise ValueError("Matched-variance audit minima must both be at least 2")

        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.run_dir = Path(run_dir) if run_dir is not None else Path(runs_root) / self.exp_id / ts
        self.output_dir = self.run_dir / "matched_variance"
        self.seed_dir = self.output_dir / "seed_scores"
        self.seed_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logging(
            self.__class__.__name__,
            self.config,
            output_file=str(self.run_dir / "matched_variance.log"),
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
        if not getattr(self.reranker, "supports_query_batching", False):
            raise TypeError("Matched-variance control requires a reranker with cross-query rank_query_batch()")
        configured_width = int(
            (self.config.get("reranker") or {}).get(
                "docs_per_score_forward",
                0,
            )
        )
        if configured_width != self.serving_width:
            raise ValueError(
                "matched-variance serving-width mismatch: "
                f"reranker.docs_per_score_forward={configured_width}, "
                f"matched_variance_control.serving_width={self.serving_width}"
            )

        self.dataloader = instantiate_dataloader(self.config, self.data_config)
        qrels_path = (self.config.get("eval") or {}).get("qrels_path")
        if not qrels_path:
            raise ValueError("Matched-variance control requires eval.qrels_path")
        self.qrels = load_qrels(qrels_path)
        if run_dir is None:
            write_resolved_run_config(
                self.run_dir,
                self.config,
                config_path=self.config_path,
                ts=ts,
                data_config=self.data_config,
            )

    def _seed_path(self, seed: int) -> Path:
        return self.seed_dir / f"seed_{seed:02d}.json"

    def _load_seed(self, seed: int) -> dict[str, Any] | None:
        path = self._seed_path(seed)
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if int(payload.get("seed", -1)) != seed:
            raise ValueError(f"{path}: seed mismatch")
        return payload

    def _eligible_pools(
        self,
    ) -> tuple[dict[str, dict], dict[str, list[dict]], list[str]]:
        run_path = str(self.data_config["run_path"])
        allowlist = _load_qids_allowlist(self.config)
        query_by_qid: dict[str, dict] = {}
        pools: dict[str, list[dict]] = {}
        for query in self.dataloader.get_qs_from_run(run_path):
            qid = str(query["qid"])
            if allowlist and qid not in allowlist:
                continue
            passages = list(self.dataloader.get_psgs_from_run(run_path, qid) or [])
            if self.pool_size is not None:
                passages = passages[: self.pool_size]
            if not passages or qid not in self.qrels:
                continue
            pids = [str(passage["pid"]) for passage in passages]
            if len(pids) != len(set(pids)):
                raise ValueError(f"qid={qid}: duplicate candidate pid")
            query_by_qid[qid] = query
            pools[qid] = passages
        eligible = list(pools)
        selected = shard_qids(
            eligible,
            shard_count=self.qid_shard_count,
            shard_index=self.qid_shard_index,
        )
        if not selected:
            raise ValueError(
                f"Matched-variance shard is empty: index={self.qid_shard_index} count={self.qid_shard_count}"
            )
        return query_by_qid, pools, selected

    def _request_records(
        self,
        query_by_qid: dict[str, dict],
        pools: dict[str, list[dict]],
        selected_qids: list[str],
    ) -> list[RequestRecord]:
        records: list[RequestRecord] = []
        for qid in selected_qids:
            passages = pools[qid]
            for start in range(0, len(passages), self.serving_width):
                chunk = passages[start : start + self.serving_width]
                records.append(
                    (
                        qid,
                        f"{start}:{start + len(chunk)}",
                        query_by_qid[qid],
                        chunk,
                    )
                )
        return records

    def _score_seed(
        self,
        seed: int,
        records: list[RequestRecord],
        progress: ProgressTracker,
        *,
        seed_index: int,
    ) -> dict[str, Any]:
        shuffled = list(records)
        random.Random(seed).shuffle(shuffled)
        scores: dict[str, dict[str, float]] = {}
        layout: dict[str, dict[str, list[Any]]] = {}
        n_batches = (len(shuffled) + self.request_batch_size - 1) // self.request_batch_size
        for batch_index, start in enumerate(
            range(0, len(shuffled), self.request_batch_size),
            start=1,
        ):
            chunk = shuffled[start : start + self.request_batch_size]
            digest = _composition_digest(chunk)
            label = f"seed={seed} batch={batch_index}/{n_batches}"
            work_index, suffix = progress.start_unit(
                label=label,
                q_index=seed_index,
                extra={
                    "seed": seed,
                    "batch_index": batch_index,
                    "n_requests": len(chunk),
                },
            )
            self.logger.info("Scoring %s (%s)", label, suffix)
            started = time.monotonic()
            items = [(query, passages) for _qid, _request_id, query, passages in chunk]
            results = self.reranker.rank_query_batch(items)
            if len(results) != len(chunk):
                raise RuntimeError(f"rank_query_batch returned {len(results)} for {len(chunk)} requests")
            engine_position = 0
            for record, result in zip(chunk, results, strict=True):
                qid, _request_id, _query, passages = record
                values = result.get("scores_init_order")
                if not isinstance(values, list) or len(values) != len(passages):
                    raise ValueError(
                        f"qid={qid}: fixed-width result has "
                        f"{0 if not isinstance(values, list) else len(values)} "
                        f"scores for {len(passages)} candidates"
                    )
                for slot, (passage, value) in enumerate(zip(passages, values, strict=True)):
                    pid = str(passage["pid"])
                    scores.setdefault(qid, {})[pid] = float(value)
                    layout.setdefault(qid, {})[pid] = [
                        engine_position + slot,
                        digest,
                    ]
                engine_position += len(passages)
            progress.finish_unit(
                label=label,
                work_index=work_index,
                q_index=seed_index,
                duration_s=time.monotonic() - started,
                success=True,
                extra={"seed": seed, "batch_index": batch_index},
            )
            progress.maybe_heartbeat(self.logger)
        n_candidates = sum(len(passages) for _qid, _request_id, _query, passages in records)
        if sum(len(values) for values in scores.values()) != n_candidates:
            raise RuntimeError("matched-variance seed scoring did not cover every candidate")
        return {
            "seed": seed,
            "request_batch_size": self.request_batch_size,
            "n_requests": len(records),
            "n_batches": n_batches,
            "scores": scores,
            "layout": layout,
        }

    def _reduce(
        self,
        seed_payloads: list[dict[str, Any]],
        pools: dict[str, list[dict]],
        selected_qids: list[str],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        per_query: dict[str, Any] = {}
        all_variances: list[float] = []
        all_ranges: list[float] = []
        all_unique_positions: list[int] = []
        all_unique_compositions: list[int] = []
        for qid in selected_qids:
            pids = [str(passage["pid"]) for passage in pools[qid]]
            score_matrix = np.asarray(
                [[float(payload["scores"][qid][pid]) for pid in pids] for payload in seed_payloads],
                dtype=float,
            )
            score_means = score_matrix.mean(axis=0)
            mean_scores = dict(zip(pids, score_means.tolist(), strict=True))
            ranking = rank_ids_from_scores(pids, mean_scores)
            per_seed_ndcg = [
                ndcg_at_k(
                    rank_ids_from_scores(
                        pids,
                        dict(zip(pids, row.tolist(), strict=True)),
                    ),
                    self.qrels[qid],
                    k=self.k_cutoff,
                )
                for row in score_matrix
            ]
            variances = score_matrix.var(axis=0, ddof=0)
            ranges = np.ptp(score_matrix, axis=0)
            unique_positions = [len({int(payload["layout"][qid][pid][0]) for payload in seed_payloads}) for pid in pids]
            unique_compositions = [
                len({str(payload["layout"][qid][pid][1]) for payload in seed_payloads}) for pid in pids
            ]
            all_variances.extend(variances.tolist())
            all_ranges.extend(ranges.tolist())
            all_unique_positions.extend(unique_positions)
            all_unique_compositions.extend(unique_compositions)
            per_query[qid] = {
                "n_candidates": len(pids),
                "ndcg_k10_score_average": ndcg_at_k(
                    ranking,
                    self.qrels[qid],
                    k=self.k_cutoff,
                ),
                "per_seed_ndcg": per_seed_ndcg,
                "mean_per_seed_ndcg": _mean(per_seed_ndcg),
                "mean_score_variance": float(variances.mean()),
                "max_score_range": float(ranges.max()),
                "min_unique_batch_positions": min(unique_positions),
                "min_unique_batch_compositions": min(unique_compositions),
            }

        min_positions = min(all_unique_positions)
        min_compositions = min(all_unique_compositions)
        if min_positions < self.required_unique_positions:
            raise ValueError(f"Matched-variance batch-position audit failed: minimum unique positions={min_positions}")
        if min_compositions < self.required_unique_compositions:
            raise ValueError(
                f"Matched-variance batch-composition audit failed: minimum unique compositions={min_compositions}"
            )

        aggregate = {
            "n_queries": len(per_query),
            "n_candidates": len(all_variances),
            f"mean_ndcg_cut_{self.k_cutoff}_k10_score_average": _mean(
                [row["ndcg_k10_score_average"] for row in per_query.values()]
            ),
            f"mean_ndcg_cut_{self.k_cutoff}_per_seed": _mean([row["mean_per_seed_ndcg"] for row in per_query.values()]),
            "mean_score_variance": _mean(all_variances),
            "rms_score_sd": float(np.sqrt(np.mean(all_variances))),
            "max_score_range": max(all_ranges),
            "min_unique_batch_positions": min_positions,
            "min_unique_batch_compositions": min_compositions,
            "mean_unique_batch_positions": _mean([float(value) for value in all_unique_positions]),
            "mean_unique_batch_compositions": _mean([float(value) for value in all_unique_compositions]),
        }
        return aggregate, per_query

    def run(self) -> dict[str, Any]:
        """Score all ten arrangements and write matched-variance outputs."""
        query_by_qid, pools, selected_qids = self._eligible_pools()
        records = self._request_records(
            query_by_qid,
            pools,
            selected_qids,
        )
        (self.output_dir / "selected_qids.txt").write_text(
            "".join(f"{qid}\n" for qid in selected_qids),
            encoding="utf-8",
        )
        n_batches = (len(records) + self.request_batch_size - 1) // self.request_batch_size
        write_jsonl, heartbeat_s = progress_config(self.config)
        progress = ProgressTracker(
            phase="matched_variance",
            total_work=n_batches * len(self.seeds),
            total_queries=len(self.seeds),
            jsonl_path=self.output_dir / "progress.jsonl",
            write_jsonl=write_jsonl,
            heartbeat_every_s=heartbeat_s,
        )

        seed_payloads: list[dict[str, Any]] = []
        for seed_index, seed in enumerate(self.seeds, start=1):
            payload = self._load_seed(seed)
            if payload is None:
                payload = self._score_seed(
                    seed,
                    records,
                    progress,
                    seed_index=seed_index,
                )
                self._seed_path(seed).write_text(
                    json.dumps(payload, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
            else:
                progress.skip_units(
                    count=n_batches,
                    label=f"seed={seed}",
                    q_index=seed_index,
                    reason="seed_resumed_from_disk",
                )
            seed_payloads.append(payload)

        aggregate, per_query = self._reduce(
            seed_payloads,
            pools,
            selected_qids,
        )
        metrics = {
            "schema_version": 1,
            "exp_id": self.exp_id,
            "aggregate": aggregate,
            "protocol": {
                "estimand": (
                    f"B={self.serving_width} fixed-order candidate score "
                    "averaged over ten request-batching arrangements "
                    "before ranking"
                ),
                "seeds": self.seeds,
                "request_order": (
                    "global Fisher-Yates shuffle of all selected (query, fixed-order candidate-chunk) requests per seed"
                ),
                "request_batch_size": self.request_batch_size,
                "serving_width": self.serving_width,
                "candidate_order": "fixed first-stage order within every chunk",
                "batch_position": ("candidate-prefix position within rank_query_batch engine call"),
                "batch_composition": (
                    "SHA256 digest of the query/candidate members sharing one rank_query_batch engine call"
                ),
                "k_cutoff_for_ndcg": self.k_cutoff,
                "pool_size": self.pool_size,
                "qid_shard_count": self.qid_shard_count,
                "qid_shard_index": self.qid_shard_index,
                "required_unique_positions": self.required_unique_positions,
                "required_unique_compositions": (self.required_unique_compositions),
                "temperature": 0,
            },
        }
        metrics_path = self.output_dir / "matched_variance_metrics.json"
        metrics_path.write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (self.output_dir / "matched_variance_per_query.json").write_text(
            json.dumps(per_query, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self.logger.info("Wrote %s", metrics_path)
        return metrics
