"""Fixed-order candidate-pool perturbation evaluation.

The order-perturbation PSI harness holds the candidate pool fixed and shuffles
it. This evaluation measures the complementary deployment intervention: keep
the first-stage order fixed, remove one of the top ``N`` candidates, and
optionally append rank ``N+1``. Metrics are computed only on the retained
documents. Query sampling is deterministically seeded; a B=1 reference is
defined as zero change without an additional inference pass.
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

from presentation_dependence.eval.psi import kendall_tau, ndcg_at_k
from presentation_dependence.eval.runner_setup import (
    instantiate_dataloader,
    instantiate_reranker,
    load_qrels,
    partial_run_error,
    validate_rank_result_if_enabled,
    write_resolved_run_config,
)
from presentation_dependence.utils.setup_logging import setup_logging
from presentation_dependence.utils.trec import qid_to_dirname, write_trec_run
from presentation_dependence.utils.progress import ProgressTracker, progress_config


PERTURBATIONS = ("replace", "drop")


def stable_seed(seed: int, qid: str, namespace: str) -> int:
    """Return a process- and Python-version-independent seed."""
    payload = f"{int(seed)}\0{qid}\0{namespace}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def choose_drop_index(qid: str, *, seed: int, pool_size: int) -> int:
    """Choose a uniform zero-based rank in ``[0, pool_size)`` per query."""
    if pool_size < 1:
        raise ValueError("pool_size must be positive")
    return random.Random(stable_seed(seed, str(qid), "drop-rank")).randrange(pool_size)


def build_fixed_order_pools(
    passages: list[dict],
    *,
    drop_index: int,
    pool_size: int = 100,
) -> dict[str, list[dict]]:
    """Build canonical, replace, and drop pools without reordering survivors."""
    if len(passages) < pool_size + 1:
        raise ValueError(f"pool perturbation needs at least {pool_size + 1} candidates, got {len(passages)}")
    if not 0 <= drop_index < pool_size:
        raise ValueError(f"drop_index must be in [0, {pool_size}), got {drop_index}")

    canonical = list(passages[:pool_size])
    retained = canonical[:drop_index] + canonical[drop_index + 1 :]
    return {
        "canonical": canonical,
        "replace": retained + [passages[pool_size]],
        "drop": retained,
    }


def restrict_ranking(ranking: list[str], retained_ids: set[str]) -> list[str]:
    """Restrict a ranking to retained IDs and require exact coverage."""
    restricted = [str(pid) for pid in ranking if str(pid) in retained_ids]
    if len(restricted) != len(retained_ids) or set(restricted) != retained_ids:
        missing = sorted(retained_ids - set(restricted))
        raise ValueError(
            "reranker output does not cover every retained document "
            f"(expected={len(retained_ids)}, got={len(restricted)}, missing={missing[:5]})"
        )
    return restricted


def evaluate_pool_pair(
    canonical_ranking: list[str],
    perturbed_ranking: list[str],
    *,
    retained_ids: set[str],
    qrels_for_q: dict[str, int] | None,
    k: int = 10,
) -> dict[str, Any]:
    """Evaluate one canonical/perturbed pair on retained documents only."""
    canonical = restrict_ranking(canonical_ranking, retained_ids)
    perturbed = restrict_ranking(perturbed_ranking, retained_ids)
    tau = kendall_tau(canonical, perturbed)
    pool_psi = float(np.clip((1.0 - tau) / 2.0, 0.0, 1.0))

    top_k_set_changed = set(canonical[:k]) != set(perturbed[:k])
    out: dict[str, Any] = {
        "n_retained": len(retained_ids),
        "kendall_tau": tau,
        "pool_psi": pool_psi,
        "top_k_set_changed": top_k_set_changed,
        "canonical_top_k": canonical[:k],
        "perturbed_top_k": perturbed[:k],
        "canonical_ndcg_at_k_retained": None,
        "perturbed_ndcg_at_k_retained": None,
        "delta_ndcg_at_k_retained": None,
    }
    if qrels_for_q is not None:
        retained_qrels = {pid: int(qrels_for_q.get(pid, 0)) for pid in retained_ids}
        q0 = ndcg_at_k(canonical, retained_qrels, k=k)
        q1 = ndcg_at_k(perturbed, retained_qrels, k=k)
        out.update(
            {
                "canonical_ndcg_at_k_retained": q0,
                "perturbed_ndcg_at_k_retained": q1,
                "delta_ndcg_at_k_retained": q1 - q0,
            }
        )
    return out


def aggregate_pool_metrics(per_query: dict[str, dict], perturbation: str) -> dict[str, Any]:
    """Aggregate one perturbation over queries."""
    rows = [row[perturbation] for row in per_query.values()]
    if not rows:
        return {"n_queries": 0}

    def summarize(key: str) -> tuple[float | None, float | None]:
        vals = [float(row[key]) for row in rows if row.get(key) is not None]
        if not vals:
            return None, None
        return float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

    mean_tau, std_tau = summarize("kendall_tau")
    mean_psi, std_psi = summarize("pool_psi")
    mean_q0, std_q0 = summarize("canonical_ndcg_at_k_retained")
    mean_q1, std_q1 = summarize("perturbed_ndcg_at_k_retained")
    mean_dq, std_dq = summarize("delta_ndcg_at_k_retained")
    abs_dq = [
        abs(float(row["delta_ndcg_at_k_retained"])) for row in rows if row.get("delta_ndcg_at_k_retained") is not None
    ]
    return {
        "n_queries": len(rows),
        "mean_kendall_tau": mean_tau,
        "std_kendall_tau": std_tau,
        "mean_pool_psi": mean_psi,
        "std_pool_psi": std_psi,
        "top_k_set_flip_rate": float(np.mean([bool(row["top_k_set_changed"]) for row in rows])),
        "n_top_k_set_flips": sum(bool(row["top_k_set_changed"]) for row in rows),
        "mean_canonical_ndcg_at_k_retained": mean_q0,
        "std_canonical_ndcg_at_k_retained": std_q0,
        "mean_perturbed_ndcg_at_k_retained": mean_q1,
        "std_perturbed_ndcg_at_k_retained": std_q1,
        "mean_delta_ndcg_at_k_retained": mean_dq,
        "std_delta_ndcg_at_k_retained": std_dq,
        "mean_abs_delta_ndcg_at_k_retained": float(np.mean(abs_dq)) if abs_dq else None,
    }


def _ranking_from_result(result: dict) -> list[str]:
    return [str(p["pid"]) for p in (result.get("top_k_psgs") or [])]


def _load_qid_allowlist(block: dict) -> set[str]:
    qids = {str(qid) for qid in (block.get("qids_to_run") or [])}
    path = block.get("qids_to_run_path")
    if path:
        with open(path, "r", encoding="utf-8") as f:
            qids.update(line.strip() for line in f if line.strip())
    return qids


class PoolPerturbationRunner:
    """Run canonical/replace/drop scoring and retained-set metrics."""

    def __init__(
        self,
        config_path: str | Path,
        runs_root: str | Path = "runs",
        run_dir: str | Path | None = None,
        reranker: object | None = None,
    ):
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")
        self.config: dict = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        self.exp_id = str(self.config["id"])
        self.data_config = self.config.get("data") or {}
        self.pool_cfg = self.config.get("pool_perturbation") or {}
        if not self.pool_cfg:
            raise ValueError("config.pool_perturbation is required")

        self.pool_size = int(self.pool_cfg.get("pool_size", 100))
        self.source_depth = int(self.pool_cfg.get("source_depth", self.pool_size + 1))
        if self.source_depth < self.pool_size + 1:
            raise ValueError("pool_perturbation.source_depth must be at least pool_size + 1")
        self.seed = int(self.pool_cfg.get("seed", 0))
        self.query_sample_seed = int(self.pool_cfg.get("query_sample_seed", 0))
        self.max_queries = int(self.pool_cfg.get("max_queries", 100))
        if self.max_queries < 1:
            raise ValueError("pool_perturbation.max_queries must be positive")
        self.k_cutoff = int(self.pool_cfg.get("k_cutoff_for_ndcg", 10))
        self.perturbations = list(self.pool_cfg.get("perturbations") or PERTURBATIONS)
        unknown = sorted(set(self.perturbations) - set(PERTURBATIONS))
        if unknown:
            raise ValueError(f"unknown pool perturbation(s): {unknown}; known={list(PERTURBATIONS)}")

        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.run_dir = Path(run_dir) if run_dir is not None else Path(runs_root) / self.exp_id / ts
        self.output_dir = self.run_dir / "pool_perturbation"
        self.results_dir = self.output_dir / "per_query_results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logging(
            self.__class__.__name__,
            self.config,
            output_file=str(self.run_dir / "pool_perturbation.log"),
        )

        if reranker is None:
            self.reranker = instantiate_reranker(self.config, self.logger)
        else:
            from presentation_dependence.eval.experiment_manager import _apply_prompt_overrides

            self.reranker = reranker
            _apply_prompt_overrides(
                self.reranker,
                self.config.get("reranker") or {},
                self.logger,
            )
        self.dataloader = instantiate_dataloader(self.config, self.data_config)
        qrels_path = (self.config.get("eval") or {}).get("qrels_path")
        self.qrels = load_qrels(qrels_path) if qrels_path else {}
        if run_dir is None:
            write_resolved_run_config(
                self.run_dir,
                self.config,
                config_path=self.config_path,
                ts=ts,
                data_config=self.data_config,
            )

    def _select_queries(self, queries: list[dict], run_path: str) -> list[tuple[dict, list[dict]]]:
        allowlist = _load_qid_allowlist(self.pool_cfg)
        eligible: list[tuple[int, str, dict, list[dict]]] = []
        for query in queries:
            qid = str(query["qid"])
            if allowlist and qid not in allowlist:
                continue
            passages = self.dataloader.get_psgs_from_run(run_path, qid) or []
            if len(passages) < self.source_depth:
                continue
            sample_key = stable_seed(self.query_sample_seed, qid, "query-sample")
            eligible.append((sample_key, qid, query, passages[: self.source_depth]))
        eligible.sort(key=lambda row: (row[0], row[1]))
        if len(eligible) < self.max_queries:
            raise ValueError(
                f"pool perturbation requires {self.max_queries} queries with at least "
                f"{self.source_depth} candidates; found {len(eligible)}"
            )
        selected = eligible[: self.max_queries]
        return [(query, passages) for _key, _qid, query, passages in selected]

    def _query_dir(self, qid: str) -> Path:
        return self.results_dir / qid_to_dirname(qid)

    def _load_completed(self, qid: str) -> dict | None:
        path = self._query_dir(qid) / "metrics.json"
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def _write_variant(self, qid: str, variant: str, result: dict) -> None:
        vdir = self._query_dir(qid) / variant
        vdir.mkdir(parents=True, exist_ok=True)
        (vdir / "detailed_results.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        write_trec_run(
            vdir / "trec_results_raw.txt",
            qid=qid,
            ranked_docs=result["top_k_psgs"],
            tag=f"presentation_dependence:pool:{self.exp_id}:{variant}",
        )

    def _rank_variants(self, query: dict, pools: dict[str, list[dict]]) -> dict[str, dict]:
        names = ["canonical", *self.perturbations]
        items = [(query, pools[name]) for name in names]
        if getattr(self.reranker, "supports_query_batching", False):
            results = self.reranker.rank_query_batch(items)
        else:
            results = [self.reranker.rank(q, passages) for q, passages in items]
        if len(results) != len(names):
            raise RuntimeError(f"reranker returned {len(results)} results for {len(names)} pool variants")
        out = dict(zip(names, results, strict=True))
        for name in names:
            validate_rank_result_if_enabled(self.config, out[name], pools[name])
            if not out[name].get("top_k_psgs"):
                raise ValueError(f"empty reranker output for variant={name}")
        return out

    def run(self) -> dict:
        run_path = str(self.data_config["run_path"])
        queries = self.dataloader.get_qs_from_run(run_path)
        selected = self._select_queries(queries, run_path)
        selected_qids = [str(query["qid"]) for query, _passages in selected]
        (self.output_dir / "selected_qids.txt").write_text(
            "".join(f"{qid}\n" for qid in selected_qids),
            encoding="utf-8",
        )

        per_query: dict[str, dict] = {}
        failed_qids: list[str] = []
        write_progress_jsonl, progress_heartbeat_s = progress_config(self.config)
        progress = ProgressTracker(
            phase="pool_perturbation",
            total_work=len(selected),
            total_queries=len(selected),
            jsonl_path=self.output_dir / "progress.jsonl",
            write_jsonl=write_progress_jsonl,
            heartbeat_every_s=progress_heartbeat_s,
        )
        for q_index, (query, passages) in enumerate(selected, start=1):
            qid = str(query["qid"])
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
            self.logger.info("Pool perturbation qid=%s (%s)", qid, suffix)
            t0 = time.monotonic()
            try:
                drop_index = choose_drop_index(qid, seed=self.seed, pool_size=self.pool_size)
                pools = build_fixed_order_pools(
                    passages,
                    drop_index=drop_index,
                    pool_size=self.pool_size,
                )
                results = self._rank_variants(query, pools)
                qdir = self._query_dir(qid)
                qdir.mkdir(parents=True, exist_ok=True)
                for name, result in results.items():
                    self._write_variant(qid, name, result)

                dropped_pid = str(pools["canonical"][drop_index]["pid"])
                retained_ids = {str(p["pid"]) for i, p in enumerate(pools["canonical"]) if i != drop_index}
                canonical_ranking = _ranking_from_result(results["canonical"])
                row: dict[str, Any] = {
                    "qid": qid,
                    "drop_rank_1based": drop_index + 1,
                    "dropped_pid": dropped_pid,
                    "replacement_pid": str(passages[self.pool_size]["pid"]),
                    "canonical_input_pids": [str(p["pid"]) for p in pools["canonical"]],
                }
                for perturbation in self.perturbations:
                    row[perturbation] = evaluate_pool_pair(
                        canonical_ranking,
                        _ranking_from_result(results[perturbation]),
                        retained_ids=retained_ids,
                        qrels_for_q=self.qrels.get(qid),
                        k=self.k_cutoff,
                    )
                (qdir / "metrics.json").write_text(
                    json.dumps(row, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
                per_query[qid] = row
                progress.finish_unit(
                    label=label,
                    work_index=work_index,
                    q_index=q_index,
                    duration_s=time.monotonic() - t0,
                    success=True,
                    extra={"qid": qid, "drop_rank_1based": drop_index + 1},
                )
                progress.maybe_heartbeat(self.logger)
            except Exception as exc:  # noqa: BLE001 - finish other qids, refuse partial aggregate
                self.logger.error("Failed pool perturbation qid=%s: %s", qid, exc, exc_info=True)
                failed_qids.append(qid)
                progress.finish_unit(
                    label=label,
                    work_index=work_index,
                    q_index=q_index,
                    duration_s=time.monotonic() - t0,
                    success=False,
                    extra={"qid": qid, "reason": type(exc).__name__},
                )

        if failed_qids:
            raise partial_run_error(
                failed_qids,
                phase="pool perturbation",
                aggregate_label="aggregate pool-perturbation metrics",
            )

        aggregate = {
            perturbation: aggregate_pool_metrics(per_query, perturbation) for perturbation in self.perturbations
        }
        metrics = {
            "schema_version": 1,
            "exp_id": self.exp_id,
            "aggregate": aggregate,
            "protocol": {
                "id": "fixed-order-pool-perturbation-v1",
                "pool_size": self.pool_size,
                "source_depth": self.source_depth,
                "canonical_order": "first-stage rank order, unchanged",
                "perturbations": self.perturbations,
                "drop_rank_sampling": "uniform rank in 1..pool_size; deterministic SHA256 seed per qid",
                "seed": self.seed,
                "query_sampling": "deterministic SHA256 ordering over eligible qids",
                "query_sample_seed": self.query_sample_seed,
                "max_queries": self.max_queries,
                "selected_qids_path": "pool_perturbation/selected_qids.txt",
                "metric_scope": "both rankings restricted to the pool_size-1 retained documents",
                "pool_psi_formula": "(1 - Kendall tau_a) / 2",
                "top_k": self.k_cutoff,
                "ndcg_gain": "linear; qrels and ideal ranking restricted to retained documents",
                "temperature": 0,
            },
            "b1_reference": {
                "mean_pool_psi": 0.0,
                "note": "At B=1 every document is scored alone, so another document cannot change its score.",
                "computed": False,
            },
        }
        (self.output_dir / "pool_metrics.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (self.output_dir / "pool_per_query.json").write_text(
            json.dumps(per_query, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        self.logger.info("Wrote %s", self.output_dir / "pool_metrics.json")
        return metrics
