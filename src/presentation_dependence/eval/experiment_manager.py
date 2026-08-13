"""Experiment runner: load a reranker, iterate queries, write per-query outputs.

Each experiment uses one configured reranker and writes a fixed output layout,
so any consumer of the run tree can read it without knowing which reranker
produced it.

Run layout:

    runs/<ID>/<timestamp>/
        resolved_config.yaml
        experiment.log
        per_query_results/<qid>/
            detailed_results.json
            trec_results_raw.txt
"""

from __future__ import annotations

import json
import math
import time
from datetime import datetime
from pathlib import Path

import yaml

from presentation_dependence.eval.runner_setup import instantiate_dataloader, instantiate_reranker, partial_run_error
from presentation_dependence.eval.runner_setup import validate_rank_result_if_enabled
from presentation_dependence.eval.runner_setup import check_resume_fingerprint, run_fingerprint
from presentation_dependence.eval.runner_setup import write_resolved_run_config
from presentation_dependence.utils.progress import ProgressTracker, progress_config
from presentation_dependence.utils.setup_logging import setup_logging
from presentation_dependence.utils.trec import dirname_to_qid, qid_to_dirname, write_trec_run


class ExperimentManager:
    """Drives a single experiment from a `configs/experiments/<ID>.yaml` file."""

    def __init__(
        self,
        config_path: str | Path,
        runs_root: str | Path = "runs",
        run_dir: str | Path | None = None,
        reranker=None,
    ):
        # Callers may inject a loaded reranker for reuse across surfaces. Prompt
        # settings from this config are still applied, allowing one model to use
        # different web and legal instructions. Without an injected instance,
        # the runner constructs the reranker from ``config.reranker``.
        self._injected_reranker = reranker
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")

        with open(self.config_path, "r") as f:
            self.config: dict = yaml.safe_load(f)

        self.exp_id = self.config["id"]
        self.runs_root = Path(runs_root)

        # Each invocation normally gets its own timestamped run dir. Reruns of
        # the same config are additive; existing per-query results are reused
        # (see ``_load_existing_results``). Experiment IDs remain stable across
        # config changes. Microsecond precision avoids collisions when
        # two invocations land in the same second (CI, tests, tight sweeps).
        #
        # When ``run_dir`` is supplied, reuse it instead of creating a new one.
        # This is the resume path: a re-launch (after a crash, laptop sleep,
        # or credential expiry) points back at the same dir so the per-query
        # files already on disk are skipped and only missing qids are
        # re-ranked. This avoids repeating paid API calls for completed queries.
        # It mirrors PsiExperimentRunner's ``run_dir`` override.
        if run_dir is not None:
            self.run_dir = Path(run_dir)
            ts = self.run_dir.name
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            self.run_dir = self.runs_root / self.exp_id / ts
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.logger = setup_logging(
            self.__class__.__name__,
            self.config,
            output_file=str(self.run_dir / "experiment.log"),
        )

        self.data_config = self.config.get("data", {})
        self.results_dir = self.run_dir / "per_query_results"
        self.results_dir.mkdir(exist_ok=True)

        # Cross-query batching sends up to N single-pass queries to one engine
        # call. Rerankers that do not implement rank_query_batch continue to run
        # per query. Both paths construct the same prefixes and scores; only the
        # engine-call count changes.
        eval_cfg = self.config.get("eval") or {}
        _qbs = eval_cfg.get("query_batch_size", 16)
        self.query_batch_size = int(_qbs if _qbs is not None else 16)
        if self.query_batch_size < 1:
            raise ValueError("eval.query_batch_size must be >= 1")

        self._load_reranker()
        self._load_dataloader()

        # Compare against the configuration that wrote any existing per-query
        # results before this call overwrites resolved_config.yaml.
        check_resume_fingerprint(
            self.run_dir,
            run_fingerprint(self.config, self.data_config),
            self.logger,
        )
        write_resolved_run_config(
            self.run_dir,
            self.config,
            config_path=self.config_path,
            ts=ts,
            data_config=self.data_config,
        )

    def _load_reranker(self) -> None:
        if self._injected_reranker is not None:
            self.reranker = self._injected_reranker
            self.logger.info("Reusing injected reranker %s", type(self.reranker).__name__)
            _apply_prompt_overrides(self.reranker, self.config.get("reranker", {}), self.logger)
            return
        self.reranker = instantiate_reranker(self.config, self.logger)

    def _load_dataloader(self) -> None:
        self.dataloader = instantiate_dataloader(self.config, self.data_config)

    def run(self) -> None:  # noqa: C901
        self.logger.info("Starting experiment %s (run_dir=%s)", self.exp_id, self.run_dir)

        run_path = self.data_config["run_path"]
        queries = self.dataloader.get_qs_from_run(run_path)
        self.logger.info("Loaded %d queries from %s", len(queries), run_path)

        existing = self._load_existing_results()
        qids_allowlist = set(map(str, self.config.get("qids_to_run", []) or []))
        qids_to_run_path = self.config.get("qids_to_run_path")
        if qids_to_run_path:
            with open(qids_to_run_path, "r", encoding="utf-8") as f:
                qids_allowlist.update(line.strip() for line in f if line.strip())
        k_input = (
            int(self.data_config.get("k_input", math.inf) or math.inf)
            if self.data_config.get("k_input") is not None
            else math.inf
        )
        failed_qids: list[str] = []
        attempted_qids: list[str] = []
        work_queries = [
            query
            for query in queries
            if str(query["qid"]) not in existing and (not qids_allowlist or str(query["qid"]) in qids_allowlist)
        ]
        write_progress_jsonl, progress_heartbeat_s = progress_config(self.config)
        progress = ProgressTracker(
            phase="eval",
            total_work=len(work_queries),
            total_queries=len(work_queries),
            jsonl_path=self.run_dir / "progress.jsonl",
            write_jsonl=write_progress_jsonl,
            heartbeat_every_s=progress_heartbeat_s,
        )

        use_query_batch = self.query_batch_size > 1 and getattr(self.reranker, "supports_query_batching", False)
        if use_query_batch:
            self._run_query_batched(work_queries, run_path, k_input, progress, attempted_qids, failed_qids)

        work_q_index = 0
        for query in [] if use_query_batch else queries:
            qid = str(query["qid"])
            if qid in existing:
                self.logger.info("Results already present for qid=%s", qid)
                continue
            if qids_allowlist and qid not in qids_allowlist:
                continue

            active_label: str | None = None
            active_work_index: int | None = None
            try:
                attempted_qids.append(qid)
                work_q_index += 1
                active_label = f"qid={qid}"
                active_work_index, suffix = progress.start_unit(
                    label=active_label,
                    q_index=work_q_index,
                    extra={"qid": qid},
                )
                self.logger.info("Ranking qid=%s (%s)", qid, suffix)
                passages = self.dataloader.get_psgs_from_run(run_path, qid) or []
                if k_input != math.inf:
                    passages = passages[: int(k_input)]

                t0 = time.monotonic()
                result = self.reranker.rank(query, passages)
                validate_rank_result_if_enabled(self.config, result, passages)

                if not result.get("top_k_psgs"):
                    self.logger.error("Empty top_k_psgs for qid=%s; skipping write", qid)
                    failed_qids.append(qid)
                    progress.finish_unit(
                        label=active_label,
                        work_index=active_work_index,
                        q_index=work_q_index,
                        duration_s=time.monotonic() - t0,
                        success=False,
                        extra={"qid": qid, "reason": "empty_top_k_psgs"},
                    )
                    continue

                self._write_query_result(qid, result)
                progress.finish_unit(
                    label=active_label,
                    work_index=active_work_index,
                    q_index=work_q_index,
                    duration_s=time.monotonic() - t0,
                    success=True,
                    extra={"qid": qid},
                )
                active_label = None
                active_work_index = None
                progress.maybe_heartbeat(self.logger)
            except Exception as e:
                self.logger.error("Failed to rank qid=%s: %s", qid, e, exc_info=True)
                failed_qids.append(qid)
                if active_label is not None and active_work_index is not None:
                    progress.finish_unit(
                        label=active_label,
                        work_index=active_work_index,
                        q_index=work_q_index,
                        success=False,
                        extra={"qid": qid, "reason": type(e).__name__},
                    )

        succeeded = len(attempted_qids) - len(failed_qids)
        self.logger.info(
            "Experiment %s finished: attempted=%d succeeded=%d failed=%d",
            self.exp_id,
            len(attempted_qids),
            succeeded,
            len(failed_qids),
        )
        if failed_qids:
            raise partial_run_error(failed_qids, phase="reranking", aggregate_label="aggregate metrics")

    def _record_query_result(
        self,
        qid: str,
        result: dict,
        passages: list,
        progress,
        label: str,
        work_index: int,
        q_index: int,
        duration_s: float,
        failed_qids: list[str],
    ) -> None:
        """Validate and write one query result, then close its progress unit.

        Shared by the per-query and batched paths so both apply identical
        validation, empty-output handling, and artefact writing.
        """
        validate_rank_result_if_enabled(self.config, result, passages)
        if not result.get("top_k_psgs"):
            self.logger.error("Empty top_k_psgs for qid=%s; skipping write", qid)
            failed_qids.append(qid)
            progress.finish_unit(
                label=label,
                work_index=work_index,
                q_index=q_index,
                duration_s=duration_s,
                success=False,
                extra={"qid": qid, "reason": "empty_top_k_psgs"},
            )
            return
        self._write_query_result(qid, result)
        progress.finish_unit(
            label=label,
            work_index=work_index,
            q_index=q_index,
            duration_s=duration_s,
            success=True,
            extra={"qid": qid},
        )

    def _run_query_batched(  # noqa: C901
        self,
        work_queries: list,
        run_path: str,
        k_input: float,
        progress,
        attempted_qids: list[str],
        failed_qids: list[str],
    ) -> None:
        """Score ``work_queries`` in cross-query batches of ``query_batch_size``.

        Each batch builds ``(query, passages)`` items and sends them in one
        ``rank_query_batch`` engine call, then writes each result via
        :meth:`_record_query_result`. If the batched call fails, the method
        retries each query with ``rank()``. It is called only when
        ``query_batch_size > 1`` and the reranker supports batching.
        """
        n = self.query_batch_size
        work_q_index = 0
        for start in range(0, len(work_queries), n):
            chunk = work_queries[start : start + n]
            items: list[tuple[dict, list]] = []
            metas: list[tuple[str, list, int, int, str]] = []
            for query in chunk:
                qid = str(query["qid"])
                attempted_qids.append(qid)
                work_q_index += 1
                label = f"qid={qid}"
                work_index, _suffix = progress.start_unit(label=label, q_index=work_q_index, extra={"qid": qid})
                passages = self.dataloader.get_psgs_from_run(run_path, qid) or []
                if k_input != math.inf:
                    passages = passages[: int(k_input)]
                items.append((query, passages))
                metas.append((qid, passages, work_q_index, work_index, label))

            # Emit progress, ETA, and GPU status before the blocking batch call.
            # Per-query completion lines appear only after the call returns.
            self.logger.info("Ranking batch of %d queries (qids=%s)", len(items), [m[0] for m in metas])
            progress.maybe_heartbeat(self.logger, force=True)
            t0 = time.monotonic()
            results = None
            try:
                results = self.reranker.rank_query_batch(items)
                if len(results) != len(items):
                    raise RuntimeError(f"rank_query_batch returned {len(results)} results for {len(items)} items")
            except Exception as e:
                self.logger.error(
                    "Batched rank failed (qids=%s): %s; falling back to per-query",
                    [m[0] for m in metas],
                    e,
                    exc_info=True,
                )
                results = None

            if results is None:
                for (query, passages), (qid, _p, wqi, wi, label) in zip(items, metas):
                    ti = time.monotonic()
                    try:
                        res = self.reranker.rank(query, passages)
                        self._record_query_result(
                            qid, res, passages, progress, label, wi, wqi, time.monotonic() - ti, failed_qids
                        )
                    except Exception as e2:
                        self.logger.error("Failed to rank qid=%s: %s", qid, e2, exc_info=True)
                        failed_qids.append(qid)
                        progress.finish_unit(
                            label=label,
                            work_index=wi,
                            q_index=wqi,
                            success=False,
                            extra={"qid": qid, "reason": type(e2).__name__},
                        )
                continue

            per_item_dt = (time.monotonic() - t0) / max(len(items), 1)
            for result, (qid, passages, wqi, wi, label) in zip(results, metas):
                try:
                    self._record_query_result(qid, result, passages, progress, label, wi, wqi, per_item_dt, failed_qids)
                except Exception as e2:
                    self.logger.error("Failed handling batched result qid=%s: %s", qid, e2, exc_info=True)
                    failed_qids.append(qid)
                    progress.finish_unit(
                        label=label,
                        work_index=wi,
                        q_index=wqi,
                        success=False,
                        extra={"qid": qid, "reason": type(e2).__name__},
                    )
            progress.maybe_heartbeat(self.logger)

    def _load_existing_results(self) -> set[str]:
        """Qids whose detailed_results.json + trec_results_raw.txt both exist."""
        done: set[str] = set()
        if not self.results_dir.exists():
            return done
        for qid_dir in self.results_dir.iterdir():
            if not qid_dir.is_dir():
                continue
            if (qid_dir / "detailed_results.json").exists() and (qid_dir / "trec_results_raw.txt").exists():
                # dir name may have `/` escaped (see utils.trec); reverse
                # so the returned set matches the raw qid strings our
                # loaders emit.
                done.add(dirname_to_qid(qid_dir.name))
        return done

    def _write_query_result(self, qid: str, result: dict) -> None:
        qdir = self.results_dir / qid_to_dirname(qid)
        qdir.mkdir(exist_ok=True)

        with open(qdir / "detailed_results.json", "w") as f:
            json.dump(result, f, indent=2)

        write_trec_run(
            qdir / "trec_results_raw.txt",
            qid=qid,
            ranked_docs=result["top_k_psgs"],
            tag=f"presentation_dependence:{self.exp_id}",
        )


# These per-surface attributes may differ while a bundle reuses the same model
# and engine. An override is applied only when the config supplies a value and
# the reranker exposes the corresponding attribute.
_PROMPT_OVERRIDE_ATTRS = ("instruction", "max_doc_chars")


def _apply_prompt_overrides(reranker, reranker_cfg: dict, logger=None) -> None:
    for attr in _PROMPT_OVERRIDE_ATTRS:
        if attr not in reranker_cfg:
            continue
        if not hasattr(reranker, attr):
            continue
        value = reranker_cfg[attr]
        if getattr(reranker, attr) != value:
            setattr(reranker, attr, value)
            if logger is not None:
                logger.info("Bundle: set reranker.%s=%r for this surface", attr, value)

    # Shared-base bundles select this surface's LoRA adapter, or the base model
    # when lora_path is absent. Single-model bundles remain unchanged.
    setter = getattr(reranker, "set_active_adapter", None)
    if setter is not None:
        target = reranker_cfg.get("lora_path")
        if getattr(reranker, "active_adapter", None) != (str(target) if target else None):
            setter(target)
            if logger is not None:
                logger.info("Bundle: set active LoRA adapter=%r for this surface", target)
