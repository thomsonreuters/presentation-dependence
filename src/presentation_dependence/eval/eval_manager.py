"""Evaluate per-query TREC runs with pytrec_eval and write ``metrics.json``.

Robustness metrics are computed separately by the PSI runner.

Inputs:
    run_dir/resolved_config.yaml   -- carries eval.qrels_path, eval.measures
    run_dir/per_query_results/<qid>/trec_results_raw.txt
Outputs:
    run_dir/per_query_results/<qid>/{trec_results_deduplicated.txt, eval_results.jsonl}
    run_dir/all_queries_eval_results.jsonl
    run_dir/metrics.json   -- canonical headline {mean,std} values per measure
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytrec_eval  # type: ignore
import yaml

from presentation_dependence.utils.setup_logging import setup_logging
from presentation_dependence.utils.trec import dirname_to_qid, qid_to_dirname


class EvalManager:
    def __init__(self, run_dir: str | Path, skip_existing: bool = False):
        self.run_dir = Path(run_dir)
        if not self.run_dir.is_dir():
            raise FileNotFoundError(f"Run dir not found: {self.run_dir}")

        resolved_cfg_path = self.run_dir / "resolved_config.yaml"
        if not resolved_cfg_path.exists():
            raise FileNotFoundError(
                f"{resolved_cfg_path} not found; this does not look like a run directory "
                "produced by presentation_dependence.eval.experiment_manager."
            )

        with open(resolved_cfg_path, "r") as f:
            self.config: dict = yaml.safe_load(f)

        self.eval_config = self.config.get("eval", {})
        self.qrels_path = self.eval_config.get("qrels_path")
        if not self.qrels_path:
            raise ValueError("config.eval.qrels_path is required for evaluation")

        self.measures = list(self.eval_config.get("measures") or pytrec_eval.supported_measures)
        self.skip_existing = skip_existing

        self.logger = setup_logging(
            self.__class__.__name__,
            self.config,
            output_file=str(self.run_dir / "evaluation.log"),
        )

        with open(self.qrels_path, "r") as f:
            self.qrels = pytrec_eval.parse_qrel(f)

        self.per_query_dir = self.run_dir / "per_query_results"
        self.all_results_path = self.run_dir / "all_queries_eval_results.jsonl"
        self.metrics_path = self.run_dir / "metrics.json"

        # Every query this run declined to score, and why. Aggregation happens
        # over whatever remains, so without this record a headline mean can
        # silently cover fewer queries than the run intended.
        self._skipped: list[dict[str, str]] = []
        self._malformed_lines = 0
        self._query_dirs_found = 0

    def _skip(self, qid: str, reason: str) -> None:
        self._skipped.append({"qid": qid, "reason": reason})

    def _coverage(self, evaluated: int) -> dict:
        """Describe how much of the intended query set the metrics cover."""
        return {
            "complete": not self._skipped and self._malformed_lines == 0,
            "query_dirs_found": self._query_dirs_found,
            "queries_evaluated": evaluated,
            "queries_skipped": len(self._skipped),
            "malformed_lines_dropped": self._malformed_lines,
            "skipped": self._skipped,
        }

    def _log_coverage(self, coverage: dict) -> None:
        if coverage["complete"]:
            return
        by_reason: dict[str, int] = {}
        for row in self._skipped:
            by_reason[row["reason"]] = by_reason.get(row["reason"], 0) + 1
        self.logger.warning(
            "Incomplete evaluation: %d/%d query directories scored; skipped %s; "
            "%d malformed TREC line(s) dropped. metrics.json aggregates only the "
            "queries that were scored; see coverage.skipped for the rest.",
            coverage["queries_evaluated"],
            coverage["query_dirs_found"],
            by_reason or "none",
            self._malformed_lines,
        )

    def run(self) -> dict:  # noqa: C901
        if self.skip_existing and self.metrics_path.exists():
            self.logger.info("Skipping eval: %s already exists", self.metrics_path)
            return json.loads(self.metrics_path.read_text())

        if not self.per_query_dir.is_dir():
            raise FileNotFoundError(f"{self.per_query_dir} not found; run the experiment first.")

        # Dedup each query's raw TREC run, write trec_results_deduplicated.txt,
        # and collect every qid's lines into a single combined list.
        combined_lines: list[str] = []
        eligible_qids: list[str] = []
        for qdir in sorted(self.per_query_dir.iterdir()):
            if not qdir.is_dir():
                continue
            self._query_dirs_found += 1
            try:
                dedup_lines = self._dedup_query_run_file(qdir)
            except Exception as e:
                self.logger.error("Error reading TREC run for %s: %s", qdir.name, e, exc_info=True)
                self._skip(dirname_to_qid(qdir.name), f"unreadable:{type(e).__name__}")
                continue
            if dedup_lines is None:
                self._skip(dirname_to_qid(qdir.name), "missing_or_empty_trec_run")
                continue
            combined_lines.extend(dedup_lines)
            # The directory name may escape the raw qid, because a qid can
            # contain `/`. The TREC
            # file content uses the raw qid, so `pytrec_eval.parse_run`
            # will produce a `per_query` dict keyed by the raw qid. Store
            # the raw qid here so downstream lookups match.
            eligible_qids.append(dirname_to_qid(qdir.name))

        if not combined_lines:
            self.logger.warning("No per-query TREC runs found under %s", self.per_query_dir)
            return self._write_empty_metrics()

        # pytrec_eval retains all requested `ndcg_cut` values only when
        # evaluate() receives one multi-query run dict. Calling evaluate()
        # per-query silently drops shorter cutoffs after the first call,
        # yielding `mean_ndcg_cut_5` computed over a single row with
        # `std_ndcg_cut_5 = 0.0`. Evaluate every cutoff in one call.
        evaluator = pytrec_eval.RelevanceEvaluator(self.qrels, set(self.measures))
        results = pytrec_eval.parse_run(combined_lines)
        per_query = evaluator.evaluate(results)

        all_rows: list[dict] = []
        for qid in eligible_qids:
            metrics_for_qid = per_query.get(qid, {})
            if not metrics_for_qid:
                self.logger.warning("No metrics returned for qid=%s", qid)
                self._skip(qid, "no_metrics_returned")
                continue
            row = {"qid": qid, **metrics_for_qid}
            all_rows.append(row)
            # Use the escaped form to locate the on-disk dir; qid itself
            # (with possible `/`) stays unchanged inside the JSONL row.
            (self.per_query_dir / qid_to_dirname(qid) / "eval_results.jsonl").write_text(json.dumps(row) + "\n")

        with open(self.all_results_path, "w") as f:
            for row in all_rows:
                f.write(json.dumps(row) + "\n")

        metrics = self._aggregate(all_rows)
        metrics["coverage"] = self._coverage(len(all_rows))
        self._log_coverage(metrics["coverage"])
        with open(self.metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        self.logger.info("Wrote %s", self.metrics_path)
        return metrics

    def _dedup_query_run_file(self, qdir: Path) -> list[str] | None:
        """Deduplicate one query's TREC run by document ID.

        Writes ``trec_results_deduplicated.txt`` and returns newline-terminated
        rows. Returns ``None`` when the raw file is missing or empty.
        """
        raw_path = qdir / "trec_results_raw.txt"
        dedup_path = qdir / "trec_results_deduplicated.txt"

        if not raw_path.exists() or raw_path.stat().st_size == 0:
            self.logger.warning("Empty/missing TREC run for qid=%s; skipping", qdir.name)
            return None

        with open(raw_path, "r") as f:
            lines = f.readlines()

        seen: set[str] = set()
        deduped: list[str] = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 6:
                # A dropped line truncates this query's ranking, which changes
                # its metrics rather than merely omitting it. Counted so the
                # coverage block can flag the run.
                if line.strip():
                    self._malformed_lines += 1
                continue
            docid = parts[2]
            if docid in seen:
                continue
            seen.add(docid)
            deduped.append(line if line.endswith("\n") else line + "\n")

        dedup_path.write_text("".join(deduped))
        return deduped

    def _write_empty_metrics(self) -> dict:
        metrics = {"n_queries": 0, "exp_id": self.config.get("id")}
        metrics["coverage"] = self._coverage(evaluated=0)
        self._log_coverage(metrics["coverage"])
        with open(self.metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        return metrics

    def _aggregate(self, rows: list[dict]) -> dict:
        if not rows:
            return {"n_queries": 0}

        measures_present = set()
        for row in rows:
            measures_present.update(k for k in row if k != "qid")

        agg: dict = {"n_queries": len(rows), "exp_id": self.config.get("id")}
        for m in sorted(measures_present):
            values = [float(r[m]) for r in rows if m in r]
            if not values:
                continue
            agg[f"mean_{m}"] = float(np.mean(values))
            agg[f"std_{m}"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        return agg
