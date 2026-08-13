"""Silver-label generation orchestrator.

The orchestrator is model-agnostic: it builds pointwise B=1 requests from the
configured data loader, hands them to a `SilverClient`, parses the returned
teacher output with the configured prompt template, and writes resumable JSONL
artifacts.
"""

from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml

from presentation_dependence.eval.runner_setup import instantiate_dataloader
from presentation_dependence.silver_data.client import SilverClient, SilverRequest
from presentation_dependence.silver_data.openai_client import OpenAICompatibleClient
from presentation_dependence.silver_data.prompts import get_prompt_template
from presentation_dependence.utils.setup_logging import setup_logging


ResumeKey = tuple[str, str, str, str]


class BudgetExceeded(RuntimeError):
    """Raised when a run crosses `cost.max_budget_usd`."""


class RunDirLocked(RuntimeError):
    """Raised when another `SilverGenerator` already owns this `run_dir`."""


@dataclass(frozen=True)
class WorkItem:
    query_id: str
    doc_id: str
    query: str
    document: str
    prompt_template_id: str


class SilverGenerator:
    def __init__(
        self,
        config: dict[str, Any],
        *,
        config_path: str | Path | None = None,
        client: SilverClient | None = None,
        run_dir: str | Path | None = None,
    ) -> None:
        self.config = config
        self.config_path = Path(config_path) if config_path is not None else None
        self.experiment_id = str(config.get("experiment_id") or config.get("id") or "silver-run")
        self.teacher = config.get("teacher") or {}
        self.teacher_model_id = str(self.teacher.get("model_id") or "")
        if not self.teacher_model_id:
            raise ValueError("teacher.model_id is required")
        self.prompt_ids = list(config.get("prompts") or [])
        if not self.prompt_ids:
            raise ValueError("prompts must contain at least one prompt_template_id")

        self.run_dir = Path(run_dir) if run_dir is not None else self._new_run_dir()
        self.raw_dir = self.run_dir / "raw_responses"
        self.metrics_dir = self.run_dir / "metrics"
        self.logs_dir = self.run_dir / "logs"
        for path in (self.raw_dir, self.metrics_dir, self.logs_dir):
            path.mkdir(parents=True, exist_ok=True)

        self.logger = setup_logging(
            self.__class__.__name__,
            config,
            output_file=str(self.logs_dir / "generation.log"),
        )
        self.client = client or self._build_default_client()
        self.labels_path = self.run_dir / "silver_labels.jsonl"
        self.cost_report_path = self.run_dir / "cost_report.json"

    def run(self) -> Path:
        with _RunDirLock(self.run_dir):
            self._write_config()
            completed = _load_completed_keys(self.labels_path)
            work = [item for item in self._build_work_items() if self._key(item) not in completed]
            self._write_manifest(work, skipped=len(completed))
            cost_state = self._load_cost_state()

            return self._run_locked(work, completed, cost_state)

    def _run_locked(
        self,
        work: list[WorkItem],
        completed: set[ResumeKey],
        cost_state: dict[str, Any],
    ) -> Path:
        n_succeeded = 0
        n_failed = 0
        max_budget = float((self.config.get("cost") or {}).get("max_budget_usd", 100.0))
        self._write_cost_report(cost_state)
        if float(cost_state["total_cost_usd"]) > max_budget:
            self._write_coverage(
                n_succeeded=0,
                n_failed=0,
                skipped=len(completed),
            )
            raise BudgetExceeded(
                f"Silver run already exceeds max_budget_usd={max_budget}: total_cost_usd={cost_state['total_cost_usd']}"
            )

        # Build all requests up front so we can correlate streaming
        # completions back to the originating WorkItem in O(1).
        requests: list[SilverRequest] = []
        item_by_call_id: dict[str, WorkItem] = {}
        for item in work:
            req = self._request_for_item(item)
            if req.call_id in item_by_call_id:
                raise ValueError(
                    f"duplicate request call_id {req.call_id!r}; query/doc/prompt/model identities must be unique"
                )
            requests.append(req)
            item_by_call_id[req.call_id] = item

        with open(self.labels_path, "a", encoding="utf-8") as f:
            for request, response in self._iter_responses(requests):
                item = item_by_call_id[request.call_id]
                if response is None:
                    n_failed += 1
                    record = self._failed_record(item, "missing response")
                elif not response.success:
                    n_failed += 1
                    record = self._failed_record(
                        item,
                        response.error or "client response failed",
                        response=response,
                    )
                else:
                    record = self._label_record(item, response)
                    if record["score_parsed"] is None:
                        n_failed += 1
                        record["error"] = (
                            f"unparseable score_raw={response.score_raw!r} "
                            f"for prompt_template_id={item.prompt_template_id}"
                        )
                    else:
                        n_succeeded += 1

                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
                _update_cost_state(cost_state, record)
                self._write_cost_report(cost_state)
                if float(cost_state["total_cost_usd"]) > max_budget:
                    self._write_coverage(
                        n_succeeded=n_succeeded,
                        n_failed=n_failed,
                        skipped=len(completed),
                    )
                    raise BudgetExceeded(
                        f"Silver run exceeded max_budget_usd={max_budget}: "
                        f"total_cost_usd={cost_state['total_cost_usd']}"
                    )

        self._write_coverage(n_succeeded=n_succeeded, n_failed=n_failed, skipped=len(completed))
        self._write_score_distribution()
        self._write_report(n_succeeded=n_succeeded, n_failed=n_failed, skipped=len(completed))
        return self.run_dir

    def _iter_responses(self, requests: list[SilverRequest]) -> Iterable[tuple[SilverRequest, "Any"]]:
        """Yield ``(request, response)`` pairs for the given requests.

        Prefers the client's ``submit_streaming`` (rolling executor; no
        head-of-line blocking) when available; falls back to chunked
        ``submit_batch`` for clients that only implement the simpler
        protocol (currently `OpenAICompatibleClient`).
        """
        if getattr(self.client, "query_batch", False) and hasattr(self.client, "submit_query_batches"):
            yield from self.client.submit_query_batches(requests, self.raw_dir)
            return
        if hasattr(self.client, "submit_streaming"):
            yield from self.client.submit_streaming(requests, self.raw_dir)
            return
        chunk_size = max(1, self._chunk_size())
        for start in range(0, len(requests), chunk_size):
            chunk = requests[start : start + chunk_size]
            responses = self.client.submit_batch(chunk, self.raw_dir)
            resp_by_id = {r.call_id: r for r in responses}
            for req in chunk:
                yield req, resp_by_id.get(req.call_id)

    def _new_run_dir(self) -> Path:
        base_dir = Path((self.config.get("output") or {}).get("base_dir") or f"runs/silver/{self.experiment_id}")
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        return base_dir / ts

    def _chunk_size(self) -> int:
        """Return the chunk size used to feed the underlying client.

        Each chunk is one `submit_batch` call.
        """
        return int((self.teacher.get("scoring") or {}).get("batch_size", 1) or 1)

    def _build_default_client(self) -> SilverClient:
        kind = str(self.teacher.get("client") or "openai").lower()
        if kind in {"openai", "openai_compatible", "chat_completions"}:
            return self._build_openai_client()
        raise ValueError(f"Unknown teacher.client {kind!r}; expected 'openai'.")

    def _build_openai_client(self) -> OpenAICompatibleClient:
        args = dict(self.teacher.get("scoring") or {})
        endpoint = self.teacher.get("endpoint") or {}
        return OpenAICompatibleClient(
            model_id=self.teacher_model_id,
            base_url=endpoint.get("base_url"),
            subset_size=int(args.get("subset_size", 20)),
            runs=int(args.get("runs", 10)),
            run_seeds=args.get("run_seeds"),
            score_min=float(args.get("score_min", 0)),
            score_max=float(args.get("score_max", 3)),
            max_tokens=int(args.get("max_tokens", 1536)),
            instruction=args.get("instruction"),
            grade_rubric_id=args.get("grade_rubric_id"),
            max_doc_chars=args.get("max_doc_chars"),
        )

    def _write_config(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with open(self.run_dir / "config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(self.config, f, sort_keys=False)

    def _build_work_items(self) -> list[WorkItem]:
        input_cfg = self.config.get("input") or {}
        data_cfg = _data_config_from_input(input_cfg)
        loader_cfg = {**self.config, "data": data_cfg}
        dataloader = instantiate_dataloader(loader_cfg, data_cfg)
        run_path = data_cfg["run_path"]
        queries = dataloader.get_qs_from_run(run_path)
        qids_to_run = _load_qids_to_run(self.config, input_cfg)
        if qids_to_run is not None:
            queries = [query for query in queries if str(query["qid"]) in qids_to_run]

        query_count = input_cfg.get("query_count", "all")
        if query_count != "all":
            rng = random.Random(int(input_cfg.get("query_seed", 42)))
            queries = list(queries)
            rng.shuffle(queries)
            queries = queries[: int(query_count)]

        candidates_per_query = int(input_cfg.get("candidates_per_query") or data_cfg.get("k_input") or 100)
        items: list[WorkItem] = []
        for query in queries:
            qid = str(query["qid"])
            passages = dataloader.get_psgs_from_run(run_path, qid) or []
            passages = passages[:candidates_per_query]
            for passage in passages:
                for prompt_id in self.prompt_ids:
                    items.append(
                        WorkItem(
                            query_id=qid,
                            doc_id=str(passage["pid"]),
                            query=str(query["text"]),
                            document=str(passage["text"]),
                            prompt_template_id=str(prompt_id),
                        )
                    )
        return items

    def _write_manifest(self, work: list[WorkItem], *, skipped: int) -> None:
        prompt_counts = Counter(item.prompt_template_id for item in work)
        qids = sorted({item.query_id for item in work})
        manifest = {
            "experiment_id": self.experiment_id,
            "teacher_model_id": self.teacher_model_id,
            "prompt_template_ids": self.prompt_ids,
            "n_pending_items": len(work),
            "n_skipped_existing": skipped,
            "n_queries_pending": len(qids),
            "prompt_counts": dict(prompt_counts),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        (self.run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def _request_for_item(self, item: WorkItem) -> SilverRequest:
        template = get_prompt_template(item.prompt_template_id)
        return SilverRequest(
            call_id=_call_id(item, self.teacher_model_id),
            query_id=item.query_id,
            doc_id=item.doc_id,
            query=item.query,
            document=item.document,
            prompt=template.render(query=item.query, document=item.document),
            prompt_template_id=item.prompt_template_id,
            teacher_model_id=self.teacher_model_id,
            params=dict(self.teacher.get("scoring") or {}),
        )

    def _label_record(self, item: WorkItem, response) -> dict[str, Any]:
        template = get_prompt_template(item.prompt_template_id)
        parsed = template.parse(response.score_raw or "")
        record = {
            "query_id": item.query_id,
            "doc_id": item.doc_id,
            "score_raw": response.score_raw,
            "score_parsed": parsed.score_parsed,
            "score_normalized": parsed.score_normalized,
            "score_raw_vector": getattr(response, "score_raw_vector", None),
            "score_continuous": parsed.score_parsed,
            "prompt_template_id": item.prompt_template_id,
            "teacher_model_id": self.teacher_model_id,
            "teacher_protocol": str(self.teacher.get("protocol") or self.teacher.get("client") or "unknown"),
            "batch_id": response.batch_id,
            "cli_call_id": response.cli_call_id,
            "cost_usd": response.cost_usd,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if record["score_raw_vector"] is None:
            record.pop("score_raw_vector")
        if record["score_continuous"] is None:
            record.pop("score_continuous")
        return record

    def _failed_record(self, item: WorkItem, error: str, *, response=None) -> dict[str, Any]:
        record = {
            "query_id": item.query_id,
            "doc_id": item.doc_id,
            "score_raw": response.score_raw if response is not None else None,
            "score_parsed": None,
            "score_normalized": None,
            "score_raw_vector": getattr(response, "score_raw_vector", None) if response is not None else None,
            "prompt_template_id": item.prompt_template_id,
            "teacher_model_id": self.teacher_model_id,
            "teacher_protocol": str(self.teacher.get("protocol") or self.teacher.get("client") or "unknown"),
            "batch_id": response.batch_id if response is not None else None,
            "cli_call_id": response.cli_call_id if response is not None else None,
            "cost_usd": response.cost_usd if response is not None else None,
            "error": error,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if record["score_raw_vector"] is None:
            record.pop("score_raw_vector")
        return record

    def _key(self, item: WorkItem) -> ResumeKey:
        return (item.query_id, item.doc_id, item.prompt_template_id, self.teacher_model_id)

    def _load_cost_state(self) -> dict[str, Any]:
        """Rebuild cost state from the durable `silver_labels.jsonl`.

        Do not trust the prior `cost_report.json`: a partial run that crashed
        mid-flush could leave that file out of sync with the labels (e.g. fewer
        recorded calls than there are jsonl rows). Labels are the single source
        of truth on disk because they are written line-by-line with fsync, so
        reconstruct everything from them.
        """
        state: dict[str, Any] = {
            "total_cost_usd": 0.0,
            "per_model_cost_usd": defaultdict(float),
            "per_call_cost_avg_usd": 0.0,
            "n_calls_total": 0,
            "n_calls_failed": 0,
            "cost_per_query_usd": 0.0,
        }
        if not self.labels_path.exists():
            return state
        per_model: dict[str, float] = defaultdict(float)
        n_total = 0
        n_failed = 0
        total_cost = 0.0
        with open(self.labels_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                n_total += 1
                if rec.get("error"):
                    n_failed += 1
                cost = rec.get("cost_usd")
                if cost is None:
                    continue
                cost_f = float(cost)
                total_cost += cost_f
                per_model[str(rec.get("teacher_model_id"))] += cost_f
        state.update(
            {
                "total_cost_usd": total_cost,
                "per_model_cost_usd": per_model,
                "n_calls_total": n_total,
                "n_calls_failed": n_failed,
            }
        )
        return state

    def _write_cost_report(self, state: dict[str, Any]) -> None:
        n_queries = len(_unique_queries(self.labels_path))
        n_calls = int(state.get("n_calls_total") or 0)
        total = float(state.get("total_cost_usd") or 0.0)
        state["per_call_cost_avg_usd"] = total / n_calls if n_calls else 0.0
        state["cost_per_query_usd"] = total / n_queries if n_queries else 0.0
        state["per_model_cost_usd"] = dict(state.get("per_model_cost_usd") or {})
        self.cost_report_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    def _write_coverage(self, *, n_succeeded: int, n_failed: int, skipped: int) -> None:
        coverage = {
            "n_calls_succeeded": n_succeeded,
            "n_calls_failed": n_failed,
            "n_skipped_existing": skipped,
            "n_calls_total": n_succeeded + n_failed + skipped,
            "error_counts": _error_counts(self.labels_path),
        }
        (self.metrics_dir / "coverage.json").write_text(json.dumps(coverage, indent=2), encoding="utf-8")

    def _write_score_distribution(self) -> None:
        values: list[float] = []
        if self.labels_path.exists():
            with open(self.labels_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    score = json.loads(line).get("score_normalized")
                    if score is not None:
                        values.append(float(score))
        buckets = Counter(round(v, 1) for v in values)
        payload = {
            "n_scores": len(values),
            "histogram_rounded_0_1": {str(k): buckets[k] for k in sorted(buckets)},
        }
        (self.metrics_dir / "score_distribution.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _write_report(self, *, n_succeeded: int, n_failed: int, skipped: int) -> None:
        cost = json.loads(self.cost_report_path.read_text(encoding="utf-8")) if self.cost_report_path.exists() else {}
        report = (
            f"# Silver Generation Report\n\n"
            f"- experiment_id: `{self.experiment_id}`\n"
            f"- teacher_model_id: `{self.teacher_model_id}`\n"
            f"- succeeded: {n_succeeded}\n"
            f"- failed: {n_failed}\n"
            f"- skipped_existing: {skipped}\n"
            f"- total_cost_usd: {cost.get('total_cost_usd', 0.0)}\n"
        )
        (self.run_dir / "REPORT.md").write_text(report, encoding="utf-8")


class _RunDirLock:
    """Per-`run_dir` advisory lock to prevent two `SilverGenerator` runs
    from racing on the same `silver_labels.jsonl`.

    Uses ``fcntl.flock`` against ``run_dir/.silver.lock``. The lock holder
    writes its PID into the file for forensic purposes; the lock itself is
    released when the file descriptor is closed (i.e., when this context
    manager exits or the process dies).
    """

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self.lock_path = self.run_dir / ".silver.lock"
        self._fd: int | None = None

    def __enter__(self) -> "_RunDirLock":
        self.run_dir.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(fd)
            if exc.errno in (errno.EWOULDBLOCK, errno.EAGAIN):
                holder = _read_lock_pid(self.lock_path)
                holder_str = f" held by PID {holder}" if holder else ""
                raise RunDirLocked(
                    f"Another silver run is already active in {self.run_dir}"
                    f"{holder_str}. Refusing to start a second writer; this "
                    "would race on silver_labels.jsonl and create duplicates."
                ) from None
            raise
        os.ftruncate(fd, 0)
        os.write(fd, f"{os.getpid()}\n".encode("utf-8"))
        os.fsync(fd)
        self._fd = fd
        return self

    def __exit__(self, *_: object) -> None:
        if self._fd is None:
            return
        try:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        finally:
            os.close(self._fd)
            self._fd = None


def _read_lock_pid(lock_path: Path) -> int | None:
    try:
        text = lock_path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _data_config_from_input(input_cfg: dict[str, Any]) -> dict[str, Any]:
    if "data" in input_cfg:
        return dict(input_cfg["data"])
    data_cfg = {
        k: v
        for k, v in input_cfg.items()
        if k
        in {
            "dataloader_class",
            "run_path",
            "fixture_path",
            "topics",
            "topics_tsv",
            "index",
            "k_input",
            "unsorted_run_file",
        }
    }
    if "candidates_per_query" in input_cfg and "k_input" not in data_cfg:
        data_cfg["k_input"] = input_cfg["candidates_per_query"]
    return data_cfg


def _load_qids_to_run(config: dict[str, Any], input_cfg: dict[str, Any]) -> set[str] | None:
    qids = config.get("qids_to_run") or input_cfg.get("qids_to_run")
    qids_path = config.get("qids_to_run_path") or input_cfg.get("qids_to_run_path")
    out: set[str] = set()
    if qids:
        out.update(str(qid) for qid in qids)
    if qids_path:
        with open(qids_path, "r", encoding="utf-8") as f:
            for line in f:
                qid = line.strip()
                if qid:
                    out.add(qid)
    return out or None


def _load_completed_keys(labels_path: Path) -> set[ResumeKey]:
    done: set[ResumeKey] = set()
    if not labels_path.exists():
        return done
    with open(labels_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            done.add(
                (
                    str(rec["query_id"]),
                    str(rec["doc_id"]),
                    str(rec["prompt_template_id"]),
                    str(rec["teacher_model_id"]),
                )
            )
    return done


def _update_cost_state(state: dict[str, Any], record: dict[str, Any]) -> None:
    state["n_calls_total"] = int(state.get("n_calls_total") or 0) + 1
    if record.get("error"):
        state["n_calls_failed"] = int(state.get("n_calls_failed") or 0) + 1
    cost = record.get("cost_usd")
    if cost is None:
        return
    cost_float = float(cost)
    model = str(record.get("teacher_model_id"))
    state["total_cost_usd"] = float(state.get("total_cost_usd") or 0.0) + cost_float
    per_model = state.get("per_model_cost_usd") or {}
    per_model[model] = float(per_model.get(model, 0.0)) + cost_float
    state["per_model_cost_usd"] = per_model


def _unique_queries(labels_path: Path) -> set[str]:
    out: set[str] = set()
    if not labels_path.exists():
        return out
    with open(labels_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.add(str(json.loads(line)["query_id"]))
    return out


def _error_counts(labels_path: Path) -> dict[str, int]:
    counts: Counter[str] = Counter()
    if not labels_path.exists():
        return {}
    with open(labels_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            error = json.loads(line).get("error")
            if error:
                label = str(error).split(":", 1)[0]
                counts[label] += 1
    return dict(counts)


def _chunks(items: list[WorkItem], size: int) -> Iterable[list[WorkItem]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _call_id(item: WorkItem, teacher_model_id: str) -> str:
    raw = "\0".join(
        [
            item.query_id,
            item.doc_id,
            item.prompt_template_id,
            teacher_model_id,
        ]
    )
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"call_{digest}"
