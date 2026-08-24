"""K-shot batched-self-consistency (BSC) teacher driver.

Pipeline
--------
For each ``(query, candidate_set)`` tuple in the experiment dataloader:

1. Run K Fisher-Yates permutations of the candidate set with seeds
   ``0..K-1`` (same seeding scheme as :mod:`presentation_dependence.eval.psi_manager`,
   allowing post-hoc cross-validation between silver labels and PSI artifacts).
2. For each permutation, call the configured reranker's ``rank()`` (the
   reranker must chunk into B-doc batched-PW grade prompts internally and
   emit continuous expected-grade scores via the grade-token readout).
3. Map the per-perm scalar scores back to *original* docid order.
4. Aggregate per docid: ``score_continuous = grade_max · mean(K_scores)``
   and ``score_raw_vector = K_scores · grade_max``. The reranker emits in
   ``[0, 1]`` (standard contract); we scale to ``[0, grade_max]`` so the
   silver record uses the configured grade scale.
5. Persist :class:`~presentation_dependence.self_distill.silver_io.SilverLabel`
   records to ``silver/silver_labels.jsonl`` plus a run-level
   ``silver/manifest.json``.

Implemented scope
-----------------
- ``protocol: k_shot_bsc`` only.
- Refuses partial-run aggregates: any per-query failure raises
  ``RuntimeError`` rather than publishing an incomplete
  ``silver_labels.jsonl``. Same discipline as
  :class:`PsiExperimentRunner` (see ``runner_setup.partial_run_error``).

Config surface
--------------
Reuses the standard experiment YAML (``id``/``reranker``/``data``/``eval``)
with one new optional top-level block::

    teacher:
      protocol: k_shot_bsc           # only mode this commit handles
      k_perms: 15                    # default permutation count
      seeds: [0, 1, ..., k_perms-1]  # optional; defaults to range(k_perms)
      prompt_template_id: grade_int_v1  # provenance for silver records
      grade_max: 3                   # [0, grade_max] output scale
      output_subdir: silver          # under runs/self-distill/<id>/<ts>/

The ``reranker:`` block configures the teacher model (``class`` =
``Qwen3InstructGradeReranker`` for the instruct teacher, ``Qwen3Reranker``
with ``continuous_readout: true`` for the reranker-specialized teacher).
The ``data:`` block points at the
first-stage candidate set (BM25 top-100, etc.). Output lands under
``runs/self-distill/<exp_id>/<ts>/``.
"""

from __future__ import annotations

import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from presentation_dependence.eval.runner_setup import (
    instantiate_dataloader,
    instantiate_reranker,
    partial_run_error,
    validate_rank_result_if_enabled,
    write_resolved_run_config,
)
from presentation_dependence.self_distill.silver_io import (
    SilverLabel,
    silver_run_dir,
    write_manifest,
)
from presentation_dependence.utils.setup_logging import setup_logging
from presentation_dependence.utils.trec import dirname_to_qid, qid_to_dirname


_DEFAULT_GRADE_MAX = 3
_DEFAULT_PROMPT_TEMPLATE_ID = "grade_int_v1"
_DEFAULT_OUTPUT_SUBDIR = "silver"
_PER_QID_SUBDIR = "per_qid"


def _per_qid_path(per_qid_dir: Path, qid: str) -> Path:
    """Return the per-qid silver file path, escaping unsafe qid characters."""
    return per_qid_dir / f"{qid_to_dirname(str(qid))}.jsonl"


def _scan_completed_qids(per_qid_dir: Path) -> set[str]:
    """Return qids whose per-qid silver file already exists on disk.

    The per-qid file is written atomically via tmp+rename, so its presence
    guarantees a complete K-shot record set for that query. Used for resume:
    a re-launched teacher run skips qids that already have a final file.
    """
    if not per_qid_dir.exists():
        return set()
    return {dirname_to_qid(p.stem) for p in per_qid_dir.glob("*.jsonl") if not p.name.endswith(".tmp")}


def _atomic_write_per_qid_records(per_qid_dir: Path, qid: str, records: list[SilverLabel]) -> Path:
    """Write all records for one query to a per-qid file atomically.

    Atomicity: write to ``<qid>.jsonl.tmp``, fsync, then ``replace`` to
    ``<qid>.jsonl``. POSIX rename is atomic, so partial writes never produce
    a half-finished per-qid file. This is what makes resume safe even when
    the process is hard-killed mid-query.
    """
    per_qid_dir.mkdir(parents=True, exist_ok=True)
    final = _per_qid_path(per_qid_dir, qid)
    tmp = final.with_suffix(final.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec.to_dict(), ensure_ascii=False))
            f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(final)
    return final


def _concat_per_qid_to_silver(per_qid_dir: Path, dest: Path) -> int:
    """Concatenate every per-qid file into a single ``silver_labels.jsonl``.

    Sorted by escaped filename for determinism. ``silver_labels.jsonl`` is
    a build artefact derived from the per-qid files, so keeping the same
    fixture content always produces the same byte-for-byte concatenation.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(dest, "w", encoding="utf-8") as out_f:
        for per_qid in sorted(per_qid_dir.glob("*.jsonl")):
            if per_qid.name.endswith(".tmp"):
                continue
            with open(per_qid, "r", encoding="utf-8") as in_f:
                for line in in_f:
                    if not line.strip():
                        continue
                    out_f.write(line if line.endswith("\n") else line + "\n")
                    n += 1
    return n


def _per_qid_record_counts(per_qid_dir: Path) -> dict[str, int]:
    """Count durable records per completed qid, rejecting empty files."""
    counts: dict[str, int] = {}
    for path in sorted(per_qid_dir.glob("*.jsonl")):
        if path.name.endswith(".tmp"):
            continue
        qid = dirname_to_qid(path.stem)
        count = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
        if count == 0:
            raise RuntimeError(f"Completed per-qid silver file is empty: {path}. Remove the invalid file and resume.")
        counts[qid] = count
    return counts


def load_qids_allowlist(config: dict) -> set[str]:
    """Return qid allowlist from ``qids_to_run`` and/or ``qids_to_run_path``.

    ``qids_to_run_path`` is a top-level sibling of the existing
    ``qids_to_run`` list so large prefix samples (1K, 30K, 100K) don't bloat
    YAML configs. This is what lets the MS MARCO self-distill dataset extend
    non-redundantly: every pilot/production config points at the same growing
    fixture but constrains itself to a cheap prefix file.
    """
    out = set(map(str, config.get("qids_to_run", []) or []))
    path_s = config.get("qids_to_run_path")
    if path_s:
        path = Path(path_s)
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                qid = line.strip()
                if qid:
                    out.add(qid)
    return out


def teacher_model_id_from_config(config: dict) -> str:
    """Return a manifest-friendly teacher id, including LoRA when configured."""
    rc = config.get("reranker") or {}
    base = str(rc.get("model_name") or rc.get("class") or "unknown")
    lora_path = rc.get("lora_path")
    if not lora_path:
        return base

    lora_ref = str(lora_path).rstrip("/")
    channel_name = Path(lora_ref).name
    extra_channels = (config.get("execution") or {}).get("extra_input_channels") or {}
    if isinstance(extra_channels, dict):
        lora_ref = str(extra_channels.get(channel_name) or lora_ref)
    return f"{base}+lora:{lora_ref}"


def fisher_yates_shuffle(items: list, seed: int) -> list:
    """Deterministic Fisher-Yates shuffle.

    Uses the same Fisher-Yates seeds as
    :func:`presentation_dependence.eval.psi_manager._random_shuffle`. Both call sites use
    ``random.Random(int(seed)).shuffle(...)``.
    """
    rng = random.Random(int(seed))
    out = list(items)
    rng.shuffle(out)
    return out


class KShotBSCTeacher:
    """Drives a K-shot BSC teacher pass and persists silver labels.

    Parameters
    ----------
    config_path : str | Path
        Experiment YAML. Must contain ``id``, ``reranker``, ``data``, plus an
        optional ``teacher`` block (see module docstring for schema).
    runs_root : str | Path, default ``"runs"``
        Root for run artefacts. Output lives at
        ``<runs_root>/self-distill/<exp_id>/<timestamp>/``.
    run_dir : str | Path | None, default ``None``
        When set, reuse this directory instead of creating a new timestamped
        one. Useful for pipelines that pre-allocate the directory.
    reranker : object | None, default ``None``
        Pre-instantiated reranker to reuse (saves the HF model load on local
        smoke runs). When ``None``, instantiates from ``config.reranker``.
    """

    def __init__(  # noqa: C901
        self,
        config_path: str | Path,
        runs_root: str | Path = "runs",
        run_dir: str | Path | None = None,
        reranker: object | None = None,
    ):
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")

        with open(self.config_path, "r") as f:
            self.config: dict = yaml.safe_load(f) or {}
        if not isinstance(self.config, dict):
            raise ValueError(f"{self.config_path} did not parse to a dict")

        self.exp_id = str(self.config["id"])
        self.runs_root = Path(runs_root)

        # Teacher block: protocol, permutation count, seeds, prompt template ID,
        # and output scale.
        teacher_cfg = self.config.get("teacher") or {}
        self.protocol = str(teacher_cfg.get("protocol", "k_shot_bsc")).strip().lower()
        if self.protocol != "k_shot_bsc":
            # Unsupported protocols must fail rather than use the BSC path.
            raise ValueError(
                f"teacher.protocol={self.protocol!r} is not implemented in this module. Only 'k_shot_bsc' is available."
            )
        self.k_perms = int(teacher_cfg.get("k_perms", 15))
        if self.k_perms <= 0:
            raise ValueError(f"teacher.k_perms must be positive, got {self.k_perms}")
        seeds_raw = teacher_cfg.get("seeds")
        if seeds_raw is None:
            self.seeds: list[int] = list(range(self.k_perms))
        else:
            self.seeds = [int(s) for s in seeds_raw]
            if len(self.seeds) != self.k_perms:
                raise ValueError(f"teacher.seeds ({len(self.seeds)}) must equal teacher.k_perms ({self.k_perms})")
        self.prompt_template_id = str(teacher_cfg.get("prompt_template_id", _DEFAULT_PROMPT_TEMPLATE_ID))
        self.grade_max = int(teacher_cfg.get("grade_max", _DEFAULT_GRADE_MAX))
        if self.grade_max <= 0:
            raise ValueError(f"teacher.grade_max must be positive, got {self.grade_max}")
        self.output_subdir = str(teacher_cfg.get("output_subdir", _DEFAULT_OUTPUT_SUBDIR))
        # Cross-query batching: how many queries' prefixes to dispatch in one
        # ``Reranker.rank_query_batch`` call (one engine forward batch). 1 =
        # legacy per-query loop. Higher values amortise scheduler overhead and
        # raise GPU utilisation under vLLM, at the cost of larger failure
        # granularity (a hard kill mid-batch loses the in-flight batch's
        # work).
        self.cross_query_batch = int(teacher_cfg.get("cross_query_batch", 1))
        if self.cross_query_batch <= 0:
            raise ValueError(f"teacher.cross_query_batch must be positive, got {self.cross_query_batch}")

        # Run directory: <runs_root>/self-distill/<exp_id>/<ts>/.
        # Namespacing under "self-distill/" keeps these passes
        # disjoint from quality / PSI runs in scripts/query_results.py etc.
        if run_dir is not None:
            self.run_dir = Path(run_dir)
            ts = self.run_dir.name
            self._reuses_run_dir = True
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            self.run_dir = silver_run_dir(self.runs_root, self.exp_id, ts)
            self._reuses_run_dir = False
        self.silver_dir = self.run_dir / self.output_subdir
        self.silver_dir.mkdir(parents=True, exist_ok=True)

        self.logger = setup_logging(
            self.__class__.__name__,
            self.config,
            output_file=str(self.run_dir / "teacher_run.log"),
        )

        self.data_config = self.config.get("data", {})
        if reranker is not None:
            self.reranker = reranker
            self.logger.info("Reusing pre-loaded reranker (%s)", type(reranker).__name__)
        else:
            self.reranker = instantiate_reranker(self.config, self.logger)
        self.dataloader = instantiate_dataloader(self.config, self.data_config)
        if not self._reuses_run_dir:
            write_resolved_run_config(
                self.run_dir,
                self.config,
                config_path=self.config_path,
                ts=ts,
                data_config=self.data_config,
            )

        # Surface teacher_model_id for the silver manifest and record schema.
        # LoRA-backed iterative silver must name the adapter as well as the
        # base model; otherwise v1 silver looks identical to off-the-shelf v0.
        self.teacher_model_id = teacher_model_id_from_config(self.config)
        self.logger.info(
            "KShotBSCTeacher exp_id=%s K=%d teacher=%s prompt_template_id=%s grade_max=%d",
            self.exp_id,
            self.k_perms,
            self.teacher_model_id,
            self.prompt_template_id,
            self.grade_max,
        )

    # -- Per-query K-shot inference -----------------------------------------

    def _absorb_perm_scores(
        self,
        per_doc_scores: dict[str, list[float]],
        per_doc_grade_vectors: dict[str, list[list[float]]],
        shuffled_passages: list[dict],
        result: dict,
        seed: int,
        qid: str,
    ) -> None:
        """Append one perm's reranker output into the per-docid score list."""
        validate_rank_result_if_enabled(self.config, result, shuffled_passages)
        scores_init = result.get("scores_init_order")
        if scores_init is None:
            raise RuntimeError(
                f"qid={qid} perm={seed}: reranker returned "
                "scores_init_order=None; K-shot BSC teacher requires per-doc "
                "continuous scalars. Confirm the reranker is configured for "
                "continuous_readout."
            )
        if len(scores_init) != len(shuffled_passages):
            raise RuntimeError(
                f"qid={qid} perm={seed}: scores_init_order length "
                f"{len(scores_init)} != #shuffled passages {len(shuffled_passages)}"
            )
        grade_vectors = result.get("grade_probabilities_init_order")
        if grade_vectors is None:
            raise RuntimeError(
                f"qid={qid} perm={seed}: reranker returned no "
                "grade_probabilities_init_order; vector-loss silver requires "
                "per-slot P(g) vectors from the continuous grade readout."
            )
        if len(grade_vectors) != len(shuffled_passages):
            raise RuntimeError(
                f"qid={qid} perm={seed}: grade_probabilities_init_order length "
                f"{len(grade_vectors)} != #shuffled passages {len(shuffled_passages)}"
            )
        # Reranker scores are in [0, 1] (standard contract); scale them to
        # [0, grade_max] for the silver record.
        expected_len = self.grade_max + 1
        for passage, score, vector in zip(shuffled_passages, scores_init, grade_vectors, strict=True):
            pid = str(passage["pid"])
            vector_f = [float(x) for x in vector]
            if len(vector_f) != expected_len:
                raise RuntimeError(
                    f"qid={qid} doc={pid} perm={seed}: grade vector length "
                    f"{len(vector_f)} != grade_max + 1 ({expected_len})"
                )
            per_doc_scores[pid].append(float(score) * self.grade_max)
            per_doc_grade_vectors[pid].append(vector_f)

    def _build_silver_records(
        self,
        qid: str,
        passages: list[dict],
        per_doc_scores: dict[str, list[float]],
        per_doc_grade_vectors: dict[str, list[list[float]]],
    ) -> list[SilverLabel]:
        """Convert a fully-populated per-doc K-perm score map into SilverLabels."""
        ts_now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        records: list[SilverLabel] = []
        for passage in passages:
            pid = str(passage["pid"])
            raw = per_doc_scores[pid]
            if len(raw) != self.k_perms:
                raise RuntimeError(
                    f"qid={qid} doc={pid}: collected {len(raw)} scores "
                    f"across K={self.k_perms} perms — internal invariant violated."
                )
            raw_vectors = per_doc_grade_vectors[pid]
            if len(raw_vectors) != self.k_perms:
                raise RuntimeError(
                    f"qid={qid} doc={pid}: collected {len(raw_vectors)} grade vectors "
                    f"across K={self.k_perms} perms — internal invariant violated."
                )
            mean_score = sum(raw) / float(self.k_perms)
            n_grades = self.grade_max + 1
            mean_grade_vector = [
                sum(vector[g] for vector in raw_vectors) / float(self.k_perms) for g in range(n_grades)
            ]
            expected_score_from_vector = sum(float(g) * p for g, p in enumerate(mean_grade_vector))
            if abs(expected_score_from_vector - mean_score) > 1e-5:
                raise RuntimeError(
                    f"qid={qid} doc={pid}: E[score_grade_vector]={expected_score_from_vector:.8f} "
                    f"does not match score_continuous={mean_score:.8f}"
                )
            records.append(
                SilverLabel(
                    query_id=qid,
                    doc_id=pid,
                    score_continuous=mean_score,
                    score_raw_vector=list(raw),
                    score_grade_vector=mean_grade_vector,
                    teacher_model_id=self.teacher_model_id,
                    teacher_protocol="k_shot_bsc",
                    prompt_template_id=self.prompt_template_id,
                    timestamp=ts_now,
                    k_perms=self.k_perms,
                )
            )
        return records

    def _kshot_label_one_query(
        self,
        query: dict,
        passages: list[dict],
    ) -> list[SilverLabel]:
        """Run K perms for one query and emit per-doc SilverLabel records."""
        qid = str(query.get("qid"))
        original_pids = [str(p["pid"]) for p in passages]
        # docid -> list of K continuous scores in [0, grade_max], in seed order.
        per_doc_scores: dict[str, list[float]] = {pid: [] for pid in original_pids}
        per_doc_grade_vectors: dict[str, list[list[float]]] = {pid: [] for pid in original_pids}

        for perm_idx, seed in enumerate(self.seeds):
            shuffled = fisher_yates_shuffle(passages, seed)
            result = self.reranker.rank(query, shuffled)
            self._absorb_perm_scores(per_doc_scores, per_doc_grade_vectors, shuffled, result, seed, qid)
            self.logger.debug(
                "qid=%s perm=%d/%d seed=%d  scored_docs=%d",
                qid,
                perm_idx + 1,
                self.k_perms,
                seed,
                len(result.get("scores_init_order") or ()),
            )

        return self._build_silver_records(qid, passages, per_doc_scores, per_doc_grade_vectors)

    def _kshot_label_query_batch(
        self,
        batch: list[tuple[dict, list[dict]]],
    ) -> dict[str, list[SilverLabel]]:
        """Cross-query batched K-shot pass over ``batch`` of (query, passages).

        For each perm seed, shuffle every batch entry's passages with the same
        seed and submit them all to ``reranker.rank_query_batch`` in a single
        call. The reranker (with vLLM as the engine) packs the prefixes from
        all queries into one engine forward batch, lifting GPU utilisation
        past what within-query batching alone achieves.

        After all K perm passes complete, build :class:`SilverLabel` records
        per query and return the map keyed by qid. Caller must perform atomic
        per-qid writes (so a successful batch flushes per-query files at
        end-of-batch boundaries).

        On any per-perm failure the whole batch is lost — no per-qid files
        get written for the batch. Callers re-run with the same run_dir;
        previously-completed batches resume cleanly via ``_scan_completed_qids``.
        """
        if not batch:
            return {}

        # Per-query, per-doc K-perm score accumulator.
        per_doc_scores: dict[str, dict[str, list[float]]] = {
            str(query.get("qid")): {str(p["pid"]): [] for p in passages} for query, passages in batch
        }
        per_doc_grade_vectors: dict[str, dict[str, list[list[float]]]] = {
            str(query.get("qid")): {str(p["pid"]): [] for p in passages} for query, passages in batch
        }

        for perm_idx, seed in enumerate(self.seeds):
            shuffled_items = [(query, fisher_yates_shuffle(passages, seed)) for query, passages in batch]
            results = self.reranker.rank_query_batch(shuffled_items)
            if len(results) != len(shuffled_items):
                raise RuntimeError(
                    f"rank_query_batch returned {len(results)} results for "
                    f"{len(shuffled_items)} items — invariant violated."
                )
            for (query, shuffled), result in zip(shuffled_items, results):
                qid = str(query.get("qid"))
                self._absorb_perm_scores(
                    per_doc_scores[qid],
                    per_doc_grade_vectors[qid],
                    shuffled,
                    result,
                    seed,
                    qid,
                )

            self.logger.debug(
                "batch perm=%d/%d seed=%d  n_queries=%d",
                perm_idx + 1,
                self.k_perms,
                seed,
                len(batch),
            )

        records_by_qid: dict[str, list[SilverLabel]] = {}
        for query, passages in batch:
            qid = str(query.get("qid"))
            records_by_qid[qid] = self._build_silver_records(
                qid, passages, per_doc_scores[qid], per_doc_grade_vectors[qid]
            )
        return records_by_qid

    # -- Path implementations -----------------------------------------------

    def _log_per_query_score_summary(self, qid: str, records: list[SilverLabel]) -> None:
        """Emit the per-query score distribution log line used at smoke time."""
        continuous_vals = [r.score_continuous for r in records]
        if not continuous_vals:
            return
        self.logger.info(
            "qid=%s silver: n=%d  min=%.3f  mean=%.3f  max=%.3f",
            qid,
            len(records),
            min(continuous_vals),
            sum(continuous_vals) / len(continuous_vals),
            max(continuous_vals),
        )

    def _run_per_query_path(
        self,
        *,
        queries: list[dict],
        qids_allowlist: set[str],
        already_done_qids: set[str],
        run_path: str,
        per_qid_dir: Path,
        failed_qids: list[str],
    ) -> tuple[int, int]:
        """Legacy per-query loop: one rank() call per perm per query, atomic
        per-qid write at end-of-query.
        """
        n_queries_processed = 0
        max_n_documents = 0
        for query in queries:
            qid = str(query.get("qid"))
            if qids_allowlist and qid not in qids_allowlist:
                continue
            if qid in already_done_qids:
                continue
            try:
                passages = self.dataloader.get_psgs_from_run(run_path, qid) or []
                if not passages:
                    self.logger.error("qid=%s: empty passage list", qid)
                    failed_qids.append(qid)
                    continue
                self.logger.info("qid=%s n_docs=%d K=%d", qid, len(passages), self.k_perms)
                records = self._kshot_label_one_query(query, passages)
                _atomic_write_per_qid_records(per_qid_dir, qid, records)
                n_queries_processed += 1
                max_n_documents = max(max_n_documents, len(passages))
                self._log_per_query_score_summary(qid, records)
            except Exception as e:
                self.logger.error("Failed on qid=%s: %s", qid, e, exc_info=True)
                failed_qids.append(qid)
        return n_queries_processed, max_n_documents

    def _warn_on_teacher_identity_drift(self, n_reused: int) -> None:
        """Warn when resuming over shards a different teacher configuration wrote.

        Per-qid shards are reused by filename, so nothing otherwise notices that
        the labels being concatenated came from two different teachers. The
        previous run's ``manifest.json`` is the only identity record available;
        an interrupted run leaves none, so this is best effort and never fatal.
        """
        manifest_path = self.silver_dir / "manifest.json"
        if not manifest_path.is_file():
            return
        try:
            prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            self.logger.warning("Could not read %s to verify resume: %s", manifest_path, exc)
            return
        current = {
            "teacher_model_id": self.teacher_model_id,
            "teacher_protocol": "k_shot_bsc",
            "prompt_template_id": self.prompt_template_id,
            "k_perms": self.k_perms,
            "seeds": list(self.seeds),
            "grade_max": self.grade_max,
        }
        differing = {
            key: {"previous": prior.get(key), "current": value}
            for key, value in current.items()
            if key in prior and prior.get(key) != value
        }
        if not differing:
            return
        self.logger.warning(
            "Resume identity mismatch in %s: %d reused per-qid shard(s) were written by a "
            "different teacher configuration (%s). silver_labels.jsonl would mix both. "
            "Use a fresh output directory unless the difference is immaterial.",
            self.silver_dir,
            n_reused,
            ", ".join(f"{k}: {v['previous']!r} -> {v['current']!r}" for k, v in differing.items()),
        )

    def _run_batched_path(  # noqa: C901
        self,
        *,
        queries: list[dict],
        qids_allowlist: set[str],
        already_done_qids: set[str],
        run_path: str,
        per_qid_dir: Path,
        failed_qids: list[str],
    ) -> tuple[int, int]:
        """Cross-query batched path: K perm passes over a batch of N queries
        per engine call. Per-qid files are atomically written at end-of-batch
        only. A hard kill mid-batch loses up to ``cross_query_batch`` queries
        of in-flight work but never previously-completed batches.
        """
        # Materialise the queue of queries we'll actually run, so partial
        # success across batches stays trackable.
        runnable: list[dict] = []
        for query in queries:
            qid = str(query.get("qid"))
            if qids_allowlist and qid not in qids_allowlist:
                continue
            if qid in already_done_qids:
                continue
            runnable.append(query)

        n_queries_processed = 0
        max_n_documents = 0
        for batch_start in range(0, len(runnable), self.cross_query_batch):
            batch_queries = runnable[batch_start : batch_start + self.cross_query_batch]
            batch_data: list[tuple[dict, list[dict]]] = []
            batch_qids_for_logging: list[str] = []
            for query in batch_queries:
                qid = str(query.get("qid"))
                try:
                    passages = self.dataloader.get_psgs_from_run(run_path, qid) or []
                except Exception as e:
                    self.logger.error("Failed loading passages for qid=%s: %s", qid, e, exc_info=True)
                    failed_qids.append(qid)
                    continue
                if not passages:
                    self.logger.error("qid=%s: empty passage list", qid)
                    failed_qids.append(qid)
                    continue
                batch_data.append((query, passages))
                batch_qids_for_logging.append(qid)
                max_n_documents = max(max_n_documents, len(passages))

            if not batch_data:
                continue

            self.logger.info(
                "batch %d: n_queries=%d K=%d qids=%s",
                batch_start // self.cross_query_batch,
                len(batch_data),
                self.k_perms,
                batch_qids_for_logging[:8] + (["..."] if len(batch_qids_for_logging) > 8 else []),
            )
            try:
                records_by_qid = self._kshot_label_query_batch(batch_data)
            except Exception as e:
                # Whole-batch failure: mark every qid in this batch as failed
                # so the post-loop ``partial_run_error`` raises with the full
                # picture. Per-qid files for previously-completed batches are
                # preserved on disk; resume picks them up on the next run.
                affected = [str(q.get("qid")) for q, _ in batch_data]
                self.logger.error(
                    "Failed on batch starting at qid=%s (%d queries): %s",
                    affected[0],
                    len(affected),
                    e,
                    exc_info=True,
                )
                failed_qids.extend(affected)
                continue

            # Successful batch: atomically write each query's per-qid file.
            for query, passages in batch_data:
                qid = str(query.get("qid"))
                records = records_by_qid[qid]
                _atomic_write_per_qid_records(per_qid_dir, qid, records)
                n_queries_processed += 1
                self._log_per_query_score_summary(qid, records)
        return n_queries_processed, max_n_documents

    # -- Main loop ----------------------------------------------------------

    def run(self) -> dict[str, Any]:
        """Execute the K-shot BSC pass, streaming per-qid silver files.

        Streaming and resume behavior:

        - Per-query records are written to ``silver/per_qid/<qid>.jsonl`` via
          ``tmp + rename`` so each query is atomically completed before the
          next one starts. A hard kill at any point loses at most
          the in-flight query, never previously-completed work.
        - On startup, ``run()`` scans ``silver/per_qid/`` and skips qids
          whose per-qid file already exists. Pass ``run_dir=<existing>`` to
          ``KShotBSCTeacher`` (or ``--run-dir`` from the CLI) to resume.
        - On successful completion, all per-qid files are concatenated into
          the canonical ``silver/silver_labels.jsonl`` and ``manifest.json``
          is written. The aggregated artefact is regenerated every run from
          the per-qid files, so re-running an already-finished job is a
          deterministic no-op.

        Returns a small summary dict that the CLI surfaces at exit.
        """
        self.logger.info("Starting silver-data run %s (run_dir=%s)", self.exp_id, self.run_dir)

        run_path = self.data_config["run_path"]
        queries = self.dataloader.get_qs_from_run(run_path)
        self.logger.info("Loaded %d queries", len(queries))

        qids_allowlist = load_qids_allowlist(self.config)

        # Validate fixture coverage BEFORE spending teacher compute. Catches
        # the common "qids_30k.txt staged but fixture.jsonl is still 1K"
        # mistake without burning forwards on it.
        if qids_allowlist:
            fixture_qids = {str(q.get("qid")) for q in queries}
            missing = sorted(qids_allowlist - fixture_qids)
            if missing:
                preview = ", ".join(missing[:10])
                more = "" if len(missing) <= 10 else f" (+{len(missing) - 10} more)"
                raise RuntimeError(
                    f"{len(missing)} qids from qids_to_run/qids_to_run_path were not present "
                    f"in the dataloader input: {preview}{more}. Refusing to publish "
                    "silver_labels.jsonl from a prefix whose fixture is not fully materialized."
                )

        per_qid_dir = self.silver_dir / _PER_QID_SUBDIR
        already_done_qids = _scan_completed_qids(per_qid_dir)
        if already_done_qids:
            self.logger.info(
                "Resume: %d qids already complete in %s; skipping.",
                len(already_done_qids),
                per_qid_dir,
            )
            self._warn_on_teacher_identity_drift(len(already_done_qids))

        # Decide between per-query and cross-query batched paths. Cross-query
        # batching only helps when the reranker actually overrides
        # ``rank_query_batch`` (true for self-distill continuous-readout
        # wrappers backed by vLLM). For wrappers that stay on the base-class
        # default the batched path would dispatch one rank() per query
        # internally: same wall-clock as the legacy loop but with larger
        # failure granularity. Fall back loudly so operators don't silently
        # over-promise the speedup.
        use_batched_path = self.cross_query_batch > 1
        if use_batched_path and not getattr(self.reranker, "supports_query_batching", False):
            self.logger.warning(
                "teacher.cross_query_batch=%d but %s does not implement real "
                "cross-query batching (supports_query_batching=False). Falling "
                "back to per-query path; expected wall-clock unchanged.",
                self.cross_query_batch,
                type(self.reranker).__name__,
            )
            use_batched_path = False
        if use_batched_path:
            self.logger.info(
                "Cross-query batching enabled: cross_query_batch=%d, reranker=%s",
                self.cross_query_batch,
                type(self.reranker).__name__,
            )

        failed_qids: list[str] = []
        n_queries_processed_this_run = 0
        max_n_documents = 0

        if use_batched_path:
            n_queries_processed_this_run, max_n_documents = self._run_batched_path(
                queries=queries,
                qids_allowlist=qids_allowlist,
                already_done_qids=already_done_qids,
                run_path=run_path,
                per_qid_dir=per_qid_dir,
                failed_qids=failed_qids,
            )
        else:
            n_queries_processed_this_run, max_n_documents = self._run_per_query_path(
                queries=queries,
                qids_allowlist=qids_allowlist,
                already_done_qids=already_done_qids,
                run_path=run_path,
                per_qid_dir=per_qid_dir,
                failed_qids=failed_qids,
            )

        # If any qid failed, do NOT publish the aggregated silver_labels.jsonl —
        # but the per-qid files we did finish remain on disk so the next run
        # resumes from where this one left off. As in PsiExperimentRunner,
        # partial runs do not publish aggregate output.
        if failed_qids:
            raise partial_run_error(
                failed_qids,
                phase="K-shot BSC teacher inference",
                aggregate_label="silver_labels.jsonl",
            )

        record_counts = _per_qid_record_counts(per_qid_dir)
        n_queries_total = len(record_counts)
        max_n_documents = max(record_counts.values(), default=0)
        labels_path = self.silver_dir / "silver_labels.jsonl"
        n_records = _concat_per_qid_to_silver(per_qid_dir, labels_path)

        manifest_path = self.silver_dir / "manifest.json"
        manifest = {
            "experiment_id": self.exp_id,
            "teacher_model_id": self.teacher_model_id,
            "teacher_protocol": "k_shot_bsc",
            "prompt_template_id": self.prompt_template_id,
            "k_perms": self.k_perms,
            "seeds": list(self.seeds),
            "grade_max": self.grade_max,
            "cross_query_batch": self.cross_query_batch,
            "n_queries": n_queries_total,
            "n_queries_processed_this_run": n_queries_processed_this_run,
            "n_queries_resumed": len(already_done_qids),
            "n_documents": max_n_documents,
            "n_records": n_records,
            "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        write_manifest(manifest_path, manifest)
        self.logger.info(
            "Wrote %s (%d records, %d qids; this-run=%d resumed=%d)",
            labels_path,
            n_records,
            n_queries_total,
            n_queries_processed_this_run,
            len(already_done_qids),
        )
        self.logger.info("Wrote %s", manifest_path)

        return {
            "run_dir": str(self.run_dir),
            "silver_labels": str(labels_path),
            "manifest": str(manifest_path),
            "n_queries": n_queries_total,
            "n_queries_processed_this_run": n_queries_processed_this_run,
            "n_queries_resumed": len(already_done_qids),
            "n_records": n_records,
            "k_perms": self.k_perms,
            "teacher_model_id": self.teacher_model_id,
            "teacher_protocol": "k_shot_bsc",
        }
