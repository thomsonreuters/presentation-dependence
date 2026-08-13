"""Shared setup for experiment and PSI runners."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytrec_eval  # type: ignore
import yaml

from presentation_dependence.eval.loaders import LOADER_CLASSES
from presentation_dependence.rerankers import get_reranker_class, validate_rank_result
from presentation_dependence.rerankers.base import Passage, RankResult
from presentation_dependence.utils.dataset_meta import meta_for_run
from presentation_dependence.utils.repo_meta import git_head_sha


def instantiate_reranker(config: dict, logger: Any | None = None):
    """Instantiate the configured reranker with consistent validation."""
    reranker_cfg = config.get("reranker", {})
    class_name = reranker_cfg.get("class")
    if not class_name:
        raise ValueError("config.reranker.class is required")
    cls = get_reranker_class(class_name)
    if logger is not None:
        logger.info("Instantiating reranker %s", class_name)
    return cls(config)


def require_first_stage_input(run_path: str | Path) -> Path:
    """Return ``run_path``, raising an actionable error when it is not staged.

    ``data/`` is gitignored, so a fresh checkout has no first-stage inputs at
    all. Checking here keeps the failure fast and legible: ``PyseriniLoader``
    would otherwise download a ~1.5 GB prebuilt index before hitting the
    missing run file.
    """
    path = Path(run_path)
    if path.is_file():
        return path
    raise FileNotFoundError(
        f"First-stage input not found: {path}\n"
        "data/ is gitignored, so a fresh checkout has to materialize it first:\n"
        "  uv run python scripts/setup_reproduction_data.py plan\n"
        "  uv run python scripts/setup_reproduction_data.py run\n"
        "Per-dataset and gated/internal access notes are in docs/DATA-SETUP.md.\n"
        "For a check that needs no data, network, or GPU, run the synthetic smoke:\n"
        "  uv run python scripts/data/build_smoke_fixture.py\n"
        "  uv run python scripts/run_experiment.py -e _smoke-fixture"
    )


def instantiate_dataloader(config: dict, data_config: dict | None = None):
    """Instantiate the configured dataloader and resolve sorted run files."""
    dc = data_config if data_config is not None else config.get("data", {})
    class_name = dc.get("dataloader_class")
    if not class_name:
        raise ValueError("config.data.dataloader_class is required")
    try:
        cls = LOADER_CLASSES[class_name]
    except KeyError:
        known = ", ".join(sorted(LOADER_CLASSES))
        raise ValueError(f"Unknown dataloader class {class_name!r}. Known: {known}.") from None

    if dc.get("run_path"):
        require_first_stage_input(dc["run_path"])

    dataloader = cls(config)
    if not dc.get("unsorted_run_file", False):
        run_path = dc["run_path"]
        sorted_path = dataloader.ensure_sorted_run_file(run_path)
        if sorted_path != run_path:
            dc["run_path"] = sorted_path
    return dataloader


def load_qrels(qrels_path: str | Path) -> dict[str, dict[str, int]]:
    """Load TREC qrels via pytrec_eval."""
    with open(qrels_path, "r") as f:
        return pytrec_eval.parse_qrel(f)


def should_validate_rank_result(config: dict) -> bool:
    """Return whether debug rank-result validation is enabled."""
    return bool((config.get("debug") or {}).get("validate_rank_result", False))


def validate_rank_result_if_enabled(config: dict, result: RankResult, passages: list[Passage]) -> None:
    """Validate ``result`` when ``debug.validate_rank_result`` is true."""
    if should_validate_rank_result(config):
        validate_rank_result(result, passages)


# Config fields that change what a scored query means. Resume reuses per-query
# shards by filename, so if any of these differ between invocations the run
# directory ends up holding shards from two different experiments.
_FINGERPRINT_RERANKER_KEYS = (
    "class",
    "model_name",
    "model_release",
    "model_size",
    "revision",
    "lora_path",
    "lora_adapters",
    "instruction",
    "grade_rubric_id",
    "prompt_template_id",
    "grade_skeleton_dummy",
    "grade_dummy",
    "scoring_mode",
    "scoring_method",
    "continuous_readout",
    "chunk_assignment",
    "docs_per_score_forward",
    "max_length",
    "max_doc_chars",
    "chat_template_kwargs",
)
_FINGERPRINT_DATA_KEYS = ("run_path", "topics", "topics_tsv", "k_input", "dataloader_class")


def run_fingerprint(config: dict, data_config: dict) -> dict:
    """Return the identity of a run: what would make earlier shards incomparable.

    Runs on every invocation via ``build_resolved_run_config``, so it never
    raises. A config shape it cannot digest yields an ``unavailable`` digest,
    which disables the resume comparison rather than failing the run.
    """
    try:
        reranker = config.get("reranker") or {}
        if not isinstance(reranker, dict):
            reranker = {}
        fields: dict[str, Any] = {
            f"reranker.{k}": reranker[k] for k in _FINGERPRINT_RERANKER_KEYS if reranker.get(k) is not None
        }
        if isinstance(data_config, dict):
            fields.update(
                {f"data.{k}": data_config[k] for k in _FINGERPRINT_DATA_KEYS if data_config.get(k) is not None}
            )
        eval_cfg = config.get("eval")
        qrels = eval_cfg.get("qrels_path") if isinstance(eval_cfg, dict) else None
        if qrels:
            fields["eval.qrels_path"] = qrels
        fields["git_sha"] = git_head_sha(Path(__file__).parent)
        # sort_keys raises on a nested dict with mixed-type keys; repr is a
        # stable enough fallback for an identity digest.
        try:
            payload = json.dumps(fields, sort_keys=True, default=str)
        except TypeError:
            payload = repr(sorted(fields.items(), key=lambda kv: kv[0]))
        digest = hashlib.sha256(payload.encode()).hexdigest()[:12]
        return {"digest": digest, "fields": fields}
    except Exception:  # identity metadata must never break a run
        return {"digest": "unavailable", "fields": {}}


def check_resume_fingerprint(run_dir: str | Path, fingerprint: dict, logger: logging.Logger) -> dict | None:
    """Warn when resuming into a run directory produced by a different configuration.

    Best effort and non-fatal: existing per-query shards are reused by filename,
    so a mismatch means the run directory mixes two experiments. Returns the
    differing fields, or ``None`` when there is nothing to compare against.
    """
    prior_path = Path(run_dir) / "resolved_config.yaml"
    if not prior_path.is_file():
        return None
    try:
        prior = yaml.safe_load(prior_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:  # a corrupt prior config should not stop the run
        logger.warning("Could not read %s to verify resume: %s", prior_path, exc)
        return None
    prior_fp = prior.get("_run_fingerprint") or {}
    prior_fields = prior_fp.get("fields")
    if not isinstance(prior_fields, dict) or not prior_fields:
        # Written before fingerprinting existed, or a config neither side could
        # digest. Nothing to compare, so stay quiet rather than guess.
        return None
    if "unavailable" in (prior_fp.get("digest"), fingerprint.get("digest")):
        return None
    if prior_fp.get("digest") == fingerprint["digest"]:
        return None

    new_fields = fingerprint["fields"]
    differing = {
        key: {"previous": prior_fields.get(key), "current": new_fields.get(key)}
        for key in sorted(set(prior_fields) | set(new_fields))
        if prior_fields.get(key) != new_fields.get(key)
    }
    logger.warning(
        "Resume fingerprint mismatch in %s: this invocation differs from the one that "
        "wrote the existing per-query results in %d field(s): %s. Completed queries are "
        "reused by filename, so the run directory now mixes both configurations. Use a "
        "fresh run directory unless you know the difference is immaterial.",
        run_dir,
        len(differing),
        ", ".join(differing),
    )
    _append_resume_history(Path(run_dir), prior_fp.get("digest"), fingerprint["digest"], differing, logger)
    return differing


def _append_resume_history(run_dir: Path, previous: object, current: object, differing: dict, logger) -> None:
    """Record the mismatch beside the run so it survives the log."""
    path = run_dir / "resume_history.json"
    try:
        history = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []
        if not isinstance(history, list):
            history = []
        history.append(
            {
                "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "previous_digest": previous,
                "current_digest": current,
                "differing_fields": differing,
            }
        )
        path.write_text(json.dumps(history, indent=2, default=str) + "\n", encoding="utf-8")
    except Exception as exc:  # recording is a courtesy, never a failure mode
        logger.warning("Could not record resume history at %s: %s", path, exc)


def build_resolved_run_config(
    config: dict,
    *,
    config_path: str | Path,
    ts: str,
    data_config: dict,
) -> dict:
    """Return the resolved config snapshot written into a run directory."""
    resolved = dict(config)
    resolved["_run_timestamp"] = ts
    resolved["_git_sha"] = git_head_sha(Path(__file__).parent)
    resolved["_source_config"] = str(Path(config_path).resolve())
    resolved["_run_fingerprint"] = run_fingerprint(config, data_config)
    run_path = data_config.get("run_path")
    if run_path:
        ds_meta = meta_for_run(run_path)
        if ds_meta is not None:
            resolved["_dataset_meta"] = ds_meta
    return resolved


def write_resolved_run_config(
    run_dir: str | Path,
    config: dict,
    *,
    config_path: str | Path,
    ts: str,
    data_config: dict,
) -> Path:
    """Write ``resolved_config.yaml`` and return its path."""
    path = Path(run_dir) / "resolved_config.yaml"
    resolved = build_resolved_run_config(config, config_path=config_path, ts=ts, data_config=data_config)
    with open(path, "w") as f:
        yaml.safe_dump(resolved, f, sort_keys=False)
    return path


def partial_run_error(failed_qids: list[str], *, phase: str, aggregate_label: str) -> RuntimeError:
    """Build the standard refusal error for partial aggregate-producing runs."""
    preview = ", ".join(failed_qids[:10])
    more = "" if len(failed_qids) <= 10 else f" (+{len(failed_qids) - 10} more)"
    return RuntimeError(
        f"{len(failed_qids)} queries failed during {phase}: {preview}{more}. "
        f"Refusing to produce {aggregate_label} from a partial run."
    )
