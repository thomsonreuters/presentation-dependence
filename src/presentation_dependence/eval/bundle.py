"""Run several evaluation surfaces in one job with one model load.

A bundle evaluates one model configuration on several datasets in a single
run. It loads the reranker once and reuses it across datasets, reducing
model-loading time without changing the computation for each dataset. For a
large base on a small surface, that load dominates the cost.

Bundles reference existing per-surface configs in
``configs/experiments/<id>.yaml``. Each config retains its reranker settings,
data channel, evaluation measures, and optional ``robustness:`` block. The
driver:

1. checks that all members share the model configuration required for reuse;
   Only the per-surface prompt knobs (``instruction``, ``max_doc_chars``) may
   differ, and ``ExperimentManager._apply_prompt_overrides`` reapplies them;
2. builds the reranker from the first member, then runs
   ``ExperimentManager`` -> ``EvalManager`` -> (optional) ``PsiExperimentRunner``
   for each member with the shared reranker; and
3. writes each surface to ``<runs_root>/<member_id>/<ts>/`` so the standard
   recovery path remains unchanged. A timeout
   loses at most the active surface or query; completed surfaces remain on disk
   for the normal top-up path.

Limited I/O and dependency injection allow the driver to run offline with
``FixtureLoader`` and ``IdentityReranker``.
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any, Callable

# Drivers configure loggers under their class names, and ``setup_logging``
# caches each name with an ``is_configured`` flag. Without a reset, only the
# first surface installs file handlers; later surfaces write their logs to the
# first run directory. Resetting between surfaces points each logger to the
# correct directory.
_RUN_SCOPED_LOGGERS = (
    "ExperimentManager",
    "EvalManager",
    "PsiExperimentRunner",
    "PoolPerturbationRunner",
    "ContextDecompositionRunner",
    "MatchedVarianceControlRunner",
)


def _reset_run_scoped_loggers() -> None:
    for name in _RUN_SCOPED_LOGGERS:
        lg = logging.getLogger(name)
        for h in list(lg.handlers):
            try:
                h.close()
            except Exception:  # pragma: no cover - best-effort handler teardown
                pass
            lg.removeHandler(h)
        if getattr(lg, "is_configured", False):
            lg.is_configured = False


def _reset_reranker_transient_state(reranker) -> None:
    """Clear mutable state before reusing a reranker on another surface.

    PSI normally resets ``render_variant`` after each rank call. A crash between
    assignment and reset would otherwise carry the variant into the next
    surface. ``ExperimentManager`` reapplies prompt settings separately.
    """
    if getattr(reranker, "render_variant", None) is not None:
        reranker.render_variant = None


# These reranker keys may differ across bundled surfaces without changing the
# loaded model or engine. All other ``reranker:`` values must match.
PER_SURFACE_RERANKER_KEYS = {"instruction", "max_doc_chars"}


def model_signature(reranker_cfg: dict) -> dict:
    """Return the model-identifying config after removing prompt settings."""
    return {k: v for k, v in (reranker_cfg or {}).items() if k not in PER_SURFACE_RERANKER_KEYS}


# Shared-base ("Case B") bundles also permit different LoRA adapters. One base
# engine serves the base model and all adapters, switching adapters by surface.
SHARED_BASE_VARYING_KEYS = {"lora_path", "lora_adapters", "max_lora_rank"}


def assert_shared_model(member_cfgs: list[dict], *, allow_varying: set[str] | None = None) -> None:
    """Require bundle members to share one reusable model configuration.

    ``allow_varying`` adds ``reranker:`` keys that may differ between members.
    Shared-base bundles use it for LoRA adapter settings.
    """
    if not member_cfgs:
        raise ValueError("bundle has no member configs")
    per_member = PER_SURFACE_RERANKER_KEYS | (allow_varying or set())

    def sig(reranker_cfg: dict) -> dict:
        # Ignore per-member keys and null values. For model identity,
        # ``revision: null`` is equivalent to omitting ``revision``; treating them
        # differently would reject compatible base and adapter configs.
        return {k: v for k, v in (reranker_cfg or {}).items() if k not in per_member and v is not None}

    base = sig(member_cfgs[0].get("reranker", {}))
    base_id = member_cfgs[0].get("id")
    for cfg in member_cfgs[1:]:
        s = sig(cfg.get("reranker", {}))
        if s != base:
            diffs = sorted(k for k in set(base) | set(s) if base.get(k) != s.get(k))
            raise ValueError(
                f"bundle members must share one model config; {cfg.get('id')!r} differs "
                f"from {base_id!r} on reranker keys {diffs}. Only {sorted(per_member)} "
                f"may vary across a bundle."
            )

    # Prompt overrides are applied only when the member config supplies them.
    # Requiring each key on all members or none prevents an omitted key from
    # inheriting the previous surface's value.
    for key in sorted(PER_SURFACE_RERANKER_KEYS):
        present = [cfg.get("id") for cfg in member_cfgs if key in (cfg.get("reranker") or {})]
        if present and len(present) != len(member_cfgs):
            missing = [cfg.get("id") for cfg in member_cfgs if key not in (cfg.get("reranker") or {})]
            raise ValueError(
                f"bundle: reranker.{key!r} is set on {present} but omitted on {missing}; "
                f"specify it on all members or none (an omitting surface would inherit the "
                f"prior surface's value on the reused reranker)."
            )


def _collect_adapters(member_cfgs: list[dict]) -> list[str]:
    """Return unique member ``reranker.lora_path`` values in member order.

    Members without ``lora_path`` use the base model. Equal paths identify the
    same mounted adapter.
    """
    paths: list[str] = []
    for cfg in member_cfgs:
        p = (cfg.get("reranker") or {}).get("lora_path")
        if p and str(p) not in paths:
            paths.append(str(p))
    return paths


def run_bundle(  # noqa: C901
    member_cfgs: list[dict],
    *,
    output_dir: str | Path,
    runs_root: str | Path | None = None,
    tmp_root: str | Path | None = None,
    config_writer: Callable[[dict, Path], Path] | None = None,
    logger: Any | None = None,
    emit_base_metrics: Callable[[dict, list[str]], None] | None = None,
    emit_psi_metrics: Callable[[dict], None] | None = None,
    emit_pool_metrics: Callable[[dict], None] | None = None,
    emit_context_metrics: Callable[[dict], None] | None = None,
    emit_matched_variance_metrics: Callable[[dict], None] | None = None,
    shared_base: bool = False,
) -> list[dict]:
    """Run each member config serially, reusing one loaded reranker.

    ``member_cfgs`` contains resolved configs whose paths refer to readable
    local or mounted files. The function returns one summary per member and
    writes ``<output_dir>/bundle_manifest.json`` with member status and headline
    metrics.

    The optional metric callbacks run immediately after the corresponding pass,
    letting a caller emit per-surface ``METRIC`` lines before the run finishes.
    """
    from presentation_dependence.eval import EvalManager, ExperimentManager

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_root = Path(runs_root) if runs_root is not None else output_dir / "runs"
    tmp_root = Path(tmp_root) if tmp_root is not None else Path("/tmp/slm_bundle")
    tmp_root.mkdir(parents=True, exist_ok=True)

    if config_writer is None:
        from presentation_dependence.utils.config import write_resolved_config as config_writer  # type: ignore

    def _log(msg: str) -> None:
        if logger is not None:
            logger.info(msg)
        else:
            print(f"[bundle] {msg}", flush=True)

    shared_reranker = None
    if shared_base:
        # Case B loads one base model with every member adapter, then selects the
        # active adapter for each surface. Adapter settings may vary; the rest of
        # the model configuration must match.
        assert_shared_model(member_cfgs, allow_varying=SHARED_BASE_VARYING_KEYS)
        adapters = _collect_adapters(member_cfgs)
        from presentation_dependence.eval.runner_setup import instantiate_reranker

        boot_cfg = copy.deepcopy(member_cfgs[0])
        boot_rc = boot_cfg.setdefault("reranker", {})
        boot_rc.pop("lora_path", None)
        if adapters:
            boot_rc["lora_adapters"] = adapters
            # The first member may be the base model and omit max_lora_rank.
            # Use the max across adapter members so vLLM accepts every mounted
            # adapter.
            ranks = [
                int(r) for cfg in member_cfgs if (r := (cfg.get("reranker") or {}).get("max_lora_rank")) is not None
            ]
            if ranks:
                boot_rc["max_lora_rank"] = max(ranks)
        shared_reranker = instantiate_reranker(boot_cfg, logger)
        if adapters:
            # Check adapter switching before any surface runs. Unsupported engines
            # raise here instead of after producing partial results.
            shared_reranker.set_active_adapter(adapters[0])
            shared_reranker.set_active_adapter(None)
        _log(
            f"shared-base: loaded {type(shared_reranker).__name__} once with base + "
            f"{len(adapters)} adapter(s); switching active adapter per surface"
        )
    else:
        assert_shared_model(member_cfgs)

    summaries: list[dict] = []
    n = len(member_cfgs)
    n_ok = 0
    for i, cfg in enumerate(member_cfgs):
        member_id = str(cfg["id"])
        _log(f"surface {i + 1}/{n}: {member_id}")
        # Point each run-scoped logger at this surface's directory.
        _reset_run_scoped_loggers()
        if shared_reranker is not None:
            _reset_reranker_transient_state(shared_reranker)
        # Isolate failures by surface so completed output survives a later failure.
        # Failed surfaces are recorded for a later top-up run.
        run_dir = None
        try:
            resolved_path = config_writer(cfg, tmp_root / f"{member_id}.yaml")

            if cfg.get("matched_variance_control"):
                from presentation_dependence.eval.matched_variance_control import (
                    MatchedVarianceControlRunner,
                )

                matched = MatchedVarianceControlRunner(
                    config_path=resolved_path,
                    runs_root=runs_root,
                    reranker=shared_reranker,
                )
                run_dir = matched.run_dir
                matched_metrics = matched.run()
                if shared_reranker is None:
                    shared_reranker = matched.reranker
                    _log(f"loaded reranker {type(shared_reranker).__name__} once; reusing for remaining surfaces")
                if emit_matched_variance_metrics is not None:
                    emit_matched_variance_metrics(matched_metrics)
                matched_metrics_path = run_dir / "matched_variance" / "matched_variance_metrics.json"
                summaries.append(
                    {
                        "id": member_id,
                        "status": "ok",
                        "run_dir": str(run_dir),
                        "metrics_path": None,
                        "psi_metrics_path": None,
                        "pool_metrics_path": None,
                        "context_decomposition_metrics_path": None,
                        "matched_variance_metrics_path": (
                            str(matched_metrics_path) if matched_metrics_path.exists() else None
                        ),
                        "mean_ndcg_cut_10": None,
                    }
                )
                n_ok += 1
                continue

            if cfg.get("context_decomposition"):
                from presentation_dependence.eval.context_decomposition import (
                    ContextDecompositionRunner,
                )

                context = ContextDecompositionRunner(
                    config_path=resolved_path,
                    runs_root=runs_root,
                    reranker=shared_reranker,
                )
                run_dir = context.run_dir
                context_metrics = context.run()
                if shared_reranker is None:
                    shared_reranker = context.reranker
                    _log(f"loaded reranker {type(shared_reranker).__name__} once; reusing for remaining surfaces")
                if emit_context_metrics is not None:
                    emit_context_metrics(context_metrics)
                context_metrics_path = run_dir / "context_decomposition" / "context_decomposition_metrics.json"
                summaries.append(
                    {
                        "id": member_id,
                        "status": "ok",
                        "run_dir": str(run_dir),
                        "metrics_path": None,
                        "psi_metrics_path": None,
                        "pool_metrics_path": None,
                        "context_decomposition_metrics_path": (
                            str(context_metrics_path) if context_metrics_path.exists() else None
                        ),
                        "mean_ndcg_cut_10": None,
                    }
                )
                n_ok += 1
                continue

            if cfg.get("pool_perturbation"):
                from presentation_dependence.eval.pool_perturbation import PoolPerturbationRunner

                pool = PoolPerturbationRunner(
                    config_path=resolved_path,
                    runs_root=runs_root,
                    reranker=shared_reranker,
                )
                run_dir = pool.run_dir
                pool_metrics = pool.run()
                if shared_reranker is None:
                    shared_reranker = pool.reranker
                    _log(f"loaded reranker {type(shared_reranker).__name__} once; reusing for remaining surfaces")
                if emit_pool_metrics is not None:
                    emit_pool_metrics(pool_metrics)
                pool_metrics_path = run_dir / "pool_perturbation" / "pool_metrics.json"
                summaries.append(
                    {
                        "id": member_id,
                        "status": "ok",
                        "run_dir": str(run_dir),
                        "metrics_path": None,
                        "psi_metrics_path": None,
                        "pool_metrics_path": (str(pool_metrics_path) if pool_metrics_path.exists() else None),
                        "context_decomposition_metrics_path": None,
                        "mean_ndcg_cut_10": None,
                    }
                )
                n_ok += 1
                continue

            exp = ExperimentManager(
                config_path=resolved_path,
                runs_root=runs_root,
                reranker=shared_reranker,
            )
            run_dir = exp.run_dir  # available post-construction; partial results persist here
            exp.run()
            if shared_reranker is None:
                shared_reranker = exp.reranker  # load once; reuse for the rest
                _log(f"loaded reranker {type(shared_reranker).__name__} once; reusing for remaining surfaces")

            metrics = EvalManager(run_dir=run_dir).run()
            metrics_path = run_dir / "metrics.json"
            # Emit base metrics before the longer PSI pass, matching the
            # single-job entrypoint.
            measures = list((cfg.get("eval") or {}).get("measures") or [])
            summary = " ".join(
                f"{m}={metrics['mean_' + m]:.4f}"
                for m in measures
                if isinstance(metrics.get("mean_" + m), (int, float))
            )
            if summary:
                _log(f"surface {member_id}: {summary}")
            if emit_base_metrics is not None:
                emit_base_metrics(metrics, measures)
            ndcg = metrics.get("mean_ndcg_cut_10")

            psi_metrics_path = None
            if cfg.get("robustness"):
                from presentation_dependence.eval.psi_manager import PsiExperimentRunner

                _log(f"surface {member_id}: robustness block present; running PSI")
                psi = PsiExperimentRunner(
                    config_path=resolved_path,
                    run_dir=run_dir,
                    reranker=shared_reranker,
                ).run()
                psi_metrics_path = run_dir / "psi" / "psi_metrics.json"
                if emit_psi_metrics is not None:
                    emit_psi_metrics(psi)
                headline = (psi.get("aggregate", {}) or {}).get("zeng_psi_corpus")
                if isinstance(headline, (int, float)):
                    _log(f"surface {member_id}: zeng_psi_corpus={headline:.4f}")

            summaries.append(
                {
                    "id": member_id,
                    "status": "ok",
                    "run_dir": str(run_dir),
                    "metrics_path": str(metrics_path) if metrics_path.exists() else None,
                    "psi_metrics_path": (
                        str(psi_metrics_path) if psi_metrics_path and psi_metrics_path.exists() else None
                    ),
                    "pool_metrics_path": None,
                    "context_decomposition_metrics_path": None,
                    "mean_ndcg_cut_10": ndcg if isinstance(ndcg, (int, float)) else None,
                }
            )
            n_ok += 1
        except Exception as exc:  # noqa: BLE001 - isolate one surface; report + continue
            import traceback

            _log(f"surface {member_id}: FAILED: {exc!r}")
            _log(traceback.format_exc())
            summaries.append(
                {
                    "id": member_id,
                    "status": "failed",
                    "error": repr(exc),
                    "run_dir": str(run_dir) if run_dir is not None else None,
                    "metrics_path": None,
                    "psi_metrics_path": None,
                    "pool_metrics_path": None,
                    "context_decomposition_metrics_path": None,
                    "mean_ndcg_cut_10": None,
                }
            )

    n_failed = n - n_ok
    manifest = {
        "n_surfaces": n,
        "n_ok": n_ok,
        "n_failed": n_failed,
        "model": model_signature(member_cfgs[0].get("reranker", {})),
        "surfaces": summaries,
    }
    (output_dir / "bundle_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    _log(f"bundle complete: {n_ok}/{n} surfaces ok, {n_failed} failed -> {output_dir / 'bundle_manifest.json'}")
    if n_ok == 0:
        raise RuntimeError(f"bundle: all {n} surfaces failed; see per-surface logs / bundle_manifest.json")
    return summaries
