#!/usr/bin/env python
r"""Container entry point: YAML-driven for any experiment config.

Not the local runner. This file assumes the layout of the images in
``scripts/train-sft/`` and ``scripts/train-vllm/``: data arrives on mounted
channels, artifacts go to a mounted output directory, and TensorBoard and
checkpoints default under ``/opt/ml``. On a bare host use
``scripts/run_experiment.py``, ``run_psi.py``, ``run_bundle.py``,
``run_silver_generation.py``, or ``run_self_distill_sft.py``, which take those
paths from the config. Artifact layouts differ between the two: see
``docs/RUN-ARTIFACTS.md``.

Underneath it calls the same evaluation and training code as those runners,
with three adaptations for the container layout:

1. The experiment YAML is read from the mounted code bundle rather than
   hand-rolled from argparse flags.
2. ``data.run_path`` and ``eval.qrels_path`` are rewritten to point at the
   mounted input channel (the legacy-compatible fallback is
   ``/opt/ml/input/data/<channel>/...``).
3. ``data.dataloader_class`` is forced to ``FixtureLoader``: the
   container has no Pyserini/Java/index, only a pre-built
   ``fixture.jsonl`` on a mounted channel. Build one with
   ``scripts/data/build_fixture_pyserini.py``.

Reranker choice, hyperparameters, eval measures, and logging level flow
through unchanged.

Job-runner arguments
-------------------------------------------------------------------------
``--config-path <path>`` (required)
    Path to the experiment YAML inside the code bundle. Conventionally
    ``configs/experiments/<ID>.yaml``.

``--override <key=value>`` (repeatable, 0..N)
    Dotted-key overrides applied to the YAML before the container-side
    path rewrites. Same syntax as ``scripts/run_experiment.py``, e.g.
    ``--override reranker.batch_size=64 --override data.k_input=50``.
    See ``presentation_dependence.utils.config`` for the full semantics.

``--overrides-b64 <base64>`` (optional)
    Alternative to ``--override`` for runners that mangle quoting. Accepts a base64-encoded
    JSON list of ``"key=value"`` strings. Base64 is used (not raw JSON)
    because a job runner that re-parses JSON-looking values would strip
    inner double-quotes on the argv round-trip. Merged in on top of any
    ``--override`` flags.

``--channel <name>`` (optional)
    Input channel name. Defaults to the value derived from the
    YAML by ``resolve_channel()``. The launcher is expected to set this
    explicitly so the container never has to guess.

METRIC lines
------------
For each metric m in ``config["eval"]["measures"]``, emits exactly one
line of the form ``METRIC <m>=<float>``. Matches the regex
``METRIC <m>=([0-9\\.]+)``, a shape a job runner can scrape from the log
into its own metric store.

Environment reference
---------------------
``SLM_CHANNEL_<NAME>`` / legacy ``SM_CHANNEL_<NAME>``  a mounted data channel;
                                legacy fallback ``/opt/ml/input/data/<channel>/``.
``SLM_OUTPUT_DIR`` / legacy ``SM_OUTPUT_DATA_DIR``     where artifacts are written;
                                legacy fallback ``/opt/ml/output/data``.
``SLM_NUM_GPUS`` / legacy ``SM_NUM_GPUS``              GPU count; autodetected
                                when neither is set.
``SLM_TRIAL_NAME`` / legacy ``TRAINING_JOB_NAME``      run-directory trial id;
                                defaults to a local timestamp.

The ``SLM_`` names take precedence. The ``SM_`` aliases and ``/opt/ml``
defaults are retained only for legacy container compatibility. ``--channel``
overrides the channel lookup entirely.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import copy
import json
import os
import sys
from pathlib import Path

from presentation_dependence.self_distill.local_dp import run_teacher_local_dp, worker_count
from presentation_dependence.utils.config import apply_bundle_member_overrides
from presentation_dependence.utils.gpu import (
    distributed_mode as _distributed_mode,
    in_torchrun_worker,
    log_cuda_diagnostics,
    maybe_reexec_torchrun,
    num_gpus as _num_gpus,
)


def _decode_b64_json(payload: str) -> object:
    """Decode a base64-encoded JSON payload (urlsafe, padded or not).

    A job runner's argument dispatcher may silently re-parse any value
    that happens to look like JSON, which mangles nested quotes on the
    argv round-trip. Encoding the JSON payload as base64 first keeps
    it opaque to that re-parser.
    """
    padding = "=" * (-len(payload) % 4)
    raw = base64.urlsafe_b64decode(payload + padding)
    return json.loads(raw.decode("utf-8"))


def _default_output_dir() -> str:
    """Resolve output with legacy SM and /opt/ml container compatibility.

    Refuses the ``/opt/ml`` fallback when that tree does not already exist.
    The run directory is created with ``parents=True``, so on a bare host the
    fallback would otherwise silently mkdir ``/opt/ml`` and write the run
    somewhere the caller never named and will not think to look.
    """
    explicit = os.environ.get("SLM_OUTPUT_DIR") or os.environ.get("SM_OUTPUT_DATA_DIR")
    if explicit:
        return explicit
    legacy = "/opt/ml/output/data"
    if not Path("/opt/ml/output").is_dir():
        raise SystemExit(
            "[entrypoint][FATAL] No output directory.\n"
            f"  Nothing set SLM_OUTPUT_DIR and the container layout ({legacy}) is "
            "not present, so this is not running inside one of the images in "
            "scripts/train-sft/ or scripts/train-vllm/.\n"
            "  On a bare host use the local runners instead; they write to "
            "runs/<id>/<timestamp>/ under the repository.\n"
            "  To use this entrypoint anyway, set SLM_OUTPUT_DIR to a writable directory."
        )
    return legacy


def _default_channel_dir(channel: str) -> str:
    """Resolve the mounted channel path from the environment.

    ``SLM_CHANNEL_<NAME>`` first, then the legacy ``SM_CHANNEL_<NAME>`` alias,
    then the legacy container path ``/opt/ml/input/data/<channel>``.
    ``--channel`` overrides all three.

    Falling through to ``/opt/ml`` outside a container means nothing declared a
    channel, which is what running this file on a bare host looks like. Say so
    here, because the alternative is a bare "no such file" naming a path the
    caller never chose and cannot create.
    """
    suffix = channel.upper().replace("-", "_").replace(".", "_")
    explicit = os.environ.get(f"SLM_CHANNEL_{suffix}") or os.environ.get(f"SM_CHANNEL_{suffix}")
    if explicit:
        return explicit
    legacy = f"/opt/ml/input/data/{channel}"
    if not Path("/opt/ml/input").is_dir():
        raise SystemExit(
            f"[entrypoint][FATAL] No input channel for {channel!r}.\n"
            f"  Nothing set SLM_CHANNEL_{suffix} and the container layout "
            f"({legacy}) is not present, so this is not running inside one of "
            "the images in scripts/train-sft/ or scripts/train-vllm/.\n"
            "  On a bare host use the local runners instead: scripts/run_experiment.py, "
            "run_psi.py, run_bundle.py, run_silver_generation.py, or "
            "run_self_distill_sft.py, which read paths from the config.\n"
            "  To use this entrypoint anyway, pass --input-dir or set "
            f"SLM_CHANNEL_{suffix} to the directory holding the channel's data."
        )
    return legacy


def _student_resolved_config_path(tmp_root: Path, exp_id: str) -> Path:
    """Return a temp config path that is safe under torchrun DDP.

    Every local DDP rank executes ``_run_self_distill_student``. If they all
    write the resolved config to the same path, one rank can read while another
    rank is truncating/re-writing the file, yielding a partially parsed YAML
    and ``KeyError: 'silver_labels_path'`` at startup.
    Keep single-process behavior stable, but isolate DDP workers by global
    rank.
    """
    if os.environ.get("SLM_TORCHRUN_WORKER") != "1":
        return tmp_root / f"{exp_id}.yaml"
    rank = os.environ.get("RANK") or os.environ.get("LOCAL_RANK") or "unknown"
    return tmp_root / f"{exp_id}-rank{rank}.yaml"


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-path",
        type=str,
        required=True,
        help="YAML path inside the code bundle, e.g. configs/experiments/<ID>.yaml.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Dotted-key override, e.g. reranker.batch_size=64. Repeatable.",
    )
    parser.add_argument(
        "--overrides-b64",
        type=str,
        default=None,
        help=(
            "Base64-encoded JSON list of overrides. Preferred when a runner "
            "channel (avoids hyperparameter re-parser shell-escape bugs)."
        ),
    )
    parser.add_argument(
        "--overrides-json",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--channel",
        type=str,
        default=None,
        help="Input channel name. Defaults to resolve_channel(cfg).",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Override the mounted channel path (otherwise SLM_CHANNEL_<CHANNEL>, legacy SM alias, then /opt/ml).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=f"Where to copy the run_dir + metrics.json (default: {_default_output_dir()}).",
    )
    return parser


def _rewrite_channel_file_by_basename(block: dict, key: str, input_dir: Path) -> None:
    raw = block.get(key)
    if raw:
        block[key] = str(input_dir / Path(str(raw)).name)


def _rewrite_paths_to_channel(cfg: dict, input_dir: Path) -> dict:
    """Rewrite data.run_path and eval.qrels_path to live under ``input_dir``.

    A staged bundle always has the three-file layout
    ``<channel>/{fixture.jsonl, qrels.txt, topics.tsv}``.
    Replace the YAML's laptop-side paths with the mounted channel paths.
    """
    data = cfg.setdefault("data", {})
    data["dataloader_class"] = "FixtureLoader"
    data["run_path"] = str(input_dir / "fixture.jsonl")

    ev = cfg.setdefault("eval", {})
    ev["qrels_path"] = str(input_dir / "qrels.txt")
    _rewrite_channel_file_by_basename(cfg, "qids_to_run_path", input_dir)
    robustness = cfg.get("robustness") or {}
    _rewrite_channel_file_by_basename(robustness, "qids_to_run_path", input_dir)
    pool_perturbation = cfg.get("pool_perturbation") or {}
    _rewrite_channel_file_by_basename(pool_perturbation, "qids_to_run_path", input_dir)
    beta_gamma = robustness.get("beta_gamma") or {}
    if beta_gamma.get("permutation_manifest_path"):
        manifest_name = Path(str(beta_gamma["permutation_manifest_path"])).name
        beta_gamma["permutation_manifest_path"] = str(input_dir / manifest_name)
    return cfg


def _emit_metric_lines(metrics: dict, measures: list[str]) -> None:
    """Emit ``METRIC <name>=<value>`` for every configured measure.

    The experiment's ``eval.measures`` list is canonical for emitted metrics;
    the launcher builds ``metric_definitions`` from the same list, so every
    declared measure gets scraped.
    """
    for measure in measures:
        key = f"mean_{measure}"
        if key not in metrics:
            print(f"[entrypoint][WARN] no {key} in metrics.json", flush=True)
            continue
        value = metrics[key]
        if not isinstance(value, (int, float)):
            print(f"[entrypoint][WARN] {key} is not numeric: {value!r}", flush=True)
            continue
        print(f"METRIC {measure}={value:.6f}", flush=True)


# Robustness headline metrics to scrape from the log when a PSI run is
# present. Kept in sync with ``PsiExperimentRunner.run()``'s `aggregate:`
# payload (see ``src/presentation_dependence/eval/psi.py::_aggregate``). The launcher
# passes all of these into ``metric_definitions`` so they show up under
# the scraped metric series alongside nDCG.
ROBUSTNESS_METRICS: tuple[str, ...] = (
    "zeng_psi_corpus",  # leaderboard-comparable (Zeng et al. Table 1)
    "mean_zeng_psi",  # per-query averaged PSI (noisier, useful for spread)
    "mean_kendall_tau",  # ranking-stability across K perms
    "mean_tau_based_psi",  # τ transformed into a PSI-style score
    "mean_delta_ndcg",  # only when form-homogeneous (stratified OR max_min)
    "mean_delta_ndcg_stratified",  # lost-in-the-middle signal (can be negative)
    "mean_delta_ndcg_max_min",  # plain stability proxy (always ≥0)
    "mean_rank_variance",  # per-doc rank variance across K
    "mean_gold_rank_variance",  # qrels-positive doc rank variance across K
    "mean_score_variance",  # per-doc score variance (None for rank-only rerankers)
)

# Protocol METRIC lines (scraped like robustness aggregates when ``robustness:`` present):
# ``METRIC psi_permutation_K=…``: random-shuffle count from ``robustness.K`` / psi JSON.
# ``METRIC tau_psi_context_batch_size=…``: headline B for τ-PSI@B (top-level PSI JSON).


def _emit_psi_K_metric_lines(psi_metrics: dict) -> None:
    """Emit permutation count **K** for log scraping.

    Mirrors top-level ``K`` in ``psi/psi_metrics.json`` (``robustness.K`` in YAML).
    """
    k = psi_metrics.get("K")
    if k is None:
        return
    if not isinstance(k, (int, float)):
        print(f"[entrypoint][WARN] psi_metrics.K not numeric: {k!r}", flush=True)
        return
    print(f"METRIC psi_permutation_K={int(k)}", flush=True)


def _emit_robustness_metric_lines(psi_metrics: dict) -> None:
    """Emit ``METRIC <name>=<value>`` for every robustness aggregate that
    came back non-None.

    `zeng_psi_corpus` lives at the top level of ``aggregate``; everything
    else is a `mean_*` key. None values (e.g. `score_variance` for a
    rank-only reranker like RankZephyr, or `delta_ndcg` for a non-
    homogeneous run) are skipped silently: an expected absence, not an error.
    """
    agg = psi_metrics.get("aggregate", {})
    for name in ROBUSTNESS_METRICS:
        value = agg.get(name)
        if value is None:
            continue
        if not isinstance(value, (int, float)):
            print(f"[entrypoint][WARN] robustness metric {name} is not numeric: {value!r}", flush=True)
            continue
        print(f"METRIC {name}={value:.6f}", flush=True)


def _emit_tau_psi_B_metric_lines(psi_metrics: dict) -> None:
    """Emit **B** for τ-PSI@B scraping (paired with ``mean_tau_based_psi``).

    Mirrors ``tau_psi_context_batch_size`` / ``tau_psi_at_B_caption`` in
    ``psi/psi_metrics.json``.
    """
    b = psi_metrics.get("tau_psi_context_batch_size")
    if b is None:
        return
    if not isinstance(b, (int, float)):
        print(f"[entrypoint][WARN] tau_psi_context_batch_size not numeric: {b!r}", flush=True)
        return
    print(f"METRIC tau_psi_context_batch_size={int(b)}", flush=True)


def _emit_pool_metric_lines(pool_metrics: dict) -> None:
    """Emit the fixed-order pool-perturbation headline metrics."""
    for perturbation, block in (pool_metrics.get("aggregate") or {}).items():
        fields = {
            f"{perturbation}_pool_psi": block.get("mean_pool_psi"),
            f"{perturbation}_top_k_set_flip_rate": block.get("top_k_set_flip_rate"),
            f"{perturbation}_delta_ndcg_retained": block.get("mean_delta_ndcg_at_k_retained"),
        }
        for name, value in fields.items():
            if isinstance(value, (int, float)):
                print(f"METRIC {name}={value:.6f}", flush=True)


def _emit_context_decomposition_metric_lines(metrics: dict) -> None:
    """Emit Category-H ranking decomposition effects."""
    aggregate = metrics.get("aggregate") or {}
    for reading in ("canonical", "order_averaged_k10"):
        if metrics.get("protocol", {}).get("mode") == "a4_screen":
            value = (aggregate.get(reading) or {}).get("cumulative_skeleton", {}).get("mean")
            if isinstance(value, (int, float)):
                print(
                    f"METRIC context_{reading}_cumulative_skeleton={value:.6f}",
                    flush=True,
                )
            continue
        decomposition = (aggregate.get(reading) or {}).get("decomposition") or {}
        for name, block in decomposition.items():
            value = block.get("mean") if isinstance(block, dict) else None
            if isinstance(value, (int, float)):
                print(
                    f"METRIC context_{reading}_{name}={value:.6f}",
                    flush=True,
                )


def _emit_matched_variance_metric_lines(metrics: dict) -> None:
    """Emit matched-variance repeated-B=1 headline metrics."""
    aggregate = metrics.get("aggregate") or {}
    for name, value in aggregate.items():
        if isinstance(value, (int, float)):
            print(f"METRIC matched_variance_{name}={value:.6f}", flush=True)


def _run_eval_bundle(cfg: dict, bundle_config_path: Path, output_dir: Path) -> int:  # noqa: C901
    """Run a multi-surface eval bundle (one model load, N datasets).

    The bundle YAML carries ``bundle.member_configs: [<exp_id>, ...]`` referencing
    existing per-surface configs that live alongside it. Each member is loaded,
    its data/qrels paths rewritten to *its own* mounted channel, then
    all members run serially via ``presentation_dependence.eval.bundle.run_bundle`` reusing
    one loaded reranker. Per-surface output lands under
    ``<output_dir>/runs/<member_id>/<ts>/``, the standard run-tree shape.
    """
    from presentation_dependence.eval.bundle import run_bundle
    from presentation_dependence.utils.config import load_experiment_config, resolve_channel

    members = (cfg.get("bundle") or {}).get("member_configs") or []
    if not members:
        print("[entrypoint][bundle][FATAL] bundle.member_configs is empty", file=sys.stderr)
        return 2
    if cfg.get("reranker"):
        print(
            "[entrypoint][bundle][FATAL] top-level reranker overrides are ignored by bundle configs; "
            "use bundle.reranker_overrides.* instead.",
            file=sys.stderr,
        )
        return 2

    bundle_cfg = cfg.get("bundle") or {}

    bundle_dir = Path(bundle_config_path).parent
    member_cfgs: list[dict] = []
    for member in members:
        member_s = str(member)
        member_path = Path(member_s) if member_s.endswith(".yaml") else bundle_dir / f"{member_s}.yaml"
        if not member_path.exists():
            print(f"[entrypoint][bundle][FATAL] member config not found: {member_path}", file=sys.stderr)
            return 2
        _mpath, member_cfg = load_experiment_config(str(member_path))
        member_cfg = apply_bundle_member_overrides(member_cfg, bundle_cfg)
        channel = resolve_channel(member_cfg)
        member_input_dir = Path(_default_channel_dir(channel))
        member_cfg = _rewrite_paths_to_channel(member_cfg, member_input_dir)
        print(
            f"[entrypoint][bundle] member={member_cfg.get('id')} channel={channel} input_dir={member_input_dir}",
            flush=True,
        )
        member_cfgs.append(member_cfg)

    # Emit METRIC scrape lines per surface, INLINE as each pass finishes (same
    # timing as the single-job path), so a monitor isn't blank and a timeout
    # still captures the surfaces already done. The same measure name recurs
    # once per surface -> a time series in the UI (per-surface metrics.json remain
    # the canonical artifacts).
    def _emit_psi(psi_metrics: dict) -> None:
        _emit_psi_K_metric_lines(psi_metrics)
        _emit_tau_psi_B_metric_lines(psi_metrics)
        _emit_robustness_metric_lines(psi_metrics)

    shared_base = bool((cfg.get("bundle") or {}).get("shared_base", False))
    if shared_base:
        print("[entrypoint][bundle] shared_base=true — one base load serves the base + adapters", flush=True)
    summaries = run_bundle(
        member_cfgs,
        output_dir=output_dir,
        runs_root=output_dir / "runs",
        tmp_root=Path("/tmp/slm_bundle"),
        emit_base_metrics=_emit_metric_lines,
        emit_psi_metrics=_emit_psi,
        emit_pool_metrics=_emit_pool_metric_lines,
        emit_context_metrics=_emit_context_decomposition_metric_lines,
        emit_matched_variance_metrics=_emit_matched_variance_metric_lines,
        shared_base=shared_base,
    )

    n_ok = sum(1 for s in summaries if s.get("status") == "ok")
    for s in summaries:
        if s.get("status") != "ok":
            print(f"[entrypoint][bundle] FAILED surface {s.get('id')}: {s.get('error')}", flush=True)
    n_failed = len(summaries) - n_ok
    print(f"[entrypoint][bundle] {n_ok} ok, {n_failed} failed of {len(summaries)} surfaces", flush=True)
    return 1 if n_failed else 0


def _run_self_distill_teacher(cfg: dict, input_dir: Path, output_dir: Path) -> int:
    """Container entry path for K-shot BSC silver-data generation.

    Invoked when the YAML carries a ``teacher:`` block. Runs
    :class:`presentation_dependence.self_distill.teacher.KShotBSCTeacher` against the
    mounted input channel and writes:

    - ``<output_dir>/runs/<exp_id>/<trial>/silver/{silver_labels.jsonl,manifest.json}``:
      under the same ``runs/<exp_id>/<trial>/`` shape that
      ``ExperimentManager`` produces for quality runs, so a collector needs no
      special-casing.
    - ``<output_dir>/silver_labels.jsonl`` and ``<output_dir>/manifest.json``
      at the tarball root: symmetric with the ``metrics.json`` /
      ``psi_metrics.json`` duplication in the standard path; lets a caller
      read headline silver provenance without unpacking the
      whole tarball.

    Container runs land at ``runs/<exp_id>/<trial>/silver/`` (no
    ``runs/self-distill/`` namespace prefix), unlike local open-weight runs from
    ``scripts/run_silver_generation.py`` which use the
    ``runs/self-distill/<id>/<ts>/silver/`` layout. The container convention
    is constrained by the ``runs/<exp_id>/<trial>/`` tarball root pattern; the
    local namespacing is convenience.
    """
    from presentation_dependence.self_distill.teacher import KShotBSCTeacher
    from presentation_dependence.utils.config import write_resolved_config
    from presentation_dependence.utils.run_paths import resolve_trial_name

    exp_id = str(cfg["id"])
    trial_name = resolve_trial_name()

    # Pre-allocate the run dir so the teacher writes under the
    # ``runs/<exp_id>/<trial>/`` shape the tarball-safety check expects.
    run_dir = output_dir / "runs" / exp_id / trial_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # The teacher reads its own config from disk; serialise the rewritten
    # cfg (with channel-rewritten data paths) into a tmp file first.
    tmp_root = Path("/tmp/slm_self_distill")
    tmp_root.mkdir(parents=True, exist_ok=True)
    resolved_path = write_resolved_config(cfg, tmp_root / f"{exp_id}.yaml")
    print(f"[entrypoint][teacher] resolved config -> {resolved_path}", flush=True)
    print(f"[entrypoint][teacher] run_dir         = {run_dir}", flush=True)

    local_dp_workers = worker_count(cfg)
    if local_dp_workers > 1:
        summary = run_teacher_local_dp(
            cfg=cfg,
            exp_id=exp_id,
            trial_name=trial_name,
            run_dir=run_dir,
            tmp_root=tmp_root,
            n_workers=local_dp_workers,
            visible_gpus=_num_gpus(),
        )
    else:
        teacher = KShotBSCTeacher(
            config_path=resolved_path,
            runs_root=str(run_dir.parents[1]),  # = output_dir/runs (unused with run_dir set)
            run_dir=run_dir,
        )
        summary = teacher.run()

    _publish_self_distill_summary(summary, output_dir, run_dir)
    return 0


def _run_self_distill_student(cfg: dict, input_dir: Path, output_dir: Path) -> int:  # noqa: C901
    """Container entry path for LoRA student SFT on completed silver labels."""
    import shutil
    from presentation_dependence.self_distill.student import train_student_sft_from_config
    from presentation_dependence.utils.config import write_resolved_config
    from presentation_dependence.utils.run_paths import resolve_trial_name

    exp_id = str(cfg["id"])
    trial_name = resolve_trial_name()
    run_dir = output_dir / "runs" / exp_id / trial_name
    student_dir = run_dir / "student"
    student_dir.mkdir(parents=True, exist_ok=True)

    # Rewrite student data paths to the mounted channel when the
    # config carries repo-relative paths. Keep explicit absolute paths intact.
    cfg = copy.deepcopy(cfg)
    student = cfg.setdefault("student", {})
    for key in ("silver_labels_path", "eval_silver_labels_path", "fixture_path", "qrels_path"):
        raw = student.get(key)
        if not raw:
            continue
        p = Path(str(raw))
        if not p.is_absolute():
            candidate = input_dir / p.name
            if candidate.exists():
                student[key] = str(candidate)
    student_data = student.setdefault("data", {})
    for key in ("train_qids_path", "exclude_qids_path"):
        raw = student_data.get(key)
        if not raw:
            continue
        p = Path(str(raw))
        if not p.is_absolute():
            candidate = input_dir / p.name
            if candidate.exists():
                student_data[key] = str(candidate)
    student["output_dir"] = str(student_dir)
    tensorboard = student.setdefault("observability", {}).setdefault("tensorboard", {})
    if tensorboard.get("enabled") and not tensorboard.get("log_dir"):
        # Legacy container location for live TensorBoard output; the run tree
        # under the output dir remains the portable artifact.
        tensorboard["log_dir"] = "/opt/ml/output/tensorboard"
    checkpoint = student.setdefault("checkpoint", {})
    if checkpoint.get("enabled") and not checkpoint.get("dir"):
        # Legacy container checkpoint location. Resumable state lives outside
        # the output tree so a resume does not mutate the run artifacts.
        checkpoint["dir"] = "/opt/ml/checkpoints/student"

    tmp_root = Path("/tmp/slm_student_sft")
    tmp_root.mkdir(parents=True, exist_ok=True)
    resolved_path = write_resolved_config(cfg, _student_resolved_config_path(tmp_root, exp_id))
    print(f"[entrypoint][student] resolved config -> {resolved_path}", flush=True)
    print(f"[entrypoint][student] output_dir      = {student_dir}", flush=True)

    summary = train_student_sft_from_config(resolved_path)
    if summary.get("is_main_process") is False:
        print(
            f"[entrypoint][student] non-main rank {summary.get('rank')} finished; artifacts are rank-0 only.",
            flush=True,
        )
        return 0
    print(f"[entrypoint][student] checkpoint = {summary['checkpoint']}", flush=True)
    print(f"[entrypoint][student] n_chunks   = {summary['n_chunks']}", flush=True)
    print(f"[entrypoint][student] steps      = {summary['global_steps']}", flush=True)
    if isinstance(summary.get("loss_initial"), (int, float)):
        print(f"METRIC student_loss_initial={float(summary['loss_initial']):.6f}", flush=True)
    if isinstance(summary.get("loss_final"), (int, float)):
        print(f"METRIC student_loss_final={float(summary['loss_final']):.6f}", flush=True)
    if isinstance(summary.get("global_steps"), (int, float)):
        print(f"METRIC student_steps={float(summary['global_steps']):.6f}", flush=True)
    if isinstance(summary.get("n_chunks"), (int, float)):
        print(f"METRIC student_n_chunks={float(summary['n_chunks']):.6f}", flush=True)
    view_counts = summary.get("view_counts") or {}
    if isinstance(view_counts, dict):
        for metric_name, key in (("student_n_groups", "groups"),):
            value = view_counts.get(key)
            if isinstance(value, (int, float)):
                print(f"METRIC {metric_name}={float(value):.6f}", flush=True)

    for fname in ("training_summary.json", "resolved_student_config.json", "progress.jsonl"):
        src = student_dir / fname
        if src.is_file():
            dst = output_dir / fname
            shutil.copyfile(src, dst)
            print(f"[entrypoint][student] {fname} -> {dst}", flush=True)
    return 0


def _publish_self_distill_summary(summary: dict, output_dir: Path, run_dir: Path) -> None:
    """Emit metrics and copy headline silver artifacts to the tarball root."""
    import shutil

    print(f"[entrypoint][teacher] silver_labels = {summary['silver_labels']}", flush=True)
    print(f"[entrypoint][teacher] manifest      = {summary['manifest']}", flush=True)
    print(
        "[entrypoint][teacher] n_queries={n_queries}  n_records={n_records}  K={k_perms}".format(**summary),
        flush=True,
    )
    for metric_name, summary_key in (
        ("silver_n_queries", "n_queries"),
        ("silver_n_records", "n_records"),
        ("silver_k_perms", "k_perms"),
    ):
        value = summary.get(summary_key)
        if isinstance(value, (int, float)):
            print(f"METRIC {metric_name}={float(value):.6f}", flush=True)

    # Mirror metrics.json / psi_metrics.json convention: copy the headline
    # artefacts to the tarball root so post-job inspectors can pull silver
    # provenance with one streaming tar command.
    silver_dir = run_dir / "silver"
    for fname in ("manifest.json", "silver_labels.jsonl"):
        src = silver_dir / fname
        if src.is_file():
            dst = output_dir / fname
            shutil.copyfile(src, dst)
            print(f"[entrypoint][teacher] {fname} -> {dst}", flush=True)


def _log_torch_cuda_diagnostics() -> None:
    """Emit CUDA diagnostics under the entrypoint's log prefix."""
    log_cuda_diagnostics("[entrypoint][diagnostics]")


def _install_tf5_vllm_compat_pth() -> None:
    """Install the tf5/vLLM shim only for an installed vLLM older than 0.12.

    ``presentation_dependence.self_distill.engines._tf5_compat`` restores
    ``PreTrainedTokenizerBase.all_special_tokens_extended`` back on tf 5.x
    for the legacy vLLM 0.10.x stack retained by the Qwen3-4B-Instruct-2507
    and Qwen3-Reranker-4B silver configs. Importing the vLLM engine applies
    the shim in the parent process. ``VLLM_WORKER_MULTIPROC_METHOD=spawn``
    starts a fresh subprocess that may not import the engine module before
    touching the tokenizer, so the spawned worker hits unpatched
    transformers and crashes with::

        AttributeError: Qwen2Tokenizer has no attribute all_special_tokens_extended

    Python's ``site`` module processes each ``.pth`` file at interpreter
    startup and executes lines starting with ``import`` as Python statements.
    Installing ``slm_tf5_vllm_compat.pth`` applies the shim in parent
    and worker processes before vLLM accesses the tokenizer. The failure first
    appeared in the 1K self-distillation pilots after the container moved
    to ``tf>=5.4``.

    Modern vLLM images contain the upstream tokenizer fix, and HF-only jobs do
    not need a worker hook. Detecting the installed distribution here prevents
    the compatibility patch from leaking into either runtime.
    """
    try:
        from importlib import metadata
    except Exception as e:  # pragma: no cover
        print(f"[entrypoint] could not inspect vLLM version for tf5 shim: {e}", flush=True)
        return

    try:
        installed_vllm = metadata.version("vllm")
    except metadata.PackageNotFoundError:
        print("[entrypoint] vLLM is not installed; skipping tf5 compatibility shim", flush=True)
        return
    try:
        version_parts = installed_vllm.split("+", 1)[0].split("-", 1)[0].split(".")
        vllm_major_minor = (int(version_parts[0]), int(version_parts[1]))
    except (IndexError, ValueError):
        print(
            f"[entrypoint] could not parse vLLM version {installed_vllm!r}; skipping tf5 compatibility shim",
            flush=True,
        )
        return
    if vllm_major_minor >= (0, 12):
        print(
            f"[entrypoint] vLLM {installed_vllm} includes the tokenizer fix; skipping tf5 compatibility shim",
            flush=True,
        )
        return

    try:
        import site
    except Exception as e:  # pragma: no cover
        print(f"[entrypoint] could not import site to install tf5 .pth: {e}", flush=True)
        return

    candidates = list(site.getsitepackages() or [])
    user_site = site.getusersitepackages()
    if user_site:
        candidates.append(user_site)

    target_dir = next((p for p in candidates if p and os.path.isdir(p)), None)
    if target_dir is None:
        print(
            "[entrypoint] no writable site-packages directory found; tf5 vLLM-subprocess shim will NOT be active.",
            flush=True,
        )
        return

    pth_path = os.path.join(target_dir, "slm_tf5_vllm_compat.pth")
    pth_line = "import presentation_dependence.self_distill.engines._tf5_compat as _t; _t.ensure_tf5_vllm_compat()\n"
    try:
        with open(pth_path, "w", encoding="utf-8") as f:
            f.write(pth_line)
        print(f"[entrypoint] installed tf5 vLLM-subprocess shim at {pth_path}", flush=True)
    except OSError as e:
        print(
            f"[entrypoint] failed to write {pth_path}: {e}; vLLM subprocess may crash on tf 5 tokenizers.",
            flush=True,
        )


def _maybe_install_flash_attn_for_xformers() -> None:
    """Optionally pin standalone ``flash-attn`` for XFORMERS backend tests.

    The retained container requirements do not pin standalone
    ``flash-attn`` because vLLM normally uses its vendored ``vllm-flash-attn``
    package. The alternate `VLLM_ATTENTION_BACKEND=XFORMERS` path, however,
    imports xformers' Flash-Attention integration and rejects `flash-attn`
    2.8.3 with:

        Requires Flash-Attention version >=2.7.1,<=2.8.2 but got 2.8.3

    Adding `flash-attn<=2.8.2` to `requirements.txt` is not viable: pip
    build isolation cannot see the image-provided torch and fails before the
    entrypoint starts. This opt-in hook runs *after* the image's requirements
    install, with torch importable, and uses `--no-build-isolation`.

    Enable with:

        execution.environment.SLM_FLASH_ATTN_VERSION: "2.8.2"

    Scoped to XFORMERS smoke configs; do not run on normal HF/vLLM jobs.
    """
    wanted = os.environ.get("SLM_FLASH_ATTN_VERSION")
    if not wanted:
        return

    try:
        from importlib import metadata
    except Exception as e:  # pragma: no cover
        print(f"[entrypoint] cannot inspect flash-attn version: {e}", flush=True)
        metadata = None

    if metadata is not None:
        try:
            current = metadata.version("flash-attn")
        except metadata.PackageNotFoundError:
            current = None
        if current == wanted:
            print(f"[entrypoint] flash-attn=={wanted} already installed", flush=True)
            return
        print(f"[entrypoint] flash-attn current={current!r}; installing {wanted}", flush=True)

    import subprocess

    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        f"flash-attn=={wanted}",
        "--no-build-isolation",
    ]
    proc = subprocess.run(cmd, text=True)
    if proc.returncode != 0:
        print(
            f"[entrypoint][FATAL] failed to install flash-attn=={wanted} "
            f"with --no-build-isolation (exit={proc.returncode})",
            flush=True,
        )
        sys.exit(2)


def main() -> int:  # noqa: C901
    args = get_parser().parse_args()

    from presentation_dependence.utils.config import (
        apply_execution_environment,
        apply_overrides,
        load_experiment_config,
        resolve_channel,
        write_resolved_config,
    )
    from presentation_dependence.utils.log_redaction import redact_override_list

    config_path, cfg = load_experiment_config(args.config_path)

    overrides: list[str] = list(args.override)
    for flag_name, raw, decoder in (
        ("--overrides-b64", args.overrides_b64, _decode_b64_json),
        ("--overrides-json", args.overrides_json, json.loads),
    ):
        if not raw:
            continue
        try:
            extra = decoder(raw)
        except (json.JSONDecodeError, binascii.Error, ValueError) as e:
            print(f"[entrypoint][FATAL] {flag_name} is not valid: {e}", file=sys.stderr)
            return 2
        if not isinstance(extra, list) or not all(isinstance(x, str) for x in extra):
            print(
                f"[entrypoint][FATAL] {flag_name} must decode to a list[str], got {extra!r}",
                file=sys.stderr,
            )
            return 2
        overrides.extend(extra)
    cfg = apply_overrides(cfg, overrides)
    apply_execution_environment(cfg)

    _install_tf5_vllm_compat_pth()
    _log_torch_cuda_diagnostics()
    _maybe_install_flash_attn_for_xformers()

    from presentation_dependence.silver_data import SilverConfigKind, validate_silver_config

    silver_kind = None
    if cfg.get("teacher"):
        try:
            silver_kind = validate_silver_config(cfg, path=config_path).kind
        except ValueError as exc:
            print(f"[entrypoint][FATAL] invalid silver config: {exc}", file=sys.stderr)
            return 2
        if silver_kind is SilverConfigKind.HOSTED:
            print(
                "[entrypoint][FATAL] hosted closed_model_generated_bsc configs "
                "run through scripts/run_silver_generation.py; the container "
                "entrypoint supports open-weight k_shot_bsc configs",
                file=sys.stderr,
            )
            return 2

    # Multi-surface bundle branch: one job, one model load, N surfaces. The
    # bundle config references existing per-surface configs (no single data
    # channel of its own), so it must be handled before the single-channel
    # resolution below. See presentation_dependence.eval.bundle.
    if cfg.get("bundle"):
        output_dir = Path(args.output_dir or _default_output_dir())
        print(f"[entrypoint] bundle config detected — exp_id={cfg.get('id')}", flush=True)
        return _run_eval_bundle(cfg, config_path, output_dir)

    channel = args.channel or resolve_channel(cfg)
    input_dir = Path(args.input_dir or _default_channel_dir(channel))
    output_dir = Path(args.output_dir or _default_output_dir())

    print(f"[entrypoint] config     = {config_path}", flush=True)
    print(f"[entrypoint] exp_id     = {cfg.get('id')}", flush=True)
    print(f"[entrypoint] channel    = {channel}", flush=True)
    print(f"[entrypoint] input_dir  = {input_dir}", flush=True)
    print(f"[entrypoint] output_dir = {output_dir}", flush=True)
    if overrides:
        print(f"[entrypoint] overrides  = {redact_override_list(overrides)}", flush=True)

    cfg = _rewrite_paths_to_channel(cfg, input_dir)
    if cfg.get("student"):
        student = cfg.setdefault("student", {})
        for key in ("silver_labels_path", "eval_silver_labels_path", "fixture_path", "qrels_path"):
            raw = student.get(key)
            if not raw:
                continue
            p = Path(str(raw))
            if not p.is_absolute():
                candidate = input_dir / p.name
                if candidate.exists():
                    student[key] = str(candidate)
        student_data = student.setdefault("data", {})
        for key in ("train_qids_path", "exclude_qids_path"):
            raw = student_data.get(key)
            if not raw:
                continue
            p = Path(str(raw))
            if not p.is_absolute():
                candidate = input_dir / p.name
                if candidate.exists():
                    student_data[key] = str(candidate)
        # adapter_init_path may be mounted as an extra input channel.
        # Resolve the configured path against the channel mount when present;
        # otherwise pass through unchanged (local path or absolute).
        student_lora = student.setdefault("lora", {})
        raw_adapter = student_lora.get("adapter_init_path")
        if raw_adapter:
            adapter_p = Path(str(raw_adapter))
            if not adapter_p.is_absolute():
                # Resolve legacy channel-relative /opt/ml paths before trying
                # the generic mounted-input sibling.
                for candidate in (
                    Path("/opt/ml/input/data") / adapter_p.name,
                    input_dir.parent / adapter_p.name,
                ):
                    if candidate.exists() and (candidate / "adapter_config.json").exists():
                        student_lora["adapter_init_path"] = str(candidate)
                        break

    if _distributed_mode(cfg) == "ddp" and cfg.get("student") and not in_torchrun_worker():
        tmp_root = Path("/tmp/slm_entrypoint_ddp")
        tmp_root.mkdir(parents=True, exist_ok=True)
        resolved_worker_config = write_resolved_config(cfg, tmp_root / f"{cfg['id']}.yaml")
        # Workers must read the already-rewritten config. Otherwise they reload
        # the repo YAML and miss mounted student data paths.
        argv = sys.argv[1:]
        worker_argv = ["--config-path", str(resolved_worker_config)]
        for flag in ("--channel", "--input-dir", "--output-dir"):
            if flag in argv:
                idx = argv.index(flag)
                if idx + 1 < len(argv):
                    worker_argv.extend([flag, argv[idx + 1]])
        ddp_exit = maybe_reexec_torchrun(
            cfg,
            worker_argv,
            script=Path(__file__).resolve(),
            log_prefix="[entrypoint][ddp]",
        )
        if ddp_exit is not None:
            return int(ddp_exit)

    needed_files = ["fixture.jsonl", "qrels.txt"]
    if cfg.get("qids_to_run_path"):
        needed_files.append(Path(str(cfg["qids_to_run_path"])).name)
    robustness = cfg.get("robustness") or {}
    if robustness.get("qids_to_run_path"):
        needed_files.append(Path(str(robustness["qids_to_run_path"])).name)
    pool_perturbation = cfg.get("pool_perturbation") or {}
    if pool_perturbation.get("qids_to_run_path"):
        needed_files.append(Path(str(pool_perturbation["qids_to_run_path"])).name)
    beta_gamma = robustness.get("beta_gamma") or {}
    if beta_gamma.get("permutation_manifest_path"):
        needed_files.append(Path(str(beta_gamma["permutation_manifest_path"])).name)
    if cfg.get("student"):
        student_data = (cfg.get("student") or {}).get("data") or {}
        for key in ("train_qids_path", "exclude_qids_path"):
            if student_data.get(key):
                needed_files.append(Path(str(student_data[key])).name)
    for needed in needed_files:
        p = input_dir / needed
        if not p.exists():
            print(f"[entrypoint][FATAL] missing input {p}", file=sys.stderr)
            return 2

    # Open-weight silver branch, selected by the validated protocol rather than
    # the config namespace. This preserves moved configs/silver jobs and rejects
    # hosted configs before they can fall through to eval/PSI.
    if silver_kind is SilverConfigKind.OPEN_WEIGHT:
        return _run_self_distill_teacher(cfg, input_dir, output_dir)
    if cfg.get("student"):
        return _run_self_distill_student(cfg, input_dir, output_dir)
    from presentation_dependence.eval import EvalManager, ExperimentManager

    tmp_root = Path("/tmp/slm_entrypoint")
    tmp_root.mkdir(parents=True, exist_ok=True)
    resolved_path = write_resolved_config(cfg, tmp_root / f"{cfg['id']}.yaml")
    print(f"[entrypoint] resolved config -> {resolved_path}", flush=True)

    runs_root = output_dir / "runs"
    if cfg.get("matched_variance_control"):
        from presentation_dependence.eval.matched_variance_control import (
            MatchedVarianceControlRunner,
        )

        print(
            "[entrypoint] matched_variance_control block present — running fixed-width request-batching control",
            flush=True,
        )
        matched_runner = MatchedVarianceControlRunner(
            config_path=resolved_path,
            runs_root=runs_root,
        )
        matched_metrics = matched_runner.run()
        _emit_matched_variance_metric_lines(matched_metrics)
        top_matched = output_dir / "matched_variance_metrics.json"
        with open(top_matched, "w", encoding="utf-8") as f:
            json.dump(
                matched_metrics,
                f,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )
        print(
            f"[entrypoint] matched_variance_metrics.json -> {top_matched}",
            flush=True,
        )
        return 0

    if cfg.get("context_decomposition"):
        from presentation_dependence.eval.context_decomposition import (
            ContextDecompositionRunner,
        )

        print(
            "[entrypoint] context_decomposition block present — running A1-A4 control",
            flush=True,
        )
        context_runner = ContextDecompositionRunner(
            config_path=resolved_path,
            runs_root=runs_root,
        )
        context_metrics = context_runner.run()
        _emit_context_decomposition_metric_lines(context_metrics)
        top_context = output_dir / "context_decomposition_metrics.json"
        with open(top_context, "w", encoding="utf-8") as f:
            json.dump(
                context_metrics,
                f,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )
        print(
            f"[entrypoint] context_decomposition_metrics.json -> {top_context}",
            flush=True,
        )
        return 0

    if cfg.get("pool_perturbation"):
        from presentation_dependence.eval.pool_perturbation import PoolPerturbationRunner

        print(
            "[entrypoint] pool_perturbation block present — running fixed-order pool control",
            flush=True,
        )
        pool_runner = PoolPerturbationRunner(
            config_path=resolved_path,
            runs_root=runs_root,
        )
        pool_metrics = pool_runner.run()
        _emit_pool_metric_lines(pool_metrics)
        top_pool = output_dir / "pool_metrics.json"
        with open(top_pool, "w", encoding="utf-8") as f:
            json.dump(pool_metrics, f, indent=2, sort_keys=True, ensure_ascii=False)
        print(f"[entrypoint] pool_metrics.json -> {top_pool}", flush=True)
        return 0

    exp = ExperimentManager(config_path=resolved_path, runs_root=runs_root)
    exp.run()
    run_dir = exp.run_dir
    print(f"[entrypoint] run_dir    = {run_dir}", flush=True)

    eval_mgr = EvalManager(run_dir=run_dir)
    metrics = eval_mgr.run()

    measures = list((cfg.get("eval") or {}).get("measures") or [])
    _emit_metric_lines(metrics, measures)

    # Duplicate metrics.json at the tarball root so a collector can read it
    # without unpacking the whole archive. The
    # canonical copy is the one ExperimentManager wrote under
    # ``runs/<exp_id>/<ts>/``; a collector picks that subtree.
    top_metrics = output_dir / "metrics.json"
    with open(top_metrics, "w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    print(f"[entrypoint] metrics.json -> {top_metrics}", flush=True)

    headline_key = "mean_ndcg_cut_10"
    if headline_key in metrics:
        print(f"[entrypoint] headline {headline_key} = {metrics[headline_key]:.4f}", flush=True)

    # ---- Optional co-located PSI pass ------------------------------------
    # Opt-in PSI pass. If ``robustness:`` is in the YAML, run
    # PsiExperimentRunner against the same run_dir so everything lands in
    # one tarball. The reranker is reused (skips a second HF model load).
    #
    # EvalManager and PsiExperimentRunner share one container job, model load, run
    # directory, and output tarball while remaining separate drivers.
    if cfg.get("robustness"):
        from presentation_dependence.eval.psi_manager import PsiExperimentRunner

        print("[entrypoint] robustness block present — running PSI pass", flush=True)
        psi_runner = PsiExperimentRunner(
            config_path=resolved_path,
            run_dir=run_dir,
            reranker=exp.reranker,
        )
        psi_metrics = psi_runner.run()
        _emit_psi_K_metric_lines(psi_metrics)
        _emit_tau_psi_B_metric_lines(psi_metrics)
        _emit_robustness_metric_lines(psi_metrics)

        k_proto = psi_metrics.get("K")
        b_proto = psi_metrics.get("tau_psi_context_batch_size")
        if isinstance(k_proto, (int, float)) or b_proto is not None:
            print(
                f"[entrypoint] PSI protocol K={k_proto} B={b_proto} (τ-PSI@B headline width)",
                flush=True,
            )

        # Duplicate psi_metrics.json at the tarball root too, symmetric
        # with the metrics.json convention above. Same fetch-idiom,
        # different filename.
        top_psi = output_dir / "psi_metrics.json"
        with open(top_psi, "w", encoding="utf-8") as f:
            json.dump(psi_metrics, f, indent=2, sort_keys=True, ensure_ascii=False)
        print(f"[entrypoint] psi_metrics.json -> {top_psi}", flush=True)

        psi_headline = psi_metrics.get("aggregate", {}).get("zeng_psi_corpus")
        if isinstance(psi_headline, (int, float)):
            print(f"[entrypoint] headline zeng_psi_corpus = {psi_headline:.4f}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
