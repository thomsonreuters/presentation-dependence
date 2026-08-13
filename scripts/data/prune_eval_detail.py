#!/usr/bin/env python
r"""Reclaim disk by deleting regenerable / redundant detail from finished run
directories while preserving every artefact that downstream training, eval, or
analysis actually needs.

Three independent, individually-guarded prune categories run together
(all enabled by default; toggle with ``--no-*`` flags):

1. **Eval detail** (PSI / quality eval trials).
2. **Training ``data_views/``** (SFT / self-distill trials).
3. **Silver resume state** (``*-bsc-*`` teacher trials).

Background
----------

*Eval detail.* A PSI trial fans out per query into 13 permutations
(10 random shuffles + 3 middle-injection positions), and each
permutation writes a fat ``detailed_results.json`` under
``psi/per_query_results/<qid>/permutation_<k>/``. On big surfaces the
per-perm tree alone reaches ~4 GB per trial. The aggregates we keep
(``metrics.json``, ``psi/psi_metrics.json``, ``psi/psi_per_query.json``,
``psi/sc_*.json``) are < 1 MB combined.

*Training data_views.* Every SFT / self-distill run writes inspection
sidecars to ``student/data_views/`` (regression / pairwise / listwise /
permutation JSONL views). They are written by ``student.py`` purely for
inspection (``write_sidecars``) and are read by **nothing**: training
consumes in-memory chunks, eval never touches them. They regenerate
deterministically from the silver labels, at ~2-4 GB per run.

*Silver resume state.* A ``*-bsc-*`` teacher run keeps ``silver/per_qid/``
(per-query resume shards) and ``workers/`` (per-worker shards). Both are
merged into the canonical ``silver/silver_labels.jsonl``, the only file
``student_data.py`` reads. ``manifest.json`` records ``n_records``, written
last, only on full success. Once published, the shards are redundant.

What this prunes (per category, if present and guards pass)
-----------------------------------------------------------

Eval detail:
  * ``psi/per_query_results/``: per-perm fan-out
  * ``per_query_results/``: top-level eval detail
  * ``all_queries_eval_results.jsonl``: flat per-qid metric dump

Training data_views:
  * ``student/data_views/``: regenerable inspection sidecars

Silver resume state:
  * ``silver/per_qid/``: per-query resume shards
  * ``workers/``: per-worker shards

Always keeps: ``metrics.json``, ``resolved_config.yaml``,
``psi/psi_metrics.json``, ``psi/sc_metrics.json``,
``psi/psi_per_query.json``, ``psi/sc_per_query.json``,
``student/checkpoint-final/``, ``student/progress.jsonl``,
``student/training_summary.json``, ``student/resolved_student_config.json``,
``silver/silver_labels.jsonl``, ``silver/manifest.json``, all ``*.log``,
and anything else not on the explicit deletion list.

Safety guards
-------------

Eval detail is **refused** when ``psi/per_query_results/`` exists but
``psi/psi_metrics.json`` is absent (would block
package-level partial-PSI recovery). Pass ``--allow-unaggregated``
to override.

Eval detail is **skipped** when neither ``metrics.json`` nor
``psi/psi_metrics.json`` exists (no aggregate to fall back on).

``data_views/`` (regenerable inspection sidecars, read by nothing) is
dropped once the run is no longer being written — i.e. it has a final
checkpoint, a training summary, trained weights under ``student/``
(``checkpoints/`` layout). It is **refused** only
for a local, in-flight training job with none of those — exactly where
racing a live writer would be unsafe.

Silver shards are **refused** when present but the teacher run is not
verified complete: ``silver/silver_labels.jsonl`` + ``silver/manifest.json``
must both exist, and the line count of ``silver_labels.jsonl`` must equal
``manifest.json``'s ``n_records``. This guarantees the canonical merged
silver is whole before its source shards are deleted.

Idempotency
-----------

After a successful prune, a ``.pruned.json`` marker is written at the
trial root recording what was removed and when. Re-running on an
already-pruned trial reports ``noop`` (nothing left to prune).

Discovery modes
---------------

Combine freely:

  * Positional trial dirs:  ``runs/<ID>/<ts>/`` ...
  * ``--exp-id <ID>``: all timestamps under that exp id.
  * ``--all``: every trial under ``--runs-root``.
  * ``--sweep <sweep-id>``: every job listed in
                              ``sweeps/<id>/manifest.json``.

By default the script runs in **dry-run** mode and prints what would
be deleted. Pass ``--apply`` to actually delete. All three prune
categories are on by default; disable any with ``--no-prune-eval-detail``,
``--no-prune-data-views``, ``--no-prune-silver-resume``.

Examples:
--------
::

    # Dry-run: how much would we save by pruning every trial?
    uv run python scripts/data/prune_eval_detail.py --all

    # Apply on one trial:
    uv run python scripts/data/prune_eval_detail.py --apply \\
        runs/<exp-id>/<timestamp>

    # Apply on every trial:
    uv run python scripts/data/prune_eval_detail.py --apply --all

    # Apply on every timestamp of one exp id, including unaggregated PSI:
    uv run python scripts/data/prune_eval_detail.py --apply \\
        --exp-id <exp-id> \\
        --allow-unaggregated
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from presentation_dependence.utils.run_paths import default_runs_root  # noqa: E402

PRUNE_MARKER_NAME = ".pruned.json"

# Eval-detail targets (relative to trial root).
PSI_PER_QUERY = Path("psi/per_query_results")
TOP_PER_QUERY = Path("per_query_results")
ALL_QUERIES_JSONL = Path("all_queries_eval_results.jsonl")

# Training-trial targets / markers.
DATA_VIEWS = Path("student/data_views")
CHECKPOINT_FINAL = Path("student/checkpoint-final")
TRAINING_SUMMARY = Path("student/training_summary.json")

# Teacher silver-run targets / markers.
SILVER_PER_QID = Path("silver/per_qid")
WORKERS = Path("workers")
SILVER_LABELS = Path("silver/silver_labels.jsonl")
SILVER_MANIFEST = Path("silver/manifest.json")


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _dir_size_bytes(path: Path) -> int:
    """Return cumulative byte size of a directory tree, ignoring symlinks."""
    total = 0
    for root, _dirs, files in os.walk(path, followlinks=False):
        for name in files:
            fp = Path(root) / name
            try:
                total += fp.lstat().st_size
            except OSError:
                continue
    return total


def _is_training_trial(trial_dir: Path) -> bool:
    """Heuristic: training trials carry a ``student/`` subdir with checkpoints,
    not an eval ``per_query_results/`` tree.
    """
    return (trial_dir / "student").is_dir()


def _looks_like_trial_root(trial_dir: Path) -> bool:
    """A directory is a prunable trial root if it carries any of the recognised
    run markers: an eval config, a fetch marker, a training ``student/`` tree,
    or a teacher ``silver/`` tree.
    """
    return (
        (trial_dir / "resolved_config.yaml").is_file()
        or (trial_dir / "student").is_dir()
        or (trial_dir / "silver").is_dir()
    )


def _has_student_weights(trial_dir: Path) -> bool:
    """True if any LoRA adapter weights exist under ``student/`` — covers runs
    that keep the trained model under ``student/checkpoints/checkpoint-step-*/``
    rather than ``student/checkpoint-final/``.
    """
    student = trial_dir / "student"
    if not student.is_dir():
        return False
    for _ in student.rglob("*.safetensors"):
        return True
    return False


def _data_views_safe_to_drop(trial_dir: Path) -> bool:
    """``student/data_views/`` is regenerable inspection output, never read by
    training or eval. It is safe to delete once the run is no longer being
    written. We treat a run as no-longer-written when ANY of:

    * it has a final checkpoint or end-of-run training summary, or
    * it has trained weights under ``student/`` (``checkpoints/`` layout).

    The only case left refused is an in-flight training run with no weights and
    no summary yet, which is exactly where racing a live writer would be
    unsafe.
    """
    return (
        (trial_dir / CHECKPOINT_FINAL).is_dir()
        or (trial_dir / TRAINING_SUMMARY).is_file()
        or _has_student_weights(trial_dir)
    )


def _count_lines(path: Path) -> int:
    """Count newlines in a file with bounded memory. ``silver_labels.jsonl``
    is written one record per line with a trailing newline, so this equals
    the published record count.
    """
    n = 0
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            n += chunk.count(b"\n")
    return n


def _teacher_silver_verified(trial_dir: Path) -> tuple[bool, str | None]:
    """Return ``(ok, reason_if_not)`` for whether the canonical merged silver
    is provably whole, so its ``per_qid/`` + ``workers/`` source shards are
    safe to delete.

    Cheap + authoritative: compares ``silver_labels.jsonl``'s line count to
    ``manifest.json``'s ``n_records`` (the teacher writes the manifest last,
    only on full success — see ``self_distill/teacher.py``).
    """
    labels = trial_dir / SILVER_LABELS
    manifest = trial_dir / SILVER_MANIFEST
    if not labels.is_file() or not manifest.is_file():
        return False, "teacher run not complete (silver/silver_labels.jsonl + silver/manifest.json required)"
    try:
        with open(manifest, "r", encoding="utf-8") as f:
            n_records = json.load(f).get("n_records")
    except Exception as e:
        return False, f"silver/manifest.json unreadable ({e})"
    if not isinstance(n_records, int):
        return False, "silver/manifest.json has no integer n_records"
    actual = _count_lines(labels)
    if actual != n_records:
        return False, f"silver_labels.jsonl lines ({actual}) != manifest n_records ({n_records}); merge not verified"
    return True, None


def _human(n_bytes: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    size = float(n_bytes)
    idx = 0
    while size >= 1024 and idx < len(units) - 1:
        size /= 1024
        idx += 1
    return f"{size:.1f} {units[idx]}"


def _collect_eval_targets(
    trial_dir: Path,
    *,
    keep_top_per_query: bool,
    keep_all_queries_jsonl: bool,
    allow_unaggregated: bool,
) -> tuple[list[Path], str | None, str | None]:
    """Eval-detail category. Returns ``(targets, refusal, skip)``.

    Applies only to eval/PSI trials (those with ``resolved_config.yaml`` and
    no ``student/`` tree). Training and teacher trials yield no eval targets
    and no reason, so the other categories handle them.
    """
    if not (trial_dir / "resolved_config.yaml").is_file():
        return [], None, None
    if _is_training_trial(trial_dir):
        return [], None, None

    has_metrics = (trial_dir / "metrics.json").is_file()
    has_psi_aggregate = (trial_dir / "psi" / "psi_metrics.json").is_file()
    has_psi_perq = (trial_dir / PSI_PER_QUERY).is_dir()

    if not has_metrics and not has_psi_aggregate:
        return [], None, "no metrics.json and no psi/psi_metrics.json (no aggregates to fall back on)"

    if has_psi_perq and not has_psi_aggregate and not allow_unaggregated:
        return (
            [],
            (
                "psi/per_query_results/ present but psi/psi_metrics.json missing — "
                "pruning would block partial-PSI aggregate recovery"
            ),
            None,
        )

    targets: list[Path] = []
    if has_psi_perq:
        targets.append(trial_dir / PSI_PER_QUERY)
    if not keep_top_per_query and (trial_dir / TOP_PER_QUERY).is_dir():
        targets.append(trial_dir / TOP_PER_QUERY)
    if not keep_all_queries_jsonl and (trial_dir / ALL_QUERIES_JSONL).is_file():
        targets.append(trial_dir / ALL_QUERIES_JSONL)
    return targets, None, None


def _collect_data_views_targets(trial_dir: Path, *, enabled: bool) -> tuple[list[Path], str | None, str | None]:
    """Training ``data_views/`` category. Returns ``(targets, refusal, skip)``."""
    if not enabled:
        return [], None, None
    dv = trial_dir / DATA_VIEWS
    if not dv.is_dir():
        return [], None, None
    if not _data_views_safe_to_drop(trial_dir):
        return (
            [],
            (
                "student/data_views/ present but run looks like a local in-flight training job "
                "(no checkpoint-final/, no training_summary.json, no weights, no fetch marker)"
            ),
            None,
        )
    return [dv], None, None


def _collect_silver_targets(trial_dir: Path, *, enabled: bool) -> tuple[list[Path], str | None, str | None]:
    """Teacher silver-resume category. Returns ``(targets, refusal, skip)``."""
    if not enabled:
        return [], None, None
    per_qid = trial_dir / SILVER_PER_QID
    workers = trial_dir / WORKERS
    if not per_qid.is_dir() and not workers.is_dir():
        return [], None, None
    ok, reason = _teacher_silver_verified(trial_dir)
    if not ok:
        return [], f"silver resume files present but {reason}", None
    targets: list[Path] = []
    if per_qid.is_dir():
        targets.append(per_qid)
    if workers.is_dir():
        targets.append(workers)
    return targets, None, None


def _measure_targets(targets: Iterable[Path]) -> int:
    total = 0
    for tgt in targets:
        if tgt.is_dir():
            total += _dir_size_bytes(tgt)
        elif tgt.is_file():
            try:
                total += tgt.lstat().st_size
            except OSError:
                pass
    return total


def _delete_targets(targets: Iterable[Path]) -> None:
    for tgt in targets:
        if tgt.is_dir():
            shutil.rmtree(tgt)
        elif tgt.is_file():
            tgt.unlink()


def _write_marker(trial_dir: Path, *, removed_rel: list[str], bytes_reclaimed: int) -> None:
    marker = {
        "pruned_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "removed": sorted(removed_rel),
        "bytes_reclaimed": bytes_reclaimed,
        "tool": "scripts/data/prune_eval_detail.py",
        "schema_version": 1,
    }
    existing = trial_dir / PRUNE_MARKER_NAME
    if existing.is_file():
        try:
            with open(existing, "r", encoding="utf-8") as f:
                prev = json.load(f)
            history = list(prev.get("history") or [])
            history.append({k: v for k, v in prev.items() if k != "history"})
            marker["history"] = history
        except Exception:
            pass
    with open(existing, "w", encoding="utf-8") as f:
        json.dump(marker, f, indent=2, sort_keys=True)


def prune_trial(  # noqa: C901
    trial_dir: Path,
    *,
    apply: bool = False,
    allow_unaggregated: bool = False,
    keep_top_per_query: bool = False,
    keep_all_queries_jsonl: bool = False,
    prune_eval_detail: bool = True,
    prune_data_views: bool = True,
    prune_silver_resume: bool = True,
    measure: bool = True,
) -> dict:
    """Prune (or report on) a single trial dir across all enabled categories.

    Returns a dict describing the outcome. The caller is responsible
    for printing — ``main()`` does that — so this function stays importable
    from other scripts without fighting stdout.
    """
    trial_dir = trial_dir.resolve()
    if not trial_dir.is_dir():
        return _result(trial_dir, "skipped", f"not a directory: {trial_dir}", apply)
    if not _looks_like_trial_root(trial_dir):
        return _result(
            trial_dir,
            "skipped",
            "no resolved_config.yaml / student/ / silver/ (not a recognizable run/trial dir)",
            apply,
        )

    targets: list[Path] = []
    refusals: list[str] = []
    skips: list[str] = []

    ev_t, ev_ref, ev_skip = (
        _collect_eval_targets(
            trial_dir,
            keep_top_per_query=keep_top_per_query,
            keep_all_queries_jsonl=keep_all_queries_jsonl,
            allow_unaggregated=allow_unaggregated,
        )
        if prune_eval_detail
        else ([], None, None)
    )
    dv_t, dv_ref, dv_skip = _collect_data_views_targets(trial_dir, enabled=prune_data_views)
    sv_t, sv_ref, sv_skip = _collect_silver_targets(trial_dir, enabled=prune_silver_resume)

    for t in (ev_t, dv_t, sv_t):
        targets.extend(t)
    for r in (ev_ref, dv_ref, sv_ref):
        if r:
            refusals.append(r)
    for s in (ev_skip, dv_skip, sv_skip):
        if s:
            skips.append(s)

    # Always measure on `--apply` so the .pruned.json marker records reclaimed
    # bytes faithfully. For dry-run, allow the caller to skip the walk; this
    # cuts wall-clock by an order of magnitude on large trees.
    bytes_reclaimed = _measure_targets(targets) if (targets and (apply or measure)) else 0
    rel = [str(t.relative_to(trial_dir)) for t in targets]

    if not targets:
        if refusals:
            return _result(trial_dir, "refused", "; ".join(refusals), apply)
        if skips:
            return _result(trial_dir, "skipped", "; ".join(skips), apply)
        return _result(trial_dir, "noop", "nothing to prune", apply)

    # Some categories may yield targets while another refuses; surface the
    # deferred refusal as a note but still reclaim what is safe.
    note = ("; ".join(refusals)) if refusals else None

    if not apply:
        return _result(trial_dir, "dry-run", note, apply, bytes_reclaimed=bytes_reclaimed, removed=rel)

    _delete_targets(targets)
    _write_marker(trial_dir, removed_rel=rel, bytes_reclaimed=bytes_reclaimed)
    return _result(trial_dir, "applied", note, apply, bytes_reclaimed=bytes_reclaimed, removed=rel)


def _result(
    trial_dir: Path,
    status: str,
    reason: str | None,
    apply: bool,
    *,
    bytes_reclaimed: int = 0,
    removed: list[str] | None = None,
) -> dict:
    return {
        "trial_dir": str(trial_dir),
        "status": status,
        "reason": reason,
        "bytes_reclaimed": bytes_reclaimed,
        "removed": removed or [],
        "dry_run": not apply,
    }


def _iter_trial_dirs_under_runs_root(runs_root: Path) -> Iterable[Path]:
    """Yield ``runs/<exp_id>/<trial>/`` for every recognisable trial dir —
    eval/PSI (``resolved_config.yaml``), training (``student/``), or teacher
    (``silver/``). See ``_looks_like_trial_root``.
    """
    if not runs_root.is_dir():
        return
    for exp_dir in sorted(runs_root.iterdir()):
        if not exp_dir.is_dir():
            continue
        for trial_dir in sorted(exp_dir.iterdir()):
            if not trial_dir.is_dir():
                continue
            if _looks_like_trial_root(trial_dir):
                yield trial_dir


def _iter_trial_dirs_for_exp_id(runs_root: Path, exp_id: str) -> Iterable[Path]:
    exp_dir = runs_root / exp_id
    if not exp_dir.is_dir():
        return
    for trial_dir in sorted(exp_dir.iterdir()):
        if trial_dir.is_dir() and _looks_like_trial_root(trial_dir):
            yield trial_dir


def _iter_trial_dirs_for_sweep(sweep_id: str, runs_root: Path) -> Iterable[Path]:
    from _paths import sweeps_root  # type: ignore[import-not-found]

    manifest_path = sweeps_root() / sweep_id / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"sweep manifest not found: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    for job in manifest.get("jobs", []):
        exp_id = job.get("exp_id")
        if not exp_id:
            continue
        yield from _iter_trial_dirs_for_exp_id(runs_root, exp_id)


def collect_trial_dirs(args: argparse.Namespace, runs_root: Path) -> list[Path]:  # noqa: C901
    """Resolve the union of all CLI-selected discovery modes into a flat list."""
    selected: list[Path] = []
    seen: set[Path] = set()

    def _add(p: Path) -> None:
        rp = p.resolve()
        if rp in seen:
            return
        seen.add(rp)
        selected.append(rp)

    for raw in args.trial_dirs or []:
        _add(Path(raw))
    if args.exp_id:
        for td in _iter_trial_dirs_for_exp_id(runs_root, args.exp_id):
            _add(td)
    if args.sweep_id:
        for td in _iter_trial_dirs_for_sweep(args.sweep_id, runs_root):
            _add(td)
    if args.all:
        for td in _iter_trial_dirs_under_runs_root(runs_root):
            _add(td)
    return selected


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI args."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "trial_dirs",
        nargs="*",
        help=("Specific trial dirs to prune (runs/<ID>/<ts>/). Combine freely with --exp-id / --sweep / --all."),
    )
    p.add_argument(
        "--exp-id",
        dest="exp_id",
        default=None,
        help="Prune every trial under runs/<exp-id>/.",
    )
    p.add_argument(
        "--sweep",
        dest="sweep_id",
        default=None,
        help="Read sweeps/<sweep-id>/manifest.json and prune every trial for every job in it.",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Prune every trial dir under --runs-root. Use with care.",
    )
    p.add_argument(
        "--runs-root",
        type=Path,
        default=default_runs_root(),
        help="Local runs root (default: SLM_RUNS_ROOT, else ./runs).",
    )
    p.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete. Without this flag the script runs in dry-run mode.",
    )
    p.add_argument(
        "--allow-unaggregated",
        action="store_true",
        help="Permit pruning psi/per_query_results/ even when psi/psi_metrics.json is missing. "
        "This blocks future partial-PSI aggregate recovery for that trial.",
    )
    p.add_argument(
        "--keep-top-per-query",
        action="store_true",
        help="Keep the top-level per_query_results/ tree (only prune psi/per_query_results/ and the jsonl).",
    )
    p.add_argument(
        "--keep-all-queries-jsonl",
        action="store_true",
        help="Keep all_queries_eval_results.jsonl.",
    )
    p.add_argument(
        "--prune-eval-detail",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prune eval/PSI per-query detail (on by default; --no-prune-eval-detail to disable).",
    )
    p.add_argument(
        "--prune-data-views",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prune student/data_views/ on completed training runs (on by default; --no-prune-data-views to disable).",
    )
    p.add_argument(
        "--prune-silver-resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Prune silver/per_qid/ + workers/ on verified-complete teacher runs "
            "(on by default; --no-prune-silver-resume to disable)."
        ),
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Only print per-trial errors and the final summary.",
    )
    p.add_argument(
        "--no-measure",
        action="store_true",
        help=(
            "Skip byte-counting target trees in dry-run mode. Reported "
            "'would reclaim' bytes will be 0; trial-by-trial listings still "
            "show which paths would be removed. Use for fast previews on "
            "large trees. Ignored when --apply is set (apply always measures)."
        ),
    )
    return p.parse_args(argv)


def _format_row(report: dict) -> str:
    tag = report["status"].upper()
    saved = _human(report["bytes_reclaimed"])
    rel = ",".join(report.get("removed") or []) or "-"
    reason = f"  [{report['reason']}]" if report.get("reason") else ""
    return f"  {tag:<8s} {saved:>9s}  {report['trial_dir']}{reason}  ({rel})"


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = parse_args(argv)
    project = _project_root()
    runs_root = args.runs_root if args.runs_root.is_absolute() else project / args.runs_root

    if not (args.trial_dirs or args.exp_id or args.sweep_id or args.all):
        print(
            "[prune] no targets selected. Pass trial dirs, --exp-id, --sweep, or --all.",
            file=sys.stderr,
        )
        return 2

    trial_dirs = collect_trial_dirs(args, runs_root)
    if not trial_dirs:
        print("[prune] no trial dirs matched.", file=sys.stderr)
        return 0

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"[prune] {mode}: {len(trial_dirs)} trial dir(s) considered.", flush=True)

    totals = {
        "applied": 0,
        "dry-run": 0,
        "skipped": 0,
        "refused": 0,
        "noop": 0,
    }
    bytes_total = 0
    refused: list[dict] = []
    for td in trial_dirs:
        try:
            report = prune_trial(
                td,
                apply=args.apply,
                allow_unaggregated=args.allow_unaggregated,
                keep_top_per_query=args.keep_top_per_query,
                keep_all_queries_jsonl=args.keep_all_queries_jsonl,
                prune_eval_detail=args.prune_eval_detail,
                prune_data_views=args.prune_data_views,
                prune_silver_resume=args.prune_silver_resume,
                measure=not args.no_measure,
            )
        except Exception as e:
            print(f"  ERROR    {td}: {e}", file=sys.stderr)
            totals["skipped"] += 1
            continue

        totals[report["status"]] = totals.get(report["status"], 0) + 1
        bytes_total += report["bytes_reclaimed"]
        if report["status"] == "refused":
            refused.append(report)
            print(_format_row(report), file=sys.stderr)
        elif not args.quiet:
            print(_format_row(report))

    print("", flush=True)
    verb = "would reclaim" if not args.apply else "reclaimed"
    print(
        f"[prune] {mode} summary: {verb} {_human(bytes_total)} total. "
        f"applied={totals.get('applied', 0)} dry-run={totals.get('dry-run', 0)} "
        f"noop={totals.get('noop', 0)} skipped={totals.get('skipped', 0)} "
        f"refused={totals.get('refused', 0)}",
        flush=True,
    )

    if refused:
        print(
            "[prune] note: refused trials had unaggregated PSI (finalize the PSI aggregate or pass "
            "--allow-unaggregated), an incomplete training run (data_views kept), or an unverified "
            "teacher silver merge (silver shards kept). See per-trial reasons above.",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
