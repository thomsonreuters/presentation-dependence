#!/usr/bin/env python
r"""Expand a sweep YAML and run every cell locally.

The matrix is expanded by ``presentation_dependence.utils.sweeps``, then each cell is
dispatched to the runner its config implies: silver generation, student SFT, a
permutation sweep, or a single reranking pass.

Usage::

    python scripts/run_sweep.py configs/sweeps/<id>.yaml
    python scripts/run_sweep.py configs/sweeps/<id>.yaml --dry-run
    python scripts/run_sweep.py configs/sweeps/<id>.yaml --only example-passage-b20-psi
    python scripts/run_sweep.py configs/sweeps/<id>.yaml --only 2 --only 3
    python scripts/run_sweep.py configs/sweeps/<id>.yaml --jobs 2

Concurrency comes from the sweep's ``execution.max_parallel``, or from
``--jobs`` when given. It counts processes on this machine, each loading its
own copy of the model, so a value above 1 only makes sense when the cells do
not contend for the same GPU. Tracked sweeps declare 1.

A manifest lands at ``sweeps/<sweep_id>/manifest.json`` recording every cell,
its command, and its exit status, so a partial sweep can be resumed with
``--only``.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from presentation_dependence.utils.config import ConfigOverrideError, apply_overrides  # noqa: E402
from presentation_dependence.utils.log_redaction import (  # noqa: E402
    manifest_override_lists_for_disk,
    redact_launch_argv_for_log,
)
from presentation_dependence.utils.sweeps import (  # noqa: E402
    claims_every_gpu,
    expand_jobs,
    load_sweep,
    local_command,
    max_parallel,
    resolve_config,
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sweep_yaml", type=Path, help="Path to configs/sweeps/<id>.yaml.")
    p.add_argument("--dry-run", action="store_true", help="Print the expanded matrix and the commands; run nothing.")
    p.add_argument(
        "--only", action="append", default=[], help="Run only these exp_id(s) or cell index/indices. Repeatable."
    )
    p.add_argument(
        "--skip", action="append", default=[], help="Skip these exp_id(s) or cell index/indices. Repeatable."
    )
    p.add_argument("--jobs", type=int, default=None, help="Concurrent cells. Default: the sweep's max_parallel.")
    p.add_argument("--keep-going", action="store_true", help="Continue after a failing cell instead of stopping.")
    p.add_argument("--manifest-root", type=Path, default=None, help="Where to write the manifest. Default: sweeps/.")
    return p.parse_args()


def _claims_every_gpu(job: dict, root: Path) -> bool:
    """Whether this cell's config sizes itself from the visible GPUs.

    The cell's overrides are applied first: a sweep can turn a config into a
    ``ddp`` student, and the warning has to see the cell as it will run.
    """
    try:
        _, cfg = resolve_config(job["exp_id"], root / "configs")
    except FileNotFoundError:
        return False
    try:
        cfg = apply_overrides(cfg, job["overrides"])
    except ConfigOverrideError:
        # A bad override is the child's error to report, with its own message.
        pass
    return claims_every_gpu(cfg)


def _matches(job: dict, tokens: set[str]) -> bool:
    """Match a cell by exp_id or by its index in the sweep.

    A sweep that varies the dataset reuses one config across every cell, so
    exp_id alone cannot name a single cell. The index printed beside each cell
    can.
    """
    return job["exp_id"] in tokens or str(job["index"]) in tokens


def _select(jobs: list[dict], only: list[str], skip: list[str]) -> list[dict]:
    if only:
        jobs = [j for j in jobs if _matches(j, set(only))]
    if skip:
        jobs = [j for j in jobs if not _matches(j, set(skip))]
    return jobs


def _run_one(job: dict) -> int:
    shown = " ".join(redact_launch_argv_for_log(job["command"]))
    print(f"[sweep] [{job['index']}] {job['exp_id']}: {shown}", flush=True)
    return subprocess.run(job["command"], check=False).returncode


def main() -> int:  # noqa: C901
    """Expand the sweep and dispatch every selected cell."""
    args = parse_args()
    root = _project_root()
    sweep = load_sweep(args.sweep_yaml)
    sweep_id = str(sweep.get("id") or args.sweep_yaml.stem)

    jobs = _select(expand_jobs(sweep), args.only, args.skip)
    if not jobs:
        print("[sweep][FATAL] no cells selected", file=sys.stderr)
        return 2

    for job in jobs:
        job["command"] = local_command(job, python=sys.executable, project_root=root)

    n_parallel = args.jobs if args.jobs is not None else max_parallel(sweep)
    print(f"[sweep] {sweep_id}: {len(jobs)} cell(s), concurrency={n_parallel}", flush=True)

    if n_parallel > 1:
        greedy = [job["exp_id"] for job in jobs if _claims_every_gpu(job, root)]
        if greedy:
            # A sweep whose cells are named by path lists 24 absolute paths here
            # otherwise, which buries the warning it is trying to deliver.
            names = sorted({Path(exp_id).stem for exp_id in greedy})
            shown_names = ", ".join(names[:4]) + (f", and {len(names) - 4} more" if len(names) > 4 else "")
            print(
                f"[sweep][WARN] concurrency={n_parallel} but {len(greedy)} cell(s) size themselves "
                f"from the visible GPUs ({shown_names}); they will double-book "
                "the same devices. Run these sequentially.",
                file=sys.stderr,
                flush=True,
            )

    if args.dry_run:
        for job in jobs:
            shown = " ".join(redact_launch_argv_for_log(job["command"]))
            print(f"[sweep] [{job['index']}] {job['exp_id']}\n    {shown}")
        return 0

    results: dict[int, int] = {}
    if n_parallel <= 1:
        for job in jobs:
            results[job["index"]] = _run_one(job)
            if results[job["index"]] and not args.keep_going:
                break
    else:
        with futures.ThreadPoolExecutor(max_workers=n_parallel) as pool:
            pending = {pool.submit(_run_one, job): job for job in jobs}
            for future in futures.as_completed(pending):
                results[pending[future]["index"]] = future.result()

    manifest_root = args.manifest_root or (root / "sweeps")
    manifest_dir = manifest_root / sweep_id
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "sweep_id": sweep_id,
        "description": sweep.get("description"),
        "written_at": datetime.now(timezone.utc).isoformat(),
        "jobs": [
            {
                "index": job["index"],
                "exp_id": job["exp_id"],
                "overrides": manifest_override_lists_for_disk(job["overrides"]),
                "command": redact_launch_argv_for_log(job["command"]),
                "returncode": results.get(job["index"]),
            }
            for job in jobs
        ],
    }
    (manifest_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[sweep] manifest -> {manifest_dir / 'manifest.json'}", flush=True)

    failed = [i for i, code in results.items() if code]
    skipped = [job["index"] for job in jobs if job["index"] not in results]
    if failed or skipped:
        print(f"[sweep] {len(failed)} failed, {len(skipped)} not run", file=sys.stderr)
        return 1
    print(f"[sweep] {len(jobs)} cell(s) complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
