"""Writing sweep manifests, and expanding them again.

A sweep YAML is an ordered matrix of jobs that share setup and vary along a few
axes (dataset, model, hyperparameter). `configs/sweeps/_schema.md` defines the
format. Both halves of its lifetime live here: the reproduction pipelines
serialize one per training and evaluation stage, and `scripts/run_sweep.py`
expands it again to run the cells.

Dispatch mirrors the container entrypoint, which selects a code path from the
config body rather than from the filename: a `teacher:` block means silver
generation, `student:` means student SFT, `robustness:` means a permutation
sweep, and anything else is a single reranking pass.

Kept in `utils` rather than `reproduction` so that expanding and previewing a
sweep stays a yaml-only import; `reproduction/__init__` pulls the whole eval
stack, which would make a dry run require pytrec_eval and torch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from presentation_dependence.utils.config import load_experiment_config, load_yaml_mapping

EXECUTION_BLOCK = "execution"

# `max_parallel` configures the sweep runner itself: it counts concurrent child
# processes on this machine, so it is meaningless inside a cell. Every other
# execution key describes the job (`environment`, `distributed`,
# `fixture_channel`, `extra_input_channels`) and is read by the child from its
# own config, so it has to be forwarded rather than dropped.
SWEEP_SCOPED_EXECUTION_KEYS = frozenset({"max_parallel"})
_SWEEP_SCOPED_PREFIXES = tuple(f"{EXECUTION_BLOCK}.{key}=" for key in sorted(SWEEP_SCOPED_EXECUTION_KEYS))

# Config subdirectories searched for a bare exp_id, in order.
CONFIG_SUBDIRS = ("experiments", "self-distill", "silver")

RUNNERS = {
    "bundle": "scripts/run_bundle.py",
    "silver": "scripts/run_silver_generation.py",
    "student": "scripts/run_self_distill_sft.py",
    "psi": "scripts/run_psi.py",
    "rerank": "scripts/run_experiment.py",
}

# run_silver_generation.py takes -c; the rest take -e.
_CONFIG_FLAG = {"silver": "-c"}


def flatten_overrides(prefix: str, value: Any, out: list[str]) -> None:
    """Turn ``{"batch_size": 64}`` under prefix ``reranker`` into ``["reranker.batch_size=64"]``.

    Nested dicts recurse. Scalars are YAML-safe-dumped so strings, bools and
    floats round-trip through ``apply_overrides`` unchanged on the other side.
    """
    if isinstance(value, dict):
        for key, sub in value.items():
            sub_prefix = f"{prefix}.{key}" if prefix else key
            flatten_overrides(sub_prefix, sub, out)
        return
    dumped = yaml.safe_dump(value, default_flow_style=True).strip()
    if dumped.endswith("\n..."):  # yaml dump edge case
        dumped = dumped[:-4]
    out.append(f"{prefix}={dumped}")


def expand_jobs(sweep: dict) -> list[dict]:
    """Return a list of ``{index, exp_id, overrides}`` entries, in sweep order.

    Precedence, highest first: per-job ``overrides``, then ``common_overrides``,
    then the execution block, then the values in the experiment config itself.
    """
    execution = sweep.get(EXECUTION_BLOCK) or {}
    common_overrides = sweep.get("common_overrides") or {}

    base_flat: list[str] = []
    flatten_overrides(
        EXECUTION_BLOCK,
        {k: v for k, v in execution.items() if k not in SWEEP_SCOPED_EXECUTION_KEYS},
        base_flat,
    )
    flatten_overrides("", common_overrides, base_flat)

    jobs = sweep.get("jobs") or []
    if not jobs:
        raise ValueError("Sweep YAML has empty / missing 'jobs:' list.")

    expanded: list[dict] = []
    for i, spec in enumerate(jobs):
        if "exp_id" not in spec:
            raise ValueError(f"jobs[{i}] is missing required 'exp_id'.")
        per_job: list[str] = []
        flatten_overrides("", spec.get("overrides") or {}, per_job)
        expanded.append({"index": i, "exp_id": spec["exp_id"], "overrides": [*base_flat, *per_job]})
    return expanded


def load_sweep(path: str | Path) -> dict:
    """Load and minimally validate a sweep YAML."""
    sweep = load_yaml_mapping(Path(path))
    if not sweep.get("jobs"):
        raise ValueError(f"{path} has no 'jobs:' list")
    return sweep


def max_parallel(sweep: dict, default: int = 1) -> int:
    """Return the concurrent-job cap declared by the sweep."""
    return int((sweep.get(EXECUTION_BLOCK) or {}).get("max_parallel", default) or default)


def resolve_config(exp_id: str, configs_root: Path | None = None) -> tuple[Path, dict]:
    """Resolve a bare exp_id against every config subdirectory that can hold one."""
    last: Exception | None = None
    for subdir in CONFIG_SUBDIRS:
        try:
            return load_experiment_config(exp_id, configs_root, subdir=subdir)
        except FileNotFoundError as exc:
            last = exc
    searched = ", ".join(f"configs/{d}/" for d in CONFIG_SUBDIRS)
    raise FileNotFoundError(f"No config for exp_id={exp_id!r} in {searched}") from last


def runner_kind(cfg: dict) -> str:
    """Classify a resolved config the way the container entrypoint does.

    Order matters and mirrors the entrypoint: a bundle carries the member
    configs rather than a reranker of its own, so it is checked first.
    """
    if cfg.get("bundle"):
        return "bundle"
    if cfg.get("teacher"):
        return "silver"
    if cfg.get("student"):
        return "student"
    if cfg.get("robustness"):
        return "psi"
    return "rerank"


def claims_every_gpu(cfg: Mapping[str, Any]) -> bool:
    """True when one cell of this config expects the whole machine.

    A ``ddp`` student re-execs under torchrun with one worker per visible GPU,
    and a teacher with ``local_data_parallel_workers`` puts one replica on each.
    Either way the cell sizes itself from the visible devices, so running two
    such cells at once double-books every one of them.
    """
    execution = cfg.get(EXECUTION_BLOCK) or {}
    mode = execution.get("distributed") or execution.get("distribution")
    if isinstance(mode, dict):
        mode = "ddp" if (mode.get("ddp") or mode.get("type") == "ddp") else None
    if mode is not None and str(mode).lower() == "ddp" and cfg.get("student"):
        return True
    return int((cfg.get("teacher") or {}).get("local_data_parallel_workers", 0) or 0) > 1


def local_command(
    job: dict,
    *,
    python: str,
    project_root: Path,
    configs_root: Path | None = None,
) -> list[str]:
    """Build the local invocation for one expanded sweep cell.

    Only the sweep runner's own keys are withheld; see
    :data:`SWEEP_SCOPED_EXECUTION_KEYS`.
    """
    # Anchored on the project root rather than the caller's cwd, so a sweep
    # resolves the same way wherever it is invoked from.
    config_path, cfg = resolve_config(job["exp_id"], configs_root or project_root / "configs")
    kind = runner_kind(cfg)
    command = [python, str(project_root / RUNNERS[kind]), _CONFIG_FLAG.get(kind, "-e"), str(config_path)]
    for override in job["overrides"]:
        if override.startswith(_SWEEP_SCOPED_PREFIXES):
            continue
        command += ["--override", override]
    return command


# --------------------------------------------------------------------- writing

# Only the serialization is shared: callers build the mapping themselves so each
# keeps its own key order, which is load-bearing because these files are dumped
# with `sort_keys=False` and compared byte for byte.


def write_generated_yaml(path: Path, payload: Mapping[str, Any], *, allow_unicode: bool = False) -> Path:
    """Write one generated YAML document and return its path.

    Key order is the caller's and long values are not wrapped, so generated
    trees stay comparable byte for byte across runs.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            dict(payload),
            sort_keys=False,
            width=1000,
            allow_unicode=allow_unicode,
        ),
        encoding="utf-8",
    )
    return path


def write_sweep(path: Path, sweep: Mapping[str, Any], *, allow_unicode: bool = False) -> Path:
    """Write one sweep manifest and return its path."""
    return write_generated_yaml(path, sweep, allow_unicode=allow_unicode)


def write_config(path: Path, config: Mapping[str, Any]) -> Path:
    """Write one generated experiment or bundle config."""
    return write_generated_yaml(path, config)
