"""Experiment-config plumbing shared by local and container entry points.

All entry points (``scripts/run_experiment.py`` locally and
``scripts/train/entrypoint.py`` in a container) go through the
same four primitives:

1. :func:`load_experiment_config`: resolve an exp-id or explicit YAML
   path to a ``(Path, dict)`` pair.
2. :func:`apply_overrides`: patch the loaded dict with dotted-key CLI
   overrides (e.g. ``reranker.batch_size=64``). Fails hard on unknown
   top-level keys to catch typos early.
3. :func:`apply_execution_environment`: apply trusted runtime defaults without
   overriding values supplied by the process environment.
4. :func:`resolve_channel`: derive the data channel name from the ``data:``
   block. A container run mounts the channel under that name.

Override syntax
---------------
One ``--override KEY=VALUE`` per option. KEY is a dotted path whose
first segment must be one of the whitelisted top-level keys
(:data:`TOP_LEVEL_KEYS`). VALUE is parsed via YAML's safe loader so
``64`` is ``int``, ``1.5`` is ``float``, ``true`` is ``bool``, and
bare strings stay strings.

Examples::

    --override reranker.batch_size=64
    --override data.k_input=50
    --override reranker.batch_size=64
    --override logging.level=DEBUG
"""

from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


TOP_LEVEL_KEYS: frozenset[str] = frozenset(
    {
        "id",
        "reranker",
        "data",
        "eval",
        "robustness",
        "pool_perturbation",
        "context_decomposition",
        "matched_variance_control",
        "self_consistency",
        "debug",
        "qids_to_run",
        "qids_to_run_path",
        "training",
        "logging",
        "execution",
        "bundle",
        "student",
        "teacher",
    }
)


class ConfigOverrideError(ValueError):
    """Raised when an ``--override`` spec is malformed or targets an unknown top-level key."""


_ENVIRONMENT_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def apply_execution_environment(cfg: Mapping[str, Any]) -> list[str]:
    """Apply trusted ``execution.environment`` values to this process.

    Existing process variables win over tracked YAML. This preserves explicit
    operator, ``.env``, container, and scheduler settings while letting a config
    provide portable defaults. Only variable names are returned so callers can
    report what was applied without leaking values.

    Call this only from runtime entrypoints, after config/CLI overrides and
    before importing model backends, inspecting accelerators, or spawning child
    processes. Config loaders and planning/materialization paths remain
    side-effect free.
    """
    execution = cfg.get("execution")
    if execution is None:
        return []
    if not isinstance(execution, Mapping):
        raise ValueError("execution must be a mapping")
    environment = execution.get("environment")
    if environment is None:
        return []
    if not isinstance(environment, Mapping):
        raise ValueError("execution.environment must be a mapping")

    applied: list[str] = []
    for raw_key, value in environment.items():
        key = str(raw_key)
        if not _ENVIRONMENT_KEY.fullmatch(key):
            raise ValueError(
                f"Invalid execution.environment key {key!r}; expected a POSIX-style environment variable name"
            )
        if value is None or isinstance(value, (dict, list, tuple, set)):
            raise ValueError(f"execution.environment.{key} must be a scalar, not {type(value).__name__}")
        if key in os.environ:
            continue
        if isinstance(value, bool):
            os.environ[key] = "1" if value else "0"
        else:
            os.environ[key] = str(value)
        applied.append(key)
    return applied


def load_yaml_mapping(path: str | Path) -> dict[str, Any]:
    """Load a YAML file that must contain a top-level mapping."""
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"YAML file not found: {resolved}")
    value = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{resolved} did not parse to a dict (got {type(value).__name__})")
    return value


def find_configs_root(start: Path | None = None) -> Path:
    """Walk up from ``start`` (default cwd) to find a ``configs/experiments/`` dir.

    Do not resolve from ``__file__`` because the same code runs from the
    repository root locally and may run from the legacy ``/opt/ml/code/``
    container layout. In both cases ``configs/`` sits at the project root
    relative to wherever the entry script was launched from.
    """
    cur = (start or Path.cwd()).resolve()
    for candidate in [cur, *cur.parents]:
        if (candidate / "configs" / "experiments").is_dir():
            return candidate / "configs"
    raise FileNotFoundError(
        f"Could not find a 'configs/experiments/' directory from {cur}. "
        "Run from the project root or pass --config-path explicitly."
    )


def load_experiment_config(
    exp_id_or_path: str | Path,
    configs_root: Path | None = None,
    *,
    subdir: str = "experiments",
) -> tuple[Path, dict]:
    """Resolve either a short exp-id or an explicit path, return ``(Path, config dict)``.

    Resolution order:

    1. If ``exp_id_or_path`` ends with ``.yaml`` or points at an existing file,
       treat it as a literal path.
    2. Otherwise treat it as an exp-id and look for
       ``<configs_root>/<subdir>/<exp_id>.yaml``.

    Parameters
    ----------
    subdir : str, default ``"experiments"``
        Sub-directory under ``configs/`` to search for short-id configs. Use
        ``"silver"`` for teacher generation and ``"self-distill"`` for
        student training.
    """
    candidate = Path(exp_id_or_path)
    if candidate.suffix == ".yaml" or candidate.is_file():
        path = candidate
    else:
        root = configs_root or find_configs_root()
        path = root / subdir / f"{exp_id_or_path}.yaml"

    if not path.is_file():
        raise FileNotFoundError(f"Experiment config not found: {path}")

    cfg = load_yaml_mapping(path)
    return path, cfg


def _parse_override(spec: str) -> tuple[list[str], Any]:
    """Parse ``"reranker.batch_size=64"`` into ``(["reranker", "batch_size"], 64)``.

    The value side goes through ``yaml.safe_load`` so YAML scalar types
    (``true``/``false``, ``null``, ints, floats, quoted strings) all
    behave as expected. Unquoted strings (e.g. ``foo.bar=baz``) are kept
    as ``"baz"``.
    """
    if "=" not in spec:
        raise ConfigOverrideError(f"Override {spec!r} must be of the form KEY=VALUE.")
    key, raw_value = spec.split("=", 1)
    key = key.strip()
    raw_value = raw_value.strip()
    if not key:
        raise ConfigOverrideError(f"Override {spec!r} has an empty key.")
    parts = key.split(".")
    if any(not p for p in parts):
        raise ConfigOverrideError(f"Override key {key!r} has an empty segment.")
    try:
        value = yaml.safe_load(raw_value)
    except yaml.YAMLError as e:
        raise ConfigOverrideError(f"Could not parse value in {spec!r}: {e}") from e
    return parts, value


def apply_overrides(
    cfg: dict,
    overrides: Iterable[str],
    *,
    strict_top_level: bool = True,
) -> dict:
    """Return a deep copy of ``cfg`` with ``overrides`` applied.

    ``overrides`` is an iterable of ``"dotted.key=value"`` strings. When
    ``strict_top_level`` is True (the default) the first segment of each
    key must be one of :data:`TOP_LEVEL_KEYS` — this catches typos like
    ``--override rranker.batch_size=64``.

    Intermediate path segments are allowed to introduce new keys (e.g.
    adding ``execution.max_parallel`` when the YAML has no ``execution:``
    block yet).
    """
    out = copy.deepcopy(cfg)
    for spec in overrides:
        parts, value = _parse_override(spec)
        if strict_top_level and parts[0] not in TOP_LEVEL_KEYS:
            raise ConfigOverrideError(
                f"Unknown top-level key {parts[0]!r} in override {spec!r}. Expected one of {sorted(TOP_LEVEL_KEYS)}."
            )
        cur: Any = out
        for seg in parts[:-1]:
            if not isinstance(cur, dict):
                raise ConfigOverrideError(
                    f"Cannot descend into non-dict at {'.'.join(parts[: parts.index(seg)])} while applying {spec!r}."
                )
            cur = cur.setdefault(seg, {})
        if not isinstance(cur, dict):
            raise ConfigOverrideError(f"Cannot set leaf on non-dict at {'.'.join(parts[:-1])} while applying {spec!r}.")
        cur[parts[-1]] = value
    return out


def _deep_merge_mapping(target: dict[str, Any], patch: Mapping[str, Any]) -> None:
    """Recursively merge ``patch`` into ``target`` without sharing mutable values."""
    for key, value in patch.items():
        current = target.get(key)
        if isinstance(current, dict) and isinstance(value, Mapping):
            _deep_merge_mapping(current, value)
        else:
            target[key] = copy.deepcopy(value)


def apply_bundle_member_overrides(
    member_cfg: Mapping[str, Any],
    bundle_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the two bundle override layers to one resolved member config.

    ``member_overrides`` patches the complete member configuration, then
    ``reranker_overrides`` patches only its ``reranker`` block. Keeping this
    transformation here makes local and container bundle execution identical.
    """
    out = copy.deepcopy(dict(member_cfg))
    member_overrides = bundle_cfg.get("member_overrides") or {}
    reranker_overrides = bundle_cfg.get("reranker_overrides") or {}
    if not isinstance(member_overrides, Mapping):
        raise ValueError("bundle.member_overrides must be a mapping")
    if not isinstance(reranker_overrides, Mapping):
        raise ValueError("bundle.reranker_overrides must be a mapping")

    _deep_merge_mapping(out, member_overrides)
    if reranker_overrides:
        reranker = out.setdefault("reranker", {})
        if not isinstance(reranker, dict):
            raise ValueError("bundle.reranker_overrides requires the member reranker to be a mapping")
        _deep_merge_mapping(reranker, reranker_overrides)
    return out


def resolve_channel(cfg: dict) -> str:
    """Return the data channel name for this experiment.

    Precedence (first match wins):

    1. ``execution.fixture_channel`` — explicit override in the YAML.
    2. Parent-directory name of ``data.run_path``. This matches the
       on-disk layout (e.g. ``data/dl19-passage/run.*`` → ``dl19-passage``,
       ``data/beir-v1.0.0-trec-covid-test/run.*`` →
       ``beir-v1.0.0-trec-covid-test``).
    3. ``data.topics`` — fallback for configs that don't have a run_path
       yet.

    Channel names are restricted to ``[A-Za-z0-9_.-]+`` so they are safe as
    directory and mount names; all
    three sources produce legal names for existing experiments.
    """
    execution_block = cfg.get("execution") or {}
    explicit = execution_block.get("fixture_channel")
    if explicit:
        return str(explicit)

    data = cfg.get("data") or {}
    run_path = data.get("run_path")
    if run_path:
        parent = Path(run_path).parent.name
        if parent:
            return parent

    topics = data.get("topics")
    if topics:
        return str(topics)

    raise ValueError(
        "Could not resolve channel: YAML has no execution.fixture_channel, no data.run_path, and no data.topics."
    )


def write_resolved_config(cfg: dict, dest: Path) -> Path:
    """Write the merged config for entry points that hand a path to ``ExperimentManager``."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return dest
