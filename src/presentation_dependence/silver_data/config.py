"""Resolution, classification, and validation for silver-generation configs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from presentation_dependence.utils.config import find_configs_root, load_yaml_mapping


class SilverConfigKind(str, Enum):
    """Supported silver-generation config variants."""

    OPEN_WEIGHT = "open_weight"
    HOSTED = "hosted"


@dataclass(frozen=True)
class SilverConfig:
    """A validated silver-generation config and its source path."""

    path: Path
    data: dict[str, Any]
    kind: SilverConfigKind
    experiment_id: str


_SHORT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_OPEN_PROTOCOL = "k_shot_bsc"
_HOSTED_PROTOCOL = "closed_model_generated_bsc"


def resolve_silver_config_path(
    config_id_or_path: str | Path,
    *,
    configs_root: Path | None = None,
) -> Path:
    """Resolve a short ID from ``configs/silver`` or accept an explicit YAML path."""
    raw = str(config_id_or_path)
    candidate = Path(raw)
    if candidate.suffix.lower() in {".yaml", ".yml"} or candidate.is_file():
        path = candidate
    else:
        if not _SHORT_ID_RE.fullmatch(raw) or raw in {".", ".."}:
            raise ValueError(
                "Silver config short IDs may contain only letters, digits, '.', "
                "'_', and '-' and may not contain path separators. Pass an explicit "
                "YAML path instead."
            )
        root = configs_root or find_configs_root()
        path = root / "silver" / f"{raw}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Silver config not found: {path}")
    return path


def classify_silver_config(config: Mapping[str, Any]) -> SilverConfigKind:
    """Classify a silver config from its required ``teacher.protocol`` discriminator."""
    teacher = config.get("teacher")
    if not isinstance(teacher, Mapping):
        raise ValueError("Silver config requires a top-level 'teacher' mapping")
    protocol = teacher.get("protocol")
    if not isinstance(protocol, str) or not protocol.strip():
        raise ValueError(
            f"Silver config requires teacher.protocol; expected {_OPEN_PROTOCOL!r} or {_HOSTED_PROTOCOL!r}"
        )
    normalized = protocol.strip().lower()
    if normalized == _OPEN_PROTOCOL:
        return SilverConfigKind.OPEN_WEIGHT
    if normalized == _HOSTED_PROTOCOL:
        return SilverConfigKind.HOSTED
    raise ValueError(f"Unsupported teacher.protocol={protocol!r}; expected {_OPEN_PROTOCOL!r} or {_HOSTED_PROTOCOL!r}")


def validate_silver_config(
    config: Mapping[str, Any],
    *,
    path: str | Path = Path("<memory>"),
) -> SilverConfig:
    """Validate the discriminated union and return a normalized loaded config."""
    source = Path(path)
    data = dict(config)
    kind = classify_silver_config(data)
    if kind is SilverConfigKind.OPEN_WEIGHT:
        experiment_id = _validate_open_weight(data)
    else:
        experiment_id = _validate_hosted(data)
    return SilverConfig(
        path=source,
        data=data,
        kind=kind,
        experiment_id=experiment_id,
    )


def load_silver_config(
    config_id_or_path: str | Path,
    *,
    configs_root: Path | None = None,
) -> SilverConfig:
    """Resolve, safely load, classify, and validate one silver config."""
    path = resolve_silver_config_path(config_id_or_path, configs_root=configs_root)
    config = load_yaml_mapping(path)
    return validate_silver_config(config, path=path)


def _required_nonempty_string(
    mapping: Mapping[str, Any],
    key: str,
    *,
    label: str,
) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}.{key} must be a non-empty string")
    return value.strip()


def _required_mapping(
    mapping: Mapping[str, Any],
    key: str,
    *,
    label: str = "config",
) -> Mapping[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{label}.{key} must be a mapping")
    return value


def _validate_open_weight(config: Mapping[str, Any]) -> str:
    experiment_id = _required_nonempty_string(config, "id", label="config")
    if config.get("experiment_id") not in (None, experiment_id):
        raise ValueError("Open-weight configs use id; experiment_id must be absent or equal to id")
    if "student" in config:
        raise ValueError("Open-weight silver configs may not contain a student block")

    reranker = _required_mapping(config, "reranker")
    if not (reranker.get("class") or reranker.get("model_name")):
        raise ValueError("config.reranker requires class or model_name")
    _required_mapping(config, "data")
    teacher = _required_mapping(config, "teacher")

    k_perms = teacher.get("k_perms", 15)
    if isinstance(k_perms, bool):
        raise ValueError("config.teacher.k_perms must be a positive integer")
    try:
        k_perms_int = int(k_perms)
    except (TypeError, ValueError) as exc:
        raise ValueError("config.teacher.k_perms must be a positive integer") from exc
    if k_perms_int <= 0:
        raise ValueError("config.teacher.k_perms must be a positive integer")
    seeds = teacher.get("seeds")
    if seeds is not None:
        if not isinstance(seeds, list):
            raise ValueError("config.teacher.seeds must be a list when provided")
        if len(seeds) != k_perms_int:
            raise ValueError(
                f"config.teacher.seeds length must equal config.teacher.k_perms ({len(seeds)} != {k_perms_int})"
            )
    return experiment_id


def _validate_hosted(config: Mapping[str, Any]) -> str:
    experiment_id = _hosted_experiment_id(config)
    for forbidden in ("student", "reranker", "data"):
        if forbidden in config:
            raise ValueError(f"Hosted silver configs may not contain top-level {forbidden!r}")

    input_config = _required_mapping(config, "input")
    nested_data = input_config.get("data")
    if nested_data is not None and not isinstance(nested_data, Mapping):
        raise ValueError("config.input.data must be a mapping when provided")
    effective_input = nested_data if isinstance(nested_data, Mapping) else input_config
    if not effective_input.get("run_path"):
        raise ValueError("Hosted silver configs require input.run_path or input.data.run_path")

    teacher = _required_mapping(config, "teacher")
    _required_nonempty_string(teacher, "model_id", label="config.teacher")
    prompts = config.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("config.prompts must be a non-empty list")
    if not all(isinstance(prompt, str) and prompt.strip() for prompt in prompts):
        raise ValueError("config.prompts entries must be non-empty strings")
    _validate_optional_mappings(config, "cost", "output")
    return experiment_id


def _hosted_experiment_id(config: Mapping[str, Any]) -> str:
    id_value = config.get("experiment_id") or config.get("id")
    if not isinstance(id_value, str) or not id_value.strip():
        raise ValueError("Hosted silver configs require non-empty experiment_id or id")
    if config.get("experiment_id") and config.get("id") and config["experiment_id"] != config["id"]:
        raise ValueError("Hosted config experiment_id and id must match when both are set")
    return id_value.strip()


def _validate_optional_mappings(config: Mapping[str, Any], *keys: str) -> None:
    for key in keys:
        value = config.get(key)
        if value is not None and not isinstance(value, Mapping):
            raise ValueError(f"config.{key} must be a mapping when provided")
