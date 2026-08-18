"""Generate and verify immutable reproduction source identities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from presentation_dependence.eval.dataset_catalog import load_dataset_population
from presentation_dependence.utils.config import load_yaml_mapping

from .common import sha256_file
from .data_setup import NECTAR_HF_REVISION
from .errors import ReproductionError


LOCK_PATH = Path("configs/reproduction/source-lock.yaml")
PIPELINES = (
    "configs/reproduction/shared.yaml",
    "configs/reproduction/passage-reranking.yaml",
    "configs/reproduction/multi-document-qa.yaml",
    "configs/reproduction/response-ranking.yaml",
)
POPULATIONS = (
    "reranking-primary-18",
    "reranking-frozen-18",
    "multi-document-qa-3",
    "response-ranking-5",
)
DATA_KEYS = ("run_path", "qrels_path", "topics_tsv", "qids_to_run_path")
DATASET_PINS = {"berkeley-nest/Nectar": NECTAR_HF_REVISION}
FIXTURE_GLOBS = (
    "configs/reproduction/evidence/cohorts/*.txt",
    "configs/reproduction/evidence/metric-discordance/*.json",
    "configs/reproduction/evidence/response-width/*.json",
    "configs/reproduction/evidence/round-robin/*.json",
    "configs/reproduction/fixtures/taupsi-qids/*.txt",
)


def _mappings(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        yield value
        for nested in value.values():
            yield from _mappings(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _mappings(nested)


def declared_model_pins(project_root: Path) -> dict[str, str]:
    """Return unique model-name to immutable revision declarations."""
    pins: dict[str, str] = {}
    for relative in PIPELINES:
        payload = load_yaml_mapping(project_root / relative)
        for row in _mappings(payload):
            model_name = row.get("model_name")
            if not model_name:
                continue
            revision = row.get("revision")
            if not isinstance(revision, str) or len(revision) != 40:
                raise ReproductionError(f"Model {model_name} has no immutable 40-character revision in {relative}")
            prior = pins.get(str(model_name))
            if prior and prior != revision:
                raise ReproductionError(f"Model {model_name} has conflicting revisions: {prior}, {revision}")
            pins[str(model_name)] = revision
    return dict(sorted(pins.items()))


def declared_data_files(project_root: Path) -> list[str]:
    """Return canonical local dataset files used by all task populations."""
    paths = set()
    for population in POPULATIONS:
        rows = load_dataset_population(population, configs_root=project_root / "configs")
        for row in rows:
            # Internal collections cannot be materialized outside the licence
            # that covers them, so locking their hashes would make `verify`
            # permanently unsatisfiable. They are not needed for any published
            # public-collection result.
            if str(row.get("access") or "") == "internal":
                continue
            paths.update(str(row[key]) for key in DATA_KEYS if row.get(key))
    # Frozen reporting cohorts and tau-PSI qid lists determine published
    # reductions. They are not inference populations, so pin them by path.
    for pattern in FIXTURE_GLOBS:
        paths.update(str(path.relative_to(project_root)) for path in project_root.glob(pattern))
    return sorted(paths)


def build_source_lock(project_root: Path) -> dict[str, Any]:
    """Build the deterministic source lock from current canonical inputs."""
    files = {}
    missing = []
    for relative in declared_data_files(project_root):
        path = project_root / relative
        if not path.is_file():
            missing.append(relative)
            continue
        files[relative] = {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    if missing:
        raise ReproductionError(f"Cannot lock missing canonical data files: {missing}")
    return {
        "schema_version": 1,
        "datasets": dict(sorted(DATASET_PINS.items())),
        "models": declared_model_pins(project_root),
        "files": files,
    }


def write_source_lock(project_root: Path) -> Path:
    """Write the tracked deterministic source lock."""
    destination = project_root / LOCK_PATH
    destination.write_text(
        yaml.safe_dump(build_source_lock(project_root), sort_keys=True),
        encoding="utf-8",
    )
    return destination


def verify_source_lock(project_root: Path, *, check_files: bool = True) -> dict[str, Any]:
    """Verify declarations and local data against the tracked source lock."""
    path = project_root / LOCK_PATH
    if not path.is_file():
        raise ReproductionError(f"Missing reproduction source lock: {path}")
    expected = load_yaml_mapping(path)
    actual = {
        "datasets": dict(sorted(DATASET_PINS.items())),
        "models": declared_model_pins(project_root),
    }
    mismatches = []
    for section in ("datasets", "models"):
        if expected.get(section) != actual.get(section):
            mismatches.append(section)
    expected_files = expected.get("files")
    declared_files = declared_data_files(project_root)
    if not isinstance(expected_files, Mapping) or set(expected_files) != set(declared_files):
        mismatches.append("file-declarations")
    if check_files:
        actual_files = build_source_lock(project_root)["files"]
        if expected_files != actual_files:
            mismatches.append("files")
    return {
        "status": "ok" if not mismatches else "mismatch",
        "lock": str(LOCK_PATH),
        "datasets": len(actual["datasets"]),
        "models": len(actual["models"]),
        "files": len(expected_files or {}),
        "files_checked": check_files,
        "mismatches": mismatches,
    }
