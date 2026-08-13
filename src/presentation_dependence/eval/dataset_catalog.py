"""Load reproduction dataset populations for config generators and studies."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from presentation_dependence.utils.config import find_configs_root


def load_dataset_population(
    population_id: str,
    *,
    configs_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Load one reproduction population by ID."""
    root = configs_root or find_configs_root()
    path = root / "reproduction" / "populations" / f"{population_id}.yaml"
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("id") != population_id:
        raise ValueError(f"Invalid dataset population: {path}")
    datasets = value.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError(f"Dataset population has no datasets: {path}")
    base_population = value.get("base_population")
    if base_population:
        if str(base_population) == population_id:
            raise ValueError(f"Dataset population cannot extend itself: {path}")
        base = load_dataset_population(str(base_population), configs_root=root)
        overrides = {str(dataset["id"]): dict(dataset) for dataset in datasets}
        base_ids = {str(dataset["id"]) for dataset in base}
        unknown = set(overrides) - base_ids
        if unknown:
            raise ValueError(f"Population {population_id} overrides unknown datasets: {sorted(unknown)}")
        return [{**dataset, **overrides.get(str(dataset["id"]), {})} for dataset in base]
    return [dict(dataset) for dataset in datasets]


def merge_dataset_population(
    existing: Iterable[Mapping[str, Any]],
    population_id: str,
    *,
    configs_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Overlay population fields onto a broader existing registry."""
    canonical = {
        str(dataset["id"]): dataset for dataset in load_dataset_population(population_id, configs_root=configs_root)
    }
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in existing:
        dataset = dict(raw)
        dataset_id = str(dataset["id"])
        if dataset_id in canonical:
            dataset.update(canonical[dataset_id])
            seen.add(dataset_id)
        merged.append(dataset)
    merged.extend(dataset for dataset_id, dataset in canonical.items() if dataset_id not in seen)
    return merged
