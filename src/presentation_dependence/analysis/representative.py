"""YAML-driven representative appendix analysis execution."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from presentation_dependence.reproduction.common import sha256_file
from presentation_dependence.reproduction.errors import ReproductionError
from presentation_dependence.utils.config import load_yaml_mapping

from .metrics import compute_analysis


REGISTRY_PATH = Path("configs/reproduction/representative-analyses.yaml")
OUTPUT_ROOT = Path("build/reproduction/representative")


def _canonical_hash(value: Mapping[str, Any]) -> str:
    payload = {key: item for key, item in value.items() if key not in {"content_hash"}}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _registry(project_root: Path) -> dict[str, Any]:
    registry = load_yaml_mapping(project_root / REGISTRY_PATH)
    if registry.get("schema_version") != 1:
        raise ReproductionError("Representative registry must use schema_version 1")
    families = registry.get("families")
    if not isinstance(families, Mapping) or not families:
        raise ReproductionError("Representative registry has no families")
    return registry


def _families(project_root: Path, family_id: str | None = None) -> list[tuple[str, dict[str, Any]]]:
    families = _registry(project_root)["families"]
    if family_id is not None:
        if family_id not in families:
            raise ReproductionError(f"Unknown representative family: {family_id}")
        return [(family_id, dict(families[family_id]))]
    return [(str(key), dict(value)) for key, value in families.items()]


def _read_fixture(project_root: Path, spec: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = project_root / str(spec["fixture"])
    if not path.is_file():
        raise ReproductionError(f"Representative fixture is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ReproductionError(f"Representative fixture must be a mapping: {path}")
    return path, payload


def _resolve_path(payload: Mapping[str, Any], path: str) -> Any:
    value: Any = payload
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise ReproductionError(f"Representative result has no path: {path}")
        value = value[part]
    return value


def _evaluate_check(payload: Mapping[str, Any], check: Mapping[str, Any]) -> dict[str, Any]:
    observed = _resolve_path(payload, str(check["path"]))
    mode = str(check.get("mode", "absolute"))
    result = {
        "id": str(check["id"]),
        "path": str(check["path"]),
        "mode": mode,
        "observed": observed,
    }
    if mode == "absolute":
        expected = float(check["expected"])
        atol = float(check.get("atol", 0.0))
        passed = math.isclose(float(observed), expected, abs_tol=atol, rel_tol=0.0)
        result.update({"expected": expected, "atol": atol, "passed": passed})
        return result
    if mode == "range":
        lower, upper = [float(value) for value in check["expected"]]
        passed = lower <= float(observed) <= upper
        result.update({"expected": [lower, upper], "passed": passed})
        return result
    if mode == "ci-excludes-zero":
        interval = [float(value) for value in observed]
        passed = len(interval) == 2 and (interval[0] > 0.0 or interval[1] < 0.0)
        result.update({"passed": passed})
        return result
    raise ReproductionError(f"Unknown representative check mode: {mode}")


def plan_representative(project_root: Path, *, family_id: str | None = None) -> dict[str, Any]:
    """Report representative fixtures, methods, and paper anchors."""
    rows = []
    for current_id, spec in _families(project_root, family_id):
        fixture = project_root / str(spec["fixture"])
        rows.append(
            {
                "family": current_id,
                "operation": spec["operation"],
                "paper": spec["paper"],
                "fixture": str(spec["fixture"]),
                "fixture_exists": fixture.is_file(),
                "checks": len(spec.get("checks") or []),
            }
        )
    return {
        "schema_version": 1,
        "families": rows,
        "summary": {
            "declared": len(rows),
            "ready": sum(row["fixture_exists"] for row in rows),
        },
    }


def _run_family(project_root: Path, family_id: str, spec: Mapping[str, Any]) -> dict[str, Any]:
    fixture_path, fixture = _read_fixture(project_root, spec)
    if fixture.get("family") != family_id:
        raise ReproductionError(f"Fixture family mismatch for {family_id}: {fixture.get('family')}")
    payload = compute_analysis(str(spec["operation"]), fixture["payload"])
    checks = [_evaluate_check(payload, check) for check in spec.get("checks") or []]
    result = {
        "schema_version": 1,
        "family_id": family_id,
        "paper": dict(spec["paper"]),
        "operation": str(spec["operation"]),
        "evidence_level": fixture.get("evidence_level"),
        "source": fixture.get("source"),
        "fixture": {
            "path": str(fixture_path.relative_to(project_root)),
            "sha256": sha256_file(fixture_path),
        },
        "status": "complete" if all(row["passed"] for row in checks) else "failed",
        "coverage": {
            "expected_checks": len(checks),
            "passed_checks": sum(row["passed"] for row in checks),
        },
        "payload": payload,
        "checks": checks,
    }
    result["content_hash"] = _canonical_hash(result)
    output_root = project_root / OUTPUT_ROOT / family_id
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_root / "provenance.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "family_id": family_id,
                "fixture": result["fixture"],
                "content_hash": result["content_hash"],
                "checks": result["coverage"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return result


def run_representative(project_root: Path, *, family_id: str | None = None) -> dict[str, Any]:
    """Run one or all representative analysis families offline."""
    results = {
        current_id: _run_family(project_root, current_id, spec)
        for current_id, spec in _families(project_root, family_id)
    }
    manifest = {
        "schema_version": 1,
        "families": {
            current_id: {
                "status": result["status"],
                "content_hash": result["content_hash"],
                "result": str(OUTPUT_ROOT / current_id / "result.json"),
            }
            for current_id, result in results.items()
        },
        "summary": {
            "families": len(results),
            "complete": sum(result["status"] == "complete" for result in results.values()),
        },
    }
    output_root = project_root / OUTPUT_ROOT
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def validate_representative(project_root: Path, *, family_id: str | None = None) -> dict[str, Any]:
    """Run representative analyses and write the aggregate validation receipt."""
    manifest = run_representative(project_root, family_id=family_id)
    status = "complete" if manifest["summary"]["complete"] == manifest["summary"]["families"] else "incomplete"
    result = {
        "schema_version": 1,
        "status": status,
        "summary": manifest["summary"],
        "families": manifest["families"],
    }
    output = project_root / OUTPUT_ROOT / "validate-all.json"
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result
