"""Plan and run the programs behind registered appendix artifacts."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

from presentation_dependence.utils.config import load_yaml_mapping

from .common import tracked_files
from .errors import ReproductionError


REGISTRY_PATH = Path("configs/reproduction/appendix.yaml")
PROGRAMS_PATH = Path("configs/reproduction/appendix-programs.yaml")
VALID_STATUSES = {
    "canonical",
    "legacy",
    "static",
    "partial",
    "manual",
    "unsupported",
}


def _python_script_command(project_root: Path, script: str, *arguments: str) -> list[str]:
    path = Path(script)
    if (
        path.suffix == ".py"
        and len(path.parts) > 2
        and path.parts[:2]
        in {
            ("scripts", "analyze"),
            ("scripts", "data"),
            ("scripts", "gen"),
        }
    ):
        module = ".".join(path.with_suffix("").parts)
        return [sys.executable, "-m", module, *arguments]
    return [sys.executable, str(project_root / path), *arguments]


def load_appendix_registry(project_root: Path) -> dict[str, Any]:
    """Load and minimally validate the appendix artifact registry."""
    registry = load_yaml_mapping(project_root / REGISTRY_PATH)
    artifacts = registry.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ReproductionError("Appendix registry has no artifacts")
    for artifact_id, artifact in artifacts.items():
        if not isinstance(artifact, dict):
            raise ReproductionError(f"Appendix artifact {artifact_id!r} must be a mapping")
        if artifact.get("status") not in VALID_STATUSES:
            raise ReproductionError(f"Appendix artifact {artifact_id!r} has invalid status {artifact.get('status')!r}")
    return registry


def load_appendix_programs(project_root: Path) -> dict[str, Any]:
    """Load the shared from-scratch appendix execution programs."""
    payload = load_yaml_mapping(project_root / PROGRAMS_PATH)
    programs = payload.get("programs")
    if not isinstance(programs, dict) or not programs:
        raise ReproductionError("Appendix program registry has no programs")
    return programs


def _tracked(project_root: Path, relative: str) -> bool:
    normalized = relative.rstrip("/")
    tracked = tracked_files(project_root)
    return normalized in tracked or any(path.startswith(f"{normalized}/") for path in tracked)


def _expand_template_section(section: Mapping[str, Any]) -> list[str]:
    paths = [str(path) for path in section.get("fixed") or []]
    paths.extend(str(path) for path in section.get("teacher_silver") or [])
    for family in section.get("families") or []:
        pattern = str(family["pattern"])
        codes = family.get("code") or []
        paths.extend(pattern.format(code=code) for code in codes)
    return paths


def _template_manifest_audit(project_root: Path, relative: str) -> dict[str, Any]:
    path = project_root / relative
    if not path.is_file():
        return {
            "path": relative,
            "templates": [],
            "missing": [relative],
            "untracked": [],
            "valid": False,
        }
    manifest = load_yaml_mapping(path)
    templates = []
    for section in (manifest.get("primary") or {}).values():
        templates.extend(_expand_template_section(section))
    templates.extend(_expand_template_section(manifest.get("scaling") or manifest.get("appendix_scaling") or {}))
    templates = sorted(set(templates))
    missing = [template for template in templates if not (project_root / template).is_file()]
    untracked = [
        template
        for template in templates
        if (project_root / template).is_file() and not _tracked(project_root, template)
    ]
    return {
        "path": relative,
        "templates": templates,
        "missing": missing,
        "untracked": untracked,
        "valid": not missing and not untracked and _tracked(project_root, relative),
    }


def _program_audit(
    project_root: Path,
    program_id: str | None,
) -> dict[str, Any] | None:
    if program_id is None:
        return None
    programs = load_appendix_programs(project_root)
    try:
        program = programs[program_id]
    except KeyError as exc:
        raise ReproductionError(f"Unknown appendix reproduction program: {program_id}") from exc
    inputs = [str(path) for path in program.get("inputs") or []]
    missing_inputs = [path for path in inputs if not (project_root / path).exists()]
    untracked_inputs = [path for path in inputs if (project_root / path).exists() and not _tracked(project_root, path)]
    study_source = str(program["study"]) if program.get("study") else None
    study = (
        load_yaml_mapping(project_root / study_source)
        if study_source and (project_root / study_source).is_file()
        else None
    )
    study_error = None
    if study_source and study is None:
        study_error = f"missing study declaration: {study_source}"
    study_materializer_specs = list(study.get("materializers") or []) if isinstance(study, dict) else []
    study_materializers = [
        str(materializer["entrypoint"] if isinstance(materializer, dict) else materializer)
        for materializer in study_materializer_specs
    ]
    study_collector_specs = list(study.get("collectors") or []) if isinstance(study, dict) else []
    study_collectors = [
        str(collector["entrypoint"] if isinstance(collector, dict) else collector)
        for collector in study_collector_specs
    ]
    script_keys = (
        "materializers",
        "collectors",
        "reducers",
        "benchmarks",
        "figures",
    )
    scripts = (
        [str(path) for key in script_keys for path in program.get(key) or []] + study_materializers + study_collectors
    )
    missing_scripts = [path for path in scripts if not (project_root / path).is_file()]
    untracked_scripts = [
        path for path in scripts if (project_root / path).is_file() and not _tracked(project_root, path)
    ]
    sweeps = [str(path) for path in program.get("sweeps") or []]
    missing_sweeps = [path for path in sweeps if not (project_root / path).is_file()]
    untracked_sweeps = [path for path in sweeps if (project_root / path).is_file() and not _tracked(project_root, path)]
    config_globs = {
        pattern: len(list(project_root.glob(str(pattern)))) for pattern in program.get("config_globs") or []
    }
    missing_globs = [pattern for pattern, count in config_globs.items() if count == 0]
    untracked_globs = {
        pattern: [
            str(path.relative_to(project_root))
            for path in project_root.glob(str(pattern))
            if not _tracked(project_root, str(path.relative_to(project_root)))
        ]
        for pattern in config_globs
    }
    study_source_missing = bool(study_source and not (project_root / study_source).is_file())
    study_source_untracked = bool(
        study_source and not study_source_missing and not _tracked(project_root, study_source)
    )
    template_manifest = (
        str(study.get("template_manifest")) if isinstance(study, dict) and study.get("template_manifest") else None
    )
    template_audit = _template_manifest_audit(project_root, template_manifest) if template_manifest else None
    return {
        "id": program_id,
        "inputs": inputs,
        "missing_inputs": missing_inputs,
        "untracked_inputs": untracked_inputs,
        "scripts": scripts,
        "missing_scripts": missing_scripts,
        "untracked_scripts": untracked_scripts,
        "sweeps": sweeps,
        "missing_sweeps": missing_sweeps,
        "untracked_sweeps": untracked_sweeps,
        "config_globs": config_globs,
        "missing_config_globs": missing_globs,
        "untracked_config_globs": untracked_globs,
        "study": study.get("id") if isinstance(study, dict) else None,
        "study_spec": study,
        "study_error": study_error,
        "study_source": study_source,
        "study_source_missing": study_source_missing,
        "study_source_untracked": study_source_untracked,
        "template_manifest": template_audit,
        "valid": not (
            missing_inputs
            or untracked_inputs
            or missing_scripts
            or untracked_scripts
            or missing_sweeps
            or untracked_sweeps
            or missing_globs
            or any(untracked_globs.values())
            or study_error
            or study_source_missing
            or study_source_untracked
            or (template_audit and not template_audit["valid"])
        ),
        "spec": program,
    }


def _artifact_spec(project_root: Path, artifact_id: str) -> Mapping[str, Any]:
    artifacts = load_appendix_registry(project_root)["artifacts"]
    try:
        return artifacts[artifact_id]
    except KeyError as exc:
        raise ReproductionError(f"Unknown appendix artifact: {artifact_id}") from exc


def plan_appendix_artifact(project_root: Path, artifact_id: str) -> dict[str, Any]:
    """Resolve the from-scratch dependencies for one appendix artifact."""
    artifact = _artifact_spec(project_root, artifact_id)
    program = _program_audit(project_root, artifact.get("program"))
    analyzer = artifact.get("analyzer")
    analyzer_exists = (project_root / str(analyzer)).is_file() if analyzer else None
    analyzer_untracked = bool(analyzer and analyzer_exists and not _tracked(project_root, str(analyzer)))
    preferred = artifact.get("preferred_input")
    preferred_exists = (project_root / str(preferred)).is_file() if preferred else None
    sources = [str(path) for path in artifact.get("sources") or []]
    missing_sources = [path for path in sources if not (project_root / path).is_file()]
    command = None
    if analyzer and analyzer_exists:
        command = _python_script_command(project_root, str(analyzer))
        if preferred:
            command.extend(["--canonical", str(project_root / preferred)])
    blockers = []
    if artifact["status"] == "static":
        blockers.append("static artifact: content is committed, not regenerated")
    elif analyzer is None:
        blockers.append("artifact declares no analyzer")
    elif not analyzer_exists:
        blockers.append(f"analyzer not found: {analyzer}")
    elif analyzer_untracked:
        blockers.append(f"analyzer is untracked: {analyzer}")
    if preferred and not preferred_exists:
        blockers.append(f"declared reproduction program has not produced analyzer input: {preferred}")
    if missing_sources:
        blockers.append(f"missing sources: {missing_sources}")
    if program is not None and not program["valid"]:
        blockers.append(f"declared reproduction program is invalid: {program['id']}")
    workflow = _program_commands(project_root, program) if program else []
    return {
        "artifact_id": artifact_id,
        "declared_status": artifact["status"],
        "runnable": bool(command) and not blockers,
        "analyzer": analyzer,
        "analyzer_exists": analyzer_exists,
        "preferred_input": preferred,
        "preferred_input_exists": preferred_exists,
        "sources": sources,
        "command": command,
        "program": program,
        "workflow": workflow,
        "blockers": blockers,
    }


def audit_appendix(project_root: Path, artifact_id: str | None = None) -> dict[str, Any]:
    """Report analyzer runnability for all or one registered artifact."""
    artifacts = load_appendix_registry(project_root)["artifacts"]
    selected = [artifact_id] if artifact_id is not None else list(artifacts)
    if artifact_id is not None and artifact_id not in artifacts:
        raise ReproductionError(f"Unknown appendix artifact: {artifact_id}")
    rows = [plan_appendix_artifact(project_root, key) for key in selected]
    return {
        "schema_version": 1,
        "registry": str(REGISTRY_PATH),
        "artifacts": rows,
        "summary": {
            "declared": len(rows),
            "runnable": sum(row["runnable"] for row in rows),
            "blocked": sum(bool(row["blockers"]) for row in rows),
            "analyzer_inputs_present": sum(row["preferred_input_exists"] is True for row in rows),
        },
    }


def _program_commands(  # noqa: C901
    project_root: Path,
    program: Mapping[str, Any],
) -> list[dict[str, Any]]:
    spec = program["spec"]
    commands = []
    materializers = list(spec.get("materializers") or [])
    materializers.extend((program.get("study_spec") or {}).get("materializers") or [])
    seen_commands = set()
    for materializer in materializers:
        if isinstance(materializer, dict):
            script = str(materializer["entrypoint"])
            arguments = [str(value) for value in materializer.get("arguments") or []]
        else:
            script = str(materializer)
            arguments = []
        key = (script, *arguments)
        if key in seen_commands:
            continue
        seen_commands.add(key)
        commands.append(
            {
                "phase": "materialize",
                "command": _python_script_command(project_root, script, *arguments),
            }
        )
    tasks = spec.get("tasks") or ([spec["task"]] if spec.get("task") else [])
    if spec.get("study_commands"):
        for task in tasks:
            for arguments in spec["study_commands"]:
                commands.append(
                    {
                        "phase": str(arguments[1]),
                        "command": [
                            sys.executable,
                            str(project_root / "scripts/study.py"),
                            task,
                            *map(str, arguments),
                        ],
                    }
                )
    else:
        for task in tasks:
            for stage in spec.get("study_stages") or []:
                for command in ("materialize", "validate", "execute", "collect"):
                    arguments = [
                        sys.executable,
                        str(project_root / "scripts/study.py"),
                        task,
                        stage,
                        command,
                    ]
                    if command == "execute":
                        arguments.append("--dry-run")
                    commands.append(
                        {
                            "phase": command,
                            "command": arguments,
                            "operator_inputs_required": (stage == "silver" and command == "collect"),
                        }
                    )
    for sweep in spec.get("sweeps") or []:
        commands.append(
            {
                "phase": "execute",
                "command": [
                    sys.executable,
                    str(project_root / "scripts/run_sweep.py"),
                    str(project_root / sweep),
                    "--dry-run",
                ],
            }
        )
    collectors = list(spec.get("collectors") or [])
    collectors.extend((program.get("study_spec") or {}).get("collectors") or [])
    for collector in collectors:
        if isinstance(collector, dict):
            script = str(collector["entrypoint"])
            arguments = [str(value) for value in collector.get("arguments") or []]
            operator_inputs_required = bool(collector.get("operator_inputs_required"))
        else:
            script = str(collector)
            arguments = []
            operator_inputs_required = False
        key = (script, *arguments)
        if key in seen_commands:
            continue
        seen_commands.add(key)
        commands.append(
            {
                "phase": "collect",
                "command": _python_script_command(project_root, script, *arguments),
                "operator_inputs_required": operator_inputs_required,
            }
        )
    for script in [
        *(spec.get("reducers") or []),
        *(spec.get("benchmarks") or []),
        *(spec.get("figures") or []),
    ]:
        commands.append(
            {
                "phase": "analyze",
                "command": _python_script_command(project_root, str(script)),
            }
        )
    return commands


def run_appendix_artifact(project_root: Path, artifact_id: str) -> int:
    """Run one analyzer from declared reproduction outputs."""
    plan = plan_appendix_artifact(project_root, artifact_id)
    if not plan["command"]:
        raise ReproductionError(
            f"Appendix artifact {artifact_id!r} is not independently reproducible from declared inputs"
        )
    return subprocess.run(plan["command"], cwd=project_root, check=False).returncode


def run_cli(argv: list[str], project_root: Path) -> int:
    """Run the appendix audit/plan/run command group."""
    parser = argparse.ArgumentParser(prog="study.py appendix")
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit")
    audit.add_argument("artifact_id", nargs="?")
    plan = commands.add_parser("plan")
    plan.add_argument("artifact_id")
    run = commands.add_parser("run")
    run.add_argument("artifact_id")
    args = parser.parse_args(argv)
    if args.command == "audit":
        print(
            json.dumps(
                audit_appendix(project_root, args.artifact_id),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "plan":
        result = plan_appendix_artifact(project_root, args.artifact_id)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["runnable"] else 2
    return run_appendix_artifact(project_root, args.artifact_id)
