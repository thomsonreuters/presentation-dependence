"""Check that tracked source is sufficient to rebuild reproduction plans.

This checks structural completeness only. It does not claim that GPU or API
execution has completed, and it does not verify that a fresh run re-derives the
published numbers. See ``docs/REPRODUCE.md#reproduction-fidelity``.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable


from .appendix import (
    audit_appendix,
)
from .data_setup import data_plan
from .common import sha256_file, tracked_files
from .errors import ReproductionError
from presentation_dependence.utils.config import load_yaml_mapping
from presentation_dependence.utils.sweeps import CONFIG_SUBDIRS
from .pipelines import load_pipeline
from .source_lock import verify_source_lock
from .training import plan_training


TASKS = (
    "passage-reranking",
    "multi-document-qa",
    "response-ranking",
)
EXPECTED_TRAINING_JOBS = {
    "passage-reranking": 24,
    "multi-document-qa": 18,
    "response-ranking": 30,
}
CANONICAL_STUDY_MATERIALIZERS = (
    "scripts/gen/materialize_condition.py",
    "src/presentation_dependence/reproduction/study_materialization.py",
)


def _script_package_markers(project_root: Path, relative: Path) -> set[str]:
    markers = set()
    parent = relative.parent
    while parent.parts and parent.parts[0] == "scripts":
        marker = parent / "__init__.py"
        if (project_root / marker).is_file():
            markers.add(marker.as_posix())
        parent = parent.parent
    return markers


def _existing_script_modules(project_root: Path, module_path: Path) -> set[str]:
    candidates = set()
    targets = (module_path.with_suffix(".py"), module_path / "__init__.py")
    for relative in targets:
        if not (project_root / relative).is_file():
            continue
        candidates.add(relative.as_posix())
        candidates.update(_script_package_markers(project_root, relative))
    return candidates


def _script_candidates(  # noqa: C901
    tree: ast.AST, project_root: Path, source: str
) -> set[str]:
    candidates = set()
    source_parent = Path(source).parent
    for node in ast.walk(tree):
        module_paths: list[Path] = []
        if isinstance(node, ast.Import):
            for alias in node.names:
                module = alias.name
                module_paths.append(
                    Path(module.replace(".", "/"))
                    if module.startswith("scripts.")
                    else source_parent / module.replace(".", "/")
                )
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = source_parent
                for _ in range(node.level - 1):
                    base = base.parent
                module_path = base
                if node.module:
                    module_path /= node.module.replace(".", "/")
            elif node.module and node.module.startswith("scripts."):
                module_path = Path(node.module.replace(".", "/"))
            elif node.module:
                module_path = source_parent / node.module.replace(".", "/")
            else:
                continue
            module_paths.append(module_path)
            module_paths.extend(module_path / alias.name for alias in node.names if alias.name != "*")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.startswith("scripts."):
            module_paths.append(Path(node.value.replace(".", "/")))
        for module_path in module_paths:
            candidates.update(_existing_script_modules(project_root, module_path))
    return candidates


def _python_import_closure(project_root: Path, scripts: Iterable[str]) -> set[str]:
    closure = set(scripts)
    for relative in tuple(closure):
        closure.update(_script_package_markers(project_root, Path(relative)))
    queue = list(closure)
    while queue:
        relative = queue.pop()
        path = project_root / relative
        if not path.is_file() or path.suffix != ".py":
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for candidate in _script_candidates(tree, project_root, relative) - closure:
            closure.add(candidate)
            queue.append(candidate)
    return closure


def required_source_paths(project_root: Path) -> set[str]:  # noqa: C901
    """Return the exact local source closure required for handoff."""
    required = {
        ".dockerignore",
        ".env.example",
        ".gitattributes",
        ".gitignore",
        "CITATION.cff",
        "LICENSE",
        "README.md",
        "SECURITY.md",
        "THIRD_PARTY_NOTICES.md",
        "pyproject.toml",
        "setup.sh",
        "scripts/study.py",
        "scripts/collect_study_results.py",
        "scripts/check_smoke_result.py",
        "scripts/setup_reproduction_data.py",
        "scripts/source_lock.py",
        "docs/DATA-SETUP.md",
        "docs/EVAL-PROTOCOL.md",
        "docs/HARDWARE.md",
        "docs/MODELS.md",
        "docs/REPRODUCE.md",
        "docs/RUN-ARTIFACTS.md",
        "docs/TRAINING.md",
        "docs/TROUBLESHOOTING.md",
        "docs/SCORING.md",
        "docs/results_index.yaml",
        "configs/reader/README.md",
        "configs/experiments/README.md",
        "configs/experiments/_schema.md",
        "configs/reproduction/fixtures/nectar-dedup-clean-qids.txt",
        "configs/self-distill/_schema.md",
        "configs/silver/_schema.md",
        "configs/studies/README.md",
        "configs/studies/first-stage-transfer-checkpoints.example.yaml",
        "third_party_licenses/Apache-2.0.txt",
        "third_party_licenses/PINE-MIT.txt",
        "third_party_licenses/README.md",
        "third_party_licenses/SQuAD-MIT.txt",
    }
    required.update(
        str(path.relative_to(project_root)) for path in (project_root / "configs/experiments").glob("example-*.yaml")
    )
    required.update(
        str(path.relative_to(project_root))
        for directory in ("configs/self-distill", "configs/silver", "configs/reader")
        for path in (project_root / directory).glob("example-*.yaml")
    )
    required.update(
        str(path.relative_to(project_root))
        for root in (
            project_root / "configs/reproduction",
            project_root / "src/presentation_dependence/analysis",
            project_root / "src/presentation_dependence/reproduction",
            project_root / "src/presentation_dependence/eval",
            project_root / "src/presentation_dependence/self_distill",
        )
        for path in root.rglob("*")
        if path.is_file() and path.suffix in {".json", ".py", ".yaml"}
    )
    for task in TASKS:
        pipeline = load_pipeline(task, project_root)
        plan = plan_training(pipeline, include_ablations=True)
        required.update(str(job["template"]) for job in [*plan["jobs"], *plan["ablation_jobs"]])
    appendix = audit_appendix(project_root)
    scripts = {path for path in CANONICAL_STUDY_MATERIALIZERS if path.startswith("scripts/")}
    for row in appendix["artifacts"]:
        scripts.update(row["sources"])
        if row["analyzer"]:
            scripts.add(str(row["analyzer"]))
        program = row["program"]
        if program:
            required.update(program["inputs"])
            required.update(
                path
                for path in (
                    program["study_source"],
                    program["template_manifest"]["path"] if program["template_manifest"] else None,
                )
                if path
            )
            if program["template_manifest"]:
                required.update(program["template_manifest"]["templates"])
            scripts.update(program["scripts"])
    for step in data_plan(project_root)["steps"]:
        command = step["command"]
        if not command:
            continue
        for index, token in enumerate(command):
            candidate = token
            if index and command[index - 1] == "-m":
                candidate = f"{token.replace('.', '/')}.py"
            if candidate.endswith((".py", ".sh")) and (project_root / candidate).is_file():
                scripts.add(candidate)
    required.update(_python_import_closure(project_root, scripts))
    return {path for path in required if path}


FORBIDDEN_PUBLIC_PREFIXES = ("data/", "runs/", "checkpoints/")


def _public_payload_layer(project_root: Path) -> dict[str, Any]:
    """Verify the source-only export boundary encoded in tracked files."""
    tracked = tracked_files(project_root)
    forbidden = sorted(
        relative
        for relative in tracked
        if relative in {"data", "runs", "checkpoints"} or relative.startswith(FORBIDDEN_PUBLIC_PREFIXES)
    )
    attributes_path = project_root / ".gitattributes"
    export_ignore = False
    if attributes_path.is_file():
        export_ignore = any(
            line.split() == ["docs/RELEASE.md", "export-ignore"]
            for line in attributes_path.read_text(encoding="utf-8").splitlines()
        )
    return {
        "status": "pass" if not forbidden and export_ignore else "fail",
        "forbidden_tracked_paths": forbidden,
        "release_workpaper_export_ignored": export_ignore,
    }


# Config keys that addressed a job scheduler this distribution does not talk
# to. Nothing reads them, so one reappearing means a config was copied from an
# older tree and now documents behaviour that will not happen. `aws` is the
# former name of the `execution` block: still load-bearing under the new name,
# so a config carrying the old one loses its execution settings silently.
RETIRED_CONFIG_KEYS = frozenset(
    {
        "aws",
        "checkpoint_bucket",
        "cli_args",
        "framework_version",
        "image_uri",
        "instance_count",
        "instance_type",
        "instance_types",
        "max_run_s",
        "py_version",
        "sagemaker_experiment_name",
        "volume_size",
    }
)

# Strings that must never reach a published config: an account, a workspace, a
# private index, or a bucket URI.
RETIRED_CONFIG_VALUES = (
    re.compile(r"\ba\d{6}-"),
    re.compile(r"\bs3://"),
    re.compile(r"\.dkr\.ecr\."),
    re.compile(r"\bjfrog\b", re.IGNORECASE),
)

# Checkpoints under a non-commercial licence: jina-reranker-v3 is CC-BY-NC-4.0,
# SPLADE++ EnsembleDistil is CC-BY-NC-SA-4.0. Either may be evaluated. Neither
# may train anything, because a candidate set, silver label set, or adapter
# derived from one inherits terms this repository cannot satisfy, and
# share-alike makes the SPLADE case worse than the jina one.
NON_COMMERCIAL_CHECKPOINTS = (
    re.compile(r"jina-?reranker", re.IGNORECASE),
    re.compile(r"splade", re.IGNORECASE),
)

# Blocks that put a checkpoint into a training role. `reranker:`, reference
# lists, and `first_stage_tokens:` are evaluation surfaces and stay
# unrestricted, which is why this keys off the block rather than the checkpoint.
TRAINING_ROLE_BLOCKS = frozenset({"teacher", "student"})
TRAINING_ROLE_KEYS = frozenset({"base_model", "teacher_model", "silver_model"})


def _walk_config(node: Any, path: str = "") -> Iterable[tuple[str, str, Any]]:
    """Yield ``(dotted_path, key, value)`` for every mapping entry, at any depth."""
    if isinstance(node, dict):
        for key, value in node.items():
            here = f"{path}.{key}" if path else str(key)
            yield here, str(key), value
            yield from _walk_config(value, here)
    elif isinstance(node, list):
        for index, item in enumerate(node):
            yield from _walk_config(item, f"{path}[{index}]")


def _retired_config_usage(project_root: Path) -> list[str]:
    """Return every tracked config still carrying a retired key or value.

    The earlier removals were anchored on the block a key usually sits in, so
    copies in differently shaped blocks survived. This looks at every mapping
    entry at any depth instead, which is what makes it a check rather than
    another sweep.
    """
    findings: list[str] = []
    for path in sorted((project_root / "configs").rglob("*.yaml")):
        try:
            document = load_yaml_mapping(path)
        except Exception:  # unreadable YAML is the yaml layer's problem
            continue
        relative = path.relative_to(project_root)
        for dotted, key, value in _walk_config(document):
            if key in RETIRED_CONFIG_KEYS:
                findings.append(f"{relative}: retired key `{dotted}`")
            if isinstance(value, str) and any(rx.search(value) for rx in RETIRED_CONFIG_VALUES):
                findings.append(f"{relative}: retired value at `{dotted}`")
    findings.extend(_retired_generator_usage(project_root))
    findings.extend(_dangling_sweep_references(project_root))
    return findings


def _non_commercial_training_usage(project_root: Path) -> list[str]:
    """Return tracked configs placing a non-commercial checkpoint in a training role.

    ``THIRD_PARTY_NOTICES.md`` states the boundary; this check enforces it.
    Promoting a reference model to a teacher is one config line and changes the
    licensing question as well as the science, so the gate fails that edit
    rather than relying on reviewer knowledge of the rule.
    """
    findings: list[str] = []
    for path in sorted((project_root / "configs").rglob("*.yaml")):
        try:
            document = load_yaml_mapping(path)
        except Exception:  # unreadable YAML is the yaml layer's problem
            continue
        relative = path.relative_to(project_root)
        for dotted, key, value in _walk_config(document):
            if not isinstance(value, str):
                continue
            if not any(rx.search(value) for rx in NON_COMMERCIAL_CHECKPOINTS):
                continue
            if dotted.split(".", 1)[0] in TRAINING_ROLE_BLOCKS or key in TRAINING_ROLE_KEYS:
                findings.append(f"{relative}: non-commercial checkpoint in a training role at `{dotted}`")
    return findings


def _dangling_sweep_references(project_root: Path) -> list[str]:
    """Return tracked sweeps whose jobs name a config that is not tracked.

    ``configs/experiments/*.yaml`` is gitignored apart from the reviewed
    examples, while ``configs/sweeps/`` is not, so materialising a study leaves
    sweeps behind that resolve locally and point at nothing in a fresh clone.
    Only tracked sweeps are judged; the leftovers are the thing being guarded
    against, not a failure in themselves.
    """
    tracked = tracked_files(project_root)
    findings: list[str] = []
    for path in sorted((project_root / "configs" / "sweeps").glob("*.yaml")):
        relative = str(path.relative_to(project_root))
        if relative not in tracked:
            continue
        try:
            document = load_yaml_mapping(path)
        except Exception:
            continue
        for job in document.get("jobs") or []:
            exp_id = str((job or {}).get("exp_id") or "").strip()
            if not exp_id:
                continue
            candidates = (
                [exp_id]
                if exp_id.endswith(".yaml")
                else [f"configs/{subdir}/{exp_id}.yaml" for subdir in CONFIG_SUBDIRS]
            )
            if not any(candidate in tracked for candidate in candidates):
                findings.append(f"{relative}: job `{exp_id}` resolves to no tracked config")
    return findings


# The generators emit YAML from string templates, so a retired key in one is
# invisible to the walk above until someone runs it and commits the output.
_GENERATOR_KEY_LINE = re.compile(
    rf"^\s*({'|'.join(sorted(RETIRED_CONFIG_KEYS))})\s*:\s*\S",
    re.MULTILINE,
)


def _retired_generator_usage(project_root: Path) -> list[str]:
    """Return every config generator whose template still writes a retired key."""
    findings: list[str] = []
    for path in sorted((project_root / "scripts" / "gen").rglob("*.py")):
        relative = path.relative_to(project_root)
        text = path.read_text(encoding="utf-8")
        for line_number, line in enumerate(text.splitlines(), start=1):
            match = _GENERATOR_KEY_LINE.match(line)
            if match:
                findings.append(f"{relative}:{line_number}: generator emits retired key `{match.group(1)}`")
    return findings


def _tracking_layer(project_root: Path) -> dict[str, Any]:
    required = sorted(required_source_paths(project_root))
    tracked = tracked_files(project_root)
    missing = [relative for relative in required if not (project_root / relative).is_file()]
    untracked = [relative for relative in required if (project_root / relative).is_file() and relative not in tracked]
    return {
        "status": "pass" if not missing and not untracked else "fail",
        "required_count": len(required),
        "missing_required": missing,
        "untracked_required": untracked,
    }


def _recipe_layer(project_root: Path) -> dict[str, Any]:
    rows = []
    mismatches = []
    for task in TASKS:
        pipeline = load_pipeline(task, project_root)
        plan = plan_training(pipeline)
        actual = int(plan["job_count"])
        expected = EXPECTED_TRAINING_JOBS[task]
        rows.append(
            {
                "task": task,
                "training_jobs": actual,
                "expected_training_jobs": expected,
            }
        )
        if actual != expected:
            mismatches.append(f"{task}: expected {expected} training jobs, found {actual}")
        ids = [str(job["id"]) for job in plan["jobs"]]
        if len(ids) != len(set(ids)):
            mismatches.append(f"{task}: duplicate training config IDs")
    mismatches.extend(_internal_campaign_ids(project_root))
    mismatches.extend(_first_stage_checkpoint_hygiene(project_root))
    return {
        "status": "pass" if not mismatches else "fail",
        "tasks": rows,
        "mismatches": mismatches,
    }


def _first_stage_checkpoint_hygiene(project_root: Path) -> list[str]:
    """Keep physical checkpoint identities out of the tracked study declaration."""
    relative = Path("configs/studies/first-stage-transfer.yaml")
    document = load_yaml_mapping(project_root / relative)
    findings: list[str] = []
    binding_path = Path(str(document.get("checkpoint_bindings") or ""))
    if (
        binding_path.is_absolute()
        or not binding_path.parts
        or binding_path.parts[0] != "build"
        or ".." in binding_path.parts
    ):
        findings.append(f"{relative}: checkpoint_bindings must point under ignored build/")
    scorers = document.get("scorers")
    if not isinstance(scorers, dict):
        return [*findings, f"{relative}: scorers must be a mapping"]
    physical_keys = {"experiment", "trial", "step", "checkpoint_uri", "lora_path"}
    seen_refs: set[str] = set()
    for scorer_id, scorer in scorers.items():
        if not isinstance(scorer, dict):
            findings.append(f"{relative}: scorer {scorer_id!r} must be a mapping")
            continue
        present = sorted(physical_keys & set(scorer))
        if present:
            findings.append(f"{relative}: scorer {scorer_id!r} embeds physical checkpoint keys {present}")
        if scorer.get("kind") != "lora":
            continue
        checkpoint_ref = str(scorer.get("checkpoint_ref") or "")
        if not checkpoint_ref:
            findings.append(f"{relative}: LoRA scorer {scorer_id!r} has no checkpoint_ref")
        elif checkpoint_ref in seen_refs:
            findings.append(f"{relative}: duplicate checkpoint_ref {checkpoint_ref!r}")
        else:
            seen_refs.add(checkpoint_ref)
        if scorer.get("training_seed") is None:
            findings.append(f"{relative}: LoRA scorer {scorer_id!r} has no training_seed")
    return findings


# Case-sensitive on purpose. These identifiers are always written one way, and
# matching case-insensitively across the whole config tree hits opaque data:
# `ae84` inside a Signal-1M qid UUID is not a campaign name.
INTERNAL_CAMPAIGN_ID = re.compile(r"\b(?:AE\d+|A-E\d+|A8c|gen_ae)\b")


def _internal_campaign_ids(project_root: Path) -> list[str]:
    """Return tracked config-tree files exposing a historical campaign identifier.

    This used to check only the recipe files, which let the same identifiers sit
    in the evidence tables, the schemas, and analysis scripts that indexed a run
    registry nobody outside can see. Everything tracked under ``configs/`` and
    ``scripts/`` is checked instead.

    ``src/`` is exempt because that is where the ban lives: the pattern below,
    and the generated-id guards in the training and direct-eval stages, all have
    to name what they reject.
    """
    tracked = tracked_files(project_root)
    findings: list[str] = []
    for path in sorted([*(project_root / "configs").rglob("*"), *(project_root / "scripts").rglob("*")]):
        if not path.is_file():
            continue
        relative = str(path.relative_to(project_root))
        if relative not in tracked:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if INTERNAL_CAMPAIGN_ID.search(text):
            findings.append(f"{relative}: exposes a historical campaign identifier")
    return findings


def _historical_materializer_reads(project_root: Path) -> list[str]:
    forbidden = (
        "resolved_config.yaml",
        'ROOT / "runs"',
        "latest completed",
        "s3://",
        "configs/experiments/AE",
        "configs/experiments/A8c",
        "a8c-s1",
    )
    return [
        f"{relative}: {token}"
        for relative in CANONICAL_STUDY_MATERIALIZERS
        for token in forbidden
        if token in (project_root / relative).read_text(encoding="utf-8")
    ]


def _legacy_figure_tree_files(project_root: Path) -> list[str]:
    """Return files that violate the collapsed legacy-figure-tree invariant."""
    root = project_root / "docs" / "figures"
    if not root.exists():
        return []
    return sorted(str(path.relative_to(project_root)) for path in root.rglob("*") if path.is_file())


def _legacy_figure_script_references(project_root: Path) -> list[str]:
    """Reject maintained scripts that could recreate the retired tree."""
    token = "docs" + "/figures"
    return sorted(
        str(path.relative_to(project_root))
        for path in (project_root / "scripts").rglob("*")
        if path.is_file()
        and path.suffix in {".py", ".sh"}
        and token in path.read_text(encoding="utf-8", errors="replace")
    )


FIDELITY_DISCLAIMER = {
    "statement": (
        "Post-hoc, best-attempt reproduction. Exact, bit-identical agreement with published numbers is not guaranteed."
    ),
    "what_this_checks": (
        "Tracked source is sufficient to rebuild the plans and high-level "
        "declared model/file identities match the source lock. Local file "
        "contents are not checked by this audit, and complete does not mean "
        "the reported metrics were recomputed."
    ),
    "details": "docs/REPRODUCE.md#reproduction-fidelity",
}


def audit_structural_completeness(project_root: Path) -> dict[str, Any]:
    """Audit source closure independently from generated execution evidence."""
    historical_reads = _historical_materializer_reads(project_root)
    retired_config_usage = _retired_config_usage(project_root)
    non_commercial_training_usage = _non_commercial_training_usage(project_root)
    legacy_figure_files = _legacy_figure_tree_files(project_root)
    legacy_figure_script_references = _legacy_figure_script_references(project_root)
    try:
        source_lock = verify_source_lock(project_root, check_files=False)
        source_lock["verification_status"] = source_lock["status"]
        source_lock["status"] = "pass" if source_lock["verification_status"] == "ok" else "fail"
    except ReproductionError as exc:
        source_lock = {"status": "error", "error": str(exc)}
    layers = {
        "git_tracking": _tracking_layer(project_root),
        "public_payload": _public_payload_layer(project_root),
        "compact_recipes": _recipe_layer(project_root),
        "config_hygiene": {
            "status": "pass" if not retired_config_usage and not non_commercial_training_usage else "fail",
            "retired_config_usage": retired_config_usage,
            "non_commercial_training_usage": non_commercial_training_usage,
        },
        "source_closure": {
            "status": "pass"
            if not historical_reads and not legacy_figure_files and not legacy_figure_script_references
            else "fail",
            "historical_materializer_reads": historical_reads,
            "legacy_figure_files": legacy_figure_files,
            "legacy_figure_script_references": legacy_figure_script_references,
        },
        "source_lock": source_lock,
    }
    complete = all(layer["status"] == "pass" for layer in layers.values())
    return {
        "schema_version": 1,
        "status": "complete" if complete else "incomplete",
        "disclaimer": FIDELITY_DISCLAIMER,
        "layers": layers,
        "summary": {
            "passing_layers": sum(layer["status"] == "pass" for layer in layers.values()),
            "layers": len(layers),
            "untracked_required": len(layers["git_tracking"]["untracked_required"]),
            "missing_required": len(layers["git_tracking"]["missing_required"]),
        },
    }


def write_structural_snapshot(project_root: Path, output: Path | None = None) -> Path:
    """Write a hash-bearing structural-completeness receipt."""
    result = audit_structural_completeness(project_root)
    result["created_at"] = datetime.now(UTC).isoformat()
    tracked_sources = sorted(
        set(required_source_paths(project_root)) - set(result["layers"]["git_tracking"]["untracked_required"])
    )
    result["tracked_source_hashes"] = {
        relative: sha256_file(project_root / relative)
        for relative in tracked_sources
        if (project_root / relative).is_file()
    }
    destination = output or (project_root / "build/reproduction/structural-proof/snapshot.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def run_cli(argv: list[str], project_root: Path) -> int:
    """Run structural audit or snapshot commands."""
    parser = argparse.ArgumentParser(prog="study.py structural")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("audit")
    snapshot = commands.add_parser("snapshot")
    snapshot.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.command == "snapshot":
        print(write_structural_snapshot(project_root, args.output))
        return 0
    result = audit_structural_completeness(project_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "complete" else 2
