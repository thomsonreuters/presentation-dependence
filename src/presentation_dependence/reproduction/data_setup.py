"""Plan and execute canonical paper data materialization."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from presentation_dependence.eval.dataset_catalog import load_dataset_population

NECTAR_HF_REVISION = "3c6b4c47fa1cc38869f9f32dce1699f7abad8b06"


@dataclass(frozen=True)
class DataStep:
    """One deterministic setup command or documented access-gated step."""

    id: str
    task: str
    command: tuple[str, ...] | None
    outputs: tuple[str, ...]
    access: str = "public"
    note: str | None = None


@dataclass(frozen=True)
class DatasetSetupAdapter:
    """Setup mechanics for one canonical reranking dataset."""

    mechanic: str
    access: str = "public"
    topics: str | None = None
    qrels: str | None = None
    slug: str | None = None
    remove_query: bool = False
    note: str | None = None


def _python(script: str, *arguments: str) -> tuple[str, ...]:
    module = ".".join(Path(script).with_suffix("").parts)
    return (sys.executable, "-m", module, *arguments)


RERANKING_ADAPTERS: dict[str, DatasetSetupAdapter] = {
    "dl19": DatasetSetupAdapter("pyserini"),
    "dl20": DatasetSetupAdapter("pyserini", topics="dl20"),
    "dl21": DatasetSetupAdapter(
        "manual",
        note=(
            "Manual: obtain the official TREC DL 2021 scoreddocs run and judged "
            "topics/qrels, then build the fixture with "
            "scripts/data/build_fixture_irds.py using dataset "
            "msmarco-passage-v2/trec-dl-2021. Source input paths are not "
            "declared, so no command runs implicitly."
        ),
    ),
    "dl22": DatasetSetupAdapter(
        "manual",
        note=(
            "Manual: obtain the official TREC DL 2022 scoreddocs run and judged "
            "topics/qrels, then build the fixture with "
            "scripts/data/build_fixture_irds.py using dataset "
            "msmarco-passage-v2/trec-dl-2022. Source input paths are not "
            "declared, so no command runs implicitly."
        ),
    ),
    "dl23": DatasetSetupAdapter(
        "manual",
        note=(
            "Manual: obtain the official TREC DL 2023 scoreddocs run and judged "
            "topics/qrels, then build the fixture with "
            "scripts/data/build_fixture_irds.py using dataset "
            "msmarco-passage-v2/trec-dl-2023. Source input paths are not "
            "declared, so no command runs implicitly."
        ),
    ),
    "touche2020": DatasetSetupAdapter("pyserini"),
    "fiqa": DatasetSetupAdapter("pyserini"),
    "nfcorpus": DatasetSetupAdapter("pyserini"),
    "arguana": DatasetSetupAdapter("pyserini", remove_query=True),
    "climate-fever": DatasetSetupAdapter("pyserini"),
    "trec-covid": DatasetSetupAdapter("pyserini"),
    "dbpedia-entity": DatasetSetupAdapter("pyserini", slug="dbpedia"),
    "scifact": DatasetSetupAdapter("pyserini"),
    "signal1m": DatasetSetupAdapter(
        "pyserini",
        access="gated",
        note=(
            "Access-gated Signal-1M data is not fetched by default. Confirm the "
            "Signal Media license and local Pyserini access before using "
            "--include-gated."
        ),
    ),
    "trec-news": DatasetSetupAdapter(
        "pyserini",
        access="gated",
        note=(
            "Access-gated TREC News data is not fetched by default. Obtain the "
            "required NIST collection access before using --include-gated."
        ),
    ),
    "robust04": DatasetSetupAdapter(
        "pyserini",
        access="gated",
        note=(
            "Access-gated Robust04 data is not fetched by default. Obtain the "
            "required TREC/NIST collection access before using --include-gated."
        ),
    ),
    "legal-a": DatasetSetupAdapter(
        "manual",
        access="internal",
        note=(
            "Internal/manual: materialize the authorized Legal-A "
            "fixture and qrels at the declared paths. No setup "
            "command runs implicitly; the fixture must not be "
            "redistributed."
        ),
    ),
    "legal-b": DatasetSetupAdapter(
        "manual",
        access="internal",
        note=(
            "Internal/manual: materialize the authorized Legal-B "
            "fixture and qrels at the declared paths. No setup "
            "command runs implicitly; the fixture must not be "
            "redistributed."
        ),
    ),
}


def _declared_outputs(dataset: dict[str, Any]) -> tuple[str, ...]:
    """Return deduplicated required population outputs in declaration order."""
    return tuple(dict.fromkeys(str(dataset[key]) for key in ("run_path", "qrels_path") if dataset.get(key)))


def _pyserini_command(dataset: dict[str, Any], adapter: DatasetSetupAdapter) -> tuple[str, ...]:
    """Build one fetch command that produces the population's exact paths."""
    dataset_id = str(dataset["id"])
    missing = [key for key in ("topics", "index", "run_path", "qrels_path") if not dataset.get(key)]
    if missing:
        raise ValueError(f"Pyserini dataset {dataset_id!r} is missing fields: {missing}")

    run_path = Path(str(dataset["run_path"]))
    qrels_path = Path(str(dataset["qrels_path"]))
    topics_path = Path(str(dataset.get("topics_tsv", run_path.parent / "topics.tsv")))
    if (
        qrels_path.parent != run_path.parent
        or qrels_path.name != "qrels.txt"
        or topics_path.parent != run_path.parent
        or topics_path.name != "topics.tsv"
    ):
        raise ValueError(
            f"Pyserini dataset {dataset_id!r} does not use the supported single-directory topics.tsv/qrels.txt layout"
        )

    command = [
        *_python("scripts/data/fetch_pyserini_dataset.py"),
        "--topics",
        adapter.topics or str(dataset["topics"]),
        "--qrels",
        adapter.qrels or str(dataset["topics"]),
        "--index",
        str(dataset["index"]),
        "--slug",
        adapter.slug or dataset_id,
        "--out",
        str(run_path.parent),
        "--run-name",
        run_path.name,
    ]
    if adapter.remove_query:
        command.append("--remove-query")
    return tuple(command)


def _reranking_steps(project_root: Path) -> tuple[DataStep, ...]:
    """Derive canonical passage-reranking rows from the population YAML."""
    datasets = load_dataset_population("reranking-primary-18", configs_root=project_root / "configs")
    population_ids = {str(dataset["id"]) for dataset in datasets}
    registry_ids = set(RERANKING_ADAPTERS)
    if population_ids != registry_ids:
        raise ValueError(
            "Reranking setup registry drift: "
            f"missing={sorted(population_ids - registry_ids)}, "
            f"extra={sorted(registry_ids - population_ids)}"
        )

    steps = []
    for dataset in datasets:
        dataset_id = str(dataset["id"])
        adapter = RERANKING_ADAPTERS[dataset_id]
        declared_access = str(dataset.get("access", adapter.access))
        if declared_access != adapter.access:
            raise ValueError(
                f"Access metadata disagrees for {dataset_id!r}: "
                f"population={declared_access!r}, adapter={adapter.access!r}"
            )
        command = _pyserini_command(dataset, adapter) if adapter.mechanic == "pyserini" else None
        steps.append(
            DataStep(
                dataset_id,
                "passage-reranking",
                command,
                _declared_outputs(dataset),
                access=adapter.access,
                note=adapter.note,
            )
        )
    return tuple(steps)


AUXILIARY_STEPS = (
    DataStep(
        "hotpot-train",
        "multi-document-qa",
        _python(
            "scripts/data/setup_hotpotqa_support_dataset.py",
            "--split",
            "train",
            "--max-queries",
            "30500",
        ),
        ("data/hotpotqa-distractor-support-train/fixture.jsonl",),
    ),
    DataStep(
        "hotpot-dev",
        "multi-document-qa",
        _python(
            "scripts/data/setup_hotpotqa_support_dataset.py",
            "--split",
            "dev",
            "--max-queries",
            "200",
        ),
        (
            "data/hotpotqa-distractor-support-dev/fixture.jsonl",
            "data/hotpotqa-distractor-support-dev/qrels.txt",
        ),
    ),
    *(
        DataStep(
            dataset,
            "multi-document-qa",
            _python(
                "scripts/data/setup_multihopqa_support_dataset.py",
                "--dataset",
                dataset,
                "--max-queries",
                "200",
            ),
            (f"data/{slug}/fixture.jsonl", f"data/{slug}/qrels.txt"),
        )
        for dataset, slug in (
            ("2wiki", "2wiki-distractor-support-dev"),
            ("musique", "musique-support-dev"),
        )
    ),
    *(
        DataStep(
            f"{dataset}-answers",
            "multi-document-qa",
            _python("scripts/data/setup_qa_answers.py", "--dataset", dataset),
            (f"data/{slug}/answers.jsonl",),
        )
        for dataset, slug in (
            ("hotpotqa", "hotpotqa-distractor-support-dev"),
            ("2wiki", "2wiki-distractor-support-dev"),
            ("musique", "musique-support-dev"),
        )
    ),
    DataStep(
        "climate-fever-verdicts",
        "multi-document-qa",
        _python(
            "scripts/data/setup_verdict_data.py",
            "--dataset",
            "climate-fever",
        ),
        (
            "data/beir-v1.0.0-climate-fever-test/verdicts.jsonl",
            "data/beir-v1.0.0-climate-fever-test/verdict_qids.txt",
        ),
    ),
    DataStep(
        "ultrafeedback-train",
        "response-ranking",
        _python(
            "scripts/data/setup_ultrafeedback_response_quality_dataset.py",
            "--max-queries",
            "30700",
        ),
        ("data/ultrafeedback-response-quality-train/fixture.jsonl",),
    ),
    DataStep(
        "rewardbench2",
        "response-ranking",
        _python("scripts/data/setup_rewardbench2_response_quality_dataset.py"),
        (
            "data/rewardbench2-response-quality-test/fixture.jsonl",
            "data/rewardbench2-response-quality-test/qrels.txt",
        ),
    ),
    DataStep(
        "nectar",
        "response-ranking",
        _python(
            "scripts/data/setup_nectar_response_quality_dataset.py",
            "--revision",
            NECTAR_HF_REVISION,
            "--max-queries",
            "500",
        ),
        (
            "data/nectar-response-quality/fixture.jsonl",
            "data/nectar-response-quality/qrels.txt",
        ),
    ),
    DataStep(
        "ppe-math",
        "response-ranking",
        _python(
            "scripts/data/setup_ppe_response_quality_dataset.py",
            "--dataset",
            "lmarena-ai/PPE-MATH-Best-of-K",
            "--out",
            "data/ppe-math-response-quality",
        ),
        (
            "data/ppe-math-response-quality/fixture.jsonl",
            "data/ppe-math-response-quality/qrels.txt",
        ),
    ),
    DataStep(
        "ppe-mmlu-pro",
        "response-ranking",
        _python(
            "scripts/data/setup_ppe_response_quality_dataset.py",
            "--dataset",
            "lmarena-ai/PPE-MMLU-Pro-Best-of-K",
            "--out",
            "data/ppe-mmlu-pro-response-quality",
        ),
        (
            "data/ppe-mmlu-pro-response-quality/fixture.jsonl",
            "data/ppe-mmlu-pro-response-quality/qrels.txt",
        ),
    ),
    DataStep(
        "rmbench",
        "response-ranking",
        _python("scripts/data/setup_rmbench_response_quality_dataset.py"),
        (
            "data/rmbench-response-quality/fixture.jsonl",
            "data/rmbench-response-quality/qrels.txt",
        ),
    ),
)


def data_plan(
    project_root: Path,
    *,
    task: str = "all",
    include_internal: bool = False,
    include_gated: bool = False,
) -> dict[str, Any]:
    """Return setup commands and readiness for selected canonical tasks."""
    steps = (*_reranking_steps(project_root), *AUXILIARY_STEPS)
    selected = [step for step in steps if (task == "all" or step.task == task)]
    rows = []
    for step in selected:
        missing = [output for output in step.outputs if not (project_root / output).is_file()]
        enabled = (
            step.access == "public"
            or (step.access == "internal" and include_internal)
            or (step.access == "gated" and include_gated)
        )
        rows.append(
            {
                "id": step.id,
                "task": step.task,
                "access": step.access,
                "command": list(step.command) if step.command else None,
                "outputs": list(step.outputs),
                "missing_outputs": missing,
                "ready": not missing,
                "enabled": enabled,
                "will_run": enabled and step.command is not None and bool(missing),
                "note": step.note,
            }
        )
    return {
        "task": task,
        "include_internal": include_internal,
        "include_gated": include_gated,
        "steps": rows,
        "summary": {
            "steps": len(rows),
            "ready": sum(row["ready"] for row in rows),
            "manual": sum(row["command"] is None for row in rows),
            "will_run": sum(row["will_run"] for row in rows),
        },
    }


def run_data_setup(
    project_root: Path,
    *,
    task: str,
    include_internal: bool,
    include_gated: bool = False,
) -> int:
    """Execute missing automated steps in declared order."""
    plan = data_plan(
        project_root,
        task=task,
        include_internal=include_internal,
        include_gated=include_gated,
    )
    for step in plan["steps"]:
        if not step["will_run"]:
            continue
        completed = subprocess.run(step["command"], cwd=project_root, check=False)
        if completed.returncode:
            return completed.returncode
    return 0


def validate_population_outputs(project_root: Path, task: str) -> list[str]:
    """Return missing paths from the canonical evaluation population."""
    populations = {
        "passage-reranking": (
            "reranking-primary-18",
            "reranking-frozen-18",
        ),
        "multi-document-qa": ("multi-document-qa-3",),
        "response-ranking": ("response-ranking-5",),
    }[task]
    datasets = [
        dataset
        for population in populations
        for dataset in load_dataset_population(population, configs_root=project_root / "configs")
    ]
    return sorted(
        {
            str(path)
            for dataset in datasets
            for key in ("run_path", "qrels_path")
            if dataset.get(key) and not (path := project_root / str(dataset[key])).is_file()
        }
    )
