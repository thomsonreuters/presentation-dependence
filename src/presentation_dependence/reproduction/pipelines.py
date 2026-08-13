"""Load reproduction task pipelines and shared declarations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from presentation_dependence.utils.config import load_yaml_mapping


PIPELINE_FILES = {
    "passage-reranking": "passage-reranking.yaml",
    "multi-document-qa": "multi-document-qa.yaml",
    "response-ranking": "response-ranking.yaml",
}


def load_pipeline(task: str, project_root: Path) -> dict[str, Any]:
    """Load one task YAML and attach its shared declarations."""
    load_dotenv(project_root / ".env", override=False)
    try:
        filename = PIPELINE_FILES[task]
    except KeyError as exc:
        raise ValueError(f"Unknown reproduction task: {task}") from exc
    path = project_root / "configs/reproduction" / filename
    pipeline = load_yaml_mapping(path)
    pipeline["_pipeline_path"] = str(path.relative_to(project_root))
    shared = load_yaml_mapping(project_root / str(pipeline["shared"]))
    pipeline["_shared"] = shared
    return pipeline


def load_passage_reranking(project_root: Path) -> dict[str, Any]:
    """Load the passage-reranking pipeline."""
    return load_pipeline("passage-reranking", project_root)


def load_multi_document_qa(project_root: Path) -> dict[str, Any]:
    """Load the multi-document-QA pipeline."""
    return load_pipeline("multi-document-qa", project_root)


def load_response_ranking(project_root: Path) -> dict[str, Any]:
    """Load the response-ranking pipeline."""
    return load_pipeline("response-ranking", project_root)
