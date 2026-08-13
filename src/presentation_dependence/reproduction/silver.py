"""Plan, materialize, validate, and collect passage-reranking silver labels."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from presentation_dependence.silver_data.config import validate_silver_config
from presentation_dependence.silver_data.transforms import (
    derive_silver_shard,
    split_silver_cohort,
)
from presentation_dependence.utils.config import load_yaml_mapping
from presentation_dependence.utils.sweeps import write_config

from .common import (
    sha256_file,
    silver_block,
    write_silver_receipts,
)
from .errors import SilverStageError
from .pipelines import load_passage_reranking as load_passage_reranking


def _teacher_id(pipeline: Mapping[str, Any]) -> str:
    silver = silver_block(pipeline)
    return f"{silver['id']}--{silver['teacher']['id']}"


def _build_root(project_root: Path, override: Path | None = None) -> Path:
    return override or project_root / "build/reproduction/passage-reranking/silver"


def plan_silver(pipeline: Mapping[str, Any]) -> dict[str, Any]:
    """Return the passage-reranking silver workflow without writing files."""
    silver = silver_block(pipeline)
    pool = silver["candidate_pool"]
    split = silver["split"]
    return {
        "task": pipeline["task"]["id"],
        "stage": "silver",
        "teacher_config_id": _teacher_id(pipeline),
        "query_count": pool["query_count"],
        "presentations": silver["teacher"]["presentations"],
        "training_queries": split["training_queries"],
        "heldout_queries": split["heldout_queries"],
        "products": list(silver["products"]),
        "steps": [
            {
                "id": "candidate-pool",
                "script": "scripts/data/setup_msmarco_self_distill.py",
            },
            {"id": "teacher", "script": "scripts/run_silver_generation.py"},
            {
                "id": "split",
                "script": "scripts/data/prepare_heldout_eval_cohort.py",
            },
            {
                "id": "derive-k1",
                "script": "scripts/derive_k1_silver_from_bsc.py",
            },
        ],
    }


def _teacher_config(pipeline: Mapping[str, Any]) -> dict[str, Any]:
    silver = silver_block(pipeline)
    teacher = silver["teacher"]
    pool = silver["candidate_pool"]
    model = pipeline["_shared"]["models"][teacher["model"]]
    seeds = pipeline["_shared"]["seeds"][teacher["presentation_seeds"]]
    data_root = Path(pool["output_dir"])
    return {
        "id": _teacher_id(pipeline),
        "reranker": {
            "class": model["class"],
            "model_name": model["model_name"],
            "model_size": model["model_size"],
            "model_release": model["model_release"],
            "revision": model.get("revision"),
            "device": "cuda",
            "dtype": model["dtype"],
            "inference_engine": model["inference_engine"],
            "chat_template_kwargs": model["chat_template_kwargs"],
            "vllm_settings": {
                "max_model_len": model["max_length"],
                "gpu_memory_utilization": 0.85,
                "enable_prefix_caching": True,
                "tensor_parallel_size": 1,
            },
            "instruction": model["instruction"],
            "max_length": model["max_length"],
            "docs_per_score_forward": teacher["serving_width"],
            "batch_size": teacher["serving_width"],
            "max_doc_chars": model["max_doc_chars"],
        },
        "data": {
            "dataloader_class": "FixtureLoader",
            "topics": "msmarco-passage-train",
            "topics_tsv": str(data_root / pool["topics"]),
            "run_path": str(data_root / pool["fixture"]),
            "k_input": pool["candidates_per_query"],
        },
        "qids_to_run_path": str(data_root / pool["selected_qids"]),
        "eval": {
            "qrels_path": str(data_root / pool["qrels"]),
            "strict_qrels_filter": False,
            "measures": ["ndcg_cut_10"],
        },
        "teacher": {
            "protocol": "k_shot_bsc",
            "k_perms": teacher["presentations"],
            "seeds": seeds,
            "prompt_template_id": teacher["prompt_template"].replace("-", "_"),
            "grade_max": teacher["grade_max"],
            "cross_query_batch": teacher["cross_query_batch"],
            "local_data_parallel_workers": teacher["local_data_parallel_workers"],
            "output_subdir": teacher["output_subdir"],
        },
        "execution": silver["execution"],
        "logging": {"level": "INFO"},
    }


def materialize_silver(
    pipeline: Mapping[str, Any],
    project_root: Path,
    *,
    build_root: Path | None = None,
) -> dict[str, Any]:
    """Write the passage-reranking teacher config under the build root."""
    root = _build_root(project_root, build_root)
    path = root / str(silver_block(pipeline)["outputs"]["config"])
    path.parent.mkdir(parents=True, exist_ok=True)
    write_config(path, _teacher_config(pipeline))
    result = {**plan_silver(pipeline), "teacher_config": str(path)}
    (root / "materialization.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def validate_silver(
    pipeline: Mapping[str, Any],
    project_root: Path,
    *,
    build_root: Path | None = None,
    check_data: bool = True,
) -> dict[str, Any]:
    """Check the passage teacher config and candidate-pool files."""
    root = _build_root(project_root, build_root)
    config_path = root / str(silver_block(pipeline)["outputs"]["config"])
    if not config_path.is_file():
        raise SilverStageError(f"Missing teacher config: {config_path}")
    config = load_yaml_mapping(config_path)
    try:
        validate_silver_config(config, path=config_path)
    except ValueError as exc:
        raise SilverStageError(f"Invalid teacher config: {exc}") from exc
    if config.get("id") != _teacher_id(pipeline):
        raise SilverStageError("Teacher config ID drifted")
    missing: list[str] = []
    if check_data:
        pool = silver_block(pipeline)["candidate_pool"]
        data_root = project_root / str(pool["output_dir"])
        missing = [
            str(data_root / str(pool[key]))
            for key in (
                "sample_order",
                "selected_qids",
                "topics",
                "qrels",
                "run",
                "fixture",
            )
            if not (data_root / str(pool[key])).is_file()
        ]
    if missing:
        raise SilverStageError("Missing candidate-pool files: " + ", ".join(missing))
    return {
        "task": pipeline["task"]["id"],
        "stage": "silver",
        "teacher_config": str(config_path),
        "data_checked": check_data,
        "missing": missing,
    }


def collect_silver(
    pipeline: Mapping[str, Any],
    project_root: Path,
    *,
    teacher_run: Path,
    output_root: Path | None = None,
) -> dict[str, Any]:
    """Split and derive silver products, then record their SHA-256 hashes."""
    source = teacher_run.resolve() / "silver/silver_labels.jsonl"
    if not source.is_file():
        raise SilverStageError(f"Missing teacher silver: {source}")
    silver = silver_block(pipeline)
    pool = silver["candidate_pool"]
    split = silver["split"]
    destination = _build_root(project_root, output_root)
    destination.mkdir(parents=True, exist_ok=True)
    data_dir = project_root / str(pool["output_dir"])
    split_silver_cohort(
        source,
        n_heldout=int(split["heldout_queries"]),
        sample_order_path=data_dir / "sample_qids_seed42.txt",
        training_qids_path=data_dir / "qids_30k.txt",
        out_dir=destination,
        out_stem="k10",
        force=True,
    )
    products_by_id = {product["id"]: product for product in silver["products"]}
    for source_name, output_name in (
        (
            f"k10_train_{split['training_queries']}.jsonl",
            products_by_id["k1-train"]["filename"],
        ),
        (
            f"k10_heldout_{split['heldout_queries']}.jsonl",
            products_by_id["k1-heldout"]["filename"],
        ),
    ):
        derive_silver_shard(
            destination / source_name,
            destination / output_name,
            k_out=1,
            start_index=0,
            force=True,
        )
    expected = {product["id"]: destination / product["filename"] for product in silver["products"]}
    missing = [str(path) for path in expected.values() if not path.is_file()]
    if missing:
        raise SilverStageError("Missing derived silver products: " + ", ".join(missing))
    products = {product_id: {"path": str(path), "sha256": sha256_file(path)} for product_id, path in expected.items()}
    manifest = {
        "schema_version": 1,
        "task": pipeline["task"]["id"],
        "stage": "silver",
        "teacher_config_id": _teacher_id(pipeline),
        "teacher_run": str(teacher_run),
        "source": str(source),
        "source_sha256": sha256_file(source),
        "queries": split["training_queries"] + split["heldout_queries"],
        "training_queries": split["training_queries"],
        "heldout_queries": split["heldout_queries"],
        "products": products,
        "consumers": silver["consumers"],
    }
    write_silver_receipts(
        pipeline,
        destination,
        manifest,
        products,
        title="Passage-reranking silver",
        details=(
            f"- Teacher run: `{teacher_run}`",
            f"- Source: `{source}`",
        ),
    )
    return manifest
