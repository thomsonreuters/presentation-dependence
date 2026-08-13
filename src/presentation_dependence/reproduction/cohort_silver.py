"""Plan two-cohort silver data for QA and response ranking."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Mapping

from presentation_dependence.silver_data.config import load_silver_config
from presentation_dependence.silver_data.transforms import derive_silver_shard
from presentation_dependence.utils.sweeps import write_config

from .common import (
    prepare_generated_directory,
    sha256_file,
    silver_block,
    write_silver_receipts,
)
from .errors import SilverStageError
from .pipelines import (
    load_multi_document_qa as load_multi_document_qa,
    load_response_ranking as load_response_ranking,
)


def _root(
    pipeline: Mapping[str, Any],
    project_root: Path,
    override: Path | None = None,
) -> Path:
    return override or project_root / str(silver_block(pipeline)["outputs"]["root"])


def _teacher_id(pipeline: Mapping[str, Any], teacher: Mapping[str, Any]) -> str:
    return f"{silver_block(pipeline)['id']}--{teacher['id']}"


def plan_cohort_silver(pipeline: Mapping[str, Any]) -> dict[str, Any]:
    """Return a two-cohort silver workflow without writing files."""
    silver = silver_block(pipeline)
    pool = silver["candidate_pool"]
    task_id = pipeline["task"]["id"]
    setup_script = (
        "scripts/data/setup_hotpotqa_support_dataset.py"
        if task_id == "multi-document-qa"
        else "scripts/data/setup_ultrafeedback_response_quality_dataset.py"
    )
    steps = [
        {"id": "candidate-pool", "script": setup_script},
        {"id": "teacher", "script": "scripts/run_silver_generation.py"},
        {"id": "derive-k1", "script": "scripts/derive_k1_silver_from_bsc.py"},
    ]
    if task_id == "multi-document-qa":
        steps.insert(
            1,
            {"id": "heldout-qids", "script": "scripts/data/make_qid_delta.py"},
        )
    return {
        "task": task_id,
        "stage": "silver",
        "teachers": [
            {
                "id": _teacher_id(pipeline, teacher),
                "cohort": teacher["cohort"],
                "queries": (pool["training_queries"] if teacher["cohort"] == "training" else pool["heldout_queries"]),
            }
            for teacher in silver["teachers"]
        ],
        "products": list(silver["products"]),
        "steps": steps,
    }


def _teacher_config(pipeline: Mapping[str, Any], teacher: Mapping[str, Any]) -> dict[str, Any]:
    silver = silver_block(pipeline)
    pool = silver["candidate_pool"]
    model = pipeline["_shared"]["models"][teacher["model"]]
    scoring = silver["scoring"]
    data_root = Path(pool["output_dir"])
    execution_block = {**silver["execution"], **teacher.get("execution", {})}
    presentations = int(teacher["presentations"])
    teacher_block = {
        "protocol": "k_shot_bsc",
        "k_perms": presentations,
        "seeds": list(pipeline["_shared"]["seeds"]["presentations"])[:presentations],
        "prompt_template_id": scoring["prompt_template"].replace("-", "_"),
        "grade_max": 3,
        "cross_query_batch": teacher["cross_query_batch"],
        "output_subdir": "silver",
    }
    if teacher.get("local_data_parallel_workers"):
        teacher_block["local_data_parallel_workers"] = teacher["local_data_parallel_workers"]
    return {
        "id": _teacher_id(pipeline, teacher),
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
                "max_model_len": scoring["max_length"],
                "gpu_memory_utilization": 0.85,
                "enable_prefix_caching": True,
                "tensor_parallel_size": 1,
                "top_logprobs": 50,
            },
            "instruction": scoring["instruction"],
            "grade_rubric_id": scoring["grade_rubric"],
            "max_length": scoring["max_length"],
            "docs_per_score_forward": teacher["serving_width"],
            "batch_size": teacher["serving_width"],
            "max_doc_chars": scoring["max_doc_chars"],
        },
        "data": {
            "dataloader_class": "FixtureLoader",
            "topics": scoring["topics_id"],
            "topics_tsv": str(data_root / pool["topics"]),
            "run_path": str(data_root / pool["fixture"]),
            "k_input": pool["candidates_per_query"],
        },
        "qids_to_run_path": str(data_root / teacher["qids"]),
        "eval": {
            "qrels_path": str(data_root / pool["qrels"]),
            "strict_qrels_filter": True,
            "measures": ["ndcg_cut_10"],
        },
        "teacher": teacher_block,
        "execution": execution_block,
        "logging": {"level": "INFO"},
    }


def materialize_cohort_silver(
    pipeline: Mapping[str, Any],
    project_root: Path,
    *,
    build_root: Path | None = None,
) -> dict[str, Any]:
    """Write training and held-out teacher configs."""
    root = _root(pipeline, project_root, build_root)
    configs_dir = prepare_generated_directory(root / str(silver_block(pipeline)["outputs"]["configs_dir"]))
    config_paths: list[str] = []
    for teacher in silver_block(pipeline)["teachers"]:
        path = configs_dir / f"{teacher['id']}.yaml"
        write_config(path, _teacher_config(pipeline, teacher))
        config_paths.append(str(path))
    result = {**plan_cohort_silver(pipeline), "teacher_configs": config_paths}
    (root / "materialization.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def validate_cohort_silver(
    pipeline: Mapping[str, Any],
    project_root: Path,
    *,
    build_root: Path | None = None,
    check_data: bool = True,
) -> dict[str, Any]:
    """Validate cohort teacher configs and candidate-pool files."""
    root = _root(pipeline, project_root, build_root)
    configs = sorted((root / str(silver_block(pipeline)["outputs"]["configs_dir"])).glob("*.yaml"))
    if len(configs) != len(silver_block(pipeline)["teachers"]):
        raise SilverStageError("Expected one teacher config per cohort")
    for config_path in configs:
        try:
            load_silver_config(config_path)
        except (FileNotFoundError, ValueError) as exc:
            raise SilverStageError(f"Invalid teacher config {config_path}: {exc}") from exc
    missing: list[str] = []
    if check_data:
        pool = silver_block(pipeline)["candidate_pool"]
        data_root = project_root / str(pool["output_dir"])
        keys = [
            "all_qids",
            "training_qids",
            "heldout_qids",
            "topics",
            "qrels",
            "run",
            "fixture",
        ]
        if pool.get("dev_qids"):
            keys.append("dev_qids")
        missing = [str(data_root / str(pool[key])) for key in keys if not (data_root / str(pool[key])).is_file()]
    if missing:
        raise SilverStageError("Missing candidate-pool files: " + ", ".join(missing))
    return {
        "task": pipeline["task"]["id"],
        "stage": "silver",
        "teacher_configs": [str(path) for path in configs],
        "data_checked": check_data,
        "missing": missing,
    }


def collect_cohort_silver(
    pipeline: Mapping[str, Any],
    project_root: Path,
    *,
    training_teacher_run: Path,
    heldout_teacher_run: Path,
    pointwise_training_teacher_run: Path | None = None,
    pointwise_heldout_teacher_run: Path | None = None,
    output_root: Path | None = None,
) -> dict[str, Any]:
    """Derive K=1 train and held-out products from completed K=10 silver."""
    silver = silver_block(pipeline)
    destination = _root(pipeline, project_root, output_root)
    destination.mkdir(parents=True, exist_ok=True)
    sources = {
        "k10-train": training_teacher_run.resolve() / "silver/silver_labels.jsonl",
        "k10-heldout": heldout_teacher_run.resolve() / "silver/silver_labels.jsonl",
    }
    for product_id, source in sources.items():
        if not source.is_file():
            raise SilverStageError(f"Missing {product_id} teacher silver: {source}")
    derived = {
        "k1-train": destination / "k1-train.jsonl",
        "k1-heldout": destination / "k1-heldout.jsonl",
    }
    for source_id, output_id in (
        ("k10-train", "k1-train"),
        ("k10-heldout", "k1-heldout"),
    ):
        derive_silver_shard(
            sources[source_id],
            derived[output_id],
            k_out=1,
            start_index=0,
            force=True,
        )
    products_by_id = {str(product["id"]): product for product in silver["products"]}
    pointwise_products = {
        product_id: product
        for product_id, product in products_by_id.items()
        if str(product.get("labels")) == "pointwise-single"
    }
    if pointwise_products:
        if pointwise_training_teacher_run is None or pointwise_heldout_teacher_run is None:
            raise SilverStageError("Pointwise products require pointwise training and heldout teacher runs")
        pointwise_runs = {
            "pointwise-train": pointwise_training_teacher_run,
            "pointwise-heldout": pointwise_heldout_teacher_run,
        }
        for product_id, run_dir in pointwise_runs.items():
            source = run_dir.resolve() / "silver/silver_labels.jsonl"
            if not source.is_file():
                raise SilverStageError(f"Missing {product_id} teacher silver: {source}")
            target = destination / str(pointwise_products[product_id]["filename"])
            shutil.copyfile(source, target)
            sources[product_id] = target
    products = {
        product_id: {"path": str(path), "sha256": sha256_file(path)}
        for product_id, path in {**sources, **derived}.items()
    }
    manifest = {
        "schema_version": 1,
        "task": pipeline["task"]["id"],
        "stage": "silver",
        "teacher_runs": {
            "training": str(training_teacher_run),
            "heldout": str(heldout_teacher_run),
            **(
                {
                    "pointwise_training": str(pointwise_training_teacher_run),
                    "pointwise_heldout": str(pointwise_heldout_teacher_run),
                }
                if pointwise_products
                else {}
            ),
        },
        "products": products,
        "consumers": silver["consumers"],
    }
    write_silver_receipts(
        pipeline,
        destination,
        manifest,
        products,
        title=f"{pipeline['task']['id']} silver",
    )
    return manifest


plan_qa_silver = plan_cohort_silver
materialize_qa_silver = materialize_cohort_silver
validate_qa_silver = validate_cohort_silver
collect_qa_silver = collect_cohort_silver
plan_response_silver = plan_cohort_silver
materialize_response_silver = materialize_cohort_silver
validate_response_silver = validate_cohort_silver
collect_response_silver = collect_cohort_silver
