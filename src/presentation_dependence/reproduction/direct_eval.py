"""Plan, materialize, and validate reproduction direct-evaluation jobs."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping


from presentation_dependence.eval.dataset_catalog import load_dataset_population
from presentation_dependence.utils.config import load_yaml_mapping
from presentation_dependence.utils.sweeps import write_config, write_sweep

from .common import prepare_generated_directory
from .errors import DirectEvalStageError


def _direct(pipeline: Mapping[str, Any]) -> Mapping[str, Any]:
    value = pipeline.get("direct_eval")
    if not isinstance(value, Mapping):
        raise DirectEvalStageError("direct_eval must be a mapping")
    return value


def _population(pipeline: Mapping[str, Any], project_root: Path) -> list[dict[str, Any]]:
    direct = _direct(pipeline)
    return load_dataset_population(
        str(direct["population"]),
        configs_root=project_root / "configs",
    )


def _checkpoint_rows(pipeline: Mapping[str, Any], project_root: Path) -> list[dict[str, Any]]:
    path = project_root / str(_direct(pipeline)["checkpoint_catalog"])
    if not path.is_file():
        raise DirectEvalStageError(f"Missing checkpoint catalog: {path}")
    catalog = json.loads(path.read_text(encoding="utf-8"))
    rows = catalog.get("checkpoints")
    if not isinstance(rows, list) or not rows:
        raise DirectEvalStageError("Checkpoint catalog has no rows")
    catalog_missing = catalog.get("missing") or []
    if catalog_missing:
        raise DirectEvalStageError(
            f"Checkpoint catalog is incomplete: {len(catalog_missing)} missing rows; first={catalog_missing[0]}"
        )
    expected = {
        (str(variant["id"]), int(seed))
        for variant in pipeline["training"]["variants"]
        for seed in pipeline["training"]["seeds"]
    }
    actual_list = [(str(row["variant"]), int(row["training_seed"])) for row in rows]
    actual = set(actual_list)
    if len(actual) != len(actual_list):
        raise DirectEvalStageError("Checkpoint catalog contains duplicate variant/seed rows")
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise DirectEvalStageError(f"Checkpoint catalog matrix mismatch: missing={missing[:5]} extra={extra[:5]}")
    return rows


def _jobs(
    pipeline: Mapping[str, Any],
    project_root: Path,
    *,
    include_controls: bool = False,
) -> list[dict[str, Any]]:
    datasets = _population(pipeline, project_root)
    direct = _direct(pipeline)
    checkpoints = _checkpoint_rows(pipeline, project_root)
    jobs = [
        {
            "variant": row["variant"],
            "training_seed": int(row["training_seed"]),
            "checkpoint_uri": row["checkpoint_uri"],
            "dataset": dataset["id"],
            "kind": "trained",
        }
        for row in checkpoints
        for dataset in datasets
    ]
    jobs.extend(
        {
            "variant": reference["id"],
            "training_seed": None,
            "dataset": dataset["id"],
            "kind": reference["kind"],
            "model": reference.get("model"),
            "order_invariant": bool(reference.get("order_invariant")),
        }
        for reference in direct["variants"]["references"]
        for dataset in datasets
    )
    if include_controls:
        jobs.extend(
            _control_job(
                control,
                dataset["id"],
                _control_checkpoint_rows(control, project_root, checkpoints),
            )
            for control in direct["optional_controls"]["variants"]
            if control.get("runnable")
            for dataset in datasets
        )
    return jobs


def _control_job(
    control: Mapping[str, Any],
    dataset: str,
    checkpoints: list[dict[str, Any]],
) -> dict[str, Any]:
    source_variant = control.get("source_variant")
    seed = control.get("training_seed")
    job = {
        "variant": control["id"],
        "training_seed": int(seed) if seed is not None else None,
        "dataset": dataset,
        "kind": "control",
        "control": control["id"],
        "control_spec": dict(control),
        "order_invariant": bool(control.get("order_invariant")),
    }
    if source_variant and source_variant != "off-shelf":
        source = next(
            (row for row in checkpoints if row["variant"] == source_variant and int(row["training_seed"]) == int(seed)),
            None,
        )
        if source is None:
            raise DirectEvalStageError(
                f"Control {control['id']} requires missing checkpoint {source_variant} seed {seed}"
            )
        job["kind"] = "trained-control"
        job["checkpoint_uri"] = source["checkpoint_uri"]
        job["source_variant"] = source_variant
    return job


def _control_checkpoint_rows(
    control: Mapping[str, Any],
    project_root: Path,
    primary: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    catalog_path = control.get("checkpoint_catalog")
    if not catalog_path:
        return primary
    path = project_root / str(catalog_path)
    if not path.is_file():
        raise DirectEvalStageError(f"Control {control['id']} requires checkpoint catalog: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("checkpoints")
    if not isinstance(rows, list):
        raise DirectEvalStageError(f"Control checkpoint catalog has no rows: {path}")
    return rows


def _config_id(pipeline: Mapping[str, Any], job: Mapping[str, Any]) -> str:
    seed = f"--seed{job['training_seed']}" if job.get("training_seed") is not None else ""
    return f"{pipeline['task']['id']}-direct-eval--{job['variant']}{seed}--{job['dataset']}"


def plan_direct_eval(
    pipeline: Mapping[str, Any],
    project_root: Path,
    *,
    include_controls: bool = False,
) -> dict[str, Any]:
    """Return the checkpoint, dataset, and reference-variant cross-product."""
    direct = _direct(pipeline)
    jobs = [
        {**job, "id": _config_id(pipeline, job)}
        for job in _jobs(
            pipeline,
            project_root,
            include_controls=include_controls,
        )
    ]
    return {
        "task": pipeline["task"]["id"],
        "stage": "direct-eval",
        "datasets": len(_population(pipeline, project_root)),
        "jobs": jobs,
        "job_count": len(jobs),
        "trained_checkpoint_rows": len(_checkpoint_rows(pipeline, project_root)),
        "reference_variants": [row["id"] for row in direct["variants"]["references"]],
        "optional_controls": (direct["optional_controls"]["variants"] if include_controls else []),
        "optional_controls_enabled": include_controls,
    }


def _dataset_map(pipeline: Mapping[str, Any], project_root: Path) -> dict[str, dict[str, Any]]:
    return {str(dataset["id"]): dataset for dataset in _population(pipeline, project_root)}


def _qwen_reranker(
    pipeline: Mapping[str, Any],
    dataset: Mapping[str, Any],
) -> dict[str, Any]:
    model = pipeline["_shared"]["models"]["qwen3-4b-nonthinking"]
    max_length = int(dataset.get("max_model_len", 4096))
    reranker = {
        "class": model["class"],
        "model_name": model["model_name"],
        "model_size": model["model_size"],
        "model_release": model["model_release"],
        "revision": model.get("revision"),
        "device": "auto",
        "dtype": model["dtype"],
        "scoring_mode": "setwise_grade_prompt",
        "continuous_readout": True,
        "inference_engine": model["inference_engine"],
        "chat_template_kwargs": model["chat_template_kwargs"],
        "vllm_settings": {
            "max_model_len": max_length,
            "gpu_memory_utilization": 0.85,
            "enable_prefix_caching": True,
            "tensor_parallel_size": 1,
            "top_logprobs": 50,
        },
        "instruction": dataset["instruction"],
        "max_length": max_length,
        "docs_per_score_forward": int(_direct(pipeline)["protocol"]["serving_width"]),
        "batch_size": int(_direct(pipeline)["protocol"]["serving_width"]),
        "max_doc_chars": int(dataset["max_doc_chars"]),
    }
    if dataset.get("grade_rubric"):
        reranker["grade_rubric_id"] = dataset["grade_rubric"]
    return reranker


def _external_reranker(
    pipeline: Mapping[str, Any],
    dataset: Mapping[str, Any],
    model: Mapping[str, Any],
) -> dict[str, Any]:
    reranker = {
        **dict(model),
        "device": "auto",
        "max_doc_chars": int(dataset["max_doc_chars"]),
        "docs_per_score_forward": int(_direct(pipeline)["protocol"]["serving_width"]),
    }
    if reranker.get("class") == "CapCalReranker":
        base = dict(reranker["base_reranker"])
        base["instruction"] = dataset["instruction"]
        base["max_doc_chars"] = int(dataset["max_doc_chars"])
        reranker["base_reranker"] = base
        reranker["instruction"] = dataset["instruction"]
    return reranker


def _data_config(pipeline: Mapping[str, Any], dataset: Mapping[str, Any]) -> dict[str, Any]:
    data = {
        "dataloader_class": dataset["loader"],
        "run_path": dataset["run_path"],
        "k_input": int(
            dataset.get(
                "pool_depth",
                _direct(pipeline)["protocol"].get("pool_depth", 100),
            )
        ),
    }
    for key in ("topics", "topics_tsv", "index"):
        if dataset.get(key) is not None:
            data[key] = dataset[key]
    return data


def _safe_channel(config_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", config_id).strip("-")[:63]


def _job_reranker(
    pipeline: Mapping[str, Any],
    dataset: Mapping[str, Any],
    job: Mapping[str, Any],
) -> dict[str, Any]:
    reranker = (
        _external_reranker(pipeline, dataset, job["model"])
        if job["kind"] == "external"
        else _qwen_reranker(pipeline, dataset)
    )
    if job.get("control") in {"pointwise-serving", "true-pointwise-b1"}:
        reranker["docs_per_score_forward"] = 1
        reranker["batch_size"] = 1
    elif job.get("control") == "round-robin-chunks":
        reranker["chunk_assignment"] = "interleaved"
    elif job.get("control") == "reverse-order":
        reranker["presentation_order"] = "reverse"
        reranker.pop("chunk_assignment", None)
    return reranker


def _job_execution(
    direct: Mapping[str, Any],
    dataset: Mapping[str, Any],
    job: Mapping[str, Any],
    reranker: dict[str, Any],
) -> dict[str, Any]:
    execution_block = {key: value for key, value in direct["execution"].items() if key != "max_parallel"}
    if dataset.get("fixture_channel"):
        execution_block["fixture_channel"] = dataset["fixture_channel"]
    if job["kind"] in {"trained", "trained-control"}:
        # `lora_path` is loaded verbatim, so it has to be the checkpoint the
        # training stage wrote. `extra_input_channels` keeps the channel name
        # for stagers that mount the adapter somewhere else.
        channel = _safe_channel(f"lora-{job['variant']}-seed{job['training_seed']}")
        reranker["lora_path"] = job["checkpoint_uri"]
        reranker["max_lora_rank"] = 16
        execution_block["extra_input_channels"] = {channel: job["checkpoint_uri"]}
    return execution_block


def _robustness_config(direct: Mapping[str, Any]) -> dict[str, Any]:
    protocol = direct["protocol"]
    robustness = {
        "K": len(protocol["presentation_seeds"]),
        "seeds": list(protocol["presentation_seeds"]),
        "perturbations": list(protocol["perturbations"]),
        "k_cutoff_for_ndcg": 10,
        "derive_self_consistency": bool(protocol["derive_self_consistency"]),
    }
    if protocol.get("sc_k_subsets"):
        robustness["sc_k_subsets"] = list(protocol["sc_k_subsets"])
    if protocol.get("sc_measures"):
        robustness["sc_measures"] = list(protocol["sc_measures"])
    return robustness


def build_experiment_config(
    pipeline: Mapping[str, Any],
    dataset: Mapping[str, Any],
    job: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one runnable direct-evaluation config from canonical inputs."""
    direct = _direct(pipeline)
    config_id = _config_id(pipeline, job)
    reranker = _job_reranker(pipeline, dataset, job)
    execution_block = _job_execution(direct, dataset, job, reranker)
    config = {
        "id": config_id,
        "reranker": reranker,
        "data": _data_config(pipeline, dataset),
        "eval": {
            "qrels_path": dataset["qrels_path"],
            "measures": list(direct["evaluation"]["measures"]),
        },
        "robustness": _robustness_config(direct),
        "execution": execution_block,
        "logging": {"level": "INFO"},
    }
    if str(pipeline["task"]["id"]) in {"passage-reranking", "multi-document-qa"}:
        config["robustness"]["beta_gamma"] = {
            "enabled": True,
            "recipe": str(job["variant"]),
            "checkpoint": str(job.get("checkpoint_uri") or config_id),
        }
    if dataset.get("qids_to_run_path"):
        config["qids_to_run_path"] = dataset["qids_to_run_path"]
    if dataset.get("grade_rubric"):
        config["eval"]["strict_qrels_filter"] = True
    if job.get("control") == "pointwise-serving" or job.get("order_invariant"):
        config.pop("robustness")
    if job.get("control") == "matched-variance":
        control = job["control_spec"]
        config["matched_variance_control"] = {
            "seeds": list(control["presentation_seeds"]),
            "request_batch_size": int(control["request_batch_size"]),
            "pool_size": int(
                dataset.get(
                    "pool_depth",
                    direct["protocol"].get("pool_depth", 10),
                )
            ),
            "k_cutoff_for_ndcg": 10,
            "qid_shard_count": 1,
            "qid_shard_index": 0,
            "required_unique_positions": 2,
            "required_unique_compositions": 2,
        }
        config.pop("robustness", None)
    return config


def materialize_direct_eval(
    pipeline: Mapping[str, Any],
    project_root: Path,
    *,
    build_root: Path | None = None,
    include_controls: bool = False,
) -> dict[str, Any]:
    """Write direct-evaluation configs and their launch sweep."""
    direct = _direct(pipeline)
    root = build_root or project_root / str(direct["outputs"]["root"])
    configs_dir = prepare_generated_directory(root / str(direct["outputs"]["configs_dir"]))
    datasets = _dataset_map(pipeline, project_root)
    records: list[dict[str, Any]] = []
    for job in _jobs(
        pipeline,
        project_root,
        include_controls=include_controls,
    ):
        config = build_experiment_config(pipeline, datasets[job["dataset"]], job)
        path = configs_dir / f"{config['id']}.yaml"
        write_config(path, config)
        records.append({**job, "id": config["id"], "config": str(path)})
    sweep = {
        "id": f"{pipeline['task']['id']}-direct-eval-primary",
        "description": f"{pipeline['task']['id']} direct-evaluation sweep",
        "execution": {"max_parallel": direct["execution"]["max_parallel"]},
        "jobs": [{"exp_id": row["config"]} for row in records],
    }
    sweep_path = write_sweep(root / str(direct["outputs"]["sweep"]), sweep)
    result = {
        **plan_direct_eval(
            pipeline,
            project_root,
            include_controls=include_controls,
        ),
        "configs": records,
        "sweep": str(sweep_path),
    }
    (root / "materialization.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def validate_direct_eval(
    pipeline: Mapping[str, Any],
    project_root: Path,
    *,
    build_root: Path | None = None,
    check_data: bool = True,
    include_controls: bool = False,
) -> dict[str, Any]:
    """Validate config count, identities, checkpoint channels, and data files."""
    direct = _direct(pipeline)
    root = build_root or project_root / str(direct["outputs"]["root"])
    configs = sorted((root / str(direct["outputs"]["configs_dir"])).glob("*.yaml"))
    expected = len(
        _jobs(
            pipeline,
            project_root,
            include_controls=include_controls,
        )
    )
    if len(configs) != expected:
        raise DirectEvalStageError(f"Expected {expected} direct-eval configs, found {len(configs)}")
    ids: set[str] = set()
    missing_data: list[str] = []
    for path in configs:
        config = load_yaml_mapping(path)
        config_id = str(config["id"])
        if config_id in ids:
            raise DirectEvalStageError(f"Duplicate config ID: {config_id}")
        ids.add(config_id)
        if "AE" in config_id or "A8c" in config_id:
            raise DirectEvalStageError(f"Legacy experiment token in generated config ID: {config_id}")
        if check_data:
            for value in (
                config["data"]["run_path"],
                config["eval"]["qrels_path"],
                config["data"].get("topics_tsv"),
            ):
                if value and not (project_root / str(value)).is_file():
                    missing_data.append(str(value))
    if missing_data:
        raise DirectEvalStageError("Missing direct-eval data files: " + ", ".join(sorted(set(missing_data))))
    sweep_path = root / str(direct["outputs"]["sweep"])
    sweep = load_yaml_mapping(sweep_path)
    if len(sweep["jobs"]) != expected:
        raise DirectEvalStageError("Direct-eval sweep size drifted")
    return {
        "task": pipeline["task"]["id"],
        "stage": "direct-eval",
        "configs": len(configs),
        "unique_ids": len(ids),
        "data_checked": check_data,
        "sweep": str(sweep_path),
    }
