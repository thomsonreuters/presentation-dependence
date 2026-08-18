"""Materialize semantic study configs from canonical declarations."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Mapping


from presentation_dependence.eval.dataset_catalog import load_dataset_population
from presentation_dependence.utils.config import load_yaml_mapping
from presentation_dependence.utils.sweeps import write_config, write_sweep

from .common import sha256_file
from .direct_eval import build_experiment_config
from .errors import ReproductionError
from .pipelines import load_pipeline


STUDY_FILES = {
    "mechanism-boundary": "configs/studies/mechanism-boundary.yaml",
    "fixed-weight-width": "configs/studies/fixed-weight-width.yaml",
    "setup-readout-silver": "configs/studies/setup-readout-silver.yaml",
    "first-stage-transfer": "configs/studies/first-stage-transfer.yaml",
}


def load_study(project_root: Path, study_id: str) -> dict[str, Any]:
    """Load one semantic study declaration by ID."""
    try:
        relative = STUDY_FILES[study_id]
    except KeyError as exc:
        raise ReproductionError(f"Unknown study: {study_id}") from exc
    study = load_yaml_mapping(project_root / relative)
    if study.get("id") != study_id:
        raise ReproductionError(f"Study ID mismatch in {relative}")
    study["_path"] = relative
    return study


def _pipeline(project_root: Path, task: str) -> dict[str, Any]:
    return load_pipeline(task, project_root)


def _datasets(
    project_root: Path,
    pipeline: Mapping[str, Any],
    population: str | None = None,
) -> list[dict[str, Any]]:
    population_id = population or str(pipeline["direct_eval"]["population"])
    return load_dataset_population(population_id, configs_root=project_root / "configs")


def _checkpoint_rows(
    project_root: Path,
    pipeline: Mapping[str, Any],
    checkpoint_catalog: str | None = None,
) -> list[dict[str, Any]]:
    path = project_root / str(checkpoint_catalog or pipeline["direct_eval"]["checkpoint_catalog"])
    if not path.is_file():
        raise ReproductionError(f"Study requires checkpoint catalog: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("checkpoints")
    if not isinstance(rows, list):
        raise ReproductionError(f"Invalid checkpoint catalog: {path}")
    return [dict(row) for row in rows]


def _checkpoint(
    project_root: Path,
    pipeline: Mapping[str, Any],
    variant: str,
    seed: int,
    checkpoint_catalog: str | None = None,
) -> dict[str, Any]:
    def matches(row: Mapping[str, Any]) -> bool:
        row_seed = row.get("training_seed")
        return row.get("variant") == variant and row_seed is not None and int(row_seed) == seed

    row = next(
        (row for row in _checkpoint_rows(project_root, pipeline, checkpoint_catalog) if matches(row)),
        None,
    )
    if row is None:
        raise ReproductionError(f"Missing checkpoint for {pipeline['task']['id']} {variant} seed {seed}")
    return row


def _source_config(
    project_root: Path,
    *,
    task: str,
    dataset: Mapping[str, Any],
    variant: str,
    seed: int | None,
    checkpoint_variant: str | None = None,
    checkpoint_catalog: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    pipeline = _pipeline(project_root, task)
    checkpoint = (
        _checkpoint(
            project_root,
            pipeline,
            checkpoint_variant or variant,
            int(seed),
            checkpoint_catalog,
        )
        if variant != "off-shelf" and seed is not None
        else None
    )
    job = {
        "variant": variant,
        "training_seed": seed,
        "dataset": dataset["id"],
        "kind": "trained" if checkpoint else "off-shelf",
    }
    if checkpoint:
        job["checkpoint_uri"] = checkpoint["checkpoint_uri"]
    return build_experiment_config(pipeline, dataset, job), checkpoint


def _condition(study: Mapping[str, Any], condition_id: str) -> Mapping[str, Any]:
    value = (study.get("conditions") or {}).get(condition_id)
    if not isinstance(value, Mapping):
        raise ReproductionError(f"Study {study['id']} has no condition {condition_id}")
    return value


def _set_identity(
    config: dict[str, Any],
    *,
    config_id: str,
    study: Mapping[str, Any],
    condition_id: str,
) -> None:
    del study, condition_id
    config["id"] = config_id


def _set_width(config: dict[str, Any], width: int) -> None:
    config["reranker"]["docs_per_score_forward"] = width
    config["reranker"]["batch_size"] = width
    if width == 1:
        config.pop("robustness", None)


def _set_placeholder(config: dict[str, Any], placeholder: str, assignment: str) -> None:
    config["reranker"]["grade_skeleton_dummy"] = placeholder
    config["reranker"]["chunk_assignment"] = assignment


def _set_random_protocol(
    config: dict[str, Any],
    *,
    seeds: list[int],
    perturbations: list[str],
) -> None:
    robustness = config.setdefault("robustness", {})
    robustness["K"] = len(seeds)
    robustness["seeds"] = seeds
    robustness["perturbations"] = perturbations
    robustness["derive_self_consistency"] = True


def _set_matched_variance(
    config: dict[str, Any],
    *,
    seeds: list[int],
    request_batch_size: int,
    k_cutoff: int,
) -> None:
    _set_width(config, 1)
    config["matched_variance_control"] = {
        "seeds": seeds,
        "request_batch_size": request_batch_size,
        "pool_size": int(config["data"]["k_input"]),
        "k_cutoff_for_ndcg": k_cutoff,
        "qid_shard_count": 1,
        "qid_shard_index": 0,
        "required_unique_positions": 2,
        "required_unique_compositions": 2,
    }


def _set_context_decomposition(
    config: dict[str, Any],
    *,
    condition: Mapping[str, Any],
    dataset_id: str,
    donor_seed_namespace: str,
) -> None:
    """Configure the paired four-arm expected-grade decomposition."""
    _set_width(config, int(condition["serving_width"]))
    config.pop("robustness", None)
    config.setdefault("eval", {})["measures"] = []
    bootstrap = condition["bootstrap"]
    config["context_decomposition"] = {
        "pool_size": int(condition["pool_depth"]),
        "variable_pool_size": bool(condition["variable_pool_size"]),
        "width": int(condition["serving_width"]),
        "dataset_key": dataset_id,
        "donor_seed_namespace": donor_seed_namespace,
        "qid_shard_count": 1,
        "qid_shard_index": 0,
        "screen_only": False,
        "seeds": [int(seed) for seed in condition["presentation_seeds"]],
        "k_cutoff_for_ndcg": int(condition["k_cutoff_for_ndcg"]),
        "request_batch_size": int(condition["request_batch_size"]),
        "bootstrap_samples": int(bootstrap["samples"]),
        "bootstrap_seed": int(bootstrap["seed"]),
    }


def _context_decomposition_shard_jobs(config_id: str, shard_count: int) -> list[dict[str, Any]]:
    """Build deterministic whole-query shard jobs for one dataset config."""
    if shard_count < 1:
        raise ReproductionError(f"Context-decomposition shard count must be positive: {config_id}")
    return [
        {
            "exp_id": f"configs/experiments/{config_id}.yaml",
            "overrides": {
                "id": f"{config_id}--shard{index}of{shard_count}",
                "context_decomposition": {
                    "qid_shard_count": shard_count,
                    "qid_shard_index": index,
                },
            },
        }
        for index in range(shard_count)
    ]


def _row(
    config: Mapping[str, Any],
    *,
    task: str,
    dataset: Mapping[str, Any],
    variant: str,
    seed: int | None,
    checkpoint: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "config": config["id"],
        "task": task,
        "dataset": dataset["id"],
        "variant": variant,
        "training_seed": seed,
        "checkpoint_uri": (checkpoint.get("checkpoint_uri") if checkpoint else None),
        "checkpoint_trial": checkpoint.get("trial") if checkpoint else None,
        "checkpoint_step": (checkpoint.get("selected_step") if checkpoint else None),
        "checkpoint_ref": checkpoint.get("checkpoint_ref") if checkpoint else None,
        "checkpoint_artifact_sha256": (checkpoint.get("artifact_sha256") if checkpoint else None),
        "data_path": (config.get("data") or {}).get("run_path"),
        "qrels_path": (config.get("eval") or {}).get("qrels_path"),
        "serving_width": (config.get("reranker") or {}).get("docs_per_score_forward"),
    }


def _write(
    project_root: Path,
    study: Mapping[str, Any],
    condition_id: str,
    configs: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    *,
    sweep_jobs: list[dict[str, Any]] | None = None,
    sweep_execution: Mapping[str, Any] | None = None,
    provenance_extra: Mapping[str, Any] | None = None,
    run_manifest_rows: list[dict[str, Any]] | None = None,
    sweep_id: str | None = None,
    sweep_description: str | None = None,
) -> dict[str, Any]:
    outputs = study["outputs"]
    config_root = project_root / str(outputs["config_root"])
    sweep_root = project_root / str(outputs["sweep_root"])
    config_root.mkdir(parents=True, exist_ok=True)
    sweep_root.mkdir(parents=True, exist_ok=True)
    for config in configs:
        path = config_root / f"{config['id']}.yaml"
        write_config(path, config)
    sweep_id = sweep_id or f"{condition_id}--canonical"
    sweep = {
        "id": sweep_id,
        "description": sweep_description or f"Canonical materialization of {study['id']} / {condition_id}",
        "jobs": sweep_jobs
        if sweep_jobs is not None
        else [{"exp_id": f"configs/experiments/{config['id']}.yaml"} for config in configs],
    }
    if sweep_execution:
        sweep["execution"] = dict(sweep_execution)
    sweep_path = write_sweep(sweep_root / f"{sweep_id}.yaml", sweep)
    provenance = {
        "schema_version": 1,
        "study": study["id"],
        "condition": condition_id,
        "source": study["_path"],
        "configs": rows,
        "sweep": str(sweep_path.relative_to(project_root)),
    }
    if provenance_extra:
        provenance.update(dict(provenance_extra))
    result_root = project_root / "build/reproduction/studies" / str(study["id"]) / condition_id
    result_root.mkdir(parents=True, exist_ok=True)
    provenance_path = result_root / "materialization.json"
    provenance_path.write_text(
        json.dumps(provenance, indent=2) + "\n",
        encoding="utf-8",
    )
    run_manifest_path = result_root / "run_manifest.json"
    run_manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "study": study["id"],
                "condition": condition_id,
                "runs": run_manifest_rows
                if run_manifest_rows is not None
                else [
                    {
                        "config_id": config["id"],
                        "run_dir": None,
                    }
                    for config in configs
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "study": study["id"],
        "condition": condition_id,
        "config_count": len(configs),
        "configs": [str(config["id"]) for config in configs],
        "sweep_job_count": len(sweep["jobs"]),
        "sweep": str(sweep_path.relative_to(project_root)),
        "provenance": str(provenance_path.relative_to(project_root)),
        "run_manifest": str(run_manifest_path.relative_to(project_root)),
    }


def _first_stage_datasets(
    project_root: Path,
    study: Mapping[str, Any],
    condition: Mapping[str, Any],
    pipeline: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Resolve one explicitly ordered first-stage dataset population."""
    catalog = {
        str(dataset["id"]): dict(dataset) for dataset in _datasets(project_root, pipeline, str(study["population"]))
    }
    catalog.update(
        {str(dataset_id): dict(dataset) for dataset_id, dataset in (study.get("additional_datasets") or {}).items()}
    )
    dataset_ids = [str(dataset_id) for dataset_id in condition["datasets"]]
    missing = [dataset_id for dataset_id in dataset_ids if dataset_id not in catalog]
    if missing:
        raise ReproductionError(f"First-stage study references unknown datasets: {missing}")
    return [catalog[dataset_id] for dataset_id in dataset_ids]


def _first_stage_dataset(
    study: Mapping[str, Any],
    dataset: Mapping[str, Any],
    first_stage: str,
) -> dict[str, Any]:
    """Swap only the candidate source for a dense or sparse first stage."""
    result = dict(dataset)
    if first_stage == "bm25":
        return result
    tokens = study["first_stage_tokens"]
    if first_stage not in tokens:
        raise ReproductionError(f"Unknown first-stage token: {first_stage}")
    parent = Path(str(dataset["run_path"])).parent
    dataset_id = str(dataset["id"])
    base_channel = str(dataset.get("fixture_channel") or parent.name)
    result.update(
        {
            "loader": "FixtureLoader",
            "run_path": str(parent / f"fixture.{tokens[first_stage]}.{dataset_id}.jsonl"),
            "fixture_channel": f"{base_channel}-{first_stage}",
        }
    )
    return result


def _first_stage_config_id(
    prefix: str,
    scorer_id: str,
    scorer: Mapping[str, Any],
    dataset_id: str,
    first_stage: str,
) -> str:
    kind = str(scorer["kind"])
    stage = "" if first_stage == "bm25" else f"-{first_stage}"
    return f"{prefix}-{scorer_id}-{kind}-{dataset_id}{stage}-vllm-psi"


def _first_stage_checkpoint_refs(study: Mapping[str, Any]) -> set[str]:
    """Validate first-stage scorer declarations and return their adapter refs."""
    scorers = study.get("scorers")
    if not isinstance(scorers, Mapping):
        raise ReproductionError(f"Study {study['id']} has no scorers mapping")
    required_refs: set[str] = set()
    for scorer_id, scorer in scorers.items():
        if not isinstance(scorer, Mapping):
            raise ReproductionError(f"First-stage scorer {scorer_id!r} must be a mapping")
        if scorer.get("kind") != "lora":
            continue
        checkpoint_ref = str(scorer.get("checkpoint_ref") or "")
        if not checkpoint_ref:
            raise ReproductionError(f"First-stage LoRA scorer {scorer_id!r} has no checkpoint_ref")
        if scorer.get("training_seed") is None:
            raise ReproductionError(f"First-stage LoRA scorer {scorer_id!r} has no training_seed")
        if checkpoint_ref in required_refs:
            raise ReproductionError(f"Duplicate first-stage checkpoint_ref {checkpoint_ref!r}")
        required_refs.add(checkpoint_ref)
    return required_refs


def _resolve_first_stage_adapter(
    project_root: Path,
    checkpoint_ref: str,
    raw_path: Any,
) -> dict[str, Any]:
    """Validate one bound PEFT adapter and return hash-bearing provenance."""
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ReproductionError(f"Checkpoint binding {checkpoint_ref!r} must be a non-empty path string")
    declared = Path(raw_path.strip()).expanduser()
    adapter_dir = declared if declared.is_absolute() else project_root / declared
    if not adapter_dir.is_dir():
        raise ReproductionError(f"Checkpoint binding {checkpoint_ref!r} is not a directory: {adapter_dir}")
    adapter_config = adapter_dir / "adapter_config.json"
    weight_candidates = (
        adapter_dir / "adapter_model.safetensors",
        adapter_dir / "adapter_model.bin",
    )
    missing_files = [] if adapter_config.is_file() else [adapter_config.name]
    weights = [path for path in weight_candidates if path.is_file()]
    if not weights:
        missing_files.append("adapter_model.safetensors|adapter_model.bin")
    if missing_files:
        raise ReproductionError(
            f"Checkpoint binding {checkpoint_ref!r} is not a complete PEFT adapter under "
            f"{adapter_dir}; missing {missing_files}"
        )
    hashed_files = [adapter_config, *weights]
    return {
        "checkpoint_ref": checkpoint_ref,
        "checkpoint_uri": str(declared),
        "artifact_sha256": {path.name: sha256_file(path) for path in hashed_files},
    }


def _load_first_stage_checkpoint_bindings(
    project_root: Path,
    study: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Resolve and validate every local adapter used by first-stage transfer."""
    declared_path = study.get("checkpoint_bindings")
    if not declared_path:
        raise ReproductionError(f"Study {study['id']} has no checkpoint_bindings path")
    bindings_path = Path(str(declared_path))
    if not bindings_path.is_absolute():
        bindings_path = project_root / bindings_path
    if not bindings_path.is_file():
        raise ReproductionError(
            f"First-stage checkpoint bindings are missing: {bindings_path}. "
            "Copy configs/studies/first-stage-transfer-checkpoints.example.yaml "
            f"to {declared_path} and replace each value with a selected local adapter directory."
        )

    payload = load_yaml_mapping(bindings_path)
    if payload.get("schema_version") != 1:
        raise ReproductionError(f"Unsupported first-stage checkpoint binding schema: {bindings_path}")
    if payload.get("study") != study.get("id"):
        raise ReproductionError(
            f"First-stage checkpoint bindings identify study={payload.get('study')!r}, expected {study.get('id')!r}"
        )
    raw_bindings = payload.get("checkpoints")
    if not isinstance(raw_bindings, Mapping):
        raise ReproductionError(f"First-stage checkpoint bindings have no checkpoints mapping: {bindings_path}")

    required_refs = _first_stage_checkpoint_refs(study)
    available_refs = set(map(str, raw_bindings))
    missing = sorted(required_refs - available_refs)
    extra = sorted(available_refs - required_refs)
    if missing or extra:
        raise ReproductionError(
            f"First-stage checkpoint binding keys do not match the study: missing={missing}, extra={extra}"
        )

    return {
        checkpoint_ref: _resolve_first_stage_adapter(project_root, checkpoint_ref, raw_bindings[checkpoint_ref])
        for checkpoint_ref in sorted(required_refs)
    }


def _first_stage_config(
    project_root: Path,
    study: Mapping[str, Any],
    condition_id: str,
    condition: Mapping[str, Any],
    pipeline: Mapping[str, Any],
    scorer_id: str,
    scorer: Mapping[str, Any],
    dataset: Mapping[str, Any],
    first_stage: str,
    checkpoint_bindings: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Build one first-stage transfer config."""
    dataset = _first_stage_dataset(study, dataset, first_stage)
    config_id = _first_stage_config_id(
        condition_id,
        scorer_id,
        scorer,
        str(dataset["id"]),
        first_stage,
    )
    model_len = 4096
    reranker = {
        "class": scorer["class"],
        "model_name": scorer["model_name"],
        "model_size": scorer["model_size"],
        "model_release": scorer["model_release"],
        "device": "auto",
        "dtype": "bfloat16",
        "scoring_mode": "setwise_grade_prompt",
        "score_output": "probability",
        "continuous_readout": True,
        "inference_engine": str(scorer.get("inference_engine", "vllm")),
        "vllm_settings": {
            "max_model_len": model_len,
            "gpu_memory_utilization": 0.85,
            "enable_prefix_caching": True,
            "tensor_parallel_size": 1,
            "top_logprobs": 50,
        },
        "instruction": dataset["instruction"],
        "max_length": model_len,
        "docs_per_score_forward": 20,
        "batch_size": 20,
        "max_doc_chars": int(dataset["max_doc_chars"]),
        "setwise_missing_grade": 0,
    }
    for key in ("revision", "chat_template_kwargs"):
        if scorer.get(key) is not None:
            reranker[key] = scorer[key]
    execution_block = {"environment": {"VLLM_WORKER_MULTIPROC_METHOD": "spawn"}}
    checkpoint = None
    if scorer["kind"] == "lora":
        checkpoint_ref = str(scorer["checkpoint_ref"])
        checkpoint = dict(checkpoint_bindings[checkpoint_ref])
        checkpoint_uri = str(checkpoint["checkpoint_uri"])
        # See the note in direct_eval: `lora_path` is loaded verbatim, so it
        # points at the checkpoint the training stage wrote.
        channel = f"lora-{scorer_id}"
        reranker["lora_path"] = checkpoint_uri
        reranker["max_lora_rank"] = 16
        execution_block["extra_input_channels"] = {channel: checkpoint_uri}
    if dataset.get("fixture_channel"):
        execution_block["fixture_channel"] = dataset["fixture_channel"]
    data = {
        "dataloader_class": dataset["loader"],
        "run_path": dataset["run_path"],
        "k_input": 100,
    }
    for key in ("topics", "topics_tsv", "index"):
        if dataset.get(key) is not None:
            data[key] = dataset[key]
    robustness = {
        "K": 10,
        "seeds": list(range(10)),
        "perturbations": ["random_shuffle", "middle_injection"],
        "k_cutoff_for_ndcg": 10,
    }
    if scorer.get("taupsi_cap"):
        capped_qids = f"configs/reproduction/fixtures/taupsi-qids/{dataset['id']}.txt"
        # The cap applies only where a frozen qid list is tracked. Collections
        # without one run tau-PSI over the full query set.
        if (project_root / capped_qids).is_file():
            robustness["qids_to_run_path"] = capped_qids
    config = {
        "id": config_id,
        "reranker": reranker,
        "data": data,
        "eval": {
            "qrels_path": dataset["qrels_path"],
            "measures": ["ndcg_cut_10", "ndcg_cut_5", "map", "recip_rank"],
        },
        "robustness": robustness,
        "execution": execution_block,
        "logging": {"level": "INFO"},
    }
    return config, checkpoint


def _materialize_first_stage_transfer(project_root: Path, condition_id: str) -> dict[str, Any]:
    study = _study_for_condition(project_root, condition_id)
    condition = _condition(study, condition_id)
    task = str(study["task"])
    pipeline = _pipeline(project_root, task)
    datasets = _first_stage_datasets(project_root, study, condition, pipeline)
    checkpoint_bindings = _load_first_stage_checkpoint_bindings(project_root, study)
    configs: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    configs_by_scorer: dict[str, list[dict[str, Any]]] = {}
    for scorer_id, raw_scorer in study["scorers"].items():
        scorer = dict(raw_scorer)
        for dataset in datasets:
            for first_stage in condition["first_stages"]:
                config, checkpoint = _first_stage_config(
                    project_root,
                    study,
                    condition_id,
                    condition,
                    pipeline,
                    str(scorer_id),
                    scorer,
                    dataset,
                    str(first_stage),
                    checkpoint_bindings,
                )
                configs.append(config)
                configs_by_scorer.setdefault(str(scorer_id), []).append(config)
                rows.append(
                    {
                        **_row(
                            config,
                            task=task,
                            dataset=dataset,
                            variant=str(scorer_id),
                            seed=int(scorer["training_seed"]) if checkpoint else None,
                            checkpoint=checkpoint,
                        ),
                        "first_stage": str(first_stage),
                    }
                )
    execution = condition.get("execution") or {}
    if not execution.get("bundle_per_scorer"):
        return _write(project_root, study, condition_id, configs, rows)

    config_root = project_root / str(study["outputs"]["config_root"])
    config_root.mkdir(parents=True, exist_ok=True)
    bundle_ids = []
    bundles = []
    sweep_jobs = []
    for scorer_id, members in configs_by_scorer.items():
        bundle_id = f"first-stage-newsurf-{scorer_id}-bundle0"
        first = members[0]
        member_ids = [config["id"] for config in members]
        bundle = {
            "id": bundle_id,
            "bundle": {
                "member_configs": member_ids,
                "reranker_overrides": {"vllm_settings": {"tensor_parallel_size": 1}},
            },
            "execution": {"environment": {"VLLM_WORKER_MULTIPROC_METHOD": "spawn"}},
        }
        if first["execution"].get("extra_input_channels"):
            bundle["bundle"]["shared_base"] = True
            bundle["execution"]["extra_input_channels"] = dict(first["execution"]["extra_input_channels"])
        write_config(config_root / f"{bundle_id}.yaml", bundle)
        bundle_ids.append(bundle_id)
        bundles.append({"bundle_id": bundle_id, "member_configs": member_ids})
        sweep_jobs.append({"exp_id": bundle_id})
    bundle_by_member = {member_id: bundle["bundle_id"] for bundle in bundles for member_id in bundle["member_configs"]}
    for row in rows:
        row["bundle_id"] = bundle_by_member[str(row["config"])]
    return _write(
        project_root,
        study,
        condition_id,
        configs,
        rows,
        sweep_jobs=sweep_jobs,
        sweep_execution={"max_parallel": int(execution["max_parallel"])},
        provenance_extra={"bundle_configs": bundle_ids, "bundles": bundles},
        run_manifest_rows=[
            {
                "config_id": row["config"],
                "bundle_id": row["bundle_id"],
                "run_dir": None,
            }
            for row in rows
        ],
    )


def materialize_first_stage_panel(project_root: Path) -> dict[str, Any]:
    """Materialize the 13-surface BGE/SPLADE transfer panel."""
    return _materialize_first_stage_transfer(project_root, "first-stage-panel")


def materialize_first_stage_newsurf(project_root: Path) -> dict[str, Any]:
    """Materialize the six-surface BM25 extension and scorer bundles."""
    return _materialize_first_stage_transfer(project_root, "first-stage-newsurf")


def _readout_reranker(
    model: Mapping[str, Any],
    dataset: Mapping[str, Any],
    *,
    placeholder: str,
    serving_width: int,
    chunk_assignment: str,
) -> dict[str, Any]:
    """Build one continuous grade-distribution scorer for the readout grid."""
    metadata_keys = {
        "paper_name",
        "max_model_len",
        "max_model_len_overrides",
        "placeholder_key",
        "instance_type",
    }
    reranker = {key: value for key, value in model.items() if key not in metadata_keys}
    reranker.setdefault("device", "auto")
    model_len = int((model.get("max_model_len_overrides") or {}).get(str(dataset["id"]), model["max_model_len"]))
    reranker["vllm_settings"] = {
        "max_model_len": model_len,
        "gpu_memory_utilization": 0.85,
        "enable_prefix_caching": True,
        "tensor_parallel_size": 1,
        "top_logprobs": 50,
    }
    reranker["instruction"] = dataset["instruction"]
    reranker["max_length"] = model_len
    reranker["docs_per_score_forward"] = serving_width
    reranker["batch_size"] = serving_width
    reranker["max_doc_chars"] = int(dataset["max_doc_chars"])
    reranker[str(model["placeholder_key"])] = placeholder
    model_class = str(model["class"])
    if model_class == "Qwen3InstructGradeReranker":
        reranker["scoring_mode"] = "setwise_grade_prompt"
        reranker["continuous_readout"] = True
        reranker["chunk_assignment"] = chunk_assignment
    elif model_class == "Gemma4GradeReranker":
        reranker["continuous_readout"] = True
        reranker["grade_prefix_mode"] = "cumulative"
    elif model_class == "Granite41GradeReranker":
        reranker["scoring_method"] = "continuous_grade"
        reranker["grade_prefix_mode"] = "cumulative"
    else:
        raise ReproductionError(f"Unsupported complete-grid model class: {model_class}")
    return reranker


def materialize_complete_readout_grid(
    project_root: Path,
) -> dict[str, Any]:
    """Materialize missing K=1 readout and placeholder cells on primary-18."""
    condition_id = "complete-readout-grid"
    study = _study_for_condition(project_root, condition_id)
    condition = _condition(study, condition_id)
    task = str(study["task"])
    pipeline = _pipeline(project_root, task)
    population = str(study["population"])
    datasets = _datasets(project_root, pipeline, population)
    placeholders = [str(value) for value in condition["placeholders"]]
    configs: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str, str], list[str]] = {}
    for model_id, raw_model in condition["models"].items():
        model = dict(raw_model)
        for placeholder in placeholders:
            for dataset in datasets:
                config, _ = _source_config(
                    project_root,
                    task=task,
                    dataset=dataset,
                    variant="off-shelf",
                    seed=None,
                )
                config_id = f"complete-readout-grid--{model_id}--grade-{placeholder}--{dataset['id']}"
                _set_identity(
                    config,
                    config_id=config_id,
                    study=study,
                    condition_id=condition_id,
                )
                config["reranker"] = _readout_reranker(
                    model,
                    dataset,
                    placeholder=placeholder,
                    serving_width=int(condition["serving_width"]),
                    chunk_assignment=str(condition["chunk_assignment"]),
                )
                config.pop("robustness", None)
                config.pop("qids_to_run_path", None)
                config.pop("qids_to_run", None)
                configs.append(config)
                rows.append(
                    {
                        **_row(
                            config,
                            task=task,
                            dataset=dataset,
                            variant="off-shelf",
                            seed=None,
                            checkpoint=None,
                        ),
                        "model": str(model_id),
                        "model_name": str(model["model_name"]),
                        "paper_name": str(model["paper_name"]),
                        "placeholder": placeholder,
                        "quality_metric": str(condition["quality_metric"]),
                    }
                )
                signature = json.dumps(
                    {
                        key: value
                        for key, value in config["reranker"].items()
                        if key not in {"instruction", "max_doc_chars"}
                    },
                    sort_keys=True,
                )
                grouped.setdefault((str(model_id), placeholder, signature), []).append(config_id)

    result = _write(project_root, study, condition_id, configs, rows)
    config_root = project_root / str(study["outputs"]["config_root"])
    bundle_ids: list[str] = []
    per_cell_bundle_index: dict[tuple[str, str], int] = {}
    for (model_id, placeholder, _), member_ids in grouped.items():
        key = (model_id, placeholder)
        index = per_cell_bundle_index.get(key, 0)
        per_cell_bundle_index[key] = index + 1
        bundle_id = f"complete-readout-grid--{model_id}--grade-{placeholder}--bundle{index}"
        bundle = {
            "id": bundle_id,
            "bundle": {
                "member_configs": member_ids,
                "shared_base": True,
            },
            "execution": {"environment": {"VLLM_WORKER_MULTIPROC_METHOD": "spawn"}},
        }
        write_config(config_root / f"{bundle_id}.yaml", bundle)
        bundle_ids.append(bundle_id)

    sweep_path = project_root / str(result["sweep"])
    write_sweep(
        sweep_path,
        {
            "id": "complete-readout-grid--canonical",
            "description": ("Missing K=1 quality passes for the complete readout and placeholder grids."),
            "execution": {"max_parallel": 1},
            "jobs": [{"exp_id": bundle_id} for bundle_id in bundle_ids],
        },
    )
    provenance_path = project_root / str(result["provenance"])
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance.update(
        {
            "expected_placeholder_cells": (len(condition["models"]) * len(datasets) * len(placeholders)),
            "generated_cells": len(configs),
            "bundle_configs": bundle_ids,
        }
    )
    provenance_path.write_text(
        json.dumps(provenance, indent=2) + "\n",
        encoding="utf-8",
    )
    result["bundle_count"] = len(bundle_ids)
    result["bundles"] = bundle_ids
    return result


def _study_for_condition(project_root: Path, condition_id: str) -> Mapping[str, Any]:
    """Return the one tracked study declaring ``condition_id``."""
    owners = []
    for study_id in STUDY_FILES:
        study = load_study(project_root, study_id)
        if condition_id in (study.get("conditions") or {}):
            owners.append(study)
    if not owners:
        raise ReproductionError(f"No tracked study declares condition {condition_id}")
    if len(owners) > 1:
        raise ReproductionError(
            f"Condition {condition_id} is declared by multiple studies: {sorted(study['id'] for study in owners)}"
        )
    return owners[0]


def _task_population(pipeline: Mapping[str, Any], *declared: Any) -> str:
    """Return the first declared population, else the pipeline default."""
    for value in declared:
        if value:
            return str(value)
    return str(pipeline["direct_eval"]["population"])


def _emit_cell(
    project_root: Path,
    *,
    study: Mapping[str, Any],
    condition_id: str,
    task: str,
    dataset: Mapping[str, Any],
    variant: str,
    seed: int | None,
    config_id: str,
    mutate: Callable[[dict[str, Any]], None] | None = None,
    row_extra: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build one config and its provenance row for a single study cell.

    Every materializer shares this middle: resolve the source config for a
    (task, dataset, variant, seed), stamp the generated id, apply the
    condition's protocol mutations, and record the row. Only the loop shape and
    the mutations differ, so those stay with the caller.
    """
    config, checkpoint = _source_config(
        project_root,
        task=task,
        dataset=dataset,
        variant=variant,
        seed=seed,
    )
    _set_identity(
        config,
        config_id=config_id,
        study=study,
        condition_id=condition_id,
    )
    if mutate is not None:
        mutate(config)
    row = _row(
        config,
        task=task,
        dataset=dataset,
        variant=variant,
        seed=seed,
        checkpoint=checkpoint,
    )
    return config, ({**row, **row_extra} if row_extra else row)


def _materialize_variant_condition(
    project_root: Path,
    *,
    condition_id: str,
) -> dict[str, Any]:
    """Materialize one variant x dataset condition from its declaration.

    Every axis comes from the condition block: the task, training seed,
    population, serving width, variants, and the optional grade placeholder.
    """
    study = _study_for_condition(project_root, condition_id)
    condition = _condition(study, condition_id)
    placeholder = condition.get("placeholder")
    task = str(condition["task"])
    seed = int(condition["training_seed"])
    pipeline = _pipeline(project_root, task)
    population = _task_population(pipeline, condition.get("population"), study.get("population"))

    def mutate(config: dict[str, Any]) -> None:
        if condition.get("serving_width") is not None:
            _set_width(config, int(condition["serving_width"]))
        if placeholder is None:
            return
        _set_placeholder(
            config,
            str(placeholder),
            str(condition.get("chunk_assignment") or "sorted"),
        )
        _set_random_protocol(
            config,
            seeds=[int(value) for value in condition.get("presentation_seeds") or range(10)],
            perturbations=[str(name) for name in condition.get("perturbations") or ["random_shuffle"]],
        )

    configs = []
    rows = []
    for variant in condition["variants"]:
        for dataset in _datasets(project_root, pipeline, population):
            config, row = _emit_cell(
                project_root,
                study=study,
                condition_id=condition_id,
                task=task,
                dataset=dataset,
                variant=str(variant),
                seed=seed,
                config_id=(f"{condition_id}--{variant}--seed{seed}--{dataset['id']}"),
                mutate=mutate,
            )
            configs.append(config)
            rows.append(row)
    return _write(project_root, study, condition_id, configs, rows)


def _validate_execution_partition(
    datasets: list[dict[str, Any]],
    *,
    standalone: list[str],
    bundle_groups: list[list[str]],
    label: str,
) -> None:
    """Require one explicit execution partition to cover each dataset once."""
    expected = {str(dataset["id"]) for dataset in datasets}
    covered = [*standalone, *(item for group in bundle_groups for item in group)]
    if len(covered) != len(set(covered)) or set(covered) != expected:
        raise ReproductionError(
            f"{label} execution topology must cover the population once: "
            f"expected={sorted(expected)}, covered={sorted(covered)}"
        )


def _write_bundle_config(
    project_root: Path,
    study: Mapping[str, Any],
    *,
    bundle_id: str,
    member_ids: list[str],
    first: Mapping[str, Any],
    execution: Mapping[str, Any],
) -> None:
    """Write one semantic shared-base bundle without planning metadata."""
    bundle = {
        "id": bundle_id,
        "bundle": {
            "member_configs": member_ids,
            "shared_base": True,
            "reranker_overrides": {"vllm_settings": {"tensor_parallel_size": 1}},
        },
        "execution": {
            "environment": {"VLLM_WORKER_MULTIPROC_METHOD": "spawn"},
        },
    }
    channels = dict(first["execution"].get("extra_input_channels") or {})
    if channels:
        bundle["execution"]["extra_input_channels"] = channels
    config_root = project_root / str(study["outputs"]["config_root"])
    config_root.mkdir(parents=True, exist_ok=True)
    write_config(config_root / f"{bundle_id}.yaml", bundle)


def materialize_placeholder_controls(
    project_root: Path,
) -> dict[str, Any]:
    """Materialize the in-format Grade 1/2/3 and diagnostic ? controls."""
    condition_id = "placeholder-controls"
    study = _study_for_condition(project_root, condition_id)
    condition = _condition(study, condition_id)
    task = str(condition["task"])
    pipeline = _pipeline(project_root, task)
    datasets = _datasets(
        project_root,
        pipeline,
        str(condition.get("population") or study["population"]),
    )
    informativeness = condition["informativeness"]
    execution = condition["execution"]
    standalone = [str(dataset_id) for dataset_id in execution["standalone_datasets"]]
    bundle_groups = [[str(dataset_id) for dataset_id in group] for group in execution["bundle_groups"]]
    _validate_execution_partition(
        datasets,
        standalone=standalone,
        bundle_groups=bundle_groups,
        label=condition_id,
    )

    configs: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    grade_configs: dict[tuple[str, str], dict[str, Any]] = {}
    for dataset in datasets:
        dataset_id = str(dataset["id"])
        for label, placeholder in informativeness["placeholders"].items():
            config, checkpoint = _source_config(
                project_root,
                task=task,
                dataset=dataset,
                variant="off-shelf",
                seed=None,
            )
            config_id = f"placeholder-controls--qwen3-4b--grade-{label}--{dataset_id}"
            _set_identity(
                config,
                config_id=config_id,
                study=study,
                condition_id=condition_id,
            )
            _set_width(config, int(informativeness["serving_width"]))
            _set_placeholder(
                config,
                str(placeholder),
                str(informativeness["chunk_assignment"]),
            )
            _set_random_protocol(
                config,
                seeds=list(range(10)),
                perturbations=[str(value) for value in informativeness["perturbations"]],
            )
            config["execution"].setdefault("environment", {})["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
            config["execution"].pop("extra_input_channels", None)
            configs.append(config)
            grade_configs[(str(label), dataset_id)] = config
            rows.append(
                {
                    **_row(
                        config,
                        task=task,
                        dataset=dataset,
                        variant="off-shelf",
                        seed=None,
                        checkpoint=checkpoint,
                    ),
                    "control": "informativeness",
                    "placeholder": str(placeholder),
                    "placeholder_label": str(label),
                }
            )

    bundle_ids: list[str] = []
    grade_sweeps: list[str] = []
    sweep_root = project_root / str(study["outputs"]["sweep_root"])
    sweep_root.mkdir(parents=True, exist_ok=True)
    for label in informativeness["placeholders"]:
        label = str(label)
        jobs = [{"exp_id": grade_configs[(label, dataset_id)]["id"]} for dataset_id in standalone]
        for index, group in enumerate(bundle_groups):
            members = [str(grade_configs[(label, dataset_id)]["id"]) for dataset_id in group]
            bundle_id = f"placeholder-controls--grade-{label}--bundle{index}"
            _write_bundle_config(
                project_root,
                study,
                bundle_id=bundle_id,
                member_ids=members,
                first=grade_configs[(label, group[0])],
                execution=execution,
            )
            bundle_ids.append(bundle_id)
            jobs.append({"exp_id": bundle_id})
        sweep_id = f"placeholder-controls--grade-{label}--canonical"
        sweep_path = sweep_root / f"{sweep_id}.yaml"
        write_sweep(
            sweep_path,
            {
                "id": sweep_id,
                "description": (
                    f"Grade:{informativeness['placeholders'][label]} off-shelf random-only placeholder control."
                ),
                "execution": {"max_parallel": int(execution["max_parallel"])},
                "jobs": jobs,
            },
        )
        grade_sweeps.append(str(sweep_path.relative_to(project_root)))

    result = _write(
        project_root,
        study,
        condition_id,
        configs,
        rows,
        provenance_extra={
            "informativeness_configs": len(grade_configs),
            "bundle_configs": bundle_ids,
            "sweeps": [
                *grade_sweeps,
            ],
            "execution_topology": {
                "standalone_datasets": standalone,
                "bundle_groups": bundle_groups,
            },
        },
    )
    result["bundle_count"] = len(bundle_ids)
    result["bundles"] = bundle_ids
    result["sweeps"] = [result["sweep"], *grade_sweeps]
    return result


def materialize_trained_channel_cross(project_root: Path) -> dict[str, Any]:
    """Materialize trained placeholder/chunk-assignment crossing cells."""
    condition_id = "trained-channel-cross"
    study = _study_for_condition(project_root, condition_id)
    condition = _condition(study, condition_id)
    task = str(condition["task"])
    seed = int(condition["training_seed"])
    pipeline = _pipeline(project_root, task)
    configs = []
    rows = []
    population = _task_population(pipeline, condition.get("population"), study.get("population"))
    cells = list(condition["cells"])

    def mutate(config: dict[str, Any], cell: Mapping[str, Any]) -> None:
        _set_placeholder(
            config,
            str(cell["placeholder"]),
            str(cell["chunk_assignment"]),
        )
        _set_random_protocol(
            config,
            seeds=list(range(10)),
            perturbations=["random_shuffle"],
        )

    for cell in cells:
        for dataset in _datasets(project_root, pipeline, population):
            config, row = _emit_cell(
                project_root,
                study=study,
                condition_id=condition_id,
                task=task,
                dataset=dataset,
                variant=str(condition["variant"]),
                seed=seed,
                config_id=(f"trained-channel-cross--{cell['id']}--{dataset['id']}"),
                mutate=lambda config, spec=cell: mutate(config, spec),
            )
            configs.append(config)
            rows.append(row)
    return _write(project_root, study, condition_id, configs, rows)


def materialize_round_robin_eval_grid(
    project_root: Path,
) -> dict[str, Any]:
    """Materialize the four trained arms of the full P0.4 round-robin grid."""
    condition_id = "round-robin-eval-grid"
    study = _study_for_condition(project_root, condition_id)
    condition = _condition(study, condition_id)
    task = str(condition["task"])
    seed = int(condition["training_seed"])
    pipeline = _pipeline(project_root, task)
    population = str(condition["population"])
    datasets = _datasets(project_root, pipeline, population)
    dataset_ids = {str(dataset["id"]) for dataset in datasets}
    execution = condition["execution"]
    bundle_groups = [[str(dataset_id) for dataset_id in group] for group in execution["bundle_groups"]]
    grouped_ids = [dataset_id for group in bundle_groups for dataset_id in group]
    if len(grouped_ids) != len(set(grouped_ids)) or set(grouped_ids) != dataset_ids:
        raise ReproductionError(
            "Round-robin bundle groups must cover the frozen population once: "
            f"datasets={sorted(dataset_ids)}, groups={sorted(grouped_ids)}"
        )

    configs: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    configs_by_id: dict[str, dict[str, Any]] = {}
    member_ids: dict[tuple[str, str], str] = {}
    cap_overrides = {
        str(dataset_id): int(value) for dataset_id, value in (condition.get("engine_cap_overrides") or {}).items()
    }
    checkpoint_sources = condition.get("checkpoint_sources") or {}
    for variant in condition["variants"]:
        variant_id = str(variant)
        checkpoint_source = checkpoint_sources.get(variant_id) or {}
        for dataset in datasets:
            dataset_id = str(dataset["id"])
            config, checkpoint = _source_config(
                project_root,
                task=task,
                dataset=dataset,
                variant=variant_id,
                seed=seed,
                checkpoint_variant=(
                    str(checkpoint_source["variant"]) if checkpoint_source.get("variant") is not None else None
                ),
                checkpoint_catalog=(
                    str(checkpoint_source["catalog"]) if checkpoint_source.get("catalog") is not None else None
                ),
            )
            config_id = f"round-robin-eval-grid--{variant_id}--{dataset_id}"
            _set_identity(
                config,
                config_id=config_id,
                study=study,
                condition_id=condition_id,
            )
            _set_placeholder(
                config,
                str(condition["placeholder"]),
                str(condition["chunk_assignment"]),
            )
            _set_random_protocol(
                config,
                seeds=[int(value) for value in condition["presentation_seeds"]],
                perturbations=["random_shuffle"],
            )
            config["robustness"]["sc_k_subsets"] = [int(value) for value in condition["sc_k_subsets"]]
            if dataset_id in cap_overrides:
                cap = cap_overrides[dataset_id]
                config["reranker"]["max_length"] = cap
                config["reranker"].setdefault("vllm_settings", {})["max_model_len"] = cap
            configs.append(config)
            configs_by_id[config_id] = config
            member_ids[(variant_id, dataset_id)] = config_id
            rows.append(
                {
                    **_row(
                        config,
                        task=task,
                        dataset=dataset,
                        variant=variant_id,
                        seed=seed,
                        checkpoint=checkpoint,
                    ),
                    "chunk_assignment": str(condition["chunk_assignment"]),
                    "placeholder": str(condition["placeholder"]),
                    "presentation_seeds": [int(value) for value in condition["presentation_seeds"]],
                    "sc_k_subsets": [int(value) for value in condition["sc_k_subsets"]],
                }
            )

    config_root = project_root / str(study["outputs"]["config_root"])
    config_root.mkdir(parents=True, exist_ok=True)
    sweep_jobs: list[dict[str, Any]] = []
    bundle_ids: list[str] = []
    for variant in condition["variants"]:
        variant_id = str(variant)
        for index, group in enumerate(bundle_groups):
            group_members = [member_ids[(variant_id, dataset_id)] for dataset_id in group]
            if len(group_members) == 1:
                sweep_jobs.append({"exp_id": (f"configs/experiments/{group_members[0]}.yaml")})
                continue
            bundle_id = f"round-robin-eval-grid--{variant_id}--bundle{index}"
            first = configs_by_id[group_members[0]]
            bundle = {
                "id": bundle_id,
                "bundle": {
                    "member_configs": group_members,
                    "shared_base": True,
                    "reranker_overrides": {"vllm_settings": {"tensor_parallel_size": 1}},
                },
                "execution": {
                    "environment": {"VLLM_WORKER_MULTIPROC_METHOD": "spawn"},
                    "extra_input_channels": dict(first["execution"].get("extra_input_channels") or {}),
                },
            }
            write_config(config_root / f"{bundle_id}.yaml", bundle)
            bundle_ids.append(bundle_id)
            sweep_jobs.append({"exp_id": bundle_id})

    reused_round_robin_variants = 2
    full_variant_count = len(condition["variants"]) + reused_round_robin_variants
    return _write(
        project_root,
        study,
        condition_id,
        configs,
        rows,
        sweep_jobs=sweep_jobs,
        sweep_execution={"max_parallel": int(execution["max_parallel"])},
        provenance_extra={
            "expected_grid_cells": full_variant_count * 2 * len(datasets),
            "generated_round_robin_cells": len(configs),
            "reused_contiguous_cells": full_variant_count * len(datasets),
            "reused_round_robin_cells": reused_round_robin_variants * len(datasets),
            "bundle_configs": bundle_ids,
            "protocol_note": (
                "Random-only presentation seeds 0..9. Touche-2020 uses the "
                "8192-token cap of the existing interleaved and trained-channel "
                "round-robin cells."
            ),
        },
    )


def materialize_matched_variance(
    project_root: Path,
) -> dict[str, Any]:
    """Materialize matched-variance controls across all canonical tasks."""
    condition_id = "matched-variance"
    study = _study_for_condition(project_root, condition_id)
    condition = _condition(study, condition_id)
    variant = str(condition["variant"])
    seed = int(condition["training_seed"])
    configs = []
    rows = []
    for task, task_spec in condition["tasks"].items():
        pipeline = _pipeline(project_root, str(task))
        population = _task_population(pipeline, task_spec.get("population"))
        for dataset in _datasets(project_root, pipeline, population):
            config, row = _emit_cell(
                project_root,
                study=study,
                condition_id=condition_id,
                task=str(task),
                dataset=dataset,
                variant=variant,
                seed=seed,
                config_id=(f"matched-variance--{task}--{dataset['id']}--seed{seed}"),
                mutate=lambda config, spec=task_spec: _set_matched_variance(
                    config,
                    seeds=list(condition["presentation_seeds"]),
                    request_batch_size=int(condition["request_batch_size"]),
                    k_cutoff=int(spec["k_cutoff"]),
                ),
            )
            configs.append(config)
            rows.append(row)
    return _write(project_root, study, condition_id, configs, rows)


def materialize_context_decomposition(
    project_root: Path,
) -> dict[str, Any]:
    """Materialize the primary-18 paired width/content decomposition."""
    condition_id = "context-decomposition"
    study = _study_for_condition(project_root, condition_id)
    condition = _condition(study, condition_id)
    task = str(condition["task"])
    variant = str(condition["variant"])
    seed = int(condition["training_seed"])
    pipeline = _pipeline(project_root, task)
    population = str(condition["population"])
    labels = {str(key): str(value) for key, value in condition["dataset_labels"].items()}
    shard_counts = {str(key): int(value) for key, value in condition["shard_counts"].items()}
    execution = condition["execution"]
    configs: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    sweep_jobs: list[dict[str, Any]] = []
    run_manifest_rows: list[dict[str, Any]] = []
    datasets = _datasets(project_root, pipeline, population)
    dataset_ids = {str(dataset["id"]) for dataset in datasets}
    if set(labels) != dataset_ids or set(shard_counts) != dataset_ids:
        raise ReproductionError(
            "Context-decomposition labels and shard counts must cover the "
            f"population exactly: datasets={sorted(dataset_ids)}, "
            f"labels={sorted(labels)}, shards={sorted(shard_counts)}"
        )

    for dataset in datasets:
        dataset_id = str(dataset["id"])
        config, checkpoint = _source_config(
            project_root,
            task=task,
            dataset=dataset,
            variant=variant,
            seed=seed,
        )
        config_id = f"context-decomposition--{task}--{dataset_id}--seed{seed}"
        _set_identity(
            config,
            config_id=config_id,
            study=study,
            condition_id=condition_id,
        )
        _set_context_decomposition(
            config,
            condition=condition,
            dataset_id=dataset_id,
            donor_seed_namespace=config_id,
        )
        shard_count = shard_counts[dataset_id]
        configs.append(config)
        rows.append(
            {
                **_row(
                    config,
                    task=task,
                    dataset=dataset,
                    variant=variant,
                    seed=seed,
                    checkpoint=checkpoint,
                ),
                "label": labels[dataset_id],
                "shard_count": shard_count,
                "query_basis": str(condition["query_basis"]),
                "canonical_order": str(condition["canonical_order"]),
            }
        )
        sweep_jobs.extend(_context_decomposition_shard_jobs(config_id, shard_count))
        run_manifest_rows.append(
            {
                "config_id": config_id,
                "run_dir": None,
                "expected_shards": shard_count,
                "merge_required": shard_count > 1,
            }
        )

    return _write(
        project_root,
        study,
        condition_id,
        configs,
        rows,
        sweep_jobs=sweep_jobs,
        sweep_execution={"max_parallel": int(execution["max_parallel"])},
        provenance_extra={
            "expected_cells": len(datasets),
            "shard_job_count": len(sweep_jobs),
            "query_basis": str(condition["query_basis"]),
            "contrasts": dict(condition["contrasts"]),
            "collection": {
                "artifact": ("context_decomposition/context_decomposition_metrics.json"),
                "requires_merged_dataset_runs": True,
            },
        },
        run_manifest_rows=run_manifest_rows,
    )


def materialize_instrument_width(
    project_root: Path,
) -> dict[str, Any]:
    """Materialize fixed-checkpoint B=1 instrument cells."""
    return _materialize_variant_condition(project_root, condition_id="instrument-width")


def materialize_fixed_weight_multiseed(
    project_root: Path,
) -> dict[str, Any]:
    """Materialize native/B=1 OC-SFT pairs over all tasks and seeds."""
    condition_id = "fixed-weight-multiseed"
    study = _study_for_condition(project_root, condition_id)
    condition = _condition(study, condition_id)
    variant = str(condition["variant"])
    configs = []
    rows = []
    for task, task_spec in condition["tasks"].items():
        pipeline = _pipeline(project_root, str(task))
        population = _task_population(pipeline, task_spec.get("population"))
        for seed in condition["training_seeds"]:
            for dataset in _datasets(project_root, pipeline, population):
                for width in (int(task_spec["native_width"]), 1):
                    config, row = _emit_cell(
                        project_root,
                        study=study,
                        condition_id=condition_id,
                        task=str(task),
                        dataset=dataset,
                        variant=variant,
                        seed=int(seed),
                        config_id=(f"fixed-weight-width--{task}--{dataset['id']}--seed{seed}--b{width}"),
                        mutate=lambda config, b=width: _set_width(config, b),
                        row_extra={
                            "quality_metric": str(task_spec["quality_metric"]),
                            "native_width": int(task_spec["native_width"]),
                        },
                    )
                    configs.append(config)
                    rows.append(row)
    return _write(project_root, study, condition_id, configs, rows)


def materialize_multiseed_width(
    project_root: Path,
) -> dict[str, Any]:
    """Materialize the missing seed-43/44 B=1 passage-width endpoints."""
    condition_id = "multiseed-width"
    study = _study_for_condition(project_root, condition_id)
    condition = _condition(study, condition_id)
    task = str(condition["task"])
    variant = str(condition["variant"])
    pipeline = _pipeline(project_root, task)
    datasets = _datasets(project_root, pipeline, str(condition["population"]))
    execution = condition["execution"]
    partitions = [[str(dataset_id) for dataset_id in group] for group in execution["bundle_partitions"]]
    _validate_execution_partition(
        datasets,
        standalone=[],
        bundle_groups=partitions,
        label=condition_id,
    )
    configs: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    configs_by_cell: dict[tuple[int, str], dict[str, Any]] = {}

    def mutate(config: dict[str, Any], seed: int) -> None:
        _set_width(config, int(condition["serving_width"]))
        config["eval"]["measures"] = [str(condition["quality_metric"])]

    for seed in condition["training_seeds"]:
        seed = int(seed)
        for dataset in datasets:
            dataset_id = str(dataset["id"])
            config, row = _emit_cell(
                project_root,
                study=study,
                condition_id=condition_id,
                task=task,
                dataset=dataset,
                variant=variant,
                seed=seed,
                config_id=(f"multiseed-width--{task}--{dataset_id}--seed{seed}--b{condition['serving_width']}"),
                mutate=lambda config, s=seed: mutate(config, s),
                row_extra={"quality_metric": str(condition["quality_metric"])},
            )
            configs.append(config)
            configs_by_cell[(seed, dataset_id)] = config
            rows.append(row)

    bundle_ids: list[str] = []
    for seed in condition["training_seeds"]:
        seed = int(seed)
        seed_execution = {
            **execution,
        }
        for index, partition in enumerate(partitions):
            bundle_id = f"multiseed-width--seed{seed}--b1--bundle{index}"
            _write_bundle_config(
                project_root,
                study,
                bundle_id=bundle_id,
                member_ids=[str(configs_by_cell[(seed, dataset_id)]["id"]) for dataset_id in partition],
                first=configs_by_cell[(seed, partition[0])],
                execution=seed_execution,
            )
            bundle_ids.append(bundle_id)
    result = _write(
        project_root,
        study,
        condition_id,
        configs,
        rows,
        sweep_jobs=[{"exp_id": bundle_id} for bundle_id in bundle_ids],
        sweep_execution={"max_parallel": int(execution["max_parallel"])},
        provenance_extra={
            "bundle_configs": bundle_ids,
            "bundle_partitions": partitions,
        },
    )
    result["bundle_count"] = len(bundle_ids)
    result["bundles"] = bundle_ids
    return result


def materialize_grade_one(
    project_root: Path,
) -> dict[str, Any]:
    """Materialize trained method arms under the Grade:1 control."""
    return _materialize_variant_condition(project_root, condition_id="grade-one-control")


MATERIALIZERS = {
    "complete-readout-grid": materialize_complete_readout_grid,
    "placeholder-controls": materialize_placeholder_controls,
    "first-stage-panel": materialize_first_stage_panel,
    "first-stage-newsurf": materialize_first_stage_newsurf,
    "instrument-width": materialize_instrument_width,
    "fixed-weight-multiseed": materialize_fixed_weight_multiseed,
    "multiseed-width": materialize_multiseed_width,
    "context-decomposition": materialize_context_decomposition,
    "trained-channel-cross": materialize_trained_channel_cross,
    "round-robin-eval-grid": materialize_round_robin_eval_grid,
    "matched-variance": materialize_matched_variance,
    "grade-one-control": materialize_grade_one,
}


def materialize_study_condition(
    project_root: Path,
    condition_id: str,
) -> dict[str, Any]:
    """Materialize one supported semantic condition."""
    try:
        materializer = MATERIALIZERS[condition_id]
    except KeyError as exc:
        raise ReproductionError(f"No canonical materializer for {condition_id}") from exc
    return materializer(project_root)
