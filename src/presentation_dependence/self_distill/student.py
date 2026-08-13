"""LoRA SFT for self-distillation silver labels and consistency controls."""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import random
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml

from presentation_dependence.self_distill.engines.hf import last_real_token_positions
from presentation_dependence.self_distill.readout import (
    expected_grade,
    resolve_grade_token_ids,
)
from presentation_dependence.self_distill.ips_propensity import estimate_slot_propensity, format_slot_table
from presentation_dependence.self_distill.student_data import (
    GroupedSilverExample,
    RegressionChunk,
    SupervisedConsistencyChunk,
    export_regression_jsonl,
    export_supervised_consistency_jsonl,
    load_grouped_silver_examples,
    regression_chunks,
    supervised_consistency_chunks,
)
from presentation_dependence.self_distill.student_eval import (
    permutation_invariance_diagnostics,
    silver_prediction_diagnostics,
)


@dataclass(slots=True)
class StudentSFTConfig:
    """Flat runtime configuration for self-distill student SFT."""

    id: str
    model_name: str
    silver_labels_path: str | None
    fixture_path: str
    output_dir: str
    # Optional held-out eval silver. When set, ``train_student_sft`` loads this
    # second silver JSONL and uses it as the ``examples`` source for every
    # ``evaluate_student_subset`` call (at_start, every_n_steps, at_end). The
    # training loop continues to consume ``silver_labels_path``. Default
    # ``None`` preserves the historical behavior of evaluating on a subset of
    # the training silver (which confounds learning with memorization for
    # overfitting detection). When supplied, the file must reference qids
    # that also exist in ``fixture_path`` (extend the fixture as a one-time
    # operational step alongside the teacher held-out run).
    eval_silver_labels_path: str | None = None
    reranker_class: str = "Qwen3Reranker"
    instruction: str | None = None
    grade_rubric_id: str = "relevance_v1"
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)
    grade_skeleton_dummy: str = "0"
    max_length: int = 4096
    max_doc_chars: int = 1200
    chunk_size: int = 20
    candidate_set_id: str = "msmarco_seed42_bm25_top100"
    objective_type: str = "supervised_mse"
    objective_loss: str = "mse"
    objective_alpha: float = 1.0
    objective_beta: float = 0.0
    objective_temperature: float = 1.0
    ema_decay: float = 0.999
    # DebiasFirst positional calibration: reweight the per-(doc, slot) MSE by the
    # inverse propensity of the document's canonical first-stage slot. All no-ops
    # unless ``ips_enabled``. See ``presentation_dependence.self_distill.ips_propensity``.
    ips_enabled: bool = False
    ips_relevance_threshold: int = 1
    ips_smoothing_eps: float = 1e-3
    ips_clip: float | None = None
    consistency_lambda: float = 0.0
    consistency_lambda_warmup_steps: int = 0
    consistency_lambda_warmup_init: float = 0.0
    consistency_lambda_warmup_schedule: str = "linear"
    consistency_view_seeds: list[int] = field(default_factory=lambda: [0, 1])
    consistency_view_generator: str = "permutation"
    max_train_queries: int | None = None
    train_qids_path: str | None = None
    exclude_qids_path: str | None = None
    include_doc_ids_in_prompt: bool = False
    qrels_path: str | None = None
    dtype: str = "bfloat16"
    device: str = "auto"
    attn_implementation: str | None = None
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] | str = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"]
    )
    # When set, load an existing PEFT LoRA adapter from this path (a directory
    # containing ``adapter_config.json`` + ``adapter_model.safetensors``) as
    # the starting point for further training, instead of initialising a
    # fresh LoRA. The loaded adapter's architecture (r, alpha, dropout,
    # target_modules) is taken from its own ``adapter_config.json``;
    # the ``lora_*`` fields above are ignored in that case (a warning is logged).
    # Optimizer state and step counter are NOT restored (use
    # ``resume_from_checkpoint`` for that).
    lora_adapter_init_path: str | None = None
    epochs: int = 1
    lr: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    min_lr_ratio: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    batch_size_per_device: int = 4
    gradient_accumulation_steps: int = 8
    slot_forward_batch_size: int | None = None
    gradient_checkpointing: bool = False
    # Use the single-forward-per-chunk grade readout path. When True, each
    # chunk's full prompt is tokenized once and the K slot logits are gathered
    # in one model forward: vs the legacy multi-prefix path that does K
    # redundant forwards over the shared body. Mathematically identical under
    # causal attention (see ``compute_continuous_scores_single_forward``).
    single_forward_readout: bool = False
    # Skip DDP gradient all-reduce on non-final microbatches of every
    # optimizer step (PyTorch's ``DistributedDataParallel.no_sync()``).
    # Without this, every ``.backward()`` triggers an allreduce; with
    # ``gradient_accumulation_steps=N`` you pay N allreduces per opt step
    # instead of the 1 you need.
    #
    # Set to ``False`` explicitly in a config to reproduce legacy numerical
    # behavior bit-for-bit (i.e. one allreduce per backward).
    # No effect when ``world_size==1`` or ``gradient_accumulation_steps<=1``.
    ddp_no_sync_during_grad_accum: bool = True
    max_steps: int | None = None
    log_every_n_steps: int = 10
    heartbeat_every_s: float = 60.0
    progress_jsonl: bool = True
    progress_jsonl_path: str | None = None
    eval_max_queries: int = 0
    eval_every_n_steps: int = 0
    eval_at_start: bool = True
    eval_at_end: bool = True
    # When true (and we're in a DDP world), shard eval queries across ranks
    # and all-gather predictions onto rank 0. This eliminates the ~5-10 min
    # rank-0-only eval bubble that otherwise idles the other 7 ranks.
    eval_shard_across_ranks: bool = True
    # When true, deterministically shuffle examples with ``eval_seed`` before
    # slicing to ``eval_max_queries`` / ``eval_psi_max_queries``. Avoids the
    # lexicographic-qid bias of the raw ``examples[:N]`` slice. The PSI subset
    # remains a strict prefix of the silver-eval subset (both share the same
    # shuffled list).
    eval_shuffle: bool = True
    eval_seed: int = 42
    # Opt-in in-training PSI proxy. Computes permutation_score_variance,
    # permutation_spearman, worst_permutation_ndcg by running each eval query
    # under K permutations. Cost: ~K extra forward passes per query, so keep
    # the budget small (≤25 queries × 3 seeds is enough as a trend signal).
    # The canonical PSI harness still runs post-hoc on the final checkpoint.
    eval_psi_enabled: bool = False
    eval_psi_max_queries: int = 10
    eval_psi_seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    tensorboard_enabled: bool = False
    tensorboard_log_dir: str | None = None
    checkpoint_dir: str | None = None
    save_every_n_steps: int = 0
    keep_last_n_checkpoints: int = 3
    # Milestone retention on top of the rolling ``keep_last_n_checkpoints`` tail.
    # When > 0, any saved checkpoint whose ``global_step`` is a positive multiple
    # of this value is preserved forever (never pruned). Use with an aligned
    # ``save_every_n_steps`` (e.g. save=200, keep_every=600 → milestones at
    # 600/1200/1800). Default 0 = milestones disabled.
    keep_every_n_steps: int = 0
    # Best-by-eval-metric retention. When > 0, the K saved checkpoints with the
    # best in-training eval metric (default ``qrels_ndcg_cut_10`` on the held-out
    # cohort) are pinned and never pruned, regardless of the rolling tail or
    # milestone rules. This retains mid-training optima: in the 30K v5 runs,
    # NDCG peaked at step 600, which the rolling tail had evicted by step 1400.
    # Steps without an evaluation entry are ignored.
    keep_n_best: int = 0
    keep_n_best_metric: str = "qrels_ndcg_cut_10"
    # ``"max"`` for higher-is-better metrics (NDCG, Pearson, Spearman),
    # ``"min"`` for lower-is-better (MSE, loss).
    keep_n_best_mode: str = "max"
    resume_from_checkpoint: str | None = None
    seed: int = 42
    permutation_augmentation_enabled: bool = False
    permutation_augmentation_seeds: list[int] = field(default_factory=lambda: list(range(10)))
    # Robustness SFT augmentation: when enabled, each epoch rebuilds training
    # chunks from one or more shuffled views of the same doc_id -> silver_score
    # labels. ``chunk_sizes`` defaults to [chunk_size] for the minimal B-fixed
    # intervention; set e.g. [10, 20] + views_per_epoch > 1 only after eval
    # shows residual brittleness.
    permutation_augmentation_views_per_epoch: int = 1
    permutation_augmentation_chunk_sizes: list[int] = field(default_factory=list)
    # Query-level shuffle rate for each augmented view. 1.0 means every query is
    # shuffled (the original robustness-SFT ablation); lower rates keep a
    # deterministic BM25-order anchor for the remaining queries.
    permutation_augmentation_shuffle_rate: float = 1.0
    token_cache_enabled: bool = False
    token_cache_max_entries: int = 100_000
    token_cache_prefix_strings: bool = True
    token_cache_token_ids: bool = True
    # Label-vocabulary training.
    # When enabled, each consistency view independently samples a document-ID
    # marker scheme from ``label_vocab_train_pool`` so the existing consistency
    # MSE forces label-invariance alongside order-invariance. Default disabled
    # (numeric-by-slot markers; byte-identical to pre-Stage-2 training). Only
    # valid with ``objective.type=supervised_consistency`` on the
    # Qwen3-Instruct grade student. Eval-only schemes (roman/rand3) are held out
    # of the train pool so the Stage-1 analyzer still measures generalization.
    label_vocab_enabled: bool = False
    label_vocab_train_pool: list[str] = field(default_factory=lambda: ["numeric", "alpha", "doc_n"])
    label_vocab_decorrelate_slots: bool = True


def load_student_sft_config(config_path: str | Path) -> StudentSFTConfig:  # noqa: C901
    """Parse a self-distill SFT config into the flat runtime dataclass."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    student = cfg.get("student") or {}
    objective = student.get("objective") or {}
    data = student.get("data") or {}
    token_cache = data.get("token_cache") or {}
    permutation_augmentation = data.get("permutation_augmentation") or {}
    lora = student.get("lora") or {}
    training = student.get("training") or {}
    evaluation = student.get("evaluation") or {}
    observability = student.get("observability") or {}
    tensorboard = observability.get("tensorboard") or {}
    checkpoint = student.get("checkpoint") or {}
    reranker = cfg.get("reranker") or {}
    out = student.get("output_dir") or f"runs/self-distill/{cfg.get('id', 'student-sft')}/student"
    chat_template_kwargs = student.get("chat_template_kwargs", reranker.get("chat_template_kwargs") or {})
    if not isinstance(chat_template_kwargs, dict):
        raise ValueError("student.chat_template_kwargs / reranker.chat_template_kwargs must be a mapping when provided")
    grade_skeleton_dummy = str(student.get("grade_skeleton_dummy", reranker.get("grade_skeleton_dummy", "0")))
    if not grade_skeleton_dummy:
        raise ValueError("student.grade_skeleton_dummy / reranker.grade_skeleton_dummy must not be empty")
    objective_type = str(objective.get("type", "supervised_mse"))
    if objective_type not in {
        "supervised_mse",
        "supervised_consistency",
        "mean_teacher",
        "kl_to_base",
    }:
        raise ValueError(
            "student.objective.type must be one of "
            "{'supervised_mse', 'supervised_consistency', "
            "'mean_teacher', 'kl_to_base'}, "
            f"got {objective_type!r}"
        )
    objective_loss = str(objective.get("loss", "mse"))
    if objective_loss not in {"mse", "kl_vector", "combined"}:
        raise ValueError(
            f"student.objective.loss must be one of {{'mse', 'kl_vector', 'combined'}}, got {objective_loss!r}"
        )
    if objective_type not in {"supervised_mse", "supervised_consistency"} and objective_loss != "mse":
        raise ValueError(
            f"student.objective.loss={objective_loss!r} is only supported for "
            "student.objective.type in {'supervised_mse', 'supervised_consistency'}"
        )
    silver_labels_path = str(student["silver_labels_path"]) if student.get("silver_labels_path") else None
    if (
        objective_type
        in {
            "supervised_mse",
            "supervised_consistency",
            "mean_teacher",
            "kl_to_base",
        }
        and silver_labels_path is None
    ):
        raise ValueError(f"student.silver_labels_path is required for student.objective.type={objective_type}")
    consistency_lambda = float(objective.get("lambda", objective.get("consistency_lambda", 0.0)))
    consistency_view_generator = str(objective.get("view_generator", "permutation"))
    if consistency_view_generator not in {"permutation", "dropout"}:
        raise ValueError(
            "student.objective.view_generator must be one of "
            f"{{'permutation', 'dropout'}}, got {consistency_view_generator!r}"
        )
    if consistency_view_generator == "dropout" and objective_type != "supervised_consistency":
        raise ValueError(
            "student.objective.view_generator='dropout' is only supported for "
            f"student.objective.type='supervised_consistency', got {objective_type!r}"
        )
    objective_alpha = float(objective.get("alpha", 1.0))
    if not math.isfinite(objective_alpha) or objective_alpha < 0:
        raise ValueError("student.objective.alpha must be a finite value >= 0")
    objective_beta = float(objective.get("beta", 0.0))
    if not math.isfinite(objective_beta) or objective_beta < 0:
        raise ValueError("student.objective.beta must be a finite value >= 0")
    objective_temperature = float(objective.get("temperature", 1.0))
    if not math.isfinite(objective_temperature) or objective_temperature <= 0:
        raise ValueError("student.objective.temperature must be a finite value > 0")
    ema_decay = float(objective.get("ema_decay", 0.999))
    if not math.isfinite(ema_decay) or not 0.0 <= ema_decay < 1.0:
        raise ValueError("student.objective.ema_decay must be in [0, 1)")
    ips_cfg = objective.get("ips") or {}
    ips_enabled = bool(ips_cfg.get("enabled", False))
    ips_relevance_threshold = int(ips_cfg.get("relevance_threshold", 1))
    ips_smoothing_eps = float(ips_cfg.get("smoothing_eps", 1e-3))
    ips_clip_raw = ips_cfg.get("clip")
    ips_clip = float(ips_clip_raw) if ips_clip_raw is not None else None
    if ips_enabled:
        if objective_type != "supervised_mse":
            raise ValueError(
                "student.objective.ips.enabled is only supported for "
                f"student.objective.type='supervised_mse', got {objective_type!r}"
            )
        if objective_loss != "mse":
            raise ValueError(
                f"student.objective.ips.enabled requires student.objective.loss='mse', got {objective_loss!r}"
            )
        if not (student.get("qrels_path") or (cfg.get("eval") or {}).get("qrels_path")):
            raise ValueError(
                "student.qrels_path is required for student.objective.ips "
                "(the slot propensity is estimated from the training qrels)"
            )
        if ips_relevance_threshold < 1:
            raise ValueError(f"student.objective.ips.relevance_threshold must be >= 1, got {ips_relevance_threshold}")
        if ips_smoothing_eps <= 0 or not math.isfinite(ips_smoothing_eps):
            raise ValueError(f"student.objective.ips.smoothing_eps must be finite and > 0, got {ips_smoothing_eps}")
        if ips_clip is not None and (ips_clip <= 0 or not math.isfinite(ips_clip)):
            raise ValueError(f"student.objective.ips.clip must be finite and > 0 when set, got {ips_clip}")
    if objective_type == "supervised_consistency" and consistency_lambda < 0:
        raise ValueError(f"student.objective.lambda must be >= 0 for {objective_type}")
    lambda_warmup_cfg = objective.get("lambda_warmup") or {}
    consistency_lambda_warmup_steps = int(lambda_warmup_cfg.get("steps", objective.get("lambda_warmup_steps", 0)))
    consistency_lambda_warmup_init = float(lambda_warmup_cfg.get("init", objective.get("lambda_warmup_init", 0.0)))
    consistency_lambda_warmup_schedule = str(lambda_warmup_cfg.get("schedule", "linear"))
    if objective_type == "supervised_consistency":
        if consistency_lambda_warmup_steps < 0:
            raise ValueError(
                f"student.objective.lambda_warmup.steps must be >= 0, got {consistency_lambda_warmup_steps}"
            )
        if consistency_lambda_warmup_init < 0 or not math.isfinite(consistency_lambda_warmup_init):
            raise ValueError(
                "student.objective.lambda_warmup.init must be a finite value >= 0, "
                f"got {consistency_lambda_warmup_init}"
            )
        if consistency_lambda_warmup_init > consistency_lambda:
            raise ValueError(
                "student.objective.lambda_warmup.init "
                f"({consistency_lambda_warmup_init}) must be <= the target "
                f"consistency lambda ({consistency_lambda})"
            )
        if consistency_lambda_warmup_schedule != "linear":
            raise ValueError(
                "student.objective.lambda_warmup.schedule must be 'linear' "
                f"for OC-SFT, got {consistency_lambda_warmup_schedule!r}"
            )
    # ----- Stage-2 label-vocabulary training -----
    from presentation_dependence.rerankers.render_variants import (
        HELDOUT_ID_SCHEME_POOL as _HELDOUT_ID_SCHEME_POOL,
        TRAIN_ID_SCHEME_POOL as _TRAIN_ID_SCHEME_POOL,
        _ID_SCHEMES as _KNOWN_ID_SCHEMES,
    )

    label_vocab_cfg = objective.get("label_vocab") or {}
    label_vocab_enabled = bool(label_vocab_cfg.get("enabled", False))
    label_vocab_train_pool = [str(s) for s in label_vocab_cfg.get("train_pool", list(_TRAIN_ID_SCHEME_POOL))]
    label_vocab_decorrelate_slots = bool(label_vocab_cfg.get("decorrelate_slots", True))
    if label_vocab_enabled:
        if objective_type != "supervised_consistency":
            raise ValueError(
                "student.objective.label_vocab.enabled requires "
                f"objective.type=supervised_consistency, got {objective_type!r}"
            )
        reranker_class_for_label = str(student.get("reranker_class") or reranker.get("class") or "Qwen3Reranker")
        if reranker_class_for_label != "Qwen3InstructGradeReranker":
            raise ValueError(
                "student.objective.label_vocab.enabled is only supported for "
                f"reranker_class=Qwen3InstructGradeReranker, got {reranker_class_for_label!r}"
            )
        if not label_vocab_train_pool:
            raise ValueError("student.objective.label_vocab.train_pool must be non-empty when enabled")
        unknown = [s for s in label_vocab_train_pool if s not in set(_KNOWN_ID_SCHEMES)]
        if unknown:
            raise ValueError(
                f"student.objective.label_vocab.train_pool has unknown schemes {unknown}; "
                f"valid schemes: {list(_KNOWN_ID_SCHEMES)}"
            )
        leaked = [s for s in label_vocab_train_pool if s in set(_HELDOUT_ID_SCHEME_POOL)]
        if leaked:
            raise ValueError(
                f"student.objective.label_vocab.train_pool leaks held-out eval schemes {leaked}; "
                f"keep {list(_HELDOUT_ID_SCHEME_POOL)} out of training to measure generalization"
            )
    if consistency_view_generator == "dropout" and label_vocab_enabled:
        raise ValueError("student.objective.view_generator='dropout' cannot also enable label-vocabulary views")
    lora_dropout = float(lora.get("dropout", 0.05))
    if not math.isfinite(lora_dropout) or not 0.0 <= lora_dropout < 1.0:
        raise ValueError(f"student.lora.dropout must be a finite value in [0, 1), got {lora_dropout}")
    if consistency_view_generator == "dropout" and lora_dropout <= 0:
        raise ValueError("student.objective.view_generator='dropout' requires student.lora.dropout > 0")
    min_lr_ratio = float(training.get("min_lr_ratio", 0.0))
    if min_lr_ratio < 0.0 or min_lr_ratio > 1.0 or not math.isfinite(min_lr_ratio):
        raise ValueError(f"student.training.min_lr_ratio must be a finite value in [0, 1], got {min_lr_ratio}")
    return StudentSFTConfig(
        id=str(cfg.get("id") or student.get("id") or "student-sft"),
        model_name=str(student.get("base_model") or reranker.get("model_name")),
        reranker_class=str(student.get("reranker_class") or reranker.get("class") or "Qwen3Reranker"),
        silver_labels_path=silver_labels_path,
        eval_silver_labels_path=(
            str(student["eval_silver_labels_path"]) if student.get("eval_silver_labels_path") else None
        ),
        fixture_path=str(student["fixture_path"]),
        qrels_path=student.get("qrels_path") or (cfg.get("eval") or {}).get("qrels_path"),
        output_dir=str(out),
        instruction=student.get("instruction") or reranker.get("instruction"),
        grade_rubric_id=str(student.get("grade_rubric_id", reranker.get("grade_rubric_id", "relevance_v1"))),
        chat_template_kwargs=dict(chat_template_kwargs),
        grade_skeleton_dummy=grade_skeleton_dummy,
        max_length=int(student.get("max_length", reranker.get("max_length", 4096))),
        max_doc_chars=int(student.get("max_doc_chars", reranker.get("max_doc_chars", 1200))),
        chunk_size=int(data.get("chunk_size", student.get("chunk_size", 20))),
        candidate_set_id=str(data.get("candidate_set_id", "msmarco_seed42_bm25_top100")),
        objective_type=objective_type,
        objective_loss=objective_loss,
        objective_alpha=objective_alpha,
        objective_beta=objective_beta,
        objective_temperature=objective_temperature,
        ema_decay=ema_decay,
        ips_enabled=ips_enabled,
        ips_relevance_threshold=ips_relevance_threshold,
        ips_smoothing_eps=ips_smoothing_eps,
        ips_clip=ips_clip,
        consistency_lambda=consistency_lambda,
        consistency_lambda_warmup_steps=consistency_lambda_warmup_steps,
        consistency_lambda_warmup_init=consistency_lambda_warmup_init,
        consistency_lambda_warmup_schedule=consistency_lambda_warmup_schedule,
        consistency_view_seeds=[int(seed) for seed in objective.get("view_seeds", [0, 1])],
        consistency_view_generator=consistency_view_generator,
        max_train_queries=(int(data["max_train_queries"]) if data.get("max_train_queries") is not None else None),
        train_qids_path=str(data["train_qids_path"]) if data.get("train_qids_path") else None,
        exclude_qids_path=str(data["exclude_qids_path"]) if data.get("exclude_qids_path") else None,
        include_doc_ids_in_prompt=bool(data.get("include_doc_ids_in_prompt", False)),
        dtype=str(student.get("dtype", reranker.get("dtype", "bfloat16"))),
        device=str(student.get("device", reranker.get("device", "auto"))),
        attn_implementation=student.get("attn_implementation") or reranker.get("attn_implementation"),
        lora_r=int(lora.get("r", 16)),
        lora_alpha=int(lora.get("alpha", 32)),
        lora_dropout=lora_dropout,
        lora_target_modules=(
            str(lora["target_modules"])
            if isinstance(lora.get("target_modules"), str)
            else list(lora.get("target_modules") or ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"])
        ),
        lora_adapter_init_path=(str(lora["adapter_init_path"]) if lora.get("adapter_init_path") else None),
        epochs=int(training.get("epochs", 1)),
        lr=float(training.get("lr", 2e-4)),
        weight_decay=float(training.get("weight_decay", 0.01)),
        warmup_ratio=float(training.get("warmup_ratio", 0.05)),
        min_lr_ratio=min_lr_ratio,
        adam_beta1=float(training.get("adam_beta1", 0.9)),
        adam_beta2=float(training.get("adam_beta2", 0.95)),
        adam_eps=float(training.get("adam_eps", 1e-8)),
        batch_size_per_device=int(training.get("batch_size_per_device", 4)),
        gradient_accumulation_steps=int(
            training.get("grad_accumulation_steps", training.get("gradient_accumulation_steps", 8))
        ),
        slot_forward_batch_size=(
            int(training["slot_forward_batch_size"]) if training.get("slot_forward_batch_size") is not None else None
        ),
        gradient_checkpointing=bool(training.get("gradient_checkpointing", False)),
        single_forward_readout=bool(training.get("single_forward_readout", False)),
        ddp_no_sync_during_grad_accum=bool(training.get("ddp_no_sync_during_grad_accum", True)),
        max_steps=training.get("max_steps"),
        log_every_n_steps=int(training.get("log_every_n_steps", 10)),
        heartbeat_every_s=float(training.get("heartbeat_every_s", 60.0)),
        progress_jsonl=bool(training.get("progress_jsonl", True)),
        progress_jsonl_path=training.get("progress_jsonl_path"),
        eval_max_queries=int(evaluation.get("max_queries", training.get("eval_max_queries", 0))),
        eval_every_n_steps=int(evaluation.get("every_n_steps", training.get("eval_every_n_steps", 0))),
        eval_at_start=bool(evaluation.get("at_start", training.get("eval_at_start", True))),
        eval_at_end=bool(evaluation.get("at_end", training.get("eval_at_end", True))),
        eval_shard_across_ranks=bool(evaluation.get("shard_across_ranks", True)),
        eval_shuffle=bool(evaluation.get("shuffle", True)),
        eval_seed=int(evaluation.get("seed", 42)),
        eval_psi_enabled=bool((evaluation.get("psi") or {}).get("enabled", False)),
        eval_psi_max_queries=int((evaluation.get("psi") or {}).get("max_queries", 10)),
        eval_psi_seeds=list((evaluation.get("psi") or {}).get("seeds", [0, 1, 2])),
        tensorboard_enabled=bool(tensorboard.get("enabled", False)),
        tensorboard_log_dir=tensorboard.get("log_dir"),
        checkpoint_dir=checkpoint.get("dir"),
        save_every_n_steps=int(checkpoint.get("save_every_n_steps", training.get("save_every_n_steps", 0))),
        keep_last_n_checkpoints=int(checkpoint.get("keep_last_n", checkpoint.get("keep_last_n_checkpoints", 3))),
        keep_every_n_steps=int(checkpoint.get("keep_every_n_steps", training.get("keep_every_n_steps", 0))),
        keep_n_best=int(checkpoint.get("keep_n_best", training.get("keep_n_best", 0))),
        keep_n_best_metric=str(
            checkpoint.get("keep_n_best_metric", training.get("keep_n_best_metric", "qrels_ndcg_cut_10"))
        ),
        keep_n_best_mode=str(checkpoint.get("keep_n_best_mode", training.get("keep_n_best_mode", "max"))),
        resume_from_checkpoint=checkpoint.get("resume_from_checkpoint"),
        seed=int(training.get("seed", 42)),
        permutation_augmentation_enabled=bool(permutation_augmentation.get("enabled", False)),
        permutation_augmentation_seeds=[int(seed) for seed in permutation_augmentation.get("seeds", list(range(10)))],
        permutation_augmentation_views_per_epoch=int(permutation_augmentation.get("views_per_epoch", 1)),
        permutation_augmentation_chunk_sizes=[int(size) for size in permutation_augmentation.get("chunk_sizes", [])],
        permutation_augmentation_shuffle_rate=float(permutation_augmentation.get("shuffle_rate", 1.0)),
        token_cache_enabled=bool(token_cache.get("enabled", False)),
        token_cache_max_entries=int(token_cache.get("max_entries", 100_000)),
        token_cache_prefix_strings=bool(token_cache.get("cache_prefix_strings", True)),
        token_cache_token_ids=bool(token_cache.get("cache_token_ids", True)),
        label_vocab_enabled=label_vocab_enabled,
        label_vocab_train_pool=label_vocab_train_pool,
        label_vocab_decorrelate_slots=label_vocab_decorrelate_slots,
    )


def _chat_template_kwargs_for_tokenizer(tokenizer: Any) -> dict[str, Any] | None:
    """Return run-scoped chat-template kwargs attached during student setup."""
    raw = getattr(tokenizer, "_slm_chat_template_kwargs", None)
    if not isinstance(raw, dict) or not raw:
        return None
    return dict(raw)


def _chat_template_kwargs_cache_key(tokenizer: Any) -> str:
    kwargs = _chat_template_kwargs_for_tokenizer(tokenizer) or {}
    return json.dumps(kwargs, sort_keys=True, default=str)


def _grade_rubric_id_for_tokenizer(tokenizer: Any) -> str | None:
    raw = getattr(tokenizer, "_slm_grade_rubric_id", None)
    if raw is None:
        return None
    return str(raw)


def build_prefixes_for_chunk(
    *,
    reranker_class: str,
    tokenizer: Any,
    instruction: str | None,
    query_text: str,
    passages: list[dict[str, str]],
    max_doc_chars: int,
) -> list[str]:
    """Reuse the exact teacher prompt skeleton for SFT loss construction."""
    if reranker_class == "Qwen3Reranker":
        from presentation_dependence.rerankers.qwen3 import _build_qwen3_setwise_grade_skeleton_prefixes

        _prompt, prefixes = _build_qwen3_setwise_grade_skeleton_prefixes(
            instruction, query_text, passages, max_doc_chars=max_doc_chars
        )
        return prefixes
    if reranker_class == "Qwen3InstructGradeReranker":
        from presentation_dependence.rerankers.qwen3_instruct_grade import _build_qwen3_instruct_grade_skeleton

        _prompt, prefixes = _build_qwen3_instruct_grade_skeleton(
            tokenizer,
            instruction,
            query_text,
            passages,
            max_doc_chars=max_doc_chars,
            chat_template_kwargs=_chat_template_kwargs_for_tokenizer(tokenizer),
            dummy_grade=getattr(tokenizer, "_slm_grade_skeleton_dummy", "0"),
            grade_rubric_id=_grade_rubric_id_for_tokenizer(tokenizer),
        )
        return prefixes
    if reranker_class == "Gemma4GradeReranker":
        from presentation_dependence.rerankers.gemma4 import _build_gemma4_grade_skeleton_prefixes

        _prompt, prefixes = _build_gemma4_grade_skeleton_prefixes(
            tokenizer,
            instruction,
            query_text,
            passages,
            max_doc_chars=max_doc_chars,
            chat_template_kwargs=_chat_template_kwargs_for_tokenizer(tokenizer),
            dummy_grade=getattr(tokenizer, "_slm_grade_skeleton_dummy", "0"),
            grade_rubric_id=_grade_rubric_id_for_tokenizer(tokenizer),
        )
        return prefixes
    if reranker_class == "Granite41GradeReranker":
        from presentation_dependence.rerankers.granite_41 import _build_granite_grade_skeleton_prefixes

        _prompt, prefixes = _build_granite_grade_skeleton_prefixes(
            tokenizer,
            instruction,
            query_text,
            passages,
            max_doc_chars=max_doc_chars,
            prefix_mode="cumulative",
            dummy_grade="0",
            grade_rubric_id=_grade_rubric_id_for_tokenizer(tokenizer),
        )
        return prefixes
    raise ValueError(f"Unsupported student.reranker_class={reranker_class!r}")


class StudentTokenCache:
    """Small in-memory cache for prompt prefixes and token IDs within one run.

    Structural caveats from the cache audit:

    1. Under ``epochs=1`` + ``permutation_augmentation.enabled=false`` (our
       current SFT geometry), every chunk is built exactly once during
       training. The prefix-string cache key includes ``chunk_id`` and
       ``permutation_seed`` so each training-loop ``get_prefixes`` call is
       a unique miss-then-insert that's never re-read — effective
       training hit-rate is 0% by construction.
    2. The token-ID cache is content-addressed (``sha1(prefix_text)``) so
       it does hit on eval calls (the same 25/100 fixture queries are
       re-tokenized every 200 steps). But eval is <1% of total tokenize
       volume, so the blended throughput impact is ≲1%.
    3. When ``single_forward_readout=true``, neither cache layer is on
       the hot path at all — the single-forward grade readout tokenizes
       its full prompt directly via ``prepare_single_forward``
       and never calls ``get_prefixes``/``encode_prefix``. We force-disable
       the cache in that mode below so the observability counters stay
       quiet (0/0) instead of accumulating noise.

    A single-forward-aware cache keyed on ``(qid, chunk_id, perm_seed)``
    storing ``(input_ids, slot_positions)`` would hit on eval re-runs, but
    eval is not a bottleneck today — kept as a future optimization.
    """

    def __init__(self, config: StudentSFTConfig):
        """Initialize cache according to ``student.data.token_cache`` config."""
        is_single_forward = bool(getattr(config, "single_forward_readout", False))
        if is_single_forward:
            # The single-forward path bypasses both cache layers; keep the
            # object alive (callers still pass it through) but silence the
            # counters so logs aren't littered with 0/0 cache stats.
            self.enabled = False
        else:
            self.enabled = bool(config.token_cache_enabled)
        self.cache_prefix_strings = bool(config.token_cache_prefix_strings)
        self.cache_token_ids = bool(config.token_cache_token_ids)
        self.max_entries = max(int(config.token_cache_max_entries), 0)
        self.prefixes: dict[tuple[Any, ...], list[str]] = {}
        self.token_ids: OrderedDict[tuple[str, int, str], list[int]] = OrderedDict()
        self.prefix_hits = 0
        self.prefix_misses = 0
        self.token_hits = 0
        self.token_misses = 0

    @staticmethod
    def _tokenizer_key(tokenizer: Any) -> str:
        name = getattr(tokenizer, "name_or_path", None) or type(tokenizer).__name__
        vocab_size = getattr(tokenizer, "vocab_size", None)
        return f"{name}:{vocab_size}"

    @staticmethod
    def _hash_text(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    def get_prefixes(
        self,
        *,
        chunk: RegressionChunk,
        reranker_class: str,
        instruction: str | None,
        tokenizer: Any,
        max_doc_chars: int,
    ) -> list[str]:
        """Return cached or newly-built skeleton prefixes for one chunk."""
        if not (self.enabled and self.cache_prefix_strings):
            return build_prefixes_for_chunk(
                reranker_class=reranker_class,
                tokenizer=tokenizer,
                instruction=instruction,
                query_text=chunk.query_text,
                passages=chunk.passages,
                max_doc_chars=max_doc_chars,
            )
        key = (
            reranker_class,
            instruction or "",
            _grade_rubric_id_for_tokenizer(tokenizer) or "",
            _chat_template_kwargs_cache_key(tokenizer),
            chunk.query_id,
            chunk.candidate_set_id,
            chunk.chunk_id,
            chunk.permutation_seed,
            max_doc_chars,
            tuple(chunk.doc_ids),
        )
        cached = self.prefixes.get(key)
        if cached is not None:
            self.prefix_hits += 1
            return cached
        self.prefix_misses += 1
        prefixes = build_prefixes_for_chunk(
            reranker_class=reranker_class,
            tokenizer=tokenizer,
            instruction=instruction,
            query_text=chunk.query_text,
            passages=chunk.passages,
            max_doc_chars=max_doc_chars,
        )
        self.prefixes[key] = prefixes
        return prefixes

    def encode_prefix(self, tokenizer: Any, prefix: str, *, max_length: int) -> list[int]:
        """Return cached or newly-tokenized prefix IDs."""
        if not (self.enabled and self.cache_token_ids and self.max_entries > 0):
            return tokenizer(
                prefix,
                add_special_tokens=False,
                truncation=True,
                max_length=max_length,
            )["input_ids"]
        key = (self._tokenizer_key(tokenizer), int(max_length), self._hash_text(prefix))
        cached = self.token_ids.get(key)
        if cached is not None:
            self.token_hits += 1
            self.token_ids.move_to_end(key)
            return cached
        self.token_misses += 1
        ids = tokenizer(
            prefix,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        )["input_ids"]
        self.token_ids[key] = list(ids)
        self.token_ids.move_to_end(key)
        while len(self.token_ids) > self.max_entries:
            self.token_ids.popitem(last=False)
        return ids

    def snapshot(self, *, reset: bool = False) -> dict[str, int]:
        """Return cache counters, optionally resetting hit/miss deltas.

        Returns ``{}`` when the cache is force-disabled (e.g. under
        ``single_forward_readout=true``) so progress logs / TB metrics
        don't carry 0/0 noise fields.
        """
        if not self.enabled:
            return {}
        stats = {
            "token_cache_prefix_hits": self.prefix_hits,
            "token_cache_prefix_misses": self.prefix_misses,
            "token_cache_token_hits": self.token_hits,
            "token_cache_token_misses": self.token_misses,
            "token_cache_prefix_entries": len(self.prefixes),
            "token_cache_token_entries": len(self.token_ids),
        }
        if reset:
            self.prefix_hits = 0
            self.prefix_misses = 0
            self.token_hits = 0
            self.token_misses = 0
        return stats


def compute_continuous_scores_single_forward(  # noqa: C901
    *,
    model: Any,
    tokenizer: Any,
    chunk_specs: list[tuple[str, list[str]]],
    grade_token_ids: list[int],
    max_length: int,
) -> Any:
    """One forward per chunk, multi-position grade readout (differentiable).

    Background
    ----------
    The existing :func:`compute_continuous_scores_hf` path takes each chunk's
    K slot prefixes (each = "full prompt truncated just before slot k's
    dummy grade digit") and forwards them independently — paying ``K`` copies
    of the shared body (~1.2-2.3K tokens). For K=20 that's ~20× more attention
    FLOPs than necessary on the shared prefix.

    Instead, each chunk's *full* prompt (with all K dummy ``Grade: 0`` lines
    pre-filled) is tokenized once, the K slot positions (the token *just before*
    each ``Grade: 0``'s ``0``) are located via
    :func:`presentation_dependence.rerankers._slot_tokenization.prepare_single_forward`, the
    batch is left-padded, and a single forward yields logits at all K positions
    per chunk. We then apply :func:`expected_grade` slot-wise.

    Correctness
    -----------
    Under causal attention, logits at position ``P`` depend only on tokens
    ``0..P-1``. The multi-prefix path's slot-k prefix has exactly the same
    leading tokens as the single full prompt up to slot k's readout
    position. The two paths produce *mathematically identical* slot logits —
    the refactor is exact, not approximate. (TinyModel-based parity test in
    ``test_student.py::test_single_forward_parity_*`` asserts this.)

    Gradient flow is preserved end-to-end: we don't wrap the forward in
    ``torch.no_grad()``, ``logits[b].index_select(...)`` is differentiable,
    and ``expected_grade`` is softmax+dot-product — backward over K slot
    positions in a single chunk flows through the shared body activations
    once instead of K times. Combined with ``gradient_checkpointing``, peak
    activation memory is roughly the same as the existing path with
    ``slot_forward_batch_size=1``.

    Parameters
    ----------
    model, tokenizer :
        Same contract as :func:`compute_continuous_scores_hf`. Tokenizer
        should have ``padding_side="left"`` and ``truncation_side="left"``
        for production paths; this function asserts the latter at call time.
    chunk_specs :
        One ``(full_prompt, slot_prefixes)`` tuple per chunk, exactly as
        returned by :func:`build_full_prompt_and_prefixes`. Each chunk's
        ``slot_prefixes`` must be a *char-prefix list* of ``full_prompt`` —
        :func:`presentation_dependence.rerankers._slot_tokenization.prepare_single_forward`
        validates this.
    grade_token_ids :
        Single-token IDs for the grade digits — see :func:`resolve_grade_token_ids`.
    max_length :
        Left-truncation budget applied per-chunk before batching.

    Returns:
    -------
    torch.Tensor
        Shape ``(sum_b K_b,)``: slot scores flattened in chunk order then
        slot order. Caller must split back to chunks if needed
        (use ``[len(prefs) for _, prefs in chunk_specs]``).
    """
    import torch

    from presentation_dependence.rerankers._slot_tokenization import prepare_single_forward
    from presentation_dependence.self_distill.readout import expected_grade

    if not chunk_specs:
        raise ValueError("chunk_specs must be non-empty")

    prev_trunc = getattr(tokenizer, "truncation_side", None)
    if prev_trunc is not None:
        tokenizer.truncation_side = "left"
    try:
        rows: list[tuple[list[int], list[int]]] = []
        for full_prompt, slot_prefixes in chunk_specs:
            if not slot_prefixes:
                raise ValueError("each chunk must have at least one slot_prefix")
            input_ids, positions = prepare_single_forward(tokenizer, full_prompt, slot_prefixes, max_length)
            rows.append((input_ids, positions))
    finally:
        if prev_trunc is not None:
            tokenizer.truncation_side = prev_trunc

    max_len = max(len(ids) for ids, _ in rows)
    rem = max_len % 8
    if rem:
        max_len += 8 - rem

    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = 0

    batch_input_ids: list[list[int]] = []
    batch_attn: list[list[int]] = []
    adjusted_positions: list[list[int]] = []
    for input_ids, positions in rows:
        pad_n = max_len - len(input_ids)
        batch_input_ids.append([pad_id] * pad_n + list(input_ids))
        batch_attn.append([0] * pad_n + [1] * len(input_ids))
        adjusted_positions.append([p + pad_n for p in positions])

    device = next(model.parameters()).device
    inputs = {
        "input_ids": torch.tensor(batch_input_ids, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(batch_attn, dtype=torch.long, device=device),
    }
    keep_positions = torch.unique(
        torch.tensor(
            [position for positions in adjusted_positions for position in positions],
            dtype=torch.long,
            device=device,
        )
    )
    try:
        logits = model(**inputs, logits_to_keep=keep_positions).logits
        position_to_kept_idx = {int(pos): idx for idx, pos in enumerate(keep_positions.tolist())}
    except TypeError:
        logits = model(**inputs).logits
        position_to_kept_idx = None

    per_chunk_scores: list[Any] = []
    for b, positions in enumerate(adjusted_positions):
        if position_to_kept_idx is None:
            pos_t = torch.as_tensor(positions, dtype=torch.long, device=logits.device)
        else:
            pos_t = torch.as_tensor(
                [position_to_kept_idx[int(position)] for position in positions],
                dtype=torch.long,
                device=logits.device,
            )
        slot_logits = logits[b].index_select(0, pos_t)
        per_chunk_scores.append(expected_grade(slot_logits, grade_token_ids))
    return torch.cat(per_chunk_scores, dim=0)


def compute_grade_logits_single_forward(  # noqa: C901
    *,
    model: Any,
    tokenizer: Any,
    chunk_specs: list[tuple[str, list[str]]],
    grade_token_ids: list[int],
    max_length: int,
) -> Any:
    """One forward per chunk, returning grade-token logits for every slot."""
    import torch

    from presentation_dependence.rerankers._slot_tokenization import prepare_single_forward

    if not chunk_specs:
        raise ValueError("chunk_specs must be non-empty")

    prev_trunc = getattr(tokenizer, "truncation_side", None)
    if prev_trunc is not None:
        tokenizer.truncation_side = "left"
    try:
        rows: list[tuple[list[int], list[int]]] = []
        for full_prompt, slot_prefixes in chunk_specs:
            if not slot_prefixes:
                raise ValueError("each chunk must have at least one slot_prefix")
            input_ids, positions = prepare_single_forward(tokenizer, full_prompt, slot_prefixes, max_length)
            rows.append((input_ids, positions))
    finally:
        if prev_trunc is not None:
            tokenizer.truncation_side = prev_trunc

    max_len = max(len(ids) for ids, _ in rows)
    rem = max_len % 8
    if rem:
        max_len += 8 - rem

    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = 0

    batch_input_ids: list[list[int]] = []
    batch_attn: list[list[int]] = []
    adjusted_positions: list[list[int]] = []
    for input_ids, positions in rows:
        pad_n = max_len - len(input_ids)
        batch_input_ids.append([pad_id] * pad_n + list(input_ids))
        batch_attn.append([0] * pad_n + [1] * len(input_ids))
        adjusted_positions.append([p + pad_n for p in positions])

    device = next(model.parameters()).device
    inputs = {
        "input_ids": torch.tensor(batch_input_ids, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(batch_attn, dtype=torch.long, device=device),
    }
    keep_positions = torch.unique(
        torch.tensor(
            [position for positions in adjusted_positions for position in positions],
            dtype=torch.long,
            device=device,
        )
    )
    try:
        logits = model(**inputs, logits_to_keep=keep_positions).logits
        position_to_kept_idx = {int(pos): idx for idx, pos in enumerate(keep_positions.tolist())}
    except TypeError:
        logits = model(**inputs).logits
        position_to_kept_idx = None

    grade_ids = torch.as_tensor(list(grade_token_ids), dtype=torch.long, device=logits.device)
    per_chunk_logits: list[Any] = []
    for b, positions in enumerate(adjusted_positions):
        if position_to_kept_idx is None:
            pos_t = torch.as_tensor(positions, dtype=torch.long, device=logits.device)
        else:
            pos_t = torch.as_tensor(
                [position_to_kept_idx[int(position)] for position in positions],
                dtype=torch.long,
                device=logits.device,
            )
        slot_logits = logits[b].index_select(0, pos_t).index_select(-1, grade_ids)
        per_chunk_logits.append(slot_logits)
    return torch.cat(per_chunk_logits, dim=0)


def build_full_prompt_and_prefixes(
    *,
    reranker_class: str,
    tokenizer: Any,
    instruction: str | None,
    query_text: str,
    passages: list[dict[str, str]],
    max_doc_chars: int,
    slot_labels: tuple[str, ...] | None = None,
) -> tuple[str, list[str]]:
    """Companion to :func:`build_prefixes_for_chunk` that also returns the
    *full* prompt (the variant ending with the last slot's dummy grade).

    Needed by the single-forward readout path so we can tokenize once and
    locate the K slot positions inside that single prompt.

    ``slot_labels`` (Stage-2 label-vocabulary training) overrides the per-slot
    document markers for this view; only the Qwen3-Instruct grade reranker
    (the self-distill student class) supports it.
    """
    if slot_labels is not None and reranker_class != "Qwen3InstructGradeReranker":
        raise NotImplementedError(
            f"slot_labels override is only supported for Qwen3InstructGradeReranker, not {reranker_class!r}"
        )
    if reranker_class == "Qwen3Reranker":
        from presentation_dependence.rerankers.qwen3 import _build_qwen3_setwise_grade_skeleton_prefixes

        return _build_qwen3_setwise_grade_skeleton_prefixes(
            instruction, query_text, passages, max_doc_chars=max_doc_chars
        )
    if reranker_class == "Qwen3InstructGradeReranker":
        from presentation_dependence.rerankers.qwen3_instruct_grade import _build_qwen3_instruct_grade_skeleton

        return _build_qwen3_instruct_grade_skeleton(
            tokenizer,
            instruction,
            query_text,
            passages,
            max_doc_chars=max_doc_chars,
            chat_template_kwargs=_chat_template_kwargs_for_tokenizer(tokenizer),
            dummy_grade=getattr(tokenizer, "_slm_grade_skeleton_dummy", "0"),
            grade_rubric_id=_grade_rubric_id_for_tokenizer(tokenizer),
            slot_labels_override=slot_labels,
        )
    if reranker_class == "Gemma4GradeReranker":
        from presentation_dependence.rerankers.gemma4 import _build_gemma4_grade_skeleton_prefixes

        return _build_gemma4_grade_skeleton_prefixes(
            tokenizer,
            instruction,
            query_text,
            passages,
            max_doc_chars=max_doc_chars,
            chat_template_kwargs=_chat_template_kwargs_for_tokenizer(tokenizer),
            dummy_grade=getattr(tokenizer, "_slm_grade_skeleton_dummy", "0"),
            grade_rubric_id=_grade_rubric_id_for_tokenizer(tokenizer),
        )
    if reranker_class == "Granite41GradeReranker":
        from presentation_dependence.rerankers.granite_41 import _build_granite_grade_skeleton_prefixes

        return _build_granite_grade_skeleton_prefixes(
            tokenizer,
            instruction,
            query_text,
            passages,
            max_doc_chars=max_doc_chars,
            prefix_mode="cumulative",
            dummy_grade="0",
            grade_rubric_id=_grade_rubric_id_for_tokenizer(tokenizer),
        )
    raise ValueError(f"Unsupported student.reranker_class={reranker_class!r}")


def compute_continuous_scores_hf(
    *,
    model: Any,
    tokenizer: Any,
    prefixes: list[str],
    grade_token_ids: list[int],
    max_length: int,
    token_cache: StudentTokenCache | None = None,
) -> Any:
    """Differentiable HF continuous readout for a list of slot prefixes."""
    import torch

    prev_trunc = getattr(tokenizer, "truncation_side", None)
    if prev_trunc is not None:
        tokenizer.truncation_side = "left"
    try:
        if token_cache is None:
            encoded = [
                tokenizer(
                    prefix,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=max_length,
                )["input_ids"]
                for prefix in prefixes
            ]
        else:
            encoded = [token_cache.encode_prefix(tokenizer, prefix, max_length=max_length) for prefix in prefixes]
    finally:
        if prev_trunc is not None:
            tokenizer.truncation_side = prev_trunc

    inputs = tokenizer.pad(
        [{"input_ids": ids, "attention_mask": [1] * len(ids)} for ids in encoded],
        padding=True,
        pad_to_multiple_of=8,
        return_tensors="pt",
    )
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    try:
        logits = model(**inputs, logits_to_keep=1).logits
        final_logits = logits[:, -1, :]
    except TypeError:
        logits = model(**inputs).logits
        last_positions = last_real_token_positions(
            inputs,
            padding_side=getattr(tokenizer, "padding_side", "right"),
            device=logits.device,
        )
        row_idx = torch.arange(logits.shape[0], device=logits.device)
        final_logits = logits[row_idx, last_positions, :]
    return expected_grade(final_logits, grade_token_ids)


def compute_grade_logits_hf(
    *,
    model: Any,
    tokenizer: Any,
    prefixes: list[str],
    grade_token_ids: list[int],
    max_length: int,
    token_cache: StudentTokenCache | None = None,
) -> Any:
    """Differentiable HF grade-logit readout for a list of slot prefixes."""
    import torch

    prev_trunc = getattr(tokenizer, "truncation_side", None)
    if prev_trunc is not None:
        tokenizer.truncation_side = "left"
    try:
        if token_cache is None:
            encoded = [
                tokenizer(
                    prefix,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=max_length,
                )["input_ids"]
                for prefix in prefixes
            ]
        else:
            encoded = [token_cache.encode_prefix(tokenizer, prefix, max_length=max_length) for prefix in prefixes]
    finally:
        if prev_trunc is not None:
            tokenizer.truncation_side = prev_trunc

    inputs = tokenizer.pad(
        [{"input_ids": ids, "attention_mask": [1] * len(ids)} for ids in encoded],
        padding=True,
        pad_to_multiple_of=8,
        return_tensors="pt",
    )
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    try:
        logits = model(**inputs, logits_to_keep=1).logits
        final_logits = logits[:, -1, :]
    except TypeError:
        logits = model(**inputs).logits
        last_positions = last_real_token_positions(
            inputs,
            padding_side=getattr(tokenizer, "padding_side", "right"),
            device=logits.device,
        )
        row_idx = torch.arange(logits.shape[0], device=logits.device)
        final_logits = logits[row_idx, last_positions, :]
    ids = torch.as_tensor(list(grade_token_ids), dtype=torch.long, device=final_logits.device)
    return final_logits.index_select(-1, ids)


def build_eval_provenance(
    *,
    config: StudentSFTConfig,
    grade_token_ids: list[int],
    examples: list[GroupedSilverExample],
    device: str,
) -> dict[str, Any]:
    """Describe the in-training eval cohort and scoring environment.

    The cohort hash is order-invariant over the selected qid set so provenance
    changes only when the actual evaluated queries change, not when an upstream
    loader yields the same set in a different order.
    """
    qids = sorted(str(ex.query_id) for ex in examples)
    if config.eval_shuffle:
        random.Random(int(config.eval_seed)).shuffle(qids)
    if config.eval_max_queries and config.eval_max_queries > 0:
        selected = qids[: int(config.eval_max_queries)]
    else:
        selected = qids
    digest = hashlib.sha256("\n".join(sorted(selected)).encode("utf-8")).hexdigest()

    def _version(package: str) -> str | None:
        try:
            from importlib.metadata import version

            return version(package)
        except Exception:
            return None

    return {
        "model": {
            "model_name": config.model_name,
            "reranker_class": config.reranker_class,
        },
        "scoring": {
            "dtype": config.dtype,
            "device": str(device),
            "grade_token_ids": [int(x) for x in grade_token_ids],
            "grade_rubric_id": config.grade_rubric_id,
            "max_length": config.max_length,
            "max_doc_chars": config.max_doc_chars,
        },
        "eval_cohort": {
            "eval_query_count": len(selected),
            "eval_query_ids_sha256": digest,
            "eval_max_queries": config.eval_max_queries,
            "eval_shuffle": config.eval_shuffle,
            "eval_seed": config.eval_seed,
        },
        "topology": {
            "world_size": 1,
            "device": str(device),
        },
        "env": {
            "torch_version": _version("torch"),
            "transformers_version": _version("transformers"),
        },
    }


def _teacher_grade_vectors_tensor(
    vectors: list[list[float]],
    *,
    n_grades: int,
    dtype: Any,
    device: Any,
    field_name: str = "score_grade_vector",
) -> Any:
    """Validate and tensorize teacher P(g) vectors for vector losses."""
    import torch

    if not vectors:
        raise ValueError(f"{field_name} is required for student.objective.loss using grade vectors")
    for idx, vector in enumerate(vectors):
        if len(vector) != n_grades:
            raise ValueError(f"{field_name}[{idx}] length {len(vector)} != number of grade tokens {n_grades}")
        total = sum(float(x) for x in vector)
        if not math.isfinite(total) or abs(total - 1.0) > 1e-3:
            raise ValueError(f"{field_name}[{idx}] must sum to 1.0, got {total:.8f}")
        bad = [float(x) for x in vector if float(x) < -1e-6 or float(x) > 1.0 + 1e-6]
        if bad:
            raise ValueError(f"{field_name}[{idx}] contains invalid probability {bad[0]!r}")
    return torch.as_tensor(vectors, dtype=dtype, device=device)


def _ips_weights_for_chunk(chunk: RegressionChunk, ips_weights: list[float]) -> list[float]:
    """Map a chunk's documents to their DebiasFirst slot weights.

    Keyed on ``canonical_slots`` so a document keeps the weight of its
    first-stage slot no matter which position the augmentation shuffle presents
    it at. Chunks built before that field existed fall back to presentation
    order, which is the same thing for unshuffled chunks.
    """
    slots = chunk.canonical_slots
    if slots is None:
        slots = list(range(len(chunk.target_scores)))
    if len(slots) != len(chunk.target_scores):
        raise RuntimeError(
            f"canonical_slots has {len(slots)} entries but {len(chunk.target_scores)} "
            f"targets for qid={chunk.query_id} chunk_id={chunk.chunk_id}"
        )
    out: list[float] = []
    for slot in slots:
        if not 0 <= int(slot) < len(ips_weights):
            raise ValueError(
                f"canonical slot {slot} is outside the {len(ips_weights)}-slot IPS weight "
                f"vector for qid={chunk.query_id}; re-estimate it at this chunk_size"
            )
        out.append(float(ips_weights[int(slot)]))
    return out


def _supervised_loss_terms_from_grade_logits(
    grade_logits: Any,
    *,
    target_scores: list[float],
    target_grade_vectors: list[list[float]],
    objective_loss: str,
    alpha: float,
    slot_weights: list[float] | None = None,
) -> dict[str, Any]:
    """Return summed scalar/vector supervised losses for one slot batch.

    ``slot_weights`` is the DebiasFirst inverse-propensity multiplier per slot,
    aligned to ``target_scores``. It scales the squared errors and returns the
    summed weight as ``weight_sum`` so the caller can normalize by total weight
    rather than slot count; all-ones weights reproduce the unweighted loss.
    """
    import torch

    if objective_loss not in {"mse", "kl_vector", "combined"}:
        raise ValueError(f"Unsupported objective_loss={objective_loss!r}")
    n_slots = int(grade_logits.shape[0])
    if n_slots != len(target_scores):
        raise RuntimeError(f"got {n_slots} logits rows for {len(target_scores)} scalar targets")
    if slot_weights is not None and len(slot_weights) != n_slots:
        raise RuntimeError(f"got {len(slot_weights)} slot weights for {n_slots} logits rows")

    probs = torch.nn.functional.softmax(grade_logits, dim=-1)
    values = torch.arange(grade_logits.shape[-1], dtype=probs.dtype, device=probs.device)
    preds = (probs * values).sum(dim=-1)
    target_t = torch.as_tensor(target_scores, dtype=preds.dtype, device=preds.device)
    if slot_weights is None:
        mse_sum = torch.nn.functional.mse_loss(preds, target_t, reduction="sum")
        weight_sum = float(n_slots)
    else:
        weights_t = torch.as_tensor(slot_weights, dtype=preds.dtype, device=preds.device)
        mse_sum = (torch.nn.functional.mse_loss(preds, target_t, reduction="none") * weights_t).sum()
        weight_sum = float(weights_t.sum().detach().cpu())

    if objective_loss == "mse":
        return {"loss_sum": mse_sum, "mse_sum": mse_sum, "kl_sum": None, "weight_sum": weight_sum}

    teacher_p = _teacher_grade_vectors_tensor(
        target_grade_vectors,
        n_grades=int(grade_logits.shape[-1]),
        dtype=grade_logits.dtype,
        device=grade_logits.device,
    )
    kl_sum = torch.nn.functional.kl_div(
        torch.nn.functional.log_softmax(grade_logits, dim=-1),
        teacher_p,
        reduction="sum",
    )
    if objective_loss == "kl_vector":
        loss_sum = kl_sum
    else:
        loss_sum = mse_sum + float(alpha) * kl_sum
    return {"loss_sum": loss_sum, "mse_sum": mse_sum, "kl_sum": kl_sum, "weight_sum": weight_sum}


def mse_loss_for_chunks(
    *,
    model: Any,
    tokenizer: Any,
    chunks: list[RegressionChunk],
    grade_token_ids: list[int],
    reranker_class: str,
    instruction: str | None,
    max_length: int,
    max_doc_chars: int,
    token_cache: StudentTokenCache | None = None,
) -> Any:
    """Compute differentiable MSE for a batch of regression chunks."""
    import torch

    prefixes: list[str] = []
    targets: list[float] = []
    for chunk in chunks:
        if token_cache is None:
            pfx = build_prefixes_for_chunk(
                reranker_class=reranker_class,
                tokenizer=tokenizer,
                instruction=instruction,
                query_text=chunk.query_text,
                passages=chunk.passages,
                max_doc_chars=max_doc_chars,
            )
        else:
            pfx = token_cache.get_prefixes(
                chunk=chunk,
                reranker_class=reranker_class,
                instruction=instruction,
                tokenizer=tokenizer,
                max_doc_chars=max_doc_chars,
            )
        if len(pfx) != len(chunk.target_scores):
            raise RuntimeError(
                f"prefix count {len(pfx)} != target count {len(chunk.target_scores)} for qid={chunk.query_id}"
            )
        prefixes.extend(pfx)
        targets.extend(float(x) for x in chunk.target_scores)

    preds = compute_continuous_scores_hf(
        model=model,
        tokenizer=tokenizer,
        prefixes=prefixes,
        grade_token_ids=grade_token_ids,
        max_length=max_length,
        token_cache=token_cache,
    )
    target_t = torch.as_tensor(targets, dtype=preds.dtype, device=preds.device)
    return torch.nn.functional.mse_loss(preds, target_t)


def backward_mse_loss_for_chunks_single_forward(
    *,
    model: Any,
    tokenizer: Any,
    chunks: list[RegressionChunk],
    grade_token_ids: list[int],
    reranker_class: str,
    instruction: str | None,
    max_length: int,
    max_doc_chars: int,
    backward_scale: float,
    ips_weights: list[float] | None = None,
) -> float:
    """Single-forward variant of :func:`backward_mse_loss_for_chunks`.

    For each chunk, tokenizes the full prompt once, runs one forward,
    gathers logits at all K slot positions via :func:`compute_continuous_scores_single_forward`,
    computes per-slot MSE against the silver targets, then accumulates a
    single backward call. Memory profile: one autograd graph per chunk
    instead of K — combined with ``gradient_checkpointing`` this is at
    most the memory of the legacy path with ``slot_forward_batch_size=1``.

    Loss-scaling semantics are identical to the legacy path: the chunk's
    summed MSE is divided by the total slot count across all chunks in
    the optimizer microbatch so the gradient magnitude matches
    ``F.mse_loss(...)`` averaged-then-grad-accumulated.

    Chunks are processed one-at-a-time (single chunk per forward call).
    That intentionally trades intra-step batching for predictable peak
    memory — the per-chunk activation footprint is well-understood from
    smoke matrix work; batching multiple chunks together would inflate
    peak activation memory and require a fresh memory smoke.

    ``ips_weights`` is the DebiasFirst per-slot multiplier keyed on canonical
    first-stage slot; when set, the normalizer becomes total weight instead of
    slot count, so a mean-1 vector preserves the loss scale.
    """
    import torch

    if not chunks:
        raise ValueError("Cannot train on an empty chunk batch")

    chunk_specs: list[tuple[str, list[str]]] = []
    targets: list[float] = []
    slot_weights: list[float] | None = [] if ips_weights is not None else None
    for chunk in chunks:
        full_prompt, slot_prefixes = build_full_prompt_and_prefixes(
            reranker_class=reranker_class,
            tokenizer=tokenizer,
            instruction=instruction,
            query_text=chunk.query_text,
            passages=chunk.passages,
            max_doc_chars=max_doc_chars,
        )
        if len(slot_prefixes) != len(chunk.target_scores):
            raise RuntimeError(
                f"prefix count {len(slot_prefixes)} != target count {len(chunk.target_scores)} for qid={chunk.query_id}"
            )
        chunk_specs.append((full_prompt, slot_prefixes))
        targets.extend(float(x) for x in chunk.target_scores)
        if slot_weights is not None:
            slot_weights.extend(_ips_weights_for_chunk(chunk, ips_weights))

    total_slots = sum(len(prefs) for _, prefs in chunk_specs)
    denominator = float(sum(slot_weights)) if slot_weights is not None else float(total_slots)
    if denominator <= 0.0:
        raise ValueError("total IPS slot weight for this chunk batch is not positive")
    total_loss_sum = 0.0
    target_idx = 0
    for chunk_spec in chunk_specs:
        n_slots = len(chunk_spec[1])
        preds = compute_continuous_scores_single_forward(
            model=model,
            tokenizer=tokenizer,
            chunk_specs=[chunk_spec],
            grade_token_ids=grade_token_ids,
            max_length=max_length,
        )
        target_values = targets[target_idx : target_idx + n_slots]
        weight_values = slot_weights[target_idx : target_idx + n_slots] if slot_weights is not None else None
        target_idx += n_slots
        target_t = torch.as_tensor(target_values, dtype=preds.dtype, device=preds.device)
        if weight_values is None:
            loss_sum = torch.nn.functional.mse_loss(preds, target_t, reduction="sum")
        else:
            weights_t = torch.as_tensor(weight_values, dtype=preds.dtype, device=preds.device)
            loss_sum = (torch.nn.functional.mse_loss(preds, target_t, reduction="none") * weights_t).sum()
        total_loss_sum += float(loss_sum.detach().cpu())
        (loss_sum * (float(backward_scale) / denominator)).backward()
    return total_loss_sum / denominator


def backward_supervised_loss_for_chunks_single_forward(
    *,
    model: Any,
    tokenizer: Any,
    chunks: list[RegressionChunk],
    grade_token_ids: list[int],
    reranker_class: str,
    instruction: str | None,
    max_length: int,
    max_doc_chars: int,
    backward_scale: float,
    objective_loss: str,
    alpha: float = 1.0,
    ips_weights: list[float] | None = None,
) -> dict[str, float]:
    """Backpropagate MSE, vector KL, or their weighted combination.

    ``ips_weights`` maps a canonical within-chunk slot to its DebiasFirst
    inverse-propensity multiplier. When set, the loss is normalized by total
    weight rather than slot count, so a mean-1 weight vector leaves the loss
    scale unchanged and only shifts relative emphasis across slots.
    """
    if not chunks:
        raise ValueError("Cannot train on an empty chunk batch")

    chunk_specs: list[tuple[str, list[str]]] = []
    targets: list[float] = []
    target_vectors: list[list[float]] = []
    slot_weights: list[float] | None = [] if ips_weights is not None else None
    for chunk in chunks:
        full_prompt, slot_prefixes = build_full_prompt_and_prefixes(
            reranker_class=reranker_class,
            tokenizer=tokenizer,
            instruction=instruction,
            query_text=chunk.query_text,
            passages=chunk.passages,
            max_doc_chars=max_doc_chars,
        )
        if len(slot_prefixes) != len(chunk.target_scores):
            raise RuntimeError(
                f"prefix count {len(slot_prefixes)} != target count {len(chunk.target_scores)} for qid={chunk.query_id}"
            )
        chunk_specs.append((full_prompt, slot_prefixes))
        targets.extend(float(x) for x in chunk.target_scores)
        target_vectors.extend([list(v) for v in chunk.target_grade_vectors])
        if slot_weights is not None:
            slot_weights.extend(_ips_weights_for_chunk(chunk, ips_weights))

    total_slots = sum(len(prefs) for _, prefs in chunk_specs)
    # The weights are data-only, so the normalizer is known before any forward
    # pass. Each chunk's backward must divide by the batch-wide denominator.
    denominator = float(sum(slot_weights)) if slot_weights is not None else float(total_slots)
    if denominator <= 0.0:
        raise ValueError("total IPS slot weight for this chunk batch is not positive")
    total_loss_sum = 0.0
    total_mse_sum = 0.0
    total_kl_sum = 0.0
    target_idx = 0
    for chunk_spec in chunk_specs:
        n_slots = len(chunk_spec[1])
        grade_logits = compute_grade_logits_single_forward(
            model=model,
            tokenizer=tokenizer,
            chunk_specs=[chunk_spec],
            grade_token_ids=grade_token_ids,
            max_length=max_length,
        )
        target_values = targets[target_idx : target_idx + n_slots]
        vector_values = target_vectors[target_idx : target_idx + n_slots]
        weight_values = slot_weights[target_idx : target_idx + n_slots] if slot_weights is not None else None
        target_idx += n_slots
        terms = _supervised_loss_terms_from_grade_logits(
            grade_logits,
            target_scores=target_values,
            target_grade_vectors=vector_values,
            objective_loss=objective_loss,
            alpha=alpha,
            slot_weights=weight_values,
        )
        loss_sum = terms["loss_sum"]
        mse_sum = terms["mse_sum"]
        kl_sum = terms["kl_sum"]
        total_loss_sum += float(loss_sum.detach().cpu())
        total_mse_sum += float(mse_sum.detach().cpu())
        if kl_sum is not None:
            total_kl_sum += float(kl_sum.detach().cpu())
        (loss_sum * (float(backward_scale) / denominator)).backward()
    return {
        "loss": total_loss_sum / denominator,
        "mse_loss": total_mse_sum / denominator,
        "kl_vector_loss": total_kl_sum / denominator if objective_loss in {"kl_vector", "combined"} else 0.0,
        "objective_alpha": float(alpha),
    }


def backward_mse_loss_for_chunks(
    *,
    model: Any,
    tokenizer: Any,
    chunks: list[RegressionChunk],
    grade_token_ids: list[int],
    reranker_class: str,
    instruction: str | None,
    max_length: int,
    max_doc_chars: int,
    backward_scale: float,
    slot_forward_batch_size: int | None = None,
    token_cache: StudentTokenCache | None = None,
) -> float:
    """Backpropagate continuous-readout MSE, optionally microbatching slots.

    Slot microbatching preserves the B-doc chunk target but avoids holding the
    activation graph for all B slot-prefix forwards at once.
    """
    import torch

    prefixes: list[str] = []
    targets: list[float] = []
    for chunk in chunks:
        if token_cache is None:
            pfx = build_prefixes_for_chunk(
                reranker_class=reranker_class,
                tokenizer=tokenizer,
                instruction=instruction,
                query_text=chunk.query_text,
                passages=chunk.passages,
                max_doc_chars=max_doc_chars,
            )
        else:
            pfx = token_cache.get_prefixes(
                chunk=chunk,
                reranker_class=reranker_class,
                instruction=instruction,
                tokenizer=tokenizer,
                max_doc_chars=max_doc_chars,
            )
        if len(pfx) != len(chunk.target_scores):
            raise RuntimeError(
                f"prefix count {len(pfx)} != target count {len(chunk.target_scores)} for qid={chunk.query_id}"
            )
        prefixes.extend(pfx)
        targets.extend(float(x) for x in chunk.target_scores)

    if not prefixes:
        raise ValueError("Cannot train on an empty chunk batch")

    micro = int(slot_forward_batch_size or len(prefixes))
    micro = max(1, min(micro, len(prefixes)))
    total_loss_sum = 0.0
    total_slots = len(prefixes)
    for start in range(0, total_slots, micro):
        pfx = prefixes[start : start + micro]
        target_values = targets[start : start + micro]
        preds = compute_continuous_scores_hf(
            model=model,
            tokenizer=tokenizer,
            prefixes=pfx,
            grade_token_ids=grade_token_ids,
            max_length=max_length,
            token_cache=token_cache,
        )
        target_t = torch.as_tensor(target_values, dtype=preds.dtype, device=preds.device)
        loss_sum = torch.nn.functional.mse_loss(preds, target_t, reduction="sum")
        total_loss_sum += float(loss_sum.detach().cpu())
        (loss_sum * (float(backward_scale) / total_slots)).backward()
    return total_loss_sum / total_slots


def backward_supervised_loss_for_chunks(
    *,
    model: Any,
    tokenizer: Any,
    chunks: list[RegressionChunk],
    grade_token_ids: list[int],
    reranker_class: str,
    instruction: str | None,
    max_length: int,
    max_doc_chars: int,
    backward_scale: float,
    slot_forward_batch_size: int | None = None,
    token_cache: StudentTokenCache | None = None,
    objective_loss: str,
    alpha: float = 1.0,
) -> dict[str, float]:
    """Backpropagate supervised scalar/vector loss, optionally microbatching slots."""
    prefixes: list[str] = []
    targets: list[float] = []
    target_vectors: list[list[float]] = []
    for chunk in chunks:
        if token_cache is None:
            pfx = build_prefixes_for_chunk(
                reranker_class=reranker_class,
                tokenizer=tokenizer,
                instruction=instruction,
                query_text=chunk.query_text,
                passages=chunk.passages,
                max_doc_chars=max_doc_chars,
            )
        else:
            pfx = token_cache.get_prefixes(
                chunk=chunk,
                reranker_class=reranker_class,
                instruction=instruction,
                tokenizer=tokenizer,
                max_doc_chars=max_doc_chars,
            )
        if len(pfx) != len(chunk.target_scores):
            raise RuntimeError(
                f"prefix count {len(pfx)} != target count {len(chunk.target_scores)} for qid={chunk.query_id}"
            )
        prefixes.extend(pfx)
        targets.extend(float(x) for x in chunk.target_scores)
        target_vectors.extend([list(v) for v in chunk.target_grade_vectors])

    if not prefixes:
        raise ValueError("Cannot train on an empty chunk batch")

    micro = int(slot_forward_batch_size or len(prefixes))
    micro = max(1, min(micro, len(prefixes)))
    total_loss_sum = 0.0
    total_mse_sum = 0.0
    total_kl_sum = 0.0
    total_slots = len(prefixes)
    for start in range(0, total_slots, micro):
        pfx = prefixes[start : start + micro]
        target_values = targets[start : start + micro]
        vector_values = target_vectors[start : start + micro]
        grade_logits = compute_grade_logits_hf(
            model=model,
            tokenizer=tokenizer,
            prefixes=pfx,
            grade_token_ids=grade_token_ids,
            max_length=max_length,
            token_cache=token_cache,
        )
        terms = _supervised_loss_terms_from_grade_logits(
            grade_logits,
            target_scores=target_values,
            target_grade_vectors=vector_values,
            objective_loss=objective_loss,
            alpha=alpha,
        )
        loss_sum = terms["loss_sum"]
        mse_sum = terms["mse_sum"]
        kl_sum = terms["kl_sum"]
        total_loss_sum += float(loss_sum.detach().cpu())
        total_mse_sum += float(mse_sum.detach().cpu())
        if kl_sum is not None:
            total_kl_sum += float(kl_sum.detach().cpu())
        (loss_sum * (float(backward_scale) / total_slots)).backward()
    return {
        "loss": total_loss_sum / total_slots,
        "mse_loss": total_mse_sum / total_slots,
        "kl_vector_loss": total_kl_sum / total_slots if objective_loss in {"kl_vector", "combined"} else 0.0,
        "objective_alpha": float(alpha),
    }


def _slot_labels_tuple(labels: list[str] | tuple[str, ...] | None) -> tuple[str, ...] | None:
    """Normalize a chunk's per-view ``slot_labels`` field to a tuple or ``None``."""
    if labels is None:
        return None
    return tuple(labels)


def _aligned_single_forward_scores(
    *,
    model: Any,
    tokenizer: Any,
    grade_token_ids: list[int],
    reranker_class: str,
    instruction: str | None,
    query_text: str,
    passages: list[dict[str, str]],
    view_doc_ids: list[str],
    canonical_doc_ids: list[str],
    max_length: int,
    max_doc_chars: int,
    slot_labels: tuple[str, ...] | None = None,
):
    """Score one permuted view and reorder predictions to canonical doc order."""
    import torch

    if set(view_doc_ids) != set(canonical_doc_ids):
        raise ValueError("consistency view doc ids do not match canonical doc ids")
    full_prompt, slot_prefixes = build_full_prompt_and_prefixes(
        reranker_class=reranker_class,
        tokenizer=tokenizer,
        instruction=instruction,
        query_text=query_text,
        passages=passages,
        max_doc_chars=max_doc_chars,
        slot_labels=slot_labels,
    )
    if len(slot_prefixes) != len(view_doc_ids):
        raise RuntimeError(f"prefix count {len(slot_prefixes)} != view doc count {len(view_doc_ids)}")
    scores = compute_continuous_scores_single_forward(
        model=model,
        tokenizer=tokenizer,
        chunk_specs=[(full_prompt, slot_prefixes)],
        grade_token_ids=grade_token_ids,
        max_length=max_length,
    )
    index_by_doc = {doc_id: idx for idx, doc_id in enumerate(view_doc_ids)}
    align_idx = torch.as_tensor(
        [index_by_doc[doc_id] for doc_id in canonical_doc_ids],
        dtype=torch.long,
        device=scores.device,
    )
    return scores.index_select(0, align_idx)


def _aligned_single_forward_grade_logits(
    *,
    model: Any,
    tokenizer: Any,
    grade_token_ids: list[int],
    reranker_class: str,
    instruction: str | None,
    query_text: str,
    passages: list[dict[str, str]],
    view_doc_ids: list[str],
    canonical_doc_ids: list[str],
    max_length: int,
    max_doc_chars: int,
    slot_labels: tuple[str, ...] | None = None,
):
    """Return grade logits for one permuted view aligned to canonical doc order."""
    import torch

    if set(view_doc_ids) != set(canonical_doc_ids):
        raise ValueError("consistency view doc ids do not match canonical doc ids")
    full_prompt, slot_prefixes = build_full_prompt_and_prefixes(
        reranker_class=reranker_class,
        tokenizer=tokenizer,
        instruction=instruction,
        query_text=query_text,
        passages=passages,
        max_doc_chars=max_doc_chars,
        slot_labels=slot_labels,
    )
    if len(slot_prefixes) != len(view_doc_ids):
        raise RuntimeError(f"prefix count {len(slot_prefixes)} != view doc count {len(view_doc_ids)}")
    grade_logits = compute_grade_logits_single_forward(
        model=model,
        tokenizer=tokenizer,
        chunk_specs=[(full_prompt, slot_prefixes)],
        grade_token_ids=grade_token_ids,
        max_length=max_length,
    )
    index_by_doc = {doc_id: idx for idx, doc_id in enumerate(view_doc_ids)}
    align_idx = torch.as_tensor(
        [index_by_doc[doc_id] for doc_id in canonical_doc_ids],
        dtype=torch.long,
        device=grade_logits.device,
    )
    return grade_logits.index_select(0, align_idx)


def _expected_scores_from_grade_logits(grade_logits: Any) -> Any:
    """Map grade-token logits to expected grade scores on the 0..G-1 scale."""
    import torch

    probs = torch.nn.functional.softmax(grade_logits, dim=-1)
    values = torch.arange(grade_logits.shape[-1], dtype=probs.dtype, device=probs.device)
    return (probs * values).sum(dim=-1)


def backward_supervised_consistency_loss_for_chunks_single_forward(
    *,
    model: Any,
    tokenizer: Any,
    chunks: list[SupervisedConsistencyChunk],
    grade_token_ids: list[int],
    reranker_class: str,
    instruction: str | None,
    max_length: int,
    max_doc_chars: int,
    consistency_lambda: float,
    backward_scale: float,
    objective_loss: str = "mse",
    alpha: float = 1.0,
) -> dict[str, float]:
    """Backpropagate supervised scalar/vector anchor plus two-view consistency."""
    import torch

    if not chunks:
        raise ValueError("Cannot train on an empty supervised consistency chunk batch")
    total_slots = sum(len(chunk.doc_ids) for chunk in chunks)
    if total_slots <= 0:
        raise ValueError("Cannot train on supervised consistency chunks with zero slots")

    total_sft_loss_sum = 0.0
    total_mse_loss_sum = 0.0
    total_kl_loss_sum = 0.0
    total_consistency_loss_sum = 0.0
    total_loss_sum = 0.0
    lam = float(consistency_lambda)
    alpha_f = float(alpha)

    for chunk in chunks:
        view_a_slot_labels = _slot_labels_tuple(getattr(chunk, "view_a_slot_labels", None))
        view_b_slot_labels = _slot_labels_tuple(getattr(chunk, "view_b_slot_labels", None))
        grade_logits_a = _aligned_single_forward_grade_logits(
            model=model,
            tokenizer=tokenizer,
            grade_token_ids=grade_token_ids,
            reranker_class=reranker_class,
            instruction=instruction,
            query_text=chunk.query_text,
            passages=chunk.view_a_passages,
            view_doc_ids=chunk.view_a_doc_ids,
            canonical_doc_ids=chunk.doc_ids,
            max_length=max_length,
            max_doc_chars=max_doc_chars,
            slot_labels=view_a_slot_labels,
        )
        preds_a = _expected_scores_from_grade_logits(grade_logits_a)
        preds_b = _aligned_single_forward_scores(
            model=model,
            tokenizer=tokenizer,
            grade_token_ids=grade_token_ids,
            reranker_class=reranker_class,
            instruction=instruction,
            query_text=chunk.query_text,
            passages=chunk.view_b_passages,
            view_doc_ids=chunk.view_b_doc_ids,
            canonical_doc_ids=chunk.doc_ids,
            max_length=max_length,
            max_doc_chars=max_doc_chars,
            slot_labels=view_b_slot_labels,
        )
        supervised_terms = _supervised_loss_terms_from_grade_logits(
            grade_logits_a,
            target_scores=[float(x) for x in chunk.target_scores],
            target_grade_vectors=[list(v) for v in chunk.target_grade_vectors],
            objective_loss=objective_loss,
            alpha=alpha_f,
        )
        sft_loss_sum = supervised_terms["loss_sum"]
        mse_loss_sum = supervised_terms["mse_sum"]
        kl_loss_sum = supervised_terms["kl_sum"]
        consistency_loss_sum = torch.nn.functional.mse_loss(preds_a, preds_b, reduction="sum")
        loss_sum = sft_loss_sum + lam * consistency_loss_sum
        total_sft_loss_sum += float(sft_loss_sum.detach().cpu())
        total_mse_loss_sum += float(mse_loss_sum.detach().cpu())
        if kl_loss_sum is not None:
            total_kl_loss_sum += float(kl_loss_sum.detach().cpu())
        total_consistency_loss_sum += float(consistency_loss_sum.detach().cpu())
        total_loss_sum += float(loss_sum.detach().cpu())
        (loss_sum * (float(backward_scale) / total_slots)).backward()

    return {
        "loss": total_loss_sum / total_slots,
        "sft_loss": total_sft_loss_sum / total_slots,
        "mse_loss": total_mse_loss_sum / total_slots,
        "kl_vector_loss": total_kl_loss_sum / total_slots if objective_loss in {"kl_vector", "combined"} else 0.0,
        "objective_alpha": alpha_f,
        "consistency_loss": total_consistency_loss_sum / total_slots,
        "consistency_lambda": lam,
    }


def _model_with_base_adapter_disabled(model: Any):
    """Return a context manager disabling PEFT adapters when available."""
    unwrapped = _unwrap_model(model)
    disable = getattr(unwrapped, "disable_adapter", None)
    if callable(disable):
        return disable()
    return contextlib.nullcontext()


def _kl_student_to_anchor_sum(student_logits: Any, anchor_logits: Any, *, temperature: float) -> Any:
    """Summed KL(student || anchor) over grade-token distributions."""
    import torch

    temp = float(temperature)
    log_p_student = torch.nn.functional.log_softmax(student_logits / temp, dim=-1)
    log_p_anchor = torch.nn.functional.log_softmax(anchor_logits / temp, dim=-1)
    p_student = log_p_student.exp()
    return (p_student * (log_p_student - log_p_anchor)).sum()


def backward_kl_to_base_loss_for_chunks_single_forward(
    *,
    model: Any,
    tokenizer: Any,
    chunks: list[RegressionChunk],
    grade_token_ids: list[int],
    reranker_class: str,
    instruction: str | None,
    max_length: int,
    max_doc_chars: int,
    beta: float,
    temperature: float,
    backward_scale: float,
) -> dict[str, float]:
    """Backpropagate K=1 SFT plus KL(student || frozen base) over grade logits."""
    import torch

    if not chunks:
        raise ValueError("Cannot train on an empty KL-to-base chunk batch")
    total_slots = sum(len(chunk.target_scores) for chunk in chunks)
    if total_slots <= 0:
        raise ValueError("Cannot train on KL-to-base chunks with zero slots")

    total_sft_loss_sum = 0.0
    total_kl_loss_sum = 0.0
    total_loss_sum = 0.0
    beta_f = float(beta)
    temp_f = float(temperature)

    for chunk in chunks:
        full_prompt, slot_prefixes = build_full_prompt_and_prefixes(
            reranker_class=reranker_class,
            tokenizer=tokenizer,
            instruction=instruction,
            query_text=chunk.query_text,
            passages=chunk.passages,
            max_doc_chars=max_doc_chars,
        )
        if len(slot_prefixes) != len(chunk.target_scores):
            raise RuntimeError(
                f"prefix count {len(slot_prefixes)} != target count {len(chunk.target_scores)} for qid={chunk.query_id}"
            )
        chunk_spec = (full_prompt, slot_prefixes)
        with torch.no_grad(), _model_with_base_adapter_disabled(model):
            anchor_logits = compute_grade_logits_single_forward(
                model=model,
                tokenizer=tokenizer,
                chunk_specs=[chunk_spec],
                grade_token_ids=grade_token_ids,
                max_length=max_length,
            )
        student_logits = compute_grade_logits_single_forward(
            model=model,
            tokenizer=tokenizer,
            chunk_specs=[chunk_spec],
            grade_token_ids=grade_token_ids,
            max_length=max_length,
        )
        probs = torch.nn.functional.softmax(student_logits, dim=-1)
        values = torch.arange(student_logits.shape[-1], dtype=probs.dtype, device=probs.device)
        preds = (probs * values).sum(dim=-1)
        target_t = torch.as_tensor(chunk.target_scores, dtype=preds.dtype, device=preds.device)
        sft_loss_sum = torch.nn.functional.mse_loss(preds, target_t, reduction="sum")
        kl_loss_sum = _kl_student_to_anchor_sum(student_logits, anchor_logits, temperature=temp_f)
        loss_sum = sft_loss_sum + beta_f * kl_loss_sum
        total_sft_loss_sum += float(sft_loss_sum.detach().cpu())
        total_kl_loss_sum += float(kl_loss_sum.detach().cpu())
        total_loss_sum += float(loss_sum.detach().cpu())
        (loss_sum * (float(backward_scale) / total_slots)).backward()

    return {
        "loss": total_loss_sum / total_slots,
        "sft_loss": total_sft_loss_sum / total_slots,
        "kl_anchor_loss": total_kl_loss_sum / total_slots,
        "kl_anchor_beta": beta_f,
        "kl_anchor_temperature": temp_f,
    }


class TrainableParameterEMA:
    """EMA shadow for trainable parameters, intended for LoRA adapter weights."""

    def __init__(self, model: Any, *, decay: float) -> None:
        self.decay = float(decay)
        self.shadow: dict[str, Any] = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if getattr(param, "requires_grad", False)
        }
        if not self.shadow:
            raise ValueError("EMA mean-teacher objective requires at least one trainable parameter")

    def update(self, model: Any) -> None:
        with _torch_no_grad():
            for name, param in model.named_parameters():
                if name not in self.shadow:
                    continue
                self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    def state_dict(self) -> dict[str, Any]:
        return {"decay": self.decay, "shadow": {name: tensor.detach().cpu() for name, tensor in self.shadow.items()}}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        loaded = state.get("shadow", state)
        if not isinstance(loaded, dict):
            raise ValueError("Invalid EMA state: expected a shadow-parameter mapping")
        missing = sorted(set(self.shadow) - set(loaded))
        if missing:
            raise ValueError(f"EMA checkpoint is missing {len(missing)} trainable parameter(s): {missing[:3]}")
        for name, tensor in loaded.items():
            if name in self.shadow:
                self.shadow[name].copy_(tensor.to(device=self.shadow[name].device, dtype=self.shadow[name].dtype))

    def apply_to_model(self, model: Any):
        return _ApplyEMAWeights(model, self.shadow)


class _ApplyEMAWeights:
    def __init__(self, model: Any, shadow: dict[str, Any]) -> None:
        self.model = model
        self.shadow = shadow
        self.live: dict[str, Any] = {}

    def __enter__(self) -> None:
        for name, param in self.model.named_parameters():
            if name not in self.shadow:
                continue
            self.live[name] = param.detach().clone()
            param.data.copy_(self.shadow[name].to(device=param.device, dtype=param.dtype))

    def __exit__(self, exc_type, exc, tb) -> None:
        for name, param in self.model.named_parameters():
            if name in self.live:
                param.data.copy_(self.live[name].to(device=param.device, dtype=param.dtype))
        self.live.clear()


def _torch_no_grad():
    import torch

    return torch.no_grad()


def backward_mean_teacher_loss_for_chunks_single_forward(
    *,
    model: Any,
    ema: TrainableParameterEMA,
    tokenizer: Any,
    chunks: list[SupervisedConsistencyChunk],
    grade_token_ids: list[int],
    reranker_class: str,
    instruction: str | None,
    max_length: int,
    max_doc_chars: int,
    alpha: float,
    backward_scale: float,
) -> dict[str, float]:
    """Backpropagate K=1 SFT plus EMA-teacher prediction consistency."""
    import torch

    if not chunks:
        raise ValueError("Cannot train on an empty mean-teacher chunk batch")
    total_slots = sum(len(chunk.doc_ids) for chunk in chunks)
    if total_slots <= 0:
        raise ValueError("Cannot train on mean-teacher chunks with zero slots")

    total_sft_loss_sum = 0.0
    total_consistency_loss_sum = 0.0
    total_loss_sum = 0.0
    alpha_f = float(alpha)

    for chunk in chunks:
        with torch.no_grad(), ema.apply_to_model(model):
            teacher_b = _aligned_single_forward_scores(
                model=model,
                tokenizer=tokenizer,
                grade_token_ids=grade_token_ids,
                reranker_class=reranker_class,
                instruction=instruction,
                query_text=chunk.query_text,
                passages=chunk.view_b_passages,
                view_doc_ids=chunk.view_b_doc_ids,
                canonical_doc_ids=chunk.doc_ids,
                max_length=max_length,
                max_doc_chars=max_doc_chars,
            )
        preds_a = _aligned_single_forward_scores(
            model=model,
            tokenizer=tokenizer,
            grade_token_ids=grade_token_ids,
            reranker_class=reranker_class,
            instruction=instruction,
            query_text=chunk.query_text,
            passages=chunk.view_a_passages,
            view_doc_ids=chunk.view_a_doc_ids,
            canonical_doc_ids=chunk.doc_ids,
            max_length=max_length,
            max_doc_chars=max_doc_chars,
        )
        target_t = torch.as_tensor(chunk.target_scores, dtype=preds_a.dtype, device=preds_a.device)
        teacher_t = teacher_b.to(dtype=preds_a.dtype, device=preds_a.device)
        sft_loss_sum = torch.nn.functional.mse_loss(preds_a, target_t, reduction="sum")
        consistency_loss_sum = torch.nn.functional.mse_loss(preds_a, teacher_t, reduction="sum")
        loss_sum = sft_loss_sum + alpha_f * consistency_loss_sum
        total_sft_loss_sum += float(sft_loss_sum.detach().cpu())
        total_consistency_loss_sum += float(consistency_loss_sum.detach().cpu())
        total_loss_sum += float(loss_sum.detach().cpu())
        (loss_sum * (float(backward_scale) / total_slots)).backward()

    return {
        "loss": total_loss_sum / total_slots,
        "sft_loss": total_sft_loss_sum / total_slots,
        "mean_teacher_consistency_loss": total_consistency_loss_sum / total_slots,
        "mean_teacher_alpha": alpha_f,
        "ema_decay": ema.decay,
    }


def _int_list(values: list[int], *, field_name: str, minimum: int) -> list[int]:
    """Validate augmentation integer lists from YAML."""
    out = [int(value) for value in values]
    bad = [value for value in out if value < minimum]
    if bad:
        raise ValueError(f"{field_name} values must be integers >= {minimum}, got {bad}")
    return out


def _augmentation_chunk_sizes(config: StudentSFTConfig) -> list[int]:
    """Return the B values used by robustness SFT augmentation."""
    sizes = config.permutation_augmentation_chunk_sizes or [config.chunk_size]
    return _int_list(
        sizes,
        field_name="student.data.permutation_augmentation.chunk_sizes",
        minimum=1,
    )


def _augmentation_view_specs(config: StudentSFTConfig, *, epoch: int) -> list[tuple[int | None, int]]:
    """Return ``(permutation_seed, chunk_size)`` specs for one training epoch.

    Minimal robustness SFT is one shuffled B=chunk_size view per epoch. The same
    doc-level targets are reused; only order and optional repartitioning change.
    """
    if not config.permutation_augmentation_enabled:
        return [(None, int(config.chunk_size))]
    seeds = _int_list(
        config.permutation_augmentation_seeds or [config.seed],
        field_name="student.data.permutation_augmentation.seeds",
        minimum=0,
    )
    chunk_sizes = _augmentation_chunk_sizes(config)
    views = max(int(config.permutation_augmentation_views_per_epoch), 1)
    start = int(epoch)
    return [
        (
            int(seeds[(start + view_idx) % len(seeds)]),
            int(chunk_sizes[(start + view_idx) % len(chunk_sizes)]),
        )
        for view_idx in range(views)
    ]


def _augmentation_shuffle_rate(config: StudentSFTConfig) -> float:
    """Return a clamped query-level shuffle probability."""
    rate = float(config.permutation_augmentation_shuffle_rate)
    if math.isnan(rate):
        raise ValueError("student.data.permutation_augmentation.shuffle_rate must be finite")
    return min(max(rate, 0.0), 1.0)


def _should_shuffle_example(
    example: GroupedSilverExample,
    *,
    config: StudentSFTConfig,
    epoch: int,
    view_idx: int,
    permutation_seed: int,
) -> bool:
    """Deterministically decide whether this query uses the shuffled view."""
    rate = _augmentation_shuffle_rate(config)
    if rate >= 1.0:
        return True
    if rate <= 0.0:
        return False
    key = f"{config.seed}:{epoch}:{view_idx}:{permutation_seed}:{example.query_id}"
    digest = hashlib.sha1(key.encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") / float(1 << 64)
    return bucket < rate


def _annotate_chunks_with_doc_ids(chunks: list[RegressionChunk]) -> None:
    """Prefix passage text with stable doc IDs for doc_id -> score SFT prompts."""
    for chunk in chunks:
        _annotate_passages_with_doc_ids(chunk.passages, chunk.doc_ids)


def _annotate_passages_with_doc_ids(passages: list[dict[str, str]], doc_ids: list[str]) -> None:
    """Prefix passage text with stable doc IDs for doc_id -> score prompts."""
    for passage, doc_id in zip(passages, doc_ids, strict=True):
        marker = f"doc_id: {doc_id}"
        text = str(passage.get("text") or "")
        if text.startswith(marker):
            continue
        passage["text"] = f"{marker}\n{text}"


def _annotate_supervised_consistency_chunks_with_doc_ids(chunks: list[SupervisedConsistencyChunk]) -> None:
    """Prefix both supervised consistency views with stable doc IDs."""
    for chunk in chunks:
        _annotate_passages_with_doc_ids(chunk.view_a_passages, chunk.view_a_doc_ids)
        _annotate_passages_with_doc_ids(chunk.view_b_passages, chunk.view_b_doc_ids)


def build_training_chunks_for_epoch(
    config: StudentSFTConfig,
    examples: list[GroupedSilverExample],
    *,
    epoch: int,
) -> list[RegressionChunk]:
    """Build deterministic, permutation-augmented SFT chunks for one epoch.

    Each view preserves the ``doc_id -> teacher_scores_mean`` mapping. Targets
    are not normalized within a chunk because top-100 inference compares scores
    across chunks.
    """
    chunks: list[RegressionChunk] = []
    for view_idx, (permutation_seed, chunk_size) in enumerate(_augmentation_view_specs(config, epoch=epoch)):
        if permutation_seed is None:
            chunks.extend(
                regression_chunks(
                    examples,
                    chunk_size=chunk_size,
                    permutation_seed=None,
                )
            )
            continue
        rate = _augmentation_shuffle_rate(config)
        if rate >= 1.0:
            chunks.extend(
                regression_chunks(
                    examples,
                    chunk_size=chunk_size,
                    permutation_seed=permutation_seed,
                )
            )
            continue
        if rate <= 0.0:
            chunks.extend(
                regression_chunks(
                    examples,
                    chunk_size=chunk_size,
                    permutation_seed=None,
                )
            )
            continue
        for example in examples:
            example_seed = (
                permutation_seed
                if _should_shuffle_example(
                    example,
                    config=config,
                    epoch=epoch,
                    view_idx=view_idx,
                    permutation_seed=permutation_seed,
                )
                else None
            )
            chunks.extend(
                regression_chunks(
                    [example],
                    chunk_size=chunk_size,
                    permutation_seed=example_seed,
                )
            )
    if config.include_doc_ids_in_prompt:
        _annotate_chunks_with_doc_ids(chunks)
    return chunks


def _consistency_view_seed_pair(config: StudentSFTConfig) -> tuple[int, int]:
    seeds = [int(seed) for seed in config.consistency_view_seeds]
    if len(seeds) != 2:
        raise ValueError(f"student.objective.view_seeds must contain exactly two seeds, got {seeds}")
    if seeds[0] == seeds[1]:
        raise ValueError("student.objective.view_seeds must use two distinct seeds")
    return (seeds[0], seeds[1])


def supervised_consistency_effective_lambda(config: StudentSFTConfig, *, global_step: int) -> float:
    """Linear warmup for supervised consistency continuation runs."""
    if config.consistency_lambda_warmup_steps <= 0:
        return float(config.consistency_lambda)
    init = float(config.consistency_lambda_warmup_init)
    final = float(config.consistency_lambda)
    if global_step <= 0:
        return init
    if global_step >= config.consistency_lambda_warmup_steps:
        return final
    frac = float(global_step) / float(config.consistency_lambda_warmup_steps)
    return init + (final - init) * frac


def build_supervised_consistency_chunks_for_epoch(
    config: StudentSFTConfig,
    examples: list[GroupedSilverExample],
    *,
    epoch: int,
) -> list[SupervisedConsistencyChunk]:
    """Build two-view chunks with K=1/K-shot labels as the supervised anchor."""
    chunks = supervised_consistency_chunks(
        examples,
        chunk_size=config.chunk_size,
        view_seeds=_consistency_view_seed_pair(config),
        epoch=epoch,
        view_generator=config.consistency_view_generator,
        label_scheme_pool=(config.label_vocab_train_pool if config.label_vocab_enabled else None),
        decorrelate_label_slots=config.label_vocab_decorrelate_slots,
    )
    if config.include_doc_ids_in_prompt:
        _annotate_supervised_consistency_chunks_with_doc_ids(chunks)
    return chunks


def _limit_train_examples(examples: list[Any], config: StudentSFTConfig) -> list[Any]:
    """Apply an optional deterministic small-scale cap for smoke runs."""
    if config.max_train_queries is None:
        return examples
    n = int(config.max_train_queries)
    if n <= 0:
        raise ValueError(f"student.data.max_train_queries must be positive when set, got {n}")
    return list(examples[:n])


def prepare_student_data(
    config: StudentSFTConfig,
    *,
    examples: list[GroupedSilverExample] | None = None,
    write_sidecars: bool = True,
) -> tuple[list[RegressionChunk], dict[str, int]]:
    """Load grouped examples, export the regression sidecar, and return regression chunks."""
    if examples is None:
        if config.silver_labels_path is None:
            raise ValueError("student.silver_labels_path is required for supervised_mse data preparation")
        examples = load_grouped_silver_examples(
            silver_labels_path=config.silver_labels_path,
            fixture_path=config.fixture_path,
            candidate_set_id=config.candidate_set_id,
            qrels_path=config.qrels_path,
            qids_path=config.train_qids_path,
            exclude_qids_path=config.exclude_qids_path,
        )
    examples = _limit_train_examples(examples, config)
    chunks = build_training_chunks_for_epoch(config, examples, epoch=0)

    out_dir = Path(config.output_dir)
    counts = {"regression": len(chunks), "groups": len(examples)}
    if not write_sidecars:
        return chunks, counts
    sidecars = out_dir / "data_views"
    sidecars.mkdir(parents=True, exist_ok=True)
    counts["regression"] = export_regression_jsonl(sidecars / "regression.jsonl", chunks)
    return chunks, counts


def prepare_supervised_consistency_data(
    config: StudentSFTConfig,
    *,
    examples: list[GroupedSilverExample] | None = None,
    write_sidecars: bool = True,
) -> tuple[list[SupervisedConsistencyChunk], dict[str, int]]:
    """Load silver examples and return anchored two-view consistency chunks."""
    if examples is None:
        if config.silver_labels_path is None:
            raise ValueError("student.silver_labels_path is required for supervised_consistency data preparation")
        examples = load_grouped_silver_examples(
            silver_labels_path=config.silver_labels_path,
            fixture_path=config.fixture_path,
            candidate_set_id=config.candidate_set_id,
            qrels_path=config.qrels_path,
            qids_path=config.train_qids_path,
            exclude_qids_path=config.exclude_qids_path,
        )
    examples = _limit_train_examples(examples, config)
    chunks = build_supervised_consistency_chunks_for_epoch(config, examples, epoch=0)

    out_dir = Path(config.output_dir)
    counts = {"supervised_consistency": len(chunks), "groups": len(examples)}
    if not write_sidecars:
        return chunks, counts
    sidecars = out_dir / "data_views"
    sidecars.mkdir(parents=True, exist_ok=True)
    counts["supervised_consistency"] = export_supervised_consistency_jsonl(
        sidecars / "supervised_consistency.jsonl",
        chunks,
    )
    return chunks, counts


def _pad_chunks_for_ddp_accumulation(
    chunks: list[RegressionChunk],
    *,
    world_size: int,
    batch_size_per_device: int,
    gradient_accumulation_steps: int,
) -> tuple[list[RegressionChunk], int]:
    """Pad global chunks so every DDP rank has a full final accumulation group.

    Without this, a global chunk count that is not divisible by
    ``world_size * batch_size_per_device * gradient_accumulation_steps`` gives
    some ranks more final microbatches than others. DDP then deadlocks when the
    longer ranks enter an extra gradient all-reduce after shorter ranks have
    left the train loop.
    """
    if world_size <= 1 or not chunks:
        return list(chunks), 0
    multiple = max(int(world_size), 1) * max(int(batch_size_per_device), 1) * max(int(gradient_accumulation_steps), 1)
    remainder = len(chunks) % multiple
    if remainder == 0:
        return list(chunks), 0
    pad_n = multiple - remainder
    padded = list(chunks)
    padded.extend(chunks[i % len(chunks)] for i in range(pad_n))
    return padded, pad_n


def _resolve_torch_dtype(dtype: str):
    import torch

    if dtype in {"auto", "bfloat16", "bf16"}:
        return torch.bfloat16
    if dtype in {"float16", "fp16"}:
        return torch.float16
    if dtype in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype={dtype!r}")


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _distributed_context() -> dict[str, int | bool]:
    """Initialize torch.distributed when launched by torchrun."""
    import os
    import torch
    import torch.distributed as dist

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
    return {
        "enabled": world_size > 1,
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "is_main": rank == 0,
    }


def _destroy_distributed() -> None:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _unwrap_model(model: Any) -> Any:
    return getattr(model, "module", model)


def _finite_metrics(metrics: dict[str, float]) -> dict[str, float]:
    return {
        key: float(value) for key, value in metrics.items() if isinstance(value, (int, float)) and math.isfinite(value)
    }


def _as_float(value: Any) -> float:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return float(value)


def _timing_log(phase: str, start: float, *, rank: int | None = None, extra: str = "") -> None:
    """Print a startup/training phase duration."""
    prefix = "[student][startup]"
    rank_s = f" rank={rank}" if rank is not None else ""
    extra_s = f" {extra}" if extra else ""
    print(f"{prefix}{rank_s} {phase}_s={time.monotonic() - start:.3f}{extra_s}", flush=True)


def _enable_gradient_checkpointing(model: Any) -> None:
    """Enable activation checkpointing for long-context LoRA training."""
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    base_model = getattr(model, "base_model", None)
    if base_model is not None and hasattr(base_model, "config") and hasattr(base_model.config, "use_cache"):
        base_model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    elif base_model is not None and hasattr(base_model, "gradient_checkpointing_enable"):
        base_model.gradient_checkpointing_enable()
    else:
        raise RuntimeError("gradient_checkpointing=true but model does not expose gradient_checkpointing_enable()")
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()


def _log_attention_diagnostics(model: Any) -> None:
    """Log enough attention metadata to verify FA2/SDPA/eager selection."""
    import torch

    try:
        import transformers

        transformers_version = getattr(transformers, "__version__", "unknown")
    except Exception:
        transformers_version = "unavailable"
    try:
        import flash_attn  # type: ignore[import-not-found]

        flash_attn_version = getattr(flash_attn, "__version__", "unknown")
    except Exception:
        flash_attn_version = "unavailable"

    unwrapped = _unwrap_model(model)
    base = getattr(unwrapped, "base_model", unwrapped)
    config = getattr(base, "config", getattr(unwrapped, "config", None))
    attn_impl = getattr(config, "_attn_implementation", None) or getattr(config, "attn_implementation", None)
    print(
        "[student][attention] "
        f"torch={torch.__version__} transformers={transformers_version} "
        f"flash_attn={flash_attn_version} attn_implementation={attn_impl}",
        flush=True,
    )
    seen: list[str] = []
    for name, module in unwrapped.named_modules():
        if "attn" not in name.lower():
            continue
        cls = f"{type(module).__module__}.{type(module).__name__}"
        seen.append(f"{name}:{cls}")
        if len(seen) >= 5:
            break
    if seen:
        print("[student][attention] sample_modules=" + " | ".join(seen), flush=True)


def _apply_gemma4_mixed_flash_sdpa(model: Any) -> dict[str, int]:
    """Use FA2 on Gemma4 sliding-attention layers and SDPA on full layers.

    Released Gemma4 checkpoints have ``head_dim=256`` on sliding layers and
    ``global_head_dim=512`` on full-attention layers. FlashAttention 2 hard
    fails on the latter, but each Gemma4 attention module reads
    ``config._attn_implementation`` during forward. Clone that small config
    object per module so layer-local dispatch is possible.
    """
    import copy

    counts = {"sliding_flash_attention_2": 0, "full_sdpa": 0}
    for module in model.modules():
        if module.__class__.__name__ != "Gemma4TextAttention":
            continue
        layer_type = getattr(module, "layer_type", None)
        if layer_type not in {"sliding_attention", "full_attention"}:
            continue
        module.config = copy.copy(module.config)
        if layer_type == "full_attention":
            module.config._attn_implementation = "sdpa"
            counts["full_sdpa"] += 1
        else:
            module.config._attn_implementation = "flash_attention_2"
            counts["sliding_flash_attention_2"] += 1
    if not any(counts.values()):
        raise RuntimeError("gemma4_mixed_flash_sdpa requested but no Gemma4TextAttention modules were found")
    return counts


class StudentProgressLogger:
    """Emit grep-friendly training progress and a machine-readable JSONL trace."""

    def __init__(self, config: StudentSFTConfig, output_dir: Path, device: str):
        """Prepare progress sinks for a single student training run."""
        self.log_every_n_steps = max(int(config.log_every_n_steps), 0)
        self.heartbeat_every_s = max(float(config.heartbeat_every_s), 0.0)
        self.progress_path = (
            Path(config.progress_jsonl_path) if config.progress_jsonl_path else output_dir / "progress.jsonl"
        )
        self.write_jsonl = bool(config.progress_jsonl)
        self.device = device
        self.objective_type = config.objective_type
        self.loss_ema: float | None = None
        self.last_emitted_step: int | None = None
        self.last_emit_monotonic = time.monotonic()
        self.tensorboard_log_dir: Path | None = None
        self.tb_writer: Any | None = None
        if self.write_jsonl:
            self.progress_path.parent.mkdir(parents=True, exist_ok=True)
            self.progress_path.write_text("", encoding="utf-8")
        if config.tensorboard_enabled:
            tb_dir = Path(config.tensorboard_log_dir) if config.tensorboard_log_dir else output_dir / "tensorboard"
            tb_dir.mkdir(parents=True, exist_ok=True)
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.tensorboard_log_dir = tb_dir
                self.tb_writer = SummaryWriter(log_dir=str(tb_dir))
                print(f"[student][tensorboard] log_dir={tb_dir}", flush=True)
            except Exception as e:
                raise RuntimeError(
                    "TensorBoard logging is enabled but torch.utils.tensorboard could not be initialized. "
                    "Install the `tensorboard` package in the training environment."
                ) from e

    def _gpu_memory_gb(self) -> dict[str, float]:
        if not self.device.startswith("cuda"):
            return {}
        try:
            import torch

            if not torch.cuda.is_available():
                return {}
            dev = torch.device(self.device)
            index = dev.index if dev.index is not None else torch.cuda.current_device()
            return {
                "gpu_mem_alloc_gb": torch.cuda.memory_allocated(index) / (1024**3),
                "gpu_mem_reserved_gb": torch.cuda.memory_reserved(index) / (1024**3),
            }
        except Exception:
            return {}

    def maybe_emit(  # noqa: C901
        self,
        *,
        step: int,
        epoch: int,
        batch_start: int,
        loss: float,
        lr: float,
        grad_norm: float,
        chunks_seen: int,
        slots_seen: int,
        chunks_per_s: float,
        slots_per_s: float,
        wall_s: float,
        token_cache_stats: dict[str, int] | None = None,
        extra_metrics: dict[str, float] | None = None,
        final: bool = False,
    ) -> bool:
        """Emit a progress record when a step, heartbeat, or final log is due."""
        now = time.monotonic()
        due_by_step = self.log_every_n_steps > 0 and (step == 1 or step % self.log_every_n_steps == 0)
        due_by_time = self.heartbeat_every_s > 0 and (now - self.last_emit_monotonic) >= self.heartbeat_every_s
        if not (due_by_step or due_by_time or final):
            return False

        if self.loss_ema is None:
            self.loss_ema = loss
        else:
            self.loss_ema = 0.8 * self.loss_ema + 0.2 * loss

        record: dict[str, Any] = {
            "objective": self.objective_type,
            "step": step,
            "epoch": epoch,
            "batch_start": batch_start,
            "loss": loss,
            "loss_ema": self.loss_ema,
            "lr": lr,
            "grad_norm": grad_norm,
            "chunks_seen": chunks_seen,
            "slots_seen": slots_seen,
            "chunks_per_s": chunks_per_s,
            "slots_per_s": slots_per_s,
            "wall_s": wall_s,
            "final": final,
        }
        if token_cache_stats:
            record.update(token_cache_stats)
        if extra_metrics:
            record.update(extra_metrics)
        record.update(self._gpu_memory_gb())

        if self.write_jsonl:
            with open(self.progress_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, sort_keys=True) + "\n")
        if self.tb_writer is not None:
            self.tb_writer.add_scalar("student/loss", loss, step)
            self.tb_writer.add_scalar("student/loss_ema", self.loss_ema, step)
            self.tb_writer.add_scalar("student/lr", lr, step)
            self.tb_writer.add_scalar("student/grad_norm", grad_norm, step)
            self.tb_writer.add_scalar("student/chunks_per_s", chunks_per_s, step)
            self.tb_writer.add_scalar("student/slots_per_s", slots_per_s, step)
            for key, value in (extra_metrics or {}).items():
                self.tb_writer.add_scalar(f"student/{key}", float(value), step)
            for key in ("gpu_mem_alloc_gb", "gpu_mem_reserved_gb"):
                if key in record:
                    self.tb_writer.add_scalar(f"student/{key}", float(record[key]), step)
            for key, value in (token_cache_stats or {}).items():
                self.tb_writer.add_scalar(f"student_cache/{key}", float(value), step)

        print(
            "[student][step={step}] objective={objective} epoch={epoch} loss={loss:.6f} "
            "loss_ema={loss_ema:.6f} lr={lr:.8f} grad_norm={grad_norm:.6f} "
            "chunks_s={chunks_per_s:.2f} slots_s={slots_per_s:.2f} wall_s={wall_s:.1f}".format(**record),
            flush=True,
        )
        for metric_name in (
            "student_progress_step",
            "student_loss",
            "student_loss_ema",
            "student_lr",
            "student_grad_norm",
            "student_chunks_per_s",
            "student_slots_per_s",
        ):
            key = metric_name.removeprefix("student_")
            if key == "progress_step":
                value = float(step)
            else:
                value = float(record[key])
            print(f"METRIC {metric_name}={value:.6f}", flush=True)
        for key, value in sorted((extra_metrics or {}).items()):
            print(f"METRIC student_{key}={float(value):.6f}", flush=True)
        for key in ("gpu_mem_alloc_gb", "gpu_mem_reserved_gb"):
            if key in record:
                print(f"METRIC student_{key}={float(record[key]):.6f}", flush=True)
        if token_cache_stats:
            cache_summary = " ".join(f"{k}={v}" for k, v in sorted(token_cache_stats.items()))
            print(f"[student][cache step={step}] {cache_summary}", flush=True)
        self.last_emitted_step = step
        self.last_emit_monotonic = now
        return True

    def emit_eval(self, *, step: int, phase: str, metrics: dict[str, float]) -> dict[str, float]:
        """Emit fixed-subset SFT diagnostics as logs, JSONL, and METRIC lines."""
        numeric = _finite_metrics(metrics)
        record: dict[str, Any] = {
            "event": "eval",
            "step": step,
            "phase": phase,
            **numeric,
        }
        if self.write_jsonl:
            with open(self.progress_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, sort_keys=True) + "\n")
        if self.tb_writer is not None:
            for name, value in sorted(numeric.items()):
                self.tb_writer.add_scalar(f"student_eval/{name}", value, step)

        compact = " ".join(f"{k}={v:.6f}" for k, v in sorted(numeric.items()))
        print(f"[student][eval step={step} phase={phase}] {compact}", flush=True)
        print(f"METRIC student_eval_step={float(step):.6f}", flush=True)
        for name, value in sorted(numeric.items()):
            print(f"METRIC student_eval_{name}={value:.6f}", flush=True)
        return numeric

    def close(self) -> None:
        """Flush and close optional progress sinks."""
        if self.tb_writer is not None:
            self.tb_writer.flush()
            self.tb_writer.close()


def predict_scores_for_examples_hf(
    *,
    model: Any,
    tokenizer: Any,
    examples: list[GroupedSilverExample],
    config: StudentSFTConfig,
    grade_token_ids: list[int],
    token_cache: StudentTokenCache | None = None,
) -> dict[tuple[str, str], float]:
    """Score grouped examples with the current student model."""
    import torch

    chunks = regression_chunks(examples, chunk_size=config.chunk_size)
    if config.include_doc_ids_in_prompt:
        _annotate_chunks_with_doc_ids(chunks)
    predictions: dict[tuple[str, str], float] = {}
    was_training = bool(getattr(model, "training", False))
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, len(chunks), config.batch_size_per_device):
                batch = chunks[start : start + config.batch_size_per_device]
                prefixes: list[str] = []
                keys: list[tuple[str, str]] = []
                for chunk in batch:
                    pfx = build_prefixes_for_chunk(
                        reranker_class=config.reranker_class,
                        tokenizer=tokenizer,
                        instruction=config.instruction,
                        query_text=chunk.query_text,
                        passages=chunk.passages,
                        max_doc_chars=config.max_doc_chars,
                    )
                    prefixes.extend(pfx)
                    keys.extend((chunk.query_id, doc_id) for doc_id in chunk.doc_ids)
                micro = int(config.slot_forward_batch_size or len(prefixes))
                micro = max(1, min(micro, len(prefixes)))
                for micro_start in range(0, len(prefixes), micro):
                    micro_prefixes = prefixes[micro_start : micro_start + micro]
                    micro_keys = keys[micro_start : micro_start + micro]
                    scores = compute_continuous_scores_hf(
                        model=model,
                        tokenizer=tokenizer,
                        prefixes=micro_prefixes,
                        grade_token_ids=grade_token_ids,
                        max_length=config.max_length,
                        token_cache=token_cache,
                    )
                    for key, score in zip(micro_keys, scores.detach().cpu().tolist(), strict=True):
                        predictions[key] = float(score)
    finally:
        if was_training:
            model.train()
    return predictions


def _all_gather_dicts(local: dict, *, world_size: int) -> list[dict]:
    """All-gather a Python dict from every DDP rank.

    Returns a list of ``world_size`` dicts (the gathered result is identical on
    every rank). Uses ``torch.distributed.all_gather_object`` so values can be
    arbitrary picklable Python objects (we use it for prediction maps).

    Stream/rank safety: we explicitly drain pending CUDA work and barrier on
    every rank before issuing the collective. Without this the NCCL
    size-prefix handshake inside ``all_gather_object`` (which runs on its own
    CUDA stream) can race against DDP's gradient-sync stream from the
    preceding optimizer step and deserialize a corrupted ``max_object_size``
    into a multi-exabyte ``input_tensor.resize_()`` allocation.
    """
    import torch
    import torch.distributed as dist

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dist.barrier()
    buf: list[dict] = [None] * world_size  # type: ignore[list-item]
    dist.all_gather_object(buf, local)
    return [d or {} for d in buf]


def _make_psi_scorer(
    *,
    model: Any,
    tokenizer: Any,
    config: StudentSFTConfig,
    grade_token_ids: list[int],
    token_cache: StudentTokenCache | None = None,
) -> Callable[[GroupedSilverExample, list[str]], dict[str, float]]:
    """Build a single-example scorer used by ``permutation_invariance_diagnostics``.

    The PSI proxy permutes doc order per (query, seed) and calls the student
    on the permuted prompt. We construct a transient ``GroupedSilverExample``
    in the permuted order and score it via the existing HF predict path.
    """

    def scorer(example: GroupedSilverExample, doc_order: list[str]) -> dict[str, float]:
        idx_for_doc = {d: i for i, d in enumerate(example.doc_ids)}
        try:
            perm_idxs = [idx_for_doc[d] for d in doc_order]
        except KeyError as exc:
            raise ValueError(f"unknown doc_id in permutation for qid={example.query_id}: {exc}") from exc
        permuted = GroupedSilverExample(
            group_id=example.group_id,
            query_id=example.query_id,
            query_text=example.query_text,
            candidate_set_id=example.candidate_set_id,
            doc_ids=[example.doc_ids[i] for i in perm_idxs],
            bm25_ranks=[example.bm25_ranks[i] for i in perm_idxs],
            passage_texts=[example.passage_texts[i] for i in perm_idxs],
            teacher_scores_mean=[example.teacher_scores_mean[i] for i in perm_idxs],
            teacher_scores_raw=[example.teacher_scores_raw[i] for i in perm_idxs],
            teacher_grade_vectors=[example.teacher_grade_vectors[i] for i in perm_idxs],
            teacher_scores_var=[example.teacher_scores_var[i] for i in perm_idxs],
            qrels=example.qrels,
        )
        preds = predict_scores_for_examples_hf(
            model=model,
            tokenizer=tokenizer,
            examples=[permuted],
            config=config,
            grade_token_ids=grade_token_ids,
            token_cache=token_cache,
        )
        return {d: preds[(example.query_id, d)] for d in doc_order}

    return scorer


_PSI_AGGREGATE_KEYS = (
    "permutation_score_variance_mean",
    "permutation_score_variance_median",
    "permutation_spearman_mean",
    "worst_permutation_ndcg_cut_10",
)


def _combine_psi_metrics(per_rank: list[dict[str, float]]) -> dict[str, float]:
    """Weighted-merge per-rank PSI proxy metrics across DDP ranks.

    Each rank contributes its local mean (over its query shard) plus the
    weight ``n_permutation_predictions``. The combined statistic is a
    weight-by-n mean — exact for the per-doc-variance mean and the spearman /
    NDCG means, approximate (close-enough for trend detection) for the median.
    """
    combined: dict[str, float] = {}
    total_n = sum(float(d.get("n_permutation_predictions", 0.0)) for d in per_rank)
    for key in _PSI_AGGREGATE_KEYS:
        weighted: list[tuple[float, float]] = []
        for d in per_rank:
            value = d.get(key)
            weight = float(d.get("n_permutation_predictions", 0.0))
            if value is None or weight <= 0 or math.isnan(float(value)):
                continue
            weighted.append((float(value), weight))
        if weighted:
            w_sum = sum(w for _, w in weighted)
            combined[key] = sum(v * w for v, w in weighted) / w_sum if w_sum > 0 else float("nan")
        else:
            combined[key] = float("nan")
    combined["n_permutation_predictions"] = total_n
    return combined


def evaluate_student_subset(
    *,
    model: Any,
    tokenizer: Any,
    examples: list[GroupedSilverExample],
    config: StudentSFTConfig,
    grade_token_ids: list[int],
    token_cache: StudentTokenCache | None = None,
    dist_ctx: dict | None = None,
) -> dict[str, float]:
    """Evaluate the current model against silver labels and qrels on a fixed subset.

    DDP-aware: when ``config.eval_shard_across_ranks`` is true and we're in a
    multi-rank world, examples are sharded by rank, each rank runs forward
    passes on its slice, and predictions are all-gathered onto every rank.
    Only rank 0 computes and returns the aggregate metrics (non-main ranks
    return ``{}``). When PSI proxy is enabled, the same sharding strategy is
    used for the K-permutation pass.
    """
    if config.eval_max_queries <= 0:
        return {}
    ctx = dist_ctx if dist_ctx is not None else {"enabled": False, "rank": 0, "world_size": 1, "is_main": True}
    enabled = bool(ctx.get("enabled", False)) and bool(config.eval_shard_across_ranks)
    rank = int(ctx.get("rank", 0))
    world_size = int(ctx.get("world_size", 1))
    is_main = bool(ctx.get("is_main", True))

    # Deterministically pick the eval slice. Default: shuffle once with
    # ``eval_seed`` to avoid the lexicographic-qid bias of a raw prefix slice.
    # The PSI subset is a strict prefix of the silver subset (same shuffled
    # list) so PSI metrics are comparable to the silver-eval queries.
    if config.eval_shuffle and len(examples) > 1:
        ordered = list(examples)
        random.Random(int(config.eval_seed)).shuffle(ordered)
    else:
        ordered = examples
    subset = ordered[: config.eval_max_queries]
    local_subset = subset[rank::world_size] if enabled and world_size > 1 else subset
    local_preds = predict_scores_for_examples_hf(
        model=model,
        tokenizer=tokenizer,
        examples=local_subset,
        config=config,
        grade_token_ids=grade_token_ids,
        token_cache=token_cache,
    )
    if enabled and world_size > 1:
        merged_preds: dict[tuple[str, str], float] = {}
        for shard in _all_gather_dicts(local_preds, world_size=world_size):
            merged_preds.update(shard)
    else:
        merged_preds = local_preds

    psi_combined: dict[str, float] = {}
    if config.eval_psi_enabled and config.eval_psi_max_queries > 0 and config.eval_psi_seeds:
        psi_subset = ordered[: config.eval_psi_max_queries]
        local_psi_subset = psi_subset[rank::world_size] if enabled and world_size > 1 else psi_subset
        scorer = _make_psi_scorer(
            model=model,
            tokenizer=tokenizer,
            config=config,
            grade_token_ids=grade_token_ids,
            token_cache=token_cache,
        )
        local_psi = permutation_invariance_diagnostics(local_psi_subset, scorer, seeds=config.eval_psi_seeds)
        if enabled and world_size > 1:
            psi_combined = _combine_psi_metrics(_all_gather_dicts(local_psi, world_size=world_size))
        else:
            psi_combined = local_psi

    if not is_main:
        return {}
    metrics = silver_prediction_diagnostics(subset, merged_preds)
    metrics["n_eval_queries"] = float(len(subset))
    for key, value in psi_combined.items():
        metrics[f"psi_{key}" if not key.startswith("psi_") else key] = value
    return metrics


def _checkpoint_root(config: StudentSFTConfig, output_dir: Path) -> Path:
    return Path(config.checkpoint_dir) if config.checkpoint_dir else output_dir / "checkpoints"


def _checkpoint_name(step: int) -> str:
    return f"checkpoint-step-{step:06d}"


def _resolve_resume_checkpoint(config: StudentSFTConfig, output_dir: Path) -> Path | None:
    raw = config.resume_from_checkpoint
    if not raw:
        return None
    root = _checkpoint_root(config, output_dir)
    if str(raw).lower() == "latest":
        latest = root / "latest_checkpoint.json"
        if not latest.is_file():
            raise FileNotFoundError(f"resume_from_checkpoint=latest but no marker exists at {latest}")
        marker = json.loads(latest.read_text(encoding="utf-8"))
        rel = marker.get("relative_path")
        path = root / str(rel) if rel else Path(str(marker["path"]))
        return path
    return Path(raw)


def _load_model_checkpoint(model: Any, checkpoint_dir: Path) -> None:
    """Load adapter/model weights from a checkpoint directory."""
    import torch

    pt_state = checkpoint_dir / "pytorch_model.bin"
    adapter_bin = checkpoint_dir / "adapter_model.bin"
    adapter_safe = checkpoint_dir / "adapter_model.safetensors"
    if pt_state.is_file():
        state = torch.load(pt_state, map_location="cpu")
        model.load_state_dict(state, strict=False)
        return
    if adapter_bin.is_file() or adapter_safe.is_file():
        from peft import set_peft_model_state_dict  # type: ignore[import-not-found]

        if adapter_safe.is_file():
            from safetensors.torch import load_file  # type: ignore[import-not-found]

            state = load_file(str(adapter_safe))
        else:
            state = torch.load(adapter_bin, map_location="cpu")
        set_peft_model_state_dict(model, state)
        return
    raise FileNotFoundError(f"No model/adapter weights found in checkpoint {checkpoint_dir}")


def _load_trainer_state(
    *,
    checkpoint_dir: Path,
    model: Any,
    optimizer: Any,
) -> dict[str, Any]:
    """Restore model, optimizer, and trainer cursor from ``checkpoint_dir``."""
    import torch

    _load_model_checkpoint(model, checkpoint_dir)
    opt_path = checkpoint_dir / "optimizer.pt"
    if opt_path.is_file():
        optimizer.load_state_dict(torch.load(opt_path, map_location="cpu"))
    state_path = checkpoint_dir / "trainer_state.json"
    if not state_path.is_file():
        raise FileNotFoundError(f"Checkpoint is missing trainer_state.json: {checkpoint_dir}")
    with open(state_path, "r", encoding="utf-8") as f:
        state = json.load(f)
    rng_path = checkpoint_dir / "rng_state.pt"
    if rng_path.is_file():
        rng = torch.load(rng_path, map_location="cpu")
        if "python_random_state" in rng:
            random.setstate(rng["python_random_state"])
        if "torch_cpu_rng_state" in rng:
            torch.set_rng_state(rng["torch_cpu_rng_state"])
        if torch.cuda.is_available() and "torch_cuda_rng_state_all" in rng:
            torch.cuda.set_rng_state_all(rng["torch_cuda_rng_state_all"])
    return state


def _load_ema_state(checkpoint_dir: Path) -> dict[str, Any] | None:
    """Load optional EMA shadow state from a training checkpoint."""
    import torch

    path = checkpoint_dir / "ema_state.pt"
    if not path.is_file():
        return None
    state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict):
        raise ValueError(f"Invalid EMA checkpoint state at {path}: expected dict")
    return state


def _move_optimizer_state_to_device(optimizer: Any, device: str) -> None:
    """Move optimizer state tensors after a CPU checkpoint load."""
    import torch

    target = torch.device(device)
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if hasattr(value, "to"):
                state[key] = value.to(target)


def _checkpoint_step_from_path(path: Path) -> int | None:
    """Parse ``checkpoint-step-NNNNNN`` → step int; return None on malformed names."""
    name = path.name
    prefix = "checkpoint-step-"
    if not name.startswith(prefix):
        return None
    try:
        return int(name[len(prefix) :])
    except ValueError:
        return None


def _select_best_steps(
    step_to_metric: dict[int, float],
    *,
    n: int,
    mode: str = "max",
) -> set[int]:
    """Pick the ``n`` step IDs with the best metric values.

    Steps absent from ``step_to_metric`` (or with non-finite values) are
    ignored. Ties favor the later step, avoiding an early checkpoint when a
    later checkpoint reaches the same metric value.
    """
    if n <= 0 or not step_to_metric:
        return set()
    finite = [(step, v) for step, v in step_to_metric.items() if math.isfinite(v)]
    if not finite:
        return set()
    reverse = mode != "min"
    finite.sort(key=lambda kv: (kv[1], kv[0]), reverse=reverse)
    return {step for step, _ in finite[:n]}


def _prune_checkpoints(  # noqa: C901
    root: Path,
    keep_last_n: int,
    *,
    keep_every_n_steps: int = 0,
    pinned_steps: set[int] | None = None,
) -> None:
    """Prune old checkpoint directories.

    Retention rule: a checkpoint at ``step`` is kept iff
        - it is among the last ``keep_last_n`` (by step) saved, OR
        - ``keep_every_n_steps > 0`` and ``step`` is a positive multiple of it
          (milestone snapshot — never pruned), OR
        - ``step`` is in ``pinned_steps`` (e.g. top-K by held-out NDCG).

    ``keep_last_n <= 0`` disables the rolling tail; ``keep_every_n_steps <= 0``
    disables milestones; ``pinned_steps`` empty / ``None`` disables best-by-
    metric pinning. With all three off, this function is a no-op.
    """
    if keep_last_n <= 0 and keep_every_n_steps <= 0 and not pinned_steps:
        return
    checkpoints = sorted(p for p in root.glob("checkpoint-step-*") if p.is_dir())
    if not checkpoints:
        return
    keep: set[Path] = set()
    if keep_last_n > 0:
        keep.update(checkpoints[-keep_last_n:])
    if keep_every_n_steps > 0:
        for path in checkpoints:
            step = _checkpoint_step_from_path(path)
            if step is not None and step > 0 and step % keep_every_n_steps == 0:
                keep.add(path)
    if pinned_steps:
        for path in checkpoints:
            step = _checkpoint_step_from_path(path)
            if step is not None and step in pinned_steps:
                keep.add(path)
    for path in checkpoints:
        if path in keep:
            continue
        import shutil

        shutil.rmtree(path, ignore_errors=True)


def save_student_checkpoint(
    *,
    config: StudentSFTConfig,
    output_dir: Path,
    model: Any,
    tokenizer: Any,
    optimizer: Any,
    ema: TrainableParameterEMA | None = None,
    global_step: int,
    epoch: int,
    next_batch_start: int,
    eval_metrics_history: list[dict[str, float | str]] | None = None,
) -> Path:
    """Save a resumable LoRA SFT checkpoint outside the immutable run tree.

    ``eval_metrics_history`` is consulted (when ``config.keep_n_best > 0``) to
    pin the top-K steps by ``config.keep_n_best_metric`` so the rolling-tail
    pruner never evicts them.
    """
    import torch

    root = _checkpoint_root(config, output_dir)
    root.mkdir(parents=True, exist_ok=True)
    ckpt = root / _checkpoint_name(global_step)
    tmp = root / f".{ckpt.name}.tmp"
    if tmp.exists():
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)
    save_model = _unwrap_model(model)
    if hasattr(save_model, "save_pretrained"):
        save_model.save_pretrained(tmp)
    if hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(tmp)
    torch.save(optimizer.state_dict(), tmp / "optimizer.pt")
    if ema is not None:
        torch.save(ema.state_dict(), tmp / "ema_state.pt")
    torch.save(
        {
            "python_random_state": random.getstate(),
            "torch_cpu_rng_state": torch.get_rng_state(),
            "torch_cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        tmp / "rng_state.pt",
    )
    trainer_state = {
        "global_step": global_step,
        "epoch": epoch,
        "next_batch_start": next_batch_start,
        "config_id": config.id,
        "model_name": config.model_name,
        "saved_at_unix": time.time(),
    }
    with open(tmp / "trainer_state.json", "w", encoding="utf-8") as f:
        json.dump(trainer_state, f, indent=2, sort_keys=True)
    if ckpt.exists():
        import shutil

        shutil.rmtree(ckpt, ignore_errors=True)
    tmp.rename(ckpt)
    latest = {
        "path": str(ckpt),
        "relative_path": ckpt.name,
        "global_step": global_step,
        "saved_at_unix": trainer_state["saved_at_unix"],
    }
    with open(root / "latest_checkpoint.json", "w", encoding="utf-8") as f:
        json.dump(latest, f, indent=2, sort_keys=True)
    pinned_steps: set[int] = set()
    if config.keep_n_best > 0 and eval_metrics_history:
        step_to_metric: dict[int, float] = {}
        for entry in eval_metrics_history:
            step_value = entry.get("step")
            metric_value = entry.get(config.keep_n_best_metric)
            if not isinstance(step_value, (int, float)) or not isinstance(metric_value, (int, float)):
                continue
            step_to_metric[int(step_value)] = float(metric_value)
        pinned_steps = _select_best_steps(step_to_metric, n=config.keep_n_best, mode=config.keep_n_best_mode)
    _prune_checkpoints(
        root,
        config.keep_last_n_checkpoints,
        keep_every_n_steps=config.keep_every_n_steps,
        pinned_steps=pinned_steps,
    )
    print(f"[student][checkpoint] saved step={global_step} path={ckpt}", flush=True)
    return ckpt


def train_student_sft(  # noqa: C901
    config: StudentSFTConfig,
    *,
    model_factory: Callable[[StudentSFTConfig], tuple[Any, Any]] | None = None,
) -> dict[str, Any]:
    """Run LoRA SFT and return a summary dict.

    ``model_factory`` is injectable for fast tests. Production uses HF
    ``AutoModelForCausalLM`` + PEFT LoRA.
    """
    import torch

    setup_start = time.monotonic()
    phase_start = time.monotonic()
    dist_ctx = _distributed_context()
    is_main = bool(dist_ctx["is_main"])
    _timing_log("ddp_init", phase_start, rank=int(dist_ctx["rank"]), extra=f"world_size={dist_ctx['world_size']}")
    random.seed(config.seed)
    torch.manual_seed(config.seed)

    out_dir = Path(config.output_dir)
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
    resume_checkpoint = _resolve_resume_checkpoint(config, out_dir)
    phase_start = time.monotonic()
    if config.objective_type in {"supervised_mse", "kl_to_base"}:
        if config.silver_labels_path is None:
            raise ValueError(f"student.silver_labels_path is required for {config.objective_type} training")
        examples: list[GroupedSilverExample] | None = load_grouped_silver_examples(
            silver_labels_path=config.silver_labels_path,
            fixture_path=config.fixture_path,
            candidate_set_id=config.candidate_set_id,
            qrels_path=config.qrels_path,
            qids_path=config.train_qids_path,
            exclude_qids_path=config.exclude_qids_path,
        )
        examples = _limit_train_examples(examples, config)
        train_group_count = len(examples)
    elif config.objective_type in {
        "supervised_consistency",
        "mean_teacher",
    }:
        if config.silver_labels_path is None:
            raise ValueError(f"student.silver_labels_path is required for {config.objective_type} training")
        examples = load_grouped_silver_examples(
            silver_labels_path=config.silver_labels_path,
            fixture_path=config.fixture_path,
            candidate_set_id=config.candidate_set_id,
            qrels_path=config.qrels_path,
            qids_path=config.train_qids_path,
            exclude_qids_path=config.exclude_qids_path,
        )
        examples = _limit_train_examples(examples, config)
        train_group_count = len(examples)
    else:
        raise ValueError(f"Unsupported student.objective.type={config.objective_type!r}")

    # Held-out eval cohort. For supervised SFT, this falls back to training silver.
    if config.eval_silver_labels_path:
        eval_examples = load_grouped_silver_examples(
            silver_labels_path=config.eval_silver_labels_path,
            fixture_path=config.fixture_path,
            candidate_set_id=config.candidate_set_id,
            qrels_path=config.qrels_path,
        )
        if is_main:
            print(
                f"[student][eval] held-out cohort active: "
                f"train_queries={train_group_count} eval_queries={len(eval_examples)} "
                f"path={config.eval_silver_labels_path}",
                flush=True,
            )
    elif (
        config.objective_type
        in {
            "supervised_mse",
            "kl_to_base",
            "supervised_consistency",
            "mean_teacher",
        }
        and examples is not None
    ):
        eval_examples = examples
        if is_main:
            print(
                f"[student][eval] WARNING: no eval_silver_labels_path set, evaluating on a "
                f"subset of the training silver ({len(examples)} groups). "
                f"This confounds learning with memorization for overfitting detection.",
                flush=True,
            )
    else:
        eval_examples = []
    _timing_log("data_load", phase_start, rank=int(dist_ctx["rank"]), extra=f"groups={train_group_count}")
    phase_start = time.monotonic()
    rank = int(dist_ctx["rank"])
    world_size = int(dist_ctx["world_size"])

    # DebiasFirst slot weights. Estimated once from the training qrels and the
    # first-stage order, over the same (already limited) example set the chunks
    # are built from. Counts only, so every DDP rank derives an identical vector
    # without a broadcast. ``None`` unless objective.ips.enabled.
    ips_weights: list[float] | None = None
    if config.ips_enabled:
        if examples is None:
            raise ValueError("student.objective.ips requires silver examples to estimate slot propensity")
        slot_propensity = estimate_slot_propensity(
            examples,
            chunk_size=config.chunk_size,
            relevance_threshold=config.ips_relevance_threshold,
            smoothing_eps=config.ips_smoothing_eps,
            clip=config.ips_clip,
        )
        ips_weights = slot_propensity.weights
        if is_main:
            print("[student][debias-first] " + format_slot_table(slot_propensity), flush=True)
            try:
                out_dir = Path(config.output_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / "ips_slot_propensity.json").write_text(json.dumps(slot_propensity.as_dict(), indent=2))
            except OSError as exc:  # best-effort artifact; never fail training on it
                print(f"[student][debias-first] could not write ips_slot_propensity.json: {exc}", flush=True)

    def pad_global_chunks(global_chunks: list[RegressionChunk] | list[SupervisedConsistencyChunk], *, epoch: int):
        padded, pad_n = _pad_chunks_for_ddp_accumulation(
            global_chunks,
            world_size=world_size if bool(dist_ctx["enabled"]) else 1,
            batch_size_per_device=config.batch_size_per_device,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
        )
        if pad_n and is_main:
            print(
                f"[student][ddp] epoch={epoch} padded_global_chunks={pad_n} "
                f"raw_chunks={len(global_chunks)} padded_chunks={len(padded)} "
                f"multiple={world_size * config.batch_size_per_device * config.gradient_accumulation_steps}",
                flush=True,
            )
        return padded

    if config.objective_type in {"supervised_mse", "kl_to_base"}:
        assert examples is not None
        epoch0_global_chunks_raw, view_counts = prepare_student_data(config, examples=examples, write_sidecars=is_main)
    elif config.objective_type in {
        "supervised_consistency",
        "mean_teacher",
    }:
        assert examples is not None
        epoch0_global_chunks_raw, view_counts = prepare_supervised_consistency_data(
            config, examples=examples, write_sidecars=is_main
        )
    else:
        raise ValueError(f"Unsupported student.objective.type={config.objective_type!r}")
    epoch0_global_chunks = pad_global_chunks(epoch0_global_chunks_raw, epoch=0)
    global_n_chunks = len(epoch0_global_chunks)
    _timing_log(
        "chunk_build",
        phase_start,
        rank=int(dist_ctx["rank"]),
        extra=f"chunks={global_n_chunks} raw_chunks={len(epoch0_global_chunks_raw)}",
    )

    def shard_chunks(global_chunks):
        if dist_ctx["enabled"]:
            return global_chunks[rank::world_size]
        return list(global_chunks)

    chunks = shard_chunks(epoch0_global_chunks)
    if dist_ctx["enabled"]:
        print(
            f"[student][ddp] rank={rank}/{world_size} local_rank={dist_ctx['local_rank']} "
            f"local_chunks={len(chunks)} global_chunks={global_n_chunks}",
            flush=True,
        )

    if model_factory is None:
        from peft import LoraConfig, get_peft_model  # type: ignore[import-not-found]
        from transformers import AutoModelForCausalLM, AutoTokenizer

        phase_start = time.monotonic()
        tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=False)
        tokenizer.padding_side = "left"
        if hasattr(tokenizer, "truncation_side"):
            tokenizer.truncation_side = "left"
        tokenizer._slm_chat_template_kwargs = dict(config.chat_template_kwargs)
        tokenizer._slm_grade_skeleton_dummy = str(config.grade_skeleton_dummy)
        tokenizer._slm_grade_rubric_id = str(config.grade_rubric_id)
        _timing_log("tokenizer_load", phase_start, rank=int(dist_ctx["rank"]))
        phase_start = time.monotonic()
        model_kwargs = {
            "dtype": _resolve_torch_dtype(config.dtype),
            "trust_remote_code": False,
        }
        use_gemma4_mixed_attention = config.attn_implementation == "gemma4_mixed_flash_sdpa"
        if config.attn_implementation:
            model_kwargs["attn_implementation"] = (
                "flash_attention_2" if use_gemma4_mixed_attention else str(config.attn_implementation)
            )
        model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)
        if use_gemma4_mixed_attention:
            counts = _apply_gemma4_mixed_flash_sdpa(model)
            if is_main:
                print(f"[student][attention] gemma4_mixed_flash_sdpa={counts}", flush=True)
        _timing_log("model_load", phase_start, rank=int(dist_ctx["rank"]))
        phase_start = time.monotonic()
        if config.lora_adapter_init_path:
            # Continue training an existing LoRA adapter. The adapter's
            # architecture (r, alpha, dropout, target_modules) is taken from
            # its own adapter_config.json: ``config.lora_*`` are ignored.
            from peft import PeftModel  # type: ignore[import-not-found]

            adapter_dir = Path(config.lora_adapter_init_path)
            if is_main:
                print(
                    f"[student][lora] loading existing adapter from {adapter_dir} "
                    "(config.lora_r/alpha/dropout/target_modules ignored)",
                    flush=True,
                )
            model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=True)
            # Log the loaded adapter's actual architecture so any mismatch
            # between config-as-written and config-as-loaded is visible.
            try:
                loaded_cfg = model.peft_config.get("default")
                if loaded_cfg is not None and is_main:
                    print(
                        f"[student][lora] loaded adapter: r={loaded_cfg.r} "
                        f"alpha={loaded_cfg.lora_alpha} dropout={loaded_cfg.lora_dropout} "
                        f"target_modules={loaded_cfg.target_modules}",
                        flush=True,
                    )
            except Exception:
                pass
        else:
            model = get_peft_model(
                model,
                LoraConfig(
                    r=config.lora_r,
                    lora_alpha=config.lora_alpha,
                    lora_dropout=config.lora_dropout,
                    target_modules=config.lora_target_modules,
                    task_type="CAUSAL_LM",
                ),
            )
        _timing_log("peft_wrap", phase_start, rank=int(dist_ctx["rank"]))
    else:
        phase_start = time.monotonic()
        model, tokenizer = model_factory(config)
        tokenizer._slm_chat_template_kwargs = dict(config.chat_template_kwargs)
        tokenizer._slm_grade_skeleton_dummy = str(config.grade_skeleton_dummy)
        tokenizer._slm_grade_rubric_id = str(config.grade_rubric_id)
        _timing_log("model_factory", phase_start, rank=int(dist_ctx["rank"]))

    if is_main:
        _log_attention_diagnostics(model)
    device = _resolve_device(config.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=(float(config.adam_beta1), float(config.adam_beta2)),
        eps=float(config.adam_eps),
    )
    resume_state: dict[str, Any] | None = None
    if resume_checkpoint is not None:
        resume_state = _load_trainer_state(
            checkpoint_dir=resume_checkpoint,
            model=model,
            optimizer=optimizer,
        )
        if is_main:
            print(f"[student][checkpoint] resumed from {resume_checkpoint}", flush=True)
    if config.gradient_checkpointing:
        phase_start = time.monotonic()
        _enable_gradient_checkpointing(model)
        _timing_log("gradient_checkpointing", phase_start, rank=int(dist_ctx["rank"]))
        if is_main:
            print("[student] gradient_checkpointing=true", flush=True)
    if is_main:
        print(
            f"[student] objective={config.objective_type} loss={config.objective_loss} "
            f"alpha={config.objective_alpha} beta={config.objective_beta} "
            f"temperature={config.objective_temperature} ema_decay={config.ema_decay} "
            f"single_forward_readout={config.single_forward_readout}",
            flush=True,
        )
        ddp_no_sync_active = (
            config.ddp_no_sync_during_grad_accum and dist_ctx["enabled"] and config.gradient_accumulation_steps > 1
        )
        print(
            f"[student] ddp_no_sync_during_grad_accum={config.ddp_no_sync_during_grad_accum} "
            f"(active={ddp_no_sync_active}; world_size={dist_ctx['world_size']}, "
            f"grad_accum={config.gradient_accumulation_steps})",
            flush=True,
        )
    if (
        config.objective_type
        in {
            "supervised_consistency",
            "mean_teacher",
            "kl_to_base",
        }
        and not config.single_forward_readout
    ):
        raise ValueError(
            f"{config.objective_type} training currently requires student.training.single_forward_readout=true"
        )
    if dist_ctx["enabled"] and torch.cuda.is_available():
        device = f"cuda:{int(dist_ctx['local_rank'])}"
    phase_start = time.monotonic()
    model.to(device)
    _timing_log("model_to_device", phase_start, rank=int(dist_ctx["rank"]), extra=f"device={device}")
    if dist_ctx["enabled"]:
        from torch.nn.parallel import DistributedDataParallel

        phase_start = time.monotonic()
        ddp_kwargs = {}
        if torch.cuda.is_available():
            ddp_kwargs = {
                "device_ids": [int(dist_ctx["local_rank"])],
                "output_device": int(dist_ctx["local_rank"]),
            }
        model = DistributedDataParallel(model, **ddp_kwargs)
        _timing_log("ddp_wrap", phase_start, rank=int(dist_ctx["rank"]))
    if resume_state is not None:
        _move_optimizer_state_to_device(optimizer, device)
    ema: TrainableParameterEMA | None = None
    if config.objective_type == "mean_teacher":
        ema = TrainableParameterEMA(model, decay=config.ema_decay)
        if resume_checkpoint is not None:
            ema_state = _load_ema_state(resume_checkpoint)
            if ema_state is not None:
                ema.load_state_dict(ema_state)
                if is_main:
                    print(f"[student][checkpoint] resumed EMA state from {resume_checkpoint}", flush=True)
    model.train()
    grade_token_ids = resolve_grade_token_ids(tokenizer)
    progress_logger = StudentProgressLogger(config, out_dir, device) if is_main else None
    token_cache = StudentTokenCache(config)
    eval_metrics_history: list[dict[str, float | str]] = []
    _timing_log(
        "setup_total",
        setup_start,
        rank=int(dist_ctx["rank"]),
        extra=f"local_chunks={len(chunks)} global_chunks={global_n_chunks}",
    )

    local_chunk_counts_by_epoch = [len(chunks)]
    for epoch in range(1, config.epochs):
        if config.objective_type in {"supervised_mse", "kl_to_base"}:
            assert examples is not None
            epoch_global = build_training_chunks_for_epoch(config, examples, epoch=epoch)
        elif config.objective_type in {
            "supervised_consistency",
            "mean_teacher",
        }:
            assert examples is not None
            epoch_global = build_supervised_consistency_chunks_for_epoch(config, examples, epoch=epoch)
        else:
            raise ValueError(f"Unsupported student.objective.type={config.objective_type!r}")
        local_chunk_counts_by_epoch.append(len(shard_chunks(pad_global_chunks(epoch_global, epoch=epoch))))
    total_batches = sum(math.ceil(n_chunks / config.batch_size_per_device) for n_chunks in local_chunk_counts_by_epoch)
    total_steps = max(1, math.ceil(total_batches / max(config.gradient_accumulation_steps, 1)))
    if config.max_steps is not None:
        total_steps = min(total_steps, int(config.max_steps))
    warmup_steps = int(total_steps * config.warmup_ratio)

    def lr_scale(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max((step + 1) / warmup_steps, 1e-8)
        denom = max(total_steps - warmup_steps, 1)
        progress = min(max((step - warmup_steps) / denom, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return float(config.min_lr_ratio) + (1.0 - float(config.min_lr_ratio)) * cosine

    losses: list[float] = []
    global_step = int((resume_state or {}).get("global_step", 0))
    resume_epoch = int((resume_state or {}).get("epoch", 0))
    resume_next_batch_start = int((resume_state or {}).get("next_batch_start", 0))
    accum = 0
    recent_losses: list[float] = []
    recent_aux_losses: list[dict[str, float]] = []
    recent_chunks = 0
    recent_slots = 0
    train_start = time.monotonic()
    last_window_start = train_start
    last_checkpoint_step: int | None = None
    optimizer.zero_grad(set_to_none=True)

    def chunk_slot_count(
        chunk: RegressionChunk | SupervisedConsistencyChunk,
    ) -> int:
        if isinstance(chunk, SupervisedConsistencyChunk):
            return len(chunk.doc_ids)
        return len(chunk.target_scores)

    def flush_optimizer_step(*, epoch: int, batch_start: int, final: bool = False) -> None:
        nonlocal accum, global_step, recent_chunks, recent_slots, last_window_start, last_checkpoint_step
        current_lr = config.lr * lr_scale(global_step)
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if ema is not None:
            ema.update(model)
        optimizer.zero_grad(set_to_none=True)
        accum = 0
        global_step += 1

        now = time.monotonic()
        window_s = max(now - last_window_start, 1e-9)
        window_loss = sum(recent_losses) / max(len(recent_losses), 1)
        extra_metrics = (
            {
                key: sum(item[key] for item in recent_aux_losses) / max(len(recent_aux_losses), 1)
                for key in recent_aux_losses[0]
            }
            if recent_aux_losses
            else None
        )
        if progress_logger is not None:
            emitted = progress_logger.maybe_emit(
                step=global_step,
                epoch=epoch,
                batch_start=batch_start,
                loss=window_loss,
                lr=current_lr,
                grad_norm=_as_float(grad_norm),
                chunks_seen=recent_chunks,
                slots_seen=recent_slots,
                chunks_per_s=recent_chunks / window_s,
                slots_per_s=recent_slots / window_s,
                wall_s=now - train_start,
                token_cache_stats=token_cache.snapshot(reset=True),
                extra_metrics=extra_metrics,
                final=final,
            )
        else:
            token_cache.snapshot(reset=True)
            emitted = True
        if emitted:
            recent_losses.clear()
            recent_aux_losses.clear()
            recent_chunks = 0
            recent_slots = 0
            last_window_start = now
        should_eval_step = (
            config.eval_max_queries > 0
            and bool(eval_examples)
            and config.eval_every_n_steps > 0
            and global_step % config.eval_every_n_steps == 0
        )
        if should_eval_step:
            metrics = evaluate_student_subset(
                model=model,
                tokenizer=tokenizer,
                examples=eval_examples,
                config=config,
                grade_token_ids=grade_token_ids,
                token_cache=token_cache,
                dist_ctx=dist_ctx,
            )
            if is_main and progress_logger is not None:
                emitted_metrics = progress_logger.emit_eval(step=global_step, phase="step", metrics=metrics)
                eval_metrics_history.append({"phase": "step", "step": global_step, **emitted_metrics})
        if is_main and config.save_every_n_steps > 0 and global_step % config.save_every_n_steps == 0:
            save_student_checkpoint(
                config=config,
                output_dir=out_dir,
                model=model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                ema=ema,
                global_step=global_step,
                epoch=epoch,
                next_batch_start=batch_start + config.batch_size_per_device,
                eval_metrics_history=eval_metrics_history,
            )
            last_checkpoint_step = global_step

    if resume_state is not None:
        if is_main:
            print(
                "[student][checkpoint] resume cursor: "
                f"global_step={global_step} epoch={resume_epoch} next_batch_start={resume_next_batch_start}",
                flush=True,
            )

    if resume_state is None and config.eval_max_queries > 0 and bool(eval_examples) and config.eval_at_start:
        metrics = evaluate_student_subset(
            model=model,
            tokenizer=tokenizer,
            examples=eval_examples,
            config=config,
            grade_token_ids=grade_token_ids,
            token_cache=token_cache,
            dist_ctx=dist_ctx,
        )
        if is_main and progress_logger is not None:
            emitted_metrics = progress_logger.emit_eval(step=0, phase="start", metrics=metrics)
            eval_metrics_history.append({"phase": "start", "step": 0, **emitted_metrics})

    last_epoch_chunk_count = len(chunks)
    for epoch in range(config.epochs):
        if epoch < resume_epoch:
            continue
        if epoch == 0:
            epoch_chunks = list(chunks)
        else:
            if config.objective_type in {"supervised_mse", "kl_to_base"}:
                assert examples is not None
                epoch_global = build_training_chunks_for_epoch(config, examples, epoch=epoch)
            elif config.objective_type in {
                "supervised_consistency",
                "mean_teacher",
            }:
                assert examples is not None
                epoch_global = build_supervised_consistency_chunks_for_epoch(config, examples, epoch=epoch)
            else:
                raise ValueError(f"Unsupported student.objective.type={config.objective_type!r}")
            epoch_chunks = shard_chunks(pad_global_chunks(epoch_global, epoch=epoch))
        last_epoch_chunk_count = len(epoch_chunks)
        random.Random(config.seed + epoch).shuffle(epoch_chunks)
        for start in range(0, len(epoch_chunks), config.batch_size_per_device):
            if epoch == resume_epoch and start < resume_next_batch_start:
                continue
            batch = epoch_chunks[start : start + config.batch_size_per_device]
            batch_slots = sum(chunk_slot_count(chunk) for chunk in batch)
            # Skip DDP gradient all-reduce on non-final microbatches when
            # ``config.ddp_no_sync_during_grad_accum`` is on. Each loss
            # function below calls ``.backward()`` once per chunk, and DDP
            # all-reduces gradients on every backward by default: so without
            # this wrapper we pay ``gradient_accumulation_steps`` allreduces
            # per opt step instead of the 1 that's actually needed. The
            # ``is_final_microbatch`` predicate uses ``accum`` (the count of
            # microbatches completed in the current opt step) and matches the
            # flush condition at the bottom of this loop, so the syncing
            # microbatch is exactly the one that triggers the optimizer
            # step. End-of-training partial flush is the documented caveat:
            # see ``ddp_no_sync_during_grad_accum`` docstring.
            is_final_microbatch = (accum + 1) >= config.gradient_accumulation_steps
            if (
                config.ddp_no_sync_during_grad_accum
                and dist_ctx["enabled"]
                and not is_final_microbatch
                and hasattr(model, "no_sync")
            ):
                ddp_sync_ctx = model.no_sync()
            else:
                ddp_sync_ctx = contextlib.nullcontext()
            with ddp_sync_ctx:
                aux_loss_terms: dict[str, float] | None = None
                if config.objective_type == "mean_teacher":
                    assert ema is not None
                    loss_terms = backward_mean_teacher_loss_for_chunks_single_forward(
                        model=model,
                        ema=ema,
                        tokenizer=tokenizer,
                        chunks=batch,
                        grade_token_ids=grade_token_ids,
                        reranker_class=config.reranker_class,
                        instruction=config.instruction,
                        max_length=config.max_length,
                        max_doc_chars=config.max_doc_chars,
                        alpha=config.objective_alpha,
                        backward_scale=1.0 / max(config.gradient_accumulation_steps, 1),
                    )
                    loss_value = float(loss_terms["loss"])
                    aux_loss_terms = {
                        "sft_loss": float(loss_terms["sft_loss"]),
                        "mean_teacher_consistency_loss": float(loss_terms["mean_teacher_consistency_loss"]),
                        "mean_teacher_alpha": float(loss_terms["mean_teacher_alpha"]),
                        "ema_decay": float(loss_terms["ema_decay"]),
                    }
                elif config.objective_type == "kl_to_base":
                    loss_terms = backward_kl_to_base_loss_for_chunks_single_forward(
                        model=model,
                        tokenizer=tokenizer,
                        chunks=batch,
                        grade_token_ids=grade_token_ids,
                        reranker_class=config.reranker_class,
                        instruction=config.instruction,
                        max_length=config.max_length,
                        max_doc_chars=config.max_doc_chars,
                        beta=config.objective_beta,
                        temperature=config.objective_temperature,
                        backward_scale=1.0 / max(config.gradient_accumulation_steps, 1),
                    )
                    loss_value = float(loss_terms["loss"])
                    aux_loss_terms = {
                        "sft_loss": float(loss_terms["sft_loss"]),
                        "kl_anchor_loss": float(loss_terms["kl_anchor_loss"]),
                        "kl_anchor_beta": float(loss_terms["kl_anchor_beta"]),
                        "kl_anchor_temperature": float(loss_terms["kl_anchor_temperature"]),
                    }
                elif config.objective_type == "supervised_consistency":
                    effective_lambda = supervised_consistency_effective_lambda(config, global_step=global_step)
                    loss_terms = backward_supervised_consistency_loss_for_chunks_single_forward(
                        model=model,
                        tokenizer=tokenizer,
                        chunks=batch,
                        grade_token_ids=grade_token_ids,
                        reranker_class=config.reranker_class,
                        instruction=config.instruction,
                        max_length=config.max_length,
                        max_doc_chars=config.max_doc_chars,
                        consistency_lambda=effective_lambda,
                        backward_scale=1.0 / max(config.gradient_accumulation_steps, 1),
                        objective_loss=config.objective_loss,
                        alpha=config.objective_alpha,
                    )
                    loss_value = float(loss_terms["loss"])
                    aux_loss_terms = {
                        "sft_loss": float(loss_terms["sft_loss"]),
                        "mse_loss": float(loss_terms["mse_loss"]),
                        "kl_vector_loss": float(loss_terms["kl_vector_loss"]),
                        "objective_alpha": float(loss_terms["objective_alpha"]),
                        "consistency_loss": float(loss_terms["consistency_loss"]),
                        "consistency_lambda": float(loss_terms["consistency_lambda"]),
                    }
                elif config.single_forward_readout:
                    if config.objective_loss == "mse":
                        loss_value = backward_mse_loss_for_chunks_single_forward(
                            model=model,
                            tokenizer=tokenizer,
                            chunks=batch,
                            grade_token_ids=grade_token_ids,
                            reranker_class=config.reranker_class,
                            instruction=config.instruction,
                            max_length=config.max_length,
                            max_doc_chars=config.max_doc_chars,
                            backward_scale=1.0 / max(config.gradient_accumulation_steps, 1),
                            ips_weights=ips_weights,
                        )
                    else:
                        loss_terms = backward_supervised_loss_for_chunks_single_forward(
                            model=model,
                            tokenizer=tokenizer,
                            chunks=batch,
                            grade_token_ids=grade_token_ids,
                            reranker_class=config.reranker_class,
                            instruction=config.instruction,
                            max_length=config.max_length,
                            max_doc_chars=config.max_doc_chars,
                            backward_scale=1.0 / max(config.gradient_accumulation_steps, 1),
                            objective_loss=config.objective_loss,
                            alpha=config.objective_alpha,
                        )
                        loss_value = float(loss_terms["loss"])
                        aux_loss_terms = {
                            "mse_loss": float(loss_terms["mse_loss"]),
                            "kl_vector_loss": float(loss_terms["kl_vector_loss"]),
                            "objective_alpha": float(loss_terms["objective_alpha"]),
                        }
                else:
                    if config.objective_loss == "mse":
                        loss_value = backward_mse_loss_for_chunks(
                            model=model,
                            tokenizer=tokenizer,
                            chunks=batch,
                            grade_token_ids=grade_token_ids,
                            reranker_class=config.reranker_class,
                            instruction=config.instruction,
                            max_length=config.max_length,
                            max_doc_chars=config.max_doc_chars,
                            backward_scale=1.0 / max(config.gradient_accumulation_steps, 1),
                            slot_forward_batch_size=config.slot_forward_batch_size,
                            token_cache=token_cache,
                        )
                    else:
                        loss_terms = backward_supervised_loss_for_chunks(
                            model=model,
                            tokenizer=tokenizer,
                            chunks=batch,
                            grade_token_ids=grade_token_ids,
                            reranker_class=config.reranker_class,
                            instruction=config.instruction,
                            max_length=config.max_length,
                            max_doc_chars=config.max_doc_chars,
                            backward_scale=1.0 / max(config.gradient_accumulation_steps, 1),
                            slot_forward_batch_size=config.slot_forward_batch_size,
                            token_cache=token_cache,
                            objective_loss=config.objective_loss,
                            alpha=config.objective_alpha,
                        )
                        loss_value = float(loss_terms["loss"])
                        aux_loss_terms = {
                            "mse_loss": float(loss_terms["mse_loss"]),
                            "kl_vector_loss": float(loss_terms["kl_vector_loss"]),
                            "objective_alpha": float(loss_terms["objective_alpha"]),
                        }
            losses.append(loss_value)
            recent_losses.append(loss_value)
            if aux_loss_terms is not None:
                recent_aux_losses.append(aux_loss_terms)
            recent_chunks += len(batch)
            recent_slots += batch_slots
            accum += 1
            if accum >= config.gradient_accumulation_steps:
                flush_optimizer_step(epoch=epoch, batch_start=start)
                if config.max_steps is not None and global_step >= config.max_steps:
                    break
        if config.max_steps is not None and global_step >= config.max_steps:
            break
    if accum:
        flush_optimizer_step(epoch=max(config.epochs - 1, 0), batch_start=last_epoch_chunk_count, final=True)
    elif is_main and losses and progress_logger is not None and progress_logger.last_emitted_step != global_step:
        now = time.monotonic()
        progress_logger.maybe_emit(
            step=global_step,
            epoch=max(config.epochs - 1, 0),
            batch_start=last_epoch_chunk_count,
            loss=losses[-1],
            lr=optimizer.param_groups[0]["lr"],
            grad_norm=0.0,
            chunks_seen=0,
            slots_seen=0,
            chunks_per_s=0.0,
            slots_per_s=0.0,
            wall_s=now - train_start,
            token_cache_stats=token_cache.snapshot(reset=True),
            extra_metrics=None,
            final=True,
        )

    if config.eval_max_queries > 0 and bool(eval_examples) and config.eval_at_end:
        metrics = evaluate_student_subset(
            model=model,
            tokenizer=tokenizer,
            examples=eval_examples,
            config=config,
            grade_token_ids=grade_token_ids,
            token_cache=token_cache,
            dist_ctx=dist_ctx,
        )
        if is_main and progress_logger is not None:
            emitted_metrics = progress_logger.emit_eval(step=global_step, phase="final", metrics=metrics)
            eval_metrics_history.append({"phase": "final", "step": global_step, **emitted_metrics})
    if progress_logger is not None:
        progress_logger.close()
    if is_main and config.save_every_n_steps > 0 and global_step > 0 and last_checkpoint_step != global_step:
        save_student_checkpoint(
            config=config,
            output_dir=out_dir,
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            ema=ema,
            global_step=global_step,
            epoch=max(config.epochs - 1, 0),
            next_batch_start=last_epoch_chunk_count,
            eval_metrics_history=eval_metrics_history,
        )

    if dist_ctx["enabled"]:
        import torch.distributed as dist

        dist.barrier()
    if not is_main:
        if dist_ctx["enabled"]:
            _destroy_distributed()
        return {
            "output_dir": str(out_dir),
            "checkpoint": None,
            "n_chunks": global_n_chunks,
            "local_chunks": len(chunks),
            "global_steps": global_step,
            "is_main_process": False,
            "rank": int(dist_ctx["rank"]),
            "world_size": int(dist_ctx["world_size"]),
        }

    ckpt = out_dir / "checkpoint-final"
    ckpt.mkdir(parents=True, exist_ok=True)
    save_model = _unwrap_model(model)
    if hasattr(save_model, "save_pretrained"):
        save_model.save_pretrained(ckpt)
    if hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(ckpt)
    summary = {
        "output_dir": str(out_dir),
        "checkpoint": str(ckpt),
        "n_chunks": len(chunks),
        "global_n_chunks": global_n_chunks,
        "local_chunks": len(chunks),
        "view_counts": view_counts,
        "global_steps": global_step,
        "is_main_process": True,
        "rank": int(dist_ctx["rank"]),
        "world_size": int(dist_ctx["world_size"]),
        "loss_initial": losses[0] if losses else None,
        "loss_final": losses[-1] if losses else None,
        # Only rank 0 builds a progress logger, and only rank 0 writes this summary.
        "loss_ema_final": progress_logger.loss_ema if progress_logger is not None else None,
        "progress_jsonl": (
            str(progress_logger.progress_path) if progress_logger is not None and config.progress_jsonl else None
        ),
        "tensorboard_log_dir": (
            str(progress_logger.tensorboard_log_dir)
            if progress_logger is not None and progress_logger.tensorboard_log_dir is not None
            else None
        ),
        "token_cache": token_cache.snapshot(reset=False),
        "eval_metrics": eval_metrics_history,
    }
    with open(out_dir / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "resolved_student_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2)
    if dist_ctx["enabled"]:
        _destroy_distributed()
    return summary


def train_student_sft_from_config(config_path: str | Path) -> dict[str, Any]:
    """Run student SFT from a YAML config path."""
    return train_student_sft(load_student_sft_config(config_path))
