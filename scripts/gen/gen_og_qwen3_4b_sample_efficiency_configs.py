#!/usr/bin/env python
"""Generate OG Qwen3-4B sample-efficiency (label-pool N) training configs.

Writes 16 self-distill YAMLs (4 arms x {0.1K,1K,3K,10K}) under configs/self-distill/
and a sweep manifest under configs/sweeps/.

Example:
    uv run python -m scripts.gen.gen_og_qwen3_4b_sample_efficiency_configs
    uv run python -m scripts.gen.gen_og_qwen3_4b_sample_efficiency_configs --dry-run
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs" / "self-distill"
SWEEP_DIR = REPO_ROOT / "configs" / "sweeps"

MAX_STEPS = 2305
LAMBDA = 5.0

# Effective DDP steps per epoch at matched geometry, from the A1 fixed-budget configs.
EPOCHS_BY_N = {
    100: 300,
    1000: 30,
    3000: 10,
    10000: 3,
}

N_SPECS = (
    ("0p1k", 100),
    ("1k", 1000),
    ("3k", 3000),
    ("10k", 10000),
)

ARMS = (
    {
        "arm": "k1_sft",
        "k_silver": 1,
        "loss": "sft",
        "template": "sft_k1",
        "id_tpl": "qwen3-4b-nonthink-sft-msmarco-{n_tag}-k1-labels-prefix",
        "source_silver": "silver_labels_qwen3_4b_k1_seed0_train_29500.jsonl",
        "train_silver_tpl": "silver_labels_qwen3_4b_k1_seed0_train_{n_qids}_prefix.jsonl",
        "eval_silver": "silver_labels_qwen3_4b_k1_seed0_heldout_500.jsonl",
        "view": "regression",
        "objective_block": "",
    },
    {
        "arm": "k10_sft",
        "k_silver": 10,
        "loss": "sft",
        "template": "sft_k10",
        "id_tpl": "qwen3-4b-nonthink-sft-msmarco-{n_tag}-k10-labels-prefix",
        "source_silver": "silver_labels_qwen3_4b_k10_train_29500.jsonl",
        "train_silver_tpl": "silver_labels_qwen3_4b_k10_train_{n_qids}_prefix.jsonl",
        "eval_silver": "silver_labels_qwen3_4b_k10_heldout_500.jsonl",
        "view": "regression",
        "objective_block": "",
    },
    {
        "arm": "k1_oc_sft",
        "k_silver": 1,
        "loss": "oc_sft",
        "template": "oc_sft_k1",
        "id_tpl": "qwen3-4b-nonthink-k1-supervised-consistency-lambda500-warmup500-msmarco-{n_tag}-prefix",
        "source_silver": "silver_labels_qwen3_4b_k1_seed0_train_29500.jsonl",
        "train_silver_tpl": "silver_labels_qwen3_4b_k1_seed0_train_{n_qids}_prefix.jsonl",
        "eval_silver": "silver_labels_qwen3_4b_k1_seed0_heldout_500.jsonl",
        "view": "supervised_consistency",
        "objective_block": """  objective:
    type: supervised_consistency
    loss: mse
    lambda: 5.0
    view_seeds: [0, 1]
    lambda_warmup:
      steps: 500
      init: 0.0
      schedule: linear""",
    },
    {
        "arm": "k10_oc_sft_hybrid",
        "k_silver": 10,
        "loss": "oc_sft",
        "template": "oc_sft_k10",
        "id_tpl": "qwen3-4b-nonthink-k10-supervised-consistency-lambda500-warmup500-msmarco-{n_tag}-prefix",
        "source_silver": "silver_labels_qwen3_4b_k10_train_29500.jsonl",
        "train_silver_tpl": "silver_labels_qwen3_4b_k10_train_{n_qids}_prefix.jsonl",
        "eval_silver": "silver_labels_qwen3_4b_k10_heldout_500.jsonl",
        "view": "supervised_consistency",
        "objective_block": """  objective:
    type: supervised_consistency
    loss: mse
    lambda: 5.0
    view_seeds: [0, 1]
    lambda_warmup:
      steps: 500
      init: 0.0
      schedule: linear""",
    },
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true", help="Print paths only; do not write files.")
    return p.parse_args()


def _render_config(*, arm: dict, n_tag: str, n_qids: int) -> tuple[str, str]:
    config_id = arm["id_tpl"].format(n_tag=n_tag)
    epochs = EPOCHS_BY_N[n_qids]
    train_silver = arm["train_silver_tpl"].format(n_qids=n_qids)
    objective_lines = ""
    if arm["objective_block"]:
        objective_lines = arm["objective_block"] + "\n"

    content = f"""###############################################################################
# OG Qwen3-4B sample-efficiency label-pool sweep: {arm["arm"]} @ N={n_qids}.
#
# Nested frozen-prefix silver (0.1K ⊂ 1K ⊂ 3K ⊂ 10K ⊂ 30K).
# Shared max-step ceiling ({MAX_STEPS}); checkpoint = argmax held-out MS MARCO nDCG@10.
# OC-SFT arms: lambda=5.0 with lambda-warmup-500 (deploy recipe).
###############################################################################

id: {config_id}

label_pool:
  source_silver_labels_path: {arm["source_silver"]}
  train_qids: {n_qids}
  subset_mode: prefix
  fixed_optimizer_steps: {MAX_STEPS}
  reporting_caveat: "nested frozen-prefix label pool under shared max-step ceiling"

reranker:
  class: Qwen3InstructGradeReranker
  model_name: Qwen/Qwen3-4B
  model_size: 4b
  model_release: qwen3-og
  chat_template_kwargs:
    enable_thinking: false
  instruction: "Given a web search query, retrieve relevant passages that answer the query"
  max_length: 4096
  docs_per_score_forward: 20
  batch_size: 20
  max_doc_chars: 1200

student:
  base_model: Qwen/Qwen3-4B
  reranker_class: Qwen3InstructGradeReranker
  silver_labels_path: {train_silver}
  eval_silver_labels_path: {arm["eval_silver"]}
  fixture_path: fixture.jsonl
  qrels_path: qrels.txt
  output_dir: runs/self-distill/{config_id}/student
  device: cuda
  dtype: bfloat16
  attn_implementation: flash_attention_2
  max_length: 4096
  max_doc_chars: 1200
  instruction: "Given a web search query, retrieve relevant passages that answer the query"
{objective_lines}  data:
    chunk_size: 20
    candidate_set_id: msmarco_seed42_bm25_top100
    include_doc_ids_in_prompt: false
    view: {arm["view"]}
    preserve_group_metadata: true
    token_cache:
      enabled: true
      max_entries: 100000
      cache_prefix_strings: true
      cache_token_ids: true
  lora:
    r: 16
    alpha: 32
    dropout: 0.05
    target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
  training:
    epochs: {epochs}
    lr: 0.0002
    batch_size_per_device: 1
    grad_accumulation_steps: 8
    slot_forward_batch_size: 8
    gradient_checkpointing: true
    ddp_no_sync_during_grad_accum: true
    warmup_ratio: 0.10
    lr_scheduler: cosine
    weight_decay: 0.01
    adam_beta1: 0.9
    adam_beta2: 0.95
    seed: 42
    max_steps: {MAX_STEPS}
    single_forward_readout: true
    log_every_n_steps: 10
    heartbeat_every_s: 60
    progress_jsonl: true
  evaluation:
    max_queries: 100
    every_n_steps: 200
    at_start: true
    at_end: true
    psi:
      enabled: true
      max_queries: 25
      seeds: [0, 1, 2]
  observability:
    tensorboard:
      enabled: true
  checkpoint:
    enabled: true
    save_every_n_steps: 200
    keep_last_n: 4
    keep_every_n_steps: 400
    keep_n_best: 3
    keep_n_best_metric: qrels_ndcg_cut_10
    keep_n_best_mode: max
    resume_from_checkpoint: null

eval:
  measures: [ndcg_cut_10]
  strict_qrels_filter: false

execution:
  fixture_channel: msmarco-train-selfdistill-seed42
  tensorboard:
    enabled: true
  checkpoint:
    enabled: true
  distributed: ddp

logging:
  level: INFO
"""
    return config_id, content


def _render_sweep(config_ids: list[str]) -> str:
    jobs = "\n".join(f"  - exp_id: configs/self-distill/{config_id}.yaml" for config_id in config_ids)
    return f"""###############################################################################
# OG Qwen3-4B sample-efficiency label-pool sweep (16 new LoRA cells).
#
# 2x2 arms (K1/K10 silver x SFT/OC-SFT) at N in {{0.1K,1K,3K,10K}}.
# 30K endpoints for all four arms are reused (no training in this sweep).
#
# Prereq: materialize the prefix-silver channel once.
#
# Run:
#   uv run python scripts/run_sweep.py \\
#       configs/sweeps/og-qwen3-4b-sample-efficiency.yaml
###############################################################################

id: og-qwen3-4b-sample-efficiency
description: "OG Qwen3-4B label-pool N sweep: 2x2 SFT/OC-SFT at 0.1K/1K/3K/10K (prefix silver)."

execution:
  max_parallel: 1

jobs:
{jobs}
"""


def main() -> int:
    args = parse_args()
    config_ids: list[str] = []
    written: list[Path] = []

    for arm in ARMS:
        for n_tag, n_qids in N_SPECS:
            config_id, content = _render_config(arm=arm, n_tag=n_tag, n_qids=n_qids)
            config_ids.append(config_id)
            path = CONFIG_DIR / f"{config_id}.yaml"
            if args.dry_run:
                print(path)
                continue
            path.write_text(content + "\n", encoding="utf-8")
            written.append(path)

    sweep_path = SWEEP_DIR / "og-qwen3-4b-sample-efficiency.yaml"
    sweep_content = _render_sweep(config_ids)
    if args.dry_run:
        print(sweep_path)
        return 0

    sweep_path.write_text(sweep_content + "\n", encoding="utf-8")
    print(f"[gen-sample-eff] wrote {len(written)} configs under {CONFIG_DIR}")
    print(f"[gen-sample-eff] wrote sweep {sweep_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
