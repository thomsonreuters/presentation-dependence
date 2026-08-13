"""Load and validate reader-bridge configs (``configs/reader/<id>.yaml``).

Reader jobs use a separate schema from ``reranker/data/eval`` experiments.
Required fields are checked here before run scripts apply CLI overrides.

Schema:

    reader:
      engine: vllm                  # vllm | echo
      model_name: ibm-granite/granite-4.1-8b
      family: granite               # must differ from the scorer family for clean attribution
      max_tokens: 64
      cot: false
      enable_thinking: false
      max_model_len: 8192
    dataset: hotpotqa               # hotpotqa | 2wiki | musique
    k_values: [3, 5]                # R3 and V3
    canonical_perm: 0
    scorer:                         # R3 and V3
      exp_id: P7-hotpotqa-qwen3-1p7b-ocsft-l300-warmup30k-lora-step469-vllm-psi
      recipe: OC-SFT
      size: 1p7b
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REQUIRED_TOP = ("id", "phase", "reader", "dataset")
VALID_PHASES = ("R3", "V3")


def load_reader_config(path: str | Path) -> dict[str, Any]:
    """Load a reader-bridge YAML and validate its required fields."""
    cfg = yaml.safe_load(Path(path).read_text())
    if not isinstance(cfg, dict):
        raise ValueError(f"reader config {path} did not parse to a mapping")
    missing = [k for k in REQUIRED_TOP if k not in cfg]
    if missing:
        raise ValueError(f"reader config {path} missing required keys: {missing}")
    if cfg["phase"] not in VALID_PHASES:
        raise ValueError(f"reader config {path}: phase must be one of {VALID_PHASES}, got {cfg['phase']!r}")
    if not isinstance(cfg.get("reader"), dict) or "engine" not in cfg["reader"]:
        raise ValueError(f"reader config {path}: 'reader.engine' is required")
    if cfg["phase"] in ("R3", "V3") and "exp_id" not in (cfg.get("scorer") or {}):
        raise ValueError(f"reader config {path}: {cfg['phase']} configs need 'scorer.exp_id'")
    return cfg
