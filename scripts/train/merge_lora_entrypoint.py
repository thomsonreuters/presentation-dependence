#!/usr/bin/env python
"""Merge a PEFT LoRA adapter into a Hugging Face base model.

Reads a complete base model snapshot and a LoRA adapter from two directories
and writes the merged full checkpoint to a third. Useful for serving a trained
student as a plain checkpoint, since the released adapters carry no base
weights.

Merged checkpoints are large (15 GB and up for the bigger bases), so point
--output-dir at a volume with room for one.

Local callers pass all three directories explicitly. Container callers may
provide the inputs through ``SLM_CHANNEL_BASE_MODEL`` /
``SLM_CHANNEL_LORA_ADAPTER`` (or the legacy ``SM_CHANNEL_*`` aliases).
The merge manifest defaults to the merged checkpoint directory unless an
explicit manifest directory or container output environment variable is set.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
from pathlib import Path
from typing import TypedDict


REQUIRED_BASE_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
OPTIONAL_SIDECARS = (
    "generation_config.json",
    "processor_config.json",
    "chat_template.jinja",
)
REQUIRED_ADAPTER_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
)


class WrittenFile(TypedDict):
    """One file emitted into the merged checkpoint."""

    path: str
    size_bytes: int


def _channel_default(channel: str) -> str | None:
    """Resolve a generic/local channel before its legacy container alias."""
    suffix = channel.upper().replace("-", "_").replace(".", "_")
    return os.environ.get(f"SLM_CHANNEL_{suffix}") or os.environ.get(f"SM_CHANNEL_{suffix}") or None


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    base_default = _channel_default("base-model")
    adapter_default = _channel_default("lora-adapter")
    p.add_argument(
        "--base-dir",
        default=base_default,
        required=base_default is None,
        help=(
            "Complete base-model snapshot. Required unless SLM_CHANNEL_BASE_MODEL "
            "or its legacy SM_CHANNEL_BASE_MODEL alias is set."
        ),
    )
    p.add_argument(
        "--adapter-dir",
        default=adapter_default,
        required=adapter_default is None,
        help=(
            "LoRA adapter directory. Required unless SLM_CHANNEL_LORA_ADAPTER "
            "or its legacy SM_CHANNEL_LORA_ADAPTER alias is set."
        ),
    )
    p.add_argument("--output-dir", required=True, help="Directory to write the merged checkpoint into.")
    p.add_argument(
        "--manifest-dir",
        default=None,
        help=(
            "Directory for merge_manifest.json. Default precedence: "
            "SLM_OUTPUT_DIR, legacy SM_OUTPUT_DATA_DIR, then --output-dir."
        ),
    )
    p.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    p.add_argument("--max-shard-size", default="20GB")
    return p.parse_args()


def _require_files(root: Path, names: tuple[str, ...], *, label: str) -> None:
    missing = [name for name in names if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{label} is missing required files: {missing} under {root}")


def _weight_files(root: Path) -> list[Path]:
    """Return the safetensors weight shard(s) for a model dir.

    Handles both single-file (``model.safetensors``) and sharded
    (``model-00001-of-000NN.safetensors`` + ``model.safetensors.index.json``)
    layouts. Sharded layouts are what large bases (e.g. Gemma-4 31B) ship.
    """
    single = root / "model.safetensors"
    if single.is_file():
        return [single]
    return sorted(root.glob("model-*-of-*.safetensors"))


def _require_base_weights(root: Path) -> None:
    if not _weight_files(root):
        raise FileNotFoundError(
            f"base model has no safetensors weights (neither model.safetensors nor "
            f"sharded model-*-of-*.safetensors) under {root}"
        )


def _collect_keys(files: list[Path]) -> set[str]:
    from safetensors import safe_open

    keys: set[str] = set()
    for f in files:
        with safe_open(str(f), framework="pt", device="cpu") as h:
            keys.update(h.keys())
    return keys


def _safetensors_data_size(path: Path) -> int:
    """Total data-section byte size of a safetensors file (header-only read)."""
    import json
    import struct

    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    end = 0
    for key, meta in header.items():
        if key == "__metadata__":
            continue
        end = max(end, int(meta["data_offsets"][1]))
    return end


def _torch_dtype(name: str):
    import torch

    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"unsupported dtype={name!r}")


def _describe_dir(local_dir: Path) -> list[WrittenFile]:
    """List the written checkpoint files and their sizes, for the manifest."""
    written: list[WrittenFile] = []
    for path in sorted(local_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(local_dir).as_posix()
        size = path.stat().st_size
        print(f"[merge-lora] wrote {rel} ({size} bytes)", flush=True)
        written.append({"path": rel, "size_bytes": size})
    return written


def _patch_missing_safetensor_keys(*, base_dir: Path, merged_dir: Path) -> list[str]:  # noqa: C901
    """Restore base checkpoint tensors omitted by PEFT merge/save.

    Gemma-4 has shared KV-layer tensors that PEFT/Transformers can omit when
    saving the merged model. vLLM treats those as uninitialized and refuses to
    load the checkpoint. Missing keys are untouched by LoRA, so copying them
    from the base snapshot preserves the intended merged weights.

    Supports both single-file and sharded base/merged layouts. Missing tensors
    are written into the last merged shard, and (for sharded outputs) the
    ``model.safetensors.index.json`` weight_map + total_size are updated so vLLM
    can locate them.
    """
    import json
    from collections import defaultdict

    base_files = _weight_files(base_dir)
    merged_files = _weight_files(merged_dir)
    if not base_files or not merged_files:
        return []

    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    base_keys = _collect_keys(base_files)
    merged_keys = _collect_keys(merged_files)
    missing = sorted(base_keys - merged_keys)
    if not missing:
        return []

    print(f"[merge-lora] restoring {len(missing)} missing base tensors into merged checkpoint", flush=True)

    # Locate each missing key in its base shard.
    missing_set = set(missing)
    key_to_base: dict[str, Path] = {}
    for f in base_files:
        with safe_open(str(f), framework="pt", device="cpu") as h:
            for k in h.keys():
                if k in missing_set and k not in key_to_base:
                    key_to_base[k] = f
    unresolved = [k for k in missing if k not in key_to_base]
    if unresolved:
        raise RuntimeError(f"missing base tensors not found in any base shard: {unresolved}")

    # Append the missing tensors to the last merged shard.
    target = merged_files[-1]
    tensors = load_file(str(target), device="cpu")
    with safe_open(str(target), framework="pt", device="cpu") as h:
        metadata = h.metadata()
    by_file: dict[Path, list[str]] = defaultdict(list)
    for k in missing:
        by_file[key_to_base[k]].append(k)
    for f, keys in by_file.items():
        with safe_open(str(f), framework="pt", device="cpu") as h:
            for k in keys:
                tensors[k] = h.get_tensor(k)
    tmp_file = target.with_suffix(".patched.safetensors")
    save_file(tensors, str(tmp_file), metadata=metadata)
    tmp_file.replace(target)

    # Sharded output: point the restored keys at the target shard + fix total_size.
    index_path = merged_dir / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.setdefault("weight_map", {})
        for k in missing:
            weight_map[k] = target.name
        total = sum(_safetensors_data_size(p) for p in _weight_files(merged_dir))
        index.setdefault("metadata", {})["total_size"] = total
        index_path.write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[merge-lora] updated index weight_map (+{len(missing)} keys) total_size={total}", flush=True)
    return missing


def _resolve_manifest_dir(output_dir: Path, explicit: str | Path | None) -> Path:
    """Choose a local-first manifest directory with a legacy container alias."""
    raw = explicit or os.environ.get("SLM_OUTPUT_DIR") or os.environ.get("SM_OUTPUT_DATA_DIR")
    return Path(raw) if raw else output_dir


def main() -> int:
    args = _parse_args()
    base_dir = Path(args.base_dir)
    adapter_dir = Path(args.adapter_dir)
    output_dir = Path(args.output_dir)

    _require_files(base_dir, REQUIRED_BASE_FILES, label="base model")
    _require_base_weights(base_dir)
    _require_files(adapter_dir, REQUIRED_ADAPTER_FILES, label="LoRA adapter")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    print(f"[merge-lora] base_dir      = {base_dir}", flush=True)
    print(f"[merge-lora] adapter_dir   = {adapter_dir}", flush=True)
    print(f"[merge-lora] output_dir    = {output_dir}", flush=True)

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        str(base_dir),
        dtype=_torch_dtype(args.dtype),
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model = model.merge_and_unload()
    model.save_pretrained(str(output_dir), safe_serialization=True, max_shard_size=args.max_shard_size)
    del model
    gc.collect()

    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir), trust_remote_code=False)
    tokenizer.save_pretrained(str(output_dir))
    for name in OPTIONAL_SIDECARS:
        src = base_dir / name
        if not src.exists():
            src = adapter_dir / name
        if src.exists():
            shutil.copy2(src, output_dir / name)

    missing_restored = _patch_missing_safetensor_keys(base_dir=base_dir, merged_dir=output_dir)
    written = _describe_dir(output_dir)
    manifest = {
        "base_dir": str(base_dir),
        "adapter_dir": str(adapter_dir),
        "output_dir": str(output_dir),
        "dtype": args.dtype,
        "max_shard_size": args.max_shard_size,
        "missing_base_keys_restored": missing_restored,
        "written": written,
    }
    manifest_dir = _resolve_manifest_dir(output_dir, args.manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "merge_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[merge-lora] manifest -> {manifest_path}", flush=True)
    print(f"METRIC merged_files={float(len(written)):.6f}", flush=True)
    print(f"METRIC merged_size_bytes={float(sum(int(item['size_bytes']) for item in written)):.6f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
