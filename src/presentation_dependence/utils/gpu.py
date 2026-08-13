"""GPU counting and single-node torchrun launching.

Shared by the container entrypoint and the local runners so that
``execution.distributed: ddp`` means the same thing on both. The training code
in :mod:`presentation_dependence.self_distill.student` reads ``WORLD_SIZE`` / ``RANK`` /
``LOCAL_RANK`` and wraps the model in ``DistributedDataParallel``; something has
to launch it under torchrun for those variables to exist, and this is that
something.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

TORCHRUN_WORKER_ENV = "SLM_TORCHRUN_WORKER"


def _declared() -> tuple[str, int] | None:
    """Return the declaring variable and its count, or None when none declares one.

    ``SLM_NUM_GPUS`` first for a local or SSH run, then the legacy
    ``SM_NUM_GPUS`` alias for older launchers. None and 0 are different
    answers: 0 is a caller saying "use no GPU", which must not be re-read as
    "nobody said", or setting it to force a CPU run would autodetect instead.
    The name comes back with the count so errors can name what to change.
    """
    for key in ("SLM_NUM_GPUS", "SM_NUM_GPUS"):
        raw = os.environ.get(key)
        if raw is None or not raw.strip():
            continue
        try:
            value = int(raw.strip())
        except ValueError:
            raise ValueError(f"{key}={raw!r} is not an integer GPU count") from None
        if value < 0:
            raise ValueError(f"{key}={raw!r} must not be negative")
        return key, value
    return None


def declared_gpus() -> int:
    """Return the GPU count the environment claims, 0 when it claims none.

    Does not fall back to autodetection: the CPU-fallback guard
    needs "how many GPUs should be here", and asking torch would always agree
    with torch.
    """
    declared = _declared()
    return declared[1] if declared is not None else 0


def visible_gpus() -> int | None:
    """Return the GPU count torch can see, or None when torch cannot answer."""
    try:
        import torch

        return int(torch.cuda.device_count())
    except Exception:
        return None


def num_gpus() -> int:
    """Return the usable GPU count, autodetecting when nothing declares one.

    Autodetection is what lets this run unchanged on a plain CUDA box, where
    neither environment variable is set.

    Every caller turns this number into that many processes, each binding one
    device, so a declaration torch cannot back is refused here rather than
    deferred. Over-declaring otherwise surfaces as a bare CUDA ordinal error
    raised inside a spawned worker, which names neither the variable that
    caused it nor the count that would have been legal. Declaring *fewer* GPUs
    than exist stays legal: that is how you hold devices back.
    """
    declared = _declared()
    if declared is None:
        seen = visible_gpus()
        return seen if seen is not None else 0

    key, value = declared
    seen = visible_gpus()
    # `seen is None` means torch could not answer, not that there are no
    # devices; there is nothing to contradict, so take the declaration.
    if seen is not None and value > seen:
        raise RuntimeError(
            f"{key}={value} exceeds the {seen} visible GPU(s). Each worker binds "
            f"cuda:<local rank>, so ranks >= {seen} would have no device. Lower "
            f"{key}, or unset it to autodetect; to choose which devices are used, "
            "set CUDA_VISIBLE_DEVICES instead."
        )
    return value


def log_cuda_diagnostics(log_prefix: str = "[diagnostics]") -> None:
    """Emit torch/CUDA diagnostics and reject silent CPU fallback.

    A host driver mismatch can cause ``torch.cuda.is_available()`` to silently
    return False, which makes ``device: auto`` resolve to CPU. The job then
    runs end-to-end at fp32 on CPU, orders of magnitude slower than expected,
    and only surfaces the issue indirectly, as a timeout or an out-of-memory
    failure after burning hours of compute.

    If the environment declares GPUs (``SLM_NUM_GPUS`` or its legacy
    ``SM_NUM_GPUS`` alias >= 1) but ``torch.cuda.is_available() is False``,
    abort with a clear actionable error rather than continuing on CPU. The
    guard reads only those variables, never the driver, so a CPU-only machine
    that declares nothing does not trip it, and a GPU box whose driver has
    broken still does.

    Set ``SLM_ALLOW_CPU_ON_GPU_INSTANCE=1`` in ``execution.environment`` to
    permit CPU execution on a GPU instance, for example while debugging an
    OOM. The override is disabled by default.

    Exits rather than raising: the message is several lines of remediation,
    which reads better than the same text wrapped in a traceback.
    """
    try:
        import torch
    except Exception as e:  # pragma: no cover
        print(f"{log_prefix} torch import failed: {e}", flush=True)
        return
    is_available = bool(torch.cuda.is_available())
    n_devices = int(torch.cuda.device_count()) if is_available else 0
    cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
    print(
        f"{log_prefix} torch={torch.__version__}  cuda_runtime={cuda_version}  "
        f"cuda_available={is_available}  n_gpus={n_devices}",
        flush=True,
    )
    if is_available:
        for i in range(n_devices):
            try:
                name = torch.cuda.get_device_name(i)
                cc = torch.cuda.get_device_capability(i)
                print(f"{log_prefix}   gpu[{i}]: {name}  cc={cc[0]}.{cc[1]}", flush=True)
            except Exception as e:  # pragma: no cover
                print(f"{log_prefix}   gpu[{i}]: error reading metadata: {e}", flush=True)
        return

    declared = declared_gpus()
    allow_cpu_on_gpu = os.environ.get("SLM_ALLOW_CPU_ON_GPU_INSTANCE", "0") == "1"

    if declared >= 1 and not allow_cpu_on_gpu:
        print(
            "\n"
            f"{log_prefix}[FATAL] CUDA is not available but the environment "
            f"declares {declared} GPU(s). Refusing to "
            "silently fall back to CPU.\n"
            "\n"
            "Likely root causes (in observed frequency):\n"
            "  1. ``vllm>=X`` (loose) transitively upgraded torch to a CUDA build the\n"
            "     host driver doesn't support. See scripts/train/requirements.txt\n"
            "     for the matched-pair pin (vllm 0.10.x <-> torch 2.8 <-> base image 2.8.0).\n"
            "  2. The base image's torch CUDA exceeds the host driver's max. Build\n"
            "     from a CUDA-matched PyTorch base; see docs/HARDWARE.md.\n"
            "  3. The host driver changed under a base image that used to work. Pin\n"
            "     the base image by digest rather than by a floating tag.\n"
            "\n"
            "If you genuinely want CPU on a GPU instance (debugging only), set\n"
            "``execution.environment.SLM_ALLOW_CPU_ON_GPU_INSTANCE=1`` in your YAML,\n"
            "or export SLM_ALLOW_CPU_ON_GPU_INSTANCE=1.\n",
            flush=True,
        )
        sys.exit(2)

    if declared >= 1 and allow_cpu_on_gpu:
        print(
            f"{log_prefix}[WARN] CUDA not available but "
            "SLM_ALLOW_CPU_ON_GPU_INSTANCE=1; proceeding on CPU per operator override.",
            flush=True,
        )
        return

    print(f"{log_prefix} No CUDA devices visible and none declared - running on CPU.", flush=True)


def distributed_mode(cfg: Mapping[str, Any]) -> str | None:
    """Return the declared distribution mode, or None when the config omits it."""
    execution_block = cfg.get("execution") or {}
    mode = execution_block.get("distributed") or execution_block.get("distribution")
    if mode is None:
        return None
    if isinstance(mode, dict):
        if mode.get("ddp") or mode.get("type") == "ddp":
            return "ddp"
        return None
    return str(mode).lower()


def in_torchrun_worker() -> bool:
    """True when this process is already a torchrun-spawned worker."""
    return os.environ.get(TORCHRUN_WORKER_ENV) == "1"


def maybe_reexec_torchrun(
    cfg: Mapping[str, Any],
    argv: list[str],
    *,
    script: Path,
    log_prefix: str,
) -> int | None:
    """Re-exec ``script`` under single-node torchrun, one worker per GPU.

    Returns the child's exit code when it launched workers, or None when the
    caller should carry on in this process: already inside a worker, no ``ddp``
    declared, or too few GPUs to be worth it. Raises when the environment
    declares more GPUs than exist; see :func:`num_gpus`.
    """
    if in_torchrun_worker():
        return None
    if distributed_mode(cfg) != "ddp":
        return None
    nproc = num_gpus()
    if nproc <= 1:
        print(
            f"{log_prefix} distributed=ddp requested but only {nproc} GPU(s) visible; "
            "running single process. Set SLM_NUM_GPUS to override.",
            flush=True,
        )
        return None
    env = os.environ.copy()
    env[TORCHRUN_WORKER_ENV] = "1"
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        "--nproc_per_node",
        str(nproc),
        str(script),
        *argv,
    ]
    print(f"{log_prefix} launching local torchrun nproc_per_node={nproc}", flush=True)
    print(f"{log_prefix} " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, env=env).returncode
