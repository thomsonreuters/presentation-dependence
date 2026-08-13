"""Local data-parallel teacher inference: one vLLM replica per GPU.

Each worker replicates the full model on one visible GPU
(``tensor_parallel_size=1``), runs the ordinary ``KShotBSCTeacher`` path over
its own shard of qids, and the parent merges the per-qid outputs back into the
single-run layout. That keeps the validated TP=1 fast path and avoids the TP>1
attention and cudagraph instability, at the cost of needing a model that fits
on one device.

Shared by the container entrypoint and ``scripts/run_silver_generation.py`` so
that ``teacher.local_data_parallel_workers`` means the same thing on both.
"""

from __future__ import annotations

import copy
import json
import multiprocessing as mp
import os
import shutil
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty
from typing import Any, Mapping

LOG = "[teacher][dp]"


def load_qids_from_cfg(cfg: dict) -> list[str]:
    """Resolve the qids requested by a self-distill config after path rewrite."""
    out: list[str] = []
    for qid in cfg.get("qids_to_run") or []:
        qid_s = str(qid).strip()
        if qid_s:
            out.append(qid_s)

    path_raw = cfg.get("qids_to_run_path")
    if path_raw:
        with open(Path(str(path_raw)), "r", encoding="utf-8") as f:
            for line in f:
                qid_s = line.strip()
                if qid_s:
                    out.append(qid_s)

    # Preserve order while dropping duplicates.
    seen: set[str] = set()
    deduped: list[str] = []
    for qid in out:
        if qid in seen:
            continue
        seen.add(qid)
        deduped.append(qid)
    return deduped


def split_round_robin(items: list[str], n_shards: int) -> list[list[str]]:
    """Round-robin split preserves the global prefix shape while balancing shards."""
    return [items[i::n_shards] for i in range(n_shards)]


def visible_device_ids(count: int) -> list[str]:
    """Return the device ids workers may claim, honouring ``CUDA_VISIBLE_DEVICES``.

    A worker sets ``CUDA_VISIBLE_DEVICES`` to a single id, which is interpreted
    against the *physical* devices. Numbering workers 0..N-1 would therefore
    ignore a restriction the caller set and reach for devices they excluded, so
    index into the caller's list instead.
    """
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw:
        ids = [part.strip() for part in raw.split(",") if part.strip()]
        if ids:
            return ids
    return [str(i) for i in range(count)]


def configure_shared_hf_cache() -> None:
    """Keep Hugging Face artefacts shared across local-DP worker processes.

    Per-worker ``XDG_CACHE_HOME`` is useful for torch/vLLM compile caches, but
    letting it also redirect HF caches makes every worker resolve/download the
    same checkpoint independently. A shared explicit HF cache avoids that
    thundering herd and lets HF file locks serialize any first download.
    """
    hf_home = Path(os.environ.get("SLM_SHARED_HF_HOME", "/tmp/slm_hf_cache"))
    hub_cache = hf_home / "hub"
    hf_home.mkdir(parents=True, exist_ok=True)
    hub_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HF_HUB_CACHE", str(hub_cache))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_home / "transformers"))


def maybe_prefetch_hf_model(cfg: dict) -> None:
    """Optionally prefetch the teacher checkpoint once before spawning workers.

    When the model starts as a Hub repo id, rewrite ``reranker.model_name`` to
    the local snapshot path returned by ``snapshot_download``. This prevents
    each vLLM worker from making its own metadata calls during engine init.
    """
    teacher_cfg = cfg.get("teacher") or {}
    if not bool(teacher_cfg.get("prefetch_hf_model", False)):
        return
    reranker_cfg = cfg.get("reranker") or {}
    model_name = str(reranker_cfg.get("model_name") or "")
    if not model_name:
        return
    if Path(model_name).exists():
        print(f"{LOG} model_name already local; skip HF prefetch: {model_name}", flush=True)
        return
    configure_shared_hf_cache()
    try:
        from huggingface_hub import snapshot_download
    except Exception as e:
        raise RuntimeError("teacher.prefetch_hf_model=true requires huggingface_hub") from e
    revision = (cfg.get("reranker") or {}).get("revision") or None
    max_workers = int(teacher_cfg.get("hf_snapshot_max_workers", 1))
    print(
        f"{LOG} prefetch_hf_model={model_name} revision={revision or 'default'} max_workers={max_workers}",
        flush=True,
    )
    local_snapshot = snapshot_download(
        repo_id=model_name,
        revision=revision,
        token=os.environ.get("HF_TOKEN") or None,
        max_workers=max(1, max_workers),
    )
    cfg.setdefault("reranker", {})["model_name"] = str(local_snapshot)
    cfg.setdefault("reranker", {})["revision"] = None
    print(f"{LOG} using prefetched local model snapshot: {local_snapshot}", flush=True)


def _worker_teacher(
    *,
    worker_idx: int,
    gpu_idx: str,
    cfg: dict,
    qids: list[str],
    run_dir: str,
    tmp_root: str,
    queue,
) -> None:
    """Worker process for local data-parallel teacher inference.

    Each worker owns exactly one visible GPU and runs the normal
    ``KShotBSCTeacher`` path with ``tensor_parallel_size=1``. The parent merges
    the per-qid outputs after all workers complete.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_idx
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    configure_shared_hf_cache()
    # Eight local vLLM engines on one host otherwise race on the same
    # torch.compile / vLLM cache directories. One worker failed in
    # ``torch/_inductor/standalone_compile.py`` while trying to save a compiled
    # graph to a missing subdirectory during local-DP testing. Isolating caches
    # per worker removes this cross-process write race and separates worker
    # artifacts for debugging.
    cache_root = Path("/tmp/slm_vllm_worker_cache") / f"worker_{worker_idx:02d}"
    for env_name, subdir in (
        ("TORCHINDUCTOR_CACHE_DIR", "torchinductor"),
        ("VLLM_CACHE_ROOT", "vllm"),
        ("XDG_CACHE_HOME", "xdg"),
    ):
        path = cache_root / subdir
        path.mkdir(parents=True, exist_ok=True)
        os.environ[env_name] = str(path)

    try:
        from presentation_dependence.self_distill.teacher import KShotBSCTeacher
        from presentation_dependence.utils.config import write_resolved_config

        worker_cfg = copy.deepcopy(cfg)
        worker_cfg["id"] = f"{cfg['id']}-worker{worker_idx:02d}"
        worker_cfg["qids_to_run"] = list(qids)
        worker_cfg.pop("qids_to_run_path", None)
        # The model fits on one GPU for the 4B/7B self-distill case. Local DP
        # means "replicate model per GPU", so force TP=1 in each subprocess.
        worker_rc = worker_cfg.setdefault("reranker", {})
        vllm_settings = worker_rc.setdefault("vllm_settings", {})
        vllm_settings["tensor_parallel_size"] = 1
        worker_teacher_cfg = worker_cfg.setdefault("teacher", {})
        worker_teacher_cfg.pop("local_data_parallel_workers", None)

        worker_tmp = Path(tmp_root)
        worker_tmp.mkdir(parents=True, exist_ok=True)
        cfg_path = write_resolved_config(worker_cfg, worker_tmp / f"worker_{worker_idx:02d}.yaml")

        worker_run_dir = Path(run_dir) / "workers" / f"worker_{worker_idx:02d}"
        worker_run_dir.mkdir(parents=True, exist_ok=True)
        print(f"{LOG}:{worker_idx} gpu={gpu_idx} qids={len(qids)} run_dir={worker_run_dir}", flush=True)
        teacher = KShotBSCTeacher(
            config_path=cfg_path,
            runs_root=str(worker_run_dir.parents[1]),
            run_dir=worker_run_dir,
        )
        summary = teacher.run()
        queue.put({"ok": True, "worker_idx": worker_idx, "summary": summary})
    except Exception:
        queue.put({"ok": False, "worker_idx": worker_idx, "traceback": traceback.format_exc()})


def merge_worker_outputs(
    *,
    cfg: dict,
    run_dir: Path,
    worker_summaries: list[dict],
    worker_shards: Mapping[int, list[str]],
    qids_order: list[str],
    n_workers: int,
) -> dict:
    """Merge per-worker silver outputs into the normal single-run layout.

    Each worker contributes only the qids it was assigned this run. A worker
    directory is never pruned, so resuming a run dir with a different worker
    count leaves files behind that now belong to a different shard; taking the
    whole directory would collide on them.
    """
    from presentation_dependence.self_distill.silver_io import write_manifest

    final_silver_dir = run_dir / str((cfg.get("teacher") or {}).get("output_subdir", "silver"))
    final_per_qid = final_silver_dir / "per_qid"
    final_per_qid.mkdir(parents=True, exist_ok=True)

    copied_qids: set[str] = set()
    max_docs = 0
    n_processed = 0
    n_resumed = 0
    for ws in worker_summaries:
        manifest = json.load(open(Path(ws["summary"]["manifest"]), "r", encoding="utf-8"))
        n_processed += int(manifest.get("n_queries_processed_this_run", 0))
        n_resumed += int(manifest.get("n_queries_resumed", 0))
        max_docs = max(max_docs, int(manifest.get("n_documents", 0)))
        src_per_qid = Path(ws["summary"]["manifest"]).parent / "per_qid"
        for qid in worker_shards[int(ws["worker_idx"])]:
            src = src_per_qid / f"{qid}.jsonl"
            if not src.is_file():
                continue  # reported as missing below, with the full list
            if qid in copied_qids:
                raise RuntimeError(f"Duplicate local-DP per-qid output for qid={qid}")
            shutil.copyfile(src, final_per_qid / src.name)
            copied_qids.add(qid)

    missing = [qid for qid in qids_order if qid not in copied_qids]
    if missing:
        preview = ", ".join(missing[:10])
        raise RuntimeError(f"Local-DP merge missing {len(missing)} qids: {preview}")

    labels_path = final_silver_dir / "silver_labels.jsonl"
    n_records = 0
    with open(labels_path, "w", encoding="utf-8") as out:
        for qid in qids_order:
            with open(final_per_qid / f"{qid}.jsonl", "r", encoding="utf-8") as f:
                for line in f:
                    out.write(line)
                    n_records += 1

    teacher_cfg = cfg.get("teacher") or {}
    rc = cfg.get("reranker") or {}
    manifest = {
        "experiment_id": cfg["id"],
        "teacher_model_id": str(rc.get("model_name") or rc.get("class") or "unknown"),
        "teacher_protocol": "k_shot_bsc",
        "prompt_template_id": str(teacher_cfg.get("prompt_template_id", "grade_int_v1")),
        "k_perms": int(teacher_cfg.get("k_perms", 0)),
        "seeds": list(teacher_cfg.get("seeds") or []),
        "grade_max": int(teacher_cfg.get("grade_max", 3)),
        "cross_query_batch": int(teacher_cfg.get("cross_query_batch", 1)),
        "local_data_parallel_workers": int(n_workers),
        "n_queries": len(copied_qids),
        "n_queries_processed_this_run": n_processed,
        "n_queries_resumed": n_resumed,
        "n_documents": max_docs,
        "n_records": n_records,
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    manifest_path = final_silver_dir / "manifest.json"
    write_manifest(manifest_path, manifest)
    print(
        f"{LOG} merged {len(copied_qids)} qids / {n_records} records from {n_workers} workers -> {labels_path}",
        flush=True,
    )
    return {
        "run_dir": str(run_dir),
        "silver_labels": str(labels_path),
        "manifest": str(manifest_path),
        "n_queries": len(copied_qids),
        "n_records": n_records,
        "k_perms": int(teacher_cfg.get("k_perms", 0)),
    }


def worker_count(cfg: dict) -> int:
    """Return the declared local-DP worker count, 0 when the config omits it."""
    return int((cfg.get("teacher") or {}).get("local_data_parallel_workers", 0) or 0)


def run_teacher_local_dp(  # noqa: C901
    *,
    cfg: dict,
    exp_id: str,
    trial_name: str,
    run_dir: Path,
    tmp_root: Path,
    n_workers: int,
    visible_gpus: int,
) -> dict:
    """Run local data-parallel teacher inference across multiple GPUs.

    Opt-in only: when ``teacher.local_data_parallel_workers`` is absent the
    caller runs its ordinary single-process path unchanged, including TP>1
    configs.
    """
    qids = load_qids_from_cfg(cfg)
    if not qids:
        raise RuntimeError("teacher.local_data_parallel_workers requires qids_to_run or qids_to_run_path")

    if visible_gpus and n_workers > visible_gpus:
        raise RuntimeError(f"teacher.local_data_parallel_workers={n_workers} exceeds the {visible_gpus} visible GPU(s)")

    device_ids = visible_device_ids(max(visible_gpus, n_workers))
    if len(device_ids) < n_workers:
        raise RuntimeError(
            f"teacher.local_data_parallel_workers={n_workers} exceeds the "
            f"{len(device_ids)} device(s) in CUDA_VISIBLE_DEVICES"
        )

    shards = split_round_robin(qids, n_workers)
    nonempty = [(i, shard) for i, shard in enumerate(shards) if shard]
    print(
        f"{LOG} exp_id={exp_id} trial={trial_name} workers={n_workers} "
        f"devices={device_ids[:n_workers]} qids={len(qids)} shard_sizes={[len(s) for s in shards]}",
        flush=True,
    )
    maybe_prefetch_hf_model(cfg)

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    procs = []
    stagger_s = float((cfg.get("teacher") or {}).get("local_data_parallel_worker_stagger_s", 0.0) or 0.0)
    for worker_idx, shard in nonempty:
        p = ctx.Process(
            target=_worker_teacher,
            kwargs={
                "worker_idx": worker_idx,
                "gpu_idx": device_ids[worker_idx],
                "cfg": cfg,
                "qids": shard,
                "run_dir": str(run_dir),
                "tmp_root": str(tmp_root / "local_dp"),
                "queue": queue,
            },
        )
        p.start()
        procs.append(p)
        if stagger_s > 0 and len(procs) < len(nonempty):
            print(f"{LOG} worker={worker_idx} started; staggering next worker by {stagger_s:.1f}s", flush=True)
            time.sleep(stagger_s)

    results: list[dict[str, Any]] = []
    while len(results) < len(procs):
        try:
            results.append(queue.get(timeout=5))
        except Empty:
            if all(p.exitcode is not None for p in procs):
                break
    for p in procs:
        p.join()

    failures = [r for r in results if not r.get("ok")]
    bad_exit = [p.exitcode for p in procs if p.exitcode not in (0, None)]
    if failures or bad_exit:
        for failure in failures:
            print(
                f"{LOG}[FATAL] worker={failure.get('worker_idx')} failed:\n{failure.get('traceback')}",
                flush=True,
            )
        raise RuntimeError(f"Local-DP teacher failed: {len(failures)} worker exceptions, exitcodes={bad_exit}")

    worker_summaries = sorted(results, key=lambda r: int(r["worker_idx"]))
    return merge_worker_outputs(
        cfg=cfg,
        run_dir=run_dir,
        worker_summaries=worker_summaries,
        worker_shards=dict(nonempty),
        qids_order=qids,
        n_workers=n_workers,
    )
