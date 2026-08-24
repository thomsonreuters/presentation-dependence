# Hardware, containers, and cost

Training and vLLM inference that match the reported CUDA stack need NVIDIA
CUDA. Offline smoke, data setup, reducers, and structural checks do not. The
zero-dependency smoke (`uv run --no-sync poe smoke` /
`configs/experiments/_smoke-fixture.yaml`) in
`[REPRODUCE.md](REPRODUCE.md#zero-dependency-smoke)` needs no accelerator, data,
Java, or network; compatible Hugging Face rerankers can run small CPU/MPS
development checks. Everything below assumes you control the machine, images,
and mounts.

## Supported hardware

"Local" means the process, data, checkpoints, and artifacts are under operator
control. It does not mean every accelerator backend is supported.


| Workflow                                                   | NVIDIA CUDA                     | Apple MPS                                  | CPU                                    | AMD ROCm                                             |
| ---------------------------------------------------------- | ------------------------------- | ------------------------------------------ | -------------------------------------- | ---------------------------------------------------- |
| Offline smoke, data setup, reducers, and structural checks | Not required                    | Not required                               | Yes                                    | Not required                                         |
| Hugging Face reranker inference                            | Recommended                     | Best effort for compatible wrappers/models | Best effort; usually slow              | Not validated; generic PyTorch may work but untested |
| Student SFT                                                | Required by the tracked recipes | Not supported                              | Not supported for the release workflow | Not validated                                        |
| vLLM teacher, evaluation, and reader paths                 | Required                        | Not supported                              | Not supported                          | Not supported by the used dependency/image profiles  |


CPU/MPS can exercise the offline and Hugging Face inference paths. They do not
reproduce the paper's training or vLLM execution. Adding ROCm or MPS training
would need backend-specific dependency profiles plus model/readout and numerical
parity tests; treating generic PyTorch compatibility as support would be
misleading.

## Containers

Two dependency images are built from tracked Dockerfiles. Each takes an
operator-supplied base image; choose a base whose Python, PyTorch, and CUDA
versions match the requirements below and pin it by digest for a citable run.
The repository does not supply or lock that digest:

```bash
docker build -t presentation-dependence-sft:local \
             --build-arg BASE_IMAGE=<pytorch-cuda-base> \
             -f scripts/train-sft/Dockerfile .
docker build -t presentation-dependence-vllm:local \
             --build-arg BASE_IMAGE=<pytorch-cuda-base> \
             -f scripts/train-vllm/Dockerfile .
```


| Image                 | Contents                                                          | Used stack                       |
| --------------------- | ----------------------------------------------------------------- | -------------------------------- |
| SFT-only              | `scripts/train-sft/requirements.txt`; no vLLM, Ray, or CuPy       | PyTorch 2.8 / py312 / CUDA 12.9  |
| vLLM eval and teacher | Full eval and teacher stack from `scripts/train/requirements.txt` | PyTorch 2.10 / py313 / CUDA 13.0 |


The local package metadata supports Python 3.11 only. These container stacks are
separate execution contracts: they install requirements and run mounted source
rather than installing the project wheel through its `requires-python` bound.

Neither requirements file pins `torch`: the base image supplies the
CUDA-matched wheel, and pinning it here would silently replace that wheel with
one built for a different CUDA version. These Dockerfiles install dependencies
only; they do not copy the project or define a command. They set
`WORKDIR=/workspace` and `PYTHONPATH=/workspace/src`, so mount the reviewed
source there and mount data/output separately:

```bash
mkdir -p container-output
docker run --rm --gpus all \
  -v "$PWD:/workspace:ro" \
  -v "$PWD/data/_smoke:/workspace/input:ro" \
  -v "$PWD/container-output:/workspace/output" \
  presentation-dependence-vllm:local \
  python scripts/train/entrypoint.py \
    --config-path configs/experiments/_smoke-fixture.yaml \
    --channel smoke \
    --input-dir /workspace/input \
    --output-dir /workspace/output
```

Replace the smoke channel/config with the reviewed job inputs. Do not mount a
populated `.env`; pass only the required secret environment variables. The
entrypoint retains `/opt/ml` only as a legacy container-layout fallback.

## The vLLM and CUDA constraint

Read this before changing vLLM, torch, or the base image. It is why the vLLM
image exists separately.

vLLM has a hard `torch==X.Y.Z` dependency that moves with every release, and the
upgrade happens transitively and silently. An unconstrained `vllm>=0.7` resolves
to a recent release that pulls a `torch` built for a newer CUDA than the base
image provides, which breaks GPU access at runtime rather than at install time.


| vLLM           | Pulls           | Needs a base with                 |
| -------------- | --------------- | --------------------------------- |
| `0.10.2`       | `torch==2.8.0`  | CUDA 12.8                         |
| `0.19.1+cu130` | `torch==2.10.x` | CUDA 13.0                         |
| `0.20+`        | `torch==2.11.x` | CUDA 13.0 and a PyTorch 2.11 base |


`requirements.txt` therefore selects on Python version: `vllm==0.10.2` below
3.13, and the `0.19.1+cu130` wheel at 3.13 and above. The Gemma-4 expected-grade
inference path requires the CUDA 13 build; the older pin will not serve it.
These are historical compatibility pins, not a security endorsement. Do not
expose their API servers to untrusted networks; scan the exact image and move to
a non-affected release only after prompt-logprob/readout parity tests pass.

## GPU sizing

Measured on L40S (48 GB) and A10G (24 GB) class accelerators. An L40S is roughly
twice an A10G for this workload. A larger box is not automatically faster:
tensor-parallel overhead offsets the extra devices.


| Base size         | Publicly runnable configuration            | Notes                                                                                            |
| ----------------- | ------------------------------------------ | ------------------------------------------------------------------------------------------------ |
| ≤ 14 B            | 1 × L40S                                   | Tensor parallelism 1                                                                             |
| ~30–32 B          | No validated tracked TP>1 profile          | Paper timing used multi-GPU hardware, but shipped configs remain TP=1 and require one-device fit |
| Larger / overflow | Not supported by the tracked local configs | Add and validate a new TP profile rather than inferring TP from device count                     |


Paper training used single-node DDP with LoRA, slot-forward batching, gradient
checkpointing, and rank 0 as the only writer: A10G for the 1.7B student,
A100-40GB for the 4B–8B and small Gemma/Granite students, and A100-80GB for the
14B/32B/30B-class students. These are historical configurations, not portable
capacity guarantees. Qwen/Granite recipes set `flash_attention_2`, but
`flash-attn` is not locked here: install it for CUDA if you want FA2, or
override to `sdpa` (`[TRAINING.md](TRAINING.md)`).

A dated internal benchmark observed substantially lower vLLM evaluation
throughput on its A100 training pool than on its L40S pool. That comparison is
not part of the manuscript evidence and does not establish a general
sevenfold/A100 rule. Benchmark the exact model, vLLM build, concurrency, and
hardware before choosing an evaluation device.

## Using more than one GPU

### One job, one device

Every runner is an ordinary local process. For rerankers,
`reranker.device: auto` resolves `cuda` → `mps` → `cpu`; plain `cuda` means
device 0 and does not spread work over visible devices or inspect which device
is busy. To place a reranker elsewhere, set `device: cuda:1` or restrict the
process with `CUDA_VISIBLE_DEVICES`.

Student SFT has a narrower contract. The tracked recipes set
`student.device: cuda`; its `auto` fallback resolves only `cuda` → `cpu`, and
the CPU branch is not a supported release-training path. It does not select MPS.
This keeps an unsupported backend from being presented as paper-faithful
training.

MPS is available only to compatible Hugging Face inference paths. The
zero-dependency smoke in
`[REPRODUCE.md](REPRODUCE.md#zero-dependency-smoke)` uses the identity
reranker on a synthetic fixture and does not load a model, so it validates
plumbing rather than MPS inference. Tracked paper configs often pin
`dtype: bfloat16`; for the broadest CPU/MPS compatibility, override
`reranker.dtype=auto`, which resolves to float32 away from CUDA.

### Multi-GPU mechanisms

Which mechanism applies is decided by the kind of job, not by a global setting:


| Job                         | Key                                      | What it does                                                                                                                                                      |
| --------------------------- | ---------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Student SFT                 | `execution.distributed: ddp`             | Re-execs under single-node torchrun, one worker process per GPU, sharding training examples by rank. Rank 0 is the only writer.                                   |
| Teacher / silver generation | `teacher.local_data_parallel_workers: N` | Spawns N workers, each pinned to one GPU with a full model replica, splitting qids round-robin and merging the outputs back into one run.                         |
| Independent sweep cells     | `execution.max_parallel`, or `--jobs`    | Runs several unrelated cells at once. Only safe when they do not contend for the same device: see `[../configs/sweeps/_schema.md](../configs/sweeps/_schema.md)`. |


Tracked configs do not use tensor parallelism across devices. Every tracked
config sets `tensor_parallel_size: 1`, because TP=1 is the validated fast path
and TP>1 hit attention and cudagraph instability during the study. Both
multi-GPU mechanisms above are therefore *replication*: they need a model that
fits on one device, and they buy throughput rather than capacity. A base too
large for a single GPU is not something these configs can serve as they stand.

The first two mechanisms size themselves from the visible GPUs, so a single
such job expects the whole machine. Running two at once double-books every
device; `scripts/run_sweep.py` warns when a sweep would do that.

### Telling the runners what is available

`SLM_NUM_GPUS` overrides the detected count, and `CUDA_VISIBLE_DEVICES`
restricts which devices may be used. The teacher fan-out indexes into that
restriction rather than numbering from zero, so limiting a run to devices `4,5`
puts its workers on 4 and 5.

Both mechanisms degrade rather than fail, because a workstation usually has
fewer GPUs than the 8-GPU node the study ran on. DDP on a single GPU says so
and continues in one process. The teacher caps its worker count at what is
visible and falls back to the single-process path when that leaves one. The
fallback preserves the intended model/prompt/K protocol but changes request
batching, so byte-identical prefixes need not produce bit-identical scores.
Record worker count, cross-query batch, backend, and hardware with the run.

The container entrypoint is stricter: it refuses a teacher config asking for
more workers than the container can see. On a node rented for eight GPUs,
quietly using two is more likely a misconfiguration than a preference, so the
entrypoint fails rather than honouring the lower count.

### Nothing is queued

There is no scheduler, no job IDs, and nothing to poll. `run_sweep.py` blocks
on each cell, and `study.py <task> <stage> execute` runs the whole tree in the
calling process. A stage holds the terminal for its full wall time, which for
the primary student is around half a day on an 8-GPU node, so wrap long runs in
your own `tmux`, `nohup`, or batch scheduler.

## Cost

The full study, across every base, seed, ablation, and evaluation run, is about 100,000 GPU-hours. Budget about 90 allocated GPU-hours for the single primary student end to end on an 8-GPU node. A clean recorded SFT loop took roughly seven node-hours (56 raw GPU-hours); the larger planning figure includes setup, validation, checkpointing, and rerun allowance.

Within one evaluation cell, the plain nDCG measurement is about 7% of the cost
and the τ-PSI permutation sweep is the remaining 93%, roughly a 1:13 ratio.
Budget accordingly: order-instability numbers cost about an order of magnitude
more than the ranking quality numbers reported beside them.

Long evaluation runs were capped at 120 compute hours per job. That cap is a
buffer rather than a target, since a run that exceeds it loses the aggregation
step that writes `psi_metrics.json`.