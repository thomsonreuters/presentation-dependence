# Troubleshooting

Commands assume the working directory is the package root.

Identify which layer failed before changing any scientific setting. In order:
the checkout, the environment, the data, the configuration, materialization,
execution, then collection. A wrong number at the end is usually a wrong input
near the beginning.

```bash
git status --short
./setup.sh --full
uv run --no-sync poe ci
uv run python scripts/setup_reproduction_data.py plan
uv run python scripts/study.py structural audit
```

Do not put full population validation or source-lock verification in the initial
diagnostic chain: both require manual/gated data that the default setup skips.

## Environment

`uv` **or Python is missing.** Run `./setup.sh`. Do not create a second
unmanaged virtual environment; the Python version is pinned in
`pyproject.toml`.

**Java or pyserini fails.** Data and index setup need Java 21. Cached indexes
live outside this repository, under `~/.cache/pyserini/`. Import the Lucene
searcher rather than the top-level package: plain `import pyserini` succeeds
even when the parts this repository uses do not.

```bash
java -version
echo "$JAVA_HOME"
uv run python -c "from pyserini.search.lucene import LuceneSearcher; print('pyserini ok')"
```

**pyserini fails with** `Missing credentials ... OPENAI_API_KEY`**.** Nothing here calls OpenAI. `pyserini.encode` constructs an OpenAI client at import time, and
`pyserini.search.lucene` pulls that module in, so the import fails on a machine
that has never set the variable. Every entry point in this repository already
sets a throwaway placeholder around the import and removes it afterwards, so
this should only appear from your own `import pyserini.search.lucene`. Set any
non-empty value for that shell if so. Do not leave a fake value exported: a
closed-model run would then send it and get a 401 instead of a clear "No API
key".

**An import works locally but fails in the container.** Check that the extra is
installed in the image, that `scripts/train/requirements.txt` carries the
dependency, and that the base image's CUDA matches the vLLM pin. The CUDA
mismatch is the usual cause; `[HARDWARE.md](HARDWARE.md)` has the compatibility
matrix. Run `uv run poe smoke` before rebuilding: it exercises the same import
graph on CPU in about a minute.

**vLLM is unavailable.** `./setup.sh --full` does not install it. On a supported
CUDA/Python 3.11 host, invoke the command through `uv run --extra vllm ...`.
Gemma-4 requires the separate Python 3.13/CUDA 13 container. On macOS or CPU,
use an HF-compatible example or the identity smoke.

**Student SFT fails on `flash_attention_2` / missing `flash_attn`.** `flash-attn`
is not in `uv.lock`. Tracked Qwen and Granite recipes set FA2 and need a
separate CUDA install of that package; otherwise override
`student.attn_implementation=sdpa` (Gemma recipes already use `sdpa`). See
[`TRAINING.md`](TRAINING.md).

**Hugging Face authentication fails.** Confirm `HF_TOKEN` and the pinned
revision. A token does not grant access to a gated model until you have accepted
its licence.

## Data

**A declared path is missing.**

```bash
uv run python scripts/setup_reproduction_data.py plan --task <task>
uv run python scripts/setup_reproduction_data.py run --task <task>
uv run python scripts/setup_reproduction_data.py validate --task <task>
```

`source_lock.py verify` **cannot build the actual file set.** The lock excludes
Legal-A/B but requires all other public, gated, and manual files. Provision
DL21–DL23, Signal-1M, TREC-News, and Robust04 before expecting success.

`source_lock.py verify` **reports** `files`**.** A present local fixture, run, qrels,
topic, or qid file differs from the tracked input. Do not overwrite the lock.
Identify which setup step produced the file, rematerialize it from the declared
source, and regenerate the lock only for an intentional reviewed migration.

**Legal-A or Legal-B is missing.** They are not distributed. Public
per-collection workflows do not require them, but the 18-collection aggregates
cannot be reproduced. Several non-internal collections are separately
manual/access-gated; "public" does not mean automatically downloadable.

## Configuration

```bash
uv run python scripts/run_experiment.py -e <id-or-yaml> --override reranker.device=cpu
```

A bare ID resolves under `configs/experiments/` for `run_experiment.py`,
`configs/self-distill/` for `run_self_distill_sft.py`, and `configs/silver/` for
`run_silver_generation.py`. `run_sweep.py` tries all three in that order. Reader
commands need an explicit `configs/reader/*.yaml`. Dotted overrides are
YAML-parsed and beat file values, and the result is written to
`resolved_config.yaml`.

If the wrong config runs, pass the YAML path explicitly and read the resolved
snapshot before retrying.

## Models and memory

**Out of memory.** Do not just change serving width, truncation, LoRA rank, or precision, because each of those changes the method. Check that only one model copy is loaded, inspect `max_model_len`, `max_doc_chars`, batch size and tensor parallelism against the reviewed config, and move to more GPUs or split the bundle. Consider quantization. Record any scientific change as a new config.

**HF and vLLM disagree.** Check grade-token IDs, prompt whitespace,
chat-template kwargs, dtype, revision, and readout mode. Expected-grade logit
readout is not equivalent to free-form grade generation; see
`[SCORING.md](SCORING.md)`.

**An adapter cannot be found.** Confirm the checkpoint catalog row
`(variant, training_seed)`, the selected step, `reranker.lora_path`, and
`max_lora_rank`. Do not reach for the latest checkpoint directory;
`[MODELS.md](MODELS.md#trained-students)` covers loading.

**A bundle fails shared-model validation.** Split its members by model signature
or adapter requirement. Do not weaken `assert_shared_model`.

**Training or inference resume fails.** The model, optimizer, data, objective,
and geometry must all match. Resume from an explicit checkpoint rather than
amending a completed trial in place. Base evaluation and PSI compare a
best-effort `_run_fingerprint`, warn on a mismatch, and append
`resume_history.json`; they do not refuse the resume or cover every field.
Silver resume does not have the same complete check. Never point an existing
run directory at changed settings or source data, and reject a publication run
with resume mismatch history.

## Results

**A baseline is outside its reference band.** DL19 BM25 is about 0.5058
nDCG@10, DL20 BM25 about 0.4876, and DL19 `example-passage-mxbai-large-v2`
falls in `[0.72, 0.74]`.
Check candidate depth, BM25 parameters, model revision, judged-qid filtering,
and the resolved config before treating a difference as a code regression.

**A rerun appears to do nothing.** Explicitly resumed runs reuse existing
per-query results. Inspect `per_query_results/`. If any config or input changed,
start a new run directory; do not delete a subset and mix old and new shards.
Delete the complete per-query tree only when intentionally recomputing the same
run identity from scratch.

**Only part of PSI completed.** Do not average partial and complete runs by
hand. Top up with `scripts/data/finalize_psi_topup.py`, keeping perturbation
type, K, seeds, and geometry identical.

**Numbers differ from published results.** Check in order: source-lock verification, the
model revision, the candidate pool or fixture hash, qid filtering, the
checkpoint catalog row, serving width and perturbation, and the reduction input
under `build/reproduction/`. Some divergence is expected;
`[REPRODUCE.md](REPRODUCE.md#reproduction-fidelity)` lists the known sources.

## Collection

Collection is the only layer that reads `runs/`, and it resolves explicit
bindings rather than discovering runs. Write
`build/reproduction/evidence/source-bindings/<task>/<stage>.json` to bind a stage
to exact run directories, or pass `--discover-latest` to scan for the newest
match. Never resolve an ambiguity by editing a timestamp into a manifest.

**An artifact is absent on a new checkout.** `runs/` stays empty until you
produce a run, so this usually means the stage has not been executed here. Find
the program that produces it with `study.py appendix audit`, run that, then bind
the result.

**A binding hash mismatches.** Do not overwrite the expected hash. Confirm which
run the binding points at and compare its recorded `artifact_sha256`. A mismatch
means the run was regenerated; change the binding only for a reviewed source
replacement, and re-record the hash in the same change.

**λ or checkpoint selection is missing.** OC-SFT needs the declared held-out
rule. Run `scripts/select_lambda.py` and collect
`training/checkpoints.json` before direct evaluation, per
`[EVAL-PROTOCOL.md](EVAL-PROTOCOL.md)`.

## Recovery rules

Never overwrite a selected checkpoint catalog with a partial run, mix task
populations, perturbations, or serving widths in one result, or regenerate
`source-lock.yaml` to hide drift. Create a new config or trial with a
descriptive ID instead of mutating evidence that has already been reported.