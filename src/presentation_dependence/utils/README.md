# `utils/`

Config loading, TREC run I/O, run-path resolution, provenance, and logging.
No research logic lives here.

## Config

- [`config.py`](config.py): experiment-config plumbing shared by every entry
  point: `load_experiment_config` (exp-id or YAML path to `(Path, dict)`),
  `apply_overrides` (dotted-key `--override`, fails hard on unknown top-level
  keys), `apply_execution_environment` (runtime-only, process environment wins
  over trusted YAML defaults), and `resolve_channel` (channel name from the `data:` block),
  plus `write_resolved_config`, `find_configs_root`, `load_yaml_mapping`.
- [`dataset_meta.py`](dataset_meta.py): load first-stage provenance
  (`dataset_meta.yaml`) for a dataset directory under `data/`.
- [`sweeps.py`](sweeps.py): write and expand sweep manifests. A sweep YAML is an
  ordered job matrix; the reproduction pipelines serialize one per training and
  evaluation stage, and `scripts/run_sweep.py` expands it back into jobs.
  `configs/sweeps/_schema.md` defines the format. Imports stay light so
  `run_sweep.py --dry-run` works without the eval dependencies.

## Run I/O and paths

- [`trec.py`](trec.py): TREC run-file helpers (read/write/sort, qid ↔ dirname
  conversions); output stays compatible with `pytrec_eval` and `pyserini`.
- [`run_paths.py`](run_paths.py): resolve local experiment run directories.
- [`repo_meta.py`](repo_meta.py): repository metadata for run provenance.
- [`pyserini_index.py`](pyserini_index.py): guard for prebuilt-index lookups.
  Pyserini's `from_prebuilt_index` returns `None` for an unknown index name
  instead of raising, which otherwise shows up much later as an opaque
  `AttributeError`; this fails at the lookup with the name that was missed.

## Logging and progress

- [`setup_logging.py`](setup_logging.py): configured stdlib logger.
- [`log_redaction.py`](log_redaction.py): redact secrets from run logs and
  sweep manifests.
- [`progress.py`](progress.py): small append-only progress helpers for
  eval-style loops.
- [`dry_run.py`](dry_run.py): shared `--dry-run` reporting for the local
  runners. Resolves the config, applies overrides, prints what would run, and
  checks the declared inputs exist without importing torch, touching a GPU, or
  calling a provider.

## GPUs

- [`gpu.py`](gpu.py): GPU counting and single-node torchrun launching, shared by
  the container entrypoint and the local runners so `execution.distributed: ddp`
  means the same thing on both. `num_gpus` and `declared_gpus` resolve the count,
  `distributed_mode` reads it off a config, and `maybe_reexec_torchrun`
  re-executes the caller under torchrun so `WORLD_SIZE`, `RANK`, and `LOCAL_RANK`
  exist for the `DistributedDataParallel` wrapper in `self_distill/student.py`.
  `in_torchrun_worker` guards against a worker re-launching itself.

[`__init__.py`](__init__.py) re-exports the common config primitives and
`setup_logging`.
