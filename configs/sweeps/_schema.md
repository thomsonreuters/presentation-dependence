# Sweep YAML schema

A sweep YAML describes an ordered matrix of jobs that share setup but vary
along a small set of axes (dataset, model, hyperparameters).
`scripts/run_sweep.py <sweep.yaml>` expands the matrix and runs each cell.

## Minimal example

```yaml
id: passage-rerankers-dl19-example
description: "Three rerankers on the same DL19 passage pool."

execution:
  # Execution settings. `max_parallel` configures the runner itself and stops
  # here; every other key (environment, distributed, fixture_channel,
  # extra_input_channels) is forwarded to each cell as an override.
  max_parallel: 1

common_overrides:
  # Dotted-key overrides applied to EVERY job. Same semantics as the
  # `--override` flag the runners take.
  reranker.max_length: 8192

jobs:
  # Each list item is one job. `exp_id` is required, and must resolve to a
  # tracked config: the structural audit fails a sweep that points at one which
  # is not in the repository.
  - exp_id: example-passage-mxbai-large-v2
  - exp_id: example-passage-granite-b20-psi
    # `overrides` here WIN over `common_overrides` if the same key is set.
    overrides:
      reranker.batch_size: 16
  - exp_id: example-passage-jina-v3-b20
```

## Precedence (highest wins)

1. Per-job `overrides` (in `jobs[i].overrides`)
2. Top-level `common_overrides`
3. Top-level `execution.*` block (packed as `execution.X=...` overrides)
4. Values in `configs/experiments/<exp_id>.yaml`

## What the sweep runner does

`scripts/run_sweep.py <sweep.yaml>` expands the matrix and runs each cell
locally. For each expanded cell `(exp_id, override_list)` it:

1. Resolves `exp_id` against `configs/experiments/`, `configs/self-distill/`
   and `configs/silver/`, and picks the runner the config implies: a `teacher:`
   block means silver generation, `student:` means student SFT, `robustness:`
   means a permutation sweep, anything else a single reranking pass.
2. Invokes that runner with the cell's overrides, including `execution.*` ones.
   `execution.max_parallel` is the exception: it belongs to the runner, not the
   cell, so it is not forwarded.
3. Writes a manifest to `sweeps/<sweep_id>/manifest.json` with one entry per
   cell (exp_id, overrides, command, exit status).

Use `--dry-run` to print the expanded matrix and the commands without running
anything, and `--only`/`--skip` to resume a partial sweep from the manifest.
Both take either an `exp_id` or the cell index printed beside each cell.

## `max_parallel` counts processes on one machine

`max_parallel: N` makes the runner keep N cells in flight at once, each a
subprocess on the machine you launched from, each loading its own copy of the
model. It is not a fan-out width across nodes: two cells at once means two
models resident on the same GPU.

Tracked configs therefore declare `1`. Raise it only when the cells do not
contend for the same device, which in practice means either cells that use no
GPU at all (a reranker backed by a hosted endpoint is network-bound, so several
run happily side by side) or one GPU per cell via `CUDA_VISIBLE_DEVICES`.
`--jobs N` overrides the declared value for a single run without editing the
sweep, which is the better way to try a larger number.

A cell that is itself multi-GPU (a `ddp` student, or a teacher with
`local_data_parallel_workers`) sizes itself from the visible devices and so
expects the whole machine. Two of those at once double-book every GPU, and the
runner warns when a sweep would do that. See
[`../../docs/HARDWARE.md`](../../docs/HARDWARE.md#using-more-than-one-gpu).

## Varying the dataset instead of the model

A sweep does not need one config per cell. To run a single config across
several collections, repeat the `exp_id` and override the data paths, as
[`beir-surfaces-psi-example.yaml`](beir-surfaces-psi-example.yaml) does:

```yaml
jobs:
  - exp_id: example-passage-b20-psi
    overrides:
      id: beir-example--fiqa
      data.run_path: data/beir-v1.0.0-fiqa-test/fixture.jsonl
      eval.qrels_path: data/beir-v1.0.0-fiqa-test/qrels.txt
```

Override `id` as well as the paths. A run lands in `runs/<id>/<timestamp>/`, so
cells that share a config and do not override `id` write into one directory and
are told apart only by timestamp. Overriding it also means `exp_id` no longer
names a single cell, which is why `--only` accepts an index.

## Conventions

- `id` should be unique within the repository. Use
  `<purpose>-<model>-<YYYY-MM-DD>`.
- `description` is copied into the manifest for later search.
- Sweep definitions are immutable after launch. Amend a sweep by
  writing a new YAML, never by editing the old one.
