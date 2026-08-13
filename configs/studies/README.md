# Scientific study configs

Tracked scientific study definitions declare parameter matrices and the
materializers that produce runnable configs.

## Fields

Six study files share a small core; the rest of each file is specific to what
that study varies.

| field | in | purpose |
| --- | --- | --- |
| `schema_version` | all | Format version; currently `1`. |
| `description` | all | One line naming the scientific question. |
| `id` | most | Study id, matching the filename stem. |
| `outputs` | most | Where materialization writes, as `config_root` and `sweep_root`. |
| `materializers` | most | Entry points to run, each an `entrypoint` plus `arguments`. |
| `population` | most | Dataset population id, resolved from `configs/reproduction/populations/`. |
| `conditions` | most | Named arms of the study, each declaring its datasets and any per-arm execution settings. |
| `dimensions` | most | Axes the matrix varies over. |
| `task` | some | Task id when the study is scoped to one. |
| `collectors` | some | Entry points that reduce the runs into `build/reproduction/studies/`. |

A study may add its own keys, and several do: `scorers` and
`first_stage_tokens` in the first-stage transfer study, `stages` and
`template_manifest` where materialization is multi-step, and `primary`,
`scaling` and `generated_prefixes` in the sample-efficiency study. Read the
materializer named under `materializers` to see what a given file's extra keys
mean; there is no shared validator for them.

Materializers write configs to the runtime locations:

```text
configs/experiments/<descriptive-id>.yaml
configs/sweeps/<descriptive-id>.yaml
```

Expanded files are local execution products. Reproduction inputs are the tracked
study YAML, referenced templates and populations, and materializer code.
`configs/reproduction/appendix.yaml` maps the currently registered artifact
families to programs/studies; it is not yet one entry per numbered manuscript
table and figure. Study definitions remain under `configs/studies/`.

## First-stage checkpoint bindings

`first-stage-transfer.yaml` names trained scorers with stable
`checkpoint_ref` values. It deliberately carries no run trial, checkpoint step,
or adapter path. Before materializing `first-stage-panel` or
`first-stage-newsurf`, copy
`first-stage-transfer-checkpoints.example.yaml` to the ignored path declared by
`checkpoint_bindings` and replace each value with the selected local PEFT
adapter directory.

The binding file has one small schema:

```yaml
schema_version: 1
study: first-stage-transfer
checkpoints:
  qwen3-4b-k1: checkpoints/<local-selected-adapter>
```

Relative paths resolve from the repository root; absolute paths are accepted.
Materialization fails before writing configs unless every declared reference is
bound exactly once and contains `adapter_config.json` plus
`adapter_model.safetensors` or `adapter_model.bin`. SHA-256 hashes of those
files are written into the materialization rows. The binding file stays under
`build/` because it describes this checkout's local artifacts, not the
scientific condition.

New study and experiment IDs must describe the scientific condition. Do not
introduce historical campaign numbers or date-coded series names.

Operator commands, explicit result collection, and appendix reporting are
documented in
[`docs/REPRODUCE.md`](../../docs/REPRODUCE.md).
