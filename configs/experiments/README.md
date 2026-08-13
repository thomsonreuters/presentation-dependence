# Experiment Configs

Keep this directory flat. Entry points resolve an experiment ID by loading
`configs/experiments/<id>.yaml`, and run artifacts are written under
`runs/<id>/`. Moving configs into subdirectories or renaming `id` values
requires a coordinated migration across docs, sweeps, run artifacts, and result
records.

Tracked scientific definitions live under `configs/studies/`. Their
materializers write runnable configs here and sweeps under `configs/sweeps/`.
New generated IDs use descriptive scientific names such as
`matched-variance--passage-reranking--dl19--seed42`; historical campaign
prefixes are compatibility-only and must not be introduced by new studies.

## Tracked examples

The `example-*.yaml` files are single-run configurations, not full study
matrices. They are runnable only after their declared data, model revision, and
optional backend extra are provisioned. Expected-grade students and protocols:

- `example-passage-b20-psi.yaml`: ordinary passage quality and PSI;
- `example-passage-granite-b20-psi.yaml`: alternate Granite base model;
- `example-passage-round-robin-b20-psi.yaml`: interleaved/round-robin chunks;
- `example-passage-dense-first-stage.yaml`: dense first-stage transfer;
- `example-passage-matched-variance-b1.yaml`: repeated B=1 control;
- `example-passage-pool-perturbation.yaml`: replace/drop pool sensitivity;
- `example-multi-document-qa-b10-psi.yaml`: answer-support scoring;
- `example-response-ranking-b4-psi.yaml`: response-quality scoring.

Reference rerankers, one per class:

- `example-passage-mxbai-large-v2.yaml`: pointwise order-free baseline;
- `example-passage-jina-v3-b20.yaml`: scoring-listwise reference (eval-only);
- `example-passage-rankzephyr-sliding-window.yaml`: generative listwise reference;
- `example-passage-capcal-b20-psi.yaml`: content-free probability calibration;
- `example-passage-pine-b20-psi.yaml`: position-invariant attention baseline;
- `example-passage-closed-model-genbsc-b20-psi.yaml`: closed/API generated BSC;
- `example-response-ranking-pairrm.yaml`: pairwise preference reference;
- `example-response-ranking-skywork-v2.yaml`: pointwise reward-model reference.

Related examples:

- `configs/self-distill/example-teacher-transfer-qwen3-1p7b-to-qwen3-4b.yaml`;
- `configs/reader/example-answer-reader-hotpotqa.yaml`;
- `configs/reader/example-verdict-reader-climate-fever.yaml`.

Run an example with the normal entrypoint:

```bash
uv run python scripts/data/build_fixture_pyserini.py \
  --topics data/dl19-passage/topics.tsv \
  --run data/dl19-passage/run.msmarco-v1-passage.bm25-default.dl19_sorted.txt \
  --index msmarco-v1-passage \
  --k 100 \
  --out data/dl19-passage/fixture.jsonl
uv run python scripts/run_experiment.py -e example-passage-b20-psi
```

The fixture build takes explicit flags because `example-passage-b20-psi` is a
`FixtureLoader` config: `-e` reads the first stage from the config, and this one
names the fixture rather than the run and index behind it.

Execution reads the paths declared in the config. Model weights resolve through
the Hugging Face cache, and data resolves under `data/`; see
[`docs/DATA-SETUP.md`](../../docs/DATA-SETUP.md) for how to populate it.
The unified Pyserini setup does not create the `fixture.jsonl` consumed by the
example above; the explicit builder command bridges that handoff. The Jina
example requires `trust_remote_code=True`; its wrapper rejects anything other
than an immutable 40-character revision, and the tracked example carries the
reviewed canonical commit.

## Naming

New IDs describe the scientific condition:

```text
<condition>--<task>--<variant>--seed<N>--<dataset>
```

Include only dimensions needed to distinguish the run, such as serving width,
first-stage retriever, placeholder, or control. Historical `R*`, `L*`, `A*`,
`P7`, and date-coded series remain valid artifact aliases but are not templates
for new names. Use `configs/studies/` for experiment matrices.

## Stability rules

- `id:` should match the filename stem for normal experiment configs.
- Underscore-prefixed files are helper/smoke configs and may be exempt from
  published experiment naming rules.
- Prefer adding a new YAML over mutating a published config. Record selected
  runs only in `docs/results_index.yaml`; detailed interpretation belongs with
  reproduction artifacts or paper sources.

## Schema

See `_schema.md` for supported blocks and field meanings. Common entry points:

```bash
uv run python scripts/run_experiment.py -e <id>
uv run python scripts/run_eval.py -e <id>
uv run python scripts/run_psi.py -e <id>
```
