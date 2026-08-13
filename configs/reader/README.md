# `configs/reader/`: frozen-reader bridge configs

Select a frozen reader, decoding settings, and scorer condition
for the R3 answer and V3 verdict bridges. `presentation_dependence.reader.config` loads and
validates them.

Claim verification applies the same reader path through the V3
scorer-to-verdict bridge.

Reader jobs generate free text and compute answer EM/F1, so they use a separate
schema from `reranker/data/eval`. Commands consume these files through an
explicit `--config`.

`load_reader_config` requires `id`, `phase`, `reader` and `dataset`, and
validates that `phase` is `R3` or `V3` and that a bridge phase names a
`scorer.exp_id`. Anything else is passed through unvalidated, so a misspelled
optional key fails silently. Two keys appear in the tracked examples without
being part of that contract: `verdict_qids_filename`, which the QA downstream
stage writes into generated verdict configs, and `data_channel`, which only the
container entrypoint reads and which the local runner ignores.

## Scorer–reader boundary

Reader evaluation loads existing QA students, per-passage scores
(`psi/beta_gamma_scores.parquet`), candidate sets, and scorer presentations.
Across R3 conditions the reader stays fixed and reads those stored scores; it
does not rebuild or retrain the scorer path.

That parquet is not a default PSI artifact. `run_psi.py` writes it only when
the scorer's experiment config sets `robustness.beta_gamma.enabled: true`, so
enable it *before* the scorer run; a reader pointed at a run without it exits
with `no psi/beta_gamma_scores.parquet found`. The tracked
`configs/experiments/example-*-psi.yaml` scorers referenced below already set
it, as do the configs the reproduction pipeline materializes for
`passage-reranking` and `multi-document-qa`.

## Reader lock (pin before downstream comparisons)

- Main reader: `ibm-granite/granite-4.1-8b`. Different family from the Qwen3
  scorers, RAG-oriented, integrated, and mid-size. Confirm the exact
  chat/instruct checkpoint in the model catalog before the first GPU run.
- Same-family control: `Qwen/Qwen3-4B` (Qwen3 OG, non-thinking; same family and
  release line as the Qwen3 scorers). Tests the shared-idiosyncrasy question.
  Pin `enable_thinking: false` to match the non-thinking scorers.
- Optional capability axis: `ibm-granite/granite-4.1-30b` or `Qwen/Qwen3-32B`
  tests whether scorer effects change with reader capacity.
- Decoding: greedy (`temperature=0`), fixed `max_tokens`, fixed prompt template,
  pinned CoT flag (default off). Determinism is required so answer changes are
  attributable to the input context, not sampling.
- The tracked `example-*-reader-*.yaml` configs in this directory carry these
  pins; copy one rather than restating them.

The current `VLLMReaderEngine` constructor receives `model_name` but not the
config's `revision`; recording a revision in YAML does not yet enforce it at
load time. For a citable run, set `reader.model_name` to a prefetched immutable
local snapshot path and record its commit/hash in the run artifact.

## Schema

```yaml
id: reader--answer--granite8b--hotpotqa--oc-sft
phase: R3                          # R3 (answer bridge) | V3 (verdict bridge)
reader:
  engine: vllm                     # vllm | echo (echo = offline plumbing)
  model_name: ibm-granite/granite-4.1-8b
  family: granite                  # must differ from the scorer family for clean R3 attribution
  max_tokens: 64
  cot: false
  enable_thinking: false           # Qwen3 readers: keep thinking off (pinned)
  max_model_len: 8192
dataset: hotpotqa                  # hotpotqa | 2wiki | musique
k_values: [3, 5]                   # R3 top-k report values
canonical_perm: 0                  # perm_idx treated as the canonical scorer order
scorer:                            # existing scorer run to read
  exp_id: multi-document-qa-direct-eval--oc-sft--seed43--hotpotqa
  recipe: OC-SFT
  size: 4b
```

## Naming

Use `reader--<consumer>--<reader>--<dataset>--<condition>`. Reader configs are
downstream consumers of semantic scorer IDs. Confirm result provenance against
tracked configs and `docs/results_index.yaml`.

## Run

```bash
uv run --extra vllm python scripts/run_reader.py \
  --config configs/reader/example-answer-reader-hotpotqa.yaml \
  --scorer-run runs/<semantic-scorer-id>/<trial>

uv run --extra vllm python scripts/run_reader.py \
  --config configs/reader/example-verdict-reader-climate-fever.yaml \
  --scorer-run runs/<semantic-scorer-id>/<trial>

# Container equivalent for V3:
SLM_CHANNEL_READER_DATA="$PWD/data/beir-v1.0.0-climate-fever-test" \
SLM_CHANNEL_SCORER_RUN="$PWD/runs/<semantic-scorer-id>/<trial>/psi" \
SLM_OUTPUT_DIR="$PWD/runs-reader" \
uv run --extra vllm python scripts/train/reader_entrypoint.py \
  --config-path configs/reader/example-verdict-reader-climate-fever.yaml \
  --data-channel reader-data \
  --scorer-channel scorer-run
```

### Scorer-run resolution

`scripts/run_reader.py` accepts `--scorer-run` as a parquet path, a trial
directory, an experiment directory, or a bare experiment id under `--runs-root`
(default `runs/`). When several `psi/beta_gamma_scores.parquet` files match, it
picks the newest by mtime.

Omitting `--scorer-run` is allowed when `--config` sets `scorer.exp_id`: the
runner copies that id into `--scorer-run` and resolves it the same way. That is
the path used by

```bash
uv run python scripts/study.py multi-document-qa downstream-eval execute --run
```

which materializes one reader config per job (R3 or V3 from `phase`) and launches
`scripts/run_reader.py --config <generated.yaml>` with no explicit trial. Pin a
trial with `--scorer-run runs/<exp-id>/<trial>` (or a source binding) when
latest-by-mtime is not the intended scorer.

`run_reader.py` dispatches R3 and V3 from the config's `phase`; the container
entrypoint implements the same two bridge phases. `SLM_*` variables take
precedence; the legacy `SM_CHANNEL_*` and `SM_OUTPUT_DATA_DIR` aliases remain
compatible only for older launchers. Set `SLM_TRIAL_NAME` to choose the output
trial id; legacy launchers may still provide `TRAINING_JOB_NAME`. For the
complete matrix use `study.py multi-document-qa downstream-eval`.

The configured readers are 8B+, so they need a GPU. See `docs/MODELS.md`,
`README.md`, `docs/HARDWARE.md`, and `docs/TROUBLESHOOTING.md`;
use `--reader-engine echo` for offline plumbing.

## Prereq: gold answers

`scripts/run_reader.py` needs the gold-answer sidecar
`data/<slug>/answers.jsonl`. Build it once per dataset (does not touch the
existing fixtures):

```bash
uv run python -m scripts.data.setup_qa_answers --dataset hotpotqa
uv run python -m scripts.data.setup_qa_answers --dataset 2wiki
uv run python -m scripts.data.setup_qa_answers --dataset musique
```
