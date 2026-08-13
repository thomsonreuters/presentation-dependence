# `reader/`

Frozen downstream readers that consume a scorer's rankings. The reader model,
decoding, and prompt stay fixed so measured answer or verdict changes come from
scorer-induced context, not from training the reader. Configs live under
`configs/reader/` and use a separate schema from `reranker/data/eval`
experiments (`phase` is required).

| Phase | What it measures | Score input | Entry |
| --- | --- | --- | --- |
| `R3` | Scorer → answer bridge | `psi/beta_gamma_scores.parquet` | `scripts/run_reader.py`, or `scripts/train/reader_entrypoint.py` in a container |
| `V3` | Scorer → verdict bridge | `psi/beta_gamma_scores.parquet` | `scripts/run_reader.py`, or `scripts/train/reader_entrypoint.py` in a container |

Bridge phases (`R3`/`V3`) reconstruct rankings from the stored PSI score log via
`eval.score_log.read_score_log`. They do not call the scorer again. Generation is
greedy (T=0) with a pinned prompt. Answer/verdict flip rate across scorer-input
permutations is the downstream analogue of τ-PSI. Both reader entrypoints read
`phase` from the config and select the answer or verdict bridge accordingly.

`VLLMReaderEngine` currently loads `model_name` without a separate revision
argument. A YAML `revision` is documentation unless the model name points to an
immutable local snapshot; use such a snapshot for publication runs.

## QA bridge (`R3`)

- [`pipeline.py`](pipeline.py): `run_reader_bridge`: load score log + QA
  `fixture.jsonl`, rebuild ranked top-k in scorer order, generate answers, score
  against `answers.jsonl`. Emits per-query EM/F1, answer flip rate, EM/F1
  spread, and the gold passage's reader slot. Data prep runs offline; generation
  needs a `ReaderEngine`.
- [`prompt.py`](prompt.py): pinned reader prompt and extraction
  (`build_reader_messages`, `extract_answer`).
- [`answer_eval.py`](answer_eval.py): EM/F1 and stability metrics
  (`answer_flip_rate`, `em_spread`, `f1_spread`, `best_em`, `best_f1`).

## Verdict bridge (`V3`)

- [`verdict_pipeline.py`](verdict_pipeline.py): `run_verdict_bridge` (scorer →
  verdict). Gold labels from `verdicts.jsonl`; Climate-FEVER `DISPUTED` is
  excluded and counted.
- [`verdict_prompt.py`](verdict_prompt.py): pinned prompt and extraction
  (`build_verdict_messages`, `extract_verdict`).
- [`verdict_eval.py`](verdict_eval.py): three-way labels
  `SUPPORTED` / `REFUTED` / `NEI` and stability metrics (`verdict_accuracy`,
  `verdict_flip_rate`, `accuracy_spread`, `normalize_verdict`).
- [`verdict_ids.py`](verdict_ids.py): semantic and legacy run-id helpers
  (`semantic_verdict_run_id`).

## Backends and config

- [`engine.py`](engine.py): `VLLMReaderEngine` (production: frozen instruction
  model, greedy/T=0; follows `self_distill.engines.vllm` construction rules) and
  `EchoReaderEngine` (deterministic stub for offline tests). Both take chat
  messages and return raw strings; extraction is in `prompt.py` /
  `verdict_prompt.py`.
- [`config.py`](config.py): loader/validator for `configs/reader/<id>.yaml`
  (`VALID_PHASES = R3, V3`).
- [`run_io.py`](run_io.py): how a reader run receives its config and reports
  its numbers. Only the container entrypoint uses it: a config may arrive
  inline as one shell-safe token rather than a path, and results flatten into
  `METRIC name=value` lines.

[`__init__.py`](__init__.py) is documentation-only; import pipelines and engines
from their modules.

## Notes

- Prefer a reader family different from the scorer for clean attribution (see
  `configs/reader/README.md`).
- QA datasets wired in the local driver: HotpotQA, 2Wiki, MuSiQue.
