# `self_distill/`

Self-distillation for batched-pointwise relevance grading. An open-weight
teacher writes K-shot BSC silver labels (expected grade over shuffled
presentations); a student is trained with LoRA on those labels. The default
student recipe is OC-SFT (`objective.type: supervised_consistency`): MSE on the
teacher labels plus a two-view consistency penalty.

Config split:

| Role | Config directory | Entry |
|---|---|---|
| Open-weight teacher (`k_shot_bsc`) | `configs/silver/` | `scripts/run_silver_generation.py` → `KShotBSCTeacher` |
| Student training | [`configs/self-distill/`](../../../configs/self-distill/_schema.md) | `scripts/run_self_distill_sft.py`, or the container entrypoint |

Hosted/API teachers also use `configs/silver/` but run through
`silver_data.SilverGenerator`, not code in `self_distill/`. See
`silver_data/README.md`.

## Teacher and readout

- [`teacher.py`](teacher.py): `KShotBSCTeacher`. For each query: K Fisher-Yates
  permutations (default seeds `0..K-1`, same scheme as `eval/psi_manager.py`),
  score each presentation, map scores back to original doc order, average per
  doc, write `silver/silver_labels.jsonl` plus `silver/manifest.json` under
  `runs/self-distill/<id>/<timestamp>/`. Implements only `protocol: k_shot_bsc`;
  refuses to publish partial aggregates. Configs are loaded from
  `configs/silver/`.
- [`readout.py`](readout.py): continuous-readout primitives: expected value over
  the `{0, 1, 2, 3}` grade tokens from vLLM `prompt_logprobs` or HF
  `output_logits` (`expected_grade`, `resolve_grade_token_ids`).
- [`silver_io.py`](silver_io.py): silver-label record schema (`SilverLabel`)
  and the JSONL / manifest writers. Active generation uses `k_shot_bsc`; the
  reader preserves compatibility with older record shapes.
- [`local_dp.py`](local_dp.py): data-parallel teacher inference, one vLLM
  replica per GPU. Each worker holds the whole model on one device
  (`tensor_parallel_size=1`), runs the ordinary `KShotBSCTeacher` path over its
  own shard of qids, and the parent merges the per-qid outputs back into the
  single-run layout. This keeps the validated TP=1 path instead of taking on
  the TP>1 attention and cudagraph instability, so it needs a model that fits
  on one device. `teacher.local_data_parallel_workers` sets the worker count,
  and means the same thing here as in the container entrypoint.

## Student training

- [`student.py`](student.py): LoRA SFT over silver labels. Supported
  `objective.type` values: `supervised_mse`, `supervised_consistency` (OC-SFT),
  `mean_teacher`, `kl_to_base`. Loss modes for the supervised types: `mse`,
  `kl_vector`, `combined`.
- [`student_data.py`](student_data.py): student-side dataset views and JSONL
  export for regression chunks and supervised-consistency chunks, which are the
  views the default recipes train on. Also builds the permutation groups that
  `student_eval.py` scores for the in-training PSI proxy.
- [`student_eval.py`](student_eval.py): evaluation diagnostics for student
  checkpoints during and after training.
- [`ips_propensity.py`](ips_propensity.py): inverse-propensity slot weights for
  the DebiasFirst baseline. Estimates `p(slot | relevant)` over first-stage
  input slots from the training qrels, then weights the per-(doc, slot) MSE by
  its inverse, so the over-represented early slots stop dominating. The
  functions are pure, and the estimate is written to
  `ips_slot_propensity.json` in the run directory. With uniform weights the
  objective reduces exactly to the shuffled-view augmentation arm.
- [`selection.py`](selection.py): checkpoint and OC-SFT λ selection rules
  (`select_heldout_lambda`, `best_checkpoint`, `select_amortization_matching`).
  `scripts/select_lambda.py` is the operator stage, and it runs
  `select_heldout_lambda` only. `select_amortization_matching` is the reference
  implementation of the response-ranking rule in
  [`EVAL-PROTOCOL.md`](../../../docs/EVAL-PROTOCOL.md); no stage runs it,
  because nothing in this repository produces its per-λ dev stability rows.

## Inference backends

The [`engines/`](engines/README.md) subpackage holds the logit-skeleton scoring
backends (`base`, `hf`, `vllm`) and a Transformers-v5 / vLLM compatibility shim
(`_tf5_compat`) retained only for the Qwen3-4B-Instruct-2507 and
Qwen3-Reranker-4B silver configs on vLLM 0.10.2. Modern vLLM and HF-only jobs
skip its worker hook. Reranker wrappers select the engine via
`reranker.inference_engine: hf | vllm`.

## Public API

[`__init__.py`](__init__.py) re-exports leaf-level entry points only
(`SilverLabel`, `expected_grade`, `resolve_grade_token_ids`,
`write_silver_jsonl`, `write_manifest`). Import the teacher from
`teacher.py` directly; re-exporting it here would close an import cycle through
`eval.runner_setup` → `rerankers` → `readout`.

## Notes

- λ and checkpoint picks go through `scripts/select_lambda.py`; do not hand-pick
  from TensorBoard or a global argmax.
- Teacher artifact layout stays under `runs/self-distill/` even though the YAML
  now lives in `configs/silver/`.
