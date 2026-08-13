# Third-party notices

This file records third-party code, models, and datasets referenced by this
project. The release contains source code, configurations, documentation, and
paper evidence only. It does not include model weights, training checkpoints,
adapters, silver labels, corpus text, relevance judgments, downloaded indexes,
credentials, or hosted-model requests and responses.

Users obtain models and datasets from their upstream sources under the
applicable terms. Verbatim licence texts for source code adapted here are under
[`third_party_licenses/`](third_party_licenses/).

## Models

No model weights are redistributed by this repository.

| Checkpoint | Licence | Role |
| --- | --- | --- |
| `Qwen/Qwen3-{1.7B,4B,8B,14B,32B}`, `Qwen3-4B-Instruct-2507` | Apache-2.0 | fine-tuned base and teacher |
| `ibm-granite/granite-4.1-{3b,8b,30b}` | Apache-2.0 | fine-tuned base, teacher, reader |
| `google/gemma-4-{E2B,E4B,26B-A4B,31B}-it` | Apache-2.0 | fine-tuned base and teacher |
| `Qwen/Qwen3-Reranker-4B` | Apache-2.0 | pointwise reference and specialized-base control |
| `mixedbread-ai/mxbai-rerank-{base,large}-v2` | Apache-2.0 | evaluation only |
| `Skywork/Skywork-Reward-V2-Qwen3-4B` | Apache-2.0 | evaluation only |
| `castorini/rank_zephyr_7b_v1_full` | MIT | evaluation only |
| `llm-blender/PairRM-hf` | MIT | evaluation only |
| `jinaai/jina-reranker-v3` | CC-BY-NC-4.0 | evaluation only |
| `BAAI/bge-base-en-v1.5` | MIT | dense first-stage retrieval |
| `naver/splade-cocondenser-ensembledistil` | CC-BY-NC-SA-4.0 | sparse first-stage, evaluation only |

The configuration audit keeps the non-commercial Jina and SPLADE checkpoints
out of teacher, student, fine-tuning-base, and training-candidate roles. BGE and
SPLADE weights are fetched at run time. RankZephyr is an MIT-licensed reference
checkpoint distilled upstream from GPT-4 listwise rankings.

## Adapted source code

The entries below identify the upstream licence, reviewed revision, local
files, and material modifications. The full Apache-2.0 text and the PINE and
SQuAD MIT texts are included under
[`third_party_licenses/`](third_party_licenses/). The corresponding source files
also retain copyright, attribution, and modification notices in their
docstrings. None of these audited upstream repositories ships a root `NOTICE`
file requiring additional notice text.

### PINE — MIT

Copyright (c) 2025 Ziqi Wang. <https://github.com/wzq016/PINE>

`src/presentation_dependence/rerankers/pine.py` adapts the position-invariant attention
implementation from PINE revision
[`a1af25b`](https://github.com/wzq016/PINE/tree/a1af25b790c690e3f2bf33f7aa9e1c382598331e).
It is scoped to Qwen-family expected-grade rerankers and raises on context
overflow where upstream truncates.

### LLM-Blender / PairRM — Apache-2.0 code; MIT checkpoint

Jiang, Ren & Lin, “LLM-Blender.”
<https://github.com/yuchenlin/LLM-Blender>

`src/presentation_dependence/rerankers/_pairrm_model.py` adapts the
`DebertaV2PairRM` architecture from revision
[`5d38ca9`](https://github.com/yuchenlin/LLM-Blender/blob/5d38ca9528cdeb23e89d40500ced511d08bb5996/llm_blender/pair_ranker/pairrm.py).
`src/presentation_dependence/rerankers/pairrm.py` adapts the published pair encoding.
The local path is inference-only and uses current Transformers building blocks.
The source repository is Apache-2.0; the separately fetched `PairRM-hf`
checkpoint repository is tagged MIT.

### RankLLM — Apache-2.0

Castorini. <https://github.com/castorini/rank_llm>

`src/presentation_dependence/rerankers/rankzephyr.py` reproduces the RankZephyr prompt
template and related prompt formatting from revision
[`e2ceebe`](https://github.com/castorini/rank_llm/tree/e2ceebe68126430c0960f7282e14c709865d66cb).
The wrapper is reimplemented against this project's reranker interface; the
prompt template remains byte-for-byte because the checkpoint was trained on it.

### RankGPT — Apache-2.0

Sun et al. <https://github.com/sunnweiwei/RankGPT>

`src/presentation_dependence/rerankers/rankzephyr.py` follows RankGPT's permutation
parsing and passage truncation conventions.
`src/presentation_dependence/rerankers/_windowing.py` adapts the end-to-start
sliding-window schedule from revision
[`0d62bc3`](https://github.com/sunnweiwei/RankGPT/tree/0d62bc3855c7c118048a7c47c18e719b938e291a).
The local implementation adds a final zero-based window when the stride does
not land exactly on zero and accepts bracketed identifiers only.

### Hugging Face Transformers — Apache-2.0

Copyright 2020 The HuggingFace Inc. team.
<https://github.com/huggingface/transformers>

`src/presentation_dependence/self_distill/engines/_tf5_compat.py` reimplements the
Transformers 4.57.6 `all_special_tokens_extended` behavior from revision
[`753d611`](https://github.com/huggingface/transformers/tree/753d61104116eefc8ffc977327b441ee0c8d599f)
against the Transformers 5.4.0 storage API at revision
[`276f140`](https://github.com/huggingface/transformers/tree/276f1402020831d949c2e1a80574a5603995de23).

### SQuAD evaluation — MIT; HotpotQA evaluation — Apache-2.0

SQuAD Explorer: Copyright (c) 2020 Pranav Rajpurkar.
[`evaluate-v2.0.py` at `09eac99`](https://github.com/rajpurkar/SQuAD-explorer/blob/09eac9971f46889fa057ff2c870bf71092ba9d55/evaluate-v2.0.py).

HotpotQA: Copyright 2018 Zhilin Yang, Peng Qi, Saizheng Zhang.
[`hotpot_evaluate_v1.py` at `fa3a363`](https://github.com/hotpotqa/hotpot/blob/fa3a36370899e1d85822de61e58c85ea19993154/hotpot_evaluate_v1.py).

`src/presentation_dependence/reader/answer_eval.py` reimplements their shared answer
normalization and EM/F1 behavior against this project's reader types. It keeps
the standard helper names and HotpotQA yes/no/noanswer handling. The SQuAD MIT
notice is retained conservatively.

### mxbai-rerank compatibility shim — Apache-2.0

Copyright 2025 mixedbread ai inc.

`src/presentation_dependence/rerankers/mxbai_pointwise.py` patches
`MxbaiRerankV2.prepare_inputs` from `mxbai-rerank==0.1.6` with an equivalent
Transformers-5 implementation. Reviewed source:
[`c27224a`](https://github.com/mixedbread-ai/mxbai-rerank/blob/c27224ad2cb2622fc7a4260778b8ad9d2a6b6f0a/mxbai_rerank/mxbai_rerank_v2.py).

### CapCal method attribution

`src/presentation_dependence/rerankers/capcal.py` reimplements the content-agnostic
probability calibration described by Lv et al., *Learning from Emptiness:
De-biasing Listwise Rerankers with Content-Agnostic Probability Calibration*,
ACL 2026 ([arXiv:2604.10150](https://arxiv.org/abs/2604.10150)). No upstream
source code is copied. The local implementation applies the published
decomposition to expected-grade distributions and includes a scalar fallback.

### Model-card prompts and scoring protocols

These wrappers reproduce published prompt text or scoring procedures without
vendoring model source:

| File | Model card | Reproduced behavior |
| --- | --- | --- |
| `src/presentation_dependence/rerankers/qwen3.py` | `Qwen/Qwen3-Reranker-4B` | pair prompt body and yes/no scoring |
| `src/presentation_dependence/rerankers/skywork_bt.py` | `Skywork/Skywork-Reward-V2-*` | chat-template application and duplicate-BOS guard |
| `src/presentation_dependence/rerankers/jina_listwise.py` | `jinaai/jina-reranker-v3` | `rerank()` contract and `block_size` default |

The Jina wrapper loads the model repository's custom class from the Hub at run
time; no Jina source or weights are included here.

## Datasets

No corpus, passage text, relevance judgment, or downloaded index is included.
`data/` is gitignored. The setup scripts fetch source material from upstream
and write local fixtures.

The repository includes aggregate paper evidence and compact sampling manifests
under `configs/reproduction/fixtures/taupsi-qids/`. Those manifests contain
bare query identifiers, not query or document text.

| Dataset | Published terms or access | Role |
| --- | --- | --- |
| MS MARCO passage | non-commercial research only; no licence or other IP rights granted | primary training pool |
| `hotpot_qa` (`distractor`) | CC-BY-SA-4.0 | multi-document QA support ranking |
| `allenai/reward-bench-2` | ODC-BY; embedded model outputs have separate terms | response-quality evaluation |
| `THU-KEG/RM-Bench` | ODC-BY | response-quality evaluation |
| `openbmb/UltraFeedback` | MIT; includes prompts and model outputs from multiple sources | response-quality training and evaluation |
| `berkeley-nest/Nectar` | Apache-2.0 metadata with additional dataset-card conditions | response-quality evaluation |
| `dgslibisey/MuSiQue` | proxy has no licence metadata; canonical source states CC-BY-4.0 | multi-hop QA support ranking |
| `voidful/2WikiMultihopQA` | proxy has no licence metadata; canonical source states Apache-2.0 | multi-hop QA support ranking |
| `lmarena-ai/PPE-MATH-Best-of-K` | prompts MIT; model outputs follow provider terms | response-quality evaluation |
| `lmarena-ai/PPE-MMLU-Pro-Best-of-K` | prompts MIT; model outputs follow provider terms | response-quality evaluation |
| Climate-FEVER retrieval via Pyserini/BEIR | BEIR metadata states CC-BY-SA-4.0 | passage-reranking evaluation |
| `tdiggelm/climate_fever` | dataset-content licence not stated | verdict labels |
| FiQA | original source states non-commercial; BEIR metadata states CC-BY-SA-4.0 | passage-reranking evaluation |
| NFCorpus, ArguAna, Touché-2020, DBpedia-Entity | BEIR packaging; underlying dataset terms also apply | passage-reranking evaluation |
| SciFact | dataset card states CC-BY-NC-2.0; upstream repository lists component licences | passage-reranking and verdict evaluation |
| TREC-COVID / CORD-19 | CORD-19 text/data-mining grant; underlying article rights retained | passage-reranking evaluation |
| Signal-1M | CC-BY-NC-3.0 and Signal Media access terms; access-gated | passage-reranking evaluation |
| TREC-NEWS | NIST access terms; access-gated | passage-reranking evaluation |
| Robust04 | TREC/NIST access terms; access-gated | passage-reranking evaluation |
| TREC Deep Learning topics and qrels | NIST terms | passage-reranking evaluation |

Signal-1M, TREC-NEWS, and Robust04 are skipped by default and require an
explicit opt-in after the operator has obtained access.

The Climate-FEVER verdict pipeline combines the Pyserini/BEIR retrieval
fixtures with labels fetched separately from `tdiggelm/climate_fever`; neither
source is redistributed here.

Dataset-level MIT or ODC-BY metadata does not replace applicable source-dataset
or model-provider terms for embedded prompts and generated responses. This
repository does not redistribute those records.

## Optional hosted-model client

The OpenAI-compatible client is optional, provider-neutral infrastructure.
Running it sends configured corpus text to the operator-selected endpoint.
Operators are responsible for ensuring that their corpus use and provider
agreement permit that processing. This repository includes no credentials or
generated provider output.

## Runtime dependencies

Runtime and optional dependencies are declared in `pyproject.toml` and pinned
in `uv.lock`. They are not bundled in this source release and remain under
their respective upstream terms. Container images are not distributed.
