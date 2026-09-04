# S3 LoRA adapters

Bucket: `s3://a204383-ml-workspace-lltknowledgeuurz-use1/`
Account: `a204383` (`451191978663`), `us-east-1`
All paths below are LoRA adapters, not merged models.

Qwen3-4B base revision: `Qwen/Qwen3-4B` @ `1cfa9a7208912126459214e8b04321603b3df60c`

---

## Table 1 — Qwen3-4B OC-SFT (seed 42)

### Passage reranking (MS MARCO 30k, λ=5.0, step 1000, held-out argmax)

```
s3://a204383-ml-workspace-lltknowledgeuurz-use1/slm_training/checkpoints/qwen3-4b-nonthink-k1-supervised-consistency-lambda500-warmup500-msmarco-30k/qwen3-4b-nonthink-k1-supervised-consistency-lambda500-warmup500-msmarco-30k-ee68c238-1780049766/student/checkpoint-step-001000/
```

Base: `Qwen/Qwen3-4B` @ `1cfa9a7208912126459214e8b04321603b3df60c`

### Multi-document QA (HotpotQA support 30k, λ=3.0, step 300, held-out argmax)

```
s3://a204383-ml-workspace-lltknowledgeuurz-use1/slm_training/checkpoints/qwen3-4b-nonthink-k1-supervised-consistency-lambda300-warmup500-hotpotqa-support-30k/qwen3-4b-nonthink-k1-supervised-consistency-lambda300-warmup500-hotpotqa-support-30k-49385bd0-1780487879/student/checkpoint-step-000300/
```

Base: `Qwen/Qwen3-4B` @ `1cfa9a7208912126459214e8b04321603b3df60c`

### Response ranking (UltraFeedback 30k, λ=1.0, step 469)

```
s3://a204383-ml-workspace-lltknowledgeuurz-use1/slm_training/checkpoints/qwen3-4b-nonthink-k1-supervised-consistency-lambda100-warmup500-ultrafeedback-rq-30k/qwen3-4b-nonthink-k1-supervised-consistency-lambda100-warmup500-ultrafeedback-rq-30k-e6e7c4b6-1782742779/student/checkpoint-step-000469/
```

Base: `Qwen/Qwen3-4B` @ `1cfa9a7208912126459214e8b04321603b3df60c`

Held-out nDCG argmax was step 50. Nectar eval used step 469 (SHA match to `slm_training/data/rr-lora-4b-ocl1/`).

---

## Qwen3-1.7B OC-SFT (other tasks)

### Multi-document QA (HotpotQA support 30k, λ=3.0, step 469, held-out argmax)

```
s3://a204383-ml-workspace-lltknowledgeuurz-use1/slm_training/checkpoints/qwen3-1p7b-nonthink-k1-supervised-consistency-lambda300-warmup500-hotpotqa-support-30k/qwen3-1p7b-nonthink-k1-supervised-consistency-lambda300-warmup500-hotpotqa-support-30k-a8a44192-1780307753/student/checkpoint-step-000469/
```

Base: `Qwen/Qwen3-1.7B`

### Response ranking (UltraFeedback 30k, λ=0.5, step 469)

```
s3://a204383-ml-workspace-lltknowledgeuurz-use1/slm_training/checkpoints/qwen3-1p7b-nonthink-k1-supervised-consistency-lambda050-warmup500-ultrafeedback-rq-30k/qwen3-1p7b-nonthink-k1-supervised-consistency-lambda050-warmup500-ultrafeedback-rq-30k-e6e7c4b6-1782754605/student/checkpoint-step-000469/
```

Base: `Qwen/Qwen3-1.7B`

Held-out nDCG argmax was step 100. Nectar eval used step 469 (SHA match to `slm_training/data/rr-lora-ocl05/`).

---

## Qwen3 self-distill OC-SFT scale (passage reranking, K=1)

Lambda/step from `configs/reproduction/evidence/lambda-selection/decisions.json`.

### Qwen3-1.7B (λ=4.0, step 1400)

```
s3://a204383-ml-workspace-lltknowledgeuurz-use1/slm_training/checkpoints/qwen3-1p7b-nonthink-k1-supervised-consistency-lambda400-warmup500-msmarco-30k/qwen3-1p7b-nonthink-k1-supervised-consistency-lambda400-warmup500-msmarco-30k-ee68c238-1780049891/student/checkpoint-step-001400/
```

Base: `Qwen/Qwen3-1.7B`

### Qwen3-4B (λ=5.0, step 1000)

Same adapter as Table 1 passage reranking.

Base: `Qwen/Qwen3-4B` @ `1cfa9a7208912126459214e8b04321603b3df60c`

### Qwen3-8B (λ=5.0, step 1400)

```
s3://a204383-ml-workspace-lltknowledgeuurz-use1/slm_training/checkpoints/qwen3-8b-nonthink-k1-supervised-consistency-lambda500-warmup500-msmarco-30k/qwen3-8b-nonthink-k1-supervised-consistency-lambda500-warmup500-msmarco-30k-a8a44192-1780391198/student/checkpoint-step-001400/
```

Base: `Qwen/Qwen3-8B`

### Qwen3-14B (λ=5.0, step 1400)

```
s3://a204383-ml-workspace-lltknowledgeuurz-use1/slm_training/checkpoints/qwen3-14b-nonthink-k1-supervised-consistency-lambda500-warmup500-msmarco-30k/qwen3-14b-nonthink-k1-supervised-consistency-lambda500-warmup500-msmarco-30k-49385bd0-1780496839/student/checkpoint-step-001400/
```

Base: `Qwen/Qwen3-14B`

### Qwen3-32B (λ=4.0, step 2000)

```
s3://a204383-ml-workspace-lltknowledgeuurz-use1/slm_training/checkpoints/qwen3-32b-nonthink-k1-supervised-consistency-lambda400-warmup500-msmarco-30k/qwen3-32b-nonthink-k1-supervised-consistency-lambda400-warmup500-msmarco-30k-49385bd0-1780632427/student/checkpoint-step-002000/
```

Base: `Qwen/Qwen3-32B`

---

## Gemma-4 self-distill OC-SFT scale (passage reranking, K=1)

Lambda/step from `configs/reproduction/evidence/lambda-selection/decisions.json`.

### Gemma-4-E2B-it (λ=5.0, step 1200, provisional)

```
s3://a204383-ml-workspace-lltknowledgeuurz-use1/slm_training/checkpoints/gemma4-e2b-k1-supervised-consistency-lambda500-warmup500-msmarco-30k/gemma4-e2b-k1-supervised-consistency-lambda500-warmup500-msmarco-30k-a8a44192-1780387418/student/checkpoint-step-001200/
```

Base: `google/gemma-4-E2B-it`

### Gemma-4-E4B-it (λ=2.0, step 2200, provisional)

```
s3://a204383-ml-workspace-lltknowledgeuurz-use1/slm_training/checkpoints/gemma4-k1-supervised-consistency-lambda200-warmup500-msmarco-30k/gemma4-k1-supervised-consistency-lambda200-warmup500-msmarco-30k-ee68c238-1780030692/student/checkpoint-step-002200/
```

Base: `google/gemma-4-E4B-it` @ `d6436b3d62967e1af08bbb046c6300b2a9ae8e85`

Downstream evals used step 600 (warmup spike). Canonical is step 2200.

### Gemma-4-26B-A4B-it (λ=3.0, step 1000, provisional)

```
s3://a204383-ml-workspace-lltknowledgeuurz-use1/slm_training/checkpoints/gemma4-26b-a4b-k1-supervised-consistency-lambda300-warmup500-msmarco-30k/gemma4-26b-a4b-k1-supervised-consistency-lambda300-warmup500-msmarco-30k-e6e7c4b6-1783080138/student/checkpoint-step-001000/
```

Base: `google/gemma-4-26B-A4B-it`

### Gemma-4-31B-it (λ=2.0, step 1200)

```
s3://a204383-ml-workspace-lltknowledgeuurz-use1/slm_training/checkpoints/gemma4-31b-k1-supervised-consistency-lambda200-warmup500-msmarco-30k/gemma4-31b-k1-supervised-consistency-lambda200-warmup500-msmarco-30k-49385bd0-1780670961/student/checkpoint-step-001200/
```

Base: `google/gemma-4-31B-it`

---

## Granite-4.1 self-distill OC-SFT scale (passage reranking, K=1)

Lambda/step from `configs/reproduction/evidence/lambda-selection/decisions.json`.

### Granite-4.1-3B (λ=5.0, step 1000)

```
s3://a204383-ml-workspace-lltknowledgeuurz-use1/slm_training/checkpoints/granite-41-3b-k1-supervised-consistency-lambda500-warmup500-msmarco-30k/granite-41-3b-k1-supervised-consistency-lambda500-warmup500-msmarco-30k-a8a44192-1780408238/student/checkpoint-step-001000/
```

Base: `ibm-granite/granite-4.1-3b`

### Granite-4.1-8B (λ=5.0, step 1000)

```
s3://a204383-ml-workspace-lltknowledgeuurz-use1/slm_training/checkpoints/granite-41-8b-k1-supervised-consistency-lambda500-warmup500-msmarco-30k/granite-41-8b-k1-supervised-consistency-lambda500-warmup500-msmarco-30k-a8a44192-1780448793/student/checkpoint-step-001000/
```

Base: `ibm-granite/granite-4.1-8b` @ `1504002f650e656a0a3789d99574df12e3e94ed0`

### Granite-4.1-30B (λ=4.0, step 1400, provisional)

```
s3://a204383-ml-workspace-lltknowledgeuurz-use1/slm_training/checkpoints/granite-41-30b-k1-supervised-consistency-lambda400-warmup500-msmarco-30k/granite-41-30b-k1-supervised-consistency-lambda400-warmup500-msmarco-30k-49385bd0-1780845616/student/checkpoint-step-001400/
```

Base: `ibm-granite/granite-4.1-30b` @ `4fae6278f7132abf5e971f9de49ebbad09c54cce`

---

## K=1 SFT scale (passage reranking, no consistency loss)

Ablation twin of each OC-SFT student above. Step is held-out MS MARCO nDCG@10 argmax.

### Qwen3-1.7B (step 1400)

```
s3://a204383-ml-workspace-lltknowledgeuurz-use1/slm_training/checkpoints/qwen3-1p7b-nonthink-sft-msmarco-30k-k1-labels/qwen3-1p7b-nonthink-sft-msmarco-30k-k1-labels-ffee9e9e-1779953891/student/checkpoint-step-001400/
```

Base: `Qwen/Qwen3-1.7B`

### Qwen3-4B (step 1400)

```
s3://a204383-ml-workspace-lltknowledgeuurz-use1/slm_training/checkpoints/qwen3-4b-nonthink-sft-msmarco-30k-k1-labels/qwen3-4b-nonthink-sft-msmarco-30k-k1-labels-ffee9e9e-1779963138/student/checkpoint-step-001400/
```

Base: `Qwen/Qwen3-4B` @ `1cfa9a7208912126459214e8b04321603b3df60c`

### Qwen3-8B (step 2000)

```
s3://a204383-ml-workspace-lltknowledgeuurz-use1/slm_training/checkpoints/qwen3-8b-nonthink-sft-msmarco-30k-k1-labels/qwen3-8b-nonthink-sft-msmarco-30k-k1-labels-ee68c238-1780274490/student/checkpoint-step-002000/
```

Base: `Qwen/Qwen3-8B`

### Qwen3-14B (step 200)

```
s3://a204383-ml-workspace-lltknowledgeuurz-use1/slm_training/checkpoints/qwen3-14b-nonthink-sft-msmarco-30k-k1-labels/qwen3-14b-nonthink-sft-msmarco-30k-k1-labels-ee68c238-1780275416/student/checkpoint-step-000200/
```

Base: `Qwen/Qwen3-14B`

Held-out nDCG peaked at step 200 (0.422); later steps sit ~0.38–0.41.

### Qwen3-32B (step 2000)

```
s3://a204383-ml-workspace-lltknowledgeuurz-use1/slm_training/checkpoints/qwen3-32b-nonthink-sft-msmarco-30k-k1-labels/qwen3-32b-nonthink-sft-msmarco-30k-k1-labels-49385bd0-1780632302/student/checkpoint-step-002000/
```

Base: `Qwen/Qwen3-32B`

### Gemma-4-E2B-it (step 800)

```
s3://a204383-ml-workspace-lltknowledgeuurz-use1/slm_training/checkpoints/gemma4-e2b-sft-msmarco-30k-k1-labels/gemma4-e2b-sft-msmarco-30k-k1-labels-a8a44192-1780387129/student/checkpoint-step-000800/
```

Base: `google/gemma-4-E2B-it`

### Gemma-4-E4B-it (step 1600)

```
s3://a204383-ml-workspace-lltknowledgeuurz-use1/slm_training/checkpoints/gemma4-sft-msmarco-30k-k1-labels/gemma4-sft-msmarco-30k-k1-labels-ee68c238-1780275483/student/checkpoint-step-001600/
```

Base: `google/gemma-4-E4B-it` @ `d6436b3d62967e1af08bbb046c6300b2a9ae8e85`

### Gemma-4-26B-A4B-it (step 1000)

```
s3://a204383-ml-workspace-lltknowledgeuurz-use1/slm_training/checkpoints/gemma4-26b-a4b-sft-msmarco-30k-k1-labels/gemma4-26b-a4b-sft-msmarco-30k-k1-labels-e6e7c4b6-1783080085/student/checkpoint-step-001000/
```

Base: `google/gemma-4-26B-A4B-it`

### Gemma-4-31B-it (step 1600)

```
s3://a204383-ml-workspace-lltknowledgeuurz-use1/slm_training/checkpoints/gemma4-31b-sft-msmarco-30k-k1-labels/gemma4-31b-sft-msmarco-30k-k1-labels-a2bee7f8-1780967980/student/checkpoint-step-001600/
```

Base: `google/gemma-4-31B-it`

### Granite-4.1-3B (step 1000)

```
s3://a204383-ml-workspace-lltknowledgeuurz-use1/slm_training/checkpoints/granite-41-3b-sft-msmarco-30k-k1-labels/granite-41-3b-sft-msmarco-30k-k1-labels-a8a44192-1780386130/student/checkpoint-step-001000/
```

Base: `ibm-granite/granite-4.1-3b`

### Granite-4.1-8B (step 1000)

```
s3://a204383-ml-workspace-lltknowledgeuurz-use1/slm_training/checkpoints/granite-41-8b-sft-msmarco-30k-k1-labels/granite-41-8b-sft-msmarco-30k-k1-labels-a8a44192-1780448680/student/checkpoint-step-001000/
```

Base: `ibm-granite/granite-4.1-8b` @ `1504002f650e656a0a3789d99574df12e3e94ed0`

### Granite-4.1-30B (step 1800)

```
s3://a204383-ml-workspace-lltknowledgeuurz-use1/slm_training/checkpoints/granite-41-30b-sft-msmarco-30k-k1-labels/granite-41-30b-sft-msmarco-30k-k1-labels-49385bd0-1780845563/student/checkpoint-step-001800/
```

Base: `ibm-granite/granite-4.1-30b` @ `4fae6278f7132abf5e971f9de49ebbad09c54cce`

---

## K=10 SFT scale (passage reranking, no consistency loss)

Same students as K=1 SFT. Step is held-out MS MARCO nDCG@10 argmax.

### Qwen3-1.7B (step 1800)

```
s3://a204383-ml-workspace-lltknowledgeuurz-use1/slm_training/checkpoints/qwen3-1p7b-nonthink-sft-msmarco-30k-k10-labels/qwen3-1p7b-nonthink-sft-msmarco-30k-k10-labels-ffee9e9e-1779953912/student/checkpoint-step-001800/
```

Base: `Qwen/Qwen3-1.7B`

### Qwen3-4B (step 1200)

```
s3://a204383-ml-workspace-lltknowledgeuurz-use1/slm_training/checkpoints/qwen3-4b-nonthink-sft-msmarco-30k-k10-labels/qwen3-4b-nonthink-sft-msmarco-30k-k10-labels-ffee9e9e-1779963151/student/checkpoint-step-001200/
```

Base: `Qwen/Qwen3-4B` @ `1cfa9a7208912126459214e8b04321603b3df60c`

### Qwen3-8B (step 1800)

```
s3://a204383-ml-workspace-lltknowledgeuurz-use1/slm_training/checkpoints/qwen3-8b-nonthink-sft-msmarco-30k-k10-labels/qwen3-8b-nonthink-sft-msmarco-30k-k10-labels-ee68c238-1780274520/student/checkpoint-step-001800/
```

Base: `Qwen/Qwen3-8B`

### Qwen3-14B (step 1000)

```
s3://a204383-ml-workspace-lltknowledgeuurz-use1/slm_training/checkpoints/qwen3-14b-nonthink-sft-msmarco-30k-k10-labels/qwen3-14b-nonthink-sft-msmarco-30k-k10-labels-ee68c238-1780275437/student/checkpoint-step-001000/
```

Base: `Qwen/Qwen3-14B`

### Qwen3-32B (step 2305)

```
s3://a204383-ml-workspace-lltknowledgeuurz-use1/slm_training/checkpoints/qwen3-32b-nonthink-sft-msmarco-30k-k10-labels/qwen3-32b-nonthink-sft-msmarco-30k-k10-labels-49385bd0-1780632325/student/checkpoint-step-002305/
```

Base: `Qwen/Qwen3-32B`

### Gemma-4-E2B-it (step 1200)

```
s3://a204383-ml-workspace-lltknowledgeuurz-use1/slm_training/checkpoints/gemma4-e2b-sft-msmarco-30k-k10-labels/gemma4-e2b-sft-msmarco-30k-k10-labels-a8a44192-1780387109/student/checkpoint-step-001200/
```

Base: `google/gemma-4-E2B-it`

### Gemma-4-E4B-it (step 1800)

```
s3://a204383-ml-workspace-lltknowledgeuurz-use1/slm_training/checkpoints/gemma4-sft-msmarco-30k-k10-labels/gemma4-sft-msmarco-30k-k10-labels-6599e2ca-1779196667/student/checkpoint-step-001800/
```

Base: `google/gemma-4-E4B-it` @ `d6436b3d62967e1af08bbb046c6300b2a9ae8e85`

### Gemma-4-26B-A4B-it (step 1400)

```
s3://a204383-ml-workspace-lltknowledgeuurz-use1/slm_training/checkpoints/gemma4-26b-a4b-sft-msmarco-30k-k10-labels/gemma4-26b-a4b-sft-msmarco-30k-k10-labels-e6e7c4b6-1783080068/student/checkpoint-step-001400/
```

Base: `google/gemma-4-26B-A4B-it`

### Gemma-4-31B-it (step 1200)

```
s3://a204383-ml-workspace-lltknowledgeuurz-use1/slm_training/checkpoints/gemma4-31b-sft-msmarco-30k-k10-labels/gemma4-31b-sft-msmarco-30k-k10-labels-a2bee7f8-1780931064/student/checkpoint-step-001200/
```

Base: `google/gemma-4-31B-it`

### Granite-4.1-3B (step 2200)

```
s3://a204383-ml-workspace-lltknowledgeuurz-use1/slm_training/checkpoints/granite-41-3b-sft-msmarco-30k-k10-labels/granite-41-3b-sft-msmarco-30k-k10-labels-a8a44192-1780386153/student/checkpoint-step-002200/
```

Base: `ibm-granite/granite-4.1-3b`

### Granite-4.1-8B (step 1800)

```
s3://a204383-ml-workspace-lltknowledgeuurz-use1/slm_training/checkpoints/granite-41-8b-sft-msmarco-30k-k10-labels/granite-41-8b-sft-msmarco-30k-k10-labels-6599e2ca-1779299571/student/checkpoint-step-001800/
```

Base: `ibm-granite/granite-4.1-8b` @ `1504002f650e656a0a3789d99574df12e3e94ed0`

### Granite-4.1-30B (step 1400)

```
s3://a204383-ml-workspace-lltknowledgeuurz-use1/slm_training/checkpoints/granite-41-30b-sft-msmarco-30k-k10-labels/granite-41-30b-sft-msmarco-30k-k10-labels-49385bd0-1780845580/student/checkpoint-step-001400/
```

Base: `ibm-granite/granite-4.1-30b` @ `4fae6278f7132abf5e971f9de49ebbad09c54cce`

---

## Teacher variants (passage reranking)

### Qwen3-32B teacher → Qwen3-4B (λ=5.0, step 400, held-out argmax)

```
s3://a204383-ml-workspace-lltknowledgeuurz-use1/slm_training/checkpoints/ts-qwen3-32b-to-qwen3-4b-k1-supcon-l500-warmup500-msmarco-30k/ts-qwen3-32b-to-qwen3-4b-k1-supcon-l500-warmup500-msmarco-30k-a2bee7f8-1781088580/student/checkpoint-step-000400/
```

Base: `Qwen/Qwen3-4B` @ `1cfa9a7208912126459214e8b04321603b3df60c`

### Qwen3-32B teacher → Qwen3-1.7B (λ=4.0, step 1200, held-out argmax)

```
s3://a204383-ml-workspace-lltknowledgeuurz-use1/slm_training/checkpoints/ts-qwen3-32b-to-qwen3-1p7b-k1-supcon-l400-warmup500-msmarco-30k/ts-qwen3-32b-to-qwen3-1p7b-k1-supcon-l400-warmup500-msmarco-30k-0d0ad2fa-1781240912/student/checkpoint-step-001200/
```

Base: `Qwen/Qwen3-1.7B`

### GPT-5.4 teacher → Qwen3-4B (λ=5.0, step 1200, provisional)

```
s3://a204383-ml-workspace-lltknowledgeuurz-use1/slm_training/checkpoints/gpt54-genbsc-qwen3-4b-nonthink-k1-supervised-consistency-lambda500-warmup500-msmarco-30k/gpt54-genbsc-qwen3-4b-nonthink-k1-supervised-consistency-lambda500-warmup500-msmarco-30k-a2bee7f8-1780892241/student/checkpoint-step-001200/
```

Base: `Qwen/Qwen3-4B` @ `1cfa9a7208912126459214e8b04321603b3df60c`

### GPT-5.4 teacher → Qwen3-1.7B (λ=4.0, step 1400, provisional)

```
s3://a204383-ml-workspace-lltknowledgeuurz-use1/slm_training/checkpoints/gpt54-genbsc-qwen3-1p7b-nonthink-k1-supervised-consistency-lambda400-warmup500-msmarco-30k/gpt54-genbsc-qwen3-1p7b-nonthink-k1-supervised-consistency-lambda400-warmup500-msmarco-30k-a2bee7f8-1780892280/student/checkpoint-step-001400/
```

Base: `Qwen/Qwen3-1.7B`

### GPT-5.4 teacher → Gemma-4-E2B-it (λ=1.0, step 1800, provisional)

```
s3://a204383-ml-workspace-lltknowledgeuurz-use1/slm_training/checkpoints/gpt54-genbsc-gemma4-e2b-k1-supervised-consistency-lambda100-warmup500-msmarco-30k/gpt54-genbsc-gemma4-e2b-k1-supervised-consistency-lambda100-warmup500-msmarco-30k-a2bee7f8-1780926601/student/checkpoint-step-001800/
```

Base: `google/gemma-4-E2B-it`

### GPT-5.4 teacher → Gemma-4-E4B-it (λ=3.0, step 1400, provisional)

```
s3://a204383-ml-workspace-lltknowledgeuurz-use1/slm_training/checkpoints/gpt54-genbsc-gemma4-e4b-k1-supervised-consistency-lambda300-warmup500-msmarco-30k/gpt54-genbsc-gemma4-e4b-k1-supervised-consistency-lambda300-warmup500-msmarco-30k-a2bee7f8-1781073266/student/checkpoint-step-001400/
```

Base: `google/gemma-4-E4B-it` @ `d6436b3d62967e1af08bbb046c6300b2a9ae8e85`

### GPT-5.4 teacher → Granite-4.1-8B (λ=4.0, step 1200, provisional)

```
s3://a204383-ml-workspace-lltknowledgeuurz-use1/slm_training/checkpoints/ts-gpt54-to-granite41-8b-k1-supcon-l400-warmup500-msmarco-30k/ts-gpt54-to-granite41-8b-k1-supcon-l400-warmup500-msmarco-30k-a2bee7f8-1781092398/student/checkpoint-step-001200/
```

Base: `ibm-granite/granite-4.1-8b` @ `1504002f650e656a0a3789d99574df12e3e94ed0`

---

## Qwen3-Reranker-4B self OC-SFT (λ=4.0, step 1800, provisional)

```
s3://a204383-ml-workspace-lltknowledgeuurz-use1/slm_training/checkpoints/a2-qwen3-reranker-k1-supervised-consistency-lambda400-warmup500-msmarco-30k/a2-qwen3-reranker-k1-supervised-consistency-lambda400-warmup500-msmarco-30k-e6cc4455-1784022727/student/checkpoint-step-001800/
```

Base: `Qwen/Qwen3-Reranker-4B`

---

## Notes

- GPT-5.4, reranker, Gemma-4-E2B, Gemma-4-E4B, Gemma-4-26B-A4B, and Granite-4.1-30B step selections are provisional in `configs/reproduction/evidence/lambda-selection/decisions.json`
