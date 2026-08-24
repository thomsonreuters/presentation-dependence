# Paper terminology

Use the paper names below in documentation, reports, plots, and other
user-facing output. Historical config identifiers, run paths, and serialized
keys remain unchanged so that existing artifacts and provenance continue to
resolve.

## Methods

| Paper name | Meaning | Internal or historical aliases |
| --- | --- | --- |
| Off the shelf | Untrained prompted scorer | `base`, `off-shelf` |
| CapCal | Content-agnostic logit calibration | `capcal` |
| Round-robin | Round-robin partitioning of a rank-sorted pool across prompts | `interleaved`, `round-robin-chunks` |
| Batched self-consistency (BSC) | Serving-time average over `K` candidate-order permutations | `bsc` |
| Single-order distillation | MSE distillation from one teacher order | `k1-sft`, `k1sft`, K=1 SFT |
| Order-averaged distillation | MSE distillation from a target averaged over `T` teacher orders | `k10-sft`, `k10sft`, K=10 SFT |
| DebiasFirst | Position-aware augmentation with inverse-propensity weighting | `debias-first`, `debiasfirst` |
| Permutation augmentation | Shuffled student views, each anchored to the teacher target | `shuffled-view-augmentation`, `position_augmentation`, `posaug`, `pos-aug-only` |
| Order-consistency SFT (OC-SFT) | Single-order distillation plus a penalty on score disagreement across shuffled views | `oc-sft`, `ocsft`, `oc_sft`, `supcon`, `supervised_consistency` |

In compact tables, use **Single-order**, **Order-averaged**, and **OC-SFT**.
Do not call order-averaged distillation “BSC”: BSC is the inference-time
ensemble, while order-averaged distillation moves its average into a training
target.

## Notation

| Symbol | Paper meaning |
| --- | --- |
| `B` | Candidates sharing one scoring prompt |
| `M` | Random evaluation permutations used to estimate order instability and decision flips |
| `T` | Teacher permutations averaged into an order-averaged distillation target |
| `N` | Student views in an OC-SFT training step |
| `K` | Serving-time permutations averaged by BSC |

Older configuration and artifact schemas use names such as `K_permutations`,
`presentation_seeds`, and `view_seeds` in more than one of these roles. Treat
those as stable implementation keys; use the paper symbols when explaining the
scientific protocol.

## Phenomenon and measurements

**Presentation dependence** is the broad dependence of a candidate score on
prompt construction: candidate order, the rule that partitions a pool across
prompts, slot markers, and answer-skeleton wording. **Order dependence** is the
specific channel measured by permuting candidate order while holding the other
properties fixed.

Use these measurement names:

- **ranking quality**: nDCG@10 for passage reranking and multi-document QA,
  nDCG@1 for response ranking;
- **order instability**: τ-PSI, lower is better;
- **retained-set overlap**: mean pairwise Jaccard between thresholded sets,
  higher is better;
- **answer flip**: rate at which a reader answer changes between permutations,
  lower is better;
- **pair flip**: rate at which a chosen/rejected response pair changes between
  permutations, lower is better.

The score decomposition uses **order-marginal score** for the permutation
average and **order residual** for the order-dependent component. The
difference between the order-marginal score at width `B` and at width one is
the **width effect**.
