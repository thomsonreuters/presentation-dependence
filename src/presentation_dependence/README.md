# `presentation_dependence/`

Trains and evaluates order-consistent second-stage rerankers:
small language models that grade query/document relevance with
batched-pointwise scoring and stay stable under candidate reordering.

Import-only package. Configs live under `configs/`, executable entry
points under `scripts/`, raw job output in `runs/`, and reproduction artifacts
under `build/reproduction/`. For the wider configuration, runtime, and
artifact map, see the [repository map](../../README.md#repository-map); the
subpackage list there mirrors this index.

## Data flow

```
silver_data + self_distill   ->   rerankers   ->   eval   ->   analysis
   (labels + trained student)     (scorers)      (scores,      (post-hoc
                                                  PSI/nDCG)      numbers)

reproduction  orchestrates the above end to end;  reader  consumes rankings
downstream (QA / verdict);  utils  is shared infrastructure.
```

## Subpackages

- [`rerankers/`](rerankers/README.md): model-family-specific scorer wrappers
  behind one `Reranker` contract and registry. The rest of the package produces
  or consumes their outputs.
- [`silver_data/`](silver_data/README.md): silver-label generation.
  Shared config classifier for two teacher kinds (open-weight vs. hosted/closed),
  plus pointwise prompt templates and parsers.
- [`self_distill/`](self_distill/README.md): self-distillation for
  batched-pointwise grading: open-weight teacher writes K-shot BSC silver, a LoRA
  student is trained with the OC-SFT recipe (`supervised_consistency`). Includes
  the inference [`engines/`](self_distill/engines/README.md) (HF / vLLM).
- [`eval/`](eval/README.md): evaluation primitives: per-query scores,
  ranking-quality metrics (nDCG), and order-stability metrics (PSI / τ-PSI) with
  their channel decompositions. Reads `runs/`.
- [`reader/`](reader/README.md): frozen downstream readers that consume a
  scorer's rankings (multi-document QA and verdict bridges).
- [`analysis/`](analysis/README.md): post-hoc analysis primitives and the
  representative-check path; off the training/eval hot path.
- [`reproduction/`](reproduction/README.md): the task lifecycle that rebuilds
  reported results from tracked configs (`silver → training → direct-eval →
  downstream-eval`), driven by the `scripts/study.py` operator CLI.
- [`utils/`](utils/README.md): shared helpers (config loading, TREC run I/O,
  run-path resolution, provenance, logging). No research logic.

[`__init__.py`](__init__.py) declares the package version and carries no runtime
logic; import subpackages directly.
