"""Load first-stage provenance for a dataset directory under `data/`.

Every `data/<slug>/` directory is expected to carry a `dataset_meta.yaml`
that records where the data came from and which first-stage retriever +
parameters produced each `run.*.txt` file in the directory. Downstream
`ExperimentManager` and `import_run_as_baseline.py` read this and paste the
relevant slice into each run's `resolved_config.yaml`, so six months from now a
`grep bm25_variant runs/**/resolved_config.yaml` tells you exactly
which first-stage was fed to which reranker.

pyserini's `--bm25` CLI flag silently applies MS MARCO-tuned params
(k1=0.82, b=0.68) for the `msmarco-v1-passage` index, not the vanilla
defaults (k1=0.9, b=0.4). Field convention across RankGPT, RankZephyr,
mxbai, DeAR, and jina-v3 DL19/DL20 reranking papers uses the tuned
variant, so we do too; but we record it explicitly.

Schema (free-form but these keys are convention):

    dataset: dl19-passage                    # slug
    corpus: msmarco-v1-passage               # Lucene / dense index id
    source: "copied from experiments/..."    # human-readable
    fetched: <recorded fetch date>
    first_stage:                             # keyed by retriever-token
      bm25-default:                          #   == 3rd segment in filename
        retriever: bm25
        variant: msmarco-tuned               # msmarco-tuned | vanilla | ...
        k1: 0.82
        b: 0.68
        hits: 1000
        note: "pyserini default for msmarco-v1-passage"
      splade-pp-ed-pytorch:
        retriever: splade-pp-ed
        model: naver/splade-pp-ed
        backend: pytorch
        hits: 1000
    topics:   {source: "...", n: 43}
    qrels:    {source: "...", n_judged_topics: 43}
    notes: |
      free-form caveats
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

META_FILENAME = "dataset_meta.yaml"


def retriever_token_from_run_path(run_path: str | Path) -> str | None:
    """Extract the retriever token from a run filename.

    Two naming conventions are supported (both in use in this project):

    - DL-style:   `run.<corpus>.<retriever>.<slug>.txt` (4 tokens after
      stripping "run."/".txt"). Used for MS MARCO v1 runs, e.g.
      `run.msmarco-v1-passage.bm25-default.dl19.txt`.
    - BEIR-style: `run.<retriever>.<slug>.txt` (3 tokens), e.g.
      `run.bm25.trec-covid.txt`.

    Returns None if the filename doesn't match either convention so
    callers can degrade gracefully on ad-hoc files.
    """
    name = Path(run_path).name
    if not (name.startswith("run.") and name.endswith(".txt")):
        return None
    stem = name[len("run.") : -len(".txt")]
    parts = stem.split(".")
    if len(parts) == 3:  # run.<corpus>.<retriever>.<slug>.txt
        return parts[1]
    if len(parts) == 2:  # run.<retriever>.<slug>.txt
        return parts[0]
    return None


def load_dataset_meta(data_dir: str | Path) -> dict[str, Any] | None:
    """Return the parsed `dataset_meta.yaml` in `data_dir`, or None if absent."""
    p = Path(data_dir) / META_FILENAME
    if not p.exists():
        return None
    with open(p, "r") as f:
        return yaml.safe_load(f) or {}


def meta_for_run(run_path: str | Path) -> dict[str, Any] | None:
    """Return the provenance slice describing the first-stage behind `run_path`.

    Example (given the DL19 meta and run filename ending in
    `bm25-default.dl19_sorted.txt`):

        {
            "dataset": "dl19-passage",
            "corpus": "msmarco-v1-passage",
            "first_stage": {"retriever": "bm25", "variant": "msmarco-tuned",
                            "k1": 0.82, "b": 0.68, ...},
            "topics": {...}, "qrels": {...}
        }

    Returns None if neither a `dataset_meta.yaml` nor a recognisable
    retriever token can be found — callers should still write their run
    without a `_dataset_meta` block in that case.
    """
    run_path = Path(run_path)
    meta = load_dataset_meta(run_path.parent)
    if meta is None:
        return None

    token = retriever_token_from_run_path(run_path)
    first_stage_all = meta.get("first_stage", {}) or {}
    first_stage = first_stage_all.get(token) if token else None

    slice_: dict[str, Any] = {
        "dataset": meta.get("dataset"),
        "corpus": meta.get("corpus"),
        "source": meta.get("source"),
        "fetched": meta.get("fetched"),
        "topics": meta.get("topics"),
        "qrels": meta.get("qrels"),
        "retriever_token": token,
    }
    if first_stage is not None:
        slice_["first_stage"] = first_stage
    else:
        slice_["first_stage"] = {
            "_warning": (
                f"No `first_stage[{token!r}]` entry in "
                f"{run_path.parent}/{META_FILENAME}; first-stage params not "
                "recorded. Add the entry so this run's provenance is complete."
            )
        }
    return slice_
