#!/usr/bin/env python
"""Generate the 15-collection pool-perturbation extension.

Adds 12 new collections (36 new base/k1sft/ocsft cells) to the existing
three-collection fixed-order pool-perturbation control:

- TREC-COVID uses its real BM25 run (``run.bm25.trec-covid.txt``, depth 1000,
  registered ``first_stage.bm25`` in its ``dataset_meta.yaml``), not SPLADE.
- ``max_queries`` is set per collection to the number of queries that are both
  judged (qrels) and reach ``source_depth=101`` in the first-stage run --
  the runner's ``_select_queries`` requires the eligible pool to be at least
  ``max_queries`` and the default of 100 does not hold for every collection.

Each generated config is byte-identical to the existing
``PP-qwen3-4b-{arm}-arguana.yaml`` templates outside of ``data:``,
``eval.qrels_path``, and ``pool_perturbation.max_queries``.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs" / "experiments"
SWEEPS = ROOT / "configs" / "sweeps"

ARMS = ("base", "k1sft", "ocsft")

# max_queries = count of qids that are both judged (qrels) and reach
# source_depth=101 in the first-stage run; see the handoff addendum for the
# per-collection derivation.
DATASETS: dict[str, dict] = {
    "dl19": {
        "data": {
            "dataloader_class": "PyseriniLoader",
            "topics": "dl19-passage",
            "topics_tsv": "data/dl19-passage/topics.tsv",
            "index": "msmarco-v1-passage",
            "run_path": "data/dl19-passage/run.msmarco-v1-passage.bm25-default.dl19.txt",
            "k_input": 101,
        },
        "qrels_path": "data/dl19-passage/qrels.txt",
        "max_queries": 43,
    },
    "dl20": {
        "data": {
            "dataloader_class": "PyseriniLoader",
            "topics": "dl20-passage",
            "topics_tsv": "data/dl20-passage/topics.tsv",
            "index": "msmarco-v1-passage",
            "run_path": "data/dl20-passage/run.msmarco-v1-passage.bm25-default.dl20.txt",
            "k_input": 101,
        },
        "qrels_path": "data/dl20-passage/qrels.txt",
        # 200 topics in the run file, but only 54 are judged; strict_qrels_filter
        # (default True) drops the rest before the runner ever sees them.
        "max_queries": 54,
    },
    "climate-fever": {
        "data": {
            "dataloader_class": "PyseriniLoader",
            "topics": "beir-v1.0.0-climate-fever-test",
            "topics_tsv": "data/beir-v1.0.0-climate-fever-test/topics.tsv",
            "index": "beir-v1.0.0-climate-fever.flat",
            "run_path": "data/beir-v1.0.0-climate-fever-test/run.bm25.climate-fever.txt",
            "k_input": 101,
        },
        "qrels_path": "data/beir-v1.0.0-climate-fever-test/qrels.txt",
        "max_queries": 100,
    },
    "trec-covid": {
        "data": {
            "dataloader_class": "PyseriniLoader",
            "topics": "beir-v1.0.0-trec-covid-test",
            "topics_tsv": "data/beir-v1.0.0-trec-covid-test/topics.tsv",
            "index": "beir-v1.0.0-trec-covid.flat",
            "run_path": "data/beir-v1.0.0-trec-covid-test/run.bm25.trec-covid.txt",
            "k_input": 101,
        },
        "qrels_path": "data/beir-v1.0.0-trec-covid-test/qrels.txt",
        # 50 judged queries all reach run-file depth 1000, but the prebuilt
        # beir-v1.0.0-trec-covid.flat index does not resolve every pid the BM25
        # run references (a known TREC-COVID corpus-versioning quirk), so only
        # 37 queries actually reach 101 *resolvable* passages once the staged
        # fixture is built. Confirmed by a failed run ("found 37") and by
        # inspecting data/beir-v1.0.0-trec-covid-test/fixture.jsonl.
        "max_queries": 37,
    },
    "robust04": {
        "data": {
            "dataloader_class": "PyseriniLoader",
            "topics": "beir-v1.0.0-robust04-test",
            "topics_tsv": "data/beir-v1.0.0-robust04-test/topics.tsv",
            "index": "beir-v1.0.0-robust04.flat",
            "run_path": "data/beir-v1.0.0-robust04-test/run.bm25.robust04.txt",
            "k_input": 101,
        },
        "qrels_path": "data/beir-v1.0.0-robust04-test/qrels.txt",
        "max_queries": 100,
    },
    "dbpedia-entity": {
        "data": {
            "dataloader_class": "PyseriniLoader",
            "topics": "beir-v1.0.0-dbpedia-entity-test",
            "topics_tsv": "data/beir-v1.0.0-dbpedia-entity-test/topics.tsv",
            "index": "beir-v1.0.0-dbpedia-entity.flat",
            "run_path": "data/beir-v1.0.0-dbpedia-entity-test/run.bm25.dbpedia.txt",
            "k_input": 101,
        },
        "qrels_path": "data/beir-v1.0.0-dbpedia-entity-test/qrels.txt",
        "max_queries": 100,
    },
    "scifact": {
        "data": {
            "dataloader_class": "PyseriniLoader",
            "topics": "beir-v1.0.0-scifact-test",
            "topics_tsv": "data/beir-v1.0.0-scifact-test/topics.tsv",
            "index": "beir-v1.0.0-scifact.flat",
            "run_path": "data/beir-v1.0.0-scifact-test/run.bm25.scifact.txt",
            "k_input": 101,
        },
        "qrels_path": "data/beir-v1.0.0-scifact-test/qrels.txt",
        "max_queries": 100,
    },
    "signal1m": {
        "data": {
            "dataloader_class": "PyseriniLoader",
            "topics": "beir-v1.0.0-signal1m-test",
            "topics_tsv": "data/beir-v1.0.0-signal1m-test/topics.tsv",
            "index": "beir-v1.0.0-signal1m.flat",
            "run_path": "data/beir-v1.0.0-signal1m-test/run.bm25.signal1m.txt",
            "k_input": 101,
        },
        "qrels_path": "data/beir-v1.0.0-signal1m-test/qrels.txt",
        "max_queries": 97,
    },
    "trec-news": {
        "data": {
            "dataloader_class": "PyseriniLoader",
            "topics": "beir-v1.0.0-trec-news-test",
            "topics_tsv": "data/beir-v1.0.0-trec-news-test/topics.tsv",
            "index": "beir-v1.0.0-trec-news.flat",
            "run_path": "data/beir-v1.0.0-trec-news-test/run.bm25.trec-news.txt",
            "k_input": 101,
        },
        "qrels_path": "data/beir-v1.0.0-trec-news-test/qrels.txt",
        "max_queries": 57,
    },
    "touche2020": {
        "data": {
            "dataloader_class": "PyseriniLoader",
            "topics": "beir-v1.0.0-webis-touche2020-test",
            "topics_tsv": "data/beir-v1.0.0-webis-touche2020-test/topics.tsv",
            "index": "beir-v1.0.0-webis-touche2020.flat",
            "run_path": "data/beir-v1.0.0-webis-touche2020-test/run.beir.bm25-flat.webis-touche2020.txt",
            "k_input": 101,
        },
        "qrels_path": "data/beir-v1.0.0-webis-touche2020-test/qrels.txt",
        "max_queries": 49,
    },
    "legal-a": {
        "data": {
            "dataloader_class": "FixtureLoader",
            "run_path": "data/legal-a/fixture.jsonl",
            "k_input": 101,
        },
        "qrels_path": "data/legal-a/qrels.txt",
        # 100 judged, 99 reach depth 101, but only 96 are both judged and deep.
        "max_queries": 96,
    },
    "legal-b": {
        "data": {
            "dataloader_class": "FixtureLoader",
            "run_path": "data/legal-b/fixture.jsonl",
            "k_input": 101,
        },
        "qrels_path": "data/legal-b/qrels.txt",
        "max_queries": 100,
    },
}

BUNDLES: dict[str, tuple[str, ...]] = {
    "PP-qwen3-4b-pool-perturbation-bundle1": ("dl19", "dl20", "climate-fever"),
    "PP-qwen3-4b-pool-perturbation-bundle2": ("trec-covid", "robust04", "dbpedia-entity"),
    "PP-qwen3-4b-pool-perturbation-bundle3": ("scifact", "signal1m", "trec-news"),
    "PP-qwen3-4b-pool-perturbation-bundle4": ("touche2020", "legal-a", "legal-b"),
}


# The base arm is a tracked config; the two trained arms are that same config
# with the selected adapter attached. Arm ids here are the compact form used in
# generated config ids; the catalog keys them by pipeline variant name.
TEMPLATE = CONFIGS / "example-passage-pool-perturbation.yaml"
CHECKPOINT_CATALOG = ROOT / "build/reproduction/passage-reranking/training/checkpoints.json"
ARM_VARIANT = {"k1sft": "k1-sft", "ocsft": "oc-sft"}
TRAINING_SEED = 42


def _load_checkpoints() -> dict[str, str]:
    """Return ``{arm: checkpoint_uri}`` for the trained arms."""
    if not CHECKPOINT_CATALOG.is_file():
        raise SystemExit(
            f"Missing checkpoint catalog: {CHECKPOINT_CATALOG}\n"
            "The k1sft and ocsft arms need the adapters the training stage selects. Run\n"
            "  uv run python scripts/study.py passage-reranking training collect\n"
            "first, or pass --base-only to generate the off-shelf arm alone."
        )
    rows = json.loads(CHECKPOINT_CATALOG.read_text(encoding="utf-8")).get("checkpoints") or []
    by_variant = {
        str(r["variant"]): str(r["checkpoint_uri"])
        for r in rows
        if int(r.get("training_seed", TRAINING_SEED)) == TRAINING_SEED
    }
    missing = [v for v in ARM_VARIANT.values() if v not in by_variant]
    if missing:
        raise SystemExit(f"{CHECKPOINT_CATALOG} has no seed-{TRAINING_SEED} row for: {', '.join(missing)}")
    return {arm: by_variant[variant] for arm, variant in ARM_VARIANT.items()}


def _load_template(arm: str, checkpoints: dict[str, str]) -> dict:
    """Return the arm's config template: the tracked base, plus an adapter when trained."""
    config = yaml.safe_load(TEMPLATE.read_text(encoding="utf-8"))
    if arm != "base":
        config["reranker"]["lora_path"] = checkpoints[arm]
        config["reranker"]["max_lora_rank"] = 16
    return config


def _write(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true", help="actually create files (default: dry-run)")
    parser.add_argument(
        "--base-only",
        action="store_true",
        help="generate only the off-shelf arm, so no checkpoint catalog is needed",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    arms = ("base",) if args.base_only else ARMS
    checkpoints = {} if args.base_only else _load_checkpoints()

    written: list[str] = []
    print(f"{'action':8} {'config id'}")
    for dataset, spec in DATASETS.items():
        for arm in arms:
            config = _load_template(arm, checkpoints)
            config_id = f"PP-qwen3-4b-{arm}-{dataset}"
            config["id"] = config_id
            config["data"] = copy.deepcopy(spec["data"])
            config["eval"]["qrels_path"] = spec["qrels_path"]
            config["pool_perturbation"]["max_queries"] = spec["max_queries"]
            print(f"{'WRITE' if args.write else 'would':8} {config_id}.yaml")
            if args.write:
                _write(CONFIGS / f"{config_id}.yaml", config)
            written.append(config_id)

    bundle_ids: list[str] = []
    for bundle_id, datasets in BUNDLES.items():
        member_configs = [f"PP-qwen3-4b-{arm}-{dataset}" for dataset in datasets for arm in arms]
        bundle = {
            "id": bundle_id,
            "bundle": {"member_configs": member_configs, "shared_base": True},
            "execution": {"environment": {"VLLM_WORKER_MULTIPROC_METHOD": "spawn"}},
        }
        if args.write:
            _write(CONFIGS / f"{bundle_id}.yaml", bundle)
        bundle_ids.append(bundle_id)

    sweep = {
        "id": "pp-qwen3-4b-extension-15collections",
        "description": (
            "Pool-perturbation extension: 12 new collections across bundle1-4, "
            "bringing the fixed-order pool control from 3 to 15 collections."
        ),
        "execution": {"max_parallel": 1},
        "jobs": [{"exp_id": bundle_id} for bundle_id in bundle_ids],
    }
    sweep_path = SWEEPS / f"{sweep['id']}.yaml"
    if args.write:
        _write(sweep_path, sweep)
        print(f"\n[gen] wrote {len(written)} configs, {len(bundle_ids)} bundles, {sweep_path}")
    else:
        print(
            f"\n[dry-run] would write {len(written)} configs, {len(bundle_ids)} bundles, "
            f"and {sweep_path}. Re-run with --write."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
