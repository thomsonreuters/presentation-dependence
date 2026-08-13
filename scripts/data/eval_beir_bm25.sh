#!/usr/bin/env bash
#
# Register and evaluate BM25 baselines for every BEIR dataset that has a
# fetched run file under `data/`. For canonical population members, run
# `scripts/setup_reproduction_data.py run --task passage-reranking` first.
# Idempotent: `import_run_as_baseline` writes a timestamped subdir on each run
# (re-runs add another measurement); `run_eval.py` walks all existing
# timestamped dirs.
#
# Usage:
#   bash scripts/data/eval_beir_bm25.sh 2>&1 | tee data/_eval_beir_bm25.log
#
# No Java, GPU, or network: pytrec_eval on existing run files.

set -euo pipefail

cd "$(dirname "$0")/../.."

# All configured datasets. Missing run or qrels files are skipped.
BEIR_SLUGS=(
    # slug           data/ dir                                    run filename
    "trec-covid      data/beir-v1.0.0-trec-covid-test             run.bm25.trec-covid_sorted.txt"
    "touche2020      data/beir-v1.0.0-webis-touche2020-test       run.beir.bm25-flat.webis-touche2020_sorted.txt"
    "nq              data/beir-v1.0.0-nq-test                     run.bm25.beir-nq.txt"
    "nfcorpus        data/beir-v1.0.0-nfcorpus-test               run.bm25.nfcorpus.txt"
    "scifact         data/beir-v1.0.0-scifact-test                run.bm25.scifact.txt"
    "scidocs         data/beir-v1.0.0-scidocs-test                run.bm25.scidocs.txt"
    "arguana         data/beir-v1.0.0-arguana-test                run.bm25.arguana.txt"
    "fiqa            data/beir-v1.0.0-fiqa-test                   run.bm25.fiqa.txt"
    "quora           data/beir-v1.0.0-quora-test                  run.bm25.quora.txt"
    "hotpotqa        data/beir-v1.0.0-hotpotqa-test               run.bm25.hotpotqa.txt"
    "dbpedia         data/beir-v1.0.0-dbpedia-entity-test         run.bm25.dbpedia.txt"
    "fever           data/beir-v1.0.0-fever-test                  run.bm25.fever.txt"
    "climate-fever   data/beir-v1.0.0-climate-fever-test          run.bm25.climate-fever.txt"
)

echo "[eval_beir_bm25] === BM25 evaluation over BEIR datasets ==="
echo

for row in "${BEIR_SLUGS[@]}"; do
    # shellcheck disable=SC2086
    read -r slug data_dir run_name <<< "$row"

    run_path="$data_dir/$run_name"
    qrels_path="$data_dir/qrels.txt"
    exp_id="bm25-beir-${slug}-baseline"

    if [[ ! -f "$run_path" ]]; then
        echo "[skip] $slug: no run file at $run_path"
        echo
        continue
    fi
    if [[ ! -f "$qrels_path" ]]; then
        echo "[skip] $slug: no qrels at $qrels_path"
        echo
        continue
    fi

    echo "=================================================="
    echo "[$exp_id]  $run_name"
    echo "=================================================="
    uv run python scripts/import_run_as_baseline.py \
        -r "$run_path" -q "$qrels_path" -i "$exp_id"
    uv run python scripts/run_eval.py -e "$exp_id"
    echo
done

echo "[eval_beir_bm25] Done. Metrics under runs/bm25-beir-*-baseline/<ts>/metrics.json"
