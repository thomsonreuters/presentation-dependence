#!/usr/bin/env bash
#
# Build the non-BM25 candidate sets for the Stage-1 retriever-robustness panel.
#
# For each surface in the public locked set it fetches a DENSE (BGE-base-en-v1.5,
# exact/flat Faiss) and a LEARNED-SPARSE (SPLADE++ ED, Lucene impact) top-100
# candidate set from PREBUILT pyserini indexes (no corpus indexing), writes a
# FixtureLoader fixture, and runs the recall@100 + gold-rank-histogram
# diagnostics. Commands mirror the pyserini 2CR matrices so recall@100 matches
# published first-stage numbers. Idempotent (skips an existing cell; FORCE=1
# rebuilds).
#
# ------------------------------------------------------------------------------
# Where to run (RAM is the binding constraint for dense):
# ------------------------------------------------------------------------------
# A Faiss *flat* index loads fully into RAM at search time. Corpus size maps to
# tier:
#
#   smoke  : nfcorpus, scifact, arguana          (<10k docs; laptop-friendly)
#   small  : + fiqa, trec-covid, touche2020,     (<1M docs; a few GB RAM each)
#            robust04, trec-news
#   large  : signal1m, dbpedia, climate-fever,    (2.8M-8.8M docs; MS MARCO
#            dl19, dl20                            BGE-flat ~27 GB RAM; use a
#                                                  high-RAM machine)
#
# SPLADE (impact) is a Lucene inverted index with light RAM use; sparse-only
# builds of the large tier are fine on a laptop. Restrict the paradigm with
# PARADIGMS.
#
# ------------------------------------------------------------------------------
# Usage:
#   bash scripts/data/setup_dense_sparse.sh [smoke|small|large|all|<slug>]   # default: smoke
#
# Env knobs:
#   PARADIGMS="dense sparse"   # restrict to one paradigm (default both)
#   FORCE=1                    # rebuild even if outputs exist
#   RUN_DIAGNOSTICS=0          # skip recall@100 / gold-rank diagnostics
#   HITS=100                   # candidate cutoff (default 100)
#   INCLUDE_GATED=1            # also build signal1m / trec-news / robust04,
#                              # which are access-gated (see the licence note below)
#
# Examples:
#   bash scripts/data/setup_dense_sparse.sh smoke                  # local validation (3 tiny corpora)
#   PARADIGMS=sparse bash scripts/data/setup_dense_sparse.sh all   # all SPLADE locally
#   bash scripts/data/setup_dense_sparse.sh large                  # dense large tier on a high-RAM box
#
# Prereqs:
#   - Python env: `./setup.sh && uv sync --extra dense`  (faiss-cpu for dense)
#   - JAVA_HOME -> Java 21 (pyserini Lucene/impact + text fixture build)
#   - Disk: prebuilt indexes cache under ~/.cache/pyserini/ (large tier: tens of GB)
#
# NOTE (license): Signal-1M / TREC-News / Robust04 are access-gated. Signal-1M
# requires agreeing to the Signal Media licence; TREC-News and Robust04 require
# the corresponding NIST collection access. The driver SKIPS all three unless
# INCLUDE_GATED=1, matching `setup_reproduction_data.py --include-gated`.
# Obtaining the entitlement is the operator's responsibility; a download that
# happens to fail is not a licence check, so do not rely on one.

set -uo pipefail

cd "$(dirname "$0")/../.."

TIER="${1:-smoke}"
PARADIGMS="${PARADIGMS:-dense sparse}"
FORCE="${FORCE:-0}"
RUN_DIAGNOSTICS="${RUN_DIAGNOSTICS:-1}"
HITS="${HITS:-100}"
# Access-gated collections are skipped unless the operator asserts the
# entitlement, mirroring `setup_reproduction_data.py --include-gated`.
INCLUDE_GATED="${INCLUDE_GATED:-0}"
GATED_SLUGS=" signal1m trec-news robust04 "
DIAG_OUT="runs/stage1-firststage-diagnostics/diagnostics.json"
# Python launcher: 'uv run python' locally; set PY=python inside containers (no uv).
PY="${PY:-uv run python}"

# Query-encoder conventions (pyserini 2CR: bge-base-en-v1.5.faiss / splade-pp-ed).
BGE_ENCODER="BAAI/bge-base-en-v1.5"
BGE_PREFIX="Represent this sentence for searching relevant passages:"
BGE_TOKEN="bge-base-en-v1.5"
SPLADE_ENCODER="naver/splade-cocondenser-ensembledistil"   # SPLADE++ EnsembleDistil (= splade-pp-ed)
SPLADE_TOKEN="splade-pp-ed"
BGE_ONNX="BgeBaseEn15"                                       # pyserini Lucene-dense ONNX encoder name
# MS MARCO has no Lucene-flat BGE variant: 'faiss' is exact (Waterloo mirror, may
# be unreachable). Set MSMARCO_DENSE=hnsw to use the HF-hosted approximate HNSW
# (ef-search=1000 keeps recall@100 near-exact) if the Faiss download fails.
MSMARCO_DENSE="${MSMARCO_DENSE:-faiss}"

if [[ -z "${JAVA_HOME:-}" ]]; then
    echo "[setup_dense_sparse] ERROR: JAVA_HOME is not set. Pyserini needs Java 21." >&2
    echo "    export JAVA_HOME=/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home" >&2
    exit 1
fi

force_flag=""
[[ "$FORCE" == "1" ]] && force_flag="--force"

# Catalog: "<slug> <corpus_type> <beir_ds> <topics> <qrels> <text_index> <tier>".
#   corpus_type : msmarco | beir   (drives index ids, --remove-query, encoders)
#   beir_ds     : BEIR dataset slug for beir-v1.0.0-<ds>.* indexes ('-' for msmarco)
#   text_index  : Lucene index for passage TEXT (fixture build)
# All BEIR rows get --remove-query (2CR default; only matters for ArguAna).
CATALOG=(
  # --- smoke (tiny corpora; laptop-friendly) ---
  "nfcorpus      beir nfcorpus          beir-v1.0.0-nfcorpus-test         beir-v1.0.0-nfcorpus-test         beir-v1.0.0-nfcorpus.flat          smoke"
  "scifact       beir scifact           beir-v1.0.0-scifact-test          beir-v1.0.0-scifact-test          beir-v1.0.0-scifact.flat           smoke"
  "arguana       beir arguana           beir-v1.0.0-arguana-test          beir-v1.0.0-arguana-test          beir-v1.0.0-arguana.flat           smoke"
  # --- small (<1M docs) ---
  "fiqa          beir fiqa              beir-v1.0.0-fiqa-test             beir-v1.0.0-fiqa-test             beir-v1.0.0-fiqa.flat              small"
  "trec-covid    beir trec-covid        beir-v1.0.0-trec-covid-test       beir-v1.0.0-trec-covid-test       beir-v1.0.0-trec-covid.flat        small"
  "touche2020    beir webis-touche2020  beir-v1.0.0-webis-touche2020-test beir-v1.0.0-webis-touche2020-test beir-v1.0.0-webis-touche2020.flat  small"
  "robust04      beir robust04          beir-v1.0.0-robust04-test         beir-v1.0.0-robust04-test         beir-v1.0.0-robust04.flat          small"
  "trec-news     beir trec-news         beir-v1.0.0-trec-news-test        beir-v1.0.0-trec-news-test        beir-v1.0.0-trec-news.flat         small"
  # --- large (2.8M-8.8M docs; dense needs a high-RAM box) ---
  "signal1m      beir signal1m          beir-v1.0.0-signal1m-test         beir-v1.0.0-signal1m-test         beir-v1.0.0-signal1m.flat          large"
  "dbpedia-entity beir dbpedia-entity   beir-v1.0.0-dbpedia-entity-test   beir-v1.0.0-dbpedia-entity-test   beir-v1.0.0-dbpedia-entity.flat    large"
  "climate-fever beir climate-fever     beir-v1.0.0-climate-fever-test    beir-v1.0.0-climate-fever-test    beir-v1.0.0-climate-fever.flat     large"
  "dl19          msmarco -              dl19-passage                      dl19-passage                      msmarco-v1-passage                 large"
  "dl20          msmarco -              dl20                              dl20-passage                      msmarco-v1-passage                 large"
)

out_dir_for() {
    # $1 = corpus_type, $2 = beir_ds, $3 = slug
    if [[ "$1" == "beir" ]]; then echo "data/beir-v1.0.0-$2-test"; else echo "data/$3-passage"; fi
}

want_tier() {
    case "$TIER" in
        smoke) [[ "$1" == "smoke" ]] ;;
        small) [[ "$1" == "smoke" || "$1" == "small" ]] ;;
        large) [[ "$1" == "large" ]] ;;
        all)   return 0 ;;
        *)     return 1 ;;
    esac
}

echo "[setup_dense_sparse] tier=$TIER paradigms='$PARADIGMS' force=$FORCE hits=$HITS"
echo "[setup_dense_sparse] JAVA_HOME=$JAVA_HOME"
echo "[setup_dense_sparse] started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo

# Initialize as empty arrays (not just `declare -a`) so `${#OK[@]}` / `${OK[*]}`
# are safe under `set -u` even when nothing is appended (older bash in containers).
OK=()
FAILED=()
SKIPPED=()

for row in "${CATALOG[@]}"; do
    # shellcheck disable=SC2086
    read -r slug corpus_type beir_ds topics qrels text_index tier <<< "$row"

    if [[ "$TIER" != "smoke" && "$TIER" != "small" && "$TIER" != "large" && "$TIER" != "all" ]]; then
        [[ "$slug" == "$TIER" ]] || continue          # explicit single-slug mode
    else
        want_tier "$tier" || continue
    fi

    # After selection, so the access-gated skip also covers explicit single-slug mode.
    if [[ "$INCLUDE_GATED" != "1" && "$GATED_SLUGS" == *" $slug "* ]]; then
        echo "[setup_dense_sparse] SKIP $slug: access-gated. Set INCLUDE_GATED=1 once you hold the entitlement."
        SKIPPED+=("$slug")
        continue
    fi

    out_dir="$(out_dir_for "$corpus_type" "$beir_ds" "$slug")"
    remove_query=""
    [[ "$corpus_type" == "beir" ]] && remove_query="--remove-query"

    for paradigm in $PARADIGMS; do
        if [[ "$paradigm" == "dense" ]]; then
            token="$BGE_TOKEN"
            if [[ "$corpus_type" == "msmarco" ]]; then
                if [[ "$MSMARCO_DENSE" == "hnsw" ]]; then
                    search_index="msmarco-v1-passage.bge-base-en-v1.5.hnsw"
                    paradigm_args=(--paradigm dense --onnx-encoder "$BGE_ONNX")
                else
                    search_index="msmarco-v1-passage.bge-base-en-v1.5"   # faiss (exact)
                    paradigm_args=(--paradigm dense --encoder "$BGE_ENCODER" --encoder-class auto
                                   --l2-norm --query-prefix "$BGE_PREFIX")
                fi
            else
                # BEIR: exact Lucene-flat dense (HF-hosted) via the ONNX BGE encoder.
                search_index="beir-v1.0.0-$beir_ds.bge-base-en-v1.5.flat"
                paradigm_args=(--paradigm dense --onnx-encoder "$BGE_ONNX")
            fi
        elif [[ "$paradigm" == "sparse" ]]; then
            token="$SPLADE_TOKEN"
            if [[ "$corpus_type" == "msmarco" ]]; then
                search_index="msmarco-v1-passage.splade-pp-ed"
            else
                search_index="beir-v1.0.0-$beir_ds.splade-pp-ed"
            fi
            paradigm_args=(--paradigm sparse --encoder "$SPLADE_ENCODER")
        else
            echo "[setup_dense_sparse] unknown paradigm '$paradigm'; skipping" >&2; continue
        fi

        echo "=================================================="
        echo "[$slug / $paradigm]  search=$search_index  text=$text_index${remove_query:+  $remove_query}"
        echo "=================================================="

        # Resumable prefetch of the (large) index tars before pyserini's
        # single-shot downloader: avoids md5/truncation failures on big HF
        # downloads. Set ROBUST_DOWNLOAD=0 to skip.
        if [[ "${ROBUST_DOWNLOAD:-1}" == "1" ]]; then
            $PY scripts/data/prefetch_pyserini_index.py "$search_index" || \
                echo "[setup_dense_sparse][WARN] prefetch $search_index failed; pyserini will retry."
            $PY scripts/data/prefetch_pyserini_index.py "$text_index" || \
                echo "[setup_dense_sparse][WARN] prefetch $text_index failed; pyserini will retry."
        fi

        # shellcheck disable=SC2086
        if $PY scripts/data/fetch_dense_sparse_candidates.py \
            "${paradigm_args[@]}" \
            --topics "$topics" --qrels "$qrels" --slug "$slug" \
            --search-index "$search_index" --text-index "$text_index" \
            --retriever-token "$token" \
            --out "$out_dir" --hits "$HITS" \
            $remove_query $force_flag; then
            OK+=("$slug/$paradigm")
            if [[ "${PURGE_AFTER:-0}" == "1" ]]; then
                # Reclaim disk: the fixture.jsonl is the artifact; the pyserini
                # index isn't needed by the eval path. Keep the text index (shared
                # with the other paradigm for this surface) until both are done.
                $PY scripts/data/prefetch_pyserini_index.py "$search_index" --purge || true
            fi
            if [[ "$RUN_DIAGNOSTICS" == "1" ]]; then
                # ensure_sorted_run_file only writes *_sorted.txt when the run
                # needed sorting; impact/dense outputs are already sorted.
                run_file="$out_dir/run.$token.${slug}_sorted.txt"
                [[ -f "$run_file" ]] || run_file="$out_dir/run.$token.${slug}.txt"
                $PY scripts/data/firststage_candidate_diagnostics.py \
                    --run "$run_file" --qrels "$out_dir/qrels.txt" \
                    --label "$slug:$token" --out "$DIAG_OUT" --k "$HITS" || true
            fi
        else
            echo "[setup_dense_sparse] WARN: FAILED for $slug/$paradigm; continuing." >&2
            FAILED+=("$slug/$paradigm")
        fi
        echo
    done
done

n_ok=${#OK[@]}; n_failed=${#FAILED[@]}; n_skipped=${#SKIPPED[@]}
ok_list="(none)"; failed_list="(none)"; skipped_list="(none)"
(( n_ok > 0 )) && ok_list="${OK[*]+${OK[*]}}"
(( n_failed > 0 )) && failed_list="${FAILED[*]+${FAILED[*]}}"
(( n_skipped > 0 )) && skipped_list="${SKIPPED[*]+${SKIPPED[*]}}"

echo "=================================================="
echo "[setup_dense_sparse] done at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[setup_dense_sparse]   OK:      $n_ok: $ok_list"
echo "[setup_dense_sparse]   FAILED:  $n_failed: $failed_list"
echo "[setup_dense_sparse]   GATED:   $n_skipped: $skipped_list (set INCLUDE_GATED=1 to include)"
echo "[setup_dense_sparse]   diagnostics: $DIAG_OUT"
echo "=================================================="

(( n_failed > 0 )) && exit 1 || exit 0
