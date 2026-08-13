#!/usr/bin/env python
r"""Fetch a non-BM25 first-stage candidate set (dense BGE / learned-sparse SPLADE).

Dense/learned-sparse analogue of ``scripts/data/fetch_pyserini_dataset.py`` (which
is BM25-only). Used for the Stage-1 retriever-robustness panel, which varies the
candidate source (lexical BM25 -> dense BGE -> learned-sparse SPLADE) while
holding the scorer, readout, and PSI evaluation fixed to test whether
instability is a BM25 artifact.

Two retrieval modes, both from prebuilt pyserini indexes (no corpus indexing).
The exact commands mirror the pyserini 2CR reproduction matrices
(``pyserini/2cr/{beir,msmarco-v1-passage}.yaml``) so recall@100 matches the
published first-stage numbers:

* ``--paradigm dense``  -> ``pyserini.search.faiss`` over an exact/flat Faiss
  index (``msmarco-v1-passage.bge-base-en-v1.5`` /
  ``beir-v1.0.0-<ds>.bge-base-en-v1.5``), with
  ``--encoder-class auto --encoder BAAI/bge-base-en-v1.5 --l2-norm
  --query-prefix "Represent this sentence for searching relevant passages:"``.
* ``--paradigm sparse`` -> ``pyserini.search.lucene --impact`` over a SPLADE++ ED
  impact index (``msmarco-v1-passage.splade-pp-ed`` /
  ``beir-v1.0.0-<ds>.splade-pp-ed``), with on-the-fly SPLADE query encoding
  (``--encoder naver/splade-cocondenser-ensembledistil``) or, for an exact 2CR
  match, pre-encoded query topics (``--encoded-topics
  beir-v1.0.0-<ds>.test.splade-pp-ed``).

Use the exact/flat Faiss index (not the ``.flat`` Lucene-flat / ``.hnsw``
variants) so recall@100 is not confounded by ANN approximation. For BEIR, pass
``--remove-query`` (2CR default; only changes results where a query id is also a
corpus doc, e.g. ArguAna).

The Faiss/impact index supplies only the ranking (pid + score). Passage
text for the FixtureLoader fixture is fetched separately from the matching
Lucene index (``--text-index``), exactly like ``build_fixture_pyserini.py``.

Outputs the 4-file layout presentation_dependence consumes, plus a FixtureLoader fixture so
container evaluation is a pure candidate-file swap (FixtureLoader, no container-side
pyserini):

    <out>/topics.tsv
    <out>/qrels.txt
    <out>/run.<retriever-token>.<slug>.txt
    <out>/run.<retriever-token>.<slug>_sorted.txt
    <out>/fixture.<retriever-token>.<slug>.jsonl     # {qid, query, passages:[{pid,text}]}
    <out>/dataset_meta.yaml                            # merged first_stage entry

Requires Java 21 (pyserini Lucene/impact) and, for dense, ``faiss`` (install via
the ``[dense]`` extra: ``uv sync --extra dense``) + a GPU-free torch encode of the
(small) query set. First use of an unseen prebuilt index downloads it into
``~/.cache/pyserini/`` (Faiss flat indexes are large; MS MARCO BGE-flat is ~tens
of GB and must fit in RAM at search time -- run the heavy surfaces on a high-RAM
box, see ``scripts/data/setup_dense_sparse.sh``).

Example -- BGE dense candidates for DL19 (MS MARCO v1 passage):

    uv run python scripts/data/fetch_dense_sparse_candidates.py \\
        --paradigm dense \\
        --topics dl19-passage --qrels dl19-passage --slug dl19 \\
        --search-index msmarco-v1-passage.bge-base-en-v1.5 \\
        --text-index   msmarco-v1-passage \\
        --encoder BAAI/bge-base-en-v1.5 \\
        --query-prefix "Represent this sentence for searching relevant passages:" --l2-norm \\
        --retriever-token bge-base-en-v1.5 \\
        --out data/dl19-passage/

Example -- SPLADE++ ED candidates for NFCorpus (BEIR, on-the-fly encoding):

    uv run python scripts/data/fetch_dense_sparse_candidates.py \\
        --paradigm sparse \\
        --topics beir-v1.0.0-nfcorpus-test --qrels beir-v1.0.0-nfcorpus-test --slug nfcorpus \\
        --search-index beir-v1.0.0-nfcorpus.splade-pp-ed \\
        --text-index   beir-v1.0.0-nfcorpus.flat \\
        --encoder naver/splade-cocondenser-ensembledistil --remove-query \\
        --retriever-token splade-pp-ed \\
        --out data/beir-v1.0.0-nfcorpus-test/

"""

from __future__ import annotations
from presentation_dependence.utils.pyserini_index import pyserini_import, require_prebuilt_index

import argparse
import json
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any

import yaml

# Sibling-script reuse (scripts/data is on sys.path[0] when run directly).
# Keeps the topic/qrels writers single-source.
from fetch_pyserini_dataset import write_qrels_trec, write_topics_tsv

from presentation_dependence.utils.dataset_meta import META_FILENAME
from presentation_dependence.utils.trec import ensure_sorted_run_file


# --------------------------------------------------------------------------- #
# First-stage provenance
# --------------------------------------------------------------------------- #
def first_stage_meta_entry(
    *,
    paradigm: str,
    search_index: str,
    text_index: str,
    encoder: str | None,
    onnx_encoder: str | None,
    encoder_class: str | None,
    encoded_topics: str | None,
    dense_backend: str | None,
    hits: int,
    remove_query: bool,
    query_prefix: str | None,
    l2_norm: bool,
) -> dict[str, Any]:
    """Build the ``first_stage[<token>]`` provenance slice for a dense/sparse run.

    Mirrors ``fetch_pyserini_dataset.infer_bm25_params`` but for the non-BM25
    paradigms. The exact query-encoder + normalization knobs are part of the
    candidate-source contract (they shift recall@100), so we record them.
    """
    common = {
        "search_index": search_index,
        "text_index": text_index,
        "hits": hits,
        "remove_query": remove_query,
        "encoder": encoder,
        "encoded_topics": encoded_topics,
    }
    if paradigm == "dense":
        return {
            "retriever": "dense",
            "model": encoder or onnx_encoder,
            "backend": dense_backend,  # faiss | lucene-flat (exact) | lucene-hnsw
            "onnx_encoder": onnx_encoder,
            "encoder_class": encoder_class,
            "query_prefix": query_prefix,
            "l2_norm": l2_norm,
            "ann": "exact" if dense_backend in ("faiss", "lucene-flat") else "hnsw",
            **common,
            "note": (
                "Dense first stage via pyserini BGE. Mirrors the pyserini 2CR "
                "bge-base-en-v1.5 command (faiss: encoder-class=auto + l2-norm + "
                "BGE query prefix; lucene-flat/hnsw: onnx-encoder BgeBaseEn15)."
            ),
        }
    if paradigm == "sparse":
        return {
            "retriever": "learned-sparse",
            "model": encoder if encoder else "(pre-encoded topics)",
            "backend": "lucene-impact",
            **common,
            "note": (
                "Learned-sparse first stage via pyserini.search.lucene --impact "
                "over a SPLADE++ ED impact index. Mirrors the pyserini 2CR "
                "splade-pp-ed command (on-the-fly SPLADE encoding or pre-encoded "
                "topics)."
            ),
        }
    raise ValueError(f"unknown paradigm {paradigm!r}")


def write_dataset_meta(
    out_dir: Path,
    *,
    dataset: str,
    text_index: str,
    topics_key: str,
    qrels_key: str,
    retriever_token: str,
    search_index: str,
    n_topics: int,
    n_qrel_rows: int,
    entry: dict[str, Any],
) -> None:
    """Merge a non-BM25 ``first_stage[<token>]`` entry into ``dataset_meta.yaml``.

    Merge semantics match ``fetch_pyserini_dataset.write_dataset_meta``: a new
    retriever-token is added under ``first_stage`` while existing entries (e.g.
    the BM25 reference row) and top-level fields are preserved.
    """
    meta_path = out_dir / META_FILENAME
    existing: dict[str, Any] = {}
    if meta_path.exists():
        with open(meta_path, "r") as f:
            existing = yaml.safe_load(f) or {}

    fs_all = existing.get("first_stage", {}) or {}
    fs_all[retriever_token] = entry

    merged: dict[str, Any] = {
        "dataset": existing.get("dataset", dataset),
        # `corpus` is the text/Lucene index (used for text lookup). Keep the
        # existing value if present (the BM25 fetch set it).
        "corpus": existing.get("corpus", text_index),
        "source": existing.get(
            "source",
            f"scripts/data/fetch_dense_sparse_candidates.py (topics={topics_key}, qrels={qrels_key})",
        ),
        "fetched": existing.get("fetched", str(date.today())),
        "first_stage": fs_all,
        "topics": existing.get("topics", {"source": f"pyserini.get_topics({topics_key!r})", "n": n_topics}),
        "qrels": existing.get("qrels", {"source": f"pyserini.get_qrels({qrels_key!r})", "n_rows": n_qrel_rows}),
    }
    sx = existing.get("search_indexes")
    sx = dict(sx) if isinstance(sx, dict) else {}
    sx[retriever_token] = search_index
    merged["search_indexes"] = sx
    if "notes" in existing:
        merged["notes"] = existing["notes"]
    with open(meta_path, "w") as f:
        yaml.safe_dump(merged, f, sort_keys=False)


# --------------------------------------------------------------------------- #
# Retrieval (dense / sparse)
# --------------------------------------------------------------------------- #
def resolve_dense_backend(backend: str, search_index: str) -> str:
    """Resolve ``auto`` to a concrete dense backend from the index suffix.

    pyserini ships BGE candidate indexes in three formats:
    * ``*.bge-base-en-v1.5``      -> **Faiss** flat (``pyserini.search.faiss``,
      ``--encoder``); hosted on the Waterloo mirror.
    * ``*.bge-base-en-v1.5.flat`` -> **Lucene flat dense** (exact;
      ``pyserini.search.lucene --dense --flat --onnx-encoder``); HF-hosted.
    * ``*.bge-base-en-v1.5.hnsw`` -> **Lucene HNSW** (approximate; add
      ``--hnsw --ef-search``); HF-hosted.

    The Lucene-flat path is the preferred default for BEIR (exact + HF-hosted). MS
    MARCO has no Lucene-flat variant, so its exact dense run needs the Faiss
    index (Waterloo); the ``.hnsw`` variant is the HF-hosted fallback (set
    ``--ef-search`` high to keep recall@100 near-exact).
    """
    if backend != "auto":
        return backend
    if search_index.endswith(".flat"):
        return "lucene-flat"
    if search_index.endswith(".hnsw"):
        return "lucene-hnsw"
    return "faiss"


def run_dense(  # noqa: C901
    *,
    backend: str,
    search_index: str,
    topics: str,
    out_run: Path,
    encoder: str | None,
    onnx_encoder: str | None,
    encoder_class: str,
    hits: int,
    threads: int,
    batch_size: int,
    ef_search: int,
    query_prefix: str | None,
    l2_norm: bool,
    encoder_extra: list[str],
    remove_query: bool,
) -> None:
    """Dense retrieval via Faiss flat or Lucene flat/HNSW dense (BGE)."""
    if backend == "faiss":
        if not encoder:
            raise ValueError("faiss dense backend needs --encoder")
        cmd = [
            sys.executable,
            "-m",
            "pyserini.search.faiss",
            "--index",
            search_index,
            "--topics",
            topics,
            "--encoder-class",
            encoder_class,
            "--encoder",
            encoder,
            "--output",
            str(out_run),
            "--hits",
            str(hits),
            "--threads",
            str(threads),
            "--batch-size",
            str(batch_size),
        ]
        if l2_norm:
            cmd.append("--l2-norm")
        if query_prefix:
            cmd += ["--query-prefix", query_prefix]
        if remove_query:
            cmd.append("--remove-query")
    elif backend in ("lucene-flat", "lucene-hnsw"):
        if not onnx_encoder:
            raise ValueError(f"{backend} dense backend needs --onnx-encoder (e.g. BgeBaseEn15)")
        if query_prefix or l2_norm:
            print(
                "      [WARN] --query-prefix/--l2-norm are ignored on the Lucene dense "
                "backend (the ONNX encoder owns query encoding).",
                flush=True,
            )
        # pyserini's Lucene dense *batch* path drops topic ids (KeyError in
        # __main__), so force the single-query path. BEIR query sets are small.
        cmd = [
            sys.executable,
            "-m",
            "pyserini.search.lucene",
            "--dense",
            "--flat" if backend == "lucene-flat" else "--hnsw",
            "--index",
            search_index,
            "--topics",
            topics,
            "--onnx-encoder",
            onnx_encoder,
            "--output",
            str(out_run),
            "--output-format",
            "trec",
            "--hits",
            str(hits),
            "--threads",
            "1",
            "--batch-size",
            "1",
        ]
        if backend == "lucene-hnsw":
            cmd += ["--ef-search", str(ef_search)]
        if remove_query:
            cmd.append("--remove-query")
    else:
        raise ValueError(f"unknown dense backend {backend!r}")
    cmd += encoder_extra
    print(f"      $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def run_sparse(
    *,
    search_index: str,
    topics: str,
    encoded_topics: str | None,
    out_run: Path,
    encoder: str | None,
    hits: int,
    threads: int,
    batch_size: int,
    encoder_extra: list[str],
    remove_query: bool,
) -> None:
    """Learned-sparse retrieval via pyserini.search.lucene --impact (SPLADE++ ED).

    Two query-encoding modes (mutually exclusive):
    * pre-encoded topics (``encoded_topics``): pass the SPLADE topic key, no encoder.
    * on-the-fly (``encoder``): pass the plain topic key + the SPLADE encoder.
    """
    cmd = [
        sys.executable,
        "-m",
        "pyserini.search.lucene",
        "--index",
        search_index,
        "--topics",
        encoded_topics if encoded_topics else topics,
        "--output",
        str(out_run),
        "--hits",
        str(hits),
        "--impact",
        "--threads",
        str(threads),
        "--batch-size",
        str(batch_size),
    ]
    if not encoded_topics:
        if not encoder:
            raise ValueError("sparse retrieval needs --encoder or --encoded-topics")
        cmd += ["--encoder", encoder]
    if remove_query:
        cmd.append("--remove-query")
    cmd += encoder_extra
    print(f"      $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


# --------------------------------------------------------------------------- #
# Fixture (FixtureLoader-compatible: text from the Lucene text index)
# --------------------------------------------------------------------------- #
def _load_topics_tsv(path: Path) -> dict[str, str]:
    topics: dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) == 2:
                topics[parts[0]] = parts[1]
    return topics


def _load_run_topk(path: Path, k: int) -> dict[str, list[str]]:
    """Qid -> [pid, ...] in ascending-rank order, truncated to top-k."""
    per: dict[str, list[tuple[int, str]]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            qid, _q0, pid, rank = parts[0], parts[1], parts[2], parts[3]
            try:
                rk = int(rank)
            except ValueError:
                continue
            per.setdefault(qid, []).append((rk, pid))
    out: dict[str, list[str]] = {}
    for qid, entries in per.items():
        entries.sort()
        out[qid] = [pid for _rk, pid in entries[:k]]
    return out


def build_fixture(
    *,
    topics_tsv: Path,
    run_path: Path,
    text_index: str,
    k: int,
    out_path: Path,
) -> None:
    """Materialise a FixtureLoader ``fixture.jsonl`` from the run + Lucene text index.

    Text-resolution logic matches ``build_fixture_pyserini.py`` (raw JSON
    doc -> ``contents`` | ``text``). The fixture is kept minimal (qid, query,
    passages:[{pid,text}]) so it drops straight into the existing FixtureLoader
    container path; diagnostics (recall@100 / gold-rank histogram) read the run + qrels
    directly, not the fixture.
    """
    try:
        with pyserini_import():
            from pyserini.search.lucene import LuceneSearcher  # type: ignore
    except Exception as e:  # pragma: no cover - env-specific
        print(f"[fixture][FATAL] pyserini import failed: {e}", file=sys.stderr)
        print("Hint: `uv sync && export JAVA_HOME=...`", file=sys.stderr)
        raise SystemExit(2) from e

    print(f"      building fixture: text_index={text_index} k={k}", flush=True)
    searcher = require_prebuilt_index(LuceneSearcher.from_prebuilt_index(text_index), text_index)
    topics = _load_topics_tsv(topics_tsv)
    run = _load_run_topk(run_path, k)

    n_written = 0
    n_zero = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as outf:
        for qid, pids in run.items():
            if qid not in topics:
                print(f"      [WARN] qid={qid} missing from topics.tsv", flush=True)
                continue
            passages: list[dict] = []
            for pid in pids:
                doc = searcher.doc(pid)
                if doc is None:
                    print(f"      [WARN] pid={pid} missing from {text_index}", flush=True)
                    continue
                raw = doc.raw()
                try:
                    parsed = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    parsed = {"contents": raw}
                text = parsed.get("contents") or parsed.get("text") or ""
                if not text:
                    continue
                passages.append({"pid": pid, "text": text})
            if not passages:
                n_zero += 1
            record = {"qid": qid, "query": topics[qid], "passages": passages}
            outf.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_written += 1
    size_mb = out_path.stat().st_size / 1e6
    print(f"      wrote {n_written} queries ({n_zero} with 0 passages) -> {out_path} ({size_mb:.1f} MB)", flush=True)


# --------------------------------------------------------------------------- #
# nDCG@10 reproduction self-check (against the pyserini 2CR matrices)
# --------------------------------------------------------------------------- #
# Condition-name maps: our (paradigm, dense_backend) -> the pyserini 2CR
# condition whose published nDCG@10 our build should reproduce.
_BEIR_DENSE_CONDITION = {
    "faiss": "bge-base-en-v1.5.faiss",
    "lucene-flat": "bge-base-en-v1.5.lucene-flat",
    "lucene-hnsw": "bge-base-en-v1.5.lucene-hnsw",
}
_MSMARCO_DENSE_CONDITION = {
    "faiss": "bge-base-en-v1.5.faiss-flat.pytorch",
    "lucene-hnsw": "bge-base-en-v1.5.lucene-hnsw.onnx",
}


def _pyserini_2cr_path(name: str) -> Path | None:
    try:
        import pyserini  # type: ignore
    except Exception:
        return None
    p = Path(pyserini.__file__).parent / "2cr" / name
    return p if p.exists() else None


def expected_2cr_ndcg10(  # noqa: C901
    *, paradigm: str, dense_backend: str | None, search_index: str, slug: str
) -> tuple[float | None, str]:  # noqa: C901
    """Return (expected nDCG@10, 2CR condition label) for this build, or (None, why).

    Sources the canonical number straight from the installed pyserini 2CR YAMLs
    (``beir.yaml`` / ``msmarco-v1-passage.yaml``) so the self-check has no
    hand-maintained reference to drift.
    """
    is_msmarco = search_index.startswith("msmarco-")
    if paradigm == "sparse":
        condition = "splade-pp-ed-pytorch" if is_msmarco else "splade-pp-ed"
    else:
        cmap = _MSMARCO_DENSE_CONDITION if is_msmarco else _BEIR_DENSE_CONDITION
        condition = cmap.get(dense_backend or "")
        if condition is None:
            return (
                None,
                f"no 2CR condition for dense backend {dense_backend!r} on {'msmarco' if is_msmarco else 'beir'}",
            )

    if is_msmarco:
        yml = _pyserini_2cr_path("msmarco-v1-passage.yaml")
        if yml is None:
            return None, "msmarco-v1-passage.yaml not found"
        y = yaml.safe_load(yml.read_text())
        topic_key = f"{slug}-passage"  # dl19 -> dl19-passage, dl20 -> dl20-passage
        for c in y.get("conditions", []):
            if c.get("name") != condition:
                continue
            for t in c.get("topics", []):
                if t.get("topic_key") == topic_key and t.get("scores"):
                    return float(t["scores"][0].get("nDCG@10")), f"{condition}/{topic_key}"
        return None, f"no 2CR score for {condition}/{topic_key}"

    # BEIR: dataset name = search index minus the 'beir-v1.0.0-' prefix, up to '.'.
    ds = search_index[len("beir-v1.0.0-") :].split(".")[0] if search_index.startswith("beir-v1.0.0-") else None
    yml = _pyserini_2cr_path("beir.yaml")
    if yml is None or ds is None:
        return None, "beir.yaml not found or non-BEIR index"
    y = yaml.safe_load(yml.read_text())
    for c in y.get("conditions", []):
        if c.get("name") != condition:
            continue
        for d in c.get("datasets", []):
            if d.get("dataset") == ds and d.get("scores"):
                return float(d["scores"][0].get("nDCG@10")), f"{condition}/{ds}"
    return None, f"no 2CR score for {condition}/{ds}"


def compute_ndcg10(run_path: Path, qrels_path: Path) -> float | None:
    """Mean nDCG@10 over the judged topics (trec_eval convention) via pytrec_eval."""
    try:
        import pytrec_eval  # type: ignore
    except Exception:
        return None
    with open(qrels_path) as f:
        qrels = pytrec_eval.parse_qrel(f)
    with open(run_path) as f:
        run = pytrec_eval.parse_run(f)
    if not qrels:
        return None
    ev = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut.10"})
    res = ev.evaluate(run)
    vals = [res[q]["ndcg_cut_10"] for q in res if q in qrels]
    return (sum(vals) / len(vals)) if vals else None


def verify_ndcg10(
    *,
    run_path: Path,
    qrels_path: Path,
    paradigm: str,
    dense_backend: str | None,
    search_index: str,
    slug: str,
    tol: float,
) -> bool:
    """Compute nDCG@10 and compare to the 2CR reference. Returns True on PASS/SKIP."""
    expected, label = expected_2cr_ndcg10(
        paradigm=paradigm, dense_backend=dense_backend, search_index=search_index, slug=slug
    )
    measured = compute_ndcg10(run_path, qrels_path)
    if measured is None:
        print("      [verify] nDCG@10 not computable (pytrec_eval/qrels missing); skipping.")
        return True
    if expected is None:
        print(f"      [verify] measured nDCG@10={measured:.4f}; no 2CR reference ({label}); skipping comparison.")
        return True
    delta = measured - expected
    status = "PASS" if abs(delta) <= tol else "MISMATCH"
    print(f"      [verify] nDCG@10 measured={measured:.4f} vs 2CR {label}={expected:.4f} (Δ={delta:+.4f}) -> {status}")
    if status == "MISMATCH":
        print(
            f"      [verify][WARN] |Δ|={abs(delta):.4f} > tol {tol}. Check encoder/flags vs 2CR "
            f"(SPLADE on-the-fly vs pre-encoded `--encoded-topics`; dense backend/prefix)."
        )
    return status == "PASS"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--paradigm", required=True, choices=("dense", "sparse"), help="Retrieval paradigm.")
    p.add_argument(
        "--topics", required=True, help="pyserini topics key (e.g. 'dl19-passage'). Used for topics.tsv + fixture."
    )
    p.add_argument("--qrels", required=True, help="pyserini qrels key (e.g. 'dl19-passage').")
    p.add_argument("--slug", required=True, help="Short name used in run/fixture filenames (e.g. 'dl19').")
    p.add_argument("--out", required=True, help="Output directory (created if missing).")
    p.add_argument(
        "--search-index",
        required=True,
        help="pyserini prebuilt index for RETRIEVAL (Faiss flat BGE for dense / impact SPLADE for sparse).",
    )
    p.add_argument(
        "--text-index",
        required=True,
        help="pyserini prebuilt Lucene index used for passage TEXT (fixture build).",
    )
    p.add_argument(
        "--encoder",
        default=None,
        help="Query encoder HF id (faiss dense: BAAI/bge-base-en-v1.5; sparse: naver/splade-cocondenser-ensembledistil).",
    )
    p.add_argument(
        "--onnx-encoder",
        default=None,
        help="Lucene dense ONNX encoder name (e.g. BgeBaseEn15). Required for lucene-flat/lucene-hnsw.",
    )
    p.add_argument("--encoder-class", default="auto", help="Dense FaissSearcher encoder class (default: auto).")
    p.add_argument(
        "--dense-backend",
        default="auto",
        choices=("auto", "faiss", "lucene-flat", "lucene-hnsw"),
        help=(
            "Dense searcher backend. 'auto' picks from the index suffix: '.flat' "
            "-> lucene-flat (exact, HF), '.hnsw' -> lucene-hnsw, else faiss "
            "(Waterloo). Ignored for --paradigm sparse."
        ),
    )
    p.add_argument(
        "--ef-search",
        type=int,
        default=1000,
        help="lucene-hnsw efSearch (default: 1000; high keeps recall near-exact).",
    )
    p.add_argument(
        "--encoded-topics",
        default=None,
        help="Sparse only: pre-encoded SPLADE topic key (e.g. beir-v1.0.0-<ds>.test.splade-pp-ed); omits --encoder.",
    )
    p.add_argument(
        "--retriever-token",
        required=True,
        help="Retriever token: run-filename segment + dataset_meta key (e.g. 'bge-base-en-v1.5', 'splade-pp-ed').",
    )
    p.add_argument("--hits", type=int, default=100, help="Candidate cutoff (default: 100, the B=20 chunking ceiling).")
    p.add_argument("--k-fixture", type=int, default=None, help="Top-k materialised into fixture (default: --hits).")
    p.add_argument("--threads", type=int, default=8, help="Search threads (default: 8).")
    p.add_argument("--batch-size", type=int, default=32, help="Query batch size (default: 32).")
    p.add_argument(
        "--remove-query",
        action="store_true",
        help="Pass --remove-query to the searcher (2CR default for BEIR; matters for ArguAna).",
    )
    p.add_argument("--query-prefix", default=None, help="Dense: query prefix (the BGE instruction prefix).")
    p.add_argument("--l2-norm", action="store_true", help="Dense: L2-normalize query embeddings (BGE convention).")
    p.add_argument(
        "--encoder-extra",
        action="append",
        default=[],
        help="Extra raw flag(s) passed verbatim to the pyserini searcher. Repeatable.",
    )
    p.add_argument("--skip-fixture", action="store_true", help="Skip the fixture.jsonl build (run + meta only).")
    p.add_argument(
        "--no-verify-ndcg",
        action="store_true",
        help="Skip the post-build nDCG@10 reproduction check against the pyserini 2CR matrices.",
    )
    p.add_argument("--verify-tol", type=float, default=0.01, help="nDCG@10 reproduction tolerance (default: 0.01).")
    p.add_argument(
        "--verify-only",
        action="store_true",
        help="Skip search/fixture; only verify an EXISTING built run's nDCG@10 against 2CR.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-fetch even if outputs already exist for this retriever token.",
    )
    return p.parse_args()


def main() -> int:  # noqa: C901
    args = parse_args()
    dense_backend = resolve_dense_backend(args.dense_backend, args.search_index) if args.paradigm == "dense" else None
    if args.paradigm == "dense" and not args.verify_only:
        if dense_backend == "faiss" and not args.encoder:
            raise SystemExit("[FATAL] faiss dense backend requires --encoder (e.g. BAAI/bge-base-en-v1.5)")
        if dense_backend in ("lucene-flat", "lucene-hnsw") and not args.onnx_encoder:
            raise SystemExit(f"[FATAL] {dense_backend} dense backend requires --onnx-encoder (e.g. BgeBaseEn15)")
    if args.paradigm == "sparse" and not args.verify_only and not (args.encoder or args.encoded_topics):
        raise SystemExit("[FATAL] --paradigm sparse requires --encoder or --encoded-topics")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    k_fixture = args.k_fixture if args.k_fixture is not None else args.hits

    token = args.retriever_token
    topics_path = out_dir / "topics.tsv"
    qrels_path = out_dir / "qrels.txt"
    run_path = out_dir / f"run.{token}.{args.slug}.txt"
    fixture_path = out_dir / f"fixture.{token}.{args.slug}.jsonl"
    meta_path = out_dir / META_FILENAME

    if args.verify_only:
        existing = out_dir / f"run.{token}.{args.slug}_sorted.txt"
        if not existing.exists():
            existing = run_path
        if not existing.exists():
            raise SystemExit(f"[FATAL] --verify-only: no built run at {existing}")
        ok = verify_ndcg10(
            run_path=existing,
            qrels_path=qrels_path,
            paradigm=args.paradigm,
            dense_backend=dense_backend,
            search_index=args.search_index,
            slug=args.slug,
            tol=args.verify_tol,
        )
        return 0 if ok else 1

    # Idempotency: skip if this retriever token is already fully materialised.
    if not args.force:
        done = run_path.exists() and (args.skip_fixture or fixture_path.exists()) and meta_path.exists()
        if done:
            existing_meta = yaml.safe_load(meta_path.read_text()) or {}
            if token in (existing_meta.get("first_stage") or {}):
                print(
                    f"[skip] {out_dir} already has run + fixture + meta for retriever "
                    f"'{token}'. Use --force to re-fetch."
                )
                return 0

    # topics.tsv / qrels.txt are shared across first stages for a surface; only
    # (re)write if missing so we don't clobber a hand-curated copy.
    if args.force or not topics_path.exists():
        print(f"[1/4] topics -> {topics_path}")
        n_topics = write_topics_tsv(args.topics, topics_path)
        print(f"      {n_topics} queries")
    else:
        n_topics = sum(1 for _ in open(topics_path))
        print(f"[1/4] topics (reuse) -> {topics_path} ({n_topics} queries)")

    if args.force or not qrels_path.exists():
        print(f"[2/4] qrels  -> {qrels_path}")
        n_qrels = write_qrels_trec(args.qrels, qrels_path)
        print(f"      {n_qrels} qrel rows")
    else:
        n_qrels = sum(1 for _ in open(qrels_path))
        print(f"[2/4] qrels (reuse) -> {qrels_path} ({n_qrels} rows)")

    print(
        f"[3/4] {args.paradigm} search (index={args.search_index}, "
        f"encoder={args.encoder or args.encoded_topics}, hits={args.hits}, "
        f"remove_query={args.remove_query}) -> {run_path}"
    )
    if args.paradigm == "dense":
        print(f"      dense backend: {dense_backend}")
        run_dense(
            backend=dense_backend,
            search_index=args.search_index,
            topics=args.topics,
            out_run=run_path,
            encoder=args.encoder,
            onnx_encoder=args.onnx_encoder,
            encoder_class=args.encoder_class,
            hits=args.hits,
            threads=args.threads,
            batch_size=args.batch_size,
            ef_search=args.ef_search,
            query_prefix=args.query_prefix,
            l2_norm=args.l2_norm,
            encoder_extra=args.encoder_extra,
            remove_query=args.remove_query,
        )
    else:
        run_sparse(
            search_index=args.search_index,
            topics=args.topics,
            encoded_topics=args.encoded_topics,
            out_run=run_path,
            encoder=args.encoder,
            hits=args.hits,
            threads=args.threads,
            batch_size=args.batch_size,
            encoder_extra=args.encoder_extra,
            remove_query=args.remove_query,
        )
    sorted_path = ensure_sorted_run_file(run_path)
    print(f"      sorted -> {sorted_path}")

    entry = first_stage_meta_entry(
        paradigm=args.paradigm,
        search_index=args.search_index,
        text_index=args.text_index,
        encoder=args.encoder,
        onnx_encoder=args.onnx_encoder if args.paradigm == "dense" else None,
        encoder_class=args.encoder_class if args.paradigm == "dense" else None,
        encoded_topics=args.encoded_topics if args.paradigm == "sparse" else None,
        dense_backend=dense_backend,
        hits=args.hits,
        remove_query=args.remove_query,
        query_prefix=args.query_prefix if args.paradigm == "dense" else None,
        l2_norm=args.l2_norm if args.paradigm == "dense" else False,
    )
    write_dataset_meta(
        out_dir,
        dataset=out_dir.name,
        text_index=args.text_index,
        topics_key=args.topics,
        qrels_key=args.qrels,
        retriever_token=token,
        search_index=args.search_index,
        n_topics=n_topics,
        n_qrel_rows=n_qrels,
        entry=entry,
    )
    print(f"      meta -> {meta_path}")

    if not args.no_verify_ndcg:
        verify_ndcg10(
            run_path=Path(sorted_path),
            qrels_path=qrels_path,
            paradigm=args.paradigm,
            dense_backend=dense_backend,
            search_index=args.search_index,
            slug=args.slug,
            tol=args.verify_tol,
        )

    if args.skip_fixture:
        print("[4/4] --skip-fixture set; run + meta only.")
    else:
        print(f"[4/4] fixture -> {fixture_path}")
        build_fixture(
            topics_tsv=topics_path,
            run_path=Path(sorted_path),
            text_index=args.text_index,
            k=k_fixture,
            out_path=fixture_path,
        )

    print()
    print("Next — candidate-source diagnostics (recall@100 + gold-rank histogram, no GPU):")
    print(
        f"  uv run python scripts/data/firststage_candidate_diagnostics.py \\\n"
        f"      --run {sorted_path} --qrels {qrels_path} --label {args.slug}:{token}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
