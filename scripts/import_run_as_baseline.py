#!/usr/bin/env python
r"""Import a TREC run file as an evaluable baseline run.

The command scores an existing first-stage run (BM25, SPLADE, or another TREC
run) through the same `EvalManager` path as reranker output, without Pyserini,
Java, or a reranker forward pass.

The script writes this per-query layout:

    runs/<ID>/<timestamp>/
        resolved_config.yaml
        per_query_results/<qid>/trec_results_raw.txt

Then run evaluation:

    uv run python scripts/run_eval.py -e <ID>

Example: BM25 baseline on DL19:

    uv run python scripts/import_run_as_baseline.py \\
        -r data/dl19-passage/run.msmarco-v1-passage.bm25-default.dl19_sorted.txt \\
        -q data/dl19-passage/qrels.txt \\
        -i bm25-dl19-baseline
    uv run python scripts/run_eval.py -e bm25-dl19-baseline

Expected: mean_ndcg_cut_10 ≈ 0.506 (the Anserini BM25 baseline).
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import yaml

from presentation_dependence.utils.dataset_meta import meta_for_run
from presentation_dependence.utils.run_paths import default_runs_root
from presentation_dependence.utils.trec import ensure_sorted_run_file, qid_to_dirname


DEFAULT_MEASURES = ["ndcg_cut_10", "ndcg_cut_5", "map", "recip_rank"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-r", "--run", type=str, required=True, help="Source TREC run file.")
    parser.add_argument("-q", "--qrels", type=str, required=True, help="TREC qrels file.")
    parser.add_argument("-i", "--id", type=str, required=True, help="Experiment ID to use for the runs/ subdir.")
    parser.add_argument("--runs-root", type=Path, default=default_runs_root())
    parser.add_argument(
        "--measures",
        nargs="+",
        default=DEFAULT_MEASURES,
        help=f"pytrec_eval measure names (default: {' '.join(DEFAULT_MEASURES)}).",
    )
    parser.add_argument("--tag", type=str, default=None, help="Override the TREC run tag (6th column).")
    args = parser.parse_args()

    run_path = Path(args.run)
    qrels_path = Path(args.qrels).resolve()
    if not run_path.exists():
        raise FileNotFoundError(run_path)
    if not qrels_path.exists():
        raise FileNotFoundError(qrels_path)

    run_path = ensure_sorted_run_file(run_path)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = Path(args.runs_root) / args.id / ts
    per_query_dir = run_dir / "per_query_results"
    per_query_dir.mkdir(parents=True, exist_ok=True)

    tag = args.tag or f"baseline-import:{run_path.name}"
    per_qid: dict[str, list[str]] = defaultdict(list)
    with open(run_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            qid, _q0, docid, rank, score, _tag = parts[:6]
            per_qid[qid].append(f"{qid} Q0 {docid} {rank} {score} {tag}")

    for qid, lines in per_qid.items():
        # Some collections use qids that contain `/`. Escape so `Path / qid`
        # stays a single dir. EvalManager reverses this via
        # `dirname_to_qid`, so pytrec_eval sees the original qid.
        qdir = per_query_dir / qid_to_dirname(qid)
        qdir.mkdir(exist_ok=True)
        (qdir / "trec_results_raw.txt").write_text("\n".join(lines))

    resolved = {
        "id": args.id,
        "source": "import_run_as_baseline.py",
        "source_run": str(run_path.resolve()),
        "eval": {
            "qrels_path": str(qrels_path),
            "measures": list(args.measures),
        },
        "logging": {"level": "INFO"},
        "_run_timestamp": ts,
    }
    ds_meta = meta_for_run(run_path)
    if ds_meta is not None:
        resolved["_dataset_meta"] = ds_meta
    with open(run_dir / "resolved_config.yaml", "w") as f:
        yaml.safe_dump(resolved, f, sort_keys=False)

    print(f"[OK] Wrote {len(per_qid)} qids to {per_query_dir}")
    print(f"     Run dir: {run_dir}")
    print(f"     Now eval: uv run python scripts/run_eval.py -e {args.id}")


if __name__ == "__main__":
    main()
