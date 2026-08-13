#!/usr/bin/env python
"""Freeze a partial silver run into a clean, fully-covered SFT corpus.

Given a `SilverGenerator` run directory:

  1. Reads `silver_labels.jsonl` after error rows have been removed.
  2. Computes per-query coverage against the source fixture.
  3. Writes three artifacts under the run dir:
       silver_labels.clean.jsonl    only fully covered queries
       silver_labels.partial.jsonl  queries with 1..K-1 labels
       FINAL_REPORT.md              Markdown summary
  4. Refreshes `metrics/coverage.json`, `metrics/score_distribution.json`,
     and `cost_report.json` to reflect the clean subset.
  5. Writes `.frozen` to mark the run directory as closed to resume.

Does not mutate `silver_labels.jsonl`; that file remains the input for resume
logic. Downstream training should consume `silver_labels.clean.jsonl`.
"""

from __future__ import annotations

import argparse
import collections
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def main() -> None:  # noqa: C901
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument(
        "--candidates-per-query",
        type=int,
        default=20,
        help="Number of labels expected per fully-covered query.",
    )
    args = parser.parse_args()

    labels_path = args.run_dir / "silver_labels.jsonl"
    if not labels_path.is_file():
        raise SystemExit(f"silver_labels.jsonl not found at {labels_path}")

    expected_qids = _expected_qids(args.fixture)
    rows_by_qid: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    n_rows_total = 0
    n_rows_with_error = 0
    with labels_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            n_rows_total += 1
            if rec.get("error") or rec.get("score_parsed") is None:
                n_rows_with_error += 1
                continue
            rows_by_qid[str(rec["query_id"])].append(rec)

    fully_covered: list[dict[str, Any]] = []
    partial: list[dict[str, Any]] = []
    for _qid, rows in rows_by_qid.items():
        if len(rows) >= args.candidates_per_query:
            fully_covered.extend(rows[: args.candidates_per_query])
        else:
            partial.extend(rows)

    clean_path = args.run_dir / "silver_labels.clean.jsonl"
    partial_path = args.run_dir / "silver_labels.partial.jsonl"
    _write_jsonl(clean_path, fully_covered)
    _write_jsonl(partial_path, partial)

    qids_clean = {r["query_id"] for r in fully_covered}
    qids_partial = {r["query_id"] for r in partial}
    qids_missing = expected_qids - qids_clean - qids_partial

    score_dist = collections.Counter()
    total_cost = 0.0
    per_model = collections.defaultdict(float)
    for r in fully_covered:
        s = r.get("score_parsed")
        if s is not None:
            score_dist[int(s)] += 1
        c = r.get("cost_usd")
        if c is not None:
            total_cost += float(c)
            per_model[str(r.get("teacher_model_id"))] += float(c)

    teacher = next(iter(per_model.keys()), "unknown")
    coverage = {
        "n_queries_expected": len(expected_qids),
        "n_queries_fully_covered": len(qids_clean),
        "n_queries_partial": len(qids_partial),
        "n_queries_missing": len(qids_missing),
        "n_labels_clean": len(fully_covered),
        "n_labels_partial": len(partial),
        "n_labels_dropped_with_error": n_rows_with_error,
        "candidates_per_query": args.candidates_per_query,
    }
    cost_report = {
        "total_cost_usd_clean_only": total_cost,
        "per_model_cost_usd_clean_only": dict(per_model),
        "n_calls_clean": len(fully_covered),
        "cost_per_clean_label_usd": total_cost / len(fully_covered) if fully_covered else 0.0,
    }
    score_distribution = {
        "n_scores": sum(score_dist.values()),
        "histogram_int_grade": {str(k): score_dist[k] for k in sorted(score_dist)},
    }

    metrics_dir = args.run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "coverage.clean.json").write_text(json.dumps(coverage, indent=2), encoding="utf-8")
    (metrics_dir / "score_distribution.clean.json").write_text(
        json.dumps(score_distribution, indent=2), encoding="utf-8"
    )
    (args.run_dir / "cost_report.clean.json").write_text(json.dumps(cost_report, indent=2), encoding="utf-8")

    report = _final_report_md(
        run_dir=args.run_dir,
        fixture=args.fixture,
        teacher=teacher,
        coverage=coverage,
        cost_report=cost_report,
        score_distribution=score_distribution,
    )
    (args.run_dir / "FINAL_REPORT.md").write_text(report, encoding="utf-8")

    frozen_path = args.run_dir / ".frozen"
    frozen_path.write_text(
        f"Frozen at {datetime.now(timezone.utc).isoformat()} "
        "by scripts/data/finalize_silver_run.py.\n"
        "Do not resume this run dir; downstream training should consume "
        "silver_labels.clean.jsonl.\n",
        encoding="utf-8",
    )

    print("=== finalize_silver_run summary ===")
    for k, v in coverage.items():
        print(f"  {k}: {v}")
    print(f"  cost_per_clean_label_usd: ${cost_report['cost_per_clean_label_usd']:.6f}")
    print(f"  total_cost_usd_clean_only: ${cost_report['total_cost_usd_clean_only']:.4f}")
    print(f"  score histogram: {score_distribution['histogram_int_grade']}")
    print(f"  clean: {clean_path}")
    print(f"  partial: {partial_path}")
    print(f"  report: {args.run_dir / 'FINAL_REPORT.md'}")


def _expected_qids(fixture_path: Path) -> set[str]:
    out: set[str] = set()
    with fixture_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            out.add(str(row["qid"]))
    return out


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("w", encoding="utf-8") as f:
        for rec in rows:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _final_report_md(
    run_dir: Path,
    fixture: Path,
    teacher: str,
    coverage: dict[str, Any],
    cost_report: dict[str, Any],
    score_distribution: dict[str, Any],
) -> str:
    hist = score_distribution["histogram_int_grade"]
    histogram_table = "\n".join(
        f"| {grade} | {count} | {count / max(1, sum(map(int, hist.values()))) * 100:.1f} % |"
        for grade, count in sorted(hist.items())
    )
    return (
        f"# Frozen silver-data report\n\n"
        f"- Run directory: `{run_dir}`\n"
        f"- Source fixture: `{fixture}`\n"
        f"- Teacher: `{teacher}`\n"
        f"- Frozen at: {datetime.now(timezone.utc).isoformat()}\n\n"
        f"## Clean training corpus\n\n"
        f"`silver_labels.clean.jsonl` contains only fully covered queries "
        f"(all `{coverage['candidates_per_query']}` candidates have a successful, parseable "
        f"silver grade). Downstream SFT should consume this file.\n\n"
        f"| | Value |\n|---|---:|\n"
        f"| Queries expected (fixture) | {coverage['n_queries_expected']} |\n"
        f"| Queries fully covered (clean) | {coverage['n_queries_fully_covered']} "
        f"({coverage['n_queries_fully_covered'] / coverage['n_queries_expected'] * 100:.1f} %) |\n"
        f"| Queries partially covered | {coverage['n_queries_partial']} |\n"
        f"| Queries not yet attempted | {coverage['n_queries_missing']} |\n"
        f"| Silver labels (clean) | {coverage['n_labels_clean']} |\n"
        f"| Silver labels (partial-query) | {coverage['n_labels_partial']} |\n"
        f"| Permanent failures discarded | {coverage['n_labels_dropped_with_error']} |\n\n"
        f"## Cost (clean labels only)\n\n"
        f"| | Value |\n|---|---:|\n"
        f"| Total cost (clean) | ${cost_report['total_cost_usd_clean_only']:.4f} |\n"
        f"| Cost per clean label | ${cost_report['cost_per_clean_label_usd']:.6f} |\n\n"
        f"Note: gateway-failure retries did not bill, but the partial-query and "
        f"discarded-failure rows above represent wall-clock spent without producing "
        f"clean output. The full operational cost is in `cost_report.json` (the "
        f"pre-freeze tally).\n\n"
        f"## Silver score distribution (clean)\n\n"
        f"| Grade | Count | % |\n|---|---:|---:|\n{histogram_table}\n\n"
        f"## Provenance\n\n"
        f"- Pool builder: `scripts/data/build_msmarco_train_silver_pool.py`\n"
        f"- Stratification: 3 top + 10 mid (rank 4–40) + 7 deep (rank 41–100) per query\n"
        f"- Prompt template: `grade_int_v1` (0–3 pointwise rubric)\n\n"
        f"## Extension\n\n"
        f"This run directory is frozen (`.frozen` marker present). Extending coverage "
        f"requires a new run directory and a fixture restricted to the missing qids.\n"
    )


if __name__ == "__main__":
    main()
