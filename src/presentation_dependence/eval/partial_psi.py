"""Rebuild PSI aggregates from completed per-query presentation artifacts.

A permutation sweep is the expensive half of an evaluation cell, so a run that
dies after writing per-query artifacts but before the top-level aggregate is
worth recovering rather than repeating. This rebuilds the aggregate from what
reached disk, and backs the top-up finalizer that merges a partial run with the
run that completes it. Idempotent; only overwrites ``psi/psi_metrics.json`` and
``psi/psi_per_query.json``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, cast

import pytrec_eval
import yaml

from presentation_dependence.eval.psi import Bucket, evaluate_psi
from presentation_dependence.eval.psi_artifacts import build_psi_metrics_envelope
from presentation_dependence.utils.trec import dirname_to_qid


def _load_qrels(qrels_path: str | Path) -> dict[str, dict[str, int]]:
    """Read a TREC qrels file via pytrec_eval (same path the live PSI runner takes)."""
    with open(qrels_path, encoding="utf-8") as f:
        return pytrec_eval.parse_qrel(f)


_PERM_DIR_RE = re.compile(r"^permutation_(\d+)_(.+)$")


def _read_trec_ranking(trec_path: Path) -> list[str]:
    """Parse a TREC run file and return document IDs from best to worst.

    Lines look like ``qid Q0 docid rank score tag``. Ordering uses the
    1-indexed rank column.
    """
    rows: list[tuple[int, str]] = []
    with open(trec_path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 6:
                continue
            try:
                rank = int(parts[3])
            except ValueError:
                continue
            rows.append((rank, parts[2]))
    rows.sort(key=lambda r: r[0])
    return [docid for _, docid in rows]


def _collect_qid_artefacts(qdir: Path) -> tuple[list[list[str]], list[str], list[Bucket] | None]:
    """Return ``(rankings, perm_labels, buckets)`` reconstructed from a single qid dir.

    ``perm_labels`` is aligned with ``rankings``. ``buckets`` is either fully
    aligned or ``None`` so the evaluator can use its non-stratified fallback;
    caller can audit which permutation produced which ranking. A qid dir
    is considered usable iff at least one ``permutation_<idx>_<label>/``
    sub-folder contains a non-empty ``trec_results_raw.txt``.
    """
    perm_dirs: list[tuple[int, str, Path]] = []
    for child in qdir.iterdir():
        if not child.is_dir():
            continue
        m = _PERM_DIR_RE.match(child.name)
        if not m:
            continue
        perm_dirs.append((int(m.group(1)), m.group(2), child))
    perm_dirs.sort(key=lambda x: x[0])

    rankings: list[list[str]] = []
    perm_labels: list[str] = []
    for _, label, pdir in perm_dirs:
        trec_path = pdir / "trec_results_raw.txt"
        if not trec_path.exists():
            continue
        ranking = _read_trec_ranking(trec_path)
        if not ranking:
            continue
        rankings.append(ranking)
        perm_labels.append(label)

    pos_path = qdir / "input_positions.json"
    if not pos_path.exists():
        return rankings, perm_labels, None

    with open(pos_path, encoding="utf-8") as f:
        data = json.load(f)
    labels = data.get("labels", [])
    if len(labels) < len(rankings):
        # Incomplete bucket metadata prevents stratified ΔnDCG/Zeng PSI for
        # this qid. Let evaluate_psi fall back to max-min.
        return rankings, perm_labels, None

    raw_buckets = [str(b) for b in labels[: len(rankings)]]
    if any(bucket not in {"top", "middle", "bottom"} for bucket in raw_buckets):
        # Unknown labels cannot support stratified metrics. Preserve the
        # rankings and use evaluate_psi's non-stratified fallback.
        return rankings, perm_labels, None
    buckets: list[Bucket] = [cast(Bucket, bucket) for bucket in raw_buckets]
    return rankings, perm_labels, buckets


def _load_qrels_for_config(cfg: dict) -> dict[str, dict[str, int]] | None:
    """Load qrels from a resolved run config, mapping container paths locally."""
    qrels_path = (cfg.get("eval") or {}).get("qrels_path")
    if not qrels_path:
        return None

    qrels_p = Path(qrels_path)
    # Legacy container artifacts may carry mounted paths like
    # ``/opt/ml/input/data/<slug>/qrels.txt``. Map those back to the local
    # ``data/<slug>/qrels.txt`` so old runs aggregate outside the container.
    if not qrels_p.exists() and "/opt/ml/input/data/" in str(qrels_p):
        local = Path("data") / Path(*qrels_p.parts[qrels_p.parts.index("data") + 1 :])
        if local.exists():
            qrels_p = local

    if qrels_p.exists():
        print(f"[aggregate] loaded qrels from {qrels_p}")
        return _load_qrels(qrels_p)

    print(f"[aggregate] qrels not found ({qrels_path}); ΔnDCG will be None.")
    return None


def _select_presentations(
    rankings: list[list[str]],
    perm_labels: list[str],
    buckets: list[Bucket] | None,
    *,
    label_prefix: str | None,
) -> tuple[list[list[str]], list[Bucket] | None]:
    """Filter aligned rankings and buckets to one labelled protocol."""
    if label_prefix is None:
        return rankings, buckets
    selected = [index for index, label in enumerate(perm_labels) if label.startswith(label_prefix)]
    selected_rankings = [rankings[index] for index in selected]
    selected_buckets: list[Bucket] | None = [buckets[index] for index in selected] if buckets is not None else None
    return selected_rankings, selected_buckets


def _collect_completed_psi_inputs(
    pq_dir: Path,
    *,
    min_perms: int,
    label_prefix: str | None = None,
) -> tuple[dict[str, list[list[str]]], dict[str, list[Bucket]], dict]:
    """Collect complete per-qid PSI inputs from a per_query_results directory.

    ``min_perms`` checks completeness against all generated presentations.
    ``label_prefix`` then selects which presentation-label prefix to aggregate
    without deleting the other retained outputs.
    """
    rankings_over_K: dict[str, list[list[str]]] = {}
    positions: dict[str, list[Bucket]] = {}
    stats = {
        "n_total": 0,
        "n_kept": 0,
        "n_partial": 0,
        "perm_label_counter": {},
    }

    for qdir in sorted(pq_dir.iterdir()):
        if not qdir.is_dir():
            continue
        stats["n_total"] += 1
        rankings, perm_labels, buckets = _collect_qid_artefacts(qdir)
        if not rankings:
            continue

        for label in perm_labels:
            stats["perm_label_counter"][label] = stats["perm_label_counter"].get(label, 0) + 1

        if len(rankings) < min_perms:
            stats["n_partial"] += 1
            continue

        rankings, buckets = _select_presentations(
            rankings,
            perm_labels,
            buckets,
            label_prefix=label_prefix,
        )
        if len(rankings) < min_perms:
            stats["n_partial"] += 1
            continue

        # Live PSI writes per-query directories through qid_to_dirname() so
        # qids containing "/" stay as a single path segment. Decode here before
        # passing keys to evaluate_psi(); qrels use the original raw qids.
        qid = dirname_to_qid(qdir.name)
        if qid in rankings_over_K:
            raise ValueError(
                f"Duplicate qid after decoding per-query directory name {qdir.name!r}: {qid!r}. "
                "Refusing to aggregate ambiguous PSI artefacts."
            )
        rankings_over_K[qid] = rankings
        if buckets is not None:
            positions[qid] = buckets
        stats["n_kept"] += 1

    return rankings_over_K, positions, stats


def rebuild_partial_psi_metrics(
    run_dir: Path,
    *,
    min_perms: int | None = None,
) -> dict[str, Any]:
    """Rebuild top-level PSI metrics from completed per-query artefacts."""
    psi_dir = run_dir / "psi"
    if not psi_dir.exists():
        raise FileNotFoundError(f"No psi/ subdirectory in {run_dir}")
    pq_dir = psi_dir / "per_query_results"
    if not pq_dir.exists():
        raise FileNotFoundError(f"No psi/per_query_results in {run_dir}")

    cfg_path = run_dir / "resolved_config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"No resolved_config.yaml in {run_dir}")
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Expected a YAML mapping in {cfg_path}")

    rb = cfg.get("robustness") or {}
    K = int(rb.get("K", 10))
    perturbations = rb.get("perturbations", ["random_shuffle"])
    k_cutoff = int(rb.get("k_cutoff_for_ndcg", 10))

    # The reported protocol owns completeness.  When random draws exist, the
    # three targeted injections are not required and may legitimately be absent
    # for queries with no relevant document in the candidate pool.
    random_only = "random_shuffle" in perturbations
    expected_perms = K if random_only else (3 if "middle_injection" in perturbations else 0)
    if min_perms is None:
        min_perms = expected_perms

    qrels = _load_qrels_for_config(cfg)
    rankings_over_K, positions, stats = _collect_completed_psi_inputs(
        pq_dir,
        min_perms=min_perms,
        label_prefix="random_s" if random_only else None,
    )
    n_total = stats["n_total"]
    n_kept = stats["n_kept"]
    n_partial = stats["n_partial"]
    perm_label_counter = stats["perm_label_counter"]

    print(f"[aggregate] qids found:    {n_total}")
    print(f"[aggregate] qids kept:     {n_kept}  (>= {min_perms} perms)")
    print(f"[aggregate] qids partial:  {n_partial}  (< {min_perms} perms, dropped)")
    print(f"[aggregate] expected perms per qid (config): {expected_perms}")
    print("[aggregate] perm label coverage:")
    for label, n in sorted(perm_label_counter.items()):
        print(f"             {label:>20}  {n}")

    if not rankings_over_K:
        raise ValueError(
            f"No qids met the min_perms={min_perms} threshold; "
            "lower the threshold via --min-perms or check the on-disk layout."
        )

    metrics = evaluate_psi(
        rankings_over_K=rankings_over_K,
        scores_over_K=None,  # generative-listwise has no per-doc scores
        qrels=qrels,
        injection_positions=positions or None,
        k_cutoff=k_cutoff,
    )
    metrics["protocol"]["presentation_aggregation"] = (
        "random_shuffle only" if random_only else "all generated presentations (no random_shuffle configured)"
    )
    metrics["protocol"]["excluded_from_presentation_aggregation"] = (
        [p for p in perturbations if p != "random_shuffle"] if random_only else []
    )

    compact = build_psi_metrics_envelope(
        cfg,
        metrics,
        perturbations=perturbations,
        K=K,
        seeds=rb.get("seeds", list(range(K))),
        exp_id=cfg.get("id", run_dir.parent.name),
        aggregation_note=(
            "Aggregated offline by presentation_dependence.eval.partial_psi."
            "rebuild_partial_psi_metrics — "
            f"qids_kept={n_kept}, qids_partial_dropped={n_partial}, "
            f"min_perms_threshold={min_perms}."
        ),
    )
    metrics_path = psi_dir / "psi_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(compact, f, indent=2, ensure_ascii=False)
    print(f"[aggregate] wrote {metrics_path}")

    full_path = psi_dir / "psi_per_query.json"
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(metrics["per_query"], f, indent=2, ensure_ascii=False)
    print(f"[aggregate] wrote {full_path}")

    return compact
