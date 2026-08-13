#!/usr/bin/env python
"""Record a run directory in docs/results_index.yaml."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml  # type: ignore[reportMissingModuleSource]


VALID_STATUSES = {"planned", "running", "complete", "partial", "topup-needed", "superseded", "failed", "diagnostic"}
VALID_ROLES = {"primary", "reference", "ablation", "negative-control", "smoke", "topup", "diagnostic"}

QUALITY_KEYS = (
    "n_queries",
    "mean_ndcg_cut_1",
    "mean_ndcg_cut_10",
    "mean_ndcg_cut_5",
    "mean_map",
    "mean_recip_rank",
)

PSI_KEYS = (
    "n_queries",
    "K_permutations",
    "zeng_psi_corpus",
    "mean_zeng_psi",
    "mean_tau_based_psi",
    "mean_kendall_tau",
    "mean_delta_ndcg_stratified",
    "mean_delta_ndcg_max_min",
    "mean_rank_variance",
    "mean_score_variance",
)

SURFACE_ORDER = {
    "dl19": 0,
    "dl20": 1,
    "beir": 2,
    "legal-a": 3,
    "legal-b": 4,
}

STATUS_ORDER = {
    "complete": 0,
    "partial": 1,
    "topup-needed": 2,
    "running": 3,
    "planned": 4,
    "diagnostic": 5,
    "failed": 6,
    "superseded": 7,
}

ROLE_ORDER = {
    "primary": 0,
    "reference": 1,
    "ablation": 2,
    "topup": 3,
    "diagnostic": 4,
    "negative-control": 5,
    "smoke": 6,
}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _portable_path(path: str | Path) -> str:
    """Return a project-relative path when ``path`` is inside this project."""
    value = Path(path)
    if not value.is_absolute():
        return value.as_posix()
    try:
        return value.relative_to(_project_root()).as_posix()
    except ValueError:
        return value.as_posix()


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _metric_subset(metrics: dict[str, Any]) -> dict[str, Any]:
    return {k: metrics[k] for k in QUALITY_KEYS if k in metrics}


def _matched_variance_subset(metrics: dict[str, Any]) -> dict[str, Any]:
    """Map matched-variance aggregate names onto the result-index quality schema."""
    aggregate = metrics.get("aggregate") or {}
    out = {
        key: aggregate[key]
        for key in (
            "n_queries",
            "n_candidates",
            "mean_score_variance",
            "rms_score_sd",
            "max_score_range",
            "min_unique_batch_positions",
            "min_unique_batch_compositions",
        )
        if key in aggregate
    }
    for cutoff in (1, 10):
        source = f"mean_ndcg_cut_{cutoff}_k10_score_average"
        if source in aggregate:
            out[f"mean_ndcg_cut_{cutoff}"] = aggregate[source]
    return out


def _psi_subset(psi_metrics: dict[str, Any]) -> dict[str, Any]:
    aggregate = psi_metrics.get("aggregate") or {}
    out = {k: aggregate[k] for k in PSI_KEYS if k in aggregate}
    for k in ("tau_psi_context_batch_size", "tau_psi_at_B_caption", "tau_psi_geometry_schema"):
        if k in psi_metrics:
            out[k] = psi_metrics[k]
    return out


def infer_series(exp_id: str) -> str:
    """Infer the experiment series token: the id up to its first hyphen."""
    if exp_id.startswith("_"):
        return "helper"
    return exp_id.split("-", 1)[0]


def infer_model_family(exp_id: str, reranker: dict[str, Any]) -> str:  # noqa: C901
    """Infer a stable model-family label for the dashboard/index."""
    cls = str(reranker.get("class") or "")
    model_name = str(reranker.get("model_name") or reranker.get("model_id") or "").lower()
    if "mxbai" in cls.lower() or "mxbai" in model_name:
        return "mxbai"
    if "qwen" in cls.lower() or "qwen" in model_name:
        return "qwen3"
    if "rankzephyr" in cls.lower() or "rank_zephyr" in model_name:
        return "rankzephyr"
    if "jina" in cls.lower() or "jina" in model_name:
        return "jina-v3"
    if "identity" in cls.lower():
        return "identity"
    if "claude" in model_name:
        if "opus" in model_name or "opus" in exp_id.lower():
            return "claude-opus-4.6"
        if "sonnet" in model_name:
            return "claude-sonnet-4.6"
        return "claude"
    if "gpt-5-mini" in model_name:
        return "gpt-5-mini"
    if "closedmodelgenbsc" in cls.lower() or "gpt" in model_name:
        return "gpt-5.4"
    return cls or "unknown"


def infer_paradigm(reranker: dict[str, Any]) -> str:
    """Infer the scoring-family label used in analysis tables."""
    cls = str(reranker.get("class") or "")
    if cls in {"MxbaiPointwise"}:
        return "pointwise"
    if cls in {"JinaListwiseReranker"}:
        return "scoring_listwise"
    if cls == "RankZephyrReranker":
        return "generative_listwise"
    if cls in {"Qwen3Reranker"}:
        return "pointwise" if int(reranker.get("docs_per_score_forward", 20) or 20) == 1 else "batched_pointwise"
    if cls in {"IdentityReranker"}:
        return "pointwise"
    if cls == "ClosedModelGenBscReranker":
        return "batched_pointwise"
    return "unknown"


def infer_protocol(reranker: dict[str, Any]) -> str:
    """Infer a concise protocol label."""
    cls = str(reranker.get("class") or "")
    if cls == "RankZephyrReranker":
        protocol = str(reranker.get("rankzephyr_protocol") or "sliding")
        if protocol == "block_local_rrf":
            return f"block{reranker.get('block_size', reranker.get('window_size', 20))}-rrf"
        return f"w{reranker.get('window_size', 20)}-s{reranker.get('stride', 10)}"
    if cls == "Qwen3Reranker":
        mode = str(reranker.get("scoring_mode") or "official_pairwise")
        width = reranker.get("docs_per_score_forward", 1 if mode == "official_pairwise" else 20)
        return f"{mode}-b{width}"
    if cls in {"MxbaiPointwise", "JinaListwiseReranker", "IdentityReranker"}:
        return "native"
    if cls == "ClosedModelGenBscReranker":
        scoring = reranker.get("scoring") or {}
        width = scoring.get("subset_size", reranker.get("docs_per_score_forward", 20))
        return f"pathc-genbsc-b{width}"
    return "unknown"


def infer_surface(dataset: str) -> str:
    """Map dataset slug to a broad collection label (`surface` index field)."""
    ds = dataset.lower()
    if "dl19" in ds:
        return "dl19"
    if "dl20" in ds:
        return "dl20"
    if ds.startswith("beir-") or "beir-" in ds:
        return "beir"
    if "legal-a" in ds:
        return "legal-a"
    if "legal-b" in ds:
        return "legal-b"
    return dataset or "unknown"


def infer_benchmark_group(surface: str) -> str:
    """Return a dashboard benchmark group for a collection label."""
    if surface in {"dl19", "dl20"}:
        return "msmarco-passage"
    if surface in {"legal-a", "legal-b"}:
        return "legal"
    return surface


def build_result_record(  # noqa: C901
    run_dir: Path,
    *,
    status: str,
    role: str = "primary",
    surface: str | None = None,
    dataset: str | None = None,
    protocol: str | None = None,
    summary: str | None = None,
    notes: list[str] | None = None,
    caveats: list[str] | None = None,
    results_anchor: str | None = None,
    supersedes: list[str] | None = None,
    superseded_by: str | None = None,
) -> dict[str, Any]:
    """Build a result-index record from a run directory."""
    if status not in VALID_STATUSES:
        raise ValueError(f"Unknown status {status!r}; expected one of {sorted(VALID_STATUSES)}")
    if role not in VALID_ROLES:
        raise ValueError(f"Unknown role {role!r}; expected one of {sorted(VALID_ROLES)}")

    cfg_path = run_dir / "resolved_config.yaml"
    metrics_path = run_dir / "metrics.json"
    matched_variance_path = run_dir / "matched_variance" / "matched_variance_metrics.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Missing {cfg_path}")
    if not metrics_path.is_file() and not matched_variance_path.is_file():
        raise FileNotFoundError(f"Missing {metrics_path} and {matched_variance_path}")

    cfg = _load_yaml(cfg_path)
    if metrics_path.is_file():
        metrics = _load_json(metrics_path)
        indexed_metrics = _metric_subset(metrics)
    else:
        metrics_path = matched_variance_path
        metrics = _load_json(metrics_path)
        indexed_metrics = _matched_variance_subset(metrics)
    exp_id = str(cfg.get("id") or run_dir.parent.name)
    data = cfg.get("data") or {}
    reranker = cfg.get("reranker") or {}
    inferred_dataset = Path(str(data.get("run_path", ""))).parent.name or str(data.get("topics", "-"))
    dataset = dataset or inferred_dataset
    surface = surface or infer_surface(dataset)

    record: dict[str, Any] = {
        "exp_id": exp_id,
        "status": status,
        "role": role,
        "series": infer_series(exp_id),
        "model_family": infer_model_family(exp_id, reranker),
        "model_name": str(reranker.get("model_name") or reranker.get("model_id") or "-"),
        "model_class": str(reranker.get("class", "-")),
        "paradigm": infer_paradigm(reranker),
        "protocol": protocol or infer_protocol(reranker),
        "surface": surface,
        "dataset": dataset,
        "benchmark_group": infer_benchmark_group(surface),
        "run_dir": _portable_path(run_dir),
        "config": _portable_path(cfg.get("_source_config") or f"configs/experiments/{exp_id}.yaml"),
        "metrics_path": _portable_path(metrics_path),
        "metrics": indexed_metrics,
    }
    if metrics_path == matched_variance_path:
        record["matched_variance_metrics_path"] = _portable_path(matched_variance_path)

    psi_path = run_dir / "psi" / "psi_metrics.json"
    if psi_path.is_file():
        psi_metrics = _load_json(psi_path)
        record["psi_metrics_path"] = _portable_path(psi_path)
        record["psi"] = _psi_subset(psi_metrics)

    if summary:
        record["summary"] = summary
    if notes:
        record["notes"] = notes
    if caveats:
        record["caveats"] = caveats
    if results_anchor:
        record["results_anchor"] = results_anchor
    if supersedes:
        record["supersedes"] = supersedes
    if superseded_by:
        record["superseded_by"] = superseded_by
    return record


def upsert_result(index: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    """Insert or replace a record keyed by exp_id + run_dir."""
    out = dict(index)
    rows = list(out.get("results") or [])
    key = (record["exp_id"], record["run_dir"])
    rows = [r for r in rows if (r.get("exp_id"), r.get("run_dir")) != key]
    rows.append(record)
    out["results"] = sort_result_records(rows)
    out.setdefault("schema_version", 1)
    return out


def _rank(mapping: dict[str, int], value: Any) -> tuple[int, str]:
    value_s = str(value or "")
    return (mapping.get(value_s, len(mapping)), value_s)


def result_sort_key(record: dict[str, Any]) -> tuple[Any, ...]:
    """Return the canonical result-index sort key.

    The YAML is optimized for research scanning: first by collection (`surface`
    field) and dataset, then by model family and protocol, with status/role as
    tie-breakers.
    """
    return (
        _rank(SURFACE_ORDER, record.get("surface")),
        str(record.get("dataset") or ""),
        str(record.get("series") or ""),
        str(record.get("model_family") or ""),
        str(record.get("protocol") or ""),
        _rank(STATUS_ORDER, record.get("status")),
        _rank(ROLE_ORDER, record.get("role")),
        str(record.get("exp_id") or ""),
        str(record.get("run_dir") or ""),
    )


def sort_result_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return result records in canonical YAML order."""
    return sorted(rows, key=result_sort_key)


def validate_result_record(record: dict[str, Any], *, index_root: Path | None = None) -> None:  # noqa: C901
    """Validate a single result-index record."""
    required = (
        "exp_id",
        "status",
        "role",
        "series",
        "model_family",
        "model_class",
        "paradigm",
        "protocol",
        "surface",
        "dataset",
        "benchmark_group",
        "run_dir",
        "config",
        "metrics_path",
        "metrics",
    )
    missing = [k for k in required if k not in record]
    if missing:
        raise ValueError(f"Result record {record.get('exp_id', '<unknown>')} missing fields: {missing}")
    if record["status"] not in VALID_STATUSES:
        raise ValueError(f"Invalid status {record['status']!r}")
    if record["role"] not in VALID_ROLES:
        raise ValueError(f"Invalid role {record['role']!r}")
    if not isinstance(record.get("metrics"), dict):
        raise ValueError("metrics must be a mapping")
    if record["status"] == "complete" and not ({"mean_ndcg_cut_1", "mean_ndcg_cut_10"} & set(record["metrics"])):
        raise ValueError(f"Complete result {record['exp_id']} missing an nDCG headline")

    if record.get("psi_metrics_path"):
        psi = record.get("psi")
        if not isinstance(psi, dict):
            raise ValueError(f"Result {record['exp_id']} has psi_metrics_path but no psi mapping")
        if "mean_tau_based_psi" not in psi and "zeng_psi_corpus" not in psi:
            raise ValueError(f"Result {record['exp_id']} has PSI metrics but no PSI headline fields")

    if index_root is not None and record["status"] not in {"planned", "running", "failed", "superseded"}:
        run_dir = index_root / record["run_dir"]
        if not run_dir.exists():
            raise ValueError(f"Indexed run_dir does not exist: {record['run_dir']}")


def validate_results_index(index: dict[str, Any], *, index_root: Path | None = None) -> None:
    """Validate a results index document."""
    if int(index.get("schema_version", 0)) != 1:
        raise ValueError("results_index.yaml must have schema_version: 1")
    rows = index.get("results")
    if not isinstance(rows, list):
        raise ValueError("results_index.yaml field `results` must be a list")
    seen: set[tuple[str, str]] = set()
    for row in rows:
        validate_result_record(row, index_root=index_root)
        key = (row["exp_id"], row["run_dir"])
        if key in seen:
            raise ValueError(f"Duplicate result entry for {key}")
        seen.add(key)
    if rows != sort_result_records(rows):
        raise ValueError("results_index.yaml results are not in canonical sort order; run --sort-index")


def emit_markdown_row(record: dict[str, Any]) -> str:
    """Emit a compact dashboard row for a result record."""
    metric = record.get("metrics", {}).get("mean_ndcg_cut_10", "—")
    metric_s = f"{metric:.3f}" if isinstance(metric, int | float) else str(metric)
    psi = record.get("psi", {})
    taupsi = psi.get("mean_tau_based_psi")
    taupsi_s = f"{taupsi:.3f}" if isinstance(taupsi, int | float) else "—"
    summary = record.get("summary", "")
    return (
        f"| `{record['exp_id']}` | {record['surface']} | {record['protocol']} | "
        f"{metric_s} | {taupsi_s} | {record['status']} | {summary} |"
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, nargs="?", help="Path to runs/<ID>/<timestamp>/")
    parser.add_argument("--status", choices=sorted(VALID_STATUSES), default="complete")
    parser.add_argument("--role", choices=sorted(VALID_ROLES), default="primary")
    parser.add_argument(
        "--collection",
        "--surface",
        dest="collection",
        default=None,
        help="Collection label written to results_index field `surface`. --surface is a deprecated alias.",
    )
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--protocol", default=None)
    parser.add_argument("--summary", default=None)
    parser.add_argument("--note", action="append", dest="notes", default=[])
    parser.add_argument("--caveat", action="append", dest="caveats", default=[])
    parser.add_argument("--results-anchor", default=None)
    parser.add_argument("--supersedes", action="append", default=[])
    parser.add_argument("--superseded-by", default=None)
    parser.add_argument(
        "--index",
        type=Path,
        default=_project_root() / "docs" / "results_index.yaml",
        help="Machine-readable results index to update.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print record but do not write the index.")
    parser.add_argument("--emit-markdown", action="store_true", help="Print a compact scoreboard row for the record.")
    parser.add_argument("--validate-index", action="store_true", help="Validate the index and exit.")
    parser.add_argument("--sort-index", action="store_true", help="Rewrite the index in canonical sort order and exit.")
    parser.add_argument(
        "--check-paths",
        action="store_true",
        help="Also require indexed local run directories to exist.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the result-recording CLI."""
    args = parse_args()
    if args.sort_index:
        index = _load_yaml(args.index)
        index["results"] = sort_result_records(list(index.get("results") or []))
        validate_results_index(
            index,
            index_root=_project_root() if args.check_paths else None,
        )
        args.index.parent.mkdir(parents=True, exist_ok=True)
        with open(args.index, "w", encoding="utf-8") as f:
            yaml.safe_dump(index, f, sort_keys=False)
        print(f"[record-result] sorted {args.index}")
        return 0
    if args.validate_index:
        index = _load_yaml(args.index)
        validate_results_index(
            index,
            index_root=_project_root() if args.check_paths else None,
        )
        print(f"[record-result] validated {args.index}")
        return 0
    if args.run_dir is None:
        raise SystemExit("run_dir is required unless --validate-index or --sort-index is set")

    record = build_result_record(
        args.run_dir,
        status=args.status,
        role=args.role,
        surface=args.collection,
        dataset=args.dataset,
        protocol=args.protocol,
        summary=args.summary,
        notes=args.notes or None,
        caveats=args.caveats or None,
        results_anchor=args.results_anchor,
        supersedes=args.supersedes or None,
        superseded_by=args.superseded_by,
    )
    if args.emit_markdown:
        print(emit_markdown_row(record))
        return 0
    if args.dry_run:
        print(yaml.safe_dump(record, sort_keys=False))
        return 0

    index = _load_yaml(args.index) if args.index.exists() else {"schema_version": 1, "results": []}
    updated = upsert_result(index, record)
    validate_results_index(
        updated,
        index_root=_project_root() if args.check_paths else None,
    )
    args.index.parent.mkdir(parents=True, exist_ok=True)
    with open(args.index, "w", encoding="utf-8") as f:
        yaml.safe_dump(updated, f, sort_keys=False)
    print(f"[record-result] recorded {record['exp_id']} -> {args.index}")
    print("[record-result] next: validate the results index")
    print("[record-result] add prose only for historical caveats or interpretation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
