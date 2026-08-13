#!/usr/bin/env python
"""Post-hoc scalarization analysis for expected-grade probability artefacts.

Reads ``per_query_results/*/detailed_results.json`` from a completed run whose
reranker emitted ``grade_probabilities_init_order`` and re-scores the same
queries under alternate scalarizations of ``P(grade in {0,1,2,3})``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np
import pytrec_eval  # type: ignore
import yaml

from presentation_dependence.utils.trec import dirname_to_qid


Scalarizer = Callable[[list[float]], float]


def _expected_grade(probs: list[float]) -> float:
    return sum(i * float(p) for i, p in enumerate(probs)) / 3.0


def _ordinal_confidence(probs: list[float]) -> float:
    best = max(range(len(probs)), key=lambda i: probs[i])
    return (best + float(probs[best])) / 4.0


def _argmax_grade(probs: list[float]) -> float:
    return max(range(len(probs)), key=lambda i: probs[i]) / 3.0


def _binary_relevance(probs: list[float]) -> float:
    return float(probs[2]) + float(probs[3])


def _very_relevant(probs: list[float]) -> float:
    return float(probs[3])


def _dcg_style(probs: list[float]) -> float:
    return (float(probs[1]) + 3.0 * float(probs[2]) + 7.0 * float(probs[3])) / 7.0


SCALARIZERS: dict[str, Scalarizer] = {
    "expected_grade": _expected_grade,
    "ordinal_confidence": _ordinal_confidence,
    "argmax_grade": _argmax_grade,
    "binary_relevance": _binary_relevance,
    "very_relevant": _very_relevant,
    "dcg_style": _dcg_style,
}


def _load_qrels(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return pytrec_eval.parse_qrel(f)


def _local_experiment_config(run_dir: Path, resolved: dict) -> dict:
    config_id = resolved.get("id")
    if not config_id:
        return {}
    path = Path.cwd() / "configs" / "experiments" / f"{config_id}.yaml"
    if not path.is_file():
        return {}
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return value if isinstance(value, dict) else {}


def _resolve_qrels(run_dir: Path) -> Path:
    cfg = yaml.safe_load((run_dir / "resolved_config.yaml").read_text(encoding="utf-8")) or {}
    raw = Path(str((cfg.get("eval") or {})["qrels_path"]))
    if raw.is_absolute() or raw.exists():
        if raw.exists():
            return raw
        legacy_container_prefix = Path("/opt/ml/input/data")
        try:
            rel = raw.relative_to(legacy_container_prefix)
        except ValueError:
            pass
        else:
            local_channel_path = Path.cwd() / "data" / rel
            if local_channel_path.exists():
                return local_channel_path
        local_cfg = _local_experiment_config(run_dir, cfg)
        local_raw = Path(str((local_cfg.get("eval") or {}).get("qrels_path", "")))
        local_path = local_raw if local_raw.is_absolute() else Path.cwd() / local_raw
        if local_path.is_file():
            return local_path
        return raw
    candidate = Path.cwd() / raw
    if candidate.exists():
        return candidate
    return raw


def _resolve_fixture(run_dir: Path) -> tuple[Path, int] | None:
    cfg = yaml.safe_load((run_dir / "resolved_config.yaml").read_text(encoding="utf-8")) or {}
    data = cfg.get("data") or {}
    if data.get("dataloader_class") != "FixtureLoader":
        return None
    raw = Path(str(data["run_path"]))
    local_cfg = _local_experiment_config(run_dir, cfg)
    local_data = local_cfg.get("data") or {}
    candidates = [raw]
    if local_data.get("run_path"):
        local_raw = Path(str(local_data["run_path"]))
        candidates.append(local_raw if local_raw.is_absolute() else Path.cwd() / local_raw)
    legacy_container_prefix = Path("/opt/ml/input/data")
    try:
        relative = raw.relative_to(legacy_container_prefix)
    except ValueError:
        pass
    else:
        candidates.append(Path.cwd() / "data" / relative)
    candidates.append(Path.cwd() / raw)
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise FileNotFoundError(f"Could not resolve fixture for {run_dir}: {raw}")
    return path, int(data["k_input"])


def _fixture_passages(run_dir: Path) -> dict[str, list[dict]] | None:
    resolved = _resolve_fixture(run_dir)
    if resolved is None:
        return None
    path, k_input = resolved
    rows: dict[str, list[dict]] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, start=1):
            record = json.loads(line)
            qid = str(record["qid"])
            if qid in rows:
                raise ValueError(f"{path}:{line_no}: duplicate qid {qid}")
            rows[qid] = list(record["passages"])[:k_input]
    return rows


def _iter_query_records(run_dir: Path):
    per_query = run_dir / "per_query_results"
    if not per_query.is_dir():
        raise FileNotFoundError(f"Missing per_query_results under {run_dir}")
    for qdir in sorted(per_query.iterdir()):
        if not qdir.is_dir():
            continue
        detailed = qdir / "detailed_results.json"
        if not detailed.is_file():
            continue
        yield dirname_to_qid(qdir.name), json.loads(detailed.read_text(encoding="utf-8"))


def _evaluate(run: dict[str, dict[str, float]], qrels: dict, measures: set[str]) -> dict[str, float]:
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, measures)
    per_query = evaluator.evaluate(run)
    out: dict[str, float] = {"n_queries": float(len(per_query))}
    if not per_query:
        return out
    metric_names = sorted({name for metrics in per_query.values() for name in metrics})
    for metric in metric_names:
        vals = [float(metrics[metric]) for metrics in per_query.values() if metric in metrics]
        out[f"mean_{metric}"] = float(np.mean(vals))
        out[f"std_{metric}"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return out


def _passages_in_init_order(qid: str, record: dict) -> list[dict]:
    """Recover input-order passages from sorted ``top_k_psgs`` + scores.

    ``scores_init_order`` is input-order, while ``top_k_psgs`` is sorted by the
    original scalarization. The original scalar score is also copied onto each
    passage, so matching those scores reconstructs the doc ids needed to align
    ``grade_probabilities_init_order``.
    """
    passages = list(record.get("top_k_psgs") or [])
    scores_init = [float(x) for x in record.get("scores_init_order") or []]
    if len(passages) != len(scores_init):
        raise ValueError(f"qid={qid}: top_k_psgs length != scores_init_order length")

    unused = list(range(len(passages)))
    out: list[dict] = []
    for score in scores_init:
        closest = min(
            (
                (abs(float(passages[passage_idx].get("score")) - score), pos, passage_idx)
                for pos, passage_idx in enumerate(unused)
            ),
            default=None,
        )
        match_pos: int | None = None
        if closest is not None:
            _, pos, passage_idx = closest
            candidate_score = float(passages[passage_idx].get("score"))
            if math.isclose(candidate_score, score, rel_tol=1e-2, abs_tol=1e-12):
                match_pos = pos
        if match_pos is None:
            raise ValueError(f"qid={qid}: could not match persisted score {score} back to a passage")
        out.append(passages[unused.pop(match_pos)])
    return out


def _score_records(
    run_dir: Path,
    scalarizer: Scalarizer,
    *,
    fixture_passages: dict[str, list[dict]] | None = None,
) -> dict[str, dict[str, float]]:
    run: dict[str, dict[str, float]] = {}
    for qid, record in _iter_query_records(run_dir):
        passages = fixture_passages[qid] if fixture_passages is not None else _passages_in_init_order(qid, record)
        vectors = record.get("grade_probabilities_init_order") or []
        scores_init = record.get("scores_init_order") or []
        if len(vectors) != len(scores_init) or len(vectors) != len(passages):
            raise ValueError(
                f"qid={qid}: cannot align grade vectors ({len(vectors)}), scores ({len(scores_init)}), "
                f"and passages ({len(passages)})"
            )
        ranked_pids = {str(passage["pid"]) for passage in record["top_k_psgs"]}
        passage_pids = {str(passage["pid"]) for passage in passages}
        if ranked_pids != passage_pids:
            raise ValueError(f"qid={qid}: fixture and persisted passage sets differ")
        doc_scores: dict[str, float] = {}
        for passage, probs in zip(passages, vectors, strict=True):
            score = float(scalarizer([float(x) for x in probs]))
            if not math.isfinite(score):
                raise ValueError(f"qid={qid} pid={passage.get('pid')}: non-finite scalarized score")
            doc_scores[str(passage["pid"])] = score
        run[str(qid)] = doc_scores
    return run


def analyze_run(
    run_dir: Path,
    *,
    measures: set[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Evaluate every registered scalarization on one explicit run directory."""
    selected_measures = measures or {
        "ndcg_cut_10",
        "ndcg_cut_5",
        "map",
        "recip_rank",
    }
    qrels = _load_qrels(_resolve_qrels(run_dir))
    fixture_passages = _fixture_passages(run_dir)
    return {
        name: _evaluate(
            _score_records(
                run_dir,
                scalarizer,
                fixture_passages=fixture_passages,
            ),
            qrels,
            selected_measures,
        )
        for name, scalarizer in SCALARIZERS.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON output path.")
    parser.add_argument(
        "--measures",
        nargs="+",
        default=["ndcg_cut_10", "ndcg_cut_5", "map", "recip_rank"],
    )
    args = parser.parse_args()

    run_dir = args.run_dir
    payload = analyze_run(run_dir, measures=set(args.measures))

    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
