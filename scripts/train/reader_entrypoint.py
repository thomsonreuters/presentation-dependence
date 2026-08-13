#!/usr/bin/env python
"""Container entry point for downstream reader bridges.

The reranker entrypoint (``entrypoint.py``) drives ``ExperimentManager`` /
``EvalManager`` and scrapes ``eval.measures``. The reader bridge generates
answers with a frozen LLM over the scorer's ranked top-k, so it has its own
entrypoint that drives the reader library directly. It mirrors the same
container conventions: read the config from the code bundle, resolve mounted
input channels from ``SLM_CHANNEL_*`` (falling back to the legacy
``SM_CHANNEL_*`` aliases), write the artifact tree under the output directory,
and emit ``METRIC <name>=<float>`` lines a job runner can scrape.

Arguments:
---------
``--config-path <path>``   reader config YAML in the bundle (configs/reader/<ID>.yaml)
``--data-channel <name>``  channel holding fixture.jsonl, qrels.txt, and task labels
``--scorer-channel <name>``  channel holding beta_gamma_scores.parquet

Reader jobs are evaluation-only. They use deterministic decoding and preserve
the scorer's ranked order.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from presentation_dependence.utils.run_paths import resolve_trial_name


def _channel_dir(channel: str) -> Path:
    """Resolve SLM-first channels with legacy SM and /opt/ml compatibility."""
    suffix = channel.upper().replace("-", "_").replace(".", "_")
    return Path(
        os.environ.get(f"SLM_CHANNEL_{suffix}")
        or os.environ.get(f"SM_CHANNEL_{suffix}")
        or f"/opt/ml/input/data/{channel}"
    )


def _output_dir() -> Path:
    """Resolve SLM-first output with legacy SM and /opt/ml compatibility."""
    return Path(os.environ.get("SLM_OUTPUT_DIR") or os.environ.get("SM_OUTPUT_DATA_DIR") or "/opt/ml/output/data")


def _trial() -> str:
    """Return an SLM-first trial name with a legacy launcher alias."""
    return resolve_trial_name()


def _emit_metrics(scalars: dict) -> None:
    for name, value in sorted(scalars.items()):
        print(f"METRIC {name}={float(value):.6f}", flush=True)


def _build_engine(reader_cfg: dict):
    from presentation_dependence.reader.engine import VLLMReaderEngine

    return VLLMReaderEngine(
        model_name=reader_cfg["model_name"],
        max_model_len=int(reader_cfg.get("max_model_len", 8192)),
        max_tokens=int(reader_cfg.get("max_tokens", 64)),
        tensor_parallel_size=int(reader_cfg.get("tensor_parallel_size", 1)),
        gpu_memory_utilization=float(reader_cfg.get("gpu_memory_utilization", 0.9)),
        enable_thinking=bool(reader_cfg.get("enable_thinking", False)),
        choices=reader_cfg.get("choices"),
    )


def main() -> int:  # noqa: C901
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config-path", required=True)
    ap.add_argument("--resolved-config-b64", default=None)
    ap.add_argument("--data-channel", required=True)
    ap.add_argument("--scorer-channel", default=None)
    ap.add_argument("--batch-size", type=int, default=0)
    args = ap.parse_args()

    from presentation_dependence.reader.config import load_reader_config
    from presentation_dependence.reader.run_io import (
        SCORER_PARQUET_FILENAME,
        decode_resolved_config,
        reader_scalar_metrics,
    )
    from presentation_dependence.reader.verdict_ids import semantic_verdict_run_id
    from presentation_dependence.utils.config import apply_execution_environment

    cfg = (
        decode_resolved_config(args.resolved_config_b64)
        if args.resolved_config_b64
        else load_reader_config(args.config_path)
    )
    apply_execution_environment(cfg)
    reader_cfg = cfg["reader"]
    phase = cfg["phase"]
    cot = bool(reader_cfg.get("cot", False))
    if phase not in ("R3", "V3"):
        raise SystemExit(f"unsupported reader phase {phase!r}; expected 'R3' or 'V3'")

    data_dir = _channel_dir(args.data_channel)
    fixture_path = data_dir / "fixture.jsonl"
    qrels_path = data_dir / "qrels.txt"
    answers_path = data_dir / "answers.jsonl"
    verdicts_path = data_dir / "verdicts.jsonl"
    verdict_qids_path = data_dir / "verdict_qids.txt"
    required = [(fixture_path, "fixture.jsonl"), (qrels_path, "qrels.txt")]
    if phase.startswith("V"):
        required.extend([(verdicts_path, "verdicts.jsonl"), (verdict_qids_path, "verdict_qids.txt")])
    else:
        required.append((answers_path, "answers.jsonl"))
    for path, label in required:
        if not path.exists():
            raise FileNotFoundError(f"missing {label} in data channel {data_dir}")

    out_dir = _output_dir()
    run_id = semantic_verdict_run_id(str(cfg["id"])) if phase.startswith("V") else str(cfg["id"])
    run_dir = out_dir / "runs" / run_id / _trial()
    engine = _build_engine(reader_cfg)

    if phase == "R3":
        from presentation_dependence.reader.pipeline import run_reader_bridge

        if not args.scorer_channel:
            raise SystemExit("R3 reader job requires --scorer-channel")
        score_log = _channel_dir(args.scorer_channel) / SCORER_PARQUET_FILENAME
        if not score_log.exists():
            raise FileNotFoundError(f"missing scorer parquet at {score_log}")
        result = run_reader_bridge(
            score_log=score_log,
            fixture_path=fixture_path,
            answers_path=answers_path,
            qrels_path=qrels_path,
            engine=engine,
            k_values=cfg.get("k_values", [3, 5]),
            cot=cot,
            canonical_perm=int(cfg.get("canonical_perm", 0)),
            batch_size=args.batch_size,
        )
        sub = run_dir / "reader"
        sub.mkdir(parents=True, exist_ok=True)
        with (sub / "per_item.jsonl").open("w", encoding="utf-8") as f:
            for it in result["items"]:
                f.write(
                    json.dumps(
                        {
                            "qid": it.qid,
                            "perm_idx": it.perm_idx,
                            "k": it.k,
                            "ranked_pids": it.ranked_pids,
                            "gold_slot": it.gold_slot,
                            "n_gold_in_topk": it.n_gold_in_topk,
                            "prediction": it.prediction,
                            "em": it.em,
                            "f1": it.f1,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        with (sub / "per_query.jsonl").open("w", encoding="utf-8") as f:
            for row in result["per_query"]:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        metrics_payload = {"meta": result["meta"], "corpus": result["corpus"]}
        (sub / "reader_metrics.json").write_text(json.dumps(metrics_payload, indent=2, sort_keys=True))
        (out_dir / "reader_metrics.json").write_text(json.dumps(metrics_payload, indent=2, sort_keys=True))
        _emit_metrics(reader_scalar_metrics("R3", result["corpus"]))

    elif phase == "V3":
        from presentation_dependence.reader.verdict_pipeline import run_verdict_bridge

        if not args.scorer_channel:
            raise SystemExit("V3 verdict job requires --scorer-channel")
        score_log = _channel_dir(args.scorer_channel) / SCORER_PARQUET_FILENAME
        if not score_log.exists():
            raise FileNotFoundError(f"missing scorer rankings at {score_log}")
        result = run_verdict_bridge(
            score_log=score_log,
            fixture_path=fixture_path,
            verdicts_path=verdicts_path,
            qrels_path=qrels_path,
            engine=engine,
            k_values=cfg.get("k_values", [5]),
            qids_path=verdict_qids_path,
            canonical_perm=int(cfg.get("canonical_perm", 0)),
            batch_size=args.batch_size,
        )
        result["meta"].update(
            {
                "dataset": cfg["dataset"],
                "scorer_exp_id": cfg["scorer"]["exp_id"],
                "scorer_recipe": cfg["scorer"].get("recipe"),
                "source_run_dir": cfg["scorer"].get("run_dir"),
            }
        )
        sub = run_dir / "verdict"
        sub.mkdir(parents=True, exist_ok=True)
        with (sub / "per_item.jsonl").open("w", encoding="utf-8") as f:
            for item in result["items"]:
                f.write(
                    json.dumps(
                        {
                            "qid": item.qid,
                            "perm_idx": item.perm_idx,
                            "k": item.k,
                            "ranked_pids": item.ranked_pids,
                            "evidence_slot": item.evidence_slot,
                            "n_evidence_in_topk": item.n_evidence_in_topk,
                            "gold_verdict": item.gold_verdict,
                            "prediction": item.prediction,
                            "raw_output": item.raw_output,
                            "accuracy": item.accuracy,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        with (sub / "per_query.jsonl").open("w", encoding="utf-8") as f:
            for row in result["per_query"]:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        payload = {"meta": result["meta"], "corpus": result["corpus"]}
        (sub / "verdict_metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
        (out_dir / "verdict_metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
        _emit_metrics(reader_scalar_metrics("V3", result["corpus"]))

    print(f"[reader-entrypoint] done phase={phase} id={cfg['id']} -> {run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
