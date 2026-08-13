#!/usr/bin/env python
"""Evaluate stored QA scorer rankings with a frozen answer or verdict reader.

Reads an existing QA scorer run's per-permutation score log
(``<run>/psi/beta_gamma_scores.parquet``). For each ``(query, scorer-input
permutation)``, it reconstructs the scorer's ranked order, passes the top-k
passages in that order to a frozen reader, and scores answer EM/F1 and stability
across the scorer-input permutations.

Offline plumbing (no GPU):

    uv run python scripts/run_reader.py \
        --scorer-run runs/<semantic-scorer-id>/<trial> \
        --dataset hotpotqa --reader-engine echo --out-id smoke-reader

Real reader (vLLM on a supported GPU host):

    uv run --extra vllm python scripts/run_reader.py \
        --scorer-run runs/<semantic-scorer-id>/<trial> \
        --dataset hotpotqa --reader-engine vllm \
        --model-name ibm-granite/granite-4.1-8b \
        --k 3,5 --out-id reader-granite8b-hotpotqa-ocsft

V3 verdict configs use the same command:

    uv run --extra vllm python scripts/run_reader.py \
        --config build/reproduction/multi-document-qa/downstream-eval/configs/<verdict-config>.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

SLM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SLM_ROOT / "src"))

from presentation_dependence.reader.engine import EchoReaderEngine, ReaderEngine  # noqa: E402
from presentation_dependence.reader.pipeline import run_reader_bridge  # noqa: E402
from presentation_dependence.utils.config import apply_execution_environment  # noqa: E402
from presentation_dependence.utils.run_paths import default_runs_root  # noqa: E402


def _default_runs_root() -> Path:
    """Anchor a relative runs root on the repo, so cwd does not move it."""
    root = default_runs_root()
    return root if root.is_absolute() else SLM_ROOT / root


QA_DATASET_DIRS = {
    "hotpotqa": "data/hotpotqa-distractor-support-dev",
    "2wiki": "data/2wiki-distractor-support-dev",
    "musique": "data/musique-support-dev",
}
VERDICT_DATASET_DIRS = {
    "climate-fever": "data/beir-v1.0.0-climate-fever-test",
}
DATASET_DIRS = {**QA_DATASET_DIRS, **VERDICT_DATASET_DIRS}


def resolve_score_log(scorer_run: str, runs_root: Path) -> Path:
    """Resolve a scorer run dir / exp id / parquet path to the score log.

    Accepts: an explicit parquet file; a single-run dir (``.../<ts>/``); an
    experiment dir holding timestamp subdirs (``runs/<exp-id>/``); or a bare
    experiment id resolved under ``runs_root``. Picks the latest matching
    ``psi/beta_gamma_scores.parquet`` when several exist.
    """
    p = Path(scorer_run)
    if p.is_file():
        return p

    search_dirs: list[Path] = []
    if p.is_dir():
        search_dirs.append(p)
    # Scorer runs live under the run root; also try the default root explicitly
    # so a redirected --runs-root (e.g. a smoke output dir) still finds the
    # canonical scorer parquet.
    for root in {Path(runs_root), _default_runs_root()}:
        candidate_exp = root / str(scorer_run)
        if candidate_exp.is_dir():
            search_dirs.append(candidate_exp)

    matches: list[Path] = []
    for base in search_dirs:
        direct = base / "psi" / "beta_gamma_scores.parquet"
        if direct.exists():
            matches.append(direct)
        matches.extend(base.glob("**/psi/beta_gamma_scores.parquet"))
    if matches:
        return max(set(matches), key=lambda m: m.stat().st_mtime)
    raise FileNotFoundError(f"no psi/beta_gamma_scores.parquet found for scorer-run {scorer_run!r}")


def resolve_dataset_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    if args.dataset:
        base = SLM_ROOT / DATASET_DIRS[args.dataset]
        fixture = args.fixture or (base / "fixture.jsonl")
        answers = args.answers or (base / "answers.jsonl")
        qrels = args.qrels or (base / "qrels.txt")
    else:
        if not (args.fixture and args.answers and args.qrels):
            raise SystemExit("Provide --dataset or all of --fixture/--answers/--qrels")
        fixture, answers, qrels = Path(args.fixture), Path(args.answers), Path(args.qrels)
    for path, label in [(fixture, "fixture"), (answers, "answers (run setup_qa_answers.py)"), (qrels, "qrels")]:
        if not Path(path).exists():
            raise FileNotFoundError(f"missing {label}: {path}")
    return Path(fixture), Path(answers), Path(qrels)


def resolve_verdict_dataset_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    """Resolve fixture, verdict labels, qrels, and paired qids for V3."""
    if args.dataset:
        if args.dataset not in VERDICT_DATASET_DIRS:
            raise SystemExit(f"V3 does not support dataset {args.dataset!r}")
        base = SLM_ROOT / VERDICT_DATASET_DIRS[args.dataset]
        fixture = args.fixture or (base / "fixture.jsonl")
        verdicts = args.verdicts or (base / "verdicts.jsonl")
        qrels = args.qrels or (base / "qrels.txt")
        verdict_qids = args.verdict_qids or (base / getattr(args, "verdict_qids_filename", "verdict_qids.txt"))
    else:
        if not (args.fixture and args.verdicts and args.qrels and args.verdict_qids):
            raise SystemExit("V3 requires --dataset or all of --fixture/--verdicts/--qrels/--verdict-qids")
        fixture = args.fixture
        verdicts = args.verdicts
        qrels = args.qrels
        verdict_qids = args.verdict_qids
    required = (
        (fixture, "fixture"),
        (verdicts, "verdicts"),
        (qrels, "qrels"),
        (verdict_qids, "verdict qids"),
    )
    for path, label in required:
        if not Path(path).exists():
            raise FileNotFoundError(f"missing {label}: {path}")
    return Path(fixture), Path(verdicts), Path(qrels), Path(verdict_qids)


def build_engine(args: argparse.Namespace) -> ReaderEngine:
    if args.reader_engine == "echo":
        return EchoReaderEngine(name="echo-reader")
    if args.reader_engine == "vllm":
        if not args.model_name:
            raise SystemExit("--model-name is required for --reader-engine vllm")
        from presentation_dependence.reader.engine import VLLMReaderEngine

        thinking = None if args.enable_thinking == "auto" else (args.enable_thinking == "true")
        return VLLMReaderEngine(
            model_name=args.model_name,
            max_model_len=args.max_model_len,
            max_tokens=args.max_tokens,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enable_thinking=thinking,
            choices=args.choices,
        )
    raise SystemExit(f"unknown reader engine {args.reader_engine!r}")


def apply_config(args: argparse.Namespace) -> argparse.Namespace:  # noqa: C901
    """Merge a reader-bridge config YAML into args; explicit CLI flags win."""
    if not args.config:
        if not args.scorer_run:
            raise SystemExit("Provide --scorer-run or --config")
        return args
    from presentation_dependence.reader.config import load_reader_config

    cfg = load_reader_config(args.config)
    apply_execution_environment(cfg)
    reader = cfg.get("reader", {})
    scorer = cfg.get("scorer", {})
    args.phase = str(cfg["phase"])
    args.verdict_qids_filename = str(cfg.get("verdict_qids_filename") or "verdict_qids.txt")
    if args.scorer_run is None:
        args.scorer_run = scorer.get("exp_id")
    if args.dataset is None:
        args.dataset = cfg.get("dataset")
    if args.reader_engine is None:
        args.reader_engine = reader.get("engine", "echo")
    if args.model_name is None:
        args.model_name = reader.get("model_name")
    if args.k is None and cfg.get("k_values"):
        args.k = ",".join(str(k) for k in cfg["k_values"])
    if not args.cot:
        args.cot = bool(reader.get("cot", False))
    if args.max_tokens is None:
        args.max_tokens = int(reader.get("max_tokens", 64))
    if args.max_model_len is None:
        args.max_model_len = int(reader.get("max_model_len", 8192))
    if args.batch_size is None:
        args.batch_size = int(reader.get("batch_size", 0))
    if args.choices is None and reader.get("choices"):
        args.choices = [str(choice) for choice in reader["choices"]]
    if args.enable_thinking is None:
        et = reader.get("enable_thinking", False)
        args.enable_thinking = "auto" if et is None else ("true" if et else "false")
    if args.canonical_perm is None:
        args.canonical_perm = int(cfg.get("canonical_perm", 0))
    if args.out_id is None:
        args.out_id = cfg.get("id")
    if not args.scorer_run:
        raise SystemExit("config has no scorer.exp_id and --scorer-run not given")
    return args


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=None, help="Reader-bridge config YAML (configs/reader/<id>.yaml).")
    ap.add_argument("--phase", choices=["R3", "V3"], default="R3")
    ap.add_argument("--scorer-run", default=None, help="Scorer run dir, exp id, or parquet path.")
    ap.add_argument("--dataset", choices=sorted(DATASET_DIRS), default=None)
    ap.add_argument("--fixture", type=Path, default=None)
    ap.add_argument("--answers", type=Path, default=None)
    ap.add_argument("--verdicts", type=Path, default=None)
    ap.add_argument("--verdict-qids", type=Path, default=None)
    ap.add_argument("--qrels", type=Path, default=None)
    ap.add_argument("--reader-engine", choices=["echo", "vllm"], default=None)
    ap.add_argument("--model-name", default=None, help="HF id for the frozen reader (vllm engine).")
    ap.add_argument("--k", default=None, help="Comma-separated top-k values to report (default 3,5).")
    ap.add_argument("--cot", action="store_true", help="Use the CoT reader template (pinned across conditions).")
    ap.add_argument("--canonical-perm", type=int, default=None, help="perm_idx treated as the canonical scorer order.")
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--max-model-len", type=int, default=None)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    ap.add_argument("--enable-thinking", choices=["auto", "true", "false"], default=None)
    ap.add_argument("--batch-size", type=int, default=None, help="0 = single generate call.")
    ap.add_argument("--choices", default=None, help="Comma-separated constrained V3 verdict labels.")
    ap.add_argument("--out-id", default=None, help="Reader run id; output goes to runs/<out-id>/<ts>/reader/.")
    ap.add_argument("--runs-root", type=Path, default=_default_runs_root())
    return ap.parse_args()


def _output_dir(args: argparse.Namespace, engine: ReaderEngine, subdir: str) -> Path:
    out_id = args.out_id or f"reader-{engine.name.replace('/', '_')}"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = Path(args.runs_root) / out_id / ts / subdir
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_answer_output(
    args: argparse.Namespace,
    engine: ReaderEngine,
    result: dict,
    *,
    score_log: Path,
    fixture: Path,
    answers: Path,
    qrels: Path,
) -> tuple[Path, dict]:
    """Write the local R3 answer-reader artifact layout."""
    out_dir = _output_dir(args, engine, "reader")
    meta = {
        **result["meta"],
        "dataset": args.dataset,
        "score_log": str(score_log),
        "fixture": str(fixture),
        "answers": str(answers),
        "qrels": str(qrels),
    }
    with (out_dir / "per_item.jsonl").open("w", encoding="utf-8") as handle:
        for item in result["items"]:
            handle.write(
                json.dumps(
                    {
                        "qid": item.qid,
                        "perm_idx": item.perm_idx,
                        "k": item.k,
                        "ranked_pids": item.ranked_pids,
                        "gold_slot": item.gold_slot,
                        "n_gold_in_topk": item.n_gold_in_topk,
                        "prediction": item.prediction,
                        "em": item.em,
                        "f1": item.f1,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    with (out_dir / "per_query.jsonl").open("w", encoding="utf-8") as handle:
        for row in result["per_query"]:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out_dir / "reader_metrics.json").write_text(
        json.dumps({"meta": meta, "corpus": result["corpus"]}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return out_dir, meta


def _write_verdict_output(
    args: argparse.Namespace,
    engine: ReaderEngine,
    result: dict,
    *,
    score_log: Path,
    fixture: Path,
    verdicts: Path,
    verdict_qids: Path,
    qrels: Path,
) -> tuple[Path, dict]:
    """Write the local V3 verdict-reader artifact layout."""
    out_dir = _output_dir(args, engine, "verdict")
    meta = {
        **result["meta"],
        "dataset": args.dataset,
        "score_log": str(score_log),
        "fixture": str(fixture),
        "verdicts": str(verdicts),
        "verdict_qids": str(verdict_qids),
        "qrels": str(qrels),
    }
    with (out_dir / "per_item.jsonl").open("w", encoding="utf-8") as handle:
        for item in result["items"]:
            handle.write(
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
    with (out_dir / "per_query.jsonl").open("w", encoding="utf-8") as handle:
        for row in result["per_query"]:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out_dir / "verdict_metrics.json").write_text(
        json.dumps({"meta": meta, "corpus": result["corpus"]}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return out_dir, meta


def main() -> int:
    args = parse_args()
    try:
        args = apply_config(args)
        # Fill hard defaults for anything neither CLI nor config set.
        args.reader_engine = args.reader_engine or "echo"
        args.k = args.k or "3,5"
        args.max_tokens = 64 if args.max_tokens is None else args.max_tokens
        args.max_model_len = 8192 if args.max_model_len is None else args.max_model_len
        args.enable_thinking = args.enable_thinking or "false"
        args.canonical_perm = 0 if args.canonical_perm is None else args.canonical_perm
        args.batch_size = 0 if args.batch_size is None else args.batch_size
        if isinstance(args.choices, str):
            args.choices = [choice.strip() for choice in args.choices.split(",") if choice.strip()]
        score_log = resolve_score_log(args.scorer_run, args.runs_root)
        k_values = [int(tok) for tok in str(args.k).split(",") if tok.strip()]
        engine = build_engine(args)
        if args.phase == "R3":
            fixture, answers, qrels = resolve_dataset_paths(args)
            result = run_reader_bridge(
                score_log=score_log,
                fixture_path=fixture,
                answers_path=answers,
                qrels_path=qrels,
                engine=engine,
                k_values=k_values,
                cot=bool(args.cot),
                canonical_perm=int(args.canonical_perm),
                batch_size=int(args.batch_size),
            )
        else:
            from presentation_dependence.reader.verdict_pipeline import run_verdict_bridge

            fixture, verdicts, qrels, verdict_qids = resolve_verdict_dataset_paths(args)
            result = run_verdict_bridge(
                score_log=score_log,
                fixture_path=fixture,
                verdicts_path=verdicts,
                qrels_path=qrels,
                engine=engine,
                k_values=k_values,
                qids_path=verdict_qids,
                canonical_perm=int(args.canonical_perm),
                batch_size=int(args.batch_size),
            )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3

    if args.phase == "R3":
        out_dir, meta = _write_answer_output(
            args,
            engine,
            result,
            score_log=score_log,
            fixture=fixture,
            answers=answers,
            qrels=qrels,
        )
    else:
        out_dir, meta = _write_verdict_output(
            args,
            engine,
            result,
            score_log=score_log,
            fixture=fixture,
            verdicts=verdicts,
            verdict_qids=verdict_qids,
            qrels=qrels,
        )

    print(json.dumps({"meta": meta, "corpus": result["corpus"], "out_dir": str(out_dir)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
