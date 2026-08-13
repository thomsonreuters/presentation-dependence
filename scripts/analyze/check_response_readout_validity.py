#!/usr/bin/env python
"""Stage 0 readout-validity guard for the response-ranking arm.

On a sample, the expected-grade readout argmax (the highest-scored response per
prompt) should agree with the model's free-form preference / the gold-preferred
response. Agreement confirms the structured graded readout reflects judgment,
not a prompt-format artifact.

Post-hoc check over an existing run directory (the
``per_query_results/<qid>/detailed_results.json`` written by ExperimentManager);
needs no GPU and is offline-unit-testable.

Two reference sources:

    --reference gold        (default) compare argmax vs the gold-preferred
                            response(s) from a graded qrels file (the
                            UltraFeedback overall_score grades). Agreement if
                            the argmax response is among the max-grade set.

    --reference file        compare argmax vs an external preference JSONL
                            (one line per prompt: {"qid": ..., "preferred_pid": ...}),
                            e.g. produced by a separate free-form "which response
                            is best?" generation pass.

Exit code is 0 when the sampled agreement >= --min-agreement, else 1 (so it can
gate Stage 0). Usage:

    uv run python scripts/analyze/check_response_readout_validity.py \
        --run-dir runs/RR-ultrafeedback-qwen3-1p7b-base-vllm-psi/<ts> \
        --qrels data/ultrafeedback-response-quality-train/qrels.txt \
        --sample 50 --min-agreement 0.6
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

REPO_SLM_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_SLM_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_SLM_ROOT / "src"))

try:
    from presentation_dependence.utils.trec import dirname_to_qid
except Exception:  # pragma: no cover - fallback if import path differs

    def dirname_to_qid(name: str) -> str:
        return name


def load_predicted_best(run_dir: Path) -> dict[str, str]:
    """Map qid -> argmax response pid from per-query detailed_results.json."""
    results_dir = run_dir / "per_query_results"
    if not results_dir.is_dir():
        raise FileNotFoundError(f"no per_query_results/ under {run_dir}")
    predicted: dict[str, str] = {}
    for qid_dir in sorted(results_dir.iterdir()):
        detailed = qid_dir / "detailed_results.json"
        if not detailed.is_file():
            continue
        payload = json.loads(detailed.read_text(encoding="utf-8"))
        top = payload.get("top_k_psgs") or []
        if not top:
            continue
        qid = dirname_to_qid(qid_dir.name)
        predicted[qid] = str(top[0]["pid"])
    if not predicted:
        raise ValueError(f"no scored queries found under {results_dir}")
    return predicted


def load_gold_best_from_qrels(qrels_path: Path) -> dict[str, set[str]]:
    """Map qid -> set of pids tied at the maximum grade."""
    grades: dict[str, dict[str, int]] = {}
    for line in qrels_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 4:
            continue
        qid, _q0, pid, rel = parts
        grades.setdefault(qid, {})[pid] = int(rel)
    best: dict[str, set[str]] = {}
    for qid, by_pid in grades.items():
        top = max(by_pid.values())
        best[qid] = {pid for pid, rel in by_pid.items() if rel == top}
    return best


def load_preferences(path: Path) -> dict[str, set[str]]:
    """Map qid -> {preferred_pid} from an external preference JSONL."""
    prefs: dict[str, set[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        qid = str(rec["qid"])
        pid = rec.get("preferred_pid") or rec.get("chosen") or rec.get("pid")
        if pid is None:
            raise ValueError(f"preference row for qid={qid} has no preferred_pid/chosen/pid")
        prefs[qid] = {str(pid)}
    return prefs


def compute_agreement(
    predicted: dict[str, str],
    reference: dict[str, set[str]],
    *,
    sample: int | None,
    seed: int,
) -> dict[str, Any]:
    shared = sorted(set(predicted) & set(reference))
    if not shared:
        raise ValueError("no overlapping qids between run dir and reference")
    if sample is not None and 0 < sample < len(shared):
        shared = random.Random(seed).sample(shared, sample)
    agree = 0
    disagreements: list[dict[str, Any]] = []
    for qid in shared:
        pred = predicted[qid]
        ref = reference[qid]
        if pred in ref:
            agree += 1
        else:
            disagreements.append({"qid": qid, "predicted": pred, "reference": sorted(ref)})
    n = len(shared)
    return {
        "n_compared": n,
        "n_agree": agree,
        "agreement": agree / n if n else 0.0,
        "n_disagree": len(disagreements),
        "disagreements_sample": disagreements[:20],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True, type=Path, help="runs/<ID>/<timestamp> directory to inspect.")
    parser.add_argument("--reference", choices=["gold", "file"], default="gold", help="Reference preference source.")
    parser.add_argument("--qrels", type=Path, default=None, help="Graded qrels file (for --reference gold).")
    parser.add_argument("--preferences", type=Path, default=None, help="Preference JSONL (for --reference file).")
    parser.add_argument("--sample", type=int, default=None, help="Sample this many overlapping qids (default: all).")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    parser.add_argument("--min-agreement", type=float, default=0.0, help="Gate: exit 1 if agreement < this value.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        predicted = load_predicted_best(args.run_dir)
        if args.reference == "gold":
            if args.qrels is None:
                raise ValueError("--reference gold requires --qrels")
            reference = load_gold_best_from_qrels(args.qrels)
        else:
            if args.preferences is None:
                raise ValueError("--reference file requires --preferences")
            reference = load_preferences(args.preferences)
        summary = compute_agreement(predicted, reference, sample=args.sample, seed=int(args.seed))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    summary["reference"] = args.reference
    summary["min_agreement"] = float(args.min_agreement)
    summary["passed"] = summary["agreement"] >= float(args.min_agreement)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
