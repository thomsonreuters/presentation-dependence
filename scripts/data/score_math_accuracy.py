#!/usr/bin/env python
r"""Verifiable exact-match accuracy for generated responses on PPE-MATH.

Extract each response's final boxed answer and check mathematical equivalence against the
ground-truth boxed answer in the evaluation pool's `gold.jsonl`. Reports
per-arm accuracy and the decisive deltas. The reward model is NOT involved, so this
is immune to judge bias / circularity in a way win-rate is not.

Equivalence backend: `math_verify` (the HF MATH grader) if installed; otherwise a
normalized-string fallback (conservative — may undercount equivalent forms, but
symmetric across arms, so the *delta* stays fair).

Usage:
    uv run python scripts/data/score_math_accuracy.py \\
      --gold data/grpo-eval-ppe-math/gold.jsonl \\
      --responses base=runs/.../base_responses.jsonl \\
      --responses ocrm_bG=runs/.../ocrm_bG_responses.jsonl \\
      --responses pointwise_b1=... --responses ocrm_b1_null=...
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def extract_last_boxed(text: str) -> str | None:
    r"""Content of the last balanced ``\boxed{...}`` in ``text``."""
    if not text:
        return None
    idx = text.rfind(r"\boxed")
    if idx < 0:
        return None
    i = idx + len(r"\boxed")
    while i < len(text) and text[i] != "{":
        if not text[i].isspace():
            return None
        i += 1
    if i >= len(text):
        return None
    depth, start = 0, i
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : j].strip()
    return None


def _normalize(s: str) -> str:
    """Loose normalization for the fallback string comparison."""
    if s is None:
        return ""
    s = s.strip()
    for a, b in (
        ("\\left", ""),
        ("\\right", ""),
        ("\\!", ""),
        ("\\,", ""),
        ("\\;", ""),
        ("\\ ", " "),
        ("$", ""),
        ("\\$", ""),
        (" ", ""),
        ("\\%", ""),
        ("%", ""),
        ("\\text{", ""),
        ("\\mathrm{", ""),
        ("^{\\circ}", ""),
        ("^\\circ", ""),
    ):
        s = s.replace(a, b)
    s = re.sub(r"\\dfrac", r"\\frac", s)
    s = re.sub(r"\\tfrac", r"\\frac", s)
    s = s.rstrip("}").strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s


def _extract_pred(response: str) -> str | None:
    """Best-effort final answer from a free-form response: boxed, else last $...$, else trailing number."""
    b = extract_last_boxed(response)
    if b is not None:
        return b
    # last inline-math span
    spans = re.findall(r"\$([^$]+)\$", response or "")
    if spans:
        return spans[-1].strip()
    # trailing number
    nums = re.findall(r"-?\d[\d,]*\.?\d*", response or "")
    return nums[-1].replace(",", "") if nums else None


def _make_equal():
    """Return an equality function using math_verify if available, else normalized string."""
    try:
        from math_verify import parse, verify  # type: ignore

        def eq(pred: str, gold: str) -> bool:
            try:
                return bool(verify(parse(gold), parse(pred)))
            except Exception:
                return _normalize(pred) == _normalize(gold)

        return eq, "math_verify"
    except Exception:

        def eq(pred: str, gold: str) -> bool:
            return _normalize(pred) == _normalize(gold)

        return eq, "normalized_string"


def _load_jsonl_map(path: Path, key: str, val: str) -> dict[str, str]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out[str(r[key])] = str(r.get(val, ""))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", type=Path, required=True, help="gold.jsonl ({qid,gold_answer}).")
    ap.add_argument(
        "--responses",
        action="append",
        required=True,
        help="LABEL=path/to/responses.jsonl ({qid,response}). Repeatable.",
    )
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    gold = _load_jsonl_map(args.gold, "qid", "gold_answer")
    eq, backend = _make_equal()

    arms = {}
    for spec in args.responses:
        if "=" not in spec:
            raise SystemExit(f"--responses must be LABEL=path, got {spec!r}")
        label, path = spec.split("=", 1)
        resp = _load_jsonl_map(Path(path), "qid", "response")
        qids = [q for q in gold if q in resp]
        n = len(qids)
        correct = 0
        n_no_pred = 0
        for q in qids:
            pred = _extract_pred(resp[q])
            if pred is None:
                n_no_pred += 1
                continue
            if eq(pred, gold[q]):
                correct += 1
        acc = correct / n if n else 0.0
        arms[label] = {"n": n, "correct": correct, "accuracy": round(acc, 4), "n_no_pred": n_no_pred}

    result = {"metric": "verifiable_exact_match_accuracy", "backend": backend, "gold_n": len(gold), "arms": arms}
    # decisive deltas
    if "ocrm_bG" in arms:
        for other in ("pointwise_b1", "ocrm_b1_null", "base"):
            if other in arms:
                result.setdefault("deltas", {})[f"ocrm_bG_minus_{other}"] = round(
                    arms["ocrm_bG"]["accuracy"] - arms[other]["accuracy"], 4
                )
    print(json.dumps(result, indent=2))
    if args.out:
        args.out.write_text(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
