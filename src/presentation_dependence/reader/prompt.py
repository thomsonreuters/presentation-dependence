"""Fixed reader prompt + answer extraction for the QA reader bridge.

The reader prompt and decoding settings are fixed across scorer conditions.
Only the scorer's ranked top-k passages change between conditions.

Passages are always rendered in the order they are passed in, which is the
scorer's output ranking. The default template requests only an answer. The CoT
template permits reasoning before a tagged final answer. The selected template
must remain fixed across conditions and be reported with the results. This
reader-side choice does not alter the scorer's single-pass protocol.
"""

from __future__ import annotations

import re
from typing import Mapping, Sequence

__all__ = [
    "ANSWER_TAG",
    "SYSTEM_PROMPT",
    "render_passages",
    "build_reader_messages",
    "extract_answer",
]

ANSWER_TAG = "Answer:"

SYSTEM_PROMPT = (
    "You are a precise question-answering assistant. Use only the provided "
    "passages to answer the question. The answer is a short span — a name, "
    "entity, date, or 'yes'/'no'. Do not explain."
)

SYSTEM_PROMPT_COT = (
    "You are a precise question-answering assistant. Use only the provided "
    "passages to answer the question. Reason step by step, then give the final "
    f"short answer on a new line prefixed with '{ANSWER_TAG}'."
)

_NO_COT_INSTRUCTION = (
    "Answer the question with the shortest exact span (a name, entity, date, or "
    "yes/no). Output only the answer, with no explanation or punctuation."
)

_COT_INSTRUCTION = (
    "Think briefly about which passages are relevant, then write the final short "
    f"answer on its own line prefixed with '{ANSWER_TAG}'."
)


def render_passages(passages: Sequence[Mapping[str, object]]) -> str:
    """Render passages as a numbered, fixed-order context block.

    Each passage is shown as ``[i] <text>`` in the given order (the scorer's
    ranked order). ``text`` already contains the title prefix in the QA fixtures.
    """
    lines: list[str] = []
    for i, passage in enumerate(passages, start=1):
        text = str(passage.get("text", "")).strip()
        lines.append(f"[{i}] {text}")
    return "\n\n".join(lines)


def build_reader_messages(
    question: str,
    passages: Sequence[Mapping[str, object]],
    *,
    cot: bool = False,
) -> list[dict[str, str]]:
    """Build the pinned chat messages for the reader given ranked passages."""
    instruction = _COT_INSTRUCTION if cot else _NO_COT_INSTRUCTION
    system = SYSTEM_PROMPT_COT if cot else SYSTEM_PROMPT
    user = f"{instruction}\n\nPassages:\n{render_passages(passages)}\n\nQuestion: {str(question).strip()}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


_ANSWER_TAG_RE = re.compile(rf"{re.escape(ANSWER_TAG)}\s*(.+)", flags=re.IGNORECASE | re.DOTALL)


def extract_answer(generated_text: str, *, cot: bool = False) -> str:
    """Extract the final short answer from the reader output.

    For CoT output, use the text after the final ``Answer:`` tag, or the final
    non-empty line when the tag is absent. For answer-only output, use the first
    non-empty line.
    """
    text = (generated_text or "").strip()
    if not text:
        return ""
    if cot:
        matches = list(_ANSWER_TAG_RE.finditer(text))
        if matches:
            return matches[-1].group(1).strip().splitlines()[0].strip()
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        return lines[-1] if lines else ""
    # no-CoT: strip a leading "Answer:" if the model added one anyway.
    first = text.splitlines()[0].strip()
    tag_match = _ANSWER_TAG_RE.match(first)
    if tag_match:
        return tag_match.group(1).strip()
    return first
