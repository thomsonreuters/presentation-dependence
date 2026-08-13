"""Shared fixed-grade prompt helpers for generic chat SLM rerankers."""

from __future__ import annotations

from typing import Any

from presentation_dependence.rerankers.base import Passage
from presentation_dependence.rerankers.grade_rubrics import build_pathc_grade_user_body


DEFAULT_GRADE_RUBRIC = (
    "Assign exactly one integer relevance grade to every document:\n"
    "- 3 = directly and completely answers the query\n"
    "- 2 = strongly relevant but incomplete\n"
    "- 1 = weakly relevant or topical background\n"
    "- 0 = not relevant\n"
)


def apply_chat_template_or_flatten(
    tokenizer: Any,
    messages: list[dict[str, str]],
) -> str:
    """Apply a tokenizer chat template, with a deterministic text fallback."""
    apply_fn = getattr(tokenizer, "apply_chat_template", None)
    if apply_fn is not None:
        try:
            return str(apply_fn(messages, tokenize=False, add_generation_prompt=True))
        except (AttributeError, TypeError, ValueError):
            pass
    parts = [msg["content"] for msg in messages if msg.get("content")]
    return "\n\n".join(parts) + "\n"


def build_grade_skeleton_user_body(
    *,
    instruction: str | None,
    default_instruction: str,
    query: str,
    passages: list[Passage],
    max_doc_chars: int,
    compare_documents: bool = False,
    extra_instructions: str = "",
    grade_rubric_id: str | None = None,
) -> str:
    """Build the common query + B-doc grade-rubric user body.

    ``None`` and ``"relevance_v1"`` both produce the passage-relevance body
    written inline below, which is *not* what
    :mod:`presentation_dependence.rerankers.grade_rubrics` builds for the same rubric id:
    this one uses ``<Instruct>:``/``<Query>:``/``<Documents>:`` where the shared
    builder uses ``Instruction:``/``Query:``/``Documents:``. Every recorded
    passage-reranking result was produced with the text below, so changing a
    byte of it makes new numbers incomparable with the published ones.

    Any other rubric id, such as ``"response_quality_v1"`` for the
    response-ranking arm, routes through the shared builder instead.
    """
    if grade_rubric_id is not None and grade_rubric_id != "relevance_v1":
        return build_pathc_grade_user_body(
            instruction,
            query,
            [str(p["text"]) for p in passages],
            max_doc_chars=max_doc_chars,
            grade_rubric_id=grade_rubric_id,
        )
    inst = instruction or default_instruction
    doc_lines: list[str] = []
    for i, passage in enumerate(passages, start=1):
        text = " ".join(str(passage["text"]).split())
        if max_doc_chars > 0:
            text = text[:max_doc_chars]
        doc_lines.append(f"[{i}] {text}")

    comparison = (
        "Evaluate every document independently for relevance to the query, while comparing documents in the batch.\n"
        if compare_documents
        else "Evaluate every document independently for relevance to the query.\n"
    )
    suffix = extra_instructions
    if suffix and not suffix.endswith("\n"):
        suffix += "\n"
    return (
        f"<Instruct>: {inst}\n"
        f"<Query>: {query}\n\n"
        "<Documents>:\n"
        + "\n".join(doc_lines)
        + "\n\n"
        + comparison
        + DEFAULT_GRADE_RUBRIC
        + "Output one line per document in the form: "
        "[<id>] Grade: <0|1|2|3>\n" + suffix
    )


def build_chat_grade_skeleton_prefixes(
    *,
    tokenizer: Any,
    system_prompt: str,
    instruction: str | None,
    default_instruction: str,
    query: str,
    passages: list[Passage],
    max_doc_chars: int,
    compare_documents: bool = False,
    extra_instructions: str = "",
    dummy_grade: str = "0",
    grade_rubric_id: str | None = None,
) -> tuple[str, list[str]]:
    """Render a chat grade skeleton and return prefixes before each dummy grade."""
    user_body = build_grade_skeleton_user_body(
        instruction=instruction,
        default_instruction=default_instruction,
        query=query,
        passages=passages,
        max_doc_chars=max_doc_chars,
        compare_documents=compare_documents,
        extra_instructions=extra_instructions,
        grade_rubric_id=grade_rubric_id,
    )
    chat_prefix = apply_chat_template_or_flatten(
        tokenizer,
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_body},
        ],
    )

    prompt = chat_prefix + "Grades:\n"
    prefixes: list[str] = []
    if not dummy_grade:
        raise ValueError("dummy_grade must not be empty")
    for i in range(1, len(passages) + 1):
        slot_prefix = f"[{i}] Grade: "
        prompt += slot_prefix + dummy_grade + "\n"
        prefixes.append(prompt[: -(len(dummy_grade) + 1)])
    return prompt, prefixes
