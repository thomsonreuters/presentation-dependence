"""Shared grade rubrics and expected-grade user-prompt body for open + closed models."""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"
DEFAULT_GRADE_RUBRIC_ID = "relevance_v1"

LEGAL_INSTRUCTION = "Given a legal search query, retrieve relevant legal passages that answer the query"


@dataclass(frozen=True)
class GradeRubric:
    system_prompt: str
    query_label: str
    task_sentence: str
    grade_lines: tuple[str, str, str, str]
    # Item-type wording. Defaults reproduce the original document/relevance
    # phrasing byte-for-byte; non-default values let the same expected-grade scoring
    # path describe other item types (e.g. responses) without a new code path.
    item_label: str = "Documents"
    item_noun: str = "document"
    grade_noun: str = "relevance"


_GRADE_RUBRICS: dict[str, GradeRubric] = {
    "relevance_v1": GradeRubric(
        system_prompt=(
            "You are a search relevance grader. For each numbered document, output one fixed integer "
            "relevance grade in {0, 1, 2, 3}. Do not rank the documents and do not explain."
        ),
        query_label="Query",
        task_sentence="Evaluate every document independently for relevance to the query.",
        grade_lines=(
            "- 3 = directly and completely answers the query",
            "- 2 = strongly relevant but incomplete",
            "- 1 = weakly relevant or topical background",
            "- 0 = not relevant",
        ),
    ),
    "answer_support_v1": GradeRubric(
        system_prompt=(
            "You are a passage evidence grader for retrieval. For each numbered passage, output one fixed "
            "integer evidence grade in {0, 1, 2, 3}. Do not rank passages and do not explain."
        ),
        query_label="Query",
        task_sentence=(
            "Evaluate every passage independently for how much factual evidence it provides for the query. "
            "Grade informational support, not topical similarity."
        ),
        grade_lines=(
            "- 3 = directly contains facts needed for the query",
            "- 2 = strong supporting evidence but incomplete alone",
            "- 1 = weak background or bridge context, insufficient alone",
            "- 0 = no useful evidence for the query",
        ),
    ),
    # Response-quality / reward-modeling rubric for the response-ranking cross-task arm.
    # Uses the same 0-3 expected-grade readout for candidate responses to a prompt.
    "response_quality_v1": GradeRubric(
        system_prompt=(
            "You are a response quality grader. For each numbered response, output one fixed "
            "integer quality grade in {0, 1, 2, 3}. Do not rank the responses and do not explain."
        ),
        query_label="Prompt",
        task_sentence=(
            "Evaluate every response independently for overall quality as an answer to the prompt: "
            "helpfulness, correctness, and instruction-following."
        ),
        grade_lines=(
            "- 3 = excellent: correct, helpful, and fully follows the prompt",
            "- 2 = good: mostly correct and helpful with minor issues",
            "- 1 = poor: partially helpful but with notable errors or omissions",
            "- 0 = unhelpful, incorrect, or off-task",
        ),
        item_label="Responses",
        item_noun="response",
        grade_noun="quality",
    ),
    "counterargument_relevance_v1": GradeRubric(
        system_prompt=(
            "You are a counterargument retrieval grader. For each numbered passage, "
            "output one fixed integer grade in {0, 1, 2, 3}. Do not rank passages "
            "and do not explain."
        ),
        query_label="Argument",
        task_sentence=(
            "Evaluate every passage independently for how directly and substantively "
            "it challenges or rebuts the given argument."
        ),
        grade_lines=(
            "- 3 = a direct, substantive counterargument addressing the central claim",
            "- 2 = a relevant counterargument that is incomplete or indirect",
            "- 1 = related argumentative context with weak rebuttal value",
            "- 0 = does not counter the argument",
        ),
        item_label="Candidate counterarguments",
        item_noun="passage",
        grade_noun="counterargument relevance",
    ),
    "argument_retrieval_v1": GradeRubric(
        system_prompt=(
            "You are an argument retrieval grader. For each numbered passage, output "
            "one fixed integer grade in {0, 1, 2, 3}. Do not rank passages and do not explain."
        ),
        query_label="Controversial question",
        task_sentence=(
            "Evaluate every passage independently for whether it presents a clear, "
            "relevant argument addressing the question."
        ),
        grade_lines=(
            "- 3 = a clear, directly relevant argument addressing the question",
            "- 2 = a relevant argument that is incomplete or weakly supported",
            "- 1 = related discussion with little explicit argumentative content",
            "- 0 = irrelevant or non-argumentative",
        ),
        item_label="Candidate arguments",
        item_noun="passage",
        grade_noun="argument relevance",
    ),
    # Faithful paraphrases of relevance_v1 for the input-symmetry rubric probe.
    # Human validation is recommended.
    "relevance_v1_p1": GradeRubric(
        system_prompt=(
            "You are a retrieval relevance judge. For every listed document, assign exactly one integer "
            "score from {0, 1, 2, 3}. Do not rank documents and do not add commentary."
        ),
        query_label="Query",
        task_sentence="Score each document on its relevance to the query, independently.",
        grade_lines=(
            "- 3 = fully relevant to the query",
            "- 2 = highly relevant but incomplete",
            "- 1 = tangentially related or background only",
            "- 0 = irrelevant",
        ),
    ),
    "relevance_v1_p2": GradeRubric(
        system_prompt=(
            "Act as a document relevance evaluator. Output a single integer grade in {0,1,2,3} per document. "
            "No ranking, no explanation."
        ),
        query_label="Search query",
        task_sentence="Judge how well each document matches the search query on its own.",
        grade_lines=(
            "- 3 = directly relevant and complete",
            "- 2 = strongly relevant but incomplete",
            "- 1 = weak topical overlap",
            "- 0 = not relevant to the query",
        ),
    ),
    "relevance_v1_p3": GradeRubric(
        system_prompt=(
            "Grade document relevance for search retrieval. Use integers 0, 1, 2, or 3 for each document. "
            "One grade per document; do not compare or rank."
        ),
        query_label="Query",
        task_sentence="For each document, decide how pertinent it is to the query.",
        grade_lines=(
            "- 3 = maximally pertinent and on-target for the query",
            "- 2 = pertinent but missing needed detail",
            "- 1 = loosely related context",
            "- 0 = no relevance",
        ),
    ),
    "relevance_v1_p4": GradeRubric(
        system_prompt=(
            "You rate passage relevance for retrieval. Emit one fixed grade per passage from "
            "{0, 1, 2, 3}. No explanations."
        ),
        query_label="Query",
        task_sentence="Assess each passage separately for relevance to the query.",
        grade_lines=(
            "- 3 = directly and fully relevant",
            "- 2 = mostly relevant, missing detail",
            "- 1 = peripheral topic overlap",
            "- 0 = unrelated",
        ),
    ),
}


def resolve_grade_rubric(grade_rubric_id: str | None) -> GradeRubric:
    rubric_id = (grade_rubric_id or DEFAULT_GRADE_RUBRIC_ID).strip()
    try:
        return _GRADE_RUBRICS[rubric_id]
    except KeyError as exc:
        known = ", ".join(sorted(_GRADE_RUBRICS))
        raise ValueError(f"Unknown grade_rubric_id={rubric_id!r}; known: {known}") from exc


def normalize_doc_text(text: str, *, max_doc_chars: int) -> str:
    normalized = " ".join(str(text).split())
    if max_doc_chars > 0:
        normalized = normalized[:max_doc_chars]
    return normalized


def _join_doc_blocks(blocks: list[str], separator: str | None) -> str:
    if separator is None or separator == "newline":
        return "\n".join(blocks)
    if separator == "dash":
        return "\n---\n".join(blocks)
    if separator == "hash":
        return "\n###\n".join(blocks)
    if separator == "xml":
        return "\n".join(f"<doc>{block}</doc>" for block in blocks)
    if separator == "passage_n":
        return "\n".join(blocks)
    raise ValueError(f"Unknown separator style: {separator!r}")


def build_doc_blocks(
    doc_texts: list[str],
    *,
    max_doc_chars: int,
    slot_labels: tuple[str, ...] | None = None,
    separator: str | None = None,
) -> list[str]:
    """Build per-document text blocks for the user prompt."""
    blocks: list[str] = []
    for slot_idx, text in enumerate(doc_texts, start=1):
        body = normalize_doc_text(text, max_doc_chars=max_doc_chars)
        if slot_labels is not None:
            marker = slot_labels[slot_idx - 1]
            if separator == "passage_n":
                label_index = slot_idx
                blocks.append(f"Passage {label_index}: {body}")
            else:
                blocks.append(f"{marker} {body}")
        elif separator == "passage_n":
            blocks.append(f"Passage {slot_idx}: {body}")
        else:
            blocks.append(f"[{slot_idx}] {body}")
    return blocks


def build_pathc_grade_user_body(
    instruction: str | None,
    query: str,
    doc_texts: list[str],
    *,
    max_doc_chars: int,
    grade_rubric_id: str | None = None,
    slot_labels: tuple[str, ...] | None = None,
    separator: str | None = None,
) -> str:
    """User-section body: instruction + query + numbered docs + grade rubric.

    When ``slot_labels`` and ``separator`` are both ``None``, output is
    byte-for-byte identical to the pre–input-symmetry implementation.
    """
    inst = instruction or DEFAULT_INSTRUCTION
    rubric = resolve_grade_rubric(grade_rubric_id)
    doc_section = _join_doc_blocks(
        build_doc_blocks(
            doc_texts,
            max_doc_chars=max_doc_chars,
            slot_labels=slot_labels,
            separator=separator,
        ),
        separator,
    )

    return (
        f"Instruction: {inst}\n"
        f"{rubric.query_label}: {query}\n\n"
        f"{rubric.item_label}:\n" + doc_section + "\n\n"
        f"{rubric.task_sentence}\n"
        f"Assign exactly one integer {rubric.grade_noun} grade to every {rubric.item_noun}:\n"
        + "\n".join(rubric.grade_lines)
        + "\n"
        f"Output one line per {rubric.item_noun} in the form: [<id>] Grade: <0|1|2|3>"
    )
