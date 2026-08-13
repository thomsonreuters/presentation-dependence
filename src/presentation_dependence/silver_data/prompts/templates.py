"""Pointwise prompt templates and parsers for silver labels.

The teacher protocol is pointwise B=1: one query/document pair per model
judgement. These templates define the audit prompt and the parser used to
turn teacher output into normalized labels.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class PromptParseResult:
    score_parsed: float | None
    score_normalized: float | None


@dataclass(frozen=True)
class PromptTemplate:
    template_id: str
    text: str
    parser: str

    def render(self, *, query: str, document: str) -> str:
        return self.text.format(query=query, document=document)

    def parse(self, response: str) -> PromptParseResult:
        if self.parser == "grade_int_0_3":
            return _parse_grade_int(response)
        if self.parser == "float_0_1":
            return _parse_float(response)
        raise ValueError(f"Unknown prompt parser: {self.parser}")


GRADE_INT_V1 = PromptTemplate(
    template_id="grade_int_v1",
    parser="grade_int_0_3",
    text="""Assign one of the labels below to the passage based on its relevance to the query. Output your answer as only a single integer label.

<Label Instructions>
3 - The passage is highly relevant and directly on-point. The passage precisely responds to the query and not a similar but different topic.

2 - The passage is partially relevant, as it may be on topic but does not address the entirety of the query.

1 - The passage is completely non-relevant. The passage might be on the same or similar topic as the query, but the passage does not answer the question and does not satisfy the information need.

0 - The passage is off-topic and totally unrelated to the information need.
</Label Instructions>

Query: {query}

Passage: {document}

Output only one integer label: 0, 1, 2, or 3.""",
)

PATHC_GRADE_V1 = PromptTemplate(
    template_id="pathc_grade_v1",
    parser="grade_int_0_3",
    text="",
)

GRADE_FLOAT_V1 = PromptTemplate(
    template_id="grade_float_v1",
    parser="float_0_1",
    text="""You are a relevance grader. Given a query and a document, output a single relevance score.

Rate the relevance from 0.00 to 1.00, where 0.00 is completely irrelevant and 1.00 is perfectly relevant. Use three decimal places.

Query: {query}

Document: {document}

Output only the decimal number. No explanation.""",
)

_TEMPLATES = {
    GRADE_INT_V1.template_id: GRADE_INT_V1,
    PATHC_GRADE_V1.template_id: PATHC_GRADE_V1,
    GRADE_FLOAT_V1.template_id: GRADE_FLOAT_V1,
}


def get_prompt_template(template_id: str) -> PromptTemplate:
    try:
        return _TEMPLATES[template_id]
    except KeyError:
        known = ", ".join(sorted(_TEMPLATES))
        raise ValueError(f"Unknown prompt_template_id {template_id!r}. Known: {known}") from None


def _parse_grade_int(response: str) -> PromptParseResult:
    match = re.search(r"(?<![\d.])([0-2](?:\.\d+)?|3(?:\.0+)?)(?![\d.])", response.strip())
    if match is None:
        return PromptParseResult(score_parsed=None, score_normalized=None)
    grade = float(match.group(1))
    return PromptParseResult(score_parsed=grade, score_normalized=grade / 3.0)


def _parse_float(response: str) -> PromptParseResult:
    match = re.search(r"(?<![\d.])(?:0(?:\.\d+)?|1(?:\.0+)?)(?![\d.])", response.strip())
    if match is None:
        return PromptParseResult(score_parsed=None, score_normalized=None)
    score = float(match.group(0))
    if score < 0.0 or score > 1.0:
        return PromptParseResult(score_parsed=None, score_normalized=None)
    return PromptParseResult(score_parsed=score, score_normalized=score)
