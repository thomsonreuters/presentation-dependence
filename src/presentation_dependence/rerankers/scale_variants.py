"""Scoring-scale input-symmetry variants for batched-pointwise expected-grade scoring.

Holds document order and slot markers fixed; varies the nominal grading scale
(0–3, 0–10, 1–5, 0–1, descriptive labels) and the corresponding logit readout.
Each variant normalizes expected grades to the standard reranker ``[0, 1]`` contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from presentation_dependence.rerankers.grade_rubrics import (
    DEFAULT_INSTRUCTION,
    DEFAULT_GRADE_RUBRIC_ID,
    build_doc_blocks,
)
from presentation_dependence.rerankers.render_variants import RenderVariant, slot_labels_for_chunk

ScaleScheme = Literal["grade_0_3", "grade_0_10", "grade_0_100", "grade_1_5", "grade_0_1", "grade_descriptive"]

SCALE_PERTURBATION = "scale"

# K=5 Stage-0 gate set (index 0 = training-seen 0–3).
SCALE_SCHEMES: tuple[ScaleScheme, ...] = (
    "grade_0_3",
    "grade_0_10",
    "grade_1_5",
    "grade_0_1",
    "grade_descriptive",
)


@dataclass(frozen=True)
class GradeScaleSpec:
    """Prompt + readout metadata for one scoring-scale variant."""

    scheme: ScaleScheme
    label: str
    system_prompt: str
    grade_lines: tuple[str, ...]
    output_format_line: str
    grade_strings: tuple[str, ...]
    grade_values: tuple[float, ...]
    score_offset: float  # affine map to [0,1]: (E[g] - offset) / divisor
    score_divisor: float
    skeleton_dummy: str
    is_canonical: bool
    seen: bool

    def normalize_expected_grade(self, expected: float) -> float:
        if self.score_divisor <= 0:
            raise ValueError(f"score_divisor must be positive for {self.scheme!r}")
        return (float(expected) - self.score_offset) / self.score_divisor


def _spec_grade_0_3() -> GradeScaleSpec:
    return GradeScaleSpec(
        scheme="grade_0_3",
        label="canonical",
        system_prompt=(
            "You are a search relevance grader. For each numbered document, output one fixed integer "
            "relevance grade in {0, 1, 2, 3}. Do not rank the documents and do not explain."
        ),
        grade_lines=(
            "- 3 = directly and completely answers the query",
            "- 2 = strongly relevant but incomplete",
            "- 1 = weakly relevant or topical background",
            "- 0 = not relevant",
        ),
        output_format_line="Output one line per document in the form: [<id>] Grade: <0|1|2|3>",
        grade_strings=("0", "1", "2", "3"),
        grade_values=(0.0, 1.0, 2.0, 3.0),
        score_offset=0.0,
        score_divisor=3.0,
        skeleton_dummy="0",
        is_canonical=True,
        seen=True,
    )


def _spec_grade_0_10() -> GradeScaleSpec:
    # Tokenizer: 0–9 are single tokens; "10" is two tokens. Use 0–9 with linear map to [0, 10].
    values = tuple(i * (10.0 / 9.0) for i in range(10))
    return GradeScaleSpec(
        scheme="grade_0_10",
        label="scale_0_10",
        system_prompt=(
            "You are a search relevance grader. For each numbered document, output one integer "
            "relevance score from 0 to 9 (0 = not relevant, 9 = maximally relevant). "
            "Do not rank documents and do not explain."
        ),
        grade_lines=(
            "- 9 = directly and completely answers the query",
            "- 6–8 = strongly relevant",
            "- 3–5 = weakly relevant or background",
            "- 0–2 = not relevant",
        ),
        output_format_line="Output one line per document in the form: [<id>] Grade: <0-9>",
        grade_strings=tuple(str(i) for i in range(10)),
        grade_values=values,
        score_offset=0.0,
        score_divisor=10.0,
        skeleton_dummy="0",
        is_canonical=False,
        seen=False,
    )


def _spec_grade_0_100() -> GradeScaleSpec:
    # Single-token quartile anchors on the 0–100 judge scale.
    return GradeScaleSpec(
        scheme="grade_0_100",
        label="scale_0_100",
        system_prompt=(
            "You are a search relevance grader. For each numbered document, output one integer "
            "relevance score on a 0–100 scale (0 = not relevant, 100 = maximally relevant). "
            "Use only the anchor values 0, 33, 66, or 100. Do not rank and do not explain."
        ),
        grade_lines=(
            "- 100 = directly and completely answers the query",
            "- 66 = strongly relevant but incomplete",
            "- 33 = weakly relevant or topical background",
            "- 0 = not relevant",
        ),
        output_format_line="Output one line per document in the form: [<id>] Grade: <0|33|66|100>",
        grade_strings=("0", "3", "6", "9"),
        grade_values=(0.0, 33.0, 66.0, 100.0),
        score_offset=0.0,
        score_divisor=100.0,
        skeleton_dummy="0",
        is_canonical=False,
        seen=False,
    )


def _spec_grade_1_5() -> GradeScaleSpec:
    return GradeScaleSpec(
        scheme="grade_1_5",
        label="scale_1_5",
        system_prompt=(
            "You are a search relevance grader. For each numbered document, output one integer "
            "Likert grade from 1 to 5 (1 = not relevant, 5 = maximally relevant). "
            "Do not rank documents and do not explain."
        ),
        grade_lines=(
            "- 5 = directly and completely answers the query",
            "- 4 = strongly relevant but incomplete",
            "- 3 = weakly relevant or topical background",
            "- 1–2 = not relevant",
        ),
        output_format_line="Output one line per document in the form: [<id>] Grade: <1|2|3|4|5>",
        grade_strings=("1", "2", "3", "4", "5"),
        grade_values=(1.0, 2.0, 3.0, 4.0, 5.0),
        score_offset=1.0,
        score_divisor=4.0,
        skeleton_dummy="1",
        is_canonical=False,
        seen=False,
    )


def _spec_grade_0_1() -> GradeScaleSpec:
    # Continuous 0.0–1.0 via single-digit anchor tokens.
    return GradeScaleSpec(
        scheme="grade_0_1",
        label="scale_0_1",
        system_prompt=(
            "You are a search relevance grader. For each numbered document, output one relevance "
            "score on a 0.0–1.0 scale. Use only the anchor values 0, 3, 6, or 9 (interpreted as "
            "0.0, 0.33, 0.67, 1.0). Do not rank and do not explain."
        ),
        grade_lines=(
            "- 9 (=1.0) = directly and completely answers the query",
            "- 6 (=0.67) = strongly relevant but incomplete",
            "- 3 (=0.33) = weakly relevant or topical background",
            "- 0 (=0.0) = not relevant",
        ),
        output_format_line="Output one line per document in the form: [<id>] Grade: <0|3|6|9>",
        grade_strings=("0", "3", "6", "9"),
        grade_values=(0.0, 0.33, 0.67, 1.0),
        score_offset=0.0,
        score_divisor=1.0,
        skeleton_dummy="0",
        is_canonical=False,
        seen=False,
    )


def _spec_grade_descriptive() -> GradeScaleSpec:
    return GradeScaleSpec(
        scheme="grade_descriptive",
        label="scale_descriptive",
        system_prompt=(
            "You are a search relevance grader. For each numbered document, output one fixed "
            "relevance label from {Not, Weak, Strong, Full}. Do not rank documents and do not explain."
        ),
        grade_lines=(
            "- Full = directly and completely answers the query",
            "- Strong = strongly relevant but incomplete",
            "- Weak = weakly relevant or topical background",
            "- Not = not relevant",
        ),
        output_format_line="Output one line per document in the form: [<id>] Grade: <Not|Weak|Strong|Full>",
        grade_strings=("Not", "Weak", "Strong", "Full"),
        grade_values=(0.0, 1.0, 2.0, 3.0),
        score_offset=0.0,
        score_divisor=3.0,
        skeleton_dummy="Not",
        is_canonical=False,
        seen=False,
    )


_SCALE_SPECS: dict[ScaleScheme, GradeScaleSpec] = {
    "grade_0_3": _spec_grade_0_3(),
    "grade_0_10": _spec_grade_0_10(),
    "grade_0_100": _spec_grade_0_100(),
    "grade_1_5": _spec_grade_1_5(),
    "grade_0_1": _spec_grade_0_1(),
    "grade_descriptive": _spec_grade_descriptive(),
}


def resolve_scale_spec(scheme: ScaleScheme) -> GradeScaleSpec:
    try:
        return _SCALE_SPECS[scheme]
    except KeyError as exc:
        known = ", ".join(sorted(_SCALE_SPECS))
        raise ValueError(f"Unknown scale scheme: {scheme!r}; known: {known}") from exc


def build_scale_grade_user_body(
    instruction: str | None,
    query: str,
    doc_texts: list[str],
    *,
    max_doc_chars: int,
    scale_spec: GradeScaleSpec,
    slot_labels: tuple[str, ...] | None = None,
) -> str:
    """User-section body with a non-canonical scoring-scale rubric."""
    inst = instruction or DEFAULT_INSTRUCTION
    if slot_labels is None:
        slot_labels = tuple(f"[{i}]" for i in range(1, len(doc_texts) + 1))
    doc_section = "\n".join(
        build_doc_blocks(
            doc_texts,
            max_doc_chars=max_doc_chars,
            slot_labels=slot_labels,
            separator="newline",
        )
    )
    return (
        f"Instruction: {inst}\n"
        f"Query: {query}\n\n"
        "Documents:\n" + doc_section + "\n\n"
        "Evaluate every document independently for relevance to the query.\n"
        "Assign exactly one relevance grade to every document:\n"
        + "\n".join(scale_spec.grade_lines)
        + "\n"
        + scale_spec.output_format_line
    )


def scale_spec_for_render_variant(variant: RenderVariant | None) -> GradeScaleSpec | None:
    if variant is None or variant.scale_scheme is None:
        return None
    return resolve_scale_spec(variant.scale_scheme)


def generate_scale_variants(
    K: int,
    seeds: list[int],
    *,
    n_slots: int = 20,
) -> list[RenderVariant]:
    """Deterministic scale variants; index 0 is always training-seen 0–3."""
    if K <= 0:
        raise ValueError("K must be positive")
    if len(seeds) != K:
        raise ValueError(f"seeds length ({len(seeds)}) must equal K ({K})")
    schemes = SCALE_SCHEMES[:K]
    labels = slot_labels_for_chunk(n_slots, id_scheme="numeric", id_slot_permutation=None)
    out: list[RenderVariant] = []
    for idx, (scheme, _seed) in enumerate(zip(schemes, seeds, strict=False)):
        spec = resolve_scale_spec(scheme)
        out.append(
            RenderVariant(
                label=spec.label,
                id_scheme="numeric",
                separator="newline",
                rubric_id=DEFAULT_GRADE_RUBRIC_ID,
                slot_labels=labels,
                is_canonical=spec.is_canonical,
                seen=spec.seen,
                strategy="scale",
                variant_index=idx,
                scale_scheme=scheme,
            )
        )
    return out
