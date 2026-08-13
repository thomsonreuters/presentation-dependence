"""Self-distillation pipeline for batched-PW relevance grading.

Module map:

- :mod:`readout`: continuous-readout primitives (vLLM ``prompt_logprobs`` /
  HF ``output_logits`` -> expected value over ``{0, 1, 2, 3}`` grade tokens).
- :mod:`silver_io`: silver-label record schema + JSONL/manifest writers.
- :mod:`teacher`: K-shot batched-self-consistency teacher driver.

"""

# There is a real import cycle here, and two things keep it survivable.
#
# :mod:`teacher` imports :mod:`presentation_dependence.eval.runner_setup`, which imports
# :mod:`presentation_dependence.rerankers`, whose registry loads qwen3.py and
# qwen3_instruct_grade.py, both of which import back into this package for
# :mod:`readout` and :mod:`engines`.
#
# The cycle does not fail today because (a) the leaf re-exports below run
# before anything reaches the rerankers, so the names already exist on this
# module by the time they are asked for, and (b) those consumers import the
# submodules directly rather than this package. Break either and it becomes a
# hard ImportError: adding a teacher re-export *above* these lines, or
# changing a consumer to ``from presentation_dependence.self_distill import ...``, both
# fail at package init. Import the teacher from
# :mod:`presentation_dependence.self_distill.teacher` instead of re-exporting it.
from presentation_dependence.self_distill.readout import (
    expected_grade,
    resolve_grade_token_ids,
)
from presentation_dependence.self_distill.silver_io import (
    SilverLabel,
    write_silver_jsonl,
    write_manifest,
)

__all__ = [
    "SilverLabel",
    "expected_grade",
    "resolve_grade_token_ids",
    "write_manifest",
    "write_silver_jsonl",
]
