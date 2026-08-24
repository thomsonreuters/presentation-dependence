"""Scientific analysis primitives and representative checks."""

from .metrics import (
    clustered_linear_calibration,
    gain_histogram,
    hierarchical_bootstrap_ci,
    mean_pairwise_jaccard,
    paired_bootstrap_ci,
    tau_psi_from_aligned_scores,
    verdict_flip_rate,
)
from .representative import (
    plan_representative,
    run_representative,
    validate_representative,
)
from .terminology import PAPER_METHOD_LABELS, canonical_method_key, paper_method_label

__all__ = [
    "PAPER_METHOD_LABELS",
    "canonical_method_key",
    "clustered_linear_calibration",
    "gain_histogram",
    "hierarchical_bootstrap_ci",
    "mean_pairwise_jaccard",
    "paper_method_label",
    "paired_bootstrap_ci",
    "plan_representative",
    "run_representative",
    "tau_psi_from_aligned_scores",
    "validate_representative",
    "verdict_flip_rate",
]
