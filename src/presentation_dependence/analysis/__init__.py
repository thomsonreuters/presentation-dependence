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

__all__ = [
    "clustered_linear_calibration",
    "gain_histogram",
    "hierarchical_bootstrap_ci",
    "mean_pairwise_jaccard",
    "paired_bootstrap_ci",
    "plan_representative",
    "run_representative",
    "tau_psi_from_aligned_scores",
    "validate_representative",
    "verdict_flip_rate",
]
