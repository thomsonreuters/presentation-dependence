"""Silver-label generation utilities for SFT and eval-gold runs."""

from typing import TYPE_CHECKING, Any

from presentation_dependence.silver_data.config import (
    SilverConfig,
    SilverConfigKind,
    classify_silver_config,
    load_silver_config,
    resolve_silver_config_path,
    validate_silver_config,
)
from presentation_dependence.silver_data.transforms import (
    derive_silver_shard,
    materialize_qid_delta,
    split_silver_cohort,
)

if TYPE_CHECKING:
    from presentation_dependence.silver_data.generate import (
        BudgetExceeded,
        RunDirLocked,
        SilverGenerator,
    )

__all__ = [
    "BudgetExceeded",
    "RunDirLocked",
    "SilverConfig",
    "SilverConfigKind",
    "SilverGenerator",
    "classify_silver_config",
    "derive_silver_shard",
    "load_silver_config",
    "materialize_qid_delta",
    "resolve_silver_config_path",
    "split_silver_cohort",
    "validate_silver_config",
]


def __getattr__(name: str) -> Any:
    """Load the generation stack only when a caller requests it.

    Transform and prompt consumers should not pull in the eval/reranker
    dependency graph when Python initializes this package.
    """
    if name in {"BudgetExceeded", "RunDirLocked", "SilverGenerator"}:
        from presentation_dependence.silver_data import generate

        return getattr(generate, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
