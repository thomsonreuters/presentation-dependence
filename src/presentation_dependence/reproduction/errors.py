"""Typed failures raised by reproduction pipeline stages."""


class ReproductionError(RuntimeError):
    """Base class for operator-facing reproduction failures."""


class SilverStageError(ReproductionError):
    """Silver declarations or artifacts are incomplete."""


class TrainingStageError(ReproductionError):
    """Training declarations, runs, or selections are invalid."""


class DirectEvalStageError(ReproductionError):
    """Direct-evaluation declarations or artifacts are invalid."""


class DownstreamStageError(ReproductionError):
    """Downstream declarations or artifacts are invalid."""
