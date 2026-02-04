"""
Enumerations for training configuration.

Provides enum types for string-based configuration values to ensure
type safety and prevent invalid configuration parameters.
"""

from enum import Enum


class LoggingStrategy(Enum):
    """Logging strategy during training."""

    STEPS = "steps"
    EPOCH = "epoch"


class EvalStrategy(Enum):
    """Evaluation strategy during training."""

    STEPS = "steps"
    EPOCH = "epoch"
    NO = "no"


class SaveStrategy(Enum):
    """Model checkpointing strategy."""

    STEPS = "steps"
    EPOCH = "epoch"
    BEST = "best"


class LRSchedulerType(Enum):
    """Learning rate scheduler type."""

    LINEAR = "linear"
    COSINE = "cosine"
    POLYNOMIAL = "polynomial"
    CONSTANT = "constant"
    COSINE_WITH_RESTARTS = "cosine_with_restarts"


class ReportingType(Enum):
    """Reporting backend for experiment tracking."""

    WANDB = "wandb"
    SWANLAB = "swanlab"
    TENSORBOARD = "tensorboard"
    NONE = "none"
    ALL = "all"
