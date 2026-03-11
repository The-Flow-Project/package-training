"""
Flow Training - Training Logic Module
"""
# Add imports here

from .train import Trainer
from .config import (
    TrainingConfig,
    DatasetConfig,
    ModelConfig,
    ReportingConfig,
    ModelCardConfig,
)

from .logging_config import setup_logger

setup_logger()

__version__ = "0.2.0"
__license__ = "MIT"

__all__ = [
    "Trainer",
    "TrainingConfig",
    "DatasetConfig",
    "ModelConfig",
    "ReportingConfig",
    "ModelCardConfig"
    "__version__",
]
