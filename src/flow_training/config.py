"""
Dataclasses for configuring the training parameters.
"""

from dataclasses import dataclass, field
from typing import List
import os

from flow_training.enums import (
    LoggingStrategy,
    EvalStrategy,
    SaveStrategy,
    LRSchedulerType,
    ReportingType,
)


@dataclass
class TrainingConfig:
    """
    Configuration parameters for training the TrOCR model.

    Attributes:
        EPOCHS: Number of training epochs (required, must be positive).
        RUN_NAME: Name of the training run.
        OUTPUT_DIR: Directory to save the trained model, logs, and checkpoints.
        OVERWRITE_OUTPUT_DIR: Whether to overwrite the output directory if it exists.
    """

    # Required parameters
    EPOCHS: int

    RUN_NAME: str = "TrOCR_Training_Run"
    OUTPUT_DIR: str | None = None
    OVERWRITE_OUTPUT_DIR: bool = False
    PUSH_TO_HUB: bool = False
    HUGGINGFACE_TOKEN: str | None = None
    HUGGINGFACE_MODEL_ID: str | None = None
    HUGGINGFACE_REPO_PRIVATE: bool = True

    DO_TRAIN: bool = True
    DO_EVAL: bool = True
    DO_PREDICT: bool = False

    SEED: int = 42
    BATCH_SIZE: int = 16
    AUTO_FIND_BATCH_SIZE: bool = True
    GRADIENT_ACCUMULATION_STEPS: int = 2
    GRADIENT_CHECKPOINTING: bool = False
    WEIGHT_DECAY: float = 0.01
    LEARNING_RATE: float = 5e-5
    LR_SCHEDULER_TYPE: LRSchedulerType = LRSchedulerType.LINEAR
    RESUME_FROM_CHECKPOINT: bool = False
    NUM_BEAMS: int = 5

    EARLY_STOPPING: bool = False

    LOGGING_FIRST_STEP: bool = True
    LOGGING_STRATEGY: LoggingStrategy = LoggingStrategy.STEPS
    LOGGING_STEPS: int = 10

    EVAL_STRATEGY: EvalStrategy = EvalStrategy.STEPS
    EVAL_STEPS: int = 20
    EVAL_ACCUMULATION_STEPS: int = 1
    EVAL_DELAY: int = 0

    SAVE_STRATEGY: SaveStrategy = SaveStrategy.BEST
    SAVE_STEPS: int = 50
    SAVE_TOTAL_LIMIT: int = 5

    LOAD_BEST_MODEL_AT_END: bool = True
    METRIC_FOR_BEST_MODEL: str = "eval_cer"
    GREATER_IS_BETTER: bool = False

    WARMUP_RATIO: float = 0.05

    DISABLE_TQDM: bool = False

    TF32: bool = False
    FP16: bool = True
    BF16: bool = False

    PREDICT_WITH_GENERATE: bool = True
    GENERATION_NUM_BEAMS: int = 5
    GENERATION_MAX_LENGTH: int | None = None

    # Derived field set in __post_init__
    LOGGING_DIR: str = field(init=False, default="")

    def __post_init__(self) -> None:
        """
        Validate and post-process configuration parameters.

        Raises:
            ValueError: If any parameter is invalid or logically inconsistent.
        """
        self._validate_numeric_params()
        self._validate_logic()
        self._setup_derived_params()

    def _validate_numeric_params(self) -> None:
        """
        Validate numeric configuration parameters.

        Raises:
            ValueError: If any numeric parameter is out of valid range.
        """
        if self.EPOCHS <= 0:
            raise ValueError(f"EPOCHS must be positive, got {self.EPOCHS}")

        if self.BATCH_SIZE <= 0:
            raise ValueError(f"BATCH_SIZE must be positive, got {self.BATCH_SIZE}")

        if self.LEARNING_RATE <= 0:
            raise ValueError(f"LEARNING_RATE must be positive, got {self.LEARNING_RATE}")

        if self.WEIGHT_DECAY < 0:
            raise ValueError(f"WEIGHT_DECAY must be non-negative, got {self.WEIGHT_DECAY}")

        if self.NUM_BEAMS <= 0:
            raise ValueError(f"NUM_BEAMS must be positive, got {self.NUM_BEAMS}")

        if self.EVAL_STEPS <= 0:
            raise ValueError(f"EVAL_STEPS must be positive, got {self.EVAL_STEPS}")

        if self.LOGGING_STEPS <= 0:
            raise ValueError(f"LOGGING_STEPS must be positive, got {self.LOGGING_STEPS}")

        if self.SAVE_STEPS <= 0:
            raise ValueError(f"SAVE_STEPS must be positive, got {self.SAVE_STEPS}")

        if self.SAVE_TOTAL_LIMIT <= 0:
            raise ValueError(f"SAVE_TOTAL_LIMIT must be positive, got {self.SAVE_TOTAL_LIMIT}")

        if not (0 <= self.WARMUP_RATIO <= 1):
            raise ValueError(f"WARMUP_RATIO must be between 0 and 1, got {self.WARMUP_RATIO}")

        if self.SEED < 0:
            raise ValueError(f"SEED must be non-negative, got {self.SEED}")

        if self.GRADIENT_ACCUMULATION_STEPS <= 0:
            raise ValueError(
                f"GRADIENT_ACCUMULATION_STEPS must be positive, got {self.GRADIENT_ACCUMULATION_STEPS}"
            )

        if self.EVAL_ACCUMULATION_STEPS <= 0:
            raise ValueError(
                f"EVAL_ACCUMULATION_STEPS must be positive, got {self.EVAL_ACCUMULATION_STEPS}"
            )

        if self.EVAL_DELAY < 0:
            raise ValueError(f"EVAL_DELAY must be non-negative, got {self.EVAL_DELAY}")

        if self.GENERATION_NUM_BEAMS <= 0:
            raise ValueError(
                f"GENERATION_NUM_BEAMS must be positive, got {self.GENERATION_NUM_BEAMS}"
            )

        if self.GENERATION_MAX_LENGTH is not None and self.GENERATION_MAX_LENGTH <= 0:
            raise ValueError(
                f"GENERATION_MAX_LENGTH must be positive or None, got {self.GENERATION_MAX_LENGTH}"
            )

    def _validate_logic(self) -> None:
        """
        Validate logical constraints between parameters.

        Raises:
            ValueError: If configuration has logical contradictions.
        """
        if self.FP16 and self.BF16:
            raise ValueError("Cannot use both FP16 and BF16 at the same time")

        if not self.DO_TRAIN and not self.DO_EVAL and not self.DO_PREDICT:
            raise ValueError("At least one of DO_TRAIN, DO_EVAL, or DO_PREDICT must be True")

        if self.AUTO_FIND_BATCH_SIZE and self.BATCH_SIZE != 16:
            import warnings

            warnings.warn(
                "AUTO_FIND_BATCH_SIZE=True will override BATCH_SIZE setting",
                UserWarning,
            )

        if self.OUTPUT_DIR is None and self.PUSH_TO_HUB is None:
            raise ValueError("OUTPUT_DIR must be specified if PUSH_TO_HUB is False")

        if self.PUSH_TO_HUB:
            if self.HUGGINGFACE_MODEL_ID is None:
                raise ValueError("HUGGINGFACE_MODEL_ID must be specified when PUSH_TO_HUB is True")
            if self.HUGGINGFACE_TOKEN is None:
                raise ValueError("HUGGINGFACE_TOKEN must be specified when PUSH_TO_HUB is True")

    def _setup_derived_params(self) -> None:
        """Set up derived parameters from configuration."""
        path = self.OUTPUT_DIR or "./"
        self.LOGGING_DIR = os.path.join(path, "logs")


@dataclass
class DatasetConfig:
    """
    Configuration parameters for the dataset used in training.

    Attributes:
        HUGGINGFACE_DATASET_SOURCE: HuggingFace Hub dataset identifier.
        Has to be a line based dataset with "image" and "text" fields.
        HUGGINGFACE_EVAL_SPLIT_NAME: Name of evaluation split if available in dataset (e.g., "validation").
        HUGGINGFACE_TOKEN: HuggingFace token for accessing private datasets (default None).
        EVAL_SPLIT_RATIO: Ratio for train/test split if no eval split provided (must be between 0 and 1, default 0.1).
        MIN_LINE_HEIGHT: Minimum image height for filtering.
    """

    HUGGINGFACE_DATASET_SOURCE: str | None = None
    HUGGINGFACE_EVAL_SPLIT_NAME: str | None = None
    HUGGINGFACE_TOKEN: str | None = None
    EVAL_SPLIT_RATIO: float = 0.1
    MIN_LINE_HEIGHT: int = 1

    def __post_init__(self) -> None:
        """
        Validate dataset configuration parameters.

        Raises:
            ValueError: If any parameter is invalid.
        """
        if not (0.0 < self.EVAL_SPLIT_RATIO < 1.0):
            raise ValueError(
                f"EVAL_SPLIT_RATIO must be between 0 and 1, got {self.EVAL_SPLIT_RATIO}"
            )

        if self.MIN_LINE_HEIGHT <= 0:
            raise ValueError(f"MIN_LINE_HEIGHT must be positive, got {self.MIN_LINE_HEIGHT}")

        if not self.HUGGINGFACE_DATASET_SOURCE:
            raise ValueError("HUGGINGFACE_DATASET_SOURCE cannot be empty")


@dataclass
class ModelConfig:
    """
    Configuration parameters for the TrOCR basemodel and processor.

    Attributes:
        BASE_MODEL_NAME: HuggingFace model identifier for the base TrOCR model.
        BASE_PROCESSOR_NAME: HuggingFace model identifier for the base TrOCR processor.

        Note:
        The default values are set to "microsoft/trocr-large-handwritten"
        which is a large TrOCR model fine-tuned on handwritten text.
    """

    BASE_MODEL_NAME: str = "microsoft/trocr-large-handwritten"
    BASE_PROCESSOR_NAME: str | None = "microsoft/trocr-large-handwritten"

    def __post_init__(self) -> None:
        """
        Validate model configuration parameters.

        Raises:
            ValueError: If any parameter is invalid.
        """
        if not self.BASE_MODEL_NAME:
            raise ValueError("BASE_MODEL_NAME cannot be empty")

        if self.BASE_PROCESSOR_NAME is None and self.BASE_MODEL_NAME:
            self.BASE_PROCESSOR_NAME = self.BASE_MODEL_NAME


@dataclass
class ModelCardConfig:
    """
    Configuration parameters for the model card metadata.

    Attributes:
        MODEL_CARD_NAME: Name of the model.
        MODEL_CARD_LANGUAGE: Language of the model (e.g., "en" for English, default None).
        MODEL_CARD_LICENSE: License information for the model card (e.g. MIT, default None).
        MODEL_CARD_TAGS: List of tags to categorize the model
        MODEL_CARD_FINETUNED_FROM: Optional field to specify the base model the fine-tuned model is derived from
        MODEL_CARD_TASKS: Field to specify the tasks the model is fine-tuned for
    """
    MODEL_CARD_NAME: str = "TrOCR-HTR-Model"
    MODEL_CARD_LANGUAGE: str | None = None
    MODEL_CARD_LICENSE: str | None = None
    MODEL_CARD_TAGS: List[str] | None = None
    MODEL_CARD_FINETUNED_FROM: str | None = None
    MODEL_CARD_TASKS: List[str] | None = field(default_factory=["handwritten-text-recognition", "text-generation"])

@dataclass
class ReportingConfig:
    """
    Configuration parameters for reporting training metrics.

    Attributes:
        REPORT_TO: Reporting backend (wandb, swanlab, tensorboard, none, all).
        REPORT_PROJECTNAME: Project name for reporting.
        PROJECT_WORKSPACE: Workspace/organization name for reporting.
    """

    REPORT_TO: str = ReportingType.SWANLAB.value
    REPORT_PROJECTNAME: str | None = "TrOCR_Training_Project"
    PROJECT_WORKSPACE: str | None = "TrOCRTraining"

    def __post_init__(self) -> None:
        """
        Validate and setup reporting configuration.

        Raises:
            ValueError: If any parameter is invalid.
        """
        if self.REPORT_TO != "none":
            if not self.REPORT_PROJECTNAME:
                raise ValueError("REPORT_PROJECTNAME cannot be empty when REPORT_TO is set")

            if not self.PROJECT_WORKSPACE:
                raise ValueError("PROJECT_WORKSPACE cannot be empty when REPORT_TO is set")

        if self.REPORT_TO == ReportingType.SWANLAB.value:
            os.environ["SWANLAB_PROJ_NAME"] = self.REPORT_PROJECTNAME
            os.environ["SWANLAB_WORKSPACE"] = self.PROJECT_WORKSPACE
