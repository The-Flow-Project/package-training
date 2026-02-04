"""
Dataclasses for configuring the training parameters.
"""

from dataclasses import dataclass
import os


@dataclass
class TrainingConfig:
    """
    Configuration parameters for training the TrOCR model.
    """

    # Required parameters
    EPOCHS: int
    RUN_NAME: str = "TrOCR_Training_Run"  # Name of the training run
    OUTPUT_DIR: str = (
        "./trocr_finetuned_model"  # Directory to save the trained model, logs, and checkpoints
    )
    OVERWRITE_OUTPUT_DIR: bool = False  # Whether to overwrite the output directory if it exists

    DO_TRAIN: bool = True  # Whether to perform training
    DO_EVAL: bool = True  # Whether to perform evaluation
    DO_PREDICT: bool = False  # Whether to perform prediction

    SEED: int = 42  # Seed for reproducibility
    BATCH_SIZE: int = 16  # Batch size for training
    AUTO_FIND_BATCH_SIZE: bool = True  # Automatically find the best batch size
    GRADIENT_ACCUMULATION_STEPS: int = 2
    GRADIENT_CHECKPOINTING: bool = False  # Enable gradient checkpointing to save memory
    WEIGHT_DECAY: float = 0.01  # Weight decay for optimizer
    LEARNING_RATE: float = 5e-5
    LR_SCHEDULER_TYPE: str = "linear"  # Type of learning rate scheduler
    RESUME_FROM_CHECKPOINT: bool = False  # Whether to resume from a checkpoint
    NUM_BEAMS: int = 5

    EARLY_STOPPING: bool = False  # Stop training early if no improvement

    LOGGING_FIRST_STEP: bool = True  # Log the first training step
    LOGGING_STRATEGY: str = "steps"  # Logging strategy: 'steps' or 'epoch'
    LOGGING_STEPS: int = 10  # Log every X steps

    EVAL_STRATEGY: str = "steps"  # Evaluation strategy: 'steps' or 'epoch'
    EVAL_STEPS: int = 20  # Evaluate every X steps
    EVAL_ACCUMULATION_STEPS: int = 1  # Number of steps to accumulate before evaluation
    EVAL_DELAY: int = 0  # Number of epochs to wait before first evaluation

    SAVE_STRATEGY: str = "best"  # Save strategy: 'steps', 'epoch', or 'best' (saving whenever best_metric is achieved)
    SAVE_STEPS: int = 50  # Save checkpoint every X steps
    SAVE_TOTAL_LIMIT: int = 5  # Maximum number of checkpoints to keep

    LOAD_BEST_MODEL_AT_END: bool = True  # Load the best model at the end of training
    METRIC_FOR_BEST_MODEL: str = "eval_cer"  # Metric to use to determine the best model
    GREATER_IS_BETTER: bool = False  # Whether the best model metric should be maximized

    WARMUP_RATIO: float = 0.05  # Warmup ratio for learning rate scheduler

    DISABLE_TQDM: bool = False  # Disable tqdm progress bars

    TF32: bool = True  # Enable TF32 on Ampere GPUs for faster training
    FP16: bool = True
    BF16: bool = False  # Use bfloat16 precision (if supported)

    PREDICT_WITH_GENERATE: bool = True  # Use generate() for predictions
    GENERATION_NUM_BEAMS: int = 5  # Number of beams for generation during prediction
    GENERATION_MAX_LENGTH: int = None  # Maximum length for generated sequences during prediction

    def __post_init__(self):
        self.LOGGING_DIR = os.path.join(self.OUTPUT_DIR, "logs")


@dataclass
class DatasetConfig:
    """
    Configuration parameters for the dataset used in training.
    """

    HUGGINGFACE_DATASET_SOURCE: str = "dh-unibe/data-towerbooks-textlines"
    HUGGINGFACE_EVAL_SPLIT_NAME: str = "validation"  # Evaluation split name in the dataset
    HUGGINGFACE_DATASET_OUTPUT: str = "output"
    HUGGINGFACE_OUTPUT_PRIVATE: bool = True  # Whether the output dataset repo is private
    HUGGINGFACE_TOKEN: str = None  # Huggingface token for authentication
    EVAL_SPLIT_RATIO: float = 0.1  # Ratio of the dataset to use for testing \
    # (if no separate eval dataset is provided or eval split is available)
    MIN_LINE_HEIGHT: int = 32  # Minimum height of line images in pixels (filtering)


@dataclass
class ModelConfig:
    """
    Configuration parameters for the TrOCR basemodel and processor.
    """

    BASE_MODEL_NAME: str = "microsoft/trocr-large-handwritten"
    BASE_PROCESSOR_NAME: str = "microsoft/trocr-large-handwritten"


@dataclass
class ReportingConfig:
    """
    Configuration parameters for reporting training metrics.
    """

    REPORT_TO: str = "swanlab"  # Options: 'wandb', 'swanlab', 'none', 'all', 'tensorboard', etc.
    REPORT_PROJECTNAME: str = "TrOCR_Training_Project"  # project name
    PROJECT_WORKSPACE: str = "TrOCRTraining"

    def __post_init__(self):
        if self.REPORT_TO == "swanlab":
            os.environ["SWANLAB_PROJ_NAME"] = self.REPORT_PROJECTNAME
            os.environ["SWANLAB_WORKSPACE"] = self.PROJECT_WORKSPACE
