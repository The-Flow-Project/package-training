"""
Example tests demonstrating configuration validation.

Run with: pytest tests/test_config_example.py -v
"""

import pytest
from flow_training.config import TrainingConfig, DatasetConfig, ReportingConfig
from flow_training.enums import LoggingStrategy, EvalStrategy, SaveStrategy, ReportingType


class TestTrainingConfigValidation:
    """Test TrainingConfig validation."""

    def test_valid_config(self):
        """Test creation of valid TrainingConfig."""
        config = TrainingConfig(EPOCHS=10)
        assert config.EPOCHS == 10
        assert config.BATCH_SIZE == 16
        assert config.LOGGING_STRATEGY == LoggingStrategy.STEPS

    def test_epochs_must_be_positive(self):
        """Test that EPOCHS must be positive."""
        with pytest.raises(ValueError, match="EPOCHS must be positive"):
            TrainingConfig(EPOCHS=0)

        with pytest.raises(ValueError, match="EPOCHS must be positive"):
            TrainingConfig(EPOCHS=-5)

    def test_batch_size_must_be_positive(self):
        """Test that BATCH_SIZE must be positive."""
        with pytest.raises(ValueError, match="BATCH_SIZE must be positive"):
            TrainingConfig(EPOCHS=10, BATCH_SIZE=0)

    def test_learning_rate_must_be_positive(self):
        """Test that LEARNING_RATE must be positive."""
        with pytest.raises(ValueError, match="LEARNING_RATE must be positive"):
            TrainingConfig(EPOCHS=10, LEARNING_RATE=0)

        with pytest.raises(ValueError, match="LEARNING_RATE must be positive"):
            TrainingConfig(EPOCHS=10, LEARNING_RATE=-1e-5)

    def test_warmup_ratio_bounds(self):
        """Test that WARMUP_RATIO is between 0 and 1."""
        with pytest.raises(ValueError, match="WARMUP_RATIO must be between 0 and 1"):
            TrainingConfig(EPOCHS=10, WARMUP_RATIO=1.5)

        with pytest.raises(ValueError, match="WARMUP_RATIO must be between 0 and 1"):
            TrainingConfig(EPOCHS=10, WARMUP_RATIO=-0.1)

    def test_cannot_use_both_fp16_and_bf16(self):
        """Test that FP16 and BF16 cannot both be True."""
        with pytest.raises(ValueError, match="Cannot use both FP16 and BF16"):
            TrainingConfig(EPOCHS=10, FP16=True, BF16=True)

    def test_at_least_one_train_eval_predict(self):
        """Test that at least one of DO_TRAIN, DO_EVAL, DO_PREDICT is True."""
        with pytest.raises(ValueError, match="At least one of DO_TRAIN, DO_EVAL, or DO_PREDICT"):
            TrainingConfig(EPOCHS=10, DO_TRAIN=False, DO_EVAL=False, DO_PREDICT=False)

    def test_logging_dir_created(self):
        """Test that LOGGING_DIR is created."""
        config = TrainingConfig(EPOCHS=10, OUTPUT_DIR="./test_output")
        assert config.LOGGING_DIR == "./test_output/logs"

    def test_enum_type_for_logging_strategy(self):
        """Test that LOGGING_STRATEGY uses enum type."""
        config = TrainingConfig(EPOCHS=10, LOGGING_STRATEGY=LoggingStrategy.EPOCH)
        assert config.LOGGING_STRATEGY == LoggingStrategy.EPOCH
        assert config.LOGGING_STRATEGY.value == "epoch"

    def test_enum_type_for_eval_strategy(self):
        """Test that EVAL_STRATEGY uses enum type."""
        config = TrainingConfig(EPOCHS=10, EVAL_STRATEGY=EvalStrategy.EPOCH)
        assert config.EVAL_STRATEGY == EvalStrategy.EPOCH

    def test_enum_type_for_save_strategy(self):
        """Test that SAVE_STRATEGY uses enum type."""
        config = TrainingConfig(EPOCHS=10, SAVE_STRATEGY=SaveStrategy.STEPS)
        assert config.SAVE_STRATEGY == SaveStrategy.STEPS


class TestDatasetConfigValidation:
    """Test DatasetConfig validation."""

    def test_valid_config(self):
        """Test creation of valid DatasetConfig."""
        config = DatasetConfig()
        assert config.EVAL_SPLIT_RATIO == 0.1
        assert config.MIN_LINE_HEIGHT == 1

    def test_eval_split_ratio_bounds(self):
        """Test that EVAL_SPLIT_RATIO is between 0 and 1."""
        with pytest.raises(ValueError, match="EVAL_SPLIT_RATIO must be between 0 and 1"):
            DatasetConfig(EVAL_SPLIT_RATIO=1.5)

        with pytest.raises(ValueError, match="EVAL_SPLIT_RATIO must be between 0 and 1"):
            DatasetConfig(EVAL_SPLIT_RATIO=0)

        with pytest.raises(ValueError, match="EVAL_SPLIT_RATIO must be between 0 and 1"):
            DatasetConfig(EVAL_SPLIT_RATIO=1.0)

    def test_min_line_height_must_be_positive(self):
        """Test that MIN_LINE_HEIGHT must be positive."""
        with pytest.raises(ValueError, match="MIN_LINE_HEIGHT must be positive"):
            DatasetConfig(MIN_LINE_HEIGHT=0)

        with pytest.raises(ValueError, match="MIN_LINE_HEIGHT must be positive"):
            DatasetConfig(MIN_LINE_HEIGHT=-5)

    def test_dataset_source_cannot_be_empty(self):
        """Test that HUGGINGFACE_DATASET_SOURCE cannot be empty."""
        with pytest.raises(ValueError, match="HUGGINGFACE_DATASET_SOURCE cannot be empty"):
            DatasetConfig(HUGGINGFACE_DATASET_SOURCE="")

    def test_eval_split_name_cannot_be_empty(self):
        """Test that HUGGINGFACE_EVAL_SPLIT_NAME cannot be empty."""
        with pytest.raises(ValueError, match="HUGGINGFACE_EVAL_SPLIT_NAME cannot be empty"):
            DatasetConfig(HUGGINGFACE_EVAL_SPLIT_NAME="")


class TestReportingConfigValidation:
    """Test ReportingConfig validation."""

    def test_valid_config(self):
        """Test creation of valid ReportingConfig."""
        config = ReportingConfig()
        assert config.REPORT_TO == ReportingType.SWANLAB
        assert config.REPORT_PROJECTNAME == "TrOCR_Training_Project"

    def test_project_name_cannot_be_empty(self):
        """Test that REPORT_PROJECTNAME cannot be empty."""
        with pytest.raises(ValueError, match="REPORT_PROJECTNAME cannot be empty"):
            ReportingConfig(REPORT_PROJECTNAME="")

    def test_workspace_cannot_be_empty(self):
        """Test that PROJECT_WORKSPACE cannot be empty."""
        with pytest.raises(ValueError, match="PROJECT_WORKSPACE cannot be empty"):
            ReportingConfig(PROJECT_WORKSPACE="")

    def test_enum_type_for_report_to(self):
        """Test that REPORT_TO uses enum type."""
        config = ReportingConfig(REPORT_TO=ReportingType.WANDB)
        assert config.REPORT_TO == ReportingType.WANDB
        assert config.REPORT_TO.value == "wandb"

    def test_swanlab_env_vars_set(self, monkeypatch):
        """Test that environment variables are set for swanlab."""
        monkeypatch.delenv("SWANLAB_PROJ_NAME", raising=False)
        monkeypatch.delenv("SWANLAB_WORKSPACE", raising=False)

        config = ReportingConfig(
            REPORT_TO=ReportingType.SWANLAB,
            REPORT_PROJECTNAME="TestProject",
            PROJECT_WORKSPACE="TestWorkspace"
        )

        import os
        assert os.environ.get("SWANLAB_PROJ_NAME") == "TestProject"
        assert os.environ.get("SWANLAB_WORKSPACE") == "TestWorkspace"


class TestIntegration:
    """Integration tests for multiple configs."""

    def test_create_all_configs_together(self):
        """Test creating all configs together."""
        training_config = TrainingConfig(
            EPOCHS=10,
            LOGGING_STRATEGY=LoggingStrategy.STEPS,
            EVAL_STRATEGY=EvalStrategy.STEPS,
            SAVE_STRATEGY=SaveStrategy.BEST
        )
        dataset_config = DatasetConfig()
        reporting_config = ReportingConfig(REPORT_TO=ReportingType.SWANLAB)

        assert training_config.EPOCHS == 10
        assert dataset_config.EVAL_SPLIT_RATIO == 0.1
        assert reporting_config.REPORT_TO == ReportingType.SWANLAB

    def test_invalid_combination(self):
        """Test that invalid combinations are caught."""
        with pytest.raises(ValueError):
            TrainingConfig(
                EPOCHS=10,
                FP16=True,
                BF16=True
            )
