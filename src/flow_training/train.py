"""
Flow Training - Main Training Module

This module provides training functionalities to train TrOCR models.
Input: Preprocessed Dataset from Huggingface Datasets containing line images and corresponding text labels.
Output: Fine-tuned TrOCR Model saved to specified directory or uploaded to Huggingface Hub.
"""

from __future__ import annotations

import os
import shutil
from loguru import logger
from pathlib import Path
from typing import TYPE_CHECKING, Any
from PIL import Image, UnidentifiedImageError

from flow_training.logging_config import setup_logger
from flow_training.utils import seed_everything, TrOCRDataCollator
from flow_training.config import (
    TrainingConfig,
    DatasetConfig,
    ModelConfig,
    ReportingConfig,
)
from flow_training.enums import ReportingType

# Type hints only - heavy imports will be done in __init__
if TYPE_CHECKING:
    from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, GenerationConfig
    from transformers.trainer_utils import EvalPrediction

setup_logger()  # default to DEBUG level


# ===============================================================================
# TRAINER CLASS
# ===============================================================================


class Trainer:
    """
    Trainer class for fine-tuning TrOCR models.
    """

    def __init__(
            self,
            training_config: TrainingConfig,
            dataset_config: DatasetConfig,
            model_config: ModelConfig | None = None,
            reporting_config: ReportingConfig | None = None,
    ):
        """
        Initialize the Trainer with dataset and configuration.

        This orchestrates the initialization process by delegating to specialized methods:
        - Config setup and validation
        - Environment and device setup
        - Model and processor loading
        - Dataset preparation and preprocessing
        - Trainer initialization

        Args:
            training_config: Training configuration - all training parameters/arguments.
            dataset_config: Dataset configuration - dataset source and output.
            model_config: Model configuration - used models for training. Defaults to ModelConfig().
            reporting_config: Reporting configuration - parameters to set up reporting/tracking.
                Defaults to ReportingConfig().
        """
        # Import heavy-weight dependencies here for lazy loading
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        from evaluate import load

        # Store config references
        self.training_config: TrainingConfig = training_config
        self.dataset_config: DatasetConfig = dataset_config
        self.model_config: ModelConfig = model_config if model_config is not None else ModelConfig()
        self.reporting_config: ReportingConfig = (
            reporting_config if reporting_config is not None else ReportingConfig()
        )

        # Lazy-load these for later use
        self.torch = torch
        self.TrOCRProcessor = TrOCRProcessor
        self.VisionEncoderDecoderModel = VisionEncoderDecoderModel
        self.cer_metric_loader = load

        # Initialize trainer
        self.trainer: Seq2SeqTrainer | None = None

        # Initialize in phases
        self._initialize_configs()
        self._setup_environment()
        self._load_model_and_processor()
        self._prepare_and_preprocess_datasets()
        self._initialize_trainer()

    def _initialize_configs(self) -> None:
        """
        Initialize and validate configurations.

        Sets up configuration objects and initializes reporting (Swanlab).
        """
        if self.reporting_config.REPORT_TO == ReportingType.SWANLAB:
            self._init_swanlab()
        logger.info("Initialized Trainer with provided configurations.")

    def _setup_environment(self) -> None:
        """
        Setup environment and reproducibility settings.

        Initializes metrics loading and sets random seeds for reproducibility.
        """
        self.max_seq_length: int | None = None
        self.cer_metric = self.cer_metric_loader("cer")

        seed_everything(self.training_config.SEED)
        logger.info(f"Set seed to {self.training_config.SEED}")

    def _load_model_and_processor(self) -> None:
        """
        Load TrOCR model and processor from pretrained weights.

        Loads model and processor from Hugging Face Hub, moves model to appropriate device
        (GPU or CPU), and configures floating point settings.
        """
        self.processor = self.TrOCRProcessor.from_pretrained(self.model_config.BASE_PROCESSOR_NAME)
        self.model = self.VisionEncoderDecoderModel.from_pretrained(
            self.model_config.BASE_MODEL_NAME
        )
        logger.info(f"Loaded model and processor from {self.model_config.BASE_MODEL_NAME}")

        self.device = self.torch.device("cuda" if self.torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.torch.backends.cuda.matmul.allow_tf32 = self.training_config.TF32
        logger.info(f"Model moved to device: {self.device}")

    def _prepare_and_preprocess_datasets(self) -> None:
        """
        Prepare datasets and apply preprocessing.

        Loads and filters datasets, computes optimal sequence length, and applies
        tokenization and image preprocessing to train and evaluation sets.
        """
        # Prepare output directory
        if self.training_config.OVERWRITE_OUTPUT_DIR and os.path.exists(
                self.training_config.OUTPUT_DIR
        ):
            shutil.rmtree(self.training_config.OUTPUT_DIR)
            logger.debug(f"Output directory removed: {self.training_config.OUTPUT_DIR}")
            os.mkdir(self.training_config.OUTPUT_DIR)

        # Load and filter datasets
        self._load_prepare_datasets()

        # Compute and set maximum sequence length
        self.max_seq_length = self._compute_max_length()
        self.processor.tokenizer.model_max_length = self.max_seq_length
        self.model.generation_config.max_length = self.max_seq_length
        logger.info(f"Set maximum sequence length to: {self.max_seq_length}")

        # Preprocess datasets
        self.train_dataset = self.train_dataset.map(
            self._preprocess,
            batched=False,
            remove_columns=self.train_dataset.column_names,
        )
        self.eval_dataset = self.eval_dataset.map(
            self._preprocess,
            batched=False,
            remove_columns=self.eval_dataset.column_names,
        )
        logger.info(f"Train dataset size: {len(self.train_dataset)}")
        logger.info(f"Eval dataset size: {len(self.eval_dataset)}")

    def _initialize_trainer(self) -> None:
        """
        Initialize the Seq2SeqTrainer.

        Sets up special tokens, generation config, and creates the trainer with all
        datasets, arguments, and data collator.
        """
        from transformers import Seq2SeqTrainer

        self._set_special_tokens()
        self.model.generation_config = self._create_generation_config()

        self.trainer = Seq2SeqTrainer(
            model=self.model,
            processing_class=self.processor.image_processor,
            compute_metrics=self._compute_metrics,
            args=self._create_training_arguments(),
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=TrOCRDataCollator(self.processor),
        )
        logger.info("Seq2SeqTrainer initialized.")

    def _init_swanlab(self) -> None:
        """
        Initialize Swanlab for experiment tracking.

        Swanlab is used to track and visualize training metrics.
        Uses configuration from ReportingConfig and TrainingConfig.

        Raises:
            ImportError: If swanlab is not installed.
            RuntimeError: If swanlab initialization fails.
        """
        try:
            import swanlab
        except ImportError as e:
            logger.error("Swanlab not installed. Install with: pip install swanlab")
            raise ImportError("swanlab is required when REPORT_TO='swanlab'") from e

        try:
            # Use the parent directory of LOGGING_DIR as the base log directory
            log_dir = Path(self.training_config.LOGGING_DIR).parent
            log_dir.mkdir(parents=True, exist_ok=True)

            swanlab.init(
                logdir=str(log_dir),
                mode="local",
                project_name=self.reporting_config.REPORT_PROJECTNAME,
                workspace=self.reporting_config.PROJECT_WORKSPACE,
            )
            logger.info(f"✓ Swanlab initialized at {log_dir}")
        except Exception as e:
            logger.error(f"Swanlab initialization failed: {e}")
            raise RuntimeError(f"Failed to initialize Swanlab: {e}") from e

    def train(self) -> None:
        """
        Train the TrOCR model.

        This is a blocking operation that should be called from an async context
        via asyncio.to_thread() or similar to avoid blocking the event loop.

        Example:
            FastAPI usage:
            await asyncio.to_thread(trainer.train)

        Raises:
            KeyboardInterrupt: If training is interrupted by user.
            RuntimeError: If GPU out of memory or training fails.
            OSError: If model/processor save fails.
        """
        try:
            logger.info("Starting training.")
            self.trainer.train(
                resume_from_checkpoint=self.training_config.OUTPUT_DIR
                if self.training_config.RESUME_FROM_CHECKPOINT
                else None,
            )
        except KeyboardInterrupt:
            logger.warning("Training interrupted by user")
            raise
        except self.torch.cuda.OutOfMemoryError as e:
            logger.error(f"GPU out of memory: {e}")
            raise RuntimeError("Not enough GPU memory for training") from e
        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            raise

        try:
            self.processor.save_pretrained(self.training_config.OUTPUT_DIR)
            self.model.save_pretrained(self.training_config.OUTPUT_DIR)
            logger.info(
                f"Training completed. Model and processor saved to {self.training_config.OUTPUT_DIR}."
            )
        except OSError as e:
            logger.error(f"Failed to save model: {e}")
            raise

    def _filter_by_height(self, example: dict[str, Any]) -> bool:
        """
        Filter function to remove images below a certain height.

        Args:
            example: A single entry from the dataset with 'image' field.

        Returns:
            True if the image height is above the minimum threshold, False otherwise.

        Raises:
            KeyError: If 'image' field is missing from example.
            UnidentifiedImageError: If image format is invalid.
        """
        try:
            img = example["image"]
            image = self._convert_to_rgb(img)
            return image.height >= self.dataset_config.MIN_LINE_HEIGHT
        except KeyError as e:
            logger.error(f"Missing 'image' field in dataset example: {e}")
            raise
        except UnidentifiedImageError as e:
            logger.error(f"Invalid image format: {e}")
            raise

    def _preprocess(self, example: dict[str, Any]) -> dict[str, Any]:
        """
        Preprocess single example.

        Args:
            example: A single entry from the dataset with 'image' and 'text' fields.

        Returns:
            Preprocessed example with 'pixel_values' and 'labels' tensors.

        Raises:
            KeyError: If required fields are missing from example.
            UnidentifiedImageError: If image format is invalid.
        """
        try:
            img = example["image"]
            image = self._convert_to_rgb(img)

            pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze(0)
            labels = self.processor.tokenizer(
                example["text"],
                padding="max_length",  # Pad to max_length for uniform tensor shapes
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="pt",
            ).input_ids.squeeze(0)

            return {
                "pixel_values": pixel_values,
                "labels": labels,
            }
        except KeyError as e:
            logger.error(f"Missing required field in dataset example: {e}")
            raise ValueError(f"Dataset entry missing field '{e.args[0]}'") from e
        except (FileNotFoundError, UnidentifiedImageError) as e:
            logger.error(f"Image processing failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error preprocessing example: {e}")
            raise

    @staticmethod
    def _convert_to_rgb(img) -> Image.Image:
        """
        Convert image to a PIL Image in RGB mode.

        Args:
            img: Either a PIL Image object or a path (str/Path) to an image file.

        Returns:
            PIL Image in RGB mode.

        Raises:
            FileNotFoundError: If img is a path that doesn't exist.
            UnidentifiedImageError: If file is not a valid image format.
            TypeError: If img is neither PIL Image nor valid path.
        """
        try:
            if hasattr(img, "convert"):
                # It's already a PIL Image
                return img.convert("RGB")
            else:
                # Assume it's a file path
                return Image.open(img).convert("RGB")
        except FileNotFoundError as e:
            logger.error(f"Image file not found: {img} - {e}")
            raise
        except UnidentifiedImageError as e:
            logger.error(f"File is not a valid image format: {img} - {e}")
            raise

    def _compute_max_length(self) -> int:
        """
        Compute the maximum sequence length for the tokenizer/model.

        Uses the 99th percentile of text lengths with a 1.2x safety buffer.
        This handles 99% of training examples while reserving space for longer
        generated sequences. These values are optimized for performance and not exposed
        to users, as they require deep understanding of NLP preprocessing.

        Returns:
            Maximum sequence length as integer.
        """
        import numpy as np

        lengths = [len(text) for text in self.train_dataset["text"]]
        max_length_99 = int(np.percentile(lengths, 99))
        max_length = int(max_length_99 * 1.2)  # Add 20% buffer for generated sequences

        logger.info(
            f"Computed max_seq_length: {max_length} "
            f"(99th percentile: {max_length_99}, buffer: 1.2x)"
        )
        return max_length

    def _load_prepare_datasets(self) -> None:
        """
        Load and prepare the training and evaluation datasets.

        Loads datasets from HuggingFace Hub, filters by image height,
        and handles train/test split if no separate evaluation dataset exists.

        Raises:
            DatasetNotFoundError: If dataset not found on HuggingFace Hub.
            RepositoryNotFoundError: If model/processor not found.
        """
        from datasets import load_dataset, get_dataset_split_names

        self.train_dataset = load_dataset(
            self.dataset_config.HUGGINGFACE_DATASET_SOURCE, split="train"
        )
        self.train_dataset = self.train_dataset.filter(self._filter_by_height)
        split_names = get_dataset_split_names(self.dataset_config.HUGGINGFACE_DATASET_SOURCE)
        self.eval_dataset = None
        if self.dataset_config.HUGGINGFACE_EVAL_SPLIT_NAME in split_names:
            self.eval_dataset = load_dataset(
                self.dataset_config.HUGGINGFACE_DATASET_SOURCE,
                split=self.dataset_config.HUGGINGFACE_EVAL_SPLIT_NAME,
            )
            self.eval_dataset = self.eval_dataset.filter(self._filter_by_height)
            logger.info(
                f"Loaded evaluation split: {self.dataset_config.HUGGINGFACE_EVAL_SPLIT_NAME}"
            )
            logger.debug("Ignored EVAL_SPLIT_RATIO since eval split is provided in the dataset.")

        if not self.eval_dataset:
            datasetdict = self.train_dataset.shuffle(
                seed=self.training_config.SEED,
            ).train_test_split(
                test_size=self.dataset_config.EVAL_SPLIT_RATIO,
                seed=self.training_config.SEED,
            )
            self.train_dataset = datasetdict["train"]
            self.eval_dataset = datasetdict["test"]
            logger.info(
                f"Split train dataset into train and eval with ratio {self.dataset_config.EVAL_SPLIT_RATIO}"
            )

    def _create_generation_config(self) -> GenerationConfig:
        """
        Create generation configuration for the model.

        Sets up the GenerationConfig with special tokens, beam search parameters,
        and other settings for text generation during evaluation.

        Returns:
            GenerationConfig: Configuration object for model generation.

        Raises:
            AttributeError: If required tokenizer attributes not found.
        """
        from transformers import GenerationConfig

        gen_config = GenerationConfig(
            decoder_start_token_id=self.processor.tokenizer.cls_token_id,
            eos_token_id=self.processor.tokenizer.sep_token_id,
            forced_eos_token_id=self.processor.tokenizer.sep_token_id,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            max_length=self.max_seq_length,
            early_stopping=self.training_config.EARLY_STOPPING,
            length_penalty=2.0,
            num_beams=self.training_config.NUM_BEAMS,
            use_cache=True,
        )
        logger.info(f"Created GenerationConfig: {gen_config}")
        return gen_config

    def _set_special_tokens(self) -> None:
        """
        Set special tokens used for creating the decoder_input_ids from the labels.

        Configures the model with special tokens for sequence-to-sequence generation:
        - decoder_start_token_id: Token to start generation
        - eos_token_id: End of sequence token
        - pad_token_id: Padding token
        - vocab_size: Total vocabulary size

        Returns:
            None
        """
        self.model.config.decoder_start_token_id = self.processor.tokenizer.cls_token_id
        self.model.config.eos_token_id = self.processor.tokenizer.sep_token_id
        # self.model.config.forced_eos_token_id = self.processor.tokenizer.sep_token_id
        self.model.config.pad_token_id = self.processor.tokenizer.pad_token_id
        self.model.config.vocab_size = self.model.config.decoder.vocab_size
        logger.info("Set special tokens in model configuration.")

    def _create_training_arguments(self) -> Seq2SeqTrainingArguments:
        """
        Create training arguments for the Seq2SeqTrainer.

        Builds comprehensive training arguments from configuration objects including
        training hyperparameters, evaluation settings, logging configuration, and
        model-specific generation parameters.

        Returns:
            Seq2SeqTrainingArguments: Configured training arguments for the trainer.
        """
        from transformers import Seq2SeqTrainingArguments

        training_args = Seq2SeqTrainingArguments(
            output_dir=self.training_config.OUTPUT_DIR,
            seed=self.training_config.SEED,
            do_train=self.training_config.DO_TRAIN,
            do_eval=self.training_config.DO_EVAL,
            do_predict=self.training_config.DO_PREDICT,
            auto_find_batch_size=self.training_config.AUTO_FIND_BATCH_SIZE,
            per_device_train_batch_size=self.training_config.BATCH_SIZE,
            per_device_eval_batch_size=self.training_config.BATCH_SIZE,
            learning_rate=self.training_config.LEARNING_RATE,
            lr_scheduler_type=self.training_config.LR_SCHEDULER_TYPE,
            weight_decay=self.training_config.WEIGHT_DECAY,
            num_train_epochs=self.training_config.EPOCHS,
            tf32=self.training_config.TF32,
            fp16=self.training_config.FP16,
            bf16=self.training_config.BF16,
            eval_strategy=self.training_config.EVAL_STRATEGY,
            eval_steps=self.training_config.EVAL_STEPS,
            eval_accumulation_steps=self.training_config.EVAL_ACCUMULATION_STEPS,
            save_strategy=self.training_config.SAVE_STRATEGY,
            save_steps=self.training_config.SAVE_STEPS,
            save_total_limit=self.training_config.SAVE_TOTAL_LIMIT,
            logging_strategy=self.training_config.LOGGING_STRATEGY,
            logging_dir=self.training_config.LOGGING_DIR,
            logging_first_step=self.training_config.LOGGING_FIRST_STEP,
            logging_steps=self.training_config.LOGGING_STEPS,
            gradient_accumulation_steps=self.training_config.GRADIENT_ACCUMULATION_STEPS,
            generation_num_beams=self.training_config.GENERATION_NUM_BEAMS,
            generation_max_length=self.max_seq_length,
            gradient_checkpointing=self.training_config.GRADIENT_CHECKPOINTING,
            disable_tqdm=self.training_config.DISABLE_TQDM,
            warmup_ratio=self.training_config.WARMUP_RATIO,
            predict_with_generate=self.training_config.PREDICT_WITH_GENERATE,
            load_best_model_at_end=self.training_config.LOAD_BEST_MODEL_AT_END,
            metric_for_best_model=self.training_config.METRIC_FOR_BEST_MODEL,
            greater_is_better=self.training_config.GREATER_IS_BETTER,
            report_to=self.reporting_config.REPORT_TO,
            run_name=self.training_config.RUN_NAME,
        )
        logger.info(f"Created Seq2SeqTrainingArguments: {training_args}")
        return training_args

    def _compute_metrics(self, pred: EvalPrediction) -> dict[str, float]:
        """
        Compute Character Error Rate (CER) metric.

        Args:
            pred: EvalPrediction object from the Trainer containing predictions and label_ids.

        Returns:
            Dictionary with CER metric score.
        """
        labels_ids = pred.label_ids
        labels_ids[labels_ids == -100] = self.processor.tokenizer.pad_token_id

        pred_logits = pred.predictions
        pred_logits[pred_logits == -100] = self.processor.tokenizer.pad_token_id
        if hasattr(pred_logits, "shape") and len(pred_logits.shape) == 3:
            pred_ids = pred_logits.argmax(-1)
        else:
            pred_ids = pred_logits

        # Dekodierte Strings prüfen
        pred_str = self.processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = self.processor.batch_decode(labels_ids, skip_special_tokens=True)

        logger.debug("=== PREDICTION INSIGHT ===")
        logger.debug(f"First 3 predictions: {pred_str[:3]}")
        logger.debug(f"First 3 references: {label_str[:3]}")

        pred_str = [pred.strip() for pred in pred_str]
        label_str = [lbl.strip() for lbl in label_str]

        cer = self.cer_metric.compute(predictions=pred_str, references=label_str)
        logger.debug(f"CER: {cer}")
        return {"cer": cer}
