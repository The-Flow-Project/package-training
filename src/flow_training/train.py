"""
Flow Training - Main Training Module

This module provides training functionalities to tran TrOCR models.
Input: Preprocessed Dataset from Huggingface Datasets containing line images and corresponding text labels.
Output: Fine-tuned TrOCR Model saved to specified directory or uploaded to Huggingface Hub.
"""

from __future__ import annotations

import os
import shutil
from loguru import logger
from typing import TYPE_CHECKING

from flow_training.logging_config import setup_logger
from flow_training.utils import seed_everything
from flow_training.config import (
    TrainingConfig,
    DatasetConfig,
    ModelConfig,
    ReportingConfig,
)

# Type hints only - heavy imports will be done in __init__
if TYPE_CHECKING:
    from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, GenerationConfig

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
            model_config: ModelConfig | None = None,  # Default using TrOCR Large Handwritten
            reporting_config: ReportingConfig | None = None,  # Default using SWANLAB
    ):
        """
        Initialize the Trainer with dataset and configuration.
        :param training_config: Training configuration - all training parameters/arguments.
        :param dataset_config: Dataset configuration - dataset source and output.
        :param model_config: Model configuration - used models for training.
        :param reporting_config: Reporting configuration - parameters to set up the reporting/tracking.

        :return: None
        """
        # Import heavy-weight dependencies here for lazy loading
        import torch
        from transformers import (
            TrOCRProcessor,
            VisionEncoderDecoderModel,
            Seq2SeqTrainer,
            default_data_collator,
        )
        from evaluate import load

        self.trainer: Seq2SeqTrainer | None = None
        self.training_config: TrainingConfig = training_config
        self.dataset_config: DatasetConfig = dataset_config
        self.model_config: ModelConfig = model_config if model_config is not None else ModelConfig()
        self.reporting_config: ReportingConfig = (
            reporting_config if reporting_config is not None else ReportingConfig()
        )
        if self.reporting_config.REPORT_TO == "swanlab":
            import swanlab
            # use swanlab watch .logs to track the experiments
            # maybe later self-docker-hosted: https://docs.swanlab.cn/en/guide_cloud/self_host/docker-deploy.html
            swanlab.init(
                logdir=".logs",
                mode="local"
            )
        logger.info("Initialized Trainer with provided configurations.")

        self.max_seq_length: int | None = None  # Will be set after loading the dataset
        self.cer_metric = load("cer")

        ##################################
        # SEED for reproducibility
        ##################################
        seed_everything(self.training_config.SEED)
        logger.info(f"Set seed to {self.training_config.SEED}")

        ##################################
        # Model and Processor Initialization
        ##################################
        self.processor = TrOCRProcessor.from_pretrained(self.model_config.BASE_PROCESSOR_NAME)
        self.model = VisionEncoderDecoderModel.from_pretrained(self.model_config.BASE_MODEL_NAME)
        logger.info(f"Loaded model and processor from {self.model_config.BASE_MODEL_NAME}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        torch.backends.cuda.matmul.allow_tf32 = self.training_config.TF32
        logger.info(f"Model moved to device: {device}")

        ##################################
        # Dataset preparing/loading
        ##################################
        if self.training_config.OVERWRITE_OUTPUT_DIR and os.path.exists(
                self.training_config.OUTPUT_DIR
        ):
            shutil.rmtree(self.training_config.OUTPUT_DIR)
            logger.debug(f"Output directory removed: {self.training_config.OUTPUT_DIR}")
            os.mkdir(self.training_config.OUTPUT_DIR)

        self._load_prepare_datasets()

        # Get maximum sequence length from training dataset
        self.max_seq_length = self._compute_max_length()
        self.processor.tokenizer.model_max_length = self.max_seq_length
        self.model.generation_config.max_length = self.max_seq_length
        logger.info(f"Set maximum sequence length to: {self.max_seq_length}")

        ##################################
        # Dataset preprocess mapping
        ##################################
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

        ##################################
        # Model configuration adjustments
        ##################################
        self._set_special_tokens()
        self.model.generation_config = self._create_generation_config()

        self.trainer = Seq2SeqTrainer(
            model=self.model,
            processing_class=self.processor.image_processor,
            compute_metrics=self._compute_metrics,
            args=self._create_training_arguments(),
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=default_data_collator,
        )
        logger.info("Seq2SeqTrainer initialized.")

    async def train(self) -> None:
        """
        Method to start the training process.
        :return: None
        """
        try:
            logger.info(f"Starting training.")
            self.trainer.train(
                resume_from_checkpoint=TrainingConfig.OUTPUT_DIR
                if TrainingConfig.RESUME_FROM_CHECKPOINT
                else None,
            )
            self.processor.save_pretrained(TrainingConfig.OUTPUT_DIR)
            self.model.save_pretrained(TrainingConfig.OUTPUT_DIR)
            logger.info(
                f"Training completed. Model and processor saved to {TrainingConfig.OUTPUT_DIR}."
            )
        except Exception as e:
            logger.error(f"An error occurred during training: {e}")
            raise e

    def _filter_by_height(self, example) -> bool:
        """
        Filter function to remove images below a certain height.
        :param example: A single entry from the dataset.
        :return: True if the image height is above the minimum threshold, False otherwise.
        """
        import PIL
        img = example["image"]
        if type(img) in [PIL.PngImagePlugin.PngImageFile, PIL.JpegImagePlugin.JpegImageFile, PIL.Image.Image]:
            image = img.convert("RGB")
        else:
            image = PIL.Image.open(img).convert("RGB")
        return image.height >= self.dataset_config.MIN_LINE_HEIGHT

    def _preprocess(self, example) -> dict:
        """
        Preprocess single example.
        :param example: A single entry from the dataset.
        :return: Preprocessed example with pixel_values and labels.
        """
        import PIL
        img = example["image"]
        if type(img) in [PIL.PngImagePlugin.PngImageFile, PIL.JpegImagePlugin.JpegImageFile, PIL.Image.Image]:
            image = img.convert("RGB")
        else:
            image = PIL.Image.open(img).convert("RGB")

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

    def _compute_max_length(self) -> int:
        """
        Compute the maximum sequence length for the tokenizer/model.
        :return: Maximum sequence length.
        """
        import numpy as np
        lengths = [len(text) for text in self.train_dataset["text"]]
        max_length_99 = int(np.percentile(lengths, 99))
        return int(max_length_99 * 1.2)  # add some buffer to the max length

    def _load_prepare_datasets(self) -> None:
        """
        Load and prepare the training and evaluation datasets.
        :return:
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
        :return: GenerationConfig object.
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
        Set special tokens used for creating the decoder_input_ids from the labels
        :return: None
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
        :return: Seq2SeqTrainingArguments object.
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

    ######################################
    # Evaluation Metrics
    ######################################
    def _compute_metrics(self, pred) -> dict:
        """
        Compute Character Error Rate (CER) metric.
        :param pred: Prediction object from the Trainer containing predictions and label_ids.
        :return: Dictionary with CER metric.
        """
        logger.debug(f"=== DEBUGGING PREDICTIONS ===")
        logger.debug(f"Label IDs shape: {pred.label_ids.shape}")
        logger.debug(f"Label IDs sample: {pred.label_ids[0][:20]}")  # Erste 20 Tokens

        # Predictions-Shape prüfen
        if hasattr(pred.predictions, 'shape'):
            logger.debug(f"Predictions shape: {pred.predictions.shape}")
            logger.debug(f"Predictions sample: {pred.predictions[0][:5]}")  # Erste 5 Tokens
        else:
            logger.debug(f"Predictions type: {type(pred.predictions)}")

        labels_ids = pred.label_ids
        labels_ids[labels_ids == self.processor.tokenizer.pad_token_id] = -100

        pred_logits = pred.predictions
        if hasattr(pred_logits, "shape") and len(pred_logits.shape) == 3:
            pred_ids = pred_logits.argmax(-1)
        else:
            pred_ids = pred_logits

        # Dekodierte Strings prüfen
        pred_str = self.processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = self.processor.batch_decode(labels_ids, skip_special_tokens=True)

        logger.debug(f"First 3 predictions: {pred_str[:3]}")
        logger.debug(f"First 3 references: {label_str[:3]}")
        logger.debug(f"Prediction lengths: {[len(p) for p in pred_str[:5]]}")

        pred_str = [p.strip() for p in pred_str]
        label_str = [l.strip() for l in label_str]

        cer = self.cer_metric.compute(predictions=pred_str, references=label_str)
        logger.debug(f"CER: {cer}")
        return {"cer": cer}

        """
        logger.debug(f"Compute Character Error Rate (CER) metric.")
        logger.debug(f"Predictions: {pred}")
        labels_ids = pred.label_ids
        labels_ids[labels_ids == self.processor.tokenizer.pad_token_id] = -100

        pred_logits = pred.predictions
        if hasattr(pred_logits, "shape") and len(pred_logits.shape) == 3:
            pred_ids = pred_logits.argmax(-1)
        else:
            pred_ids = pred_logits

        pred_str = self.processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = self.processor.batch_decode(labels_ids, skip_special_tokens=True)

        # Clean up empty strings
        pred_str = [p.strip() for p in pred_str]
        label_str = [l.strip() for l in label_str]

        logger.debug(f"Predictions: {pred_str[:5]} ...")
        logger.debug(f"References: {label_str[:5]} ...")

        # Compute CER
        cer = self.cer_metric.compute(predictions=pred_str, references=label_str)
        logger.debug(f"CER: {cer}")
        return {"cer": cer}
        """
