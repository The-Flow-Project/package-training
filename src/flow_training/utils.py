"""
Utils for the training class to make it cleaner.
"""

from transformers import TrOCRProcessor
from dataclasses import dataclass
from typing import Any
import numpy as np
import torch


##################################
# SEED for reproduction
##################################
def seed_everything(seed_value) -> None:
    """
    Seed everything for reproducibility.
    :param seed_value:
    """
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


##################################
# TrOCRDataCollator
###################################


@dataclass
class TrOCRDataCollator:
    """
    Data collator for TrOCR model training.

    Stacks pixel values and pads labels to the maximum length in the batch.
    Uses -100 as ignore index, which is standard in Hugging Face transformers
    for tokens that should be excluded from loss computation.

    Args:
        processor: TrOCRProcessor for handling images and text.
    """

    processor: TrOCRProcessor

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """
        Collate batch of features for TrOCR training.

        Stacks pixel values and pads labels to maximum length in batch,
        using -100 as ignore index for loss computation.

        Args:
            features: List of examples with 'pixel_values' and 'labels' tensors.

        Returns:
            Batch dictionary with stacked pixel_values and padded labels.

        Raises:
            ValueError: If features list is empty or missing required keys.
        """
        # Validate input
        if not features:
            raise ValueError("Features list cannot be empty")

        if not all("pixel_values" in f and "labels" in f for f in features):
            raise ValueError("All features must contain 'pixel_values' and 'labels' keys")

        # Stack pixel values
        pixel_values_list = [f["pixel_values"] for f in features]
        # Convert to tensor if not already
        pixel_values_list = [
            pv if isinstance(pv, torch.Tensor) else torch.tensor(pv, dtype=torch.float32)
            for pv in pixel_values_list
        ]
        pixel_values = torch.stack(pixel_values_list)

        # Pad labels to max length
        labels = [f["labels"] for f in features]
        max_label_length = max(len(label) for label in labels)

        padded_labels = []
        for label in labels:
            padding_length = max_label_length - len(label)
            # Convert label to list if it's a tensor
            if isinstance(label, torch.Tensor):
                label = label.tolist()
            # Pad with -100 (ignore index - excluded from loss computation)
            padded_label = list(label) + [-100] * padding_length
            padded_labels.append(padded_label)

        labels = torch.tensor(padded_labels, dtype=torch.long)

        return {
            "pixel_values": pixel_values,
            "labels": labels,
        }
