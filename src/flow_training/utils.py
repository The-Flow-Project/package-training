"""
Utils for the training class to make it cleaner.
"""

from transformers import TrOCRProcessor
import datasets
from dataclasses import dataclass
from typing import Any
import PIL
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
    Pads images and labels to the maximum length in the batch.
    """
    processor: TrOCRProcessor

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        # Ensure pixel_values are tensors and stack them
        pixel_values_list = [f["pixel_values"] for f in features]
        # Convert to tensor if not already
        pixel_values_list = [
            pv if isinstance(pv, torch.Tensor) else torch.tensor(pv, dtype=torch.float32)
            for pv in pixel_values_list
        ]
        pixel_values = torch.stack(pixel_values_list)

        labels = [f["labels"] for f in features]
        max_label_length = max(len(label) for label in labels)

        padded_labels = []
        for label in labels:
            padding_length = max_label_length - len(label)
            # Convert label to list if it's a tensor
            if isinstance(label, torch.Tensor):
                label = label.tolist()
            padded_label = list(label) + [-100] * padding_length
            padded_labels.append(padded_label)

        labels = torch.tensor(padded_labels, dtype=torch.long)

        return {
            "pixel_values": pixel_values,
            "labels": labels,
        }
