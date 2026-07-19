"""
dataset.py

Custom PyTorch Dataset for OCR fine-tuning: loads images + ground-truth text,
applies the TrOCR processor for both image preprocessing and text tokenization.
"""
import logging
from typing import Dict, List, Optional

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import TrOCRProcessor

from data.preprocess import load_image, normalize_text

logger = logging.getLogger(__name__)


class OCRDataset(Dataset):
    """
    Loads (image_path, text) pairs from a CSV manifest and prepares them
    as model-ready tensors.

    Args:
        csv_path: path to a CSV with columns [image_path, text]
        processor: a TrOCRProcessor (handles both image + text preprocessing)
        max_target_length: max token length for the text labels
    """

    def __init__(self, csv_path: str, processor: TrOCRProcessor, max_target_length: int = 128):
        self.df = pd.read_csv(csv_path)
        self.processor = processor
        self.max_target_length = max_target_length

        if self.df.empty:
            raise ValueError(f"No samples found in {csv_path}")

        logger.info("Loaded %d samples from %s", len(self.df), csv_path)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]

        try:
            image = load_image(row["image_path"])
        except (FileNotFoundError, OSError) as e:
            logger.warning("Failed to load %s (%s); substituting blank image", row["image_path"], e)
            image = Image.new("RGB", (384, 384), (255, 255, 255))

        text = normalize_text(str(row["text"]))

        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze()

        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze()

        # Replace pad token id with -100 so it's ignored in the loss computation.
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {"pixel_values": pixel_values, "labels": labels}


def ocr_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Stack a list of samples into a single batch. Padding is already handled
    per-sample via max_length tokenization, so this is a simple stack."""
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    return {"pixel_values": pixel_values, "labels": labels}
