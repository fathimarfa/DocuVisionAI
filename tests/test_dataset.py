"""
test_dataset.py

Basic unit tests for OCRDataset. Uses a tiny synthetic CSV + generated images
so tests run fast without needing the real dataset downloaded.
"""
import csv
import os

import pytest
from PIL import Image
from transformers import TrOCRProcessor

from src.dataset import OCRDataset, ocr_collate_fn

MODEL_NAME = "microsoft/trocr-base-handwritten"


@pytest.fixture(scope="module")
def processor():
    return TrOCRProcessor.from_pretrained(MODEL_NAME)


@pytest.fixture
def tiny_csv(tmp_path):
    img_dir = tmp_path / "images"
    img_dir.mkdir()

    rows = []
    for i, text in enumerate(["hello world", "fine tuning ocr"]):
        img_path = img_dir / f"{i}.png"
        Image.new("RGB", (200, 64), (255, 255, 255)).save(img_path)
        rows.append({"image_path": str(img_path), "text": text})

    csv_path = tmp_path / "manifest.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "text"])
        writer.writeheader()
        writer.writerows(rows)

    return str(csv_path)


def test_dataset_length(tiny_csv, processor):
    dataset = OCRDataset(tiny_csv, processor)
    assert len(dataset) == 2


def test_dataset_getitem_shapes(tiny_csv, processor):
    dataset = OCRDataset(tiny_csv, processor, max_target_length=32)
    sample = dataset[0]
    assert "pixel_values" in sample and "labels" in sample
    assert sample["pixel_values"].ndim == 3          # C, H, W
    assert sample["labels"].shape[0] == 32


def test_dataset_missing_image_falls_back(tmp_path, processor):
    csv_path = tmp_path / "manifest.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "text"])
        writer.writeheader()
        writer.writerow({"image_path": str(tmp_path / "does_not_exist.png"), "text": "hi"})

    dataset = OCRDataset(str(csv_path), processor)
    sample = dataset[0]  # should not raise
    assert sample["pixel_values"] is not None


def test_collate_fn_batches(tiny_csv, processor):
    dataset = OCRDataset(tiny_csv, processor)
    batch = ocr_collate_fn([dataset[0], dataset[1]])
    assert batch["pixel_values"].shape[0] == 2
    assert batch["labels"].shape[0] == 2


def test_empty_csv_raises(tmp_path, processor):
    csv_path = tmp_path / "empty.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "text"])
        writer.writeheader()

    with pytest.raises(ValueError):
        OCRDataset(str(csv_path), processor)
