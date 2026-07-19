"""
preprocess.py

Image preprocessing utilities shared by the training and inference pipelines.
Kept separate from src/dataset.py so both training code and standalone
scripts/notebooks can reuse the same transforms.
"""
from typing import Tuple

from PIL import Image, ImageOps


def load_image(path: str) -> Image.Image:
    """Load an image from disk and convert to RGB."""
    return Image.open(path).convert("RGB")


def resize_with_padding(image: Image.Image, target_size: Tuple[int, int] = (384, 384)) -> Image.Image:
    """
    Resize an image to fit within target_size while preserving aspect ratio,
    then pad with white to reach the exact target size.

    Note: TrOCRProcessor already handles resizing internally, so this is
    mainly useful for augmentation/visualization or for models (e.g. Donut)
    that expect a fixed input size prepared beforehand.
    """
    image = ImageOps.exif_transpose(image)
    image.thumbnail(target_size, Image.BICUBIC)

    padded = Image.new("RGB", target_size, (255, 255, 255))
    offset = ((target_size[0] - image.width) // 2, (target_size[1] - image.height) // 2)
    padded.paste(image, offset)
    return padded


def normalize_text(text: str) -> str:
    """Basic ground-truth text cleanup: strip whitespace, collapse spaces."""
    return " ".join(text.strip().split())
