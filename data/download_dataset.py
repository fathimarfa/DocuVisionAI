"""
download_dataset.py

Downloads and caches the handwriting recognition dataset from Hugging Face,
then writes train/val/test CSV manifests (image_path, text) into data/processed/.

Usage:
    python data/download_dataset.py --config configs/trocr_base.yaml
"""
import argparse
import logging
import os
from pathlib import Path

import yaml
from datasets import load_dataset
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def download_and_split(cfg: dict) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading dataset: %s", cfg["dataset_name"])
    ds = load_dataset(cfg["dataset_name"], split="train", cache_dir=str(RAW_DIR))

    if cfg.get("subset_size"):
        ds = ds.shuffle(seed=cfg["seed"]).select(range(min(cfg["subset_size"], len(ds))))
        logger.info("Using subset of %d samples", len(ds))

    # Save raw images to disk and build (path, text) pairs.
    images_dir = RAW_DIR / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for i, sample in enumerate(ds):
        img = sample["image"]
        text = sample["text"] if "text" in sample else sample.get("label", "")
        img_path = images_dir / f"{i:07d}.png"
        if not img_path.exists():
            img.convert("RGB").save(img_path)
        records.append({"image_path": str(img_path), "text": text})

    logger.info("Total records: %d", len(records))

    train_recs, temp_recs = train_test_split(
        records, train_size=cfg["train_split"], random_state=cfg["seed"]
    )
    rel_val = cfg["val_split"] / (cfg["val_split"] + cfg["test_split"])
    val_recs, test_recs = train_test_split(
        temp_recs, train_size=rel_val, random_state=cfg["seed"]
    )

    import pandas as pd

    for name, recs in [("train", train_recs), ("val", val_recs), ("test", test_recs)]:
        df = pd.DataFrame(recs)
        out_path = PROCESSED_DIR / f"{name}.csv"
        df.to_csv(out_path, index=False)
        logger.info("Wrote %s (%d rows) -> %s", name, len(df), out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/trocr_base.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    download_and_split(config)
