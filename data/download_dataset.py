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


def _extract_records(ds, images_dir: Path, id_prefix: str) -> list:
    records = []
    for i, sample in enumerate(ds):
        img = sample["image"]
        text = sample.get("text", sample.get("label", ""))
        img_path = images_dir / f"{id_prefix}_{i:07d}.png"
        if not img_path.exists():
            img.convert("RGB").save(img_path)
        records.append({"image_path": str(img_path), "text": text})
    return records


def download_and_split(cfg: dict) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    images_dir = RAW_DIR / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading dataset: %s", cfg["dataset_name"])

    # Prefer the dataset's own predefined splits (avoids re-splitting and
    # sidesteps datasets' automatic (and sometimes buggy) feature inference
    # that occurs when only a single 'train' split is requested).
    from datasets import get_dataset_split_names

    available_splits = get_dataset_split_names(cfg["dataset_name"])
    logger.info("Available splits: %s", available_splits)

    split_map = {"train": "train", "val": None, "test": None}
    for candidate in ["validation", "val", "dev"]:
        if candidate in available_splits:
            split_map["val"] = candidate
            break
    for candidate in ["test"]:
        if candidate in available_splits:
            split_map["test"] = candidate
            break

    import pandas as pd

    if split_map["val"] and split_map["test"]:
        # Clean path: dataset already has train/val/test.
        for out_name, hf_split in [("train", "train"), ("val", split_map["val"]), ("test", split_map["test"])]:
            ds = load_dataset(cfg["dataset_name"], split=hf_split, cache_dir=str(RAW_DIR))
            if cfg.get("subset_size") and out_name == "train":
                ds = ds.shuffle(seed=cfg["seed"]).select(range(min(cfg["subset_size"], len(ds))))
            elif cfg.get("subset_size"):
                frac = cfg["subset_size"] / max(cfg.get("subset_size"), 1)
                n = max(int(len(ds) * 0.1), 50)  # keep val/test proportionally small
                ds = ds.shuffle(seed=cfg["seed"]).select(range(min(n, len(ds))))

            records = _extract_records(ds, images_dir, out_name)
            df = pd.DataFrame(records)
            out_path = PROCESSED_DIR / f"{out_name}.csv"
            df.to_csv(out_path, index=False)
            logger.info("Wrote %s (%d rows) -> %s", out_name, len(df), out_path)
    else:
        # Fallback: only a 'train' split exists, so split it ourselves.
        ds = load_dataset(cfg["dataset_name"], split="train", cache_dir=str(RAW_DIR))
        if cfg.get("subset_size"):
            ds = ds.shuffle(seed=cfg["seed"]).select(range(min(cfg["subset_size"], len(ds))))
            logger.info("Using subset of %d samples", len(ds))

        records = _extract_records(ds, images_dir, "all")
        logger.info("Total records: %d", len(records))

        train_recs, temp_recs = train_test_split(records, train_size=cfg["train_split"], random_state=cfg["seed"])
        rel_val = cfg["val_split"] / (cfg["val_split"] + cfg["test_split"])
        val_recs, test_recs = train_test_split(temp_recs, train_size=rel_val, random_state=cfg["seed"])

        for out_name, recs in [("train", train_recs), ("val", val_recs), ("test", test_recs)]:
            df = pd.DataFrame(recs)
            out_path = PROCESSED_DIR / f"{out_name}.csv"
            df.to_csv(out_path, index=False)
            logger.info("Wrote %s (%d rows) -> %s", out_name, len(df), out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/trocr_base.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    download_and_split(config)