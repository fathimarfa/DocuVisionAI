"""
train.py

Fine-tunes TrOCR on a custom OCR dataset using the Hugging Face Seq2SeqTrainer,
with mixed precision, gradient accumulation, LR scheduling, checkpointing and
early stopping, as configured via a YAML file.

Usage:
    python src/train.py --config configs/trocr_base.yaml
"""
import argparse
import logging
import random

import numpy as np
import torch
import yaml

import functools
torch.load = functools.partial(torch.load, weights_only=False)

from transformers import (
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
)

from src.dataset import OCRDataset, ocr_collate_fn
from src.evaluate import build_compute_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model(model_name: str) -> VisionEncoderDecoderModel:
    logger.info("Loading pre-trained model: %s", model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    return model


def configure_model_for_training(model: VisionEncoderDecoderModel, processor: TrOCRProcessor, cfg: dict):
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_length = cfg["max_target_length"]
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4
    return model


def setup_training(cfg: dict) -> Seq2SeqTrainingArguments:
    return Seq2SeqTrainingArguments(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        fp16=cfg["fp16"] and torch.cuda.is_available(),
        learning_rate=cfg["learning_rate"],
        num_train_epochs=cfg["num_train_epochs"],
        warmup_steps=cfg["warmup_steps"],
        weight_decay=cfg["weight_decay"],
        logging_steps=cfg["logging_steps"],
        eval_strategy="steps",
        eval_steps=cfg["eval_steps"],
        save_steps=cfg["save_steps"],
        save_total_limit=cfg["save_total_limit"],
        load_best_model_at_end=cfg["load_best_model_at_end"],
        metric_for_best_model=cfg["metric_for_best_model"],
        greater_is_better=cfg["greater_is_better"],
        predict_with_generate=cfg["predict_with_generate"],
        report_to=["tensorboard"],
        seed=cfg["seed"],
    )


def train(cfg: dict) -> None:
    set_seed(cfg["seed"])

    processor = TrOCRProcessor.from_pretrained(cfg["model_name"])
    model = load_model(cfg["model_name"])
    model = configure_model_for_training(model, processor, cfg)

    train_dataset = OCRDataset("data/processed/train.csv", processor, cfg["max_target_length"])
    val_dataset = OCRDataset("data/processed/val.csv", processor, cfg["max_target_length"])

    training_args = setup_training(cfg)
    compute_metrics = build_compute_metrics(processor)

    trainer = Seq2SeqTrainer(
        model=model,
        processing_class=processor,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=ocr_collate_fn,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg["early_stopping_patience"])],
    )

    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=cfg.get("resume_from_checkpoint"))

    save_checkpoint(trainer, processor, f"{cfg['output_dir']}/checkpoint-final")
    logger.info("Training complete.")


def save_checkpoint(trainer: Seq2SeqTrainer, processor: TrOCRProcessor, path: str) -> None:
    trainer.save_model(path)
    processor.save_pretrained(path)
    logger.info("Saved checkpoint to %s", path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/trocr_base.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    train(config)