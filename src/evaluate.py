"""
evaluate.py

Computes Character Error Rate (CER) and Word Error Rate (WER), compares the
fine-tuned model against the base (zero-shot) model, and prints a formatted
evaluation report matching the assignment's expected output format.

Usage:
    python src/evaluate.py --checkpoint weights/checkpoint-final --base-model microsoft/trocr-base-handwritten
"""
import argparse
import logging
import random
from collections import Counter
from typing import Callable, Dict, List, Tuple

import jiwer
import torch
from torch.utils.data import DataLoader
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from src.dataset import OCRDataset, ocr_collate_fn

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_cer(reference: str, hypothesis: str) -> float:
    """Character Error Rate = (S + D + I) / len(reference), via jiwer."""
    if len(reference) == 0:
        return 0.0 if len(hypothesis) == 0 else 1.0
    return jiwer.cer(reference, hypothesis)


def compute_wer(reference: str, hypothesis: str) -> float:
    """Word Error Rate = (S + D + I) / word_count(reference), via jiwer."""
    if len(reference.split()) == 0:
        return 0.0 if len(hypothesis.split()) == 0 else 1.0
    return jiwer.wer(reference, hypothesis)


def build_compute_metrics(processor: TrOCRProcessor) -> Callable:
    """Returns a compute_metrics fn compatible with Seq2SeqTrainer (used during training eval)."""

    def compute_metrics(pred) -> Dict[str, float]:
        labels_ids = pred.label_ids
        pred_ids = pred.predictions

        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
        label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

        cer = jiwer.cer(label_str, pred_str)
        wer = jiwer.wer(label_str, pred_str)
        return {"cer": cer, "wer": wer}

    return compute_metrics


@torch.no_grad()
def generate_predictions(
    model: VisionEncoderDecoderModel,
    processor: TrOCRProcessor,
    dataset: OCRDataset,
    device: str,
    batch_size: int = 8,
) -> Tuple[List[str], List[str]]:
    model.to(device).eval()
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=ocr_collate_fn)

    all_preds, all_refs = [], []
    for batch in loader:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].clone()
        labels[labels == -100] = processor.tokenizer.pad_token_id

        generated_ids = model.generate(pixel_values, max_length=128, num_beams=4)
        preds = processor.batch_decode(generated_ids, skip_special_tokens=True)
        refs = processor.batch_decode(labels, skip_special_tokens=True)

        all_preds.extend(preds)
        all_refs.extend(refs)

    return all_preds, all_refs


def most_confused_characters(refs: List[str], preds: List[str], top_k: int = 5) -> List[Tuple[str, str, int]]:
    """Rough character-substitution frequency count via aligned diffs (not full edit-distance backtrace)."""
    counter: Counter = Counter()
    for ref, pred in zip(refs, preds):
        for r_char, p_char in zip(ref, pred):
            if r_char != p_char:
                counter[(r_char, p_char)] += 1
    return [(r, p, c) for (r, p), c in counter.most_common(top_k)]


def print_evaluation_report(base_metrics: Dict, ft_metrics: Dict, refs: List[str], preds: List[str], n_samples: int) -> None:
    print("=" * 50)
    print("MODEL EVALUATION METRICS (TEST SET)")
    print("=" * 50)
    print(f"Total Samples Evaluated: {len(refs)}\n")

    print("Base Model (Zero-Shot) Performance:")
    print(f"  Character Error Rate (CER): {base_metrics['cer']:.4f}")
    print(f"  Word Error Rate (WER): {base_metrics['wer']:.4f}\n")

    print("Fine-Tuned Model Performance:")
    print(f"  Character Error Rate (CER): {ft_metrics['cer']:.4f}")
    print(f"  Word Error Rate (WER): {ft_metrics['wer']:.4f}\n")

    cer_reduction = (base_metrics["cer"] - ft_metrics["cer"]) / max(base_metrics["cer"], 1e-9) * 100
    wer_reduction = (base_metrics["wer"] - ft_metrics["wer"]) / max(base_metrics["wer"], 1e-9) * 100
    print("Improvement:")
    print(f"  CER Reduction: {cer_reduction:.1f}%")
    print(f"  WER Reduction: {wer_reduction:.1f}%")
    print("=" * 50)

    print(f"\nSample Output Validation ({min(n_samples, len(refs))} random examples):")
    sample_idxs = random.sample(range(len(refs)), min(n_samples, len(refs)))
    for i, idx in enumerate(sample_idxs, 1):
        ref, pred = refs[idx], preds[idx]
        cer = compute_cer(ref, pred)
        match = "SUCCESS" if cer == 0 else "PARTIAL"
        print(f'[{i}] Target: "{ref}"')
        print(f'    Predicted: "{pred}"')
        print(f"    CER: {cer:.3f} | Match: {match}\n")

    print("=" * 50)
    print("Error Analysis:")
    confused = most_confused_characters(refs, preds)
    print(f"  Most Confused Characters: {confused}")
    print("=" * 50)


def run_evaluation(checkpoint_path: str, base_model_name: str, test_csv: str, device: str) -> None:
    processor = TrOCRProcessor.from_pretrained(checkpoint_path)
    test_dataset = OCRDataset(test_csv, processor)

    logger.info("Evaluating base (zero-shot) model...")
    base_model = VisionEncoderDecoderModel.from_pretrained(base_model_name)
    base_model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    base_model.config.pad_token_id = processor.tokenizer.pad_token_id
    base_preds, base_refs = generate_predictions(base_model, processor, test_dataset, device)
    base_metrics = {"cer": jiwer.cer(base_refs, base_preds), "wer": jiwer.wer(base_refs, base_preds)}

    logger.info("Evaluating fine-tuned model...")
    ft_model = VisionEncoderDecoderModel.from_pretrained(checkpoint_path)
    ft_preds, ft_refs = generate_predictions(ft_model, processor, test_dataset, device)
    ft_metrics = {"cer": jiwer.cer(ft_refs, ft_preds), "wer": jiwer.wer(ft_refs, ft_preds)}

    print_evaluation_report(base_metrics, ft_metrics, ft_refs, ft_preds, n_samples=5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="weights/checkpoint-final")
    parser.add_argument("--base-model", type=str, default="microsoft/trocr-base-handwritten")
    parser.add_argument("--test-csv", type=str, default="data/processed/test.csv")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    run_evaluation(args.checkpoint, args.base_model, args.test_csv, args.device)