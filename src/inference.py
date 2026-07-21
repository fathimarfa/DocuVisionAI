"""
inference.py

Production inference pipeline: load the fine-tuned checkpoint once, then
transcribe new images (single or batched) in under ~2 seconds each on CPU.

Usage:
    python src/inference.py --checkpoint weights/checkpoint-best --image path/to/image.png
"""
import argparse
import logging
import time
from pathlib import Path
from typing import List, Union

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class OCRPredictor:
    def __init__(self, checkpoint_path: str, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading model from %s onto %s", checkpoint_path, self.device)
        self.processor = TrOCRProcessor.from_pretrained(checkpoint_path)
        self.model = VisionEncoderDecoderModel.from_pretrained(checkpoint_path).to(self.device)
        self.model.eval()

    def preprocess_image(self, image: Union[str, Path, Image.Image]) -> torch.Tensor:
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif image.mode != "RGB":
            image = image.convert("RGB")
        return self.processor(image, return_tensors="pt").pixel_values

    @torch.no_grad()
    def predict(self, image: Union[str, Path, Image.Image], return_confidence: bool = True) -> dict:
        start = time.time()
        pixel_values = self.preprocess_image(image).to(self.device)

        outputs = self.model.generate(
            pixel_values,
            max_length=128,
            num_beams=4,
            output_scores=return_confidence,
            return_dict_in_generate=return_confidence,
        )

        if return_confidence:
            generated_ids = outputs.sequences
            confidence = self._sequence_confidence(outputs)
        else:
            generated_ids = outputs
            confidence = None

        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        elapsed = time.time() - start

        return {"text": text, "confidence": confidence, "latency_seconds": round(elapsed, 3)}

    def _sequence_confidence(self, outputs) -> float:
        """Approximate confidence as the mean max-softmax-probability across generated tokens."""
        if not outputs.scores:
            return None
        probs = [F.softmax(s, dim=-1).max().item() for s in outputs.scores]
        return round(sum(probs) / len(probs), 4)

    def batch_predict(self, images: List[Union[str, Path, Image.Image]]) -> List[dict]:
        return [self.predict(img) for img in images]


def load_model(checkpoint_path: str, device: str = None) -> OCRPredictor:
    return OCRPredictor(checkpoint_path, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="weights/checkpoint-best")
    parser.add_argument("--image", type=str, required=True, help="Path to an image file")
    args = parser.parse_args()

    predictor = load_model(args.checkpoint)
    result = predictor.predict(args.image)
    print(f"Text:       {result['text']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Latency:    {result['latency_seconds']}s")