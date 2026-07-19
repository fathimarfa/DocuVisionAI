# DocuVision AI — Fine-Tuned Vision-Language Model for Document Intelligence

Fine-tunes `microsoft/trocr-base-handwritten` on the [Handwriting Recognition Dataset](https://huggingface.co/datasets/gymprathap/Handwriting-Recognition-Dataset) to transcribe handwritten text images, targeting >50% CER reduction vs. zero-shot baseline.

## Why TrOCR over Donut
| | TrOCR (chosen) | Donut |
|---|---|---|
| Best for | Text transcription accuracy | Structured layout extraction (JSON/tables) |
| Convergence on free T4, <4h | Reliable | Harder — larger effective compute need |
| Fits this assignment's grading (CER/WER-driven) | Directly | Indirectly |
| Complexity | Lower | Higher (Swin encoder + BART decoder, layout tokens) |

Donut is documented as an alternative (`configs/donut_base.yaml`) for structured-form use cases (receipts, forms), but TrOCR is the primary path here.

## Project Structure
```
document-ocr-finetuning/
├── data/               # download + preprocessing scripts
├── src/                # dataset, train, evaluate, inference
├── configs/            # YAML hyperparameter configs
├── weights/            # saved checkpoints (best / final)
├── notebooks/          # Colab training notebook (GPU step)
├── tests/              # unit tests
├── app.py              # Gradio demo
└── requirements.txt
```

## Setup
```bash
git clone <your-repo-url>
cd document-ocr-finetuning
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Running the pipeline
Data download and training need a GPU — do these steps in **`notebooks/02_model_training.ipynb` on Google Colab** (free T4). Everything else (writing/editing code, running tests, running inference on a trained checkpoint) works locally in VS Code on CPU.

1. **Data**: `python data/download_dataset.py --config configs/trocr_base.yaml` *(on Colab)*
2. **Train**: `python src/train.py --config configs/trocr_base.yaml` *(on Colab)*
3. **Evaluate**: `python src/evaluate.py --checkpoint weights/checkpoint-best` *(on Colab, or locally on CPU — slower but works)*
4. **Inference** (locally, after downloading the trained checkpoint from Colab):
   ```bash
   python src/inference.py --checkpoint weights/checkpoint-best --image sample.png
   ```
5. **Demo**:
   ```bash
   python app.py --checkpoint weights/checkpoint-best
   ```

## Results
_Fill in after training — evaluate.py prints this automatically:_
```
Base Model CER: 0.XXXX   Fine-Tuned CER: 0.XXXX   CER Reduction: XX.X%
Base Model WER: 0.XXXX   Fine-Tuned WER: 0.XXXX   WER Reduction: XX.X%
```

## Known limitations
- Trained on names/short-phrase handwriting samples; long multi-line paragraphs (e.g. full IAM forms) may need additional fine-tuning.
- Confidence score is an approximation (mean max-softmax over generated tokens), not calibrated probability.

## Future improvements
- Data augmentation (rotation/noise/blur) for robustness to scan quality
- Beam search width tuning; try cosine LR schedule
- Quantize checkpoint (int8) for faster CPU inference

## Reproducibility
Seed fixed at `42` throughout (`configs/trocr_base.yaml`). Exact package versions pinned in `requirements.txt`.
