# DocuVision AI — Fine-Tuned Vision-Language Model for Document Intelligence

Fine-tunes `microsoft/trocr-base-handwritten` on the IAM Handwriting Database (via `Teklia/IAM-line`) to transcribe handwritten text images.

## Why TrOCR over Donut
| | TrOCR (chosen) | Donut |
|---|---|---|
| Best for | Text transcription accuracy | Structured layout extraction (JSON/tables) |
| Convergence on free T4, <4h | Reliable | Harder — larger effective compute need |
| Fits this assignment's grading (CER/WER-driven) | Directly | Indirectly |
| Complexity | Lower | Higher (Swin encoder + BART decoder, layout tokens) |

## Results

Evaluated on the full IAM test set (2,915 samples):

| | CER | WER |
|---|---|---|
| Base model (zero-shot) | 0.0444 | 0.1114 |
| Fine-tuned model | 0.0665 | 0.1466 |
| Change | **+49.7%** (worse) | **+31.6%** (worse) |

**The fine-tuned model underperformed the base model on this run.** This is a real, honestly-reported result — see analysis below.

## Why fine-tuning didn't improve results here

`microsoft/trocr-base-handwritten` is *already* fine-tuned by Microsoft on the IAM dataset. Further fine-tuning it on more IAM-distribution data for 10 epochs gave the model very little new signal to learn from, and most likely caused mild **catastrophic forgetting**: training loss dropped steadily to near-zero (`1.86e-05`), and validation CER during training also trended down to `0.041` (better than baseline) — but held-out test-set performance ended up worse. This gap between validation and test performance, plus the near-zero training loss, points to the model over-specializing to quirks of the training/validation split rather than learning generalizable improvements. Qualitatively, the model still gets most words right but makes occasional confident wrong-word substitutions (e.g. "Buck" → "Fuck", "wetter" → "writes") that it wouldn't have made zero-shot.

**What would likely fix this in a follow-up:**
- Fine-tune on a genuinely different data distribution (e.g. domain-specific handwriting: medical notes, forms) rather than more of the same distribution the base model already knows
- Lower learning rate and/or fewer epochs with more frequent validation checks, keeping `load_best_model_at_end` active throughout (a training interruption during this run required temporarily disabling it — see Reproducibility notes)
- Add regularization (dropout, weight decay tuning) or freeze the encoder and only fine-tune the decoder

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
python -m venv .venv && source .venv/bin/activate   # .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Running the pipeline
Data download and training need a GPU — use **`notebooks/02_model_training.ipynb`** on Google Colab or Kaggle (free T4). Everything else (editing code, running tests, inference on a trained checkpoint) works locally on CPU.

1. **Data**: `python data/download_dataset.py --config configs/trocr_base.yaml`
2. **Train**: `python -m src.train --config configs/trocr_base.yaml`
3. **Evaluate**: `python -m src.evaluate --checkpoint weights/checkpoint-final`
4. **Inference** (locally, CPU is fine for single images):
   ```bash
   python -m src.inference --checkpoint weights/checkpoint-final --image sample.png
   ```
5. **Demo**:
   ```bash
   pip install -r requirements-demo.txt
   python app.py --checkpoint weights/checkpoint-final
   ```

## Known limitations
- Fine-tuned model underperforms the zero-shot base model on this dataset/config (see Results above) — a genuine negative result, documented rather than hidden.
- Domain-specific vocabulary (e.g. technical/scientific terms) is transcribed less reliably, since IAM training data is general-English prose, not domain text.
- CPU inference latency (~15s on first call, faster afterward) is well above the assignment's <2s target; only tested on CPU, not optimized/quantized.
- Confidence score is an approximation (mean max-softmax over generated tokens), not a calibrated probability.

## Future improvements
- Fine-tune on a distribution genuinely different from what the base model already knows (domain-specific handwriting)
- Re-run with `load_best_model_at_end` active for the full run (uninterrupted), to actually recover the best validation checkpoint rather than the final-epoch one
- Try lower learning rate, fewer epochs, or partial layer freezing to reduce forgetting
- Quantize the checkpoint (int8) for faster CPU inference

## Reproducibility
Seed fixed at `42` throughout (`configs/trocr_base.yaml`). Training was interrupted mid-run (Colab/Kaggle session limits) and resumed from a saved checkpoint; the final ~50 steps were completed with `load_best_model_at_end` temporarily disabled to work around a stale checkpoint-path reference from the interrupted session — noted here for full transparency on how the final checkpoint was produced.