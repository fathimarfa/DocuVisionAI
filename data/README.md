# Dataset Documentation

## Source
**IAM Handwriting Database** (clean HF port): `Teklia/IAM-line`
https://huggingface.co/datasets/Teklia/IAM-line

- The IAM dataset is one of the assignment's own recommended sources; this is a properly structured Hugging Face port of it (pre-split train/validation/test, ready-to-use `image`/`text` fields).
- Line-level handwritten text images from 657 writers.

## Why this dataset
- **TrOCR-base-handwritten** was originally fine-tuned on IAM itself, so further fine-tuning on more IAM-style data converges reliably.
- Predefined train/validation/test splits — no manual annotation parsing, no re-splitting logic needed.
- Large enough to subsample for fast iteration (`subset_size` in config) and scale up later.

## Alternative datasets (documented, not used by default)
| Dataset | Use case | Why not default |
|---|---|---|
| IAM Handwriting | Multi-line cursive benchmark | Manual download + XML parsing overhead |
| SROIE | Receipts / structured OCR | Better suited to Donut, not TrOCR |
| FUNSD | Form layout understanding | Small (199 forms), layout-focused not text-focused |

## Pipeline
1. `download_dataset.py` pulls the dataset via `datasets.load_dataset`, saves images to `data/raw/images/`, and writes `data/processed/{train,val,test}.csv` (columns: `image_path`, `text`).
2. `preprocess.py` provides shared image/text cleanup utilities used by `src/dataset.py`.

## Regenerating splits
```bash
python data/download_dataset.py --config configs/trocr_base.yaml
```
Split ratios (80/10/10) and the random seed are controlled in the YAML config for reproducibility.