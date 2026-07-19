# Dataset Documentation

## Source
**Handwriting Recognition Dataset** (Hugging Face): `gymprathap/Handwriting-Recognition-Dataset`
https://huggingface.co/datasets/gymprathap/Handwriting-Recognition-Dataset

- 400,000+ clean samples of handwritten names
- Pre-split, no manual cleanup required
- Free, no account/registration needed

## Why this dataset
- **TrOCR-base-handwritten** was pre-trained on similar handwriting distributions, so fine-tuning converges fast.
- No manual annotation parsing (unlike IAM's XML forms) — lower risk of preprocessing bugs eating into project time.
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
