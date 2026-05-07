# DropleX

Open-source companion code for **DropleX: Liquid sensing on tablet touchscreens** (IMWUT 2026).

This repository contains a **minimal pipeline**: collect maXTouch-style delta grids, define regions interactively, train a touchscreen-based liquid classification model, and run exported ONNX or PyTorch checkpoints.

## Contents

| Path | Role |
|------|------|
| `measure2.py`, `measure3.py` | Load session CSVs, visualize deltas, draw regions (`region_stats.py`) |
| `train_coke_spiking_classifier.py` | Train, cross-validate, and export CNN checkpoints |
| `example_model_inference.py` | Load `.pth` or `.onnx` + metadata and run inference |
| `MODEL_EXPORT_README.md` | Export format and deployment notes |
| `my_models/` | Example multiclass ethanol-spiking checkpoint (small, for reproducibility) |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Quick pointers

- **Interactive labeling**: See `--help` on `measure2.py` / `measure3.py` for session-folder layout (`deltas_*.csv`).
- **Training + ONNX export**: `MODEL_EXPORT_README.md` and `train_coke_spiking_classifier.py --help`.
- **Inference**: `example_model_inference.py` (expects exported weights under `my_models/` or your own path).

## Citation

Add the ACM DOI / BibTeX entry here when the camera-ready proceedings entry is finalized.
