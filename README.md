# DropleX

Companion code for ["DropleX: Liquid Sensing on Tablet Touchscreens"](https://arxiv.org/pdf/2511.02694) (IMWUT 2026).

DropleX is the first system that enables liquid sensing using the capacitive touchscreen of commodity tablets with no hardware modification required. Our system is able to detect microliter-scale liquid samples, and perform through-container measurements for liquid analysis. 

---

## Hardware

Samsung Galaxy Note 10.1 (2014 edition) with Atmel maXTouch MXT1066T2 (52 × 32 electrode grid, 4.2 mm pitch). Raw delta values are read via `mxt-app` (open-source CLI, requires root).

> **Root access is required.** A standard (non-rooted) tablet will not work.

---

## Structure

- `firmware/` — Android app that streams live capacitance data and renders it as a real-time heatmap. 
- `data/` — Raw per-frame delta CSV files organized by session (`session_<name>/`). Each session has a baseline frame (`deltas_0.csv`) and subsequent measurement frames.
- `regions/` — Extracted sensor-region tensors (`.npz`) used directly by training scripts.
- `tablet_session_visualizer.py` — Desktop tool to play back recorded sessions as an animated heatmap. Click a region to extract and save it to `regions/`.
- `train_*.py` — Training scripts for each experiment:
  - `train_liquid_classifier_rf.py` — liquid type (tap / DI / ethanol)
  - `train_coke_spiking_classifier.py` — Coke ethanol-spiking level
  - `train_wine_classifier.py` — wine adulteration
  - `train_nacl_classifier.py` — NaCl concentration
- `example_model_inference.py` — Load a bundled model and run a prediction.
- `my_models/` — Pre-trained CNN (Coke/ethanol spiking), available as `.pth`.
- `ref_normal.csv` — Reference capacitance map snapshot at state of rest.

---

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Visualize a session and extract sensor regions
python tablet_session_visualizer.py --f data/session_coke_unadulterated

# Run the bundled pre-trained model
python example_model_inference.py --model my_models/coke_spiking_multiclass_*.pth --sample

# Reproduce a paper experiment
python train_coke_spiking_classifier.py \
  --classes unadulterated,ethanol10,ethanol30,ethanol50,ethanol80 --epochs 50
```

All training scripts accept `--help` for the full option list.

---

## Included Data

Sessions in `data/` cover four experiments: Coke ethanol-spiking, wine adulteration, liquid type in plastic cups, and NaCl concentration. Matching `.npz` region files are in `regions/`.

---

## Citation

BibTeX will be added once camera-ready proceedings are published.