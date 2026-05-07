# DropleX

Open-source companion code for **DropleX: Liquid sensing on tablet touchscreens** (IMWUT 2026).

## Contents

| Path | Role |
|------|------|
| `tablet_session_visualizer.py` | Interactive tool: load a session folder of maXTouch-style delta CSVs, visualize frames, outline regions (`region_stats.py`). |
| `train_coke_spiking_classifier.py` | Coke ethanol-spiking (and related) multiclass CNN; CV and optional ONNX / PyTorch export. |
| `train_conc_alcohol_classifier.py` | Alcohol **concentration** (dilution-level) classification CNN (`regions/` NPZ inputs). |
| `train_milk_adulteration_classifier.py` | Milk adulteration CNN. |
| `train_wine_classifier.py` | Wine ethanol / adulteration CNN tasks. |
| `train_container_classifier_tree2.py` | Container **liquid-type** classifier (RandomForest on spatial features from `regions/`). |
| `example_model_inference.py` | Example loader for exported `.pth` / `.onnx` + JSON metadata (see MODEL_EXPORT readme). |
| `MODEL_EXPORT_README.md` | How training exports checkpoints and normalization for deployment. |
| `my_models/` | Tiny **example** exported coke multiclass checkpoint (not a leaderboard model). |

## Example recordings (`data2/`)

Subset of captured sessions for trying the visualizer pipeline (each folder is `deltas_*.csv` per frame):

| Folder | Experiment type |
|--------|----------------|
| `data2/session_coke_unadulterated_2` | Coke baseline |
| `data2/session_coke_ethanol10` | Coke + 10% ethanol scenario |
| `data2/session_coke_ethanol50` | Coke + 50% ethanol scenario |
| `data2/session_container_heart_tap` | Heart-shaped vessel, tap water |
| `data2/session_wine_2023_ethanol10` | Wine + 10% ethanol scenario |

Try e.g.: `python3 tablet_session_visualizer.py --help` and `--f data2/session_coke_ethanol10`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Training expects region NPZ bundles under **`regions/`** produced from your recording workflow — not shipped in full here.

## Citation

Add the ACM DOI / BibTeX entry when the camera-ready proceedings entry is finalized.
