# DropleX

Open-source companion code for **DropleX: Liquid sensing on tablet touchscreens** (IMWUT 2026).

## Layout

| Path | Role |
|------|------|
| `tablet_session_visualizer.py` | Load **`data/session_*`** maXTouch delta CSVs frame-by-frame, visualize, sketch regions (`region_stats.py`). |
| `regions/*_regions.npz` | Exported region tensors — **stem matches the folder name under `data/`** (training scripts read here). |

## Training scripts (paper experiments)

| Script | Matches figure / task |
|--------|------------------------|
| `train_liquid_classifier_rf.py` | **Liquid type** in a rectangular plastic cup: `--container plastic_cup --liquids tap,di,ethanol100`. Filename mapping: **`tap`** → `session_plastic_cup_tap_water`, **`di`** → `session_plastic_cup_deionized`, **`ethanol100`** → `session_plastic_cup_ethanol`. Use `--container heart` for **`session_container_heart_*`** data. |
| `train_coke_spiking_classifier.py` | Coke / adulterated-soda multiclass (panel‑c style ethanol levels). |
| `train_wine_classifier.py` | Wine adulteration / ethanol‑in‑wine scenarios. |
| `train_milk_adulteration_classifier.py` | Milk adulteration. |
| `train_nacl_classifier.py` | **Salinity** (panel d‑style NaCl tiers). |

Run each script with `--help` for arguments.

## Published example data (`data/` + matching `regions/`)

**Coke + ethanol spike**

- **`session_coke_unadulterated`** — soda alone  
- **`session_coke_ethanol10`** … **`session_coke_ethanol80`** incl. **`session_coke_ethanol20`** (~panel “+20 % ethanol” wording)  

**Coverage gap:** **`session_coke_ethanol100`** raw CSV plus matching `regions/` were **not** on our archived `main` branch; regenerate locally if you need that tier.

**Wine**

- **`session_wine_2023`** — unadulterated wine  
- **`session_wine_2023_ethanol40`** — adulterated (**no archived `…_ethanol50` session**—`ethanol40` is the bundled mid‑strength surrogate)

**Plastic cup — DI vs ethanol vs tap water (panel b)**

- **`session_plastic_cup_tap_water`** — tap  
- **`session_plastic_cup_deionized`** — deionized water  
- **`session_plastic_cup_ethanol`** — pure ethanol in the cup  

**Salinity**

- **`session_conc_0`** — baseline water  
- **`session_conc_nacl_0-0001`**, **`session_conc_nacl_0-001`**, **`session_conc_nacl_0-01`**

Each **`data/session_*`** row has **`regions/session_*_regions.npz`** with the **same suffix** (`session_*`).

## Example model / export docs

`MODEL_EXPORT_README.md`, `example_model_inference.py`, `my_models/` — ONNX / PyTorch example for multiclass Coke‑ethanol spiking.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Citation

Add ACM DOI / BibTeX when camera-ready proceedings are available.
