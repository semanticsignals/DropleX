# DropleX

Open-source companion code for **DropleX: Liquid sensing on tablet touchscreens** (IMWUT 2026).

## Layout

| Path | Role |
|------|------|
| `tablet_session_visualizer.py` | Load **`data/session_*`** maXTouch delta CSVs frame-by-frame, visualize, sketch regions (`region_stats.py`). |
| `regions/*_regions.npz` | Exported region tensors aligned **by session name** with folders under **`data/`** (training scripts read here). |

## Training scripts (paper experiments)

| Script | Matches figure / task |
|--------|------------------------|
| `train_container_classifier_tree2.py` | **Liquid type** (plastic cup): use `--container plcup --liquids tap,di,ethanol100`. In code, **`tap` maps to** `session_container_plcup_12*_regions.npz` (cup geometry “12 mL”; same vessel class as DI/ethanol recordings). |
| `train_coke_spiking_classifier.py` | Coke / adulterated-soda multiclass (**panel c-style** discriminability; includes multiple ethanol spike levels). |
| `train_wine_classifier.py` | Wine adulteration / ethanol levels. |
| `train_milk_adulteration_classifier.py` | Milk adulteration. |
| `train_conc_alcohol_classifier.py` | Alcohol concentration (dilution) from `regions/` (distinct from spike-level coke multiclass design). |
| `train_nacl_classifier.py` | **Salinity** tiers (**panel d**, NaCl concentrations). |

See each script’s `--help`.

## Published example data (`data/` + matching `regions/`)

**Coke + ethanol spike (panels similar to multiclass adulteration):** one CSV session each — unadulterated, **10 / 20 / 30 / 50 / 80** % ethanol in filename (`session_coke_ethanol*`). The **~20 %** variant matches the qualitative **soda vs soda+20 % ethanol** split in the confusion-matrix panel. **`session_coke_ethanol100` raw CSV was not archived in git** on `main`; bundled **`regions/`** only go up through **ethanol80** for this lineage. To reproduce ethanol100, capture that session locally and regenerate `regions/`.

**Wine (unadulterated vs ethanol-adult.):**

- Unadulterated: `session_wine_2023`
- Adulterated (~mid strength filename): `session_wine_2023_ethanol40` (no **`ethanol50`** CSV in archived `main`; `ethanol40` is the closest half-ish setting we ship.)

**Plastic cup liquids (panel b semantics — DI vs ethanol vs “tap”)**

- **`session_container_plcup_di`** — deionized water  
- **`session_container_plcup_ethanol100`** — ethanol  
- **`session_container_plcup_12`** — **tap water** (paired with geometry label `12` in filenames; classifier maps `liquid=tap` → this pattern.)

**Salinity (panel d):**

- `session_conc_0` — deionized baseline  
- `session_conc_nacl_0-0001`, `session_conc_nacl_0-001`, `session_conc_nacl_0-01`

Each row above has **`regions/<same_session_name>_regions.npz`**.

## Example model / export docs

- `MODEL_EXPORT_README.md`, `example_model_inference.py`, `my_models/` — ONNX / PyTorch export of a **multiclass ethanol-spiking Coke** checkpoint (not the only model in the paper).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Citation

Add ACM DOI / BibTeX when camera-ready proceedings are available.
