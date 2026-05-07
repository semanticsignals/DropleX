# DropleX

Companion code for **"DropleX: Liquid Sensing on Tablet Touchscreens"** (IMWUT 2026).

DropleX is the first system that enables liquid sensing using the capacitive touchscreen of commodity tablets. We're able to detect microliter-scale liquid samples, and performs non-invasive, through-container measurements for liquid analysis. The tablet's capacitive sensor produces a 2-D grid of readings that shift depending on the liquid's conductive and dielectric properties. A machine-learning model trained on those grids can classify the liquid type or concentration, no hardware modifications required.

---

## Hardware


| Component              | Specification                                     |
| ---------------------- | ------------------------------------------------- |
| **Tablet**             | Samsung Galaxy Note 10.1                          |
| **Capacitive sensor**  | Atmel maXTouch MXT1066T2                          |
| **Sensor grid**        | 52 rows × 32 columns of capacitive electrodes     |
| **Electrode pitch**    | 4.2 mm                                            |
| **Sensor access tool** | `mxt-app` (open-source CLI, run as root via `su`) |


> **Root access is required.** The firmware app calls `mxt-app` as root to read raw delta values directly from the sensor IC. A standard (non-rooted) tablet will not work.

---

## Repo Layout

```
tablet_cap/
│
├── firmware/                        # Android app — live heatmap viewer (RAW mode)
│  
├── data/                            # Raw touch-sensor recordings (one folder per session)
│   └── session_<name>/              # Each folder holds per-frame delta CSV files
│       ├── deltas_0.csv             # Baseline frame (captured at session start)
│       └── deltas_<timestamp>.csv   # Subsequent frames
│
├── regions/                         # Pre-extracted sensor-region tensors (.npz)
│                                    # One file per session; used directly by training scripts
│
├── my_models/                       # Bundled pre-trained CNN (Coke / ethanol spiking)
├── tablet_session_visualizer.py     # Desktop tool: interactive heatmap viewer for recorded sessions
├── region_stats.py                  # Region geometry helpers (used by the viewer)
│
├── train_liquid_classifier_rf.py    # Experiment: liquid type (tap / DI / ethanol) — Random Forest
├── train_coke_spiking_classifier.py # Experiment: Coke ethanol-spiking level — CNN
├── train_wine_classifier.py         # Experiment: wine adulteration — CNN
├── train_nacl_classifier.py         # Experiment: NaCl (salt) concentration — CNN
│
├── example_model_inference.py       # Demo: run the bundled model on a sensor frame
├── MODEL_EXPORT_README.md           # Guide: train → export → deploy a model
│
├── requirements.txt
└── LICENSE
```

---

## How It Works (End-to-End Pipeline)

```
┌─────────────────────────────┐
│  Liquid container on tablet │
│  (Samsung Galaxy Note 10.1) │
└────────────┬────────────────┘
             │ capacitance delta values
             ▼
     maXTouch MXT1066T2
     (52 × 32 electrode grid)
             │
             ▼  mxt-app (root)
   data/session_*/deltas_*.csv     ← raw per-frame CSV files
             │
       ┌─────┴──────────────────────────────────────┐
       │                                             │
       ▼  on-tablet                                  ▼  on desktop
  firmware/ Android app                  tablet_session_visualizer.py
  (real-time heatmap, RAW mode)          (playback + region annotation)
                                                     │
                                                     ▼
                                         regions/<session>_regions.npz
                                                     │
                                                     ▼
                                          Training script
                                   (Random Forest or CNN, 5-fold CV)
                                                     │
                                          ┌──────────┴──────────┐
                                          ▼                     ▼
                                      accuracy /          my_models/
                                    confusion matrix    *.pth + *.onnx
                                                             │
                                                             ▼
                                               example_model_inference.py
                                               (load model, run prediction)
```

---

## Firmware — Real-Time Heatmap Viewer

The Android app in `firmware/` streams live capacitance data from the tablet's sensor and renders it as a colour-coded heatmap.

### What it shows

Every electrode's delta value is mapped to a colour:


| Colour                  | Meaning                                            |
| ----------------------- | -------------------------------------------------- |
| **Deep red**            | Large positive delta (strong capacitance increase) |
| **Light red / white**   | Small positive delta                               |
| **Near-white**          | Near-zero change                                   |
| **Light green / white** | Small negative delta                               |
| **Deep green**          | Large negative delta                               |


A liquid container placed on the screen typically produces a rectangular patch of large-magnitude (red or green) values in the region under the container.

### Controls


| Key         | Action                                               |
| ----------- | ---------------------------------------------------- |
| Volume Up   | Toggle electrode-position marker overlay             |
| Volume Down | No action (consumed to prevent system-volume change) |


### Build & Install

Open the `firmware/` folder as an Android Studio project (Android Gradle Plugin 8+, JDK 17). Connect the rooted Samsung Galaxy Tab S6 Lite via USB and run the app.

**Prerequisites on the tablet (one-time setup):**

```bash
# Push mxt-app to the tablet (built for arm64-v8a)
adb push mxt-app /data/local/tmp/
adb shell chmod +x /data/local/tmp/mxt-app

# Create the reference baseline frame
adb shell su -c "/data/local/tmp/mxt-app -d i2c-dev:0-004a --block-size 8 \
  --debug-dump /sdcard/logs/ref.csv --frames 1 --format 0"
```

The app polls the sensor every 2 seconds by default (`CAPTURE_INTERVAL_MS` in `MainActivity.java`).

---

## Python — Data Collection & ML Experiments

### Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Visualize a recorded session on your desktop

```bash
python tablet_session_visualizer.py --f data/session_coke_unadulterated
```

Press **Space** or the Play button to step through frames. Click the heatmap to select a sensor region; the viewer saves the region to `regions/` automatically. Press `h` for the full keyboard shortcut list.

### Try the bundled pre-trained model on a random frame

```bash
python example_model_inference.py \
  --model my_models/coke_spiking_multiclass_*.onnx \
  --sample
```

### Reproduce a paper experiment

```bash
# Liquid-type classification — tap water vs DI water vs ethanol (Random Forest)
python train_liquid_classifier_rf.py --liquids tap,di,ethanol100

# Coke ethanol-spiking classification — 5-class CNN
python train_coke_spiking_classifier.py \
  --classes unadulterated,ethanol10,ethanol30,ethanol50,ethanol80 --epochs 50

# Wine adulteration — binary CNN (unadulterated vs ethanol40)
python train_wine_classifier.py --binary

# NaCl concentration — 4-class CNN
python train_nacl_classifier.py
```

Every script accepts `--help` for the full option list.

---

## Included Data Sessions

Each folder in `data/` is one recording. The matching `regions/<session>_regions.npz` file holds the pre-extracted sensor patch that training scripts read directly.

### Coke + ethanol spiking


| Session folder               | Description          |
| ---------------------------- | -------------------- |
| `session_coke_unadulterated` | Plain Coke (control) |
| `session_coke_ethanol10`     | Coke + 10 % ethanol  |
| `session_coke_ethanol20`     | Coke + 20 % ethanol  |
| `session_coke_ethanol30`     | Coke + 30 % ethanol  |
| `session_coke_ethanol50`     | Coke + 50 % ethanol  |
| `session_coke_ethanol80`     | Coke + 80 % ethanol  |


### Wine adulteration


| Session folder                | Description                        |
| ----------------------------- | ---------------------------------- |
| `session_wine_2023`           | Unadulterated wine                 |
| `session_wine_2023_ethanol40` | Wine adulterated with 40 % ethanol |


> No `ethanol50` session is bundled; `ethanol40` is the closest available tier.

### Plastic cup — liquid type (paper panel b)


| Session folder                  | Description          |
| ------------------------------- | -------------------- |
| `session_plastic_cup_tap_water` | Tap water            |
| `session_plastic_cup_deionized` | Deionized (DI) water |
| `session_plastic_cup_ethanol`   | Pure ethanol         |


### Salinity (NaCl concentration)


| Session folder             | Description           |
| -------------------------- | --------------------- |
| `session_conc_0`           | Baseline — pure water |
| `session_conc_nacl_0-0001` | 0.0001 mol/L NaCl     |
| `session_conc_nacl_0-001`  | 0.001 mol/L NaCl      |
| `session_conc_nacl_0-01`   | 0.01 mol/L NaCl       |


---

## Python Scripts in Detail

### `tablet_session_visualizer.py` — Desktop session viewer

Loads a `data/<session>/` folder and plays back the capacitive-sensor frames as an animated heatmap. Interactive controls:


| Key / control       | Action                                                  |
| ------------------- | ------------------------------------------------------- |
| Space / Play button | Start / pause animation                                 |
| `a` / `d`           | Step one frame back / forward                           |
| Click heatmap       | Auto-detect and save the sensor region under the cursor |
| `h`                 | Print all keyboard shortcuts                            |


Saves region masks to `regions/<session>_regions.npz` for use with training scripts.

### `region_stats.py` — Region geometry utilities

Helper module (not a standalone script). Computes area, perimeter, bounding box, and roughness for sensor-region masks. Imported automatically by `tablet_session_visualizer.py`.

### `train_liquid_classifier_rf.py` — Liquid-type classifier (Random Forest)

Distinguishes tap water, deionized water, and pure ethanol placed in a plastic cup or heart-shaped vessel. Uses scikit-learn `RandomForestClassifier` with 5-fold cross-validation. Runs on CPU; no GPU needed.

```bash
python train_liquid_classifier_rf.py --liquids tap,di,ethanol100
python train_liquid_classifier_rf.py --container heart --liquids tap,di,ethanol100
```

### `train_coke_spiking_classifier.py` — Coke adulteration classifier (CNN)

Classifies Coke as unadulterated or spiked at various ethanol levels using a `SpatioTemporalCNN`. Supports 5-fold cross-validation and optional ONNX / PyTorch export for deployment.

```bash
python train_coke_spiking_classifier.py \
  --classes unadulterated,ethanol10,ethanol30,ethanol50,ethanol80 \
  --epochs 50 --save-model
```

### `train_wine_classifier.py` — Wine adulteration classifier (CNN)

Binary or multiclass classification of unadulterated vs ethanol-adulterated wine.

```bash
python train_wine_classifier.py --binary           # unadulterated vs adulterated
python train_wine_classifier.py --year-classification  # 2020 vs 2023 vintage
```

### `train_nacl_classifier.py` — Salinity classifier (CNN)

Classifies NaCl concentration tiers (0, 0.0001, 0.001, 0.01 mol/L).

```bash
python train_nacl_classifier.py
python train_nacl_classifier.py --classes 0-00005,0-0001,0-001,0-01
```

### `example_model_inference.py` — Inference demo

Loads a `.pth` or `.onnx` model file and runs a prediction. Pass `--sample` to test with a randomly generated frame, or adapt the script to feed your own sensor data.

```bash
python example_model_inference.py --model my_models/<name>.onnx --sample
python example_model_inference.py --model my_models/<name>.pth  --sample
```

See `MODEL_EXPORT_README.md` for how to train, export, and deploy your own model.

---

## Citation

BibTeX will be added once camera-ready proceedings are published.