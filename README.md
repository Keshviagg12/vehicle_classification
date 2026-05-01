# 🚗 Vehicle Type Classification via 3-D Magnetic Sensor Signals

> **Deep Learning · Time-Series · IoT Sensing · Python**  
> Author: **Keshvi Agarwal**

---

## Overview

A road-embedded 3-axis magnetic sensor captures the disturbance signature
of each vehicle as it passes overhead.  This project trains and evaluates
two deep-learning classifiers to distinguish:

| Class | Label | Count |
|-------|-------|-------|
| Motorcycle | Light (L) | 47 |
| Passenger Car | Medium (M) | 296 |
| Bus / Truck | Heavy (H) | 33 |

376 labelled vehicle passages are included in the dataset.  The heavy class
imbalance (1 : 6 : 0.7) is handled via inverse-frequency class weighting
rather than oversampling, keeping the original data distribution intact.

---
## Dashboard Preview

![Dashboard Preview](assets/dashboard_preview.png)

---

## Models

### 1 — CNN-BiLSTM (raw signals)
Designed for the raw dataset (`class3.csv`) where each sample contains
621 values = **207 time steps × 3 magnetic axes**.

```
Input (207, 3)
  Conv1D(64, k=5) → BatchNorm → ReLU → MaxPool(2)
  Conv1D(128, k=5) → BatchNorm → ReLU → MaxPool(2)
  Bidirectional LSTM(64)
  Dense(64, ReLU) → Dropout(0.3)
  Dense(3, Softmax)
```

**Why CNN + BiLSTM?**  
The CNN layers extract local magnetic waveform patterns per axis; the
Bidirectional LSTM then models temporal dependencies in both directions,
capturing the symmetric entry–exit signature of vehicles passing the sensor.

### 2 — MLP (feature-engineered inputs)
Operates on `class3_FE.csv` which contains 44 pre-computed statistical and
frequency-domain features per vehicle.

```
Input (44,)
  Dense(128) → BN → ReLU → Dropout(0.3)
  Dense(64)  → BN → ReLU → Dropout(0.3)
  Dense(32)  → BN → ReLU → Dropout(0.2)
  Dense(3, Softmax)
```

---

## Project Structure

vehicle_classification/
│
├── config.py
├── train.py
├── predict.py
├── requirements.txt
│
├── assets/                        ← ADD THIS FOLDER
│   └── dashboard_preview.png      ← PUT IMAGE HERE
│
├── data/
│   ├── class3.csv
│   └── class3_FE.csv
│
├── models/
│   ├── cnn_lstm.py
│   └── mlp_fe.py
│
├── utils/
│   ├── data_loader.py
│   ├── metrics.py
│   └── seed.py
│
├── notebooks/
│   └── exploratory_analysis.py
│
└── results/
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Place data files

```bash
cp /path/to/class3.csv     data/
cp /path/to/class3_FE.csv  data/
```

### 3. Explore the data (optional)

```bash
python notebooks/exploratory_analysis.py
```

Generates PCA / t-SNE projections, per-class waveforms, and RMS energy
box-plots in `results/`.

### 4. Train

```bash
# Train both models (recommended)
python train.py --mode both

# Train only the CNN-BiLSTM
python train.py --mode raw

# Train only the MLP
python train.py --mode fe
```

Trained models are saved to `models/`, evaluation plots to `results/`.

### 5. Predict on new data

```bash
python predict.py --model models/cnn_bilstm_vehicle.keras \
                  --input data/class3.csv --mode raw
```

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| CNN before LSTM | CNNs reduce the 207-step sequence, letting LSTM focus on high-level temporal patterns |
| Bidirectional LSTM | Vehicle signatures are symmetric; both directions improve recall |
| Class weighting | 8× imbalance between car and truck — weighting recovers minority-class F1 |
| Categorical cross-entropy | More interpretable loss with class weighting than focal loss |
| Train/val/test split | Separate validation set prevents early stopping from leaking into test |
| GlorotUniform init | Stabilises gradients on the small dataset (376 samples) |
| MinMaxScaler | Consistent with original study; preserves inter-feature relationships |

---

## Results (indicative)

Exact numbers depend on hardware and TF version.  Expected range:

| Metric | CNN-BiLSTM (raw) | MLP (FE) |
|--------|-----------------|----------|
| Accuracy | ~88–93% | ~90–94% |
| Weighted F1 | ~0.86–0.92 | ~0.88–0.93 |
| Motorcycle F1 | ~0.75–0.88 | ~0.78–0.90 |
| Truck F1 | ~0.70–0.85 | ~0.72–0.87 |

The MLP benefits from the human-designed feature set that compresses domain
knowledge; the CNN-BiLSTM learns entirely from raw signals end-to-end.

---

## Citation

If you use this dataset, please cite the original papers:

1. Kolukisa, B. et al. (2022). *A deep neural network approach with
   hyper-parameter optimization for vehicle type classification using 3-D
   magnetic sensor.* Computer Standards & Interfaces.
   <https://doi.org/10.1016/j.csi.2022.103703>

2. Kolukisa, B. et al. (2022). *Deep learning approaches for vehicle type
   classification with 3-D magnetic sensor.* Computer Networks, 217, 109326.
   <https://doi.org/10.1016/j.comnet.2022.109326>

---

## Author

**Keshvi Agarwal**  
Deep learning · Signal processing · IoT sensing

---

*Built from scratch with a clean, modular architecture — not a port of the original Colab notebooks.*
