"""
notebooks/exploratory_analysis.py
—  Exploratory Data Analysis for the 3-D magnetic sensor dataset.
   Generates and saves visualisations into results/.

Run: python notebooks/exploratory_analysis.py

Author: Keshvi Agarwal
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

RESULTS = config.RESULTS_DIR
os.makedirs(RESULTS, exist_ok=True)

# ── Palette ───────────────────────────────────────────────────────────────────
PALETTE = ["#6c63ff", "#00d4aa", "#ff6b6b"]
CLASS_COLORS = {"motorcycle": PALETTE[0], "passenger_car": PALETTE[1], "truck": PALETTE[2]}

plt.rcParams.update({
    "figure.facecolor": "#0f1117",
    "axes.facecolor":   "#1a1d27",
    "axes.edgecolor":   "#6c7a89",
    "axes.labelcolor":  "#e8e8f0",
    "xtick.color":      "#e8e8f0",
    "ytick.color":      "#e8e8f0",
    "text.color":       "#e8e8f0",
    "grid.color":       "#2a2d3a",
    "grid.linestyle":   "--",
    "grid.alpha":       0.4,
})

LABEL_MAP = {"HSS": "passenger_car", "motorcycle": "motorcycle", "truck": "truck"}

# ── Load data ─────────────────────────────────────────────────────────────────
df_raw = pd.read_csv(config.RAW_CSV, header=None)
df     = df_raw[0].str.split(";", expand=True)
labels = df[config.N_RAW_FEATS].map(LABEL_MAP)
X_raw  = df.iloc[:, :config.N_RAW_FEATS].astype(float).values
X_norm = MinMaxScaler().fit_transform(X_raw)

# ── 1. Class distribution ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
counts = labels.value_counts()
bars = ax.bar(counts.index, counts.values, color=PALETTE[:len(counts)], width=0.5)
ax.set_title("Class Distribution (376 vehicles)", fontsize=14)
ax.set_xlabel("Vehicle Type"); ax.set_ylabel("Count")
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, val + 2, str(val),
            ha="center", fontsize=11)
plt.tight_layout()
plt.savefig(f"{RESULTS}/eda_class_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("[saved] eda_class_distribution.png")

# ── 2. Representative waveforms per class ─────────────────────────────────────
#    Reshape: (376, 621) → (376, 207, 3)
X_3d     = X_norm.reshape(-1, config.N_TIMESTEPS, config.N_AXES)
axis_lbl = ["X-axis", "Y-axis", "Z-axis"]

fig, axes = plt.subplots(3, 3, figsize=(14, 9), sharey=False)
fig.suptitle("Sample Waveforms by Vehicle Type & Magnetic Axis", fontsize=14, y=1.01)

for row, cls in enumerate(["motorcycle", "passenger_car", "truck"]):
    idx = labels[labels == cls].index[0]   # pick first sample of this class
    for col, axis_name in enumerate(axis_lbl):
        ax = axes[row, col]
        ax.plot(X_3d[idx, :, col], color=PALETTE[row], lw=1.5)
        ax.set_title(f"{cls} · {axis_name}", fontsize=9)
        ax.set_xlabel("Timestep"); ax.set_ylabel("Norm. amplitude")
        ax.grid(True)

plt.tight_layout()
plt.savefig(f"{RESULTS}/eda_sample_waveforms.png", dpi=150, bbox_inches="tight")
plt.close()
print("[saved] eda_sample_waveforms.png")

# ── 3. PCA — 2D projection ────────────────────────────────────────────────────
pca   = PCA(n_components=2, random_state=config.SEED)
X_pca = pca.fit_transform(X_norm)

fig, ax = plt.subplots(figsize=(7, 5))
for cls, color in CLASS_COLORS.items():
    mask = labels == cls
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
               c=color, label=cls, alpha=0.75, s=40, edgecolors="none")
ax.set_title(f"PCA (2D) — explained var: {pca.explained_variance_ratio_.sum()*100:.1f}%",
             fontsize=13)
ax.set_xlabel(f"PC-1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
ax.set_ylabel(f"PC-2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
ax.legend(framealpha=0.3)
ax.grid(True)
plt.tight_layout()
plt.savefig(f"{RESULTS}/eda_pca_projection.png", dpi=150, bbox_inches="tight")
plt.close()
print("[saved] eda_pca_projection.png")

# ── 4. t-SNE — 2D projection ──────────────────────────────────────────────────
print("Running t-SNE (may take ~30 s) …")
tsne   = TSNE(n_components=2, perplexity=30, random_state=config.SEED)
X_tsne = tsne.fit_transform(X_norm)

fig, ax = plt.subplots(figsize=(7, 5))
for cls, color in CLASS_COLORS.items():
    mask = labels == cls
    ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
               c=color, label=cls, alpha=0.75, s=40, edgecolors="none")
ax.set_title("t-SNE (2D) Embedding", fontsize=13)
ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
ax.legend(framealpha=0.3)
ax.grid(True)
plt.tight_layout()
plt.savefig(f"{RESULTS}/eda_tsne_projection.png", dpi=150, bbox_inches="tight")
plt.close()
print("[saved] eda_tsne_projection.png")

# ── 5. Per-axis amplitude box-plots ───────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
for col, axis_name in enumerate(axis_lbl):
    vals = X_3d[:, :, col]          # (376, 207)
    rms  = np.sqrt(np.mean(vals**2, axis=1))   # RMS energy per sample

    df_box = pd.DataFrame({"class": labels.values, "rms": rms})
    ax = axes[col]
    sns.boxplot(data=df_box, x="class", y="rms", palette=PALETTE[:3], ax=ax,
                linewidth=1.2)
    ax.set_title(f"{axis_name} — RMS Energy", fontsize=11)
    ax.set_xlabel(""); ax.set_ylabel("RMS (normalised)")
    ax.grid(axis="y")

plt.suptitle("Per-Axis RMS Energy by Vehicle Type", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(f"{RESULTS}/eda_rms_boxplot.png", dpi=150, bbox_inches="tight")
plt.close()
print("[saved] eda_rms_boxplot.png")

print("\n✓ EDA complete — all plots saved to", RESULTS)
