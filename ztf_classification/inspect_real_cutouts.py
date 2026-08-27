import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CUTOUTS_DIR = PROJECT_ROOT / "ztfdata" / "ztf_own_pipeline_data_testing_small_scale" / "cutouts"
OUT_PATH = PROJECT_ROOT / "ztfdata" / "inspect_real_cutouts.png"

# Defining sources to inspect.
INSPECT_STAMPS = [
    # real-looking detections
    "src_p1_003.npy",
    "src_p1_024.npy",
    "src_m1_009.npy",
    # a few bogus-looking ones for comparison
    "src_p1_001.npy",
    "src_p1_005.npy",
    "src_m1_001.npy",
]

if not CUTOUTS_DIR.exists():
    raise FileNotFoundError(f"Cutout directory not found: {CUTOUTS_DIR}")

n = len(INSPECT_STAMPS)
fig, axes = plt.subplots(n, 3, figsize=(9, n * 3), squeeze=False)

for i, fname in enumerate(INSPECT_STAMPS):
    stamp_path = CUTOUTS_DIR / fname
    if not stamp_path.exists():
        raise FileNotFoundError(f"Stamp not found: {stamp_path}")

    npy = np.load(stamp_path)
    titles = ["Science", "Template", "Difference"]
    for j in range(3):
        ax = axes[i][j]
        ax.imshow(npy[j], cmap="gray", origin="lower")
        ax.set_title(f"{fname}\n{titles[j]}", fontsize=7)
        ax.axis("off")

plt.tight_layout()
fig.savefig(OUT_PATH, dpi=150)
print(f"Plot saved → {OUT_PATH}")