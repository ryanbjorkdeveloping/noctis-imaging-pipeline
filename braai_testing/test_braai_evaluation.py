import sys
import os

# Go up one level to target the project root and add it to python search path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.metrics import (
    precision_score,
    recall_score,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)
from astropy.table import Table
from ztf_classification.braai_realbogus import load_braai, make_triplet, braai_score

MODELS_DIR = os.path.join(PROJECT_ROOT, "braai", "models")
model = load_braai(MODELS_DIR)

CATALOG_PATH = os.path.join(
    PROJECT_ROOT,
    "ztfdata",
    "catalog",
    "ztf_20180322273264_000468_zr_c03_o_q2_stage3_catalog.ecsv"
)
catalog = Table.read(CATALOG_PATH)


# Binary labels (1 = Real, 0 = Bogus)
GOLD_LABELS = {
    "src_p1_003.npy": 1,
    "src_p1_024.npy": 1,
    "src_m1_009.npy": 1,
    "src_p1_001.npy": 0,
    "src_p1_005.npy": 0,
    "src_m1_001.npy": 0,
}

y_true = []
y_scores = []

for row in catalog:
    raw_path = row["stamp_path"]
    stamp_path = os.path.normpath(os.path.join(PROJECT_ROOT, "notebook", raw_path))
    filename = os.path.basename(stamp_path)  # Fixed typo

    if filename in GOLD_LABELS:
        try:
            npy = np.load(stamp_path)
            triplet = make_triplet(npy[0], npy[1], npy[2])
            score = braai_score(model, triplet)
            
            y_true.append(GOLD_LABELS[filename])
            y_scores.append(score)
        except Exception as e:
            print(f"SKIP {stamp_path}: {e}")

y_true = np.array(y_true)
y_scores = np.array(y_scores)  # Use plural y_scores to match model inputs

threshold = 0.5
y_pred = (y_scores >= threshold).astype(int)  # Fixed list comparison issue

prec = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

print("=== Metrics at Threshold = 0.5 ===")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"Confusion Matrix:")
print(f"  True Negatives (Bogus correctly classified): {tn}")
print(f"  False Positives (Bogus flagged as Real):     {fp}")
print(f"  False Negatives (Real missed):               {fn}")
print(f"  True Positives (Real correctly classified):  {tp}")
print("==================================\n")

# Compute Curves and AUCs
fpr, tpr, _ = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)
precision, recall, _ = precision_recall_curve(y_true, y_scores)
pr_auc = average_precision_score(y_true, y_scores)

print("=== Area Under Curve (AUC) ===")
print(f"ROC-AUC: {roc_auc:.4f}")
print(f"PR-AUC:  {pr_auc:.4f}")
print("==============================")

# Plot Curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
# ROC Curve
ax1.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
ax1.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
ax1.set_xlim([0.0, 1.0])
ax1.set_ylim([0.0, 1.05])
ax1.set_xlabel("False Positive Rate")
ax1.set_ylabel("True Positive Rate")
ax1.set_title("Receiver Operating Characteristic (ROC)")
ax1.legend(loc="lower right")

# Precision-Recall Curve
ax2.plot(recall, precision, color="blue", lw=2, label=f"PR curve (AUC = {pr_auc:.2f})")
ax2.set_xlim([0.0, 1.0])
ax2.set_ylim([0.0, 1.05])
ax2.set_xlabel("Recall")
ax2.set_ylabel("Precision")
ax2.set_title("Precision-Recall (PR) Curve")
ax2.legend(loc="lower left")

plt.tight_layout()
out_plot_path = os.path.join(PROJECT_ROOT, "ztfdata", "evaluation_curves.png")
plt.savefig(out_plot_path, dpi=150)
print(f"\nPlot saved → {out_plot_path}")
