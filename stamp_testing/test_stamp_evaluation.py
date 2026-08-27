import sys, os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")                     # headless — save to file, no popup window
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from alerce.core import Alerce
from ztf_classification.stamp_classifier import (
    fetch_stamp_probabilities, classify_astrophysical, ASTRO_CLASSES,
)
from ztf_classification.stamp_labels import fetch_examples

client = Alerce()
examples = fetch_examples(client, n_per_class=5)   # a few more per class for metrics

y_true, y_pred = [], []

for expected_class, oids in examples.items():
    for oid in oids:
        probs = fetch_stamp_probabilities(client, oid)
        if probs is None:
            continue
        top, p, astro = classify_astrophysical(probs)
        y_true.append(expected_class)
        y_pred.append(top)

#metrics
print(classification_report(y_true, y_pred, labels = ASTRO_CLASSES))
cm = confusion_matrix(y_true, y_pred, labels = ASTRO_CLASSES)
print(cm)

#heatmap set up with matplotlib
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(len(ASTRO_CLASSES))); ax.set_xticklabels(ASTRO_CLASSES)
ax.set_yticks(range(len(ASTRO_CLASSES))); ax.set_yticklabels(ASTRO_CLASSES)
ax.set_xlabel("predicted"); ax.set_ylabel("true (ALeRCE)")
for i in range(len(ASTRO_CLASSES)):
    for j in range(len(ASTRO_CLASSES)):
        ax.text(j, i, cm[i, j], ha="center", va="center")
out = os.path.join(PROJECT_ROOT, "ztfdata", "stamp_confusion_matrix.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nConfusion matrix saved → {out}")

print("\nSCOPE: metrics are on ALeRCE example objects, not our own detections")
print("       (only 2/3 of ours matched — too few to evaluate). Because y_true and")
print("       y_pred both derive from ALeRCE, this validates our decision-rule")
print("       pipeline (drop-bogus/renormalize/threshold), not independent CNN skill.")
