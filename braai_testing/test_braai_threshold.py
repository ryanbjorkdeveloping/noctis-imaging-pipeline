
import sys
import os

# braai_testing/ lives under the project root; go up one level so the project
# root (which holds ztf_classification/, braai/, ztfdata/) is importable.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.table import Table
from ztf_classification.braai_realbogus import load_braai, make_triplet, braai_score

MODELS_DIR   = os.path.join(PROJECT_ROOT, "braai", "models")
model = load_braai(MODELS_DIR)

CATALOG_PATH = os.path.join(PROJECT_ROOT, "ztfdata", "ztf_own_pipeline_data_testing_small_scale", "catalog",
                            "ztf_20180322273264_000468_zr_c03_o_q2_stage3_catalog.ecsv")
catalog = Table.read(CATALOG_PATH)
print(f"Catalog: {len(catalog)} sources")

results = []   # will hold: {"label": ..., "filename": ..., "score": ...}

for row in catalog:
    raw_path   = row["stamp_path"]
    stamp_path = os.path.normpath(os.path.join(PROJECT_ROOT, "notebook", raw_path))
    
    try:
        npy     = np.load(stamp_path)
        triplet = make_triplet(npy[0], npy[1], npy[2])
        score   = braai_score(model, triplet)
        results.append({"label": int(row["label"]),
                         "filename": os.path.basename(stamp_path),
                         "score": score})
    except Exception as e:
        print(f"SKIP {stamp_path}: {e}")

scores = np.array([r["score"] for r in results])

print(f"Min:    {scores.min():.4f}")
print(f"Max:    {scores.max():.4f}")
print(f"Mean:   {scores.mean():.4f}")
print(f"Median: {np.median(scores):.4f}")

for thresh in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    n_real  = (scores >= thresh).sum()
    n_bogus = (scores <  thresh).sum()
    print(f"P(real) >= {thresh:.1f}: {n_real:3d} REAL | {n_bogus:3d} BOGUS")

fig, ax = plt.subplots(figsize=(9, 5))

ax.hist(scores, bins=30, range=(0, 1), color="#4C9BE8", edgecolor="white")

# Draw vertical lines at two candidate thresholds
ax.axvline(0.5, color="red",    linestyle="--", linewidth=1.8, label="threshold=0.5")
ax.axvline(0.7, color="orange", linestyle=":",  linewidth=1.8, label="threshold=0.7")

ax.set_xlabel("P(real)")
ax.set_ylabel("Number of sources")
ax.set_title("braai P(real) distribution — Stage 3 cutouts")
ax.legend()

out_path = os.path.join(PROJECT_ROOT, "ztfdata", "braai_score_distribution.png")
fig.savefig(out_path, dpi=150)
print(f"Plot saved → {out_path}")

THRESHOLD = 0.5
reals = [r for r in results if r["score"] >= THRESHOLD]
for r in sorted(reals, key=lambda x: x["score"], reverse=True):
    print(f"label={r['label']:3d} | {r['filename']} | P(real)={r['score']:.4f}")
