import sys
import os

# braai_testing/ lives under the project root; go up one level so the project
# root (which holds ztf_classification/, braai/, ztfdata/) is importable.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from astropy.table import Table
from ztf_classification.braai_realbogus import load_braai, make_triplet, braai_score

# ── 1. Load the braai model ───────────────────────────────────────────────────
MODELS_DIR = os.path.join(PROJECT_ROOT, "braai", "models")
model = load_braai(MODELS_DIR)
print("Model loaded successfully\n")

# ── 2. Load Stage 3 catalog ───────────────────────────────────────────────────
# The catalog's stamp_path values are relative to the notebook/ dir,
# so we resolve them from the project root instead.
CATALOG_PATH = os.path.join(PROJECT_ROOT, "ztfdata", "ztf_own_pipeline_data_testing_small_scale", "catalog",
                            "ztf_20180322273264_000468_zr_c03_o_q2_stage3_catalog.ecsv")

catalog = Table.read(CATALOG_PATH)
print(f"Loaded catalog: {len(catalog)} sources\n")

# ── 3. Score each source from our own cutouts ─────────────────────────────────
# Channel order confirmed from Stage 3 notebook: np.stack([s, r, d], axis=0)
# So npy[0]=science, npy[1]=reference/template, npy[2]=difference  ✅

CUTOUTS_DIR = os.path.join(PROJECT_ROOT, "ztfdata",
                           "ztf_own_pipeline_data_testing_small_scale", "cutouts")
for row in catalog:
    # resolve by basename against the (moved) cutouts dir — robust to the folder move
    stamp_path = os.path.join(CUTOUTS_DIR,
                              os.path.basename(str(row["stamp_path"])))
    label = row["label"]

    try:
        npy = np.load(stamp_path)                       # shape: (3, 63, 63)
        science    = npy[0]                             # channel 0 = science
        template   = npy[1]                             # channel 1 = reference/template
        difference = npy[2]                             # channel 2 = difference

        triplet = make_triplet(science, template, difference)
        score   = braai_score(model, triplet)

        tag = "REAL" if score >= 0.5 else "BOGUS"
        print(f"label={label:3d} | {os.path.basename(stamp_path)} | P(real)={score:.4f} → {tag}")

    except Exception as e:
        print(f"FAILED — label={label} | {stamp_path}: {e}")