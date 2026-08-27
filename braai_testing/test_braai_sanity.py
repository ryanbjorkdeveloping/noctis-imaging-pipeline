"""
Step 2 — Sanity check: braai score vs ZTF's own 'rb' field
============================================================
Goal: confirm braai is correctly loaded by scoring NATIVE ZTF alert stamps
(not our own cutouts) for known real transients.

Expected result:
  - braai P(real) should be HIGH (> 0.5) for obvious real sources
  - ZTF's built-in `rb` field should also be HIGH for the same sources
  - They don't need to match exactly (different models), but both
    calling a known supernova BOGUS = preprocessing bug in our code

ALeRCE docs:
  get_stamps()        -> HDUList with sci/template/diff cutouts
  query_detections()  -> DataFrame with per-detection 'rb' field
"""

import sys
import os

# braai_testing/ lives under the project root; go up one level so the project
# root (which holds ztf_classification/, braai/, ztfdata/) is importable.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from alerce.core import Alerce
from ztf_classification.braai_realbogus import load_braai, make_triplet, braai_score

# ── 1. Load model ─────────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(PROJECT_ROOT, "braai", "models")
model = load_braai(MODELS_DIR)
print("Model loaded successfully\n")

# ── 2. Known real ZTF transients ──────────────────────────────────────────────
# These are well-documented real astrophysical sources (supernovae / variables).
# All should score high on both braai and ZTF's own rb.
KNOWN_REALS = [
    "ZTF18abee782",   # known real transient
    "ZTF18aajupnt",   # known real transient
    "ZTF17aaanmol",   # known real transient
]

# ── 3. Score each object and compare ─────────────────────────────────────────
client = Alerce()

print(f"{'Object ID':<18} {'braai P(real)':>14} {'ZTF rb (mean)':>14}  {'Agreement?':>12}")
print("-" * 65)

for oid in KNOWN_REALS:
    braai_result = None
    ztf_rb = None
    error_msg = None

    # --- braai score from native ZTF stamps ---
    try:
        hdul = client.get_stamps(oid, survey="ztf", format="HDUList")

        # HDU order: science=0, template=1, difference=2
        # (verify: science/template should look like star field,
        #  difference should be mostly noise + central blob)
        science    = hdul[0].data
        template   = hdul[1].data
        difference = hdul[2].data

        triplet      = make_triplet(science, template, difference)
        braai_result = braai_score(model, triplet)

    except Exception as e:
        error_msg = f"stamps failed: {e}"

    # --- ZTF's own rb score via query_detections ---
    try:
        detections = client.query_detections(oid, format="pandas")
        if "rb" in detections.columns:
            # rb is per-detection — take the mean across all detections
            ztf_rb = detections["rb"].dropna().mean()
        else:
            ztf_rb = None

    except Exception as e:
        if error_msg:
            error_msg += f" | rb failed: {e}"
        else:
            error_msg = f"rb failed: {e}"

    # --- Report ---
    if error_msg:
        print(f"{oid:<18}  ERROR: {error_msg}")
        continue

    braai_tag = "HIGH ✅" if braai_result and braai_result >= 0.5 else "LOW ⚠️ "
    rb_tag    = "HIGH ✅" if ztf_rb and ztf_rb >= 0.5 else ("LOW ⚠️ " if ztf_rb is not None else "N/A")

    agree = "✅ AGREE" if (braai_result and ztf_rb and
                           (braai_result >= 0.5) == (ztf_rb >= 0.5)) else "⚠️  DISAGREE"

    braai_str = f"{braai_result:.4f} ({braai_tag})" if braai_result is not None else "N/A"
    rb_str    = f"{ztf_rb:.4f} ({rb_tag})" if ztf_rb is not None else "N/A"

    print(f"{oid:<18}  braai={braai_result:.4f}  rb={ztf_rb:.4f}  {agree}")

print()
print("Interpretation guide:")
print("  Both HIGH  → braai is working correctly on native ZTF data ✅")
print("  braai LOW, rb HIGH → preprocessing bug in make_triplet() ⚠️")
print("  Both LOW   → these OIDs may not be real sources, or data unavailable")
