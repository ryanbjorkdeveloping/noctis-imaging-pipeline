"""
Score the recovered 2019 BE5 streak stamps through DeepStreaks.

Run with the streak venv:  
.venv_streaks/bin/python streak_testing/score_nea_recovery.py
"""

import os 
import sys
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from ztf_classification.deepstreaks import load_deepstreaks, score_families, cascade_decision

# resolve $ZTFDATA from .env directly (this runs in .venv_streaks, which does not
# import config / python-dotenv). recover_2019be5.py writes stamps to $ZTFDATA/nea_recovery.
def _ztfdata():
    env = os.path.join(PROJECT_ROOT, ".env")
    if os.path.exists(env):
        for line in open(env):
            if line.strip().startswith("ZTFDATA="):
                return line.split("=", 1)[1].strip()
    return os.path.join(PROJECT_ROOT, "ztfdata")

OUT = os.path.join(_ztfdata(), "nea_recovery")
batch = np.load(f"{OUT}/nea_batch.npy").astype(np.float32)
specs = json.load(open(f"{OUT}/nea_specs.json"))

models = load_deepstreaks()
scores = score_families(models, batch)

print(f"\n{'exposure':14s} {'rb':>6} {'kd':>6} {'sl':>6}  route")
for i, tag in enumerate(specs):
    d = cascade_decision(scores, i)
    rb = max(scores["rb"][a][i] for a in scores["rb"])
    kd = max(scores["kd"][a][i] for a in scores["kd"])
    sl = max(scores["sl"][a][i] for a in scores["sl"])
    print(f"{tag:14s} {rb:6.3f} {kd:6.3f} {sl:6.3f}  {d['route']}")
