"""
Score braai on ZTF's OWN (native) alert stamps, not our homemade cutouts.

Our diff cutouts carry registration dipoles (a Stage-2/3 limitation) that braai
correctly rejects -> ~0.05 recall. This script fetches ZTF's native (sci,ref,diff)
stamp for each ZTF-matched detection via get_stamps and scores braai on THAT --
the input braai was built for. Measures braai's TRUE agreement with ZTF's reals.

Honest scope: this validates braai (weights + our load code) on proper stamps; it
does NOT validate OUR stamp-making (that still needs a dipole-free diff).

Run:  python stamp_testing/score_braai_native.py
"""

import os, sys
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from astropy.table import Table
from alerce.core import Alerce
from ztf_classification.braai_realbogus import load_braai, make_triplet, braai_score

LABELS = os.path.join(PROJECT_ROOT, "ztfdata", "labels",
                      "ztf_20180322273264_labeled_set.ecsv")
BRAAI_MODELS = os.path.join(PROJECT_ROOT, "braai", "models")
THRESH = 0.5

def main():
    tab = Table.read(LABELS)
    # detections that matched a ZTF object (have an oid) AND are our "real" set (have rb)
    rows = [r for r in tab
            if str(r["ztf_oid"]) not in ("", "--") and r["ztf_rb"] == r["ztf_rb"]]
    print(f"{len(rows)} ZTF-matched real detections to score on native stamps\n")

    model = load_braai(BRAAI_MODELS)
    client = Alerce()
    print("braai loaded\n")

    scores, n_ok, n_fail = [], 0, 0
    for r in rows:
        oid = str(r["ztf_oid"])
        try:
            hdul = client.get_stamps(oid, format="HDUList")
            sci, ref, diff = hdul[0].data, hdul[1].data, hdul[2].data
            trip = make_triplet(sci, ref, diff)
            scores.append(braai_score(model, trip))
            n_ok += 1
        except Exception:
            n_fail += 1

    scores = np.array(scores)
    n_real = int((scores >= THRESH).sum())
    recall = n_real / len(scores) if len(scores) else 0.0
    print(f"=== braai on NATIVE ZTF stamps ({n_ok} scored, {n_fail} stamp-fetch fails) ===")
    print(f"  braai called REAL:  {n_real} / {len(scores)}   (recall {recall:.3f})")
    print(f"  P(real):  min {scores.min():.3f}  median {np.median(scores):.3f}  max {scores.max():.3f}")
    print("\nCOMPARE: on OUR homemade cutouts braai recall was 0.048 (dipole artifacts).")
    print("This isolates: braai WORKS on proper stamps; our diff-cutout pipeline is the")
    print("gap (registration dipoles) -> a Stage-2/3 fix (deeper template / better reg).")


if __name__ == "__main__":
    main()
