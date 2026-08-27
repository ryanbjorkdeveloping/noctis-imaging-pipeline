"""
Score braai (real/bogus) against the cross-matched answer key.

Ground truth: a detection with a ZTF rb (it's in ALeRCE = ZTF alerted on it)
is treated as REAL (y_true=1). We run braai on that detection's Stage-3 triplet
and check whether braai ALSO calls it real (P>=0.5). This measures braai's
RECALL on ZTF-real sources -- i.e. how many reals braai misses on our diffs.

Scope (honest): the rb labels are ZTF's 2023 alerts matched to our 2018
positions, valid for PERSISTENT sources; and we have no confirmed-bogus labels,
so this grades false-NEGATIVES (missed reals), not false-positives.

Run:  python stamp_testing/score_braai_vs_labels.py
"""

import os, sys
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from astropy.table import Table
from ztf_classification.braai_realbogus import load_braai, make_triplet, braai_score

LABELS = os.path.join(PROJECT_ROOT, "ztfdata", "labels",
                      "ztf_20180322273264_labeled_set.ecsv")
CUTOUTS = os.path.join(PROJECT_ROOT, "ztfdata", "ztf_own_pipeline_data_testing_small_scale", "cutouts")
BRAAI_MODELS = os.path.join(PROJECT_ROOT, "braai", "models")
THRESH = 0.5


def main():
    tab = Table.read(LABELS)
    # ground truth = detections ZTF alerted on (have a real rb value)
    rb_rows = [r for r in tab if r["ztf_rb"] == r["ztf_rb"]]   # x==x drops NaN
    print(f"{len(tab)} detections; {len(rb_rows)} have a ZTF rb (our REAL set)\n")

    model = load_braai(BRAAI_MODELS)
    print("braai loaded\n")

    y_true, y_pred, scores = [], [], []
    for r in rb_rows:
        name = os.path.basename(str(r["stamp_path"])).replace(".npy", "")
        sci, ref, diff = np.load(os.path.join(CUTOUTS, name + ".npy"))
        trip = make_triplet(sci, ref, diff)
        p = braai_score(model, trip)
        y_true.append(1)                       # ZTF alerted -> real
        y_pred.append(1 if p >= THRESH else 0) # braai's verdict
        scores.append(p)

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    n_caught = int(y_pred.sum())
    recall = n_caught / len(y_true)
    print(f"=== braai vs {len(y_true)} ZTF-real sources (threshold {THRESH}) ===")
    print(f"  braai called REAL:  {n_caught} / {len(y_true)}   (recall {recall:.3f})")
    print(f"  braai called BOGUS: {len(y_true) - n_caught} / {len(y_true)}   (missed reals)")
    print(f"  braai P(real):  min {min(scores):.3f}  median {np.median(scores):.3f}  max {max(scores):.3f}")
    print("\nSCOPE: measures braai's recall on ZTF-cataloged (persistent) sources on")
    print("       OUR difference images. Low recall = domain shift (our diff != ZTF's),")
    print("       braai's dominant error is MISSING reals, not passing bogus.")


if __name__ == "__main__":
    main()
