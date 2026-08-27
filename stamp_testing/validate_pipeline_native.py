"""
Validate the CONNECTED pipeline (braai real/bogus + stamp type) on ZTF's OWN
native stamps for labeled ALeRCE objects — professional-grade input + real labels.
Proves the pipeline is accurate on good data (our own field's diffs are reference-
limited; this isolates the pipeline's quality from our data's quality).

    python stamp_testing/validate_pipeline_native.py
"""
import os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from alerce.core import Alerce
from ztf_classification.braai_realbogus import load_braai, make_triplet, braai_score
from ztf_classification.stamp_classifier import (
    find_objects_of_class, fetch_stamp_probabilities,
    classify_astrophysical, apply_threshold, ASTRO_CLASSES,
)

N_PER_CLASS = 6
THRESH = 0.5

def main():
    client = Alerce()
    model = load_braai(os.path.join(PROJECT_ROOT, "braai", "models"))

    # 1. gather labeled objects: {oid: true_class}
    truth = {}
    for cls in ASTRO_CLASSES:
        for oid in find_objects_of_class(client, cls, n=N_PER_CLASS, min_prob=0.8):
            truth[oid] = cls
    print(f"{len(truth)} labeled ALeRCE objects across {len(ASTRO_CLASSES)} classes\n")

    braai_pass, braai_total = 0, 0
    y_true, y_pred = [], []
    for oid, true_cls in truth.items():
        try:
            hdul = client.get_stamps(oid, format="HDUList")            # native clean stamps
            trip = make_triplet(hdul[0].data, hdul[1].data, hdul[2].data)
        except Exception:
            continue

        # --- pipeline branch 1: braai real/bogus ---
        p_real = braai_score(model, trip)
        braai_total += 1
        if p_real < THRESH:
            continue                                                   # gated bogus, don't type
        braai_pass += 1

        # --- pipeline branch 2: stamp type (Path B by oid) ---
        probs = fetch_stamp_probabilities(client, oid)
        if probs is None:
            continue
        top_class, top_prob, _ = classify_astrophysical(probs)
        pred = apply_threshold(top_class, top_prob)
        if pred == "uncertain":
            continue
        y_true.append(true_cls)
        y_pred.append(pred)

    # --- grade ---
    from sklearn.metrics import confusion_matrix, classification_report
    print(f"=== braai real/bogus (on native reals) ===")
    print(f"  passed real: {braai_pass}/{braai_total}  (recall {braai_pass/braai_total:.3f})\n")

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    acc = (y_true == y_pred).mean() if len(y_true) else 0.0
    print(f"=== stamp type (pipeline, n={len(y_true)}, accuracy {acc:.3f}) ===")
    print(confusion_matrix(y_true, y_pred, labels=ASTRO_CLASSES))
    print(classification_report(y_true, y_pred, labels=ASTRO_CLASSES, zero_division=0))

if __name__ == "__main__":
    main()
