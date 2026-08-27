import sys, os
# This script lives in stamp_testing/ (one level under the project root).
# ztf_classification/ is UP one level, so put the root on sys.path so the
# `from ztf_classification...` imports resolve when run from anywhere.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from alerce.core import Alerce
from ztf_classification.stamp_classifier import (
    fetch_stamp_probabilities, classify_astrophysical, apply_threshold,
)
from ztf_classification.stamp_labels import fetch_examples

client = Alerce()
examples = fetch_examples(client, n_per_class=2)

# FOr each (expected_class, oid): fetch probs --> classify --> threshold --> print
for expected_class, oids in examples.items():
    #classify
    for oid in oids:
        probs = fetch_stamp_probabilities(client, oid)
        if probs is None:
            print(f"{oid}: no stamp classifier found")
            continue
        
        top_class, top_prob, astro = classify_astrophysical(probs)
        verdict = apply_threshold(top_class, top_prob)
        # print: oid | ALeRCE asked-for class | our verdict | probability
        # and ideally a ✓/✗ showing whether verdict matches expected_class

        match = "✓" if verdict == expected_class else "✗"
        print(f"{oid} | {expected_class} | {verdict} | {top_prob:.3f} | {match}")