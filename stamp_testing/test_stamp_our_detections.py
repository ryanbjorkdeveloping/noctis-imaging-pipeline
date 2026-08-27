#connect project to project root to actually run application without errors
import sys, os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# importing what we need from library for testing
from astropy.table import Table
from alerce.core import Alerce
from ztf_classification.stamp_classifier import classify_detection, apply_threshold

#getting our catalog path from stage 3
CATALOG_PATH = os.path.join(PROJECT_ROOT, "ztfdata", "ztf_own_pipeline_data_testing_small_scale", "catalog",
                            "ztf_20180322273264_000468_zr_c03_o_q2_stage3_catalog.ecsv")
catalog = Table.read(CATALOG_PATH)

# The 3 detections braai passed at P(real) >= 0.5 (labels 3, 24, 9). 
BRAAI_PASSED = {"src_p1_003.npy", "src_p1_024.npy", "src_m1_009.npy"}

client = Alerce()
n_matched = 0

for row in catalog:
    fname = os.path.basename(row["stamp_path"])
    if fname not in BRAAI_PASSED:
        continue
    
    result = classify_detection(client, float(row["ra"]), float(row["dec"]))

    # if not result["matched"]: print a NO-MATCH line, continue
    # else: n_matched += 1 ; verdict = apply_threshold(...) ; print the verdict line

    if not result["matched"]:
        print(f"NO MATCH for {fname}")
    else:
        n_matched += 1
        verdict = apply_threshold(result["top_class"], result["top_prob"])
        print(f"{fname}: {result['top_class']} (P={result['top_prob']:.3f}) -> {verdict}")

print(f"\nTotal matched: {n_matched} / {len(BRAAI_PASSED)}")
print("\nNOTE: a NO-MATCH is a valid result — ALeRCE only catalogs objects from ZTF's")
print("      live alert stream, so our own-pipeline detections may have no counterpart.")
print("      (Same as SkyBoT finding 0 known objects in this field.) A NO-MATCH does")
print("      NOT mean bogus — braai already confirmed these 3 are real.")
