"""
Task 2b — build the Phase-B (light-curve classifier) evaluation set.

Our own field's objects are too faint/sparse (0 of 30 have >=6 detections), so we
validate the light-curve classifier on ALERT-RICH, confidently-typed ALeRCE objects
instead (same pattern as braai's native-stamp validation). For a few light-curve
classes we query typed objects, keep those with >=6 detections, and pull each one's
full multi-epoch light curve.

Output: ztfdata/lc_eval/lightcurves.parquet (all detections, tagged by oid + true class)
        ztfdata/lc_eval/objects.csv         (per-object true class + ndet + colors)

RUN FROM THE SANDBOX:  .venv_lc/bin/python lc_testing/build_lc_eval_set.py
"""
import os, sys
import pandas as pd
from alerce.core import Alerce

PR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PR, "ztfdata", "lc_eval")
os.makedirs(OUT, exist_ok=True)

# a spread of light-curve classes with real histories: periodic + stochastic + transient
CLASSES = ["LPV", "RRL", "E", "QSO", "AGN", "SNIa"]
N_PER_CLASS = 5          # small eval set; enough to prove + grade the classifier
MIN_DET = 6              # the hard >=6-in-g-or-r requirement (approx via total ndet here)


def main():
    client = Alerce()
    obj_rows, lc_frames = [], []

    for cls in CLASSES:
        try:
            objs = client.query_objects(classifier="lc_classifier", class_name=cls,
                                        page_size=40, format="pandas")
        except Exception as e:
            print(f"  {cls}: query failed ({e})"); continue
        objs = objs.drop_duplicates("oid")
        confident = objs[objs["ndet"] >= MIN_DET].head(N_PER_CLASS)
        rich = confident.sort_values("probability", ascending=False).head(N_PER_CLASS)
        print(f"{cls}: {len(rich)} objects with >={MIN_DET} detections")

        for _, o in rich.iterrows():
            oid = o["oid"]
            try:
                det = client.query_detections(oid, format="pandas")
            except Exception:
                continue
            # per-band counts (fid 1=g, 2=r) — enforce the real >=6-g-OR-6-r cut
            g = int((det["fid"] == 1).sum()); r = int((det["fid"] == 2).sum())
            if g < MIN_DET and r < MIN_DET:
                continue
            det = det.copy()
            det["oid"] = oid
            det["sgscore1"] = 0.5
            lc_frames.append(det)
            # keep the FULL query_objects row as object_information (preprocessor needs it),
            # plus our true_class + band counts
            row = o.to_dict()
            row.update({"true_class": cls, "g_det": g, "r_det": r})
            obj_rows.append(row)

    objects = pd.DataFrame(obj_rows).drop_duplicates(subset="oid", keep="first")
    # rename to the library's object_info schema (underscore -> hyphen; deltajd -> deltamjd)
    objects = objects.rename(columns={
        "g_r_max": "g-r_max", "g_r_max_corr": "g-r_max_corr",
        "g_r_mean": "g-r_mean", "g_r_mean_corr": "g-r_mean_corr",
        "deltajd": "deltamjd"})
    # fill the few flags the preprocessor expects but query_objects omits
    for col, default in [("ndubious", 0), ("nearPS1", False), ("nearZTF", False)]:
        if col not in objects.columns:
            objects[col] = default

    lightcurves = pd.concat(lc_frames, ignore_index=True) if lc_frames else pd.DataFrame()

    objects.to_csv(os.path.join(OUT, "objects.csv"), index=False)
    lightcurves.to_parquet(os.path.join(OUT, "lightcurves.parquet"))
    print(f"\nEVAL SET: {len(objects)} objects, {len(lightcurves)} total detections")
    print("by class:"); print(objects["true_class"].value_counts())
    print(f"\nsaved -> {OUT}/objects.csv + lightcurves.parquet")


if __name__ == "__main__":
    main()