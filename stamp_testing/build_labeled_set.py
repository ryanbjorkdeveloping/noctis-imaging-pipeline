"""
Build the LABELED evaluation set for our Stage-3 detections and save it to
ztfdata/labels/. Runs the three cross-match rungs in crossmatch_labels.py
(SkyBoT asteroids + SIMBAD type + ZTF/ALeRCE rb+type) over all 71 detections.

This is the ANSWER KEY: run once (network), then score braai + the stamp
classifier against the saved table offline, as many times as you want.

Run:  python stamp_testing/build_labeled_set.py
"""

import os, sys, glob
from collections import Counter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from astropy.table import Table
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.time import Time
import astropy.units as u
from alerce.core import Alerce

from ztf_classification.crossmatch_labels import build_labeled_set

CAT = os.path.join(PROJECT_ROOT, "ztfdata", "ztf_own_pipeline_data_testing_small_scale", "catalog",
                   "ztf_20180322273264_000468_zr_c03_o_q2_stage3_catalog.ecsv")
OUT_DIR = os.path.join(PROJECT_ROOT, "ztfdata", "labels")
OUT = os.path.join(OUT_DIR, "ztf_20180322273264_labeled_set.ecsv")


def main():
    cat = Table.read(CAT)

    # epoch from the sci FITS header (SkyBoT needs the exposure time)
    sci = glob.glob(os.path.join(PROJECT_ROOT, "ztfdata", "sci",
                                 "**", "*20180322273264*sciimg.fits"), recursive=True)[0]
    hdr = fits.open(sci)[0].header
    epoch = Time(hdr["OBSJD"], format="jd")
    field_center = SkyCoord(ra=149.8 * u.deg, dec=2.2 * u.deg)

    print(f"Building labeled set over all {len(cat)} detections (3 network rungs)...")
    client = Alerce()
    labeled = build_labeled_set(cat, field_center, epoch, client)

    print("\n=== label distribution ===")
    for lab, n in Counter(labeled["label"]).most_common():
        print(f"  {lab:12s} {n}")
    print("\n=== label_source distribution ===")
    for src, n in Counter(labeled["label_source"]).most_common():
        print(f"  {src:12s} {n}")
    n_rb = sum(1 for x in labeled["ztf_rb"] if x == x)   # x==x is False for NaN
    print(f"\n{n_rb} detections carry a ZTF rb (used to grade braai).")

    os.makedirs(OUT_DIR, exist_ok=True)
    labeled.write(OUT, format="ascii.ecsv", overwrite=True)
    print(f"\nSaved answer key -> {OUT}")


if __name__ == "__main__":
    main()
