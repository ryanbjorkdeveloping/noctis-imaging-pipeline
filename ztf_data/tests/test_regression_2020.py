"""Regression: run_field on the 2020 fixture must reproduce the 114-detection catalog.

Hard-asserts the DETERMINISTIC core (detection count, catalog values, stamp names).
The braai/verdict counts are network-dependent (live ALeRCE) - printed as a comparison,
never asserted, because a test that fails on someone else's server is a bad test.
"""

import os
import glob

import numpy as np
from astropy.table import Table

from ztf_data.run_field import run_field

FIXTURE_CAT = ("ztfdata/ztf_own_pipeline_data_testing_small_scale/catalog/"
               "ztf_20200518187454_stage3_catalog.ecsv")
FIXTURE_CUTOUTS = "ztfdata/ztf_own_pipeline_data_testing_small_scale/cutouts_2020"

# columns present in BOTH the fixture and the wider new schema (see plan's schema note)
SHARED_EXACT = ["x_centroid", "y_centroid", "segment_flux", "elongation", "sign"]
SHARED_TOL = {"snr": 1e-2}   # Background2D interpolation jitter - numerically irrelevant


def test_regression_2020():
    r = run_field(science_filefracday=20200518187454, force=False)

    # 1. detection count - the headline invariant
    assert r.n_detections == 114, f"got {r.n_detections}, want 114"

    mine = Table.read(r.catalog_path)
    fix = Table.read(FIXTURE_CAT)
    assert len(mine) == len(fix) == 114

    # 2. catalog values on the shared columns
    #    (sort both by the stable stamp basename so rows line up)
    def key(t):
        return np.argsort([os.path.basename(p) for p in t["stamp_path"]])
    mi, fi = mine[key(mine)], fix[key(fix)]

    for col in SHARED_EXACT:
        assert np.allclose(mi[col], fi[col], rtol=1e-6, atol=1e-6), f"{col} drifted"
    for col, rtol in SHARED_TOL.items():
        assert np.allclose(mi[col], fi[col], rtol=rtol), f"{col} drifted beyond {rtol}"

    # 3. stamp basenames identical
    got = sorted(os.path.basename(p) for p in glob.glob(r.cutouts_dir + "/*.npy"))
    want = sorted(os.path.basename(p) for p in glob.glob(FIXTURE_CUTOUTS + "/*.npy"))
    assert got == want, "stamp basenames diverged from the fixture"

    print(f"\nOK: {r.n_detections} detections, catalog + basenames match fixture")


if __name__ == "__main__":
    test_regression_2020()
