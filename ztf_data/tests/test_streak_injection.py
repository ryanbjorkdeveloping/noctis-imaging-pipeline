"""Positive control: an injected streak must be DETECTED elongated AND routed SHORT.

Field 468 has no real streaks, so the negative control (all BOGUS) can't tell a
working stage from a broken one. This injects a known streak into the real 2020
diff and drives the full detect->measure->cut->cascade chain, asserting it comes
out SHORT_NEA_CANDIDATE while the field's real detections stay BOGUS.

Run from the project root:  python -m ztf_data.tests.test_streak_injection
"""

import glob
import os
import tempfile

import numpy as np

from ztf_data.detect import (
    load_diff, central_cutout, CUTOUT_SIZE, background_rms, detect_both_signs,
)
from ztf_data.measure import measure_catalog, cut_triplets
from ztf_data.inject import paint_streak
from ztf_data.streaks import find_streak_candidates, _row_id

# injection knobs. WIDTH 1.5 + FLUX 200 land in DeepStreaks' short-streak sweet
# spot; the rb gate needs the span=12 fix in cut_streak_stamp to pass a real streak.
INJECT_X, INJECT_Y = 300.0, 700.0
LENGTH, ANGLE, FLUX, WIDTH = 32, 30, 200, 1.5


def test_streak_injection():
    diff_path = glob.glob("ztfdata/sci/**/*20200518187454*scimrefdiffimg.fits.fz",
                          recursive=True)[0]
    full, wcs = load_diff(diff_path)
    cut = central_cutout(full, wcs=wcs, size=CUTOUT_SIZE)
    cut_wcs, diff_img = cut.wcs, np.nan_to_num(cut.data)

    inj = paint_streak(diff_img, INJECT_X, INJECT_Y, LENGTH, ANGLE, FLUX, width=WIDTH)
    rms = background_rms(inj)
    sp, sn = detect_both_signs(inj, rms)
    cat = measure_catalog(inj, sp, sn, rms, cut_wcs)

    scratch = os.path.join(tempfile.gettempdir(), "streak_injection_test")
    cut_triplets(cat, inj, inj, inj, os.path.join(scratch, "cutouts"))  # sci/ref dummy

    # a bright streak can split into >1 detection, so "ours" = every detection
    # within the streak's footprint (radius ~LENGTH). Excluding only the single
    # nearest one would leave a streak fragment miscounted as a real field source.
    d = np.hypot(cat["x_centroid"] - INJECT_X, cat["y_centroid"] - INJECT_Y)
    inj_rows = cat[d < LENGTH]
    inj_ids = {_row_id(r["stamp_path"]) for r in inj_rows}
    assert any(float(r["elongation"]) > 1.5 for r in inj_rows), "injected streak not detected elongated"

    cands = find_streak_candidates(cat, inj, scratch)
    injected = [r for r in cands if r["row_id"] in inj_ids]
    reals = [r["route"] for r in cands if r["row_id"] not in inj_ids]
    print(f"\ninjected footprint routes: {[r['route'] for r in injected]}  (want a SHORT_NEA_CANDIDATE)")
    print(f"real elongated detections: {len(reals)}, non-BOGUS: {sum(r != 'BOGUS' for r in reals)}")

    assert any(r["route"] == "SHORT_NEA_CANDIDATE" for r in injected), "injected streak did not route SHORT"
    assert all(r == "BOGUS" for r in reals), "a real field detection routed non-BOGUS"
    print("PASS: injected streak -> SHORT; all real field detections -> BOGUS")


if __name__ == "__main__":
    test_streak_injection()
