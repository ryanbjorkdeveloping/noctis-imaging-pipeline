"""Regression: the sci/ref channels must be cropped by the SAME region rule as the diff.

The bug (fixed 2026-08-11): `_prep_channel` took a `size` and always called
`central_cutout(off, size=CUTOUT_SIZE)`, while `diff_img` honoured `full_frame`. So
`run_field(full_frame=True, streaks_only=False)` measured centroids in a 2952x2944
frame and then cut sci/ref from a 1000x1000 central crop:

  - centroids past ~1031 px  -> NoOverlapError, run_field dies
  - centroids inside it      -> SILENTLY the wrong sky, offset by ~976 px

Nothing ever executed this path (every *_full run on disk is streaks_only=True), so it
sat unguarded. It is the same class as the streaks.py:95 out-of-bounds bug one layer up.

The decisive trick: reprojecting a frame onto its OWN wcs is the identity, so feeding
the diff through `_prep_channel` must reproduce `diff_img`. If the region rule regresses,
the shapes disagree (central-1000 vs full-frame) and the first assert fires; if someone
reintroduces an offset at matching shape, the content assert fires.

Caveat measured, not assumed: the comparison is NOT bit-exact. reproject_interp dilates
ZTF's chip-edge NaN mask by ~1px, which perturbs 6470 of 8.69M pixels (0.07%) - and 6449
of those sit within 2px of a NaN. So the content check excludes a 3px NaN halo and caps
the residual mismatch, rather than demanding zero. An actually-offset crop fails this by
orders of magnitude, which is the failure mode being guarded.

Needs the 2020 fixture diff on disk (no template build, no network).
"""

import os
import tempfile

import numpy as np
from astropy.table import Table
from scipy.ndimage import binary_dilation

from ztf_data.detect import (
    load_diff, central_cutout, full_frame_cutout, CUTOUT_SIZE,
)
from ztf_data.fetch import resolve_local
from ztf_data.measure import cut_triplets, STAMP
from ztf_data.run_field import _prep_channel, _footprint

FFD = 20200518187454
FIELD, CCDID, QID = 468, 3, 2
DIFF_PRODUCT = "scimrefdiffimg.fits.fz"


def _regions():
    """The two crop rules run_field chooses between, in run_field's own form."""
    return {
        "full": lambda a, w=None: full_frame_cutout(a, wcs=w),
        "central": lambda a, w=None: central_cutout(a, wcs=w, size=CUTOUT_SIZE),
    }


def test_full_frame_stamps():
    try:
        diff_path = resolve_local(FFD, DIFF_PRODUCT, field=FIELD, ccdid=CCDID, qid=QID)
    except FileNotFoundError as e:
        print(f"SKIP: fixture diff not on disk ({e})")
        return

    diff_full, diff_wcs = load_diff(diff_path)

    for name, region in _regions().items():
        diff_c = region(diff_full, diff_wcs)
        diff_img = np.nan_to_num(diff_c.data)

        # identity reproject: the "science" channel IS the diff, so a correctly
        # regioned _prep_channel must return the same array.
        sci_img = _prep_channel(diff_full, diff_wcs, diff_wcs, diff_full.shape, region)

        assert sci_img.shape == diff_img.shape, (
            f"[{name}] channel shape {sci_img.shape} != diff {diff_img.shape} - "
            f"_prep_channel is not honouring the run's region rule"
        )

        # exclude a 3px halo around ZTF's chip-edge NaN mask (reproject dilates it)
        halo = binary_dilation(np.isnan(diff_c.data), iterations=3)
        good = ~halo
        bad = (np.abs(sci_img - diff_img) > 1e-6) & good
        frac = bad.sum() / max(int(good.sum()), 1)
        assert frac < 1e-3, (
            f"[{name}] {bad.sum()} of {int(good.sum())} interior pixels ({frac:.2%}) "
            f"diverged from the diff under an identity reproject - the crop is offset"
        )

        # a footprint must be computable for the region actually detected on
        ra_c, dec_c, corners = _footprint(diff_c.wcs, diff_img.shape)
        assert len(corners) == 4 and all(len(c) == 2 for c in corners)
        assert -90.0 <= dec_c <= 90.0 and 0.0 <= ra_c <= 360.0
        print(f"  [{name}] region {diff_img.shape}  centre {ra_c:.4f} {dec_c:.4f}")

    # end-to-end through cut_triplets on the FULL-FRAME region, at centroids that the
    # old code could not reach at all (both well past the central-1000 window).
    region = _regions()["full"]
    diff_c = region(diff_full, diff_wcs)
    diff_img = np.nan_to_num(diff_c.data)
    h, w = diff_img.shape
    assert h > CUTOUT_SIZE and w > CUTOUT_SIZE, "full frame should exceed the central crop"

    sci_img = _prep_channel(diff_full, diff_wcs, diff_wcs, diff_full.shape, region)
    xs = [w - 200.0, 150.0, w / 2.0]
    ys = [h - 200.0, 150.0, h / 2.0]
    cat = Table({
        "x_centroid": xs, "y_centroid": ys,
        "sign": [1, -1, 1], "row_index": [0, 1, 2],
    })

    with tempfile.TemporaryDirectory() as tmp:
        cut_triplets(cat, sci_img, sci_img, diff_img, tmp)
        assert len(os.listdir(tmp)) == len(cat), "one stamp per detection"
        for row, x, y in zip(cat, xs, ys):
            stamp = np.load(row["stamp_path"])
            assert stamp.shape == (3, STAMP, STAMP), stamp.shape
            assert stamp.dtype == np.float32, f"stamps must be float32, got {stamp.dtype}"
            # the wrong-sky catcher: the stamp's diff channel must equal the region's
            # own pixels at that centroid. An offset crop fails here even when the
            # shapes happen to line up.
            half = STAMP // 2
            y0, x0 = int(round(y)) - half, int(round(x)) - half
            want = diff_img[y0:y0 + STAMP, x0:x0 + STAMP]
            if want.shape == (STAMP, STAMP):
                assert np.allclose(stamp[2], want, rtol=0, atol=1e-5), (
                    f"stamp at ({x:.0f},{y:.0f}) does not match the diff at that "
                    f"centroid - the channels are cut from different sky"
                )

    print(f"\nOK: both regions honoured; {len(cat)} full-frame stamps match the diff")


if __name__ == "__main__":
    test_full_frame_stamps()
