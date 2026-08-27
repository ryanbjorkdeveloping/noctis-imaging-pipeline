"""Bake per-cell REAL FITS (not PNG) so Aladin Lite v3 can render every cell the
same way index.html renders field 468 - genuine WebGL2 FITS decoding, pan/zoom,
live colormap/stretch tuning - not a flat pre-rendered image.

WHY THIS EXISTS
    build_cell_stacks.py already reprojects each cell's deep reference AND every
    epoch's official diff onto a shared TAN grid (see its downsampled_grid) - it
    just then quantizes the result to 8-bit PNG for the lightweight blink viewer.
    This script reuses THE EXACT SAME reprojection helpers (imported from
    build_cell_stacks, not reimplemented: downsampled_grid, reference_path,
    resolve_local, dedupe_epochs) and instead writes the reprojected float32
    arrays out as real FITS with a proper TAN WCS header, so fitsrs (Aladin's
    WASM FITS decoder) can load them directly - the same rendering path
    index.html already uses for field 468, now driven per-cell.

    NO NEW IRSA DOWNLOADS: the raw reference + diff FITS this reads are already
    cached on local disk from the PNG-stack build. This is CPU-bound reprojection
    only, and reprojection is cheap at this grid size (build_cell_stacks.py's own
    timings: ~2.5s for a whole single-epoch cell, once the raw files are local).

WHAT IS WRITTEN, per cell (ui/data/survey/cells/<cell_key>/fits/):
    ref_tan.fits         deep reference co-add, reprojected to TAN, float32
    <filefracday>.fits   each epoch's official diff, reprojected to TAN, float32,
                          NaN-floored + clipped for DISPLAY (see NOISE_FLOOR_SIGMA /
                          CEILING_SIGMA below — a real Aladin render of the RAW
                          reprojected diff came back as near-total static; this is
                          the fix, screenshot-verified, and it is display-only —
                          the manifest's diff_sigma is measured on the raw array)
    manifest.json         what's on disk + per-epoch obsjd/ut_date/n_detections,
                          consumed by ui/cell-aladin.js

Run: .venv/bin/python ui/build_cell_fits.py [--force] [--cell KEY]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
UI_DIR = os.path.dirname(os.path.abspath(__file__))
if UI_DIR not in sys.path:
    sys.path.insert(0, UI_DIR)

import config  # noqa: F401  LOAD-BEARING: sets $ZTFDATA before ztfquery imports

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp

from ztf_data.fetch import resolve_local

# Reuse the ALREADY-CORRECT helpers from build_cell_stacks.py rather than
# reimplementing quadrant resolution / grid construction / reference download —
# those are proven (27/29 cells built successfully) and any divergence here
# would silently produce a different grid than the PNG stack uses.
from build_cell_stacks import (
    downsampled_grid, dedupe_epochs, reference_path, robust_sigma, FILTERCODE,
    DIFF_PRODUCT, CELLS_JSON, SIZE,
)

OUT_ROOT = os.path.join(UI_DIR, "data", "survey", "cells")

# Display-time noise floor for the per-epoch diff FITS — added after the first
# real Aladin render came back as near-total salt-and-pepper static covering
# most of the frame, screenshot-verified against the RAW reprojected diff
# array (no floor at all). MEASURED, not guessed, via a live CDP session
# against this exact colormap/stretch/cuts combination (the same 'ztf-change'
# 5-stop diverging ramp index.html's change_overlay.fits uses):
#   floor=1*diff_sigma (index.html's own choice, ~30% of pixels survive)  -> STILL solid static
#   floor=1*diff_sigma, cut clipped to the exact display window           -> STILL solid static
#   floor=5*diff_sigma, ceiling=20*diff_sigma (0.13% of pixels survive)   -> clean: black sky,
#                                                                            sparse amber/cyan dots
# So — unlike index.html's ALREADY-CLIPPED change_overlay.fits, whose own
# ztf-change colormap dead-zone does the rest of the suppression — THIS
# renderer/colormap/cut combination did not visibly suppress anything between
# the NaN floor and the cut edge, no matter how the cut window was widened or
# narrowed (confirmed with a control test: clipping the raw array to exactly
# +/-60 counts, matching the display cut, was JUST AS NOISY as the unclipped
# original at a +/-5000 cut). The only lever that actually thinned the noise
# was raising what counts as "signal" at build time, i.e. a stronger NaN
# floor — so the floor here is set much higher than index.html's ~1 sigma.
# A real catalogued source's PEAK pixel sits far above this floor (SNR is a
# per-source measure over many pixels, not a single-pixel amplitude), so
# genuine detections still show through; this only mutes the raw noise field.
NOISE_FLOOR_SIGMA = 5.0
CEILING_SIGMA = 20.0


def _write_fits(arr, wcs_out, path):
    hdr = wcs_out.to_header()
    hdu = fits.PrimaryHDU(data=np.ascontiguousarray(arr.astype(np.float32)), header=hdr)
    hdu.writeto(path, overwrite=True, checksum=False, output_verify="silentfix")


def build_cell_fits(cell, force):
    key = cell["cell_key"]
    epochs = dedupe_epochs(cell.get("epochs") or [])
    if not epochs:
        return None, "0 real visits"

    out_dir = os.path.join(OUT_ROOT, key, "fits")
    manifest_path = os.path.join(out_dir, "manifest.json")
    if os.path.exists(manifest_path) and not force:
        return json.load(open(manifest_path)), "cached"
    os.makedirs(out_dir, exist_ok=True)

    field, ccdid, qid = cell["field"], cell["ccdid"], cell["qid"]

    # resolve every epoch's diff BEFORE doing any work — same discipline as
    # build_cell_stacks.py's build_cell: a cell missing frames should say so.
    paths = []
    for e in epochs:
        try:
            paths.append(resolve_local(int(e["science_filefracday"]), DIFF_PRODUCT,
                                       field=field, ccdid=ccdid, qid=qid))
        except Exception as exc:
            if cell.get("quadrant_ambiguous"):
                return None, ("quadrant label predates the resolve fix - its frames "
                              "cannot be identified, and guessing would use the wrong "
                              "patch of sky")
            return None, "epoch %s unresolved: %s" % (e["science_filefracday"], exc)

    with fits.open(paths[0]) as hdul:
        hdu = hdul[1] if len(hdul) > 1 and hdul[1].data is not None else hdul[0]
        grid_wcs = WCS(hdu.header)
        grid_shape = hdu.data.shape
    out_wcs, out_shape = downsampled_grid(grid_wcs, grid_shape, SIZE)

    ref_name = None
    # Display stats, computed ONCE per cell and shared across every epoch —
    # mirrors build_cell_stacks.py's "ONE shared stretch across all epochs"
    # choice: per-frame autoscaling would renormalise each image and erase
    # exactly the brightness changes the viewer exists to reveal. Written into
    # the manifest so ui/cell-aladin.js never has to guess a stretch or pull
    # stats out of a multi-MB array client-side.
    ref_name, ref_sky_mu, ref_sky_sigma = None, None, None
    fc = FILTERCODE.get(cell["fid"], "zr")
    rp = reference_path(field, ccdid, qid, fc)
    if rp:
        try:
            with fits.open(rp) as hdul:
                hdu = hdul[0] if hdul[0].data is not None else hdul[1]
                ref_arr, _ = reproject_interp((hdu.data.astype(float), WCS(hdu.header)),
                                              out_wcs, shape_out=out_shape)
            _write_fits(ref_arr, out_wcs, os.path.join(out_dir, "ref_tan.fits"))
            ref_name = "ref_tan.fits"
            finite = ref_arr[np.isfinite(ref_arr)]
            if finite.size:
                ref_sky_mu = float(np.median(finite[::7] if finite.size > 2_000_000 else finite))
                ref_sky_sigma = robust_sigma(ref_arr)
        except Exception as exc:
            print("     ! unusable reference for %s (%s: %s) — ref layer skipped"
                  % (key, type(exc).__name__, exc))

    out_epochs = []
    diff_sigma = None
    for e, p in zip(epochs, paths):
        with fits.open(p) as hdul:
            hdu = hdul[1] if len(hdul) > 1 and hdul[1].data is not None else hdul[0]
            data = hdu.data.astype(float)
            wcs = WCS(hdu.header)
        arr, _ = reproject_interp((data, wcs), out_wcs, shape_out=out_shape)
        if diff_sigma is None:            # from the grid (first) epoch only — shared stretch
            diff_sigma = robust_sigma(arr)
        # Display-only transform (see NOISE_FLOOR_SIGMA above) — diff_sigma
        # above and in the manifest is measured on the RAW array, honestly
        # describing this epoch's real per-pixel noise; only the WRITTEN copy
        # is floored/clipped, same split index.html's build_change_overlay.py
        # already uses (measure honest stats, then bake a display version).
        disp = arr.copy()
        floor = NOISE_FLOOR_SIGMA * diff_sigma
        ceil = CEILING_SIGMA * diff_sigma
        disp[np.abs(disp) < floor] = np.nan
        np.clip(disp, -ceil, ceil, out=disp)
        fname = "%d.fits" % e["science_filefracday"]
        _write_fits(disp, out_wcs, os.path.join(out_dir, fname))
        out_epochs.append({
            "exposure_id": e["exposure_id"],
            "science_filefracday": e["science_filefracday"],
            "obsjd": e["obsjd"], "ut_date": e["ut_date"],
            "n_detections": e.get("n_detections"), "n_short": e.get("n_short"),
            "fits": fname,
        })

    manifest = {
        "cell_key": key, "field": field, "ccdid": ccdid, "qid": qid, "fid": cell["fid"],
        "ra": cell["ra"], "dec": cell["dec"],
        "grid": {"width": out_shape[1], "height": out_shape[0]},
        "ref": ref_name,
        "ref_sky_mu": ref_sky_mu, "ref_sky_sigma": ref_sky_sigma,
        "diff_sigma": diff_sigma,
        "epochs": out_epochs,
    }
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, separators=(",", ":"))
    return manifest, "%d epochs, ref=%s" % (len(out_epochs), bool(ref_name))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="rebuild cells already baked")
    ap.add_argument("--cell", help="build only this cell_key")
    a = ap.parse_args()

    if not os.path.exists(CELLS_JSON):
        sys.exit("no %s - run: .venv/bin/python ui/build_survey.py --verify" % CELLS_JSON)
    cells = json.load(open(CELLS_JSON))["rows"]
    if a.cell:
        cells = [c for c in cells if c["cell_key"] == a.cell] or sys.exit("no such cell")

    ok, skipped = 0, 0
    for c in cells:
        t0 = time.time()
        try:
            man, msg = build_cell_fits(c, a.force)
        except Exception as exc:
            man, msg = None, "%s: %s" % (type(exc).__name__, exc)
        if man is None:
            skipped += 1
            print("  skip %-20s %s" % (c["cell_key"], msg))
            continue
        ok += 1
        print("  ok   %-20s %-28s %5.1fs" % (c["cell_key"], msg, time.time() - t0))

    print("\n%d cells with FITS, %d skipped -> %s" % (ok, skipped, OUT_ROOT))


if __name__ == "__main__":
    main()
