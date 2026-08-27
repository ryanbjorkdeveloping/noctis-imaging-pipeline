"""Bake per-CELL image stacks for the survey UI: N epochs of one patch of sky,
all resampled onto ONE shared pixel grid so they can be blinked against each other.

WHY THIS EXISTS
    views/cells.json already reassembles a cell's repeat visits as NUMBERS. This
    turns them into PICTURES. A cell is one ZTF quadrant (~0.83 deg); its epochs
    are revisits of that exact patch, and the change between them is the whole
    point of difference imaging. Stacked and blinked, you SEE it.

THE LOAD-BEARING BIT — the epochs are NOT already aligned.
    ZTF's official difference (scimrefdiffimg) is built on the SCIENCE frame's
    grid, and consecutive visits are dithered. Measured on 518/c01/q1: CRVAL
    moves ~0.0005 deg = ~1.8 arcsec = ~1.8 px between epochs. Stacking the raw
    arrays would shift every star by a couple of pixels per frame, and blinking
    that manufactures apparent MOTION on sources that never moved - the exact
    false-positive class the whole motion pipeline exists to avoid.
    So every epoch is reprojected through its own WCS onto the grid epoch's WCS.

WHAT IS BEING SHOWN - two layers, which is the whole trick
    sky.png     ZTF's DEEP REFERENCE co-add for this quadrant (kind="ref",
                refimg.fits) - tens of frames stacked, so it is far deeper and
                cleaner than any single epoch. This is the static sky, and it is
                literally the image every epoch's difference was measured
                against. ONE download per cell, not one per epoch.
    <ffd>.png   That epoch's CHANGE overlay, cut from ZTF's official difference:
                transparent wherever nothing changed, amber where something
                brightened, cyan where something faded.

    Stacked, that is "the sky, and what changed in it at each visit" - blink
    through the epochs and the changes are what move.

    Rendering the raw difference as flat greyscale instead was tried first and
    is much worse: a difference image IS subtraction residual, so at any sane
    stretch most of the frame is noise and it reads as static (the same trap
    ui/build_change_overlay.py documents for the single-field viewer).

    Display transform only. Detection and measurement ran on the FULL-RESOLUTION
    official diff; nothing here feeds back into the catalog. The overlay's
    transparency floor is deliberately the pipeline's own detect.N_SIGMA, so
    what you see lit is the population the catalog was drawn from.

    One SHARED stretch across all epochs of a cell (computed from the grid
    epoch). Per-frame autoscaling would renormalise each image and erase exactly
    the brightness changes the stack exists to reveal.

Run:  .venv/bin/python ui/build_cell_stacks.py [--force] [--cell KEY] [--size 1024]
      Needs IRSA for the deep references (~40 MB per cell, downloaded once and
      cached under $ZTFDATA/ref). --no-sky skips them and bakes overlays only.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sqlite3
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import config  # noqa: F401  LOAD-BEARING: sets $ZTFDATA before ztfquery imports

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp

from ztf_data.fetch import resolve_local
from ztf_data.paths import OWN_ROOT

DIFF_PRODUCT = "scimrefdiffimg.fits.fz"

OUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "survey", "cells")
CELLS_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "survey", "views", "cells.json")
DB_PATH = os.path.join(OWN_ROOT, "survey.db")

SIZE = 1024          # long side of the baked PNG, px
MARKER_TOP_SNR = 150  # brightest detections baked per epoch, on top of every streak candidate

# ── change-overlay display constants ────────────────────────────────────────
# Below FLOOR the overlay is fully transparent, by FULL it is fully opaque, both
# in units of the cell's own difference noise. FLOOR is the pipeline's own
# detection threshold (detect.N_SIGMA = 3.0) on purpose: what lights up is the
# same population the catalog was drawn from, so the picture and the numbers
# cannot tell different stories.
OVER_FLOOR_SIGMA = 3.0
OVER_FULL_SIGMA = 12.0
# Alpha rises as the ramp to this power. 0.5, not 1.0: measured on the live page,
# a linear ramp left the MEAN overlay contribution at ~2 of 255 levels, because
# most residuals sit near the floor - only the handful of very strong ones were
# visible at all, so the layer read as almost empty. The square root lifts a
# marginal 4-sigma residual from alpha 0.11 to 0.33 while preserving the
# ordering, so stronger is still more opaque.
OVER_GAMMA = 0.5
POS_RGB = (255, 176, 71)    # brightened / appeared - amber
NEG_RGB = (94, 200, 255)    # faded             - cyan
# Redundant with hue on purpose (amber/cyan survives colour blindness poorly on
# its own): positive is also the WARM end and is drawn over a black sky, so the
# sign is legible from brightness ordering too.

# ── deep-reference (sky) display constants ──────────────────────────────────
SKY_BLACK_SIGMA = 1.5       # black point, sky median + this many sigma -> 0
SKY_SOFT = 30.0             # asinh softening; higher = more faint detail
SKY_SPAN_SIGMA = 300.0      # white point, in sigma above the black point


# ── the shared grid ──────────────────────────────────────────────────────────
def downsampled_grid(wcs, shape, size):
    """A TAN grid with the same sky footprint as `wcs`, binned down to `size`.

    Built from pixel_scale_matrix rather than by copying CD/PC/CDELT keywords:
    ZTF writes the scale under more than one convention, and a whitelist that
    misses the one in use silently falls back to 1 deg/px (a bug this repo has
    already been bitten by once, in ui/bootstrap.sh).
    """
    ny, nx = shape
    f = max(nx, ny) / float(size)
    out_shape = (int(round(ny / f)), int(round(nx / f)))

    out = WCS(naxis=2)
    out.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    out.wcs.crval = wcs.wcs.crval
    # pixel k in the output covers pixels [k*f, (k+1)*f) of the input; FITS
    # pixels are 1-based and centred, hence the half-pixel shifts.
    out.wcs.crpix = [(wcs.wcs.crpix[0] - 0.5) / f + 0.5,
                     (wcs.wcs.crpix[1] - 0.5) / f + 0.5]
    out.wcs.cd = wcs.pixel_scale_matrix * f
    return out, out_shape


def robust_sigma(a):
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return 1.0
    sample = finite[::7] if finite.size > 2_000_000 else finite
    med = float(np.median(sample))
    return float(1.4826 * np.median(np.abs(sample - med))) or 1.0


def overlay_png(arr, sigma, path):
    """Signed difference -> RGBA change overlay.

    Transparent where nothing changed (so the deep sky underneath shows through),
    amber where flux appeared, cyan where it faded. Alpha ramps from the
    detection threshold to fully opaque, so a marginal residual reads as faint
    rather than being either hidden or shouted.
    """
    from PIL import Image

    z = np.nan_to_num(arr, nan=0.0) / sigma
    mag = np.abs(z)
    alpha = np.clip((mag - OVER_FLOOR_SIGMA) / (OVER_FULL_SIGMA - OVER_FLOOR_SIGMA), 0.0, 1.0)
    alpha = np.power(alpha, OVER_GAMMA)
    alpha[~np.isfinite(arr)] = 0.0

    pos = z > 0
    rgb = np.empty(arr.shape + (3,), np.uint8)
    for c in range(3):
        rgb[..., c] = np.where(pos, POS_RGB[c], NEG_RGB[c])

    a8 = (alpha * 255.0).astype(np.uint8)
    # FITS rows run bottom-up, PNG rows run top-down.
    im = Image.fromarray(np.dstack([rgb, a8])[::-1], mode="RGBA")
    im.save(path, optimize=True)
    return os.path.getsize(path), float((alpha > 0).mean())


def sky_png(arr, path):
    """Deep reference co-add -> 8-bit grey, black sky, asinh so faint sources live.

    Black point sits just above the measured sky so empty sky is truly 0 rather
    than a grey fog; asinh (not linear) is what keeps faint stars visible while
    a bright star still saturates cleanly.
    """
    from PIL import Image

    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0
    med = float(np.median(finite[::7]))
    sig = robust_sigma(arr)
    z = (arr - (med + SKY_BLACK_SIGMA * sig)) / (SKY_SPAN_SIGMA * sig)
    g = np.arcsinh(np.clip(z, 0, None) * SKY_SOFT) / np.arcsinh(SKY_SOFT)
    grey = (np.clip(np.nan_to_num(g, nan=0.0), 0, 1) * 255.0).astype(np.uint8)
    Image.fromarray(grey[::-1], mode="L").save(path, optimize=True)
    return os.path.getsize(path)


# ── ZTF's deep reference for a cell ─────────────────────────────────────────
def reference_path(field, ccdid, qid, filtercode, download=True):
    """Resolve (downloading once if needed) this quadrant's deep reference co-add.

    One file per (field, ccdid, qid, filter) for the whole survey - it is the
    static sky every epoch of this cell was differenced against, so it is both
    the cheapest and the most correct base image to show under the overlays.
    """
    pat = os.path.join(os.environ["ZTFDATA"], "ref", "**",
                       "ztf_%06d_%s_c%02d_q%d_refimg.fits" % (field, filtercode, ccdid, qid))
    hits = glob.glob(pat, recursive=True)
    if hits:
        return hits[0]
    if not download:
        return None
    from ztfquery import query
    zq = query.ZTFQuery()
    zq.load_metadata(kind="ref", sql_query="field=%d AND ccdid=%d AND qid=%d AND filtercode='%s'"
                                           % (field, ccdid, qid, filtercode))
    if len(zq.metatable) == 0:
        return None
    zq.download_data("refimg.fits", show_progress=False, nprocess=1)
    hits = glob.glob(pat, recursive=True)
    return hits[0] if hits else None


# ── detections, for the overlay ──────────────────────────────────────────────
def epoch_markers(con, exposure_id, out_wcs, out_shape):
    rows = con.execute(
        """SELECT det_uid, row_id, ra, dec, snr, sign, streak_route, novelty_status,
                  elongation, final_verdict
             FROM detections WHERE exposure_id = ?""", (exposure_id,)).fetchall()
    if not rows:
        return []
    short = [r for r in rows if r[6] == "SHORT_NEA_CANDIDATE"]
    rest = sorted((r for r in rows if r[6] != "SHORT_NEA_CANDIDATE"),
                  key=lambda r: -(r[4] or 0))[:MARKER_TOP_SNR]
    keep = short + rest

    ra = np.array([r[2] for r in keep], float)
    dec = np.array([r[3] for r in keep], float)
    x, y = out_wcs.world_to_pixel_values(ra, dec)
    ny, nx = out_shape

    out = []
    for i, r in enumerate(keep):
        if not (0 <= x[i] < nx and 0 <= y[i] < ny):
            continue
        out.append({
            "det_uid": r[0], "row_id": r[1],
            "x": round(float(x[i]), 1),
            "y": round(float(ny - 1 - y[i]), 1),   # PNG row order
            "snr": round(float(r[4]), 1) if r[4] is not None else None,
            "sign": r[5],
            "route": r[6],
            "novelty_status": r[7],
            "elongation": round(float(r[8]), 2) if r[8] is not None else None,
            "verdict": r[9],
        })
    return out


# ── one cell ─────────────────────────────────────────────────────────────────
FILTERCODE = {1: "zg", 2: "zr", 3: "zi"}


def dedupe_epochs(epochs):
    """One entry per REAL visit, sorted by time.

    A cell's epoch list is one row per EXPOSURE, and the same exposure can be
    processed twice - once cropped to the central 1000 px, once full-frame -
    which the store keeps apart because the region is part of a target's
    identity. Both rows carry the SAME science_filefracday and reduce the SAME
    difference image, so left alone they collide on the `<ffd>.png` filename
    and the stack claims two visits where the telescope made one. A blink over
    that pair shows nothing moving, which reads as "nothing happened here"
    rather than "this is one frame shown twice".

    Keep the row with the most detections: same pixels either way, but the
    full-frame run carries far more markers.
    """
    best = {}
    for e in epochs:
        k = e["science_filefracday"]
        if k not in best or (e.get("n_detections") or 0) > (best[k].get("n_detections") or 0):
            best[k] = e
    return sorted(best.values(), key=lambda e: e["obsjd"])


def build_cell(cell, con, size, force, want_sky=True):
    key = cell["cell_key"]
    epochs = dedupe_epochs(cell.get("epochs") or [])
    if len(epochs) < 1:
        return None, "0 real visits - nothing to show"

    out_dir = os.path.join(OUT_ROOT, key)
    manifest_path = os.path.join(out_dir, "manifest.json")
    if os.path.exists(manifest_path) and not force:
        return json.load(open(manifest_path)), "cached"
    os.makedirs(out_dir, exist_ok=True)

    field, ccdid, qid = cell["field"], cell["ccdid"], cell["qid"]

    # resolve every epoch's diff BEFORE doing any work: a cell missing frames
    # should say so, not half-bake.
    paths = []
    for e in epochs:
        try:
            paths.append(resolve_local(int(e["science_filefracday"]), DIFF_PRODUCT,
                                       field=field, ccdid=ccdid, qid=qid))
        except Exception as exc:
            # A quadrant_ambiguous cell records a (ccdid, qid) from before the
            # resolve_local fix, so its frames genuinely cannot be found under
            # the label the store carries. Do NOT relax the glob to "find them
            # anyway": the file that turns up belongs to a DIFFERENT quadrant,
            # i.e. a different patch of sky, and stacking it would fabricate an
            # aligned sequence out of unrelated frames.
            if cell.get("quadrant_ambiguous"):
                return None, ("quadrant label predates the resolve fix - its frames "
                              "cannot be identified, and guessing would stack the "
                              "wrong patch of sky")
            return None, "epoch %s unresolved: %s" % (e["science_filefracday"], exc)

    # grid = first epoch by time. Arbitrary but FIXED, and recorded in the
    # manifest, so a rebuild lands on the same pixels.
    with fits.open(paths[0]) as hdul:
        hdu = hdul[1] if len(hdul) > 1 and hdul[1].data is not None else hdul[0]
        grid_wcs = WCS(hdu.header)
        grid_data = hdu.data.astype(float)
    out_wcs, out_shape = downsampled_grid(grid_wcs, grid_data.shape, size)

    # ── the static sky: ZTF's deep reference, reprojected onto the same grid ──
    sky, sky_meta = None, None
    if want_sky:
        fc = FILTERCODE.get(cell["fid"], "zr")
        rp = reference_path(field, ccdid, qid, fc)
        if rp:
            # A reference truncated by an interrupted download raises deep inside
            # astropy. Do NOT let that kill the run: the sky is one optional
            # layer, and the epochs (the actual subject) need none of it. Drop
            # the bad file so the next run re-fetches it, and carry on.
            try:
                with fits.open(rp) as hdul:
                    hdu = hdul[0] if hdul[0].data is not None else hdul[1]
                    ref_arr, _ = reproject_interp((hdu.data.astype(float), WCS(hdu.header)),
                                                  out_wcs, shape_out=out_shape)
                    nframes = hdu.header.get("NFRAMES")
                    maglim = hdu.header.get("MAGLIM") or hdu.header.get("MAGLIMIT")
                sky_png(ref_arr, os.path.join(out_dir, "sky.png"))
                sky = "sky.png"
                sky_meta = {
                    "product": "ZTF deep reference co-add (refimg.fits)",
                    "n_frames": int(nframes) if nframes else None,
                    "maglimit": round(float(maglim), 2) if maglim else None,
                    "note": ("The static sky this cell's differences were measured against - "
                             "not one of the epochs. Deeper than any single visit."),
                }
            except Exception as exc:
                print("     ! unusable reference for %s (%s: %s) - removed, will refetch"
                      % (key, type(exc).__name__, exc))
                try:
                    os.remove(rp)
                except OSError:
                    pass

    out_epochs, total_bytes = [], 0
    sigma = None
    for e, p in zip(epochs, paths):
        with fits.open(p) as hdul:
            hdu = hdul[1] if len(hdul) > 1 and hdul[1].data is not None else hdul[0]
            data = hdu.data.astype(float)
            wcs = WCS(hdu.header)

        arr, _ = reproject_interp((data, wcs), out_wcs, shape_out=out_shape)
        if sigma is None:                      # ONE stretch for the whole cell
            sigma = robust_sigma(arr)

        png = "%d.png" % e["science_filefracday"]
        nbytes, lit = overlay_png(arr, sigma, os.path.join(out_dir, png))
        total_bytes += nbytes

        out_epochs.append({
            "lit_fraction": round(lit, 5),
            "exposure_id": e["exposure_id"],
            "science_filefracday": e["science_filefracday"],
            "obsjd": e["obsjd"],
            "ut_date": e["ut_date"],
            "n_detections": e.get("n_detections"),
            "n_short": e.get("n_short"),
            "image": png,
            "markers": epoch_markers(con, e["exposure_id"], out_wcs, out_shape),
        })

    manifest = {
        "cell_key": key,
        "field": field, "ccdid": ccdid, "qid": qid, "fid": cell["fid"],
        "ra": cell["ra"], "dec": cell["dec"],
        "quadrant_ambiguous": bool(cell.get("quadrant_ambiguous")),
        "grid": {
            "width": out_shape[1], "height": out_shape[0],
            "reference_filefracday": epochs[0]["science_filefracday"],
            "pixel_scale_arcsec": round(
                float(np.sqrt(np.abs(np.linalg.det(out_wcs.pixel_scale_matrix)))) * 3600.0, 3),
            "note": ("Every epoch reprojected through its own WCS onto this grid. "
                     "ZTF dithers between visits (~2 px here), so the raw frames are "
                     "NOT aligned; without this step a blink shows fake motion."),
        },
        "sky": sky,
        "sky_display": sky_meta,
        "display": {
            "product": "ZTF official difference image (scimrefdiffimg)",
            "sigma": round(float(sigma), 4),
            "floor_sigma": OVER_FLOOR_SIGMA,
            "full_sigma": OVER_FULL_SIGMA,
            "meaning": "transparent = unchanged, amber = brightened, cyan = faded",
            "shared_stretch": True,
            "note": ("Display transform only, computed once per cell and applied to "
                     "every epoch. Detection ran on the full-resolution diff, and the "
                     "transparency floor is the pipeline's own 3-sigma threshold."),
        },
        "epochs": out_epochs,
    }
    # The PNG name is derived from the filefracday, so two epochs sharing one
    # would silently overwrite each other and the manifest would claim more
    # visits than there are frames. dedupe_epochs prevents it; this proves it.
    names = [e["image"] for e in out_epochs]
    if len(set(names)) != len(names):
        raise AssertionError("%s: %d epochs share only %d image files"
                             % (key, len(names), len(set(names))))

    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, separators=(",", ":"))
    return manifest, "%d visits, %.1f MB" % (len(out_epochs), total_bytes / 1e6)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="rebuild cells already baked")
    ap.add_argument("--cell", help="build only this cell_key")
    ap.add_argument("--size", type=int, default=SIZE)
    ap.add_argument("--no-sky", action="store_true",
                    help="skip the deep reference base layer (no IRSA needed)")
    a = ap.parse_args()

    if not os.path.exists(CELLS_JSON):
        sys.exit("no %s - run: .venv/bin/python ui/build_survey.py --verify" % CELLS_JSON)
    cells = json.load(open(CELLS_JSON))["rows"]
    if a.cell:
        cells = [c for c in cells if c["cell_key"] == a.cell] or sys.exit("no such cell")

    os.makedirs(OUT_ROOT, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    index, skipped = [], []
    for c in cells:
        t0 = time.time()
        # One bad cell must not discard the other 25. Failures are recorded as
        # skips with their reason, the same as the deliberate ones.
        try:
            man, msg = build_cell(c, con, a.size, a.force, want_sky=not a.no_sky)
        except Exception as exc:
            man, msg = None, "%s: %s" % (type(exc).__name__, exc)
        if man is None:
            skipped.append((c["cell_key"], msg))
            print("  skip %-20s %s" % (c["cell_key"], msg))
            continue
        print("  ok   %-20s %-24s %5.1fs" % (c["cell_key"], msg, time.time() - t0))
        # the LATEST epoch is the representative thumbnail — most complete detection
        # set, and "what does this cell look like right now" is the natural default.
        # Stashed here (not left for the frontend to guess) so the grid page can draw
        # a thumbnail from THIS index alone, no per-cell manifest fetch required.
        index.append({
            "cell_key": man["cell_key"], "ra": man["ra"], "dec": man["dec"],
            "n_epochs": len(man["epochs"]),
            "has_sky": bool(man.get("sky")),
            "width": man["grid"]["width"], "height": man["grid"]["height"],
            "span_hours": round((man["epochs"][-1]["obsjd"] - man["epochs"][0]["obsjd"]) * 24.0, 3),
            "thumb": man["epochs"][-1]["image"],
        })
    con.close()

    with open(os.path.join(OUT_ROOT, "index.json"), "w") as fh:
        json.dump({
            "note": ("Every cell with at least one real visit, baked as a difference-image "
                     "stack on one shared pixel grid. 2+ epochs also get a blink comparator; "
                     "a single visit still gets its full difference-image view, just nothing "
                     "to blink against yet."),
            "stacks": index,
            "skipped": [{"cell_key": k, "reason": r} for k, r in skipped],
        }, fh, indent=1)
    print("\n%d stacks, %d cells skipped -> %s" % (len(index), len(skipped), OUT_ROOT))


if __name__ == "__main__":
    main()
