"""Stage-4 motion feeder: a run_field output -> fast-asteroid (streak) candidates.

Cuts 144x144 stamps from the DIFF ONLY (never sci/ref) and scores them through
DeepStreaks - so this discovery path is NOT blocked by the registration dipole
that limits the point-source (braai) path. The scorer itself lives in
ztf_classification/pipeline.py and is reused as-is; this module is the bridge.
"""

import numpy as np

from ztf_data.detect import load_diff, central_cutout, full_frame_cutout, CUTOUT_SIZE

from astropy.io import fits

def reconstruct_diff_img(diff_path, full_frame=False):
    """Re-derive the exact diff array run_field detected on.

    run_field returns only diff_path, but streak stamps must be cut from the SAME
    frame the catalog centroids live in. Same load_diff (.fz->HDU1) + the matching
    cutout + nan_to_num => byte-identical to run_field's internal diff_img.

    full_frame MUST match the run that produced the catalog. It is not cosmetic:
    rebuilding the central-1000 for a full-frame catalog puts 89% of the centroids
    OUT OF BOUNDS, cut_streak_stamp gray-pads those windows, and DeepStreaks routed
    546 flat slabs SHORT_NEA_CANDIDATE - a wall of false discoveries with no error
    raised anywhere. See the assert in find_streak_candidates.
    """
    diff_full, diff_wcs = load_diff(diff_path)
    cut = (full_frame_cutout(diff_full) if full_frame
           else central_cutout(diff_full, size=CUTOUT_SIZE))
    return np.nan_to_num(cut.data)

import os

from astropy.coordinates import SkyCoord
from astropy.table import Table
import astropy.units as u

from ztf_classification.pipeline import build_streak_batch, run_streak_subprocess

CANDIDATE_COLUMNS = ["row_id", "ra", "dec", "elongation", "orientation", "route",
                     "rb_pass", "kd_pass", "sl_short"]


def _row_id(stamp_path):
    """Stamp basename without extension = the join key (matches pipeline.py)."""
    return os.path.splitext(os.path.basename(str(stamp_path)))[0]


def _f(cell):
    """Coerce a catalog cell to a plain float, units and all.

    find_streak_candidates is public and takes EITHER an in-memory photutils catalog
    (where `orientation` is a Quantity in deg -> bare float() raises 'only dimensionless
    scalar quantities...') OR a disk-read ecsv catalog (a plain Column). A Quantity has
    .value; a numpy/Column scalar does not, so getattr falls back to the cell itself.
    """
    return float(getattr(cell, "value", cell))


def sky_position_angle(wcs, ra, dec, orientation_deg, step_px=5.0):
    """Sky position angle (deg E of N, folded to [0,180)) of a streak whose pixel
    major axis is `orientation_deg` (photutils' angle CCW from +x) at (ra, dec).

    A streak already encodes its direction of motion in ONE exposure - measured on
    the BE5 control, this recovers the true track PA to ~1-3 deg for well-resolved
    fragments (elong>3), ~13 deg for marginal ones. So it is a PAIR-GATE, not the
    final direction (that comes from the track fit). Folded to [0,180) because a
    streak axis is undirected (head/tail ambiguous in a single frame).

    Done through the WCS (not from CD-matrix rotation) so it is correct wherever the
    source sits: step step_px along the pixel major axis, read the sky PA of that step.
    """
    p0 = SkyCoord(ra * u.deg, dec * u.deg)
    px, py = wcs.world_to_pixel(p0)
    import numpy as np
    th = np.radians(orientation_deg)
    p1 = SkyCoord(*wcs.pixel_to_world_values(px + step_px * np.cos(th),
                                             py + step_px * np.sin(th)), unit="deg")
    return float(p0.position_angle(p1).deg % 180.0)


def find_streak_candidates(catalog, diff_img, scratch_dir, elong_thresh=1.5):
    """Score the catalog's elongated detections through DeepStreaks -> candidates table.

    Reuses pipeline.build_streak_batch (cut 144x144 dark-on-gray) + run_streak_subprocess
    (.venv_streaks cascade). One row per elongated, non-edge detection, with its route.
    0 elongated -> empty table (build_streak_batch's np.stack crashes on an empty list).
    """
    # The catalog's centroids and diff_img MUST come from the same detection region.
    # A mismatch does not raise anywhere downstream: cut_streak_stamp simply gray-pads
    # every out-of-bounds window and DeepStreaks happily routes the resulting slabs
    # SHORT_NEA_CANDIDATE. Cheap assert, caught a 546-false-positive run.
    H, W = diff_img.shape
    if len(catalog) and (catalog["x_centroid"].max() >= W or catalog["y_centroid"].max() >= H):
        raise ValueError(
            f"catalog centroids exceed diff_img {diff_img.shape}: "
            f"x<={catalog['x_centroid'].max():.0f} y<={catalog['y_centroid'].max():.0f}. "
            "reconstruct_diff_img(full_frame=) does not match the run that made this catalog."
        )

    results = [
        {"row_id": _row_id(r["stamp_path"]),
         "elongation": float(r["elongation"]),
         "on_edge": bool(r["on_edge"])}
        for r in catalog
    ]
    qualifying = [r for r in results if r["elongation"] > elong_thresh and not r["on_edge"]]
    if not qualifying:
        return Table(names=CANDIDATE_COLUMNS,
                     dtype=[str, float, float, float, float, str, bool, bool, bool])

    batch, ids = build_streak_batch(results, catalog, diff_img, elong_thresh=elong_thresh)
    routes = run_streak_subprocess(batch, ids, scratch_dir)

    by_id = {_row_id(r["stamp_path"]): r for r in catalog}
    rows = []
    for rid in ids:
        crow, d = by_id[rid], routes[rid]
        rows.append((rid, _f(crow["ra"]), _f(crow["dec"]),
                     _f(crow["elongation"]), _f(crow["orientation"]), d["route"],
                     bool(d["rb_pass"]), bool(d["kd_pass"]), bool(d["sl_short"])))
    return Table(rows=rows, names=CANDIDATE_COLUMNS)


def streaks_from_run(field_result, scratch_dir=None):
    """Full per-run pass: load catalog + re-derive diff, score, write streak_candidates.ecsv.

    Attaches the two columns the LINKER needs and the sweep used to drop: `obsjd`
    (exposure time, from the diff header) and `streak_pa` (each streak's sky position
    angle, the single-exposure motion-direction gate). ra/dec/obsjd/streak_pa is the
    minimal observation a track is built from.
    """
    cat = Table.read(field_result.catalog_path)
    diff_full, diff_wcs = load_diff(field_result.diff_path)
    diff_img = reconstruct_diff_img(field_result.diff_path,
                                    full_frame=field_result.full_frame)
    if scratch_dir is None:
        scratch_dir = os.path.join(field_result.run_dir, "streak_scratch")
    table = find_streak_candidates(cat, diff_img, scratch_dir)

    # time: the science exposure's OBSJD (HDU 1 for .fz-compressed diffs)
    hdu = 1 if str(field_result.diff_path).endswith(".fz") else 0
    obsjd = float(fits.getheader(str(field_result.diff_path), hdu) ["OBSJD"])
    table["obsjd"] = obsjd

    # direction: convert each streak's pixel orientation to a sky position angle
    table["streak_pa"] = [
        sky_position_angle(diff_wcs, float(r["ra"]), float(r["dec"]), float(r["orientation"]))
        for r in table
    ]

    table.write(os.path.join(field_result.run_dir, "streak_candidates.ecsv"),
                format="ascii.ecsv", overwrite=True)
    
    return table