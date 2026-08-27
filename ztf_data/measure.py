"""
Stage 3b: segmentation maps -> a catalog + (3,63,63) cutout triplets.

The catalog is the Stage-4 handoff. Its stamp basenames ARE the join key
(`row_id` in ztf_classification/pipeline.py), so naming is a contract, not style.
"""

import os

import numpy as np
from astropy.nddata import Cutout2D
from astropy.table import Table, vstack
from photutils.segmentation import SourceCatalog, deblend_sources

# REGRESSION-LOCKED: deblend splits merged clumps. On the 2020 fixture this
# turns 106 raw detections into 114 (+3 pos, +5 neg) - real pairs separated,
# no over-splitting. Needs scikit-image (lazy watershed backend).
N_PIXELS = 5
N_LEVELS = 32
CONTRAST = 0.001

# braai's input contract - keras.Input(shape=(63,63,3)). Not a knob.
STAMP = 63

# photutils 3.0 names. The old ones (npixels/xcentroid) warn and die in 4.0.
COLUMNS = [
    "label",
    "x_centroid",
    "y_centroid",
    "sky_centroid",
    "segment_flux",
    "segment_flux_err",
    "max_value",
    "elongation",
    "ellipticity",
    "orientation",
    "area",
]


def _measure_one_sign(img, segm, rms, cut_wcs, sign):
    """Deblend + measure ONE sign. Negatives are measured on -img so their
    flux comes out positive; `sign` preserves the truth."""
    segm_db = deblend_sources(
        img, segm, n_pixels=N_PIXELS, n_levels=N_LEVELS, contrast=CONTRAST
    )
    cat = SourceCatalog(img, segm_db, wcs=cut_wcs, error=rms)
    tbl = cat.to_table(columns=COLUMNS)
    tbl["ra"] = tbl["sky_centroid"].ra.deg
    tbl["dec"] = tbl["sky_centroid"].dec.deg
    tbl.remove_column("sky_centroid")  # does not round-trip to .ecsv
    tbl["snr"] = tbl["segment_flux"] / tbl["segment_flux_err"]
    tbl["sign"] = sign
    return tbl


def measure_catalog(img, segm_pos, segm_neg, rms, cut_wcs):
    """Both signs -> ONE sign-tagged table with a GLOBAL row_index.

    row_index is assigned AFTER the stack, so it is continuous across the sign
    flip. photutils' `label` restarts at 1 for the negatives - it is kept for
    provenance but is NOT unique and must never be used as identity.
    """
    pos = _measure_one_sign(img, segm_pos, rms, cut_wcs, sign=1)
    neg = _measure_one_sign(-img, segm_neg, rms, cut_wcs, sign=-1)
    catalog = vstack([pos, neg])
    catalog["row_index"] = np.arange(len(catalog))
    return catalog


def stamp_name(sign, row_index, width=3):
    """The Stage-4 join key. width=3 matches the existing cutouts_2020 fixture.

    Width is per-run and set by assign_stamp_ids from the detection count: a
    central-1000 run (114) keeps the fixture's 3 digits exactly, while a full-frame
    run (~950, and more on a crowded field) widens instead of colliding.
    """
    tag = "p" if sign > 0 else "m"
    return f"src_{tag}1_{row_index:0{width}d}.npy"


def assign_stamp_ids(catalog, frame_shape, cutouts_dir=""):
    """
    Assign stamp_path + on_edge WITHOUT reading pixels.

    on_edge is pure geometry (centroid vs frame bounds) and stamp_path is pure
    naming - neither needs sci/ref. Split out of cut_triplets so the motion path,
    which reads only the diff, can skip building a 40-frame deep template.
    """

    # Zero-padding width from THIS run's detection count. Full-frame yields ~950+
    # (957 on the BE5 frame), so a fixed %03d would collide on a crowded field;
    # <=1000 detections still produce the fixture's exact 3-digit basenames.
    width = max(3, len(str(max(len(catalog) - 1, 0))))
    half = STAMP / 2
    H, W = frame_shape
    catalog["stamp_path"] = [
        os.path.join(cutouts_dir, stamp_name(r["sign"], r["row_index"], width))
        for r in catalog
    ]
    catalog["on_edge"] = [
        float(r["x_centroid"]) < half or float(r["y_centroid"]) < half
        or float(r["x_centroid"]) > W - half or float(r["y_centroid"]) > H - half
        for r in catalog
    ]
    return catalog


def cut_triplets(catalog, sci_img, ref_img, diff_img, cutouts_dir):
    """Cut (3,63,63) sci/ref/diff at each centroid. Adds stamp_path + on_edge.

    Valid ONLY because all three share one aligned pixel grid. mode="partial"
    zero-pads edge sources so every stamp is uniformly 63x63. Naming + on_edge
    come from assign_stamp_ids - this function only writes pixels.
    """
    # The centroids come from diff_img. If a channel was cropped by a DIFFERENT rule
    # the same (x,y) indexes different sky in each channel - which fails loudly only
    # for centroids past the smaller frame and is SILENT for the rest. run_field used
    # to crop sci/ref to central-1000 while diff honoured full_frame; this is the
    # assert that catches it. Shapes, not sizes: any mismatch is the same bug.
    if not (sci_img.shape == ref_img.shape == diff_img.shape):
        raise ValueError(
            f"channel shape mismatch: sci{sci_img.shape} ref{ref_img.shape} "
            f"diff{diff_img.shape} - every channel must be cropped by the same region "
            f"rule as the diff the centroids were measured on"
        )
    os.makedirs(cutouts_dir, exist_ok=True)
    assign_stamp_ids(catalog, diff_img.shape, cutouts_dir)
    kw = dict(size=STAMP, mode="partial", fill_value=0.0)
    for row in catalog:
        x, y = float(row["x_centroid"]), float(row["y_centroid"])
        # float32: braai L2-normalizes per channel, so f64 buys no precision and
        # doubles the stamp (95 KB -> 48 KB). At ~1000 detections/exposure that is
        # the difference between 9.5 GB and 4.8 GB across a 200-exposure batch.
        np.save(row["stamp_path"], np.stack([
            Cutout2D(sci_img, (x, y), **kw).data,
            Cutout2D(ref_img, (x, y), **kw).data,
            Cutout2D(diff_img, (x, y), **kw).data,
        ], axis=0).astype(np.float32))
    return catalog


def write_catalog(catalog, out_path):
    catalog.write(out_path, format="ascii.ecsv", overwrite=True)
    return out_path
