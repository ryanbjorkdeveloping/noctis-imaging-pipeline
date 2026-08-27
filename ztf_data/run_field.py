"""Stage 1-3 orchestrator: one grid cell + science epoch -> catalog + cutouts + diff.

The official diff defines the working grid; the science frame and the deep template
are reprojected ONTO it so a centroid found in the diff is valid in every channel.
The catalog is a pure function of the official diff (detection reads only diff_img),
which is what makes the regression stable across template float jitter.
"""

import datetime
import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field as dc_field

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from ztf_data.paths import run_dir, run_paths, template_path
from ztf_data.epochs import query_grid, select_epochs, local_grid
from ztf_data.fetch import download_epochs, resolve_local, resolve_local_many
from ztf_data.align import load_frames, pick_grid, reproject_to, reproject_one
from ztf_data.template import stack_median, save_template
from ztf_data.detect import (
    load_diff, central_cutout, full_frame_cutout, background_rms, detect_both_signs,
    CUTOUT_SIZE,
)
from ztf_data.measure import (
    measure_catalog, cut_triplets, assign_stamp_ids, write_catalog,
)

SCI_PRODUCT = "sciimg.fits"
DIFF_PRODUCT = "scimrefdiffimg.fits.fz"


@dataclass
class FieldResult:
    run_dir: str
    catalog_path: str
    cutouts_dir: str
    diff_path: str
    template_path: str
    science_filefracday: int
    n_detections: int
    # which region the catalog's centroids live in - streaks.py MUST rebuild the
    # matching diff array or every stamp is cut from the wrong coordinate frame.
    full_frame: bool = False
    # provenance the survey store needs. obsjd is the LINKING key (link.py groups by
    # time); corners are the sky footprint of the DETECTION REGION (not the chip), so
    # a full-frame and a central run of the same exposure honestly draw different boxes.
    obsjd: float = 0.0
    ra_center: float = 0.0
    dec_center: float = 0.0
    corners: list = dc_field(default_factory=list)
    # sha1 of the reference set -> the template cache key. Always computed (it is a
    # pure function of the epoch plan), even on streaks_only runs where no template
    # is built - it records which refs WOULD have been used.
    ref_key: str = ""
    science_infobits: int = 0


def ref_key_for(plan):
    """Content hash of the reference set = the template cache key.

    Keyed on the resulting ref LIST, not on (ref_years, n_ref, science_year) that
    chose it: if selection logic ever changes, a different list yields a different
    key rather than silently reusing a stale stack under a matching-looking name.
    """
    ffds = sorted(int(x) for x in plan.refs["filefracday"])
    return hashlib.sha1(",".join(str(x) for x in ffds).encode()).hexdigest()[:16]


def _adopt_legacy_template(legacy_path, manifest_path, out_path, plan):
    """Reuse a pre-hoist run-dir template as the cache entry, if it provably matches.

    Hoisting the template to templates/<ref_key>.fits orphaned every per-run
    deep_template.fits already on disk, and a rebuild costs 5-10 min at ~9 GB peak RSS.
    But adopting one blindly would be a lie: the file is only valid for the ref set it
    was stacked from. The manifest records `reference_filefracdays`, so adopt ONLY when
    that list equals the current plan's - otherwise leave it and re-stack honestly.
    """
    if out_path.exists() or not legacy_path.exists() or not manifest_path.exists():
        return False
    try:
        with open(manifest_path) as f:
            old_refs = json.load(f).get("reference_filefracdays")
    except (OSError, ValueError):
        return False
    if not old_refs:
        return False
    if sorted(int(x) for x in old_refs) != sorted(int(x) for x in plan.refs["filefracday"]):
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:                       # hardlink: same bytes, no copy, no double disk
        os.link(legacy_path, out_path)
    except OSError:
        shutil.copy2(legacy_path, out_path)
    return True


def _build_template(ref_paths, out_path, force):
    """Deep median template on the sharpest ref frame's grid. Cached unless force."""
    if out_path.exists() and not force:
        return fits.getdata(str(out_path)).astype(float), WCS(fits.getheader(str(out_path)))
    frames = load_frames([str(p) for p in ref_paths])
    grid = pick_grid(frames)
    template = stack_median(reproject_to(frames, grid.wcs, grid.data.shape))
    save_template(template, grid.wcs, str(out_path))
    return template, grid.wcs


def _prep_channel(data, wcs, grid_wcs, grid_shape, region):
    """Reproject onto the diff grid, crop to the SAME region as the diff, NaN->0.

    `region` is a callable, not a size: the sci/ref channels must be cropped by the
    identical rule as diff_img, or the catalog's centroids index a different frame
    than the pixels they cut from. This was hardwired to central_cutout(CUTOUT_SIZE)
    while diff_img honoured full_frame, so a full-frame stationary run cut sci/ref
    from ~976 px away (silently, for centroids under ~1031) or raised NoOverlapError
    (for everything beyond it). cut_triplets now asserts the shapes match.
    """
    off = reproject_one(data, wcs, grid_wcs, grid_shape)
    return np.nan_to_num(region(off).data)


def _footprint(wcs, shape):
    """(ra_center, dec_center, [[ra,dec] x4]) for a detection region, CCW from (0,0)."""
    h, w = shape
    xs = [0, w - 1, w - 1, 0]
    ys = [0, 0, h - 1, h - 1]
    ra, dec = wcs.pixel_to_world_values(xs, ys)
    rc, dc = wcs.pixel_to_world_values((w - 1) / 2.0, (h - 1) / 2.0)
    corners = [[round(float(a), 6), round(float(d), 6)] for a, d in zip(ra, dec)]
    return round(float(rc), 6), round(float(dc), 6), corners


def _write_manifest(path, field, ccdid, qid, fid, plan, n_det, streaks_only=False,
                    full_frame=False, region_shape=None, obsjd=None, ra_center=None,
                    dec_center=None, corners=None, ref_key=None):
    """Params + the exact epoch set -> a run you can attribute and reproduce."""
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        sha = None
    manifest = {
        "field": field, "ccdid": ccdid, "qid": qid, "fid": fid,
        "science_filefracday": int(plan.science["filefracday"]),
        "reference_filefracdays": [int(x) for x in plan.refs["filefracday"]],
        "gap_days": None if plan.gap_days is None else round(plan.gap_days, 2),
        "n_reference": len(plan.refs), "n_detections": n_det,
        # survey-store provenance. obsjd is the linking key; the footprint describes
        # the DETECTION REGION, so it differs between a full-frame and a central run.
        "obsjd": obsjd, "ra_center": ra_center, "dec_center": dec_center,
        "corners": corners,
        "ref_key": ref_key,
        # != 0 means a FLAGGED science frame (allowed only via an explicit pin).
        # Motion targets often exist only on bright-sky frames - see query_grid.
        "science_infobits": int(plan.science["infobits"]),
        # streaks_only catalogs carry stamp_path (it IS the row_id contract) but the
        # .npy files were never written - braai on such a run would FileNotFoundError.
        "streaks_only": streaks_only,
        "stamps_written": not streaks_only,
        # detection region: full_frame changes the catalog wholesale (~950 vs 114),
        # so it is recorded here AND keyed into the run dir.
        "full_frame": full_frame,
        "region_shape": list(region_shape) if region_shape is not None else None,
        "git_sha": sha,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)


def run_field(field=468, ccdid=3, qid=2, fid=2,
              science_filefracday=None, science_year=None,
              ref_years=(2021, 2024), n_ref=40, force=False,
              streaks_only=False, full_frame=False, offline=False):
    """Query -> select -> download -> template -> diff -> detect -> measure -> cut.

    streaks_only skips the template, the sci/ref channels and the (3,63,63) stamps:
    the motion path reads the diff alone, so ~1.6 GB/field of references buys nothing.
    The catalog is IDENTICAL either way - detection reads only the diff.

    full_frame detects on the whole chip minus a 64px NaN border (8.7x the area of
    the default central-1000) instead of the central crop. Opt-in, and it writes to
    its own run dir, so the regression-locked central-1000 catalog is untouched.
    """
    # offline: choose epochs from the frames already on disk and skip the download
    # entirely. Same table shape, so select_epochs and everything below are unchanged.
    if offline:
        # enumerate from the product this run will actually OPEN. A motion run reads
        # the diff alone, and streaks_only never downloaded the science frames, so
        # enumerating it from sciimg finds nothing and fails on a cell that is in fact
        # fully processable. With no science frames there are also no reference frames
        # to stack - hence n_ref=0, which is exact here, not a loosened guard.
        product = DIFF_PRODUCT if streaks_only else SCI_PRODUCT
        zq, epochs = None, local_grid(field, ccdid, qid, fid, product=product)
        if streaks_only:
            n_ref = 0
    else:
        zq, epochs = query_grid(field, ccdid, qid, fid)
    plan = select_epochs(epochs, science_year=science_year,
                         science_filefracday=science_filefracday,
                         ref_years=ref_years, n_ref=n_ref)
    sci_ffd = int(plan.science["filefracday"])

    rdir = run_dir(field, ccdid, qid, fid, sci_ffd, full_frame=full_frame)
    paths = run_paths(rdir, create=True)

    if not offline:
        download_epochs(zq, plan, field, ccdid, qid, force=force,
                        streaks_only=streaks_only)
    diff_path = resolve_local(sci_ffd, DIFF_PRODUCT, field=field, ccdid=ccdid, qid=qid)

    # the official diff defines the grid AND is the sole source of the catalog:
    # detection runs before any template work, so no template can move a detection.
    diff_full, diff_wcs = load_diff(diff_path)
    # ONE region rule, applied to the diff AND to every channel cut from it below.
    region = ((lambda a, w=None: full_frame_cutout(a, wcs=w)) if full_frame
              else (lambda a, w=None: central_cutout(a, wcs=w, size=CUTOUT_SIZE)))
    diff_c = region(diff_full, diff_wcs)
    cut_wcs = diff_c.wcs
    # zero-fill for the arrays, but tell photutils where the fakes are: a zero reads
    # as ordinary sky in a diff, so unmasked bad columns depress the local rms.
    nan_mask = np.isnan(diff_c.data)
    diff_img = np.nan_to_num(diff_c.data)

    rms = background_rms(diff_img, mask=nan_mask)
    segm_pos, segm_neg = detect_both_signs(diff_img, rms, mask=nan_mask)
    catalog = measure_catalog(diff_img, segm_pos, segm_neg, rms, cut_wcs)

    rkey = ref_key_for(plan)
    if streaks_only:
        tmpl_path = None
        assign_stamp_ids(catalog, diff_img.shape, str(paths["cutouts"]))
    else:
        sci_path = resolve_local(sci_ffd, SCI_PRODUCT, field=field, ccdid=ccdid, qid=qid)
        ref_paths = resolve_local_many([int(x) for x in plan.refs["filefracday"]],
                                       SCI_PRODUCT, field=field, ccdid=ccdid, qid=qid)
        # cached per (grid cell, reference set) - NOT per science epoch. Every epoch of
        # a cell shares one 5-10 min, ~9 GB-peak stack instead of rebuilding it.
        tpath = template_path(field, ccdid, qid, fid, rkey, create=True)
        if not force:
            _adopt_legacy_template(paths["template"], paths["manifest"], tpath, plan)
        template, tmpl_wcs = _build_template(ref_paths, tpath, force)
        with fits.open(sci_path) as h:
            sci_raw, sci_wcs = h[0].data.astype(float), WCS(h[0].header)
        sci_img = _prep_channel(sci_raw, sci_wcs, diff_wcs, diff_full.shape, region)
        ref_img = _prep_channel(template, tmpl_wcs, diff_wcs, diff_full.shape, region)
        cut_triplets(catalog, sci_img, ref_img, diff_img, str(paths["cutouts"]))
        tmpl_path = str(tpath)

    ra_c, dec_c, corners = _footprint(cut_wcs, diff_img.shape)
    obsjd = float(plan.science["obsjd"])

    write_catalog(catalog, str(paths["catalog"]))
    _write_manifest(paths["manifest"], field, ccdid, qid, fid, plan, len(catalog),
                    streaks_only=streaks_only, full_frame=full_frame,
                    region_shape=diff_img.shape, obsjd=obsjd, ra_center=ra_c,
                    dec_center=dec_c, corners=corners, ref_key=rkey)

    return FieldResult(
        run_dir=str(rdir), catalog_path=str(paths["catalog"]),
        cutouts_dir=str(paths["cutouts"]), diff_path=str(diff_path),
        template_path=tmpl_path, science_filefracday=sci_ffd,
        n_detections=len(catalog), full_frame=full_frame,
        obsjd=obsjd, ra_center=ra_c, dec_center=dec_c, corners=corners,
        ref_key=rkey, science_infobits=int(plan.science["infobits"]),
    )
