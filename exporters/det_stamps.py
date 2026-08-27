"""Cut a (science | reference | difference) postage stamp for ANY harvested detection.

WHY THIS EXISTS
---------------
`build_survey.py::_thumb_for` could only render a thumbnail from a saved (3,63,63)
`.npy` triplet, and a `streaks_only` harvest never writes those (`stamps_written:
false` in every run manifest). So 109,661 of the 109,975 exported detections had
`"thumb": null` and the drawer showed nothing - the reader could open a detection
and never see the pixels it was measured from.

The pixels were on disk the whole time. Every run keeps its official ZTF difference
image (497 of 503 exposures resolve one; the 6 that do not are exactly the
`quadrant_ambiguous` legacy runs, whose frames genuinely cannot be identified), and
ZTF's own deep REFERENCE co-add for the cell is cached under $ZTFDATA/ref for 474 of
them. This module cuts stamps straight out of those.

CUT BY SKY POSITION, NOT BY PIXEL
---------------------------------
Detections carry `x_centroid`/`y_centroid` in the DETECTION REGION's frame (central
1000x1000, or full-frame minus a 64 px border) - not in the FITS frame, and not in
the reference's frame at all. Cutting with `Cutout2D(..., SkyCoord)` and each file's
own WCS sidesteps the region offset entirely and is the only thing that can put the
reference (a different grid, different epoch, 3200x3200 not 3080x3072) on the same
source. Verified: the brightest detections land on the centre pixel in both.

The reference channel here is ZTF's OWN reference image - literally the frame the
official difference was measured against - so it is a truer "before" than our own
deep template would be.
"""

from __future__ import annotations

import glob
import os
import re

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D, NoOverlapError
from astropy.wcs import WCS

STAMP = 63          # ZTF's own alert-stamp size; matches the (3,63,63) triplets
FLOOR_SIGMA = 2.5   # below this the difference is noise field - render it as black
SPAN_SIGMA = 20.0   # full colour at 20 sigma (build_cell_fits.py's CEILING_SIGMA)
# Panels are COLORIZED (brightness x hue), not tinted over a constant base. The base
# -tint form build_thumbs.py used holds the two inactive channels at 0.2, which paints
# empty sky a flat olive/purple slab - tolerable when a min-max stretch left the sky
# mid-grey anyway, but not here, where the whole point of the black point below is
# that empty sky is black in every panel and the eye goes straight to the source.
CHANNEL_COLOR = {
    "sci": (0.46, 0.62, 1.00),    # blue   - the science frame ("after")
    "ref": (0.44, 1.00, 0.52),    # green  - ZTF's deep reference ("before")
}
GUTTER = 1          # px between panels, so three 63px squares read as three images
GUTTER_RGB = (38, 40, 48)

DIFF_PRODUCT = "scimrefdiffimg.fits.fz"
SCI_PRODUCT = "sciimg.fits"


# --------------------------------------------------------------- file resolution

def _localsource():
    from ztfquery.io import LOCALSOURCE
    return LOCALSOURCE


def _one(pattern):
    """Exactly-one glob, or None. Never glob(...)[0] - see ztf_data/fetch.py."""
    hits = glob.glob(pattern, recursive=True)
    return hits[0] if len(hits) == 1 else None


def sci_tree_path(ffd, field, ccdid, qid, product):
    return _one(os.path.join(_localsource(), "sci", "**",
                             f"ztf_{int(ffd)}_{field:06d}_*c{ccdid:02d}_o_q{qid}*{product}"))


def ref_path(field, ccdid, qid):
    """ZTF's cached deep reference co-add for one cell (one file per field/ccd/quadrant)."""
    return _one(os.path.join(_localsource(), "ref", "**",
                             f"ztf_{field:06d}_*_c{ccdid:02d}_q{qid}_refimg.fits"))


_QUAD_RE = re.compile(r"_c(\d{2})_o_q(\d)_")


def _quadrant_of(path):
    """(ccdid, qid) parsed back out of a ZTF filename, or None."""
    m = _QUAD_RE.search(os.path.basename(path))
    return (int(m.group(1)), int(m.group(2))) if m else None


def diff_by_footprint(ffd, field, ra, dec):
    """Find the difference image whose own sky footprint CONTAINS (ra, dec).

    The escape hatch for the six `quadrant_ambiguous` legacy runs. Their (ccdid, qid)
    label predates the resolve_local fix and is simply wrong - exposure 11 is recorded
    as c13/q1 while the only file for that visit is c13/q2 - so the strict glob finds
    nothing and 1125 real detections had no picture.

    This is NOT the relaxed `ztf_<ffd>_<field>_*` glob that caused the original bug.
    That glob picked whichever quadrant happened to be downloaded first, silently, with
    nothing on screen to reveal the substitution. Here the candidate must PROVE it is
    the right frame: (ra, dec) is the region centre the run itself recorded from the
    WCS of the diff it opened, so the frame that contains that point is the frame it
    opened. A ZTF readout tiles the sky into 64 non-overlapping quadrants, so at most
    one can contain the point - and if zero or several do, this returns None and the
    detection keeps saying "no cutout" rather than showing a different patch of sky.
    """
    if ra is None or dec is None:
        return None
    pattern = os.path.join(_localsource(), "sci", "**",
                           f"ztf_{int(ffd)}_{field:06d}_*{DIFF_PRODUCT}")
    keep = []
    for path in glob.glob(pattern, recursive=True):
        try:
            with fits.open(path) as h:
                wcs, shape = WCS(h[1].header), h[1].data.shape
            x, y = wcs.world_to_pixel(SkyCoord(float(ra), float(dec), unit="deg"))
        except Exception:
            continue
        if 0 <= float(x) < shape[1] and 0 <= float(y) < shape[0]:
            keep.append(path)
    return keep[0] if len(keep) == 1 else None


# ------------------------------------------------------------------- rendering

# Black point of a counts stamp. A 63x63 cutout is overwhelmingly empty sky, so its
# median IS the sky level and a black point just above it puts the sky at black and
# lets the sources carry the ramp - the same "black sky, sources crisp" calibration
# build_color_layers.py and build_cell_fits.py already settled on for the full frames.
SKY_PCTL = 62.0
PEAK_PCTL = 99.7


def _stretch_positive(a):
    """Percentile-clipped sqrt stretch for a counts image (science / reference).

    A ZTF reference spans ~200 to ~49000 counts; plain min-max puts the whole field in
    the bottom 0.5% of the ramp and the stamp reads as black with one white dot. sqrt
    then compresses the bright end so a saturated star does not eat the whole ramp.
    """
    a = np.asarray(a, dtype=np.float32)
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return np.zeros(a.shape, dtype=np.float32)
    lo, hi = np.percentile(finite, (SKY_PCTL, PEAK_PCTL))
    if hi - lo < 1e-9:
        lo, hi = float(finite.min()), float(finite.max())
        if hi - lo < 1e-9:
            return np.zeros(a.shape, dtype=np.float32)
    return np.sqrt(np.clip((np.nan_to_num(a, nan=lo) - lo) / (hi - lo), 0.0, 1.0))


def _stretch_signed(a, sigma):
    """Signed difference -> [-1, +1], scaled by the FRAME's own noise, floored at it.

    Two decisions, both learned the hard way elsewhere in this project:

    * Scale by the WHOLE frame's sigma, not the stamp's own extremes. A per-stamp
      min-max slides the zero point to wherever that stamp happens to peak, so a
      purely-negative stamp and a purely-positive one render identically and a 4-sigma
      residual looks exactly as loud as a saturated star. One shared scale keeps
      stamps comparable to each other, which is the entire reason to look at them.
    * Floor at FLOOR_SIGMA. Without it the raw difference is ~2/3 noise field and the
      stamp reads as coloured confetti with the source lost inside it - the same
      finding that forced `build_cell_fits.py`'s NaN floor and `cut_streak_stamp`'s
      span change. Detection ran at 3 sigma over >=5 connected pixels, so a real
      source's peak sits far above 2.5 sigma; only the noise is muted.
    """
    a = np.nan_to_num(np.asarray(a, dtype=np.float32), nan=0.0)
    if not sigma or not np.isfinite(sigma) or sigma <= 0:
        finite = a[np.isfinite(a)]
        sigma = float(np.std(finite)) if finite.size else 1.0
        if sigma <= 0:
            return np.zeros(a.shape, dtype=np.float32)
    floor, span = FLOOR_SIGMA * sigma, SPAN_SIGMA * sigma
    mag = np.clip((np.abs(a) - floor) / max(span - floor, 1e-9), 0.0, 1.0)
    return np.sign(a) * mag


def _tint(panel, kind):
    """(H,W) in [0,1] -> (H,W,3) RGB colorized by the channel's hue; 0 stays black."""
    color = CHANNEL_COLOR[kind]
    return np.stack([panel * c for c in color], axis=-1).astype(np.float32)


def _diverging(v):
    """Signed [-1,+1] -> the site's amber/cyan change language.

    Deliberately NOT a red tint like the other two panels: a difference is the only
    SIGNED plane here, and the sign is the whole point (appeared vs faded, and a
    dipole shows both lobes at once). Amber-up / cyan-down is the same encoding the
    survey map, the cell stacks and the change overlays already use, and it drops the
    unchanged sky to near-black instead of a flat mid-tone slab, so the eye lands on
    the change rather than on the noise field.
    """
    mag = np.abs(v) ** 0.6                                  # lift faint residuals
    pos, neg = v > 0, v < 0
    rgb = np.zeros(v.shape + (3,), dtype=np.float32)
    for ch, amber, cyan in ((0, 1.00, 0.24), (1, 0.72, 0.68), (2, 0.28, 0.92)):
        rgb[..., ch] = np.where(pos, mag * amber, np.where(neg, mag * cyan, 0.0))
    return rgb


def _robust_sigma(data, sample=400000):
    """MAD-based per-pixel noise of a whole difference frame.

    MAD, not std: a difference frame is full of real sources and saturated-star
    residuals, and std is set by those outliers rather than by the noise the sources
    have to stand out from. Subsampled because this is a display statistic and 9 Mpx
    of exact median costs more than it is worth.
    """
    flat = data.reshape(-1)
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        return None
    if finite.size > sample:
        finite = finite[:: max(1, finite.size // sample)]
    med = float(np.median(finite))
    return float(1.4826 * np.median(np.abs(finite - med))) or None


# ------------------------------------------------------------------ the source

class ExposureStamps:
    """The image planes for ONE exposure, opened lazily, plus a stamp cutter.

    The difference is decompressed in full (fpack; ~0.2 s) because every detection in
    the exposure needs a piece of it. The reference and science frames are memory
    -mapped and left open, so a 63x63 cut reads 63 rows, not 40 MB.
    """

    def __init__(self, ffd, field, ccdid, qid, want_sci=True, ra=None, dec=None):
        self._planes = {}
        self._open = []
        self.diff_sigma = None
        # True when the quadrant label was wrong and the frame had to be identified by
        # its sky footprint instead (see diff_by_footprint). Surfaced so a caller can
        # say which cell the pixels actually came from rather than repeat a bad label.
        self.quadrant_recovered = False

        dp = sci_tree_path(ffd, field, ccdid, qid, DIFF_PRODUCT)
        if dp is None:
            dp = diff_by_footprint(ffd, field, ra, dec)
            if dp is not None:
                quad = _quadrant_of(dp)
                if quad:
                    # every other plane is looked up from the TRUE cell, not the label
                    ccdid, qid = quad
                    self.quadrant_recovered = True
        self.key = (int(ffd), field, ccdid, qid)
        if dp:
            with fits.open(dp) as h:           # HDU 1: fpack puts a stub in HDU 0
                data = h[1].data.astype(np.float32)
                self._planes["diff"] = (data, WCS(h[1].header))
            self.diff_sigma = _robust_sigma(data)

        rp = ref_path(field, ccdid, qid)
        if rp:
            hl = fits.open(rp, memmap=True)
            self._open.append(hl)
            self._planes["ref"] = (hl[0].data, WCS(hl[0].header))

        if want_sci:
            sp = sci_tree_path(ffd, field, ccdid, qid, SCI_PRODUCT)
            if sp:
                hl = fits.open(sp, memmap=True)
                self._open.append(hl)
                self._planes["sci"] = (hl[0].data, WCS(hl[0].header))

    # channels in the project's established order: science, reference, difference
    @property
    def channels(self):
        return [c for c in ("sci", "ref", "diff") if c in self._planes]

    @property
    def usable(self):
        return "diff" in self._planes

    def close(self):
        for hl in self._open:
            try:
                hl.close()
            except Exception:
                pass
        self._open, self._planes = [], {}

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def stamp_rgb(self, ra, dec):
        """(H, W*n_channels, 3) uint8 strip for one sky position, or None."""
        pos = SkyCoord(float(ra), float(dec), unit="deg")
        panels = []
        for kind in self.channels:
            data, wcs = self._planes[kind]
            try:
                cut = Cutout2D(data, pos, (STAMP, STAMP), wcs=wcs,
                               mode="partial", fill_value=np.nan)
            except (NoOverlapError, ValueError):
                # off this plane entirely (the reference footprint is not identical to
                # the science quadrant's). Drop the channel rather than faking pixels.
                continue
            arr = np.asarray(cut.data, dtype=np.float32)
            panels.append((kind, _diverging(_stretch_signed(arr, self.diff_sigma))
                           if kind == "diff" else _tint(_stretch_positive(arr), kind)))
        if not panels or not any(k == "diff" for k, _ in panels):
            return None, []
        pix = [np.clip(p * 255.0, 0, 255).astype(np.uint8) for _, p in panels]
        gutter = np.empty((STAMP, GUTTER, 3), dtype=np.uint8)
        gutter[:] = GUTTER_RGB
        strip = [pix[0]]
        for p in pix[1:]:
            strip += [gutter, p]
        return np.concatenate(strip, axis=1), [k for k, _ in panels]
