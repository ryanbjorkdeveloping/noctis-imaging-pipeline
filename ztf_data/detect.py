""" 
Stage 3a: turn a difference image into segmentation maps.

Deliberately inclusive - detects everything above threshold
(real movers, transients, and artifacts) and refuses to judge real vs bogus.
That is Stage 4's job; a faint real mover discarded here is lost forever.
"""

from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.stats import SigmaClip
from astropy.wcs import WCS
from photutils.background import Background2D, MedianBackground
from photutils.segmentation import detect_sources

# regression-locked: these produce 64 pos + 42 neg = 106 raw detections on the
# 2020 fixture (-> 114 after Stage 3's deblend, which splits merged clumps).
# 3 sigma over-detects on correlated diff noise on purpose - see module docstring.
N_SIGMA = 3.0
NPIXELS = 5

# Central region: away from the reproject edges where NaNs cluster. NOT fully
# NaN-free (~1.3k here) - Background2D masks them. Full-frame needs edge masking.
CUTOUT_SIZE = 1000

# Full-frame border trim. MEASURED (2026-07-22), not guessed: 83.5% of a ZTF diff's
# NaNs (220k of 264k) sit in the outer ~50px - that is ZTF's OWN chip-edge mask, not
# our reprojection (the official diff is never reprojected; only sci/template are,
# onto its grid). Trimming 64px cuts NaN 2.79% -> 0.555% and is the knee: 96px gains
# nothing (0.564%) for less area. Leaves 2952x2944 = 8.7x the central-1000 area.
FULL_FRAME_BORDER = 64

# the noise map is "the ruler" - a diff's background is already ~0
BOX_SIZE = (64,64)
FILTER_SIZE = (3, 3)

def load_diff(path):
    """ZTF's official diff. Returns (data, wcs).

    .fz is fpack-compressed: data lives in HDU 1, HDU 0 is a header stub.
    Plain .fits diffs put it in HDU 0.
    """
    with fits.open(path) as hdul:
        hdu = hdul[1] if str(path).endswith(".fz") else hdul[0]
        return hdu.data.astype(float), WCS(hdu.header)

def central_cutout(data, wcs=None, size=CUTOUT_SIZE):
    """ Cut the central size x size box, carrying the sliced wcs

    Load-bearing: use the returned .wcs downstream, never the full-frame wcs
    or every RA/Dec is offset by the crop.
    """

    center = (data.shape[1] // 2, data.shape[0] // 2)
    return Cutout2D(data, center, size, wcs=wcs)

def full_frame_cutout(data, wcs=None, border=FULL_FRAME_BORDER):
    """The WHOLE chip minus its NaN border, carrying the sliced wcs.

    Same contract as central_cutout (use the returned .wcs downstream) but a
    RECTANGULAR cutout - Cutout2D takes (ny, nx), so the full 3080x3072 chip does
    not have to be squared off to its shorter side.
    """
    H, W = data.shape
    return Cutout2D(data, (W // 2, H // 2), (H - 2 * border, W - 2 * border), wcs=wcs)

def background_rms(img, mask=None):
    """ The per-pixel noise map. Threshold is N_SIGMA * this, not a global std.

    mask (True = ignore) keeps NaN-filled pixels out of the box statistics. Callers
    zero-fill NaNs before detection, and a zero in a diff reads as ordinary sky, so
    unmasked bad columns quietly depress the local rms -> the threshold with it.
    Measured effect is small (rms floor 6.67 vs 7.07 full-frame, 3 of 891 detections)
    and it is a no-op on the central-1000 fixture - but it is free and it is correct.
    """
    bkg = Background2D(
        img,
        box_size=BOX_SIZE,
        filter_size=FILTER_SIZE,
        sigma_clip=SigmaClip(sigma=3.0),
        bkg_estimator=MedianBackground(),
        mask=mask,
    )
    return bkg.background_rms

def detect_both_signs(img, rms, mask=None):
    """
    Detect on +img and -img and returns
    (segm_pos, segm_neg)

    Negatives = faders and dipole lobes. a mover leaves both a positive
    (where it is now) and a negative (where it was), so dropping them halves recall.
    mask (True = ignore) suppresses detections on NaN-filled pixels.
    """

    threshold = N_SIGMA * rms
    segm_pos = detect_sources(img, threshold, n_pixels=NPIXELS, mask=mask)
    segm_neg = detect_sources(-img, threshold, n_pixels=NPIXELS, mask=mask)
    return segm_pos, segm_neg