"""
Stage 2b: median-stack aligned epochs into a deep reference template.

median (not mean): robus to a mover present in one epoch - a streak is huge
spike in one frame; a mean drags toward it, a median ignores it.
That robustness is how transients get washed out of the reference.

nanMEDIAN: align.py fills off-grid edges with NaN; a plain median poisons
any pixel that is NaN in even one frame.
"""

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats

def stack_median(aligned):
    """
    list of aligned arrays (same shape) -> one deep template. NaN-robust.
    """
    return np.nanmedian(np.stack(aligned, axis=0), axis=0)

def template_noise(template):
    """
    Sigma-clipped std = the honest noise floor.

    A raw std measures the brightest star, not the noise (measured
    142.95 once);
    sigma clipping removes sources first
    """

    _, _, std = sigma_clipped_stats(template, sigma=3.0)
    return std

def save_template(template, grid_wcs, out_path):
    """
    Write the template as a FITS with the shared grid WCS in the header.

    float32: a median of 40 frames has a noise floor around 3 counts, so the f64
    mantissa is storing nothing but rounding. Halves the file (72 MB -> 38 MB), and
    these are KEPT forever by the eviction policy (re-stacking costs 5-10 min and
    ~9 GB peak RSS), so the saving compounds per grid cell.
    """

    fits.PrimaryHDU(
        data=np.asarray(template, dtype=np.float32), header=grid_wcs.to_header()
    ).writeto(out_path, overwrite=True)
    return out_path