"""
Stage 1: put many epochs on ONE pixel grid.

WCS-basd (reproject), not star-pattern matchine - every ZTF header
carries a Gaia-solved WCS, so alignment is a coordinate lookup. 
astroalign is the fallback  for when a WCS is missing or bad.
"""

from collections import namedtuple

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp

Frame = namedtuple("Frame", "data wcs name seeing")

def load_frames(paths):
    """
    load FITS -> frames. seeing drive grid choice, so
    a missing one is fatal.
    """

    frames = []
    for p in paths:
        with fits.open(p) as hdul:
            hdr = hdul[0].header
            seeing = hdr.get("SEEING")
            if seeing is None:
                raise ValueError(f"Missing SEEING in {p}")
            frames.append(
                Frame(hdul[0].data.astype(float), WCS(hdr), str(p), float(seeing))
            )
    return frames

def pick_grid(frames):
    """
    The sharpest frame defines the target grid.
    """
    return frames[int(np.argmin([f.seeing for f in frames]))]

def reproject_one(data, wcs, grid_wcs, grid_shape):
    """
    Redraw one array onto the target grid. Edge pixels with no source -> NaN.
    """
    array, _ = reproject_interp((data, wcs), grid_wcs, shape_out=grid_shape)
    return array

def reproject_to(frames, grid_wcs, grid_shape):
    """
    Redraw every frame onto the grid. Returns a list of arrays,
    input order.
    """
    return [reproject_one(f.data, f.wcs, grid_wcs, grid_shape) for f in frames]