"""Source injection for the streak positive control.

Paints a synthetic asteroid streak into a COPY of a real diff, so the full
detect->measure->cascade chain can be validated on real data with a known truth
(the standard way surveys measure their own completeness). BRIGHT/positive: diff
sources are positive bumps and Stage 3 detects positive blobs; cut_streak_stamp
inverts to dark-on-gray for DeepStreaks afterward.
"""

import numpy as np


def paint_streak(diff, x, y, length, angle_deg, flux, width=1.5):
    """Add a bright elongated ridge centered at (x,y) into a COPY of `diff`.

    A clean streak = a line SEGMENT of length `length` at `angle_deg`, with a
    Gaussian cross-section (sigma=`width`) of peak amplitude `flux`. Elongated
    because it extends `length` along the axis but only ~`width` across.
    """
    out = diff.astype(np.float32).copy()
    H, W = out.shape
    ang = np.radians(angle_deg)
    dx, dy = np.cos(ang), np.sin(ang)          # unit vector along the streak
    half = length / 2.0

    rad = int(half + 4 * width) + 1            # bbox: whole segment + gaussian wings
    x0, x1 = max(0, int(x - rad)), min(W, int(x + rad) + 1)
    y0, y1 = max(0, int(y - rad)), min(H, int(y + rad) + 1)

    for yi in range(y0, y1):
        for xi in range(x0, x1):
            px, py = xi - x, yi - y
            t = np.clip(px * dx + py * dy, -half, half)   # project onto axis, clamp to segment
            perp2 = (px - t * dx) ** 2 + (py - t * dy) ** 2  # perp distance^2 to the segment
            out[yi, xi] += flux * np.exp(-perp2 / (2 * width ** 2))
    return out
