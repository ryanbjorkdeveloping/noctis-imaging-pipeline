"""Pre-flight for the blind BE5 sweep: does 2019 BE5 land INSIDE the central
1000x1000 that run_field actually detects on?

recover_2019be5.py proved the NEA is on-chip (>=72 px from the FULL 3080x3072 edge),
but run_field detects on central_cutout(size=1000) - ~10% of the frame area. If the
streak sits outside that crop, the blind sweep CANNOT find it, and we'd be debugging
a "miss" that was never a detection failure. Costs nothing: the 9 diffs are already
on disk, so this is ephemeris + WCS arithmetic only.

Run from project root:  python streak_testing/preflight_be5_crop.py
"""

import os
import sys
import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

from astropy.io import fits
from astropy.wcs import WCS
from astroquery.jplhorizons import Horizons

from ztf_data.detect import CUTOUT_SIZE

OBJECT = "2019 BE5"
LOCATION = "I41"

def main():
    pattern = os.path.join(os.environ["ZTFDATA"], "sci", "2019", "**",
                           "*scimrefdiffimg.fits.fz")
    diffs = sorted(glob.glob(pattern, recursive=True))
    print(f"{len(diffs)} diff(s) on disk\n")

    meta = []
    for p in diffs:
        h = fits.getheader(p, 1)          # .fz -> HDU 1
        meta.append((p, float(h["OBSJD"]), WCS(h), int(h["NAXIS1"]), int(h["NAXIS2"])))

    # ONE Horizons call for all unique epochs (it caps epochs per request)
    jds = sorted({round(j, 6) for _, j, _, _, _ in meta})
    eph = Horizons(id=OBJECT, location=LOCATION, epochs=jds).ephemerides()
    pos = {round(float(jds[k]), 6): (float(eph["RA"][k]), float(eph["DEC"][k]))
           for k in range(len(jds))}

    half = CUTOUT_SIZE / 2
    inside = 0
    for p, jd, w, W, H in meta:
        ra, dec = pos[round(jd, 6)]
        x, y = w.world_to_pixel_values(ra, dec)
        cx, cy = W // 2, H // 2
        x0, x1, y0, y1 = cx - half, cx + half, cy - half, cy + half
        ok = (x0 <= x < x1) and (y0 <= y < y1)
        inside += ok
        # distance OUTSIDE the crop on each axis (0 = within)
        dx = max(x0 - x, x - (x1 - 1), 0)
        dy = max(y0 - y, y - (y1 - 1), 0)
        print(f"{os.path.basename(p)[:38]:40s} full=({x:7.1f},{y:7.1f})  "
              f"{'IN CROP' if ok else f'OUTSIDE by ({dx:.0f},{dy:.0f})px'}")

    print(f"\n{inside}/{len(meta)} land inside the central {CUTOUT_SIZE}x{CUTOUT_SIZE}")
    print("-> blind sweep is viable" if inside else
          "-> blind sweep CANNOT work on these; Rung 4 (full-frame detection) first")


if __name__ == "__main__":
    main()
