"""Recover the known fast NEA 2019 BE5 from ZTF and prove the streak stage flags it
SHORT (control B, the real-data positive control). Two stages (cross-venv wall):
  1) THIS script (main .venv): ephemeris -> on-chip exposures -> download diff ->
     world_to_pixel -> cut 144x144 stamps -> save to ztfdata/nea_recovery/.
  2) score_nea_recovery.py (.venv_streaks): load the stamps -> DeepStreaks cascade.
Run from project root:  python streak_testing/recover_2019be5.py
"""
import os
import sys
import glob
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # sets $ZTFDATA before ztfquery

import numpy as np
from astroquery.jplhorizons import Horizons
from ztfquery import query
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u

from ztf_classification.pipeline import diff_gray_scale, cut_streak_stamp

OBJECT = "2019 BE5"
FIELDS = (516, 567)
JD_LO, JD_HI = 2458514.82, 2458514.90     # 2019-01-31 ~08-09 UT: the on-chip window
MAX_SEP_DEG = 0.35                         # NEA within this of a quadrant center = on-chip
OUT = os.path.join(os.environ["ZTFDATA"], "nea_recovery")


def on_chip_candidates(zq):
    """Query the fields/window, put the NEA at each exposure's EXACT obsjd, keep the
    quadrants it lands on (nearest-center < MAX_SEP_DEG). Horizons is queried at UNIQUE
    obsjds only (one per exposure) -- it caps the number of epochs per request."""
    zq.load_metadata(sql_query=f"field IN {FIELDS} AND fid=2 "
                               f"AND obsjd BETWEEN {JD_LO} AND {JD_HI}")
    m = zq.metatable
    uj = sorted(set(round(float(x), 6) for x in m["obsjd"]))
    eph = Horizons(id=OBJECT, location="I41", epochs=uj).ephemerides()
    pos = {round(float(uj[k]), 6): (float(eph["RA"][k]), float(eph["DEC"][k]))
           for k in range(len(uj))}
    cands = []
    for idx, row in m.iterrows():
        ra_n, dec_n = pos[round(float(row["obsjd"]), 6)]
        sep = SkyCoord(ra_n * u.deg, dec_n * u.deg).separation(
              SkyCoord(float(row["ra"]) * u.deg, float(row["dec"]) * u.deg)).deg
        if sep < MAX_SEP_DEG:
            cands.append((sep, idx, row, ra_n, dec_n))
    cands.sort(key=lambda c: c[0])
    return cands


def recover_stamp(zq, idx, row, ra_n, dec_n):
    """Download this exposure's diff, world_to_pixel the NEA (sky -> pixel), cut a
    144x144 dark-on-gray stamp. Returns (tag, stamp) or None if unavailable/edge."""
    zq.download_data("scimrefdiffimg.fits.fz", indexes=[idx], show_progress=False, nprocess=1)
    ff, field = int(row["filefracday"]), int(row["field"])
    ccd, qid = int(row["ccdid"]), int(row["qid"])
    hits = glob.glob(os.path.join(os.environ["ZTFDATA"], "sci", "**",
                     f"*{ff}*{field:06d}*c{ccd:02d}*q{qid}*scimrefdiffimg.fits.fz"),
                     recursive=True)
    if not hits:
        return None
    with fits.open(hits[0]) as h:
        diff, wcs = h[1].data.astype(float), WCS(h[1].header)
    x, y = wcs.world_to_pixel_values(ra_n, dec_n)
    H, W = diff.shape
    if not (72 < x < W - 72 and 72 < y < H - 72):
        return None
    med, sigma = diff_gray_scale(diff)
    return f"{field}_c{ccd}_q{qid}", cut_streak_stamp(diff, float(x), float(y), med, sigma, 144)


def main():
    os.makedirs(OUT, exist_ok=True)
    zq = query.ZTFQuery()
    cands = on_chip_candidates(zq)
    print(f"{len(cands)} on-chip candidate exposure(s)")
    stamps, specs = [], []
    for sep, idx, row, ra_n, dec_n in cands:
        r = recover_stamp(zq, idx, row, ra_n, dec_n)
        if r:
            specs.append(r[0])
            stamps.append(r[1])
            print(f"  cut {r[0]} (sep {sep:.3f} deg)")
    if not stamps:
        print("no stamp cut -> data-limited")
        return
    np.save(os.path.join(OUT, "nea_batch.npy"),
            np.stack(stamps)[..., np.newaxis].astype(np.float32))
    json.dump(specs, open(os.path.join(OUT, "nea_specs.json"), "w"))
    print(f"saved {len(stamps)} stamp(s) -> {OUT}")


if __name__ == "__main__":
    main()
