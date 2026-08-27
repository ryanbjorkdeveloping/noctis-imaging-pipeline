"""
find a ZTF exposure that contains a fast moving asteroid
(one moving fast enough to streak: >= ~1200 arcsec/hr at ZTF's 30s exposure).

Runs in the MAIN venv (astroquery + ztfquery), not .venv_streaks:
    python streak_testing/find_streaking_asteroid.py
"""

import os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
import config 

from astroquery.imcce import Skybot
from astropy.time import Time
from astropy.coordinates import SkyCoord
import astropy.units as u 

# a streak needs the object to move >= this during a 30s ZTF exposure
MIN_MOTION_ARCSEC_PER_HR = 1200 # ~10px in 30s at 1"/px

#ask SkyBoT what solar-system objcts were near (ra, dec) at time jd, and return
#any that arem oving fast enough to streak.
def scan_field(ra_deg, dec_deg, jd, radius_deg=2.0):
    field = SkyCoord(ra_deg, dec_deg, unit="deg")
    epoch = Time(jd, format="jd")
    try:
        table = Skybot.cone_search(field, radius_deg*u.deg, epoch)
    except RuntimeError:
        return []

    fast = []
    for row in table:
        # SkyBoT gives motion components dRA*cos(dec) and dDEC in "/hr... 
        # (confirm the exact column names when you run it — print table.colnames)
        # for now, print everything so we can see what's available
        fast.append(row)

    return table

if __name__ == "__main__" :
    import numpy as np 

    result = scan_field(ra_deg=150.0, dec_deg=2.0, jd=2458200.0)
    print(f"SkyBoT returned {len(result)} known objects in this field\n")

    if len(result):
        # total sky motion = sqrt(RA_rate^2 + DEC_rate^2), in arcsec/hour
        ra_rate = np.array(result["RA_rate"])
        dec_rate = np.array(result["DEC_rate"])
        total_motion = np.sqrt(ra_rate**2 + dec_rate**2)    

        # attach it and sort fastest-first
        result["total_motion"] = total_motion
        result.sort("total_motion", reverse=True)

        print(f"motion units: RA_rate is '{result['RA_rate'].unit}'")
        print(f"fastest 10 movers (total motion, arcsec/hr):")
        for row in result[:10]:
            print(f"  {str(row['Name']):12s}  V={row['V']:.1f}  "
                  f"motion={row['total_motion']:.1f}  "
                  f"(RA_rate={row['RA_rate']:.1f}, DEC_rate={row['DEC_rate']:.1f})")

        n_fast = int((total_motion >= MIN_MOTION_ARCSEC_PER_HR).sum())
        print(f"\nobjects moving >= {MIN_MOTION_ARCSEC_PER_HR} arcsec/hr (streak-capable): {n_fast}")

