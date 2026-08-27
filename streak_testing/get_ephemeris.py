"""
get the ephemeris of the fast NEA 2019 BE5 around its discovery date,
so we know where and when ztf imaged its streaking.

JPL Horizons via astroquery. Palomar (ZTF) observatory code = I41.
Runs in the main venv: python streak_testing/get_ephemeris.py
"""

from astroquery.jplhorizons import Horizons

OBJECT = "2019 BE5" # the target NEA (V=15.1, ~50deg/day)
PALOMAR = "I41" #Palomar Mountain / ZTF observatory code

# a time window around the discovery night (2019-01-31), 1-hour steps
obj = Horizons(
    id=OBJECT,
    location=PALOMAR,
    epochs={"start": "2019-01-31", "stop": "2019-02-02", "step": "1h"}
)

eph = obj.ephemerides()

# print what columns we actually got (don't guess names — same discipline as SkyBoT)
print("columns:", eph.colnames)
print(f"\n{len(eph)} ephemeris rows over the window\n")

import numpy as np

# total sky motion per hour, and streak length in a 30s ZTF exposure
ra_rate = np.array(eph["RA_rate"]); dec_rate = np.array(eph["DEC_rate"])
total_rate = np.sqrt(ra_rate**2 + dec_rate**2)          # arcsec/hr
streak_px = total_rate * (30/3600)                       # px in 30s at 1"/px

eph["total_rate"] = total_rate
eph["streak_px"] = streak_px

# OBSERVABLE = above the horizon (EL>20 deg) AND dark (no sun up)
# solar_presence is '' when the Sun is down (daytime rows are marked)
print(f"{'time':22s} {'RA':>8s} {'DEC':>7s} {'V':>5s} {'EL':>6s} {'streak_px':>9s} {'dark?':>6s}")
for row in eph:
    el = row["EL"]
    dark = (str(row["solar_presence"]).strip() == "")     # empty => Sun below horizon
    observable = el > 20 and dark
    flag = "  <== OBSERVABLE" if observable else ""
    print(f"  {row['datetime_str']:20s} {row['RA']:8.3f} {row['DEC']:7.3f} "
          f"{row['V']:5.1f} {el:6.1f} {row['streak_px']:9.1f} {str(dark):>6s}{flag}")
