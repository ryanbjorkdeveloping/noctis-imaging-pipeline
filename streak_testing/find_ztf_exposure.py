"""
find the ZTF science exposure that images 2019 BE5 streaking.
Target (from the ephemeris): RA~128.37, DEC~14.14, 2019-01-31 ~06:00 UT.

Runs in the MAIN venv:  python streak_testing/find_ztf_exposure.py
"""

import os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
import config                          # sets $ZTFDATA before ztfquery imports


from ztfquery import query

# the ephemeris target: where/when 2019 BE5 was bright, high, and streaking
TARGET_RA = 128.37
TARGET_DEC = 14.14
# JD of 2019-01-31 06:00 UT (we'll search a window around it)
TARGET_JD  = 2458514.75

zquery = query.ZTFQuery()

#cone-search ZTF's metadata around the target position, that night. 
# obsjd between a small window brackets the exposure we want. 
zquery.load_metadata(
    radec=[TARGET_RA, TARGET_DEC],
    size=0.5,
    sql_query=f"obsjd BETWEEN {TARGET_JD - 0.1} AND {TARGET_JD + 0.1}",
)

meta = zquery.metatable
print(f"found {len(meta)} ZTF exposures near the target\n")
if len(meta):
    # show the useful columns: when, which filter, field/ccd, and time offset
    cols = [c for c in ["obsjd", "filtercode", "field", "ccdid", "qid", "seeing", "maglimit"]
            if c in meta.columns]
    print(meta[cols].to_string())
    # how far (in time) is each from our exact target moment?
    print("\ntime offset from target (minutes):")
    print(((meta["obsjd"] - TARGET_JD) * 24 * 60).round(1).to_string())
