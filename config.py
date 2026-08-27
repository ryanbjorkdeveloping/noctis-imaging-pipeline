#config.py file added to root that makes sure .env configures first
#with $ZTFQUERY before being put in a ztfquery project folder with the results

"""
Project configuration loader.

Import this module FIRST in any script that uses ztfquery:

    import config          # loads .env -> sets $ZTFDATA
    from ztfquery import query   # now ztfquery sees the right path

It reads the project's .env file, validates that ZTFDATA is set, and makes
sure the download folder exists before any ztfquery code runs.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# --- NumPy compatibility shim -------------------------------------------------
# ztfquery 1.28.0 calls np.in1d, which NumPy 2.0 REMOVED (renamed to np.isin).
# We run NumPy 2.x (for the OpenCV detectors), so restore the old name as an
# alias for its identical replacement. This must run before ztfquery is imported
# (config is imported first everywhere), and is a no-op on older NumPy that still
# has in1d.
import numpy as np

if not hasattr(np, "in1d"):
    np.in1d = np.isin
# -----------------------------------------------------------------------------

#Load .env from the project root into the process environment.
# This must happen before ztfquery is imported, because ztfquery reads
# $ZTFDATA at import time. Importing this module first guarates that.

# Pin the .env lookup to THIS file's directory (the repo root). A bare
# load_dotenv() searches upward from the CWD, so running a script from
# anywhere else silently finds no .env. Note os.environ still wins over
# .env, so `export ZTFDATA=...` works with no .env file at all.
load_dotenv(Path(__file__).resolve().parent / ".env")

ZTFDATA = os.environ.get("ZTFDATA")

#boundary
if not ZTFDATA:
    raise RuntimeError(
        "ZTFDATA is not set. Create a .env file in the project root with :\n"
        "   ZTFDATA=/absolute/path/to/ztf/data\n"
        "(see .env.example for the template)."
    )

#Making sure download folder actually exist, so ztfquery can write into it. 
Path(ZTFDATA).mkdir(parents=True, exist_ok=True)