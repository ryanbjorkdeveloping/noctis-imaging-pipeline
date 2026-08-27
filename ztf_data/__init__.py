""" ZTF data acquisition + reduction (Stages 1-3):

Separate from ztf_classification (inference, no network) because
this package owns the network, the FITS, and the config-before-ztfquery
import rule below.
"""

# LOAD-BEARING: ztfquery reads $ZTFDATA at import time, so config
# (which loads .env and installs the np.in1d shim) must land first
# Enforced here so importing any ztf_data submodule directly can't skip it.

import config