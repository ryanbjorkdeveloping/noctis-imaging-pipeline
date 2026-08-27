"""Regression: an IRSA outage must be distinguishable from an empty grid cell.

Observed live 2026-08-11: IRSA answered `load_metadata` with a 502 HTML error page,
which ztfquery parsed into a DataFrame of 8 rows whose single column was named
`<!DOCTYPE HTML PUBLIC "-//IETF//DTD HTML 2.0//EN">`. That is non-empty, so
query_grid's `len(epochs)==0` guard passed it through, and the run died four lines
later on `KeyError: 'obsjd'` - naming neither IRSA nor the outage.

Why this matters more than the ugly traceback: the harvester retries failed targets and
gives up after 3 attempts. If "IRSA was down" is indistinguishable from "this target is
broken", a ten-minute blip permanently skips every target it touched. Same discipline
novelty.py applies to SkyBoT - "could not ask" is never "asked and got nothing".

Self-contained: fakes the ZTFQuery object, no network.
"""

import pandas as pd

import ztf_data.epochs as ep
from ztf_data.epochs import (
    query_grid, IRSAUnavailable, EpochSelectionError, REQUIRED_COLUMNS,
)

HTML_COL = '<!DOCTYPE HTML PUBLIC "-//IETF//DTD HTML 2.0//EN">'


class _FakeZQ:
    def __init__(self, df):
        self.metatable = df

    def load_metadata(self, **kw):
        pass


def _with_metatable(df):
    ep.query.ZTFQuery = lambda: _FakeZQ(df)


def test_irsa_guard():
    real = ep.query.ZTFQuery
    try:
        # 1. the observed 502 shape -> RETRYABLE outage, not a data verdict
        _with_metatable(pd.DataFrame({HTML_COL: ["x"] * 8}))
        try:
            query_grid(468, 3, 2, 2)
            raise AssertionError("an HTML error page must not parse as a grid")
        except IRSAUnavailable as e:
            assert "retry" in str(e).lower(), "the message must say it is retryable"
            assert "468" in str(e), "the message must name the grid cell"

        # 2. a well-formed but EMPTY answer -> the data error, unchanged
        _with_metatable(pd.DataFrame({c: [] for c in REQUIRED_COLUMNS}))
        try:
            query_grid(999, 1, 1, 2)
            raise AssertionError("an empty grid cell must still raise")
        except EpochSelectionError:
            pass

        # 3. a partial schema (one column lost) is also an outage, not silent nonsense
        cols = {c: [1] for c in REQUIRED_COLUMNS}
        cols.pop("obsjd")
        _with_metatable(pd.DataFrame(cols))
        try:
            query_grid(468, 3, 2, 2)
            raise AssertionError("a missing required column must raise")
        except IRSAUnavailable as e:
            assert "obsjd" in str(e)

        # 4. the two errors must stay distinguishable by TYPE, not by message text -
        #    the harvester branches on this to decide retry vs. mark-failed
        assert not issubclass(IRSAUnavailable, EpochSelectionError)
        assert not issubclass(EpochSelectionError, IRSAUnavailable)
    finally:
        ep.query.ZTFQuery = real

    print("OK: IRSA outage, empty cell, and partial schema are distinct + typed")


if __name__ == "__main__":
    test_irsa_guard()
