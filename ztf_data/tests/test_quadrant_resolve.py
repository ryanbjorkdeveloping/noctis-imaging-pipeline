"""Regression: resolve_local must disambiguate sibling quadrants of one field-visit.

The bug (fixed 2026-07-27, commit da4698f): resolve_local globbed
`ztf_<ffd>_<field>_*<product>`, pinning filefracday + field but NOT ccdid/qid. One ZTF
field readout is 64 quadrants that SHARE a filefracday, so every quadrant resolved to
whichever one was downloaded first - a night sweep silently processed the same diff N
times. It slipped past BE5 validation because those exposures each had a DISTINCT
filefracday. This test plants two sibling-quadrant files (same ffd+field, different qid)
and asserts the loud, correct behaviour.

Self-contained: builds a temp sci tree and monkeypatches fetch.LOCALSOURCE, so it needs
no network and no downloaded data (unlike the other ztfdata-fixture tests).
"""

import os
import tempfile
from pathlib import Path

import ztf_data.fetch as fetch
from ztf_data.fetch import resolve_local

FFD = 20190208208981
FIELD = 567
PRODUCT = "scimrefdiffimg.fits.fz"


def _touch(root, ccd, qid):
    """Plant an empty file with a real ZTF diff filename for one quadrant."""
    d = Path(root) / "sci" / "2019" / "0208" / str(FFD % 1000000)
    d.mkdir(parents=True, exist_ok=True)
    name = f"ztf_{FFD}_{FIELD:06d}_zr_c{ccd:02d}_o_q{qid}_{PRODUCT}"
    (d / name).write_bytes(b"")
    return name


def test_quadrant_resolve():
    with tempfile.TemporaryDirectory() as tmp:
        # two quadrants of ONE visit: c01/q1 and c01/q2 (same ffd, same field)
        n_q1 = _touch(tmp, 1, 1)
        n_q2 = _touch(tmp, 1, 2)
        orig = fetch.LOCALSOURCE
        fetch.LOCALSOURCE = tmp
        try:
            # 1. each quadrant resolves to ITS OWN file, not the other's
            assert resolve_local(FFD, PRODUCT, field=FIELD, ccdid=1, qid=1).name == n_q1
            assert resolve_local(FFD, PRODUCT, field=FIELD, ccdid=1, qid=2).name == n_q2

            # 2. omitting ccd/qid with siblings present is the LOUD failure (2 hits),
            #    never a silent wrong-quadrant match
            got_error = False
            try:
                resolve_local(FFD, PRODUCT, field=FIELD)
            except FileNotFoundError as e:
                got_error = True
                assert "2 hits" in str(e), f"expected a 2-hit report, got: {e}"
            assert got_error, "field-only resolve should have raised on 2 sibling quadrants"
        finally:
            fetch.LOCALSOURCE = orig

    print("\nOK: sibling quadrants resolve distinctly; field-only resolve raises loudly")


if __name__ == "__main__":
    test_quadrant_resolve()
