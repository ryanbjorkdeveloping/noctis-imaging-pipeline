"""Stage 1 (fetch): download the epochs an EpochPlan selected, resolve them on disk.

resolve_local asserts EXACTLY ONE hit - never glob(...)[0]. Zero hits = a download
silently failed (would IndexError 30 lines later); >1 = a stale duplicate. sci/ is a
flat tree shared across ALL fields.

The glob MUST pin field AND ccdid AND qid. A ZTF filename is
`ztf_<filefracday>_<field>_<filter>_c<ccd>_o_q<qid>_<product>`, and ONE field readout
produces 64 quadrants that all SHARE a filefracday. Pinning field only (the original
bug) made every quadrant of a visit resolve to whichever one was downloaded first: a
night sweep - dozens of quadrants sharing a filefracday - silently processed the same
diff N times. It slipped past the BE5 validation because those 8 exposures each had a
DISTINCT filefracday, so (ffd, field) was accidentally unique.
"""

import glob
from pathlib import Path

from ztfquery.io import LOCALSOURCE


def _indexes_for(meta, filefracdays):
    """metatable row indexes for a set of filefracday stamps (download_data wants indexes)."""
    want = [int(x) for x in filefracdays]
    return meta.index[meta["filefracday"].isin(want)].tolist()


def resolve_local(filefracday, product, field=None, ccdid=None, qid=None):
    """Locate ONE downloaded file by (filefracday, field, ccdid, qid). Asserts one hit.

    ccdid/qid are REQUIRED to disambiguate quadrants that share a filefracday (see the
    module docstring). They default to None only for backward compatibility; a caller
    that omits them on a multi-quadrant visit will hit the >1 assert, which is the loud
    failure - never a silent wrong-quadrant match.
    """
    ffd = int(filefracday)
    fpart = f"{field:06d}" if field is not None else "*"
    cpart = f"c{ccdid:02d}_o_q{qid}" if ccdid is not None and qid is not None else "*"
    pattern = str(Path(LOCALSOURCE) / "sci" / "**" / f"ztf_{ffd}_{fpart}_*{cpart}*{product}")
    hits = glob.glob(pattern, recursive=True)
    if len(hits) != 1:
        raise FileNotFoundError(
            f"{len(hits)} hits for {ffd}/{field}/c{ccdid}/q{qid}/{product} (want 1): {hits}")
    return Path(hits[0])


def resolve_local_many(filefracdays, product, field=None, ccdid=None, qid=None):
    """resolve_local for a list; preserves order."""
    return [resolve_local(f, product, field=field, ccdid=ccdid, qid=qid) for f in filefracdays]


def download_epochs(zq, plan, field, ccdid, qid, science_product="sciimg.fits",
                    diff_product="scimrefdiffimg.fits.fz", force=False,
                    streaks_only=False):
    """Download the science frame (+ its official diff) and all reference sciimg.

    force=False and everything already on disk -> genuine no-op (the sweep relies on
    this to not re-download shared frames). force=True re-downloads. The presence check
    pins ccdid/qid, or a sibling quadrant's file would make it skip a needed download.

    streaks_only -> fetch ONLY the official diff. The motion path reads the diff and
    nothing else, so the 40 reference frames (~1.6 GB/field) are pure waste there.
    """
    meta = zq.metatable
    sci_ffd = int(plan.science["filefracday"])
    ref_ffds = [int(x) for x in plan.refs["filefracday"]]

    if not force:
        try:
            resolve_local(sci_ffd, diff_product, field=field, ccdid=ccdid, qid=qid)
            if not streaks_only:
                resolve_local(sci_ffd, science_product, field=field, ccdid=ccdid, qid=qid)
                resolve_local_many(ref_ffds, science_product, field=field, ccdid=ccdid, qid=qid)
            return  # all present -> skip
        except FileNotFoundError:
            pass   # something missing -> fall through and download

    if not streaks_only:
        sci_idx = _indexes_for(meta, [sci_ffd] + ref_ffds)
        zq.download_data(science_product, indexes=sci_idx, show_progress=True, nprocess=1)
    diff_idx = _indexes_for(meta, [sci_ffd])
    zq.download_data(diff_product, indexes=diff_idx, show_progress=True, nprocess=1)
