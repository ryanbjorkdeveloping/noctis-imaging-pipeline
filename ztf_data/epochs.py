"""Stage 1 (epoch selection): query one ZTF grid cell, choose science + reference epochs.

filefracday IS the filename stamp (ztf_<filefracday>_...) - the primary key threaded
end to end. NEVER reconstruct it from obsjd: JD rolls at noon, MJD at midnight, and
ZTF's fracday follows MJD, so obsjd%1 does NOT equal the stamp.
"""

import os
from collections import namedtuple

import numpy as np
from astropy.time import Time
from ztfquery import query

# self-subtraction guards - correctness, NOT knobs. SEARCH filters which science
# candidates are acceptable (every ref must be this far away); MIN is an independent
# final assert. Keep BOTH: the guard must stay independent of the filter or it checks
# nothing. Loosening either silently reintroduces self-subtraction - INVISIBLE in output.
SEARCH_GAP_DAYS = 120
MIN_GAP_DAYS = 60

EpochPlan = namedtuple("EpochPlan", "science refs gap_days")


class EpochSelectionError(RuntimeError):
    """No valid science/reference split (asserts vanish under -O; raise instead)."""


class IRSAUnavailable(RuntimeError):
    """IRSA did not answer with a metadata table - the query is RETRYABLE.

    Distinct from EpochSelectionError on purpose: "I could not ask" is not "I asked and
    the answer was no", the same three-state discipline novelty.py applies to SkyBoT.
    An unattended harvester must retry this WITHOUT counting it as a failed target -
    otherwise a ten-minute IRSA blip permanently skips every target it touched.
    """


# the columns everything downstream indexes. Checked because ztfquery hands back
# whatever it got: an IRSA 502 arrives as an HTML error page parsed into a DataFrame
# of 8 rows and one column named '<!DOCTYPE HTML ...>', which sails straight past a
# len()==0 guard and dies 4 lines later on KeyError: 'obsjd', naming neither IRSA nor
# the outage. Observed live 2026-08-11.
REQUIRED_COLUMNS = ("filefracday", "obsjd", "infobits", "seeing")


def query_grid(field, ccdid, qid, fid):
    """One grid cell -> (zquery, epoch table with 'year' + 'is_clean' attached).

    Returns EVERY epoch, FLAGGED not filtered. infobits==0 is the right bar for
    REFERENCE frames (a deep template needs quality) and the WRONG bar for the science
    epoch of a MOTION search: a fast NEA is over your chip when it is over your chip.
    On the 2019 BE5 night every in-crop exposure carries infobits 2**26 (bright sky -
    maglim degrades 20.6 -> 18.8 across the night at flat ~2.2" seeing) while a V=15
    streak stays trivially visible. Filtering here made those frames unreachable behind
    a "not in grid" error. select_epochs applies the bar where it belongs.

    Returns the zquery too: fetch.py reuses it to download_data by row index.

    NO radec/size cone - deliberately, do not re-add one. (field, ccdid, qid) IS the
    grid-cell address; a cone carries no extra information and can only contradict it.
    IRSA treats radec/size as POS ("images must CONTAIN this point"), so a cone left at
    one field's sky position ANDs to the empty set on every other field - which then
    surfaced three frames later as an astropy 'jd format' error on the empty table,
    naming neither the field nor the cone. That silently pinned the sweep to field 468.
    """
    zq = query.ZTFQuery()
    zq.load_metadata(
        sql_query=f"fid={fid} AND field={field} AND ccdid={ccdid} AND qid={qid}",
    )

    epochs = zq.metatable.copy()
    # schema BEFORE emptiness: an IRSA error page is non-empty but has no real columns,
    # so an emptiness check alone misreads an outage as a data answer.
    missing = [c for c in REQUIRED_COLUMNS if c not in epochs.columns]
    if missing:
        got = list(epochs.columns)[:4]
        raise IRSAUnavailable(
            f"IRSA returned no usable metadata table for field={field} ccdid={ccdid} "
            f"qid={qid} fid={fid}: missing {missing}, got columns {got} "
            f"({len(epochs)} rows). Transient server error - retry."
        )
    if len(epochs) == 0:
        raise EpochSelectionError(
            f"no epochs at all for field={field} ccdid={ccdid} qid={qid} fid={fid}"
        )
    epochs["year"] = Time(epochs["obsjd"].values, format="jd").to_value("decimalyear").astype(int)
    epochs["is_clean"] = epochs["infobits"] == 0
    return zq, epochs


def local_grid(field, ccdid, qid, fid, product="sciimg.fits"):
    """Epoch table built from CACHED FITS headers instead of IRSA. Same shape as
    query_grid's, so select_epochs and run_field cannot tell the difference.

    Two reasons this exists, neither of them a workaround:
      1. Download and processing are separable. A machine can pull frames in bulk once
         and then reduce them repeatedly with no network - which is how a long harvest
         on a second machine actually wants to work.
      2. IRSA goes down. It returned 502s across two whole sessions (2026-08-11), and
         every cached frame was still sitting on disk, unusable, because the pipeline
         insisted on asking a dead server what it already knew.

    filefracday comes from the FILENAME, never from obsjd arithmetic: JD rolls at noon
    and ZTF's fracday follows MJD, so obsjd % 1 is NOT the stamp (see the module head).
    Only the header is read, not the pixels, so ~90 frames cost well under a second.

    product selects WHICH cached file defines an epoch, and that choice must match what
    the run actually opens. A motion run reads the official diff and never touches a
    science frame, so enumerating it from sciimg.fits asks for frames it will not use -
    and offline that is fatal, because streaks_only never downloaded them. Pass
    scimrefdiffimg.fits.fz for motion; the header there lives in HDU 1 (fpack puts a
    stub in HDU 0), so the extension is probed rather than assumed.
    """
    import glob
    import re

    import pandas as pd
    from astropy.io import fits
    from ztfquery.io import LOCALSOURCE

    pattern = os.path.join(
        LOCALSOURCE, "sci", "*", "*", "*",
        f"ztf_*_{int(field):06d}_*c{int(ccdid):02d}_o_q{int(qid)}*{product}")
    rows = []
    for path in sorted(glob.glob(pattern)):
        m = re.search(r"ztf_(\d{14})_", os.path.basename(path))
        if not m:
            continue
        try:
            h = fits.getheader(path)
            if "OBSJD" not in h:          # fpacked: HDU 0 is a stub, the real header is 1
                h = fits.getheader(path, 1)
        except (OSError, IndexError, KeyError):
            continue                      # a truncated download must not kill the scan
        if int(h.get("FILTERID", -1)) != int(fid):
            continue
        rows.append({
            "filefracday": int(m.group(1)),
            "obsjd": float(h["OBSJD"]),
            "seeing": float(h.get("SEEING", float("nan"))),
            "infobits": int(h.get("INFOBITS", 0)),
            "field": int(field), "ccdid": int(ccdid), "qid": int(qid), "fid": int(fid),
            "local_path": path,
        })
    if not rows:
        raise EpochSelectionError(
            f"no cached {product} for field={field} ccdid={ccdid} qid={qid} fid={fid} "
            f"under {LOCALSOURCE}/sci - nothing to work with offline"
        )
    epochs = pd.DataFrame(rows).drop_duplicates("filefracday").reset_index(drop=True)
    epochs["year"] = Time(epochs["obsjd"].values, format="jd") \
        .to_value("decimalyear").astype(int)
    epochs["is_clean"] = epochs["infobits"] == 0
    return epochs


def _by_sharpness(df):
    """Sharpest-first, DETERMINISTICALLY. Default quicksort is unstable -> a tie at
    the n_ref boundary flips the template. mergesort + filefracday tiebreak fixes it."""
    return df.sort_values(["seeing", "filefracday"], kind="mergesort")


def select_epochs(epochs, science_year=None, science_filefracday=None,
                  ref_years=(2021, 2024), n_ref=40):
    """1 science + n_ref reference epochs.

    Reference = n_ref sharpest epochs in [ref_years], EXCLUDING the science year
    (self-sub guard, enforced here). Science = pinned filefracday, else the sharpest
    frame in science_year sitting > SEARCH_GAP_DAYS from every reference epoch.

    References come from is_clean epochs ONLY - a deep template needs quality. A
    PINNED science filefracday is honoured whatever its infobits (an explicit pin is
    the caller's decision, and motion targets are often only on flagged bright-sky
    frames); an AUTO-picked science epoch stays conservative and comes from clean too.

    n_ref=0 selects science ONLY, with an empty reference set. This is NOT a relaxed
    guard - it is for streaks_only, where no template is ever stacked and the diff is
    ZTF's own product built against ZTF's own reference. With nothing of ours to
    subtract, there is nothing to self-subtract, so the gap guards below are vacuous
    rather than skipped. run_field passes it only on the streaks_only branch; asking
    for it on a stationary run would produce a template from zero frames, so
    _build_template's caller must never see n_ref=0.
    """
    lo, hi = ref_years
    clean = epochs[epochs["is_clean"]]

    if n_ref == 0:
        if science_filefracday is not None:
            hit = epochs[epochs["filefracday"] == int(science_filefracday)]
            if len(hit) == 0:
                raise EpochSelectionError(
                    f"science filefracday {science_filefracday} not in grid")
            science = hit.iloc[0]
        elif science_year is not None:
            pool = clean[clean["year"] == int(science_year)]
            if len(pool) == 0:
                raise EpochSelectionError(f"no clean {science_year} frame in grid")
            science = _by_sharpness(pool).iloc[0]
        else:
            raise EpochSelectionError("give science_filefracday or science_year")
        # gap_days is None, not inf: it is written to manifest.json, and json.dump
        # emits a bare Infinity that no strict JSON reader will take back.
        return EpochPlan(science=science, refs=epochs.iloc[0:0], gap_days=None)

    if science_filefracday is not None:
        hit = epochs[epochs["filefracday"] == int(science_filefracday)]   # full table
        if len(hit) == 0:
            raise EpochSelectionError(f"science filefracday {science_filefracday} not in grid")
        science = hit.iloc[0]
        sci_year = int(science["year"])
    elif science_year is not None:
        sci_year, science = science_year, None   # science picked below
    else:
        raise EpochSelectionError("give science_filefracday or science_year")

    pool = clean[(clean["year"] >= lo) & (clean["year"] <= hi) & (clean["year"] != sci_year)]
    refs = _by_sharpness(pool).head(n_ref).reset_index(drop=True)
    if len(refs) < n_ref:
        raise EpochSelectionError(f"only {len(refs)} reference epochs, need {n_ref}")

    if science is None:
        for _, cand in _by_sharpness(clean[clean["year"] == sci_year]).iterrows():
            if np.abs(refs["obsjd"].values - cand["obsjd"]).min() > SEARCH_GAP_DAYS:
                science = cand
                break
        if science is None:
            raise EpochSelectionError(f"no {sci_year} frame >{SEARCH_GAP_DAYS}d from all refs")

    gap = float(np.abs(refs["obsjd"].values - science["obsjd"]).min())
    if gap <= MIN_GAP_DAYS:
        raise EpochSelectionError(f"closest ref {gap:.0f}d from science (guard >{MIN_GAP_DAYS})")

    return EpochPlan(science=science, refs=refs, gap_days=gap)
