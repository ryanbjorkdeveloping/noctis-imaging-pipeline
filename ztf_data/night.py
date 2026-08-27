"""Night enumerator: which grid cells did ZTF observe on a given night, in a region?

Turns 'sweep the fields I typed' into 'sweep the sky a mover crossed'. A fast NEA leaves
a field between that field's two nightly visits, so a discovery sweep must cover EVERY
cell the object could have crossed - which is not known in advance. This lists them from
ZTF's own metadata: one target per exposure-quadrant, ready to hand to sweep_targets.

The linking stage (link.py) is what fuses detections back across those cells into one
track, so the enumerator is deliberately GENEROUS (list everything in the box); the caller
subsets by region/time and lets linking sort out which exposures share a mover.
"""

import datetime

from astropy.table import Table
from astropy.time import Time
from ztfquery import query

# exactly the kwargs run_field/sweep_targets accept - nothing else may go in a target dict
TARGET_KEYS = ("field", "ccdid", "qid", "fid", "science_filefracday")
_META_KEYS = ("ra", "dec", "obsjd", "seeing", "maglimit", "infobits", "is_clean")


class EmptyNightError(RuntimeError):
    """The night/region query returned no exposures (names the window so it is debuggable
    instead of surfacing as a downstream astropy error on an empty table - the same class
    of trap query_grid's empty-guard closed)."""


def _night_jd_bounds(date):
    """[jd_lo, jd_hi) for a UT calendar day. obsjd rolls at noon, so ZTF's Palomar
    observing night sits inside a single UT day in obsjd terms - the BE5 exposures
    (obsjd .82-.87 of JD 2458514, i.e. 2019-01-31 07-09h UT) all fall in one UT day."""
    if isinstance(date, str):
        date = datetime.date.fromisoformat(date)
    jd_lo = Time(datetime.datetime(date.year, date.month, date.day), scale="utc").jd
    return jd_lo, jd_lo + 1.0


def enumerate_night(date, ra_range=None, dec_range=None, fields=None, fid=2,
                    ut_hour_range=None, clean_only=False):
    """Every ZTF exposure on UT `date`, optionally within a sky box and/or field list and/or
    a UT-hour sub-window, one filter. Returns a Table (one row per exposure-quadrant) with
    the sweep-target keys field/ccdid/qid/fid/science_filefracday + metadata ra/dec/obsjd/
    seeing/maglimit/infobits/is_clean, sorted by time. Use night_targets() to strip it to
    sweep dicts.

    ra_range/dec_range filter the quadrant CENTER via SQL columns - NOT the radec/size POS
    cone (which means 'image contains this point' and ANDs to the empty set off-region;
    see epochs.query_grid). ut_hour_range=(h0,h1) narrows to that UT-hour slice of the night
    (h in [0,24)) - the time-window subset a targeted discovery run needs, since a night has
    hours of exposures and a fast mover sits in a sub-window. fid=None keeps all filters.
    clean_only drops infobits!=0 (a REFERENCE-quality bar; leave it False for a MOTION search
    - a fast NEA is over the chip whether or not the sky is photometric, see query_grid).
    """
    jd_lo, jd_hi = _night_jd_bounds(date)
    lo = jd_lo if ut_hour_range is None else jd_lo + ut_hour_range[0] / 24.0
    hi = jd_hi if ut_hour_range is None else jd_lo + ut_hour_range[1] / 24.0
    clauses = [f"obsjd >= {lo}", f"obsjd < {hi}"]
    if fid is not None:
        clauses.append(f"fid={fid}")
    if ra_range is not None:
        clauses.append(f"ra > {ra_range[0]} AND ra < {ra_range[1]}")
    if dec_range is not None:
        clauses.append(f"dec > {dec_range[0]} AND dec < {dec_range[1]}")
    if fields is not None:
        clauses.append("field IN (" + ",".join(str(int(f)) for f in fields) + ")")

    zq = query.ZTFQuery()
    zq.load_metadata(sql_query=" AND ".join(clauses))
    mt = zq.metatable

    if len(mt) == 0:
        raise EmptyNightError(
            f"no exposures for {date} fid={fid} ra={ra_range} dec={dec_range} fields={fields}"
        )

    cols = ["field", "ccdid", "qid", "fid", "filefracday",
            "ra", "dec", "obsjd", "seeing", "maglimit", "infobits"]
    keep = mt[cols].rename(columns={"filefracday": "science_filefracday"})
    t = Table.from_pandas(keep)
    t["is_clean"] = t["infobits"] == 0
    if clean_only:
        t = t[t["is_clean"]]
    t.sort(["obsjd", "field", "ccdid", "qid"])
    return t


def night_targets(table):
    """Extract sweep_targets dicts (ONLY run_field kwargs) from an enumerate_night table,
    dropping the metadata columns run_field does not accept."""
    return [{k: int(row[k]) for k in TARGET_KEYS} for row in table]


if __name__ == "__main__":
    import sys

    # python -m ztf_data.night 2019-01-31 [ra_lo ra_hi dec_lo dec_hi]
    date = sys.argv[1] if len(sys.argv) > 1 else "2019-01-31"
    ra_range = dec_range = None
    if len(sys.argv) >= 6:
        ra_range = (float(sys.argv[2]), float(sys.argv[3]))
        dec_range = (float(sys.argv[4]), float(sys.argv[5]))
    t = enumerate_night(date, ra_range=ra_range, dec_range=dec_range)
    cells = sorted(set((int(r["field"]), int(r["ccdid"]), int(r["qid"])) for r in t))
    print(f"{date} ra={ra_range} dec={dec_range}: {len(t)} exposures across "
          f"{len(cells)} cells, fields {sorted(set(int(r['field']) for r in t))}")
    print(f"{int(sum(t['is_clean']))} clean (infobits==0) / {len(t)} total")
