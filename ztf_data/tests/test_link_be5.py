"""Linker control: the 2019 BE5 fast-mover track, both directions.

Positive: the SHORT candidates BE5 left across 8 exposures and 2 fields (516, 567)
must collapse to ONE track with the right motion (~6900"/hr, PA ~283 deg E of N).
Negative: 48 random SHORT candidates sprinkled into the same exposures must form NO
track and must not contaminate BE5's track.

Reads the on-disk sweep runs and enriches streak_pa/obsjd on the fly (the runs predate
streaks.py emitting those columns), so it needs the BE5 run dirs + their diffs present.
Run as a module from the project root:  python -m ztf_data.tests.test_link_be5
"""

import glob
import os

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
import astropy.units as u

import ztf_data  # config-before-ztfquery
from ztf_data.detect import load_diff
from ztf_data.streaks import sky_position_angle
from ztf_data.link import collapse_exposures, link_tracklets
from ztf_data.paths import OWN_ROOT
from ztfquery.io import LOCALSOURCE

RUNS = os.path.join(OWN_ROOT, "runs")


def _resolve_cell_diff(ffd, cell):
    """The one official diff for an exact grid cell + ffd (glob keyed by ccd/qid so the
    same ffd in adjacent quadrants doesn't collide). None if it's been pruned off disk."""
    ccd, qid = int(cell[9:11]), int(cell[13])
    pat = os.path.join(LOCALSOURCE, "sci", "**",
                       f"ztf_{ffd}_{int(cell[1:7]):06d}_*_c{ccd:02d}_*_q{qid}_*scimrefdiffimg.fits.fz")
    hits = glob.glob(pat, recursive=True)
    return hits[0] if len(hits) == 1 else None

# BE5's high-confidence blind-fit track (from the 82-exposure full-window sweep). Used
# ONLY to identify which on-disk SHORT candidates ARE BE5, so the test is deterministic
# no matter what other sweeps have littered the runs/ dir. A candidate is BE5 iff it sits
# within BE5_GATE_ARCSEC of BE5's predicted position at that exposure's obsjd. Cross-cell
# contaminants (the same ffd in an adjacent quadrant when BE5 is elsewhere) are ~2900" off
# -> cleanly excluded; real BE5 fragments spread only along the ~57"/exposure trail.
BE5_REF_OBSJD, BE5_REF_RA, BE5_REF_DEC = 2458514.846820, 123.716477, 15.200475
BE5_RATE_ARCSEC_HR, BE5_PA_DEG = 6902.75, 282.78
BE5_GATE_ARCSEC = 300.0


def _be5_predict(obsjd):
    """BE5's predicted sky position at obsjd, from the reference track (linear motion)."""
    dt_hr = (obsjd - BE5_REF_OBSJD) * 24.0
    sep = BE5_RATE_ARCSEC_HR * dt_hr
    pa = BE5_PA_DEG if sep >= 0 else BE5_PA_DEG + 180.0
    ref = SkyCoord(BE5_REF_RA * u.deg, BE5_REF_DEC * u.deg)
    return ref.directional_offset_by(pa * u.deg, abs(sep) * u.arcsec)


def _load_be5_observations():
    """The BE5 SHORT candidates across the runs, as linker observations.

    Gated on proximity to BE5's known track (see BE5_REF_* above): globbing runs/ now
    picks up whole-night sweeps in non-BE5 cells, so an ungated load would ingest other
    exposures' real SHORTs as fake BE5 points. The gate keeps the fixture pinned to BE5.
    """
    obs_rows, wcs_cache = [], {}
    for p in sorted(glob.glob(f"{RUNS}/*/*/streak_candidates.ecsv")):
        run = os.path.dirname(p)
        cell = os.path.basename(os.path.dirname(run))
        ffd = os.path.basename(run).replace("_full", "")
        fld = int(cell[1:7])
        sc = Table.read(p)
        if not len(sc):
            continue
        short = sc[sc["route"] == "SHORT_NEA_CANDIDATE"]
        if not len(short):
            continue
        cat = Table.read(glob.glob(f"{run}/*catalog.ecsv")[0])
        byid = {os.path.splitext(os.path.basename(str(r["stamp_path"])))[0]: r for r in cat}
        key = (cell, ffd)
        if key not in wcs_cache:
            dp = _resolve_cell_diff(ffd, cell)
            if dp is None:
                continue                        # contaminant cell's diff pruned -> skip
            _, w = load_diff(dp)
            wcs_cache[key] = (w, float(fits.getheader(str(dp), 1)["OBSJD"]))
        w, obsjd = wcs_cache[key]
        pred = _be5_predict(obsjd)
        for r in short:
            c = byid[r["row_id"]]
            coord = SkyCoord(float(c["ra"]) * u.deg, float(c["dec"]) * u.deg)
            if coord.separation(pred).arcsec > BE5_GATE_ARCSEC:
                continue                        # not on BE5's track -> a contaminant
            obs_rows.append(dict(
                row_id=r["row_id"], ra=float(c["ra"]), dec=float(c["dec"]),
                streak_pa=sky_position_angle(w, float(c["ra"]), float(c["dec"]),
                                             float(c["orientation"])),
                elongation=float(c["elongation"]), obsjd=obsjd, field=fld))
    return obs_rows


def _be5_track(tracks, be5_ids):
    for t in tracks:
        if len(set(t["row_ids"].split(";")) & be5_ids) >= 6:
            return t
    return None


def main():
    obs_rows = _load_be5_observations()
    assert obs_rows, "no BE5 SHORT candidates on disk - run the BE5 sweep first"
    obs = Table(rows=obs_rows)
    be5_ids = {r["row_id"] for r in obs_rows}

    # --- positive control ---
    tracks = link_tracklets(collapse_exposures(obs))
    assert len(tracks) == 1, f"BE5 must form exactly 1 track, got {len(tracks)}"
    t = tracks[0]
    assert t["n_fields"] == 2, f"BE5 track must span 2 fields, got {t['n_fields']}"
    assert t["n_det"] >= 6, f"BE5 track too short: {t['n_det']}"
    assert 4000 < t["rate_arcsec_hr"] < 9000, f"BE5 rate off: {t['rate_arcsec_hr']:.0f}"
    assert t["rms_arcsec"] < 60, f"BE5 fit rms too large: {t['rms_arcsec']:.1f}"
    print(f"[positive] 1 track, n_det={t['n_det']}, {t['n_fields']} fields, "
          f"rate={t['rate_arcsec_hr']:.0f}\"/hr, pa={t['motion_pa_deg']:.1f}, "
          f"rms={t['rms_arcsec']:.1f}\"  OK")

    # --- negative control: random SHORT candidates in the same exposures ---
    rng = np.random.default_rng(0)
    ra0, dec0 = obs["ra"].mean(), obs["dec"].mean()
    fld_of = {float(r["obsjd"]): int(r["field"]) for r in obs}
    neg = []
    for e in np.unique(obs["obsjd"]):
        for _ in range(6):
            neg.append(dict(row_id=f"neg_{len(neg)}",
                            ra=float(ra0 + rng.uniform(-0.5, 0.5)),
                            dec=float(dec0 + rng.uniform(-0.5, 0.5)),
                            streak_pa=float(rng.uniform(0, 180)), elongation=3.0,
                            obsjd=float(e), field=fld_of[float(e)]))
    mixed = link_tracklets(collapse_exposures(Table(rows=obs_rows + neg)))
    b = _be5_track(mixed, be5_ids)
    assert b is not None, "BE5 track lost under random contamination"
    assert not any(x.startswith("neg_") for x in b["row_ids"].split(";")), \
        "a random negative contaminated the BE5 track"
    assert len(mixed) == 1, f"random negatives formed a spurious track: {len(mixed)} total"
    print(f"[negative] {len(neg)} random SHORTs -> 0 spurious tracks, BE5 clean "
          f"(rms={b['rms_arcsec']:.1f}\")  OK")

    # --- novelty leave-one-out (network: SkyBoT) ---
    # Same BE5 track, two catalog states: KNOWN when BE5 is catalogued, NOVEL when it is
    # hidden. Proves the discovery branch (is_novel=True) fires on a real fast mover - the
    # only path never exercised on a true positive, because BE5 is already catalogued.
    from ztf_data.novelty import annotate_track_novelty
    base = annotate_track_novelty(tracks.copy())
    if base["novelty_status"][0] == "unchecked":
        print("[novelty] SkyBoT unreachable - leave-one-out skipped (network)")
    else:
        assert not base["is_novel"][0] and str(base["nearest_known"][0]).strip() == "2019 BE5", \
            f"BE5 must read KNOWN when catalogued, got {base['novelty_status'][0]}"
        loo = annotate_track_novelty(tracks.copy(), exclude_names={"2019 BE5"})
        assert loo["is_novel"][0] and loo["novelty_status"][0] == "novel", \
            "BE5 must flip NOVEL when hidden from the catalog"
        print(f"[novelty] catalogued->known (is_novel=False, rate_ratio "
              f"{base['rate_ratio'][0]:.2f}); hidden->novel (is_novel=True)  OK")
    print("BE5 link control PASSED")


if __name__ == "__main__":
    main()
