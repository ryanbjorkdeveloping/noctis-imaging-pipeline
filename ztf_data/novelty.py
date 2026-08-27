"""Novelty filter: is a streak candidate a KNOWN solar-system object, or new?

Cross-matches a candidate's (ra, dec) at the exposure epoch against SkyBoT (IMCCE)
and reports the NEAREST catalogued object + separation + its sky motion. No object
nearby => a NOVEL discovery candidate.

LOAD-BEARING CAVEAT: fast NEOs at close approach (exactly what streaks ARE) can have
SkyBoT positions off by ARCMINUTES when their orbit came from a short arc -- e.g.
2019 BE5 at its own 2019-01-31 discovery is 6' and 6 mag off. So a tight match MISSES
a known fast mover. Report separation + motion; treat a nearby fast mover as
'possibly known, review', not an automatic novel.
"""

from astroquery.imcce import Skybot
from astropy.coordinates import SkyCoord
from astropy.time import Time
import astropy.units as u

SEARCH_RADIUS_ARCMIN = 10.0     # wide: fast movers can sit arcminutes from their catalog spot
MIN_STREAK_MOTION_ARCSEC_HR = 1000.0   # streak-capable; slower = point-source main-belt, not our streak


class SkybotUnavailable(RuntimeError):
    """SkyBoT could not be QUERIED (HTTP 500, timeout, DNS...).

    Distinct from an empty cone on purpose. "I asked and nothing is there" means
    NOVEL; "I could not ask" means UNKNOWN. Collapsing them would let an IMCCE
    outage manufacture a field full of fake NEO discoveries - the single worst
    direction for this filter to fail in.
    """


def nearest_known_object(ra, dec, obsjd, search_arcmin=SEARCH_RADIUS_ARCMIN,
                         min_motion=MIN_STREAK_MOTION_ARCSEC_HR, exclude_names=None):
    """Nearest known FAST mover (motion > min_motion) to (ra, dec) at obsjd:
    (name, sep_arcsec, motion_arcsec_per_hr) or None. A streak matches a fast mover,
    NOT a slow main-belt asteroid that merely happens to fall in the wide cone.

    None = cone searched, no known fast mover (-> novel). Raises SkybotUnavailable
    if the service could not be reached at all (-> unknown, NOT novel).

    exclude_names: object names to treat as if they were NOT catalogued (dropped from
    the SkyBoT response before the fast-mover match). This is the leave-one-out validation
    lever for the discovery branch: hide a KNOWN mover from the catalog and confirm the
    filter would have flagged it novel. Names are compared stripped of surrounding space
    (SkyBoT pads e.g. '2019 BE5').
    """
    pos = SkyCoord(ra * u.deg, dec * u.deg)
    try:
        tbl = Skybot.cone_search(pos, search_arcmin * u.arcmin, Time(obsjd, format="jd"))
    except RuntimeError:
        return None                       # astroquery's empty-cone signal
    except Exception as exc:              # HTTPError, ConnectionError, timeout...
        raise SkybotUnavailable(f"SkyBoT query failed at ({ra:.5f}, {dec:.5f})") from exc
    if len(tbl) == 0:
        return None
    if exclude_names:                     # leave-one-out: pretend these are uncatalogued
        drop = {str(n).strip() for n in exclude_names}
        tbl = tbl[[str(n).strip() not in drop for n in tbl["Name"]]]
        if len(tbl) == 0:
            return None
    motions = ((tbl["RA_rate"] ** 2 + tbl["DEC_rate"] ** 2) ** 0.5).value  # arcsec/hr
    fast = motions > min_motion
    if not fast.any():
        return None                       # no known fast mover here -> novel streak
    tblf, motf = tbl[fast], motions[fast]
    seps = pos.separation(SkyCoord(tblf["RA"], tblf["DEC"]))
    i = int(seps.argmin())
    return str(tblf["Name"][i]), float(seps[i].arcsec), float(motf[i])

def annotate_novelty(candidates, obsjd):
    """Add `nearest_known` (name or ""), `nearest_sep_arcsec`, `nearest_motion`,
    `novelty_status` and `is_novel` columns.

    novelty_status is the honest three-state answer, and is what you should read:
      "known"     - a catalogued fast mover is nearby
      "novel"     - SkyBoT answered and there is no known fast mover -> discovery candidate
      "unchecked" - SkyBoT could not be reached; novelty is UNKNOWN
    is_novel is the convenience bool (status == "novel") and is False when unchecked,
    so a service outage can never promote a candidate to a discovery. A per-row failure
    is contained: the surviving rows keep their real answers instead of the whole
    exposure's candidates being thrown away.
    """
    names, seps, motions, status = [], [], [], []
    for row in candidates:
        try:
            hit = nearest_known_object(float(row["ra"]), float(row["dec"]), obsjd)
        except SkybotUnavailable:
            names.append(""); seps.append(-1.0); motions.append(-1.0)
            status.append("unchecked")
            continue
        if hit is None:
            names.append(""); seps.append(-1.0); motions.append(-1.0)
            status.append("novel")
        else:
            n, s, mo = hit
            names.append(n); seps.append(s); motions.append(mo)
            status.append("known")
    candidates["nearest_known"] = names
    candidates["nearest_sep_arcsec"] = seps
    candidates["nearest_motion"] = motions
    candidates["novelty_status"] = status
    candidates["is_novel"] = [s == "novel" for s in status]
    return candidates

def annotate_track_novelty(tracks, exclude_names=None):
    """Track-level novelty: match each linked track's POSITION + measured RATE against
    known fast movers. A track carries a rate the single-detection filter never had, so
    matching becomes far more specific - a rate-consistent neighbour IS the object even
    when its short-arc catalog position is arcminutes off (the BE5 failure mode).

    Adds nearest_known / nearest_sep_arcsec / nearest_motion / rate_ratio (track rate /
    known motion, ~1 = consistent) / novelty_status (known|novel|unchecked) / is_novel.
    Expects the columns link_tracklets emits: ra, dec, obsjd_mid, rate_arcsec_hr.

    exclude_names: treat these object names as uncatalogued (leave-one-out validation of
    the discovery branch - see nearest_known_object).
    """
    names, seps, motions, ratios, status = [], [], [], [], []

    for row in tracks:
        try:
            hit = nearest_known_object(float(row["ra"]), float(row["dec"]),
                                       float(row["obsjd_mid"]), exclude_names=exclude_names)
        except SkybotUnavailable:
            names.append(""); seps.append(-1.0); motions.append(-1.0)
            ratios.append(-1.0); status.append("unchecked")
            continue
        if hit is None:
            names.append(""); seps.append(-1.0); motions.append(-1.0)
            ratios.append(-1.0); status.append("novel")
        else:
            n, s, mo = hit
            ratio = float(row["rate_arcsec_hr"]) / mo if mo > 0 else -1.0
            names.append(n); seps.append(s); motions.append(mo); ratios.append(ratio)
            status.append("known")
    tracks["nearest_known"] = names
    tracks["nearest_sep_arcsec"] = seps
    tracks["nearest_motion"] = motions
    tracks["rate_ratio"] = ratios
    tracks["novelty_status"] = status
    tracks["is_novel"] = [s == "novel" for s in status]
    return tracks
