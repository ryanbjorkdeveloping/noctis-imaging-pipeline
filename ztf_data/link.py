"""Tracklet linking: independent streak detections -> one moving object with a
measured motion vector.

The sweep emits one row per detection, scored + novelty-checked in ISOLATION -
nothing knows that two rows are the same asteroid. Linking closes that gap: it
groups detections across exposures (minutes apart, and often DIFFERENT fields, since
a fast mover crosses ZTF's tiling between a field's two nightly visits) into a track,
and MEASURES the motion. A track of >=3 timestamped astrometric points is the minimal
unit of a real discovery report, and its motion vector makes the novelty check far
more discriminating than a single position ever could.

Scope: INTRA-NIGHT linking, where sky motion is linear (great-circle, constant rate).
Inter-night linking (tracklet -> orbit, curved, heliocentric) is HelioLinC/THOR
territory - a separate project, deliberately not attempted here.

Scale: after the DeepStreaks cascade there are ~1-3 SHORT candidates per exposure, a
few dozen per night. Plain O(N^2) pair-building is entirely adequate; no kd-tree, no
orbit-space voting. Every separation is great-circle (SkyCoord.separation), never
dRA/dDec - the sweep crosses fields at Dec 15+, where the cos(dec) error is real.

An observation is the minimal input row: obsjd, ra, dec, streak_pa (+ optional
elongation/field/row_id provenance). streaks.streaks_from_run now emits exactly these.
"""

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table
import astropy.units as u

# --- gates: correctness bounds, NOT tuning knobs ------------------------------

# Streak-capable angular rate. Below RATE_MIN a mover is a POINT source (main-belt
# ~35"/hr) that the streak cascade would not have flagged anyway; above RATE_MAX it
# leaves the survey footprint between exposures and cannot be linked across them.
# BE5 sits at ~6700"/hr, comfortably inside. Matches novelty.MIN_STREAK_MOTION.
RATE_MIN_ARCSEC_HR = 1000.0
RATE_MAX_ARCSEC_HR = 20000.0

# How closely a pair's connecting direction must match EACH endpoint's own streak PA.
# The single-exposure streak PA is measured good to ~1-3 deg (well-resolved) / ~13 deg
# (marginal) on the BE5 control; 20 gives margin without admitting random pairs. This
# is the gate that makes linking cheap AND is the geometric FP filter: a pair is real
# only if BOTH streaks point along the line joining them.
PA_TOL_DEG = 20.0

# Prediction residual: an observation joins a growing track if it falls this close to
# the track's predicted great-circle position at its epoch. ~ fragment scatter + fit slop.
POS_TOL_ARCSEC = 60.0

# Within ONE exposure, merge detections this close into a single observation. A bright
# streak fragments into collinear pieces tens of arcsec apart along its trail; two
# unrelated fast movers landing this close in one exposure is far rarer. Also subsumes
# exact duplicates (same exposure re-run central-1000 AND full-frame -> <1" apart).
MERGE_ARCSEC = 60.0

# A track needs >=3 points. TWO points always define a line (rate+PA unconstrained), so
# a 2-"track" is meaningless; the third point - required to sit on the predicted line at
# the predicted time - is what actually rejects chance pairs.
MIN_DETECTIONS = 3


def _pa_sep_deg(a, b):
    """Folded [0,180) difference between two position angles (streak axes are
    undirected, so 179 deg and 1 deg are 2 deg apart, not 178)."""
    d = abs((a - b) % 180.0)
    return min(d, 180.0 - d)


def _circular_mean_pa(angles_deg):
    """Mean of undirected [0,180) angles via the doubled-angle trick (a plain mean
    breaks across the 0/180 wrap)."""
    ang = np.radians(2.0 * np.asarray(angles_deg, dtype=float))
    m = np.degrees(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())) / 2.0
    return float(m % 180.0)


def collapse_exposures(obs, merge_arcsec=MERGE_ARCSEC):
    """Merge same-exposure fragments/duplicates into one observation each.

    BE5 appears as 2-3 collinear fragments per exposure (3-sigma segmentation chops the
    ~57" trail into ~12" chunks). Fed raw to the linker, that ~10" spread reads as
    intra-exposure 'motion' (Delta t = 0 -> infinite rate) and corrupts every fit. Group
    by obsjd, single-link within merge_arcsec, emit one observation per physical streak:
    mean position, circular-mean streak PA, max elongation, fragment count + member ids.
    """
    need = {"obsjd", "ra", "dec", "streak_pa"}
    missing = need - set(obs.colnames)
    if missing:
        raise ValueError(f"observations missing required columns: {sorted(missing)}")

    has_elong = "elongation" in obs.colnames
    has_field = "field" in obs.colnames
    has_rowid = "row_id" in obs.colnames

    rows = []
    for jd in np.unique(np.asarray(obs["obsjd"])):
        grp = obs[np.asarray(obs["obsjd"]) == jd]
        coords = SkyCoord(np.asarray(grp["ra"]) * u.deg, np.asarray(grp["dec"]) * u.deg)
        assigned = -np.ones(len(grp), dtype=int)
        cid = 0
        for i in range(len(grp)):
            if assigned[i] >= 0:
                continue
            stack, assigned[i] = [i], cid       # single-linkage flood fill
            while stack:
                k = stack.pop()
                seps = coords[k].separation(coords).arcsec
                for j in np.where((seps < merge_arcsec) & (assigned < 0))[0]:
                    assigned[j] = cid
                    stack.append(int(j))
            cid += 1
        for c in range(cid):
            m = grp[assigned == c]
            row = {
                "obsjd": float(jd),
                "ra": float(np.asarray(m["ra"]).mean()),
                "dec": float(np.asarray(m["dec"]).mean()),
                "streak_pa": _circular_mean_pa(m["streak_pa"]),
                "n_frag": int(len(m)),
            }
            if has_elong:
                row["elongation"] = float(np.asarray(m["elongation"]).max())
            if has_field:
                row["field"] = int(np.asarray(m["field"])[0])
            if has_rowid:
                row["row_ids"] = ";".join(str(x) for x in m["row_id"])
            rows.append(row)
    return Table(rows=rows)


def _fit_motion(coords, epochs, members):
    """Least-squares LINEAR motion fit in a local tangent plane about the first member.

    A 2-point (endpoint) fit inherits the full ~30" fragment-centroid scatter of its two
    points; over a 70-min lever arm that drift blows past POS_TOL and breaks the track.
    A least-squares line through all members averages the scatter down to ~sigma/sqrt(N)
    (~10" here), which is what lets one model reach every point on the arc. Tangent-plane
    (spherical_offsets) is exact enough over a few degrees and round-trips via
    spherical_offsets_by. Returns a model dict consumed by _predict/_motion_vector.
    """
    ref = coords[members[0]]
    tt = (np.array([epochs[m] for m in members]) - epochs[members[0]]) * 24.0  # hours
    dlon, dlat = [], []
    for m in members:
        off = ref.spherical_offsets_to(coords[m])
        dlon.append(off[0].arcsec); dlat.append(off[1].arcsec)
    bl, al = np.polyfit(tt, dlon, 1)   # dlon = bl*t + al   (East, arcsec)
    bd, ad = np.polyfit(tt, dlat, 1)   # dlat = bd*t + ad   (North, arcsec)
    return {"ref": ref, "t0": epochs[members[0]], "al": al, "bl": bl, "ad": ad, "bd": bd}


def _predict(model, jd):
    """Model position at a given obsjd -> SkyCoord."""
    tt = (jd - model["t0"]) * 24.0
    return model["ref"].spherical_offsets_by((model["al"] + model["bl"] * tt) * u.arcsec,
                                             (model["ad"] + model["bd"] * tt) * u.arcsec)


def _motion_vector(model):
    """(rate_arcsec_hr, motion_pa_deg[0-360, E of N]) from a fitted model."""
    rate = float(np.hypot(model["bl"], model["bd"]))
    pa = float(np.degrees(np.arctan2(model["bl"], model["bd"])) % 360.0)  # atan2(E, N)
    return rate, pa


def link_tracklets(obs, min_detections=MIN_DETECTIONS):
    """Link collapsed observations into tracks with a measured motion vector.

    RANSAC-style: each valid seed pair (rate in band AND both streak PAs aligned with
    the connecting direction) defines a motion model; gather ALL observations consistent
    with it (predicted position within POS_TOL and streak PA aligned), least-squares
    refit, re-gather until the inlier set is stable. Keep maximal inlier sets with
    >= min_detections points. Gathering-then-refit (not incremental endpoint growth) is
    what makes one model span the whole arc. Returns one row per track with the fitted
    motion + a representative position at mid-epoch (for the novelty query).
    """
    obs = obs.copy()
    obs.sort("obsjd")
    n = len(obs)
    empty = Table(names=["track_id", "n_det", "n_fields", "arc_min", "rate_arcsec_hr",
                         "motion_pa_deg", "rms_arcsec", "obsjd_mid", "ra", "dec",
                         "row_ids"],
                  dtype=[int, int, int, float, float, float, float, float, float,
                         float, str])
    if n < min_detections:
        return empty

    coords = SkyCoord(np.asarray(obs["ra"]) * u.deg, np.asarray(obs["dec"]) * u.deg)
    epochs = np.asarray(obs["obsjd"], dtype=float)
    pa = np.asarray(obs["streak_pa"], dtype=float)

    def gather(model):
        """Indices consistent with a motion model: predicted position within POS_TOL
        and streak PA aligned with the model's direction of motion."""
        _, mpa = _motion_vector(model)
        keep = []
        for k in range(n):
            if _predict(model, epochs[k]).separation(coords[k]).arcsec < POS_TOL_ARCSEC \
                    and _pa_sep_deg(mpa, pa[k]) <= PA_TOL_DEG:
                keep.append(k)
        return keep

    inlier_sets = set()
    for i in range(n):
        for j in range(i + 1, n):
            dt_hr = (epochs[j] - epochs[i]) * 24.0
            if dt_hr <= 0:                       # same (collapsed) exposure: not a pair
                continue
            rate = coords[i].separation(coords[j]).arcsec / dt_hr
            if not (RATE_MIN_ARCSEC_HR < rate < RATE_MAX_ARCSEC_HR):
                continue
            vpa = coords[i].position_angle(coords[j]).deg
            if _pa_sep_deg(vpa, pa[i]) > PA_TOL_DEG or _pa_sep_deg(vpa, pa[j]) > PA_TOL_DEG:
                continue
            members = [i, j]                     # gather -> refit until stable
            for _ in range(5):
                got = gather(_fit_motion(coords, epochs, members))
                if set(got) == set(members) or len(got) < 2:
                    break
                members = got
            if len(members) >= min_detections:
                inlier_sets.add(frozenset(members))

    # Merge inlier sets that share >=2 observations: two points fix a line, so any two
    # candidate tracks sharing a pair are the SAME mover seen from different seeds (this
    # is what fuses the per-seed partials into one full track). Distinct crossing movers
    # never share 2 epochs, and spurious sets never reach min_detections, so it is safe.
    # Union-find over the candidate sets.
    sets = list(inlier_sets)
    parent = list(range(len(sets)))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for a in range(len(sets)):
        for b in range(a + 1, len(sets)):
            if len(sets[a] & sets[b]) >= 2:
                parent[find(a)] = find(b)
    merged = {}
    for a in range(len(sets)):
        merged.setdefault(find(a), set()).update(sets[a])
    tracks = list(merged.values())

    has_field = "field" in obs.colnames
    has_rowid = "row_ids" in obs.colnames
    out = []
    for s in sorted(tracks, key=lambda x: -len(x)):
        # Final outlier rejection: the >=2-shared union can fuse in a lucky contaminant
        # (a negative that chanced onto 2 track points), which wrecks the fit. Iteratively
        # drop the worst-residual member while it exceeds POS_TOL - the true points
        # dominate the least-squares fit, so a gross outlier is removed and the rms recovers.
        members = sorted(s)
        while len(members) > min_detections:
            model = _fit_motion(coords, epochs, members)
            resid = [(m, _predict(model, epochs[m]).separation(coords[m]).arcsec) for m in members]
            worst_m, worst_r = max(resid, key=lambda x: x[1])
            if worst_r <= POS_TOL_ARCSEC:
                break
            members = [m for m in members if m != worst_m]
        if len(members) < min_detections:
            continue
        model = _fit_motion(coords, epochs, members)
        rate, motion_pa = _motion_vector(model)
        rms = float(np.sqrt(np.mean([
            _predict(model, epochs[m]).separation(coords[m]).arcsec ** 2 for m in members])))
        arc_min = (epochs[members[-1]] - epochs[members[0]]) * 1440.0
        jd_mid = float(np.mean([epochs[m] for m in members]))
        mid = _predict(model, jd_mid)
        n_fields = (len(set(int(obs["field"][m]) for m in members)) if has_field else 0)
        rids = (";".join(str(obs["row_ids"][m]) for m in members) if has_rowid else "")
        out.append((len(out), len(members), n_fields, arc_min, rate, motion_pa, rms,
                    jd_mid, float(mid.ra.deg), float(mid.dec.deg), rids))
    if not out:
        return empty
    return Table(rows=out, names=empty.colnames)


SHORT_ROUTE = "SHORT_NEA_CANDIDATE"


def link_from_candidates(candidates, min_detections=MIN_DETECTIONS):
    """Convenience: collapse same-exposure fragments, then link. `candidates` is a
    Table (or path to an .ecsv) of streak candidates carrying obsjd/ra/dec/streak_pa.

    Restricts to SHORT_NEA_CANDIDATE rows when a `route` column is present. This is
    load-bearing: LONG_SATELLITE streaks are fast enough to LINK, but a satellite is not
    in SkyBoT's asteroid catalog, so a satellite track would read `novel` and fabricate a
    discovery. The sl gate already told asteroid-short from satellite-long - honour it.
    """
    if isinstance(candidates, str):
        candidates = Table.read(candidates)
    if "route" in candidates.colnames and len(candidates):
        candidates = candidates[np.asarray(candidates["route"]) == SHORT_ROUTE]
    return link_tracklets(collapse_exposures(candidates), min_detections=min_detections)


if __name__ == "__main__":
    import os
    import sys

    from ztf_data.paths import OWN_ROOT
    from ztf_data.novelty import annotate_track_novelty

    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(OWN_ROOT, "sweep_candidates.ecsv")
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(OWN_ROOT, "tracks.ecsv")
    tracks = link_from_candidates(src)
    if len(tracks):
        annotate_track_novelty(tracks)          # known vs NEW, by position + measured rate
    tracks.write(out, format="ascii.ecsv", overwrite=True)
    print(f"{len(tracks)} track(s) from {src} -> {out}")
    if len(tracks):
        tracks.pprint(max_lines=-1, max_width=-1)
        novel = tracks[np.asarray(tracks["is_novel"])]
        print(f"\nDISCOVERY CANDIDATES (is_novel): {len(novel)} of {len(tracks)} track(s)")
