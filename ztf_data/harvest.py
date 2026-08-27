"""Resumable ZTF survey harvester: enqueue sky, run the pipeline, ingest results.

    enqueue -> claim batch -> pool.map(_harvest_one) -> parent ingests -> link per night

MUST be run as a guarded __main__. n_workers>1 uses spawn, spawned children re-import
this module at top level, and an unguarded pool call recursively spawns
("RuntimeError: ... bootstrapping phase"). Same rule as
streak_testing/hunt_field518_20190208.py.

Relationship to sweep.py: none, deliberately. sweep_targets/_sweep_one stay exactly as
they are - hunt_field518_20190208.py and the BE5 validation depend on them, and they
write the two global ECSVs. This is a parallel path with a database behind it, not a
replacement. The only thing borrowed is _cap_worker_threads, imported rather than copied.

Concurrency: workers touch ONLY their own run dir and the shared $ZTFDATA/sci tree
(collision-safe since resolve_local pins ccdid+qid). They return plain JSON-able dicts.
The PARENT owns the only DB connection - see the rule in store.py.
"""

import argparse
import datetime
from concurrent import futures
import json
import multiprocessing
import os
import time
import traceback

import config  # noqa: F401  LOAD-BEARING: before anything reaches ztfquery

import numpy as np
from astropy.table import Table, vstack

from ztf_data import store
from ztf_data.epochs import IRSAUnavailable
from ztf_data.link import link_from_candidates
from ztf_data.night import enumerate_night, night_targets, TARGET_KEYS
from ztf_data.novelty import annotate_novelty, annotate_track_novelty
from ztf_data.run_field import run_field
from ztf_data.store import (
    STAGE_DETECT, STAGE_STAMPS, STAGE_STREAK, STAGE_NOVELTY, STAGE_BRAAI,
    STAGE_TYPE_B, STAGE_VERDICT, BRANCH_MASKS,
)
from ztf_data.streaks import streaks_from_run, _row_id
from ztf_data.sweep import _cap_worker_threads, STREAK_ROUTES

from ztf_classification.pipeline import merge_verdict

DEFAULT_WANT = BRANCH_MASKS["motion"]
LOCKFILE = store.OWN_ROOT / "harvest.lock"

# catalog columns copied straight into the detections table
_CAT_NUMERIC = ["ra", "dec", "x_centroid", "y_centroid", "snr", "segment_flux",
                "elongation", "orientation"]


# ---------------------------------------------------------------- worker side


def _target_kwargs(t):
    """The 5 keys run_field accepts. Extra enumerator columns would be a TypeError."""
    return {k: int(t[k]) for k in TARGET_KEYS}


def _rows_from_catalog(fr):
    """catalog.ecsv -> one plain dict per detection, keyed by the stamp basename.

    row_id (the stamp basename) IS the join key the whole Stage-4 pipeline uses, and
    stays the per-exposure key here; store.det_uid makes it globally unique on ingest.
    """
    cat = Table.read(fr.catalog_path)
    rows = []
    for r in cat:
        d = {"row_id": _row_id(str(r["stamp_path"]))}
        for c in _CAT_NUMERIC:
            d[c] = float(r[c]) if c in cat.colnames else None
        d["sign"] = int(r["sign"]) if "sign" in cat.colnames else None
        d["on_edge"] = bool(r["on_edge"]) if "on_edge" in cat.colnames else None
        # relative, so a run dir can move (or be synced to another machine) intact
        d["stamp_rel"] = os.path.relpath(str(r["stamp_path"]), fr.run_dir) \
            if "stamp_path" in cat.colnames else None
        rows.append(d)
    return rows


def _merge_by_row_id(rows, table, columns):
    """Copy `columns` from an astropy table onto the matching row dicts."""
    if table is None or len(table) == 0:
        return
    by_id = {str(r["row_id"]): r for r in table}
    for d in rows:
        src = by_id.get(d["row_id"])
        if src is None:
            continue
        for c in columns:
            if c in table.colnames:
                v = src[c]
                d[c] = bool(v) if isinstance(v, (bool, np.bool_)) else (
                    None if v is None else (float(v) if isinstance(v, (float, np.floating))
                                            else (int(v) if isinstance(v, (int, np.integer))
                                                  else str(v))))


def _harvest_one(args):
    """(target, want_mask, run_kwargs) -> plain-dict summary. NEVER raises.

    Mirrors sweep._sweep_one's contract so one bad target cannot kill the pool.
    scratch_dir stays None (load-bearing): streaks_from_run then writes its
    .venv_streaks handoff into the target's OWN run dir, so concurrent workers under
    spawn cannot clobber each other's stamps.
    """
    tgt, want, run_kwargs = args
    t0 = time.time()
    # full_frame is a property of the TARGET, not of the batch. run_batch defaults it
    # from targets[0], but claim() orders by time and freely mixes _F and _C rows, so
    # the batch default silently applied one target's region to all of them: a _C
    # target ran full-frame, computed the _F key, and ingested under a DIFFERENT
    # target's identity - while the _C row it was claimed as stayed 'running' forever.
    # The region is part of target_key precisely because it changes the catalog
    # wholesale (~950 detections vs ~114), so it must be read per target.
    run_kwargs = dict(run_kwargs)
    if tgt.get("full_frame") is not None:
        run_kwargs["full_frame"] = bool(tgt["full_frame"])
    tkey = store.target_key(tgt["field"], tgt["ccdid"], tgt["qid"], tgt["fid"],
                            tgt["science_filefracday"],
                            bool(run_kwargs.get("full_frame", False)))
    out = {"target_key": tkey, "error": None, "retryable": False,
           "stage_mask": 0, "detections": [], "seconds": 0.0}
    done = 0
    try:
        fr = run_field(**_target_kwargs(tgt), **run_kwargs)
        done |= STAGE_DETECT
        if not run_kwargs.get("streaks_only"):
            done |= STAGE_STAMPS
        rows = _rows_from_catalog(fr)
        out.update(run_dir=fr.run_dir, obsjd=fr.obsjd, corners=fr.corners,
                   ra_center=fr.ra_center, dec_center=fr.dec_center,
                   ref_key=fr.ref_key, science_infobits=fr.science_infobits,
                   full_frame=fr.full_frame, n_detections=fr.n_detections,
                   region_shape=list(_region_shape(fr)))

        streaks = None
        if want & STAGE_STREAK:
            # A stage that could not RUN must not discard the stages that did. The
            # DeepStreaks scorer lives in a separate venv, so it can be absent or
            # broken (observed 2026-08-12: .venv_streaks gutted to include/ + lib/,
            # no interpreter) while detection is perfectly healthy. Failing the whole
            # target there threw away ~1300 real detections per exposure AND flipped
            # already-ingested legacy targets to 'failed'.
            #
            # Leaving STAGE_STREAK unset is the load-bearing half: the target still
            # does not satisfy want_mask, so it stays claimable and re-runs the streak
            # stage once the scorer is back. An unscored row keeps streak_route NULL,
            # which the UI renders as "not run" - never as BOGUS. Same three-state
            # discipline as novelty's 'unchecked'.
            try:
                streaks = streaks_from_run(fr, scratch_dir=None)
                _merge_by_row_id(rows, streaks, ["route", "rb_pass", "kd_pass",
                                                 "sl_short", "streak_pa"])
                for d in rows:                 # `route` is the catalog's name for it
                    d["streak_route"] = d.pop("route", None)
                done |= STAGE_STREAK
            except Exception:
                streaks = None
                out["warning"] = ("streak scoring unavailable: "
                                  + traceback.format_exc().strip().splitlines()[-1])

        if (want & STAGE_NOVELTY) and streaks is not None and len(streaks):
            # only the streak-routed rows are worth a SkyBoT call - a point source is
            # not a fast mover, and the cone query is the expensive part.
            sel = streaks[np.isin(np.asarray(streaks["route"]), list(STREAK_ROUTES))]
            if len(sel):
                annotate_novelty(sel, float(fr.obsjd))
                _merge_by_row_id(rows, sel, ["nearest_known", "nearest_sep_arcsec",
                                             "nearest_motion", "novelty_status"])
            done |= STAGE_NOVELTY

        # A verdict is only honest once the branch that produces it has RUN.
        # merge_verdict reads r.get("braai_pass", False) and so returns "bogus" for a
        # motion-only row - which would label every detection in the wide tier bogus
        # when braai simply never executed. Same trap as novelty's 'unchecked'.
        if want & STAGE_VERDICT and (done & STAGE_BRAAI):
            for d in rows:
                d["final_verdict"] = merge_verdict(d)
            done |= STAGE_VERDICT

        out["detections"] = rows
    except IRSAUnavailable as e:
        out["error"] = f"{type(e).__name__}: {e}"
        out["retryable"] = True                # an outage must not burn an attempt
    except Exception:
        out["error"] = traceback.format_exc()
    out["stage_mask"] = done
    out["seconds"] = round(time.time() - t0, 2)
    return out


def _region_shape(fr):
    try:
        with open(os.path.join(fr.run_dir, "manifest.json")) as f:
            return json.load(f).get("region_shape") or (None, None)
    except (OSError, ValueError):
        return (None, None)


# ---------------------------------------------------------------- parent side


def enqueue_night(con, date, ra_range=None, dec_range=None, fields=None, fid=2,
                  ut_hour_range=None, full_frame=True, want_mask=DEFAULT_WANT,
                  max_targets=None):
    """enumerate_night -> INSERT OR IGNORE into targets. Returns rows newly added.

    Carries ra/dec/obsjd/seeing/maglimit/infobits/ut_date straight off the enumerator
    table: free provenance, and the exposure footprint needs it. Idempotent.
    """
    table = enumerate_night(date, ra_range=ra_range, dec_range=dec_range, fields=fields,
                            fid=fid, ut_hour_range=ut_hour_range)
    if max_targets and len(table) > max_targets:
        print(f"  capping {len(table)} -> {max_targets} exposures (earliest first)")
        table = table[:max_targets]
    rows = []
    for t, r in zip(night_targets(table), table):
        t = dict(t)
        t["ut_date"] = date
        t["full_frame"] = int(full_frame)
        for c in ("obsjd", "ra", "dec", "seeing", "maglimit", "infobits"):
            if c in table.colnames:
                t[c] = float(r[c]) if c != "infobits" else int(r[c])
        rows.append(t)
    return store.enqueue(con, rows, want_mask, full_frame=full_frame)


def enqueue_local(con, product="scimrefdiffimg.fits.fz", fid=None, fields=None,
                  full_frame=True, want_mask=DEFAULT_WANT, min_epochs=1):
    """Enqueue every CACHED exposure of `product`, reading headers instead of IRSA.

    The offline twin of enqueue_night. Two reasons it is a first-class entry point and
    not a debugging shim:

      1. Download and processing are separable on purpose (see epochs.local_grid). A
         machine can pull frames in bulk once and reduce them repeatedly, which is the
         only sane shape for a long harvest across two machines.
      2. IRSA goes down - fully unreachable for two sessions (2026-08-11/12, connect
         timeout, not even a 502). Every cached diff was sitting on disk, processable,
         behind a query to a dead server for facts the FILES already carry.

    ut_date comes from the filefracday stamp, which is ZTF's own MJD-based night label,
    so nights group exactly as enumerate_night would group them - and the linker's
    per-night scoping depends on that being right. obsjd/seeing/infobits are read from
    the header (HDU 1 on fpacked diffs), so they are the real values, not reconstructed.

    min_epochs drops cells with too few cached exposures to be worth a run dir; it is a
    convenience filter, never a science cut - MIN_DETECTIONS in link.py is what decides
    whether a track can form.
    """
    import glob
    import re

    from astropy.io import fits
    from ztfquery.io import LOCALSOURCE

    pattern = os.path.join(LOCALSOURCE, "sci", "*", "*", "*", f"ztf_*{product}")
    by_cell = {}
    for path in sorted(glob.glob(pattern)):
        m = re.match(
            r"ztf_(\d{14})_(\d{6})_[a-zA-Z]+_c(\d{2})_o_q(\d)_", os.path.basename(path))
        if not m:
            continue
        ffd, field, ccdid, qid = (int(m.group(1)), int(m.group(2)),
                                  int(m.group(3)), int(m.group(4)))
        if fields and field not in fields:
            continue
        try:
            h = fits.getheader(path)
            if "OBSJD" not in h:                      # fpacked: HDU 0 is a stub
                h = fits.getheader(path, 1)
        except (OSError, IndexError, KeyError):
            continue
        this_fid = int(h.get("FILTERID", -1))
        if fid is not None and this_fid != int(fid):
            continue
        by_cell.setdefault((field, ccdid, qid, this_fid), []).append({
            "field": field, "ccdid": ccdid, "qid": qid, "fid": this_fid,
            "science_filefracday": ffd,
            "ut_date": _ut_date_from_ffd(ffd),
            "full_frame": int(full_frame),
            "obsjd": float(h["OBSJD"]),
            "seeing": float(h.get("SEEING", float("nan"))),
            "infobits": int(h.get("INFOBITS", 0)),
        })

    rows, skipped = [], []
    for cell, epochs in sorted(by_cell.items()):
        (rows if len(epochs) >= min_epochs else skipped).extend(epochs)
    print(f"  cached {product}: {len(by_cell)} cells, "
          f"{sum(len(v) for v in by_cell.values())} exposures")
    for cell, epochs in sorted(by_cell.items(), key=lambda kv: -len(kv[1])):
        mark = " " if len(epochs) >= min_epochs else " (skipped, < min_epochs)"
        print(f"    {cell[0]:06d}/c{cell[1]:02d}/q{cell[2]}/f{cell[3]}: "
              f"{len(epochs)} epochs{mark}")
    if skipped:
        print(f"  {len(skipped)} exposures below min_epochs={min_epochs}, not enqueued")
    return store.enqueue(con, rows, want_mask, full_frame=full_frame)


def enqueue_cadence(con, date, n_cells=5, n_epochs=10, spread=False, **kw):
    """Enqueue the n_cells with the MOST visits that night, n_epochs each.

    This shape is not an optimization, it is what makes the run possible:
      - link.MIN_DETECTIONS=3 is unreachable on ZTF's usual ~2 visits/night per field,
        so only a high-cadence cell can ever produce a track;
      - the deep template amortizes over a cell, so 10 epochs of one cell costs ONE
        5-10 min stack while 10 distinct cells cost ten (and 10x the 1.5 GB download).

    spread=True takes the best cell from each of n_cells DISTINCT FIELDS instead of
    the global top-n. Pure visit-count ranking picks whole quadrants off one field -
    a ZTF field is read out as 64 quadrants that all share a visit count, so the top
    10 are typically 665/c01/q1..c03/q2: ten genuinely distinct patches, but tiled
    into one contiguous ~3 deg^2 block. For "ten different areas of sky" that is the
    wrong answer, and on a survey map it renders as a single blob. spread trades a
    little cadence for real angular separation; each chosen cell is still that field's
    highest-cadence one, so linking is preserved.
    """
    table = enumerate_night(date, **{k: v for k, v in kw.items()
                                     if k in ("ra_range", "dec_range", "fields", "fid",
                                              "ut_hour_range")})
    cells = {}
    for r in table:
        cells.setdefault((int(r["field"]), int(r["ccdid"]), int(r["qid"]),
                          int(r["fid"])), []).append(r)
    if spread:
        best = {}                      # field -> (cell, visits), most-visited per field
        for cell, visits in cells.items():
            if cell[0] not in best or len(visits) > len(best[cell[0]][1]):
                best[cell[0]] = (cell, visits)
        ranked = [kv for kv in sorted(best.values(), key=lambda cv: -len(cv[1]))][:n_cells]
    else:
        ranked = sorted(cells.items(), key=lambda kv: -len(kv[1]))[:n_cells]
    print(f"  {len(cells)} cells this night; taking {len(ranked)} by visit count: "
          + ", ".join(f"{k[0]}/c{k[1]:02d}/q{k[2]} ({len(v)})" for k, v in ranked))

    full_frame = kw.get("full_frame", True)
    want = kw.get("want_mask", DEFAULT_WANT)
    rows = []
    for (field, ccdid, qid, fid), visits in ranked:
        visits = sorted(visits, key=lambda r: float(r["obsjd"]))[:n_epochs]
        for r in visits:
            t = {"field": field, "ccdid": ccdid, "qid": qid, "fid": fid,
                 "science_filefracday": int(r["science_filefracday"]),
                 "ut_date": date, "full_frame": int(full_frame)}
            for c in ("obsjd", "ra", "dec", "seeing", "maglimit", "infobits"):
                if c in table.colnames:
                    t[c] = float(r[c]) if c != "infobits" else int(r[c])
            rows.append(t)
    return store.enqueue(con, rows, want, full_frame=full_frame)


def run_batch(con, n_workers=1, batch=32, want_mask=DEFAULT_WANT, **run_kwargs):
    """Claim -> run -> ingest. Safe to kill at any point. Returns counts."""
    code = store.git_sha()
    targets = store.claim(con, want_mask, limit=batch, code_version=code)
    if not targets:
        return {"claimed": 0, "done": 0, "failed": 0, "retryable": 0}

    run_kwargs = dict(run_kwargs)
    run_kwargs.setdefault("streaks_only", not (want_mask & STAGE_STAMPS))
    run_kwargs.setdefault("full_frame", bool(targets[0]["full_frame"]))
    args = [(t, want_mask, run_kwargs) for t in targets]

    counts = {"claimed": len(targets), "done": 0, "failed": 0, "retryable": 0}
    if n_workers > 1:
        _cap_worker_threads(n_workers)         # in the PARENT, before the pool starts
        ctx = multiprocessing.get_context("spawn")
        # ProcessPoolExecutor, NOT multiprocessing.Pool. When a worker is killed from
        # outside - and under memory pressure it is; each worker carries its own
        # TensorFlow + 9 DeepStreaks models on top of full-frame float arrays -
        # Pool.imap blocks forever on a result that can never arrive. Observed
        # 2026-08-12: all 3 workers gone, parent at 0% CPU, 14 targets stuck 'running',
        # no error, no timeout. For an unattended overnight harvest a silent hang is
        # the worst failure mode there is; a BrokenProcessPool is recoverable.
        try:
            with futures.ProcessPoolExecutor(max_workers=n_workers,
                                             mp_context=ctx) as ex:
                # as_completed, NOT map: map yields strictly in submission order, so a
                # single slow target blocks the ingest of every finished one behind it.
                # Each ingest is its own committed transaction, so ordering costs real
                # durability - a crash 20 targets into a batch of 24 kept none of them
                # if target 1 was still running. Results now land as they finish.
                fut = {ex.submit(_harvest_one, a): a for a in args}
                _ingest_stream(con, (f.result() for f in futures.as_completed(fut)),
                               targets, counts, code)
        except futures.process.BrokenProcessPool as e:
            # Whatever was claimed but not ingested goes back to pending, WITHOUT
            # consuming an attempt: the worker died of resource pressure, which says
            # nothing about whether the target is processable.
            n = store.reset_running(con)
            counts["broken_pool"] = True
            print(f"  POOL DIED ({e}); {n} claimed target(s) returned to pending.\n"
                  f"  Workers are memory-hungry (TF + 9 CNNs each). Retry with fewer:"
                  f" --workers {max(1, n_workers // 2)}")
    else:
        _ingest_stream(con, (_harvest_one(a) for a in args), targets, counts, code)
    return counts


def _ingest_stream(con, results, targets, counts, code):
    # A worker must report back under the SAME identity it was claimed as. If it does
    # not, the claimed row stays 'running' forever while another target's record is
    # overwritten - silent, and invisible in every count. Cheap to check, so check.
    claimed = {store.target_key(t["field"], t["ccdid"], t["qid"], t["fid"],
                                t["science_filefracday"], bool(t["full_frame"]))
               for t in targets}
    # stage_mask BEFORE the batch, so we can tell real work from a target that was
    # re-run to produce exactly what it already had - see the livelock note below.
    prior = {store.target_key(t["field"], t["ccdid"], t["qid"], t["fid"],
                              t["science_filefracday"], bool(t["full_frame"])):
             (t["stage_mask"] or 0) for t in targets}
    for i, res in enumerate(results, 1):
        label = res["target_key"]
        if label not in claimed:
            counts["misrouted"] = counts.get("misrouted", 0) + 1
            print(f"  [{i}/{len(targets)}] BUG: worker returned key {label!r}, which "
                  f"was never claimed in this batch - NOT ingesting")
            continue
        if res["error"]:
            store.record_failure(con, label, res["error"], retryable=res["retryable"])
            counts["retryable" if res["retryable"] else "failed"] += 1
            first = res["error"].strip().splitlines()[-1][:120]
            kind = "RETRY" if res["retryable"] else "FAILED"
            print(f"  [{i}/{len(targets)}] {label}: {kind} - {first}")
            continue
        store.ingest(con, res, code_version=code)
        counts["done"] += 1
        # A target that finishes with no NEW stage did work that changes nothing, and
        # claim() will offer it again next batch - a livelock that burns full CPU and
        # looks like progress in every count. Cheap to notice, so notice it.
        if (res.get("stage_mask", 0) | prior.get(label, 0)) == prior.get(label, 0):
            counts["no_progress"] = counts.get("no_progress", 0) + 1
        # a PARTIAL is not a success: say so, or a degraded harvest reads as a clean
        # one and the missing stage is only noticed as absent data in the UI.
        warn = res.get("warning")
        if warn:
            counts["partial"] = counts.get("partial", 0) + 1
        print(f"  [{i}/{len(targets)}] {label}: {res.get('n_detections', 0)} detections"
              f" in {res['seconds']}s" + (f"  [PARTIAL: {warn[:90]}]" if warn else ""))


def link_pending_nights(con, dates=None, min_detections=3):
    """Link each UT night INDEPENDENTLY, then novelty-check the tracks.

    PER-NIGHT IS A CORRECTNESS REQUIREMENT, not tidiness. link_tracklets' seed-pair
    filter is only `dt_hr > 0` plus a rate window - there is NO maximum time
    separation - so a pair 24 h apart is admitted at 6.7-133 deg of sky. Feeding it a
    multi-night candidate set fabricates tracks. link.py says "Scope: INTRA-NIGHT" and
    means it; the grouping has to happen here.

    Detections are handed to the linker with row_id REPLACED BY det_uid, so the
    ;-joined `row_ids` a track carries maps back unambiguously. row_id alone restarts
    per exposure and would fuse unrelated detections across the night.
    """
    if dates is None:
        dates = [r[0] for r in con.execute(
            "SELECT DISTINCT ut_date FROM exposures WHERE ut_date IS NOT NULL"
            " ORDER BY ut_date").fetchall()]
    link_run = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    total = 0
    for date in dates:
        rows = con.execute(
            "SELECT d.det_uid, d.ra, d.dec, d.elongation, d.streak_pa, d.streak_route,"
            "       e.obsjd, e.field, e.exposure_id, d.row_id"
            "  FROM detections d JOIN exposures e USING(exposure_id)"
            " WHERE e.ut_date = ? AND d.streak_route = 'SHORT_NEA_CANDIDATE'"
            "   AND d.streak_pa IS NOT NULL", (date,)).fetchall()
        if len(rows) < min_detections:
            continue
        cand = Table({
            "row_id": [r["det_uid"] for r in rows],          # det_uid, see docstring
            "ra": [r["ra"] for r in rows], "dec": [r["dec"] for r in rows],
            "obsjd": [r["obsjd"] for r in rows],
            "streak_pa": [r["streak_pa"] for r in rows],
            "elongation": [r["elongation"] or 0.0 for r in rows],
            "field": [r["field"] for r in rows],
            "route": ["SHORT_NEA_CANDIDATE"] * len(rows),
        })
        tracks = link_from_candidates(cand, min_detections=min_detections)
        if not len(tracks):
            continue
        annotate_track_novelty(tracks)
        by_uid = {r["det_uid"]: r for r in rows}
        for i, t in enumerate(tracks):
            uid = f"{date}:{link_run}:{i}"
            members = []
            for chunk in str(t["row_ids"]).split(";"):
                for duid in chunk.split(";"):
                    src = by_uid.get(duid.strip())
                    if src:
                        members.append({"exposure_id": src["exposure_id"],
                                        "row_id": src["row_id"], "det_uid": duid.strip(),
                                        "obsjd": src["obsjd"], "ra": src["ra"],
                                        "dec": src["dec"], "resid_arcsec": None})
            store.upsert_track(con, {
                "track_uid": uid, "n_det": int(t["n_det"]),
                "n_fields": int(t["n_fields"]), "arc_min": float(t["arc_min"]),
                "rate_arcsec_hr": float(t["rate_arcsec_hr"]),
                "motion_pa_deg": float(t["motion_pa_deg"]),
                "rms_arcsec": float(t["rms_arcsec"]),
                "obsjd_mid": float(t["obsjd_mid"]), "ra": float(t["ra"]),
                "dec": float(t["dec"]),
                "novelty_status": str(t["novelty_status"]),
                "nearest_known": str(t["nearest_known"]),
                "nearest_sep_arcsec": float(t["nearest_sep_arcsec"]),
                "nearest_motion": float(t["nearest_motion"]),
                "rate_ratio": float(t["rate_ratio"]),
            }, members, ut_date=date, link_run=link_run)
            total += 1
        print(f"  {date}: {len(tracks)} track(s) from {len(rows)} SHORT candidates")
    return total


def _ut_date_from_ffd(ffd):
    """ZTF's filefracday starts with the UT calendar date: 20190208231343 -> 2019-02-08."""
    s = str(int(ffd))
    return f"{s[0:4]}-{s[4:6]}-{s[6:8]}"


def _obsjd_from_ffd(ffd):
    """Derive obsjd from the filename stamp. Accurate to ~1 s (measured against 4 real
    headers: -1.04, -1.00, -0.02, +0.01 s).

    Used ONLY as a last-resort fallback for the exposure index. Never good enough for
    LINKING: 1 s at 6900"/hr is 1.9" of sky, and the linker's POS_TOL is 60". Runs that
    can link have a real obsjd in their streak_candidates.ecsv, which is preferred.
    The fracday follows MJD, not JD - obsjd % 1 is NOT the stamp (see epochs.py).
    """
    from astropy.time import Time
    s = str(int(ffd))
    mjd = Time(f"{s[0:4]}-{s[4:6]}-{s[6:8]}").mjd + float("0." + s[8:])
    return mjd + 2400000.5


def _legacy_footprint(run_dir, manifest, cat):
    """(ra_center, dec_center, corners, source). Prefers the diff's real WCS.

    Falls back to the detections' ra/dec extent when the diff has been evicted - close,
    since a full-frame catalog spans the chip, but NOT the surveyed region outline. The
    source is recorded so the map can render an approximate footprint honestly rather
    than presenting it as measured coverage.
    """
    from ztf_data.detect import load_diff, central_cutout, full_frame_cutout, CUTOUT_SIZE
    from ztf_data.fetch import resolve_local
    from ztf_data.run_field import _footprint
    try:
        diff = resolve_local(int(manifest["science_filefracday"]), "scimrefdiffimg.fits.fz",
                             field=int(manifest["field"]), ccdid=int(manifest["ccdid"]),
                             qid=int(manifest["qid"]))
        data, wcs = load_diff(diff)
        cut = (full_frame_cutout(data, wcs=wcs) if manifest.get("full_frame")
               else central_cutout(data, wcs=wcs, size=CUTOUT_SIZE))
        ra_c, dec_c, corners = _footprint(cut.wcs, cut.data.shape)
        return ra_c, dec_c, corners, "wcs"
    except Exception:
        pass
    if not len(cat):
        return None, None, None, None
    ra, dec = np.asarray(cat["ra"], float), np.asarray(cat["dec"], float)
    r0, r1, d0, d1 = ra.min(), ra.max(), dec.min(), dec.max()
    return (round(float(ra.mean()), 6), round(float(dec.mean()), 6),
            [[r1, d0], [r0, d0], [r0, d1], [r1, d1]], "catalog_bbox")


def import_legacy(con, runs_root=None, limit=None):
    """Adopt run dirs produced BEFORE the store existed, without re-running anything.

    115 exposures across 18 grid cells are already on disk from the BE5 recovery, the
    blind hunts and the 2020 fixture. Their catalogs and streak scores are finished
    work; re-deriving them would cost hours of IRSA and CPU for identical numbers.

    Their manifests predate the obsjd/corners/ref_key fields, so those are recovered:
    obsjd from the run's own streak_candidates.ecsv (real, from the diff header) and
    only otherwise from the filename stamp; the footprint from the diff WCS when the
    file survives, else the catalog's extent, recorded as such.
    """
    import glob
    root = runs_root or os.path.join(store.OWN_ROOT, "runs")
    manifests = sorted(glob.glob(os.path.join(root, "*", "*", "manifest.json")))
    if limit:
        manifests = manifests[:limit]
    stats = {"runs": 0, "imported": 0, "skipped": 0, "detections": 0, "bbox": 0,
             "dup_dirs": 0, "ambiguous": 0}

    canonical, dup_count = _dedupe_legacy(manifests)

    for mpath in manifests:
        stats["runs"] += 1
        if canonical.get(mpath) is False:
            stats["dup_dirs"] += 1        # another dir holds this identical catalog
            continue
        rdir = os.path.dirname(mpath)
        try:
            with open(mpath) as f:
                m = json.load(f)
            cat_path = os.path.join(rdir, "catalog.ecsv")
            if not os.path.exists(cat_path):
                stats["skipped"] += 1
                continue
            cat = Table.read(cat_path)
        except (OSError, ValueError):
            stats["skipped"] += 1
            continue

        ffd = int(m["science_filefracday"])
        full_frame = bool(m.get("full_frame", False))
        ut_date = _ut_date_from_ffd(ffd)

        streaks = None
        spath = os.path.join(rdir, "streak_candidates.ecsv")
        if os.path.exists(spath):
            try:
                streaks = Table.read(spath)
            except ValueError:
                streaks = None

        obsjd = None
        if streaks is not None and "obsjd" in streaks.colnames and len(streaks):
            obsjd = float(streaks["obsjd"][0])          # real, from the diff header
        if obsjd is None:
            obsjd = _obsjd_from_ffd(ffd)                # ~1 s, index only

        tgt = {"field": int(m["field"]), "ccdid": int(m["ccdid"]), "qid": int(m["qid"]),
               "fid": int(m["fid"]), "science_filefracday": ffd, "ut_date": ut_date,
               "obsjd": obsjd, "infobits": m.get("science_infobits"),
               "full_frame": int(full_frame)}

        stage = STAGE_DETECT
        if m.get("stamps_written"):
            stage |= STAGE_STAMPS
        if streaks is not None:
            stage |= STAGE_STREAK

        store.enqueue(con, [tgt], stage, full_frame=full_frame)
        tkey = store.target_key(tgt["field"], tgt["ccdid"], tgt["qid"], tgt["fid"],
                                ffd, full_frame)

        fr = _LegacyResult(rdir, cat_path, os.path.join(rdir, "cutouts"))
        rows = _rows_from_catalog(fr)
        if streaks is not None:
            _merge_by_row_id(rows, streaks, ["route", "rb_pass", "kd_pass",
                                             "sl_short", "streak_pa"])
            for d in rows:
                d["streak_route"] = d.pop("route", None)

        ra_c, dec_c, corners, src = _legacy_footprint(rdir, m, cat)
        if src == "catalog_bbox":
            stats["bbox"] += 1
        claimed = dup_count.get(mpath, 1)
        if claimed > 1:
            stats["ambiguous"] += 1

        store.ingest(con, {
            "target_key": tkey, "stage_mask": stage, "run_dir": rdir, "obsjd": obsjd,
            "ra_center": ra_c, "dec_center": dec_c, "corners": corners,
            "footprint_source": src, "ref_key": None,
            "quadrant_ambiguous": claimed > 1, "quadrant_claimed_by": claimed,
            "n_reference": m.get("n_reference"), "gap_days": m.get("gap_days"),
            "science_infobits": m.get("science_infobits"),
            "region_shape": m.get("region_shape") or [None, None],
            "seconds": None, "detections": rows,
        }, code_version=m.get("git_sha"))
        stats["imported"] += 1
        stats["detections"] += len(rows)

    return stats


def _catalog_fingerprint(cat_path):
    """Content hash of a catalog's centroids. Two runs sharing it ARE the same diff."""
    import hashlib
    try:
        t = Table.read(cat_path)
    except (OSError, ValueError):
        return None
    if not len(t) or "x_centroid" not in t.colnames:
        return None
    x = np.asarray(t["x_centroid"], dtype=np.float64)
    y = np.asarray(t["y_centroid"], dtype=np.float64)
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(x).tobytes())
    h.update(np.ascontiguousarray(y).tobytes())
    return f"{len(t)}:{h.hexdigest()[:16]}"


def _dedupe_legacy(manifests):
    """Find run dirs that are the SAME exposure recorded under different quadrants.

    Runs made before the resolve_local ccd/qid fix (da4698f) globbed
    `ztf_<ffd>_<field>_*<product>`, which pins filefracday and field but NOT the
    quadrant - and one ZTF field readout is 64 quadrants sharing a filefracday. So a
    night sweep resolved ONE diff for every quadrant target and wrote N identical
    catalogs. Measured here: 63 of 115 run dirs on disk are such copies, one visit
    appearing as 10 separate "exposures".

    Importing them as independent would overstate surveyed sky by ~2x and double every
    detection count. Their detections are still REAL - ra/dec come from the diff's own
    WCS - so one dir per identical catalog is kept and flagged, rather than discarded.

    Returns ({manifest_path: keep?}, {kept_manifest_path: how_many_dirs_claimed_it}).
    """
    groups = {}
    for mpath in manifests:
        cat = os.path.join(os.path.dirname(mpath), "catalog.ecsv")
        try:
            with open(mpath) as f:
                m = json.load(f)
        except (OSError, ValueError):
            continue
        fp = _catalog_fingerprint(cat)
        if fp is None:
            continue
        key = (int(m["science_filefracday"]), bool(m.get("full_frame")), fp)
        groups.setdefault(key, []).append(mpath)

    keep, claimed = {}, {}
    for key, paths in groups.items():
        paths = sorted(paths)               # deterministic: lowest ccd/qid dir wins
        keep[paths[0]] = True
        claimed[paths[0]] = len(paths)
        for p in paths[1:]:
            keep[p] = False
    return keep, claimed


class _LegacyResult:
    """The three attributes _rows_from_catalog needs, without faking a FieldResult."""

    def __init__(self, run_dir, catalog_path, cutouts_dir):
        self.run_dir = run_dir
        self.catalog_path = catalog_path
        self.cutouts_dir = cutouts_dir


def acquire_lock():
    """One harvester at a time: reset_running() would otherwise steal a live claim."""
    LOCKFILE.parent.mkdir(parents=True, exist_ok=True)
    if LOCKFILE.exists():
        try:
            pid = int(LOCKFILE.read_text().strip())
            os.kill(pid, 0)
            raise RuntimeError(f"another harvester is running (pid {pid}); "
                               f"remove {LOCKFILE} if that is stale")
        except (ValueError, ProcessLookupError, PermissionError):
            pass                                # stale lock, take it
    LOCKFILE.write_text(str(os.getpid()))
    return LOCKFILE


def release_lock():
    try:
        LOCKFILE.unlink()
    except FileNotFoundError:
        pass


def mask_from_branches(names):
    mask = 0
    for n in names:
        if n not in BRANCH_MASKS:
            raise SystemExit(f"unknown branch '{n}'; pick from {list(BRANCH_MASKS)}")
        mask |= BRANCH_MASKS[n]
    return mask or DEFAULT_WANT


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m ztf_data.harvest")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init")

    e = sub.add_parser("enqueue")
    e.add_argument("--date")
    e.add_argument("--local", action="store_true",
                   help="enqueue every CACHED exposure instead of querying IRSA")
    e.add_argument("--min-epochs", type=int, default=1,
                   help="--local only: skip cells with fewer cached exposures")
    e.add_argument("--fields", type=int, nargs="*")
    e.add_argument("--fid", type=int, default=2)
    e.add_argument("--ra", type=float, nargs=2)
    e.add_argument("--dec", type=float, nargs=2)
    e.add_argument("--hours", type=float, nargs=2)
    e.add_argument("--cadence", help="NxM = N cells x M epochs (the linkable shape)")
    e.add_argument("--spread", action="store_true",
                   help="--cadence only: one cell per DISTINCT field, so the cells are "
                        "separated sky rather than tiled quadrants of one field")
    e.add_argument("--max-targets", type=int)
    e.add_argument("--branches", default="motion")
    e.add_argument("--central", action="store_true", help="central-1000 instead of full")

    r = sub.add_parser("run")
    r.add_argument("--workers", type=int, default=1)
    r.add_argument("--batch", type=int, default=32)
    r.add_argument("--max-batches", type=int, default=1)
    r.add_argument("--branches", default="motion")
    r.add_argument("--n-ref", type=int, help="reference frames to stack (default 40)")
    r.add_argument("--offline", action="store_true",
                   help="choose epochs from cached frames; no IRSA, no download")

    ln = sub.add_parser("link")
    ln.add_argument("--date")

    s = sub.add_parser("status")
    s.add_argument("--check", action="store_true")

    rq = sub.add_parser("requeue")
    rq.add_argument("--failed", action="store_true")

    il = sub.add_parser("import-legacy")
    il.add_argument("--limit", type=int)
    il.add_argument("--runs-root")

    a = p.parse_args(argv)
    con = store.init(store.connect())

    if a.cmd == "init":
        print(f"survey.db ready at {store.DB_PATH}")
        return

    if a.cmd == "status":
        print("queue:", store.status_counts(con))
        if a.check:
            for k, v in store.integrity(con).items():
                flag = "" if (v == 0 or v == "ok") else "  <-- PROBLEM"
                print(f"  {k}: {v}{flag}")
        return

    if a.cmd == "import-legacy":
        lock = acquire_lock()
        try:
            s = import_legacy(con, runs_root=a.runs_root, limit=a.limit)
            print(f"imported {s['imported']}/{s['runs']} run dirs, "
                  f"{s['detections']} detections ({s['skipped']} skipped)")
            if s["dup_dirs"]:
                print(f"  {s['dup_dirs']} run dir(s) skipped as copies of an identical "
                      f"catalog (pre-da4698f quadrant collision); {s['ambiguous']} kept "
                      f"exposure(s) flagged quadrant_ambiguous - their detections are "
                      f"real, their (ccdid,qid) label is not")
            if s["bbox"]:
                print(f"  {s['bbox']} footprint(s) from the catalog extent, not a WCS "
                      f"(their diff has been evicted) - flagged in footprint_source")
            print("queue:", store.status_counts(con))
        finally:
            release_lock()
        return

    if a.cmd == "requeue":
        with con:
            n = con.execute("UPDATE targets SET status='pending', attempts=0"
                            " WHERE status IN ('failed','skipped')").rowcount
        print(f"requeued {n} target(s)")
        return

    if a.cmd == "enqueue":
        want = mask_from_branches(a.branches.split(","))
        if a.local:
            n = enqueue_local(con, fid=a.fid, fields=a.fields,
                              full_frame=not a.central, want_mask=want,
                              min_epochs=a.min_epochs)
            print(f"enqueued {n} new target(s); queue: {store.status_counts(con)}")
            return
        if not a.date:
            p.error("enqueue needs --date (or --local to use cached frames)")
        kw = dict(fields=a.fields, fid=a.fid, ra_range=tuple(a.ra) if a.ra else None,
                  dec_range=tuple(a.dec) if a.dec else None,
                  ut_hour_range=tuple(a.hours) if a.hours else None,
                  full_frame=not a.central, want_mask=want)
        if a.cadence:
            n_cells, n_epochs = (int(x) for x in a.cadence.lower().split("x"))
            n = enqueue_cadence(con, a.date, n_cells=n_cells, n_epochs=n_epochs,
                                spread=a.spread, **kw)
        else:
            n = enqueue_night(con, a.date, max_targets=a.max_targets, **kw)
        print(f"enqueued {n} new target(s); queue: {store.status_counts(con)}")
        return

    if a.cmd == "link":
        lock = acquire_lock()
        try:
            n = link_pending_nights(con, dates=[a.date] if a.date else None)
            print(f"{n} track(s) linked")
        finally:
            release_lock()
        return

    if a.cmd == "run":
        want = mask_from_branches(a.branches.split(","))
        acquire_lock()
        try:
            n = store.reset_running(con)
            if n:
                print(f"recovered {n} target(s) stuck in 'running' from a previous kill")
            extra = {}
            if a.offline:
                extra["offline"] = True
            if a.n_ref:
                extra["n_ref"] = a.n_ref
            for b in range(a.max_batches):
                print(f"batch {b + 1}/{a.max_batches}:")
                c = run_batch(con, n_workers=a.workers, batch=a.batch, want_mask=want,
                              **extra)
                print(f"  -> {c}")
                if c["claimed"] == 0:
                    print("  nothing left to claim")
                    break
                # Every target finished and NONE gained a stage: claim() will hand back
                # the same rows next batch and this will spin at full CPU forever,
                # looking busy. Always a want_mask asking for a stage this branch cannot
                # set - stop and name it rather than burn the night.
                if c["done"] and c.get("no_progress", 0) == c["done"]:
                    print(f"  LIVELOCK: all {c['done']} target(s) completed without "
                          f"gaining a stage. want_mask={want} "
                          f"({','.join(store.stage_list(want))}) is asking for a stage "
                          f"this branch never sets. Stopping.")
                    break
            print("queue:", store.status_counts(con))
        finally:
            release_lock()
        return


if __name__ == "__main__":
    main()
