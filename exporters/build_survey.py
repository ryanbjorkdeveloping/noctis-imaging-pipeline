"""Export the survey store to static JSON for the browser. Build-time only.

    survey.db  --(this script, offline)-->  ui/data/survey/*.json  -->  fetch()

The browser never sees SQLite and there is no backend: `serve.sh` is a plain static
file server. "Load what the user is looking at" is done by SHARDING - clicking a
footprint fetches exposures/<id>.json, and filtering inside it is JS over an
already-loaded array. Cross-exposure questions ("every SHORT candidate, by SNR") are
answered by precomputed views/*.json, one SQL query each, rather than by shipping the
database.

HONESTY (mandatory, and enforced by --verify): this pipeline has made NO novel
discoveries. Three categories are kept distinct everywhere and the word "discovery"
appears nowhere except the "NOT a discovery feed" footer:
  recovery    - found blind, but the object IS catalogued. A validation.
  unconfirmed - no catalogue match. Unvetted, and usually an artifact.
  unchecked   - SkyBoT was unreachable. Novelty is UNKNOWN, not novel.

Run:  .venv/bin/python ui/build_survey.py [--verify] [--no-thumbs] [--limit N]
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: F401,E402  LOAD-BEARING: before paths.py reaches ztfquery

from ztf_data import store  # noqa: E402

SCHEMA = 1
OUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "survey")

# Per exposure, export only detections a reader could act on. A full-frame exposure
# holds ~1000 rows and 200 exposures would be ~30 MB of JSON that nothing renders;
# the rest scored below every branch's threshold and their numbers live in the DB.
TOP_N_BY_SNR = 200
VIEW_ROW_CAP = 500

STATEMENT = ("This pipeline has made no novel discoveries. Recoveries are known "
             "objects found blind. Candidates are unconfirmed and mostly artifacts.")


def _f(v, nd=6):
    return None if v is None else round(float(v), nd)


def _corners(row):
    try:
        return json.loads(row["corners_json"]) if row["corners_json"] else None
    except (ValueError, TypeError):
        return None


def _category(d):
    """The reader-facing class of ONE detection. Never 'discovery'.

    Computed here, in Python, once - never re-derived in JS, so the honesty rules
    cannot drift between the map, the list and the drawer.
    """
    if d["track_uid"]:
        return "recovery" if d["honesty_class"] == "recovery" else "candidate"
    if d["streak_route"] == "SHORT_NEA_CANDIDATE":
        return "candidate"
    if d["braai_pass"]:
        return "verdict"
    # A catalogue-matched type IS a classification of a real source, and
    # _exposure_class already counts n_typed as a "verdicts" exposure - without this
    # an exposure could colour as verdicts on the map while every one of its rows
    # said "no classification branch flagged this one".
    if d["pathB_type"] or d["pathA_type"]:
        return "verdict"
    return "none"


def _realness(d):
    """The real/bogus axis, computed in Python beside _category for the same reason.

    THREE states, never two. "A branch never ran" is not a rejection, and the map must
    not draw it as one - the same discipline novelty_status and pathB_matched already
    follow. Returns (realness, by):

      real        a model accepted it: braai passed the stamp, ALeRCE matched a
                  catalogued object, or it linked into a track.
      bogus       a model REJECTED it: braai scored the stamp and it failed, or the
                  DeepStreaks cascade routed the streak stamp BOGUS / COSMIC_RAY.
      unevaluated nothing that answers real-vs-bogus ever looked at it.

    Deliberately NOT on this axis:
      - SHORT_NEA_CANDIDATE. The cascade accepted it AS A STREAK, but per-detection
        SHORT rows are dominated by artifacts until they link, so calling them "real"
        would overstate exactly the claim this project refuses to make. _category
        already carries them as "candidate" and the UI draws that as its own tier.
      - LONG_SATELLITE. A satellite is a real object, just not an asteroid - a
        rejection by the science question, not by a realness model.

    `by` names the branch that decided, so the marker tooltip can say "braai rejected
    this stamp" rather than the unqualified word "bogus".
    """
    if d["track_uid"]:
        return "real", "linked into a track"
    if d["braai_pass"]:
        return "real", "braai passed the stamp"
    if d["pathB_type"] or d["pathA_type"]:
        # Note this can co-exist with a failed braai score. The catalogue match is the
        # stronger evidence (braai's rb is unreliable on saturated stamps), and
        # final_verdict already carries the "braai: low confidence real" caveat.
        return "real", "matched a catalogued object"
    if d["braai_p_real"] is not None and not d["braai_pass"]:
        return "bogus", "braai rejected the stamp"
    if d["streak_route"] in ("BOGUS", "COSMIC_RAY"):
        return "bogus", "the streak cascade rejected it as " + (
            "a cosmic ray" if d["streak_route"] == "COSMIC_RAY" else "not a real streak")
    return "unevaluated", None


def _exposure_class(row, has_recovery):
    if has_recovery:
        return "recovery"
    if row["n_typed"] or row["n_braai_pass"]:
        return "verdicts"
    if row["n_short"]:
        return "candidates"
    return "processed"


def _select_detections(con, eid):
    """streak candidates + braai passers + TYPED sources + track members + top-N SNR.

    The typed clause is not redundant with the top-N. classify.py identifies a source
    by cross-matching a catalogued ALeRCE object, and that answer has nothing to do
    with how bright the source is in THIS exposure - a securely identified variable
    star sitting at rank 400 by SNR is exactly as real as one at rank 4. Before this
    clause existed, 1601 detections carrying a real object identification were
    computed, stored, and then silently dropped at export time, so the UI could not
    show them at all. Every detection the pipeline considers a real, identified
    object now reaches the browser.
    """
    return con.execute(
        "SELECT d.*, m.track_uid, t.honesty_class"
        "  FROM detections d"
        "  LEFT JOIN track_members m ON m.exposure_id=d.exposure_id AND m.row_id=d.row_id"
        "  LEFT JOIN tracks t ON t.track_uid=m.track_uid"
        " WHERE d.exposure_id=:e AND ("
        "        d.streak_route IN ('SHORT_NEA_CANDIDATE','LONG_SATELLITE')"
        "     OR d.braai_pass=1 OR m.track_uid IS NOT NULL"
        "     OR d.pathB_type IS NOT NULL OR d.pathA_type IS NOT NULL"
        "     OR d.row_id IN (SELECT row_id FROM detections WHERE exposure_id=:e"
        "                     ORDER BY snr DESC LIMIT :n))"
        " ORDER BY d.snr DESC", {"e": eid, "n": TOP_N_BY_SNR}).fetchall()


def _det_json(d, thumb_rel, thumb_channels=()):
    realness, realness_by = _realness(d)
    return {
        "det_uid": d["det_uid"], "row_id": d["row_id"],
        "ra": _f(d["ra"]), "dec": _f(d["dec"]),
        "snr": _f(d["snr"], 2), "sign": d["sign"],
        "elongation": _f(d["elongation"], 3), "on_edge": bool(d["on_edge"]),
        "category": _category(d),
        # the real/bogus axis, separate from category's honesty class
        "realness": realness,
        "realness_by": realness_by,
        "streak_route": d["streak_route"], "streak_pa": _f(d["streak_pa"], 2),
        # three literal words, never a bool. NULL = the branch never ran, which is
        # different again from 'unchecked' (it ran and the service was unreachable).
        "novelty_status": d["novelty_status"],
        "nearest_known": d["nearest_known"] or None,
        "nearest_sep_arcsec": _f(d["nearest_sep_arcsec"], 1),
        "braai_p_real": _f(d["braai_p_real"], 4),
        "braai_pass": None if d["braai_p_real"] is None else bool(d["braai_pass"]),
        "braai_stamp_source": d["braai_stamp_source"],
        # same three-state discipline as novelty_status above: NULL = the stationary
        # branch never looked at this detection at all (streak-routed, or outside
        # classify.py's snr/scope cut); false = it DID ask ALeRCE and there was
        # genuinely no catalogued object there (the common, honest outcome for a
        # random field — most background sources were never alerted on); true = a
        # real match, and final_verdict below is populated. Collapsing false into
        # "not run" (as the export did before this field existed) reads as "the
        # pipeline never touched this row", which is false — it was asked and
        # answered "nothing here".
        "pathB_matched": None if d["pathB_matched"] is None else bool(d["pathB_matched"]),
        "type": d["pathB_type"] or d["pathA_type"],
        "type_path": d["type_path"],
        "type_calibrated": None if d["pathA_calibrated"] is None
                           else bool(d["pathA_calibrated"]),
        "final_verdict": d["final_verdict"],
        "track_uid": d["track_uid"],
        "thumb": thumb_rel,
        # Which planes the stamp actually contains, left to right. NOT always three:
        # a streaks_only harvest downloads only the difference, and ZTF's cached
        # reference exists for most cells but not all. The drawer captions itself
        # from this, so it can never claim a channel the image does not hold.
        "thumb_channels": list(thumb_channels),
    }


def _write(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, separators=(",", ":"))
    return os.path.getsize(path)


def build(out_root=OUT_ROOT, limit=None, thumbs=True):
    con = store.connect(readonly=True)
    os.makedirs(out_root, exist_ok=True)
    built = _now()

    recovery_eids = {r[0] for r in con.execute(
        "SELECT DISTINCT m.exposure_id FROM track_members m JOIN tracks t"
        " USING(track_uid) WHERE t.honesty_class='recovery'")}

    exposures = con.execute(
        "SELECT * FROM exposures ORDER BY ut_date, obsjd").fetchall()
    if limit:
        exposures = exposures[:limit]

    index, n_thumbs, n_shown, n_total = [], 0, 0, 0
    # {exposure_id: (channels, {row_id: rel})} — so a view row can carry the
    # same cutout the exposure drill-down does, instead of the drawer going
    # blank the moment a reader arrives from Candidates rather than the map.
    thumb_index = {}
    for e in exposures:
        eid = e["exposure_id"]
        dets = _select_detections(con, eid)
        thumb_map, channels = (_exposure_thumbs(e, dets, out_root)
                               if thumbs else ({}, []))
        if thumb_map:
            thumb_index[eid] = (channels, thumb_map)
        rows = []
        for d in dets:
            trel = thumb_map.get(d["row_id"])
            if trel:
                n_thumbs += 1
            rows.append(_det_json(d, trel, channels if trel else []))
        n_shown += len(rows)
        n_total += e["n_detections"]

        cls = _exposure_class(e, eid in recovery_eids)
        index.append({
            "id": eid, "key": e["target_key"],
            "field": e["field"], "ccdid": e["ccdid"], "qid": e["qid"], "fid": e["fid"],
            "ffd": e["science_filefracday"], "ut_date": e["ut_date"],
            "obsjd": _f(e["obsjd"], 6),
            "ra": _f(e["ra_center"]), "dec": _f(e["dec_center"]),
            "corners": _corners(e),
            "footprint_source": e["footprint_source"],
            # a legacy pre-fix run: the detections are real, the quadrant label is not
            "quadrant_ambiguous": bool(e["quadrant_ambiguous"]),
            "n_det": e["n_detections"], "n_short": e["n_short"], "n_long": e["n_long"],
            "n_braai_pass": e["n_braai_pass"], "n_typed": e["n_typed"],
            "n_track_members": e["n_track_members"],
            "stages": store.stage_list(e["stage_mask"]),
            "class": cls,
        })
        _write(os.path.join(out_root, "exposures", f"{eid}.json"), {
            "id": eid, "key": e["target_key"], "ut_date": e["ut_date"],
            "obsjd": _f(e["obsjd"], 6), "corners": _corners(e),
            "quadrant_ambiguous": bool(e["quadrant_ambiguous"]),
            "footprint_source": e["footprint_source"],
            "stages": store.stage_list(e["stage_mask"]),
            "counts": {"detections": e["n_detections"], "short": e["n_short"],
                       "long": e["n_long"], "braai_pass": e["n_braai_pass"],
                       "typed": e["n_typed"], "track_members": e["n_track_members"]},
            # the page must be able to say "showing 214 of 1052" rather than implying
            # these are all the detections there were
            "truncated": {"total": e["n_detections"], "shown": len(rows)},
            "detections": rows,
        })

    _write(os.path.join(out_root, "index.json"),
           {"schema": SCHEMA, "built": built, "n_exposures": len(index),
            "exposures": index})

    tracks = _build_tracks(con)
    _write(os.path.join(out_root, "tracks.json"),
           {"schema": SCHEMA, "built": built, "tracks": tracks})

    views = _build_views(con, tracks, thumb_index)
    for name, payload in views.items():
        _write(os.path.join(out_root, "views", f"{name}.json"), payload)

    summary = _build_summary(con, tracks, index, n_shown, n_total, built, views)
    _write(os.path.join(out_root, "summary.json"), summary)

    return {"exposures": len(index), "detections_shown": n_shown,
            "detections_total": n_total, "tracks": len(tracks),
            "thumbs": n_thumbs, "views": sorted(views)}


def _build_tracks(con):
    out = []
    for t in con.execute("SELECT * FROM tracks ORDER BY ut_date, track_uid"):
        members = [{
            "det_uid": m["det_uid"], "exposure_id": m["exposure_id"],
            "obsjd": _f(m["obsjd"], 6), "ra": _f(m["ra"]), "dec": _f(m["dec"]),
            "resid_arcsec": _f(m["resid_arcsec"], 2),
        } for m in con.execute(
            "SELECT * FROM track_members WHERE track_uid=? ORDER BY obsjd",
            (t["track_uid"],))]
        out.append({
            "track_uid": t["track_uid"], "ut_date": t["ut_date"],
            "honesty_class": t["honesty_class"],
            "novelty_status": t["novelty_status"],
            "nearest_known": t["nearest_known"] or None,
            "nearest_sep_arcsec": _f(t["nearest_sep_arcsec"], 1),
            "rate_ratio": _f(t["rate_ratio"], 3),
            "n_det": t["n_det"], "n_fields": t["n_fields"],
            "arc_min": _f(t["arc_min"], 2),
            "rate_arcsec_hr": _f(t["rate_arcsec_hr"], 1),
            "motion_pa_deg": _f(t["motion_pa_deg"], 2),
            "rms_arcsec": _f(t["rms_arcsec"], 2),
            "obsjd_mid": _f(t["obsjd_mid"], 6),
            "ra": _f(t["ra"]), "dec": _f(t["dec"]),
            "members": members,
        })
    return out


def _view(rows, note, cap=VIEW_ROW_CAP, total=None):
    """total overrides len(rows) when the caller already capped its own query.

    Without it a view that slices before calling here reports total == shown, so a cap
    is indistinguishable from a complete answer and the page says "the complete view"
    over what is actually the top 500 of ~80,000.
    """
    shown = rows[:cap]
    return {"schema": SCHEMA, "note": note,
            "truncated": {"total": len(rows) if total is None else int(total),
                          "shown": len(shown)}, "rows": shown}


def _dedupe_physical(rows):
    """Collapse rows that are the SAME physical detection seen under two exposures.

    The pre-da4698f quadrant collision means one visit's diff was written out under
    several quadrant labels, so the same source survives as two detections with
    different det_uids - identical ra/dec and identical filefracday, one on a
    quadrant_ambiguous exposure and one on a correctly-resolved one. Both are real;
    they are just the same object counted twice. Measured on the live store: 97
    candidate rows collapse to 87 unique, so a reader counting cards over-counts
    fast-mover candidates by ~11%.

    Position is rounded to 1e-4 deg (0.36") - far below ZTF's 1.0125"/px, so two
    genuinely distinct sources can never merge, while the same source re-measured on a
    sibling quadrant always does. The NON-ambiguous exposure wins, because its
    (ccdid,qid) label is the trustworthy one.
    """
    best = {}
    for r in rows:
        if r.get("ra") is None or r.get("dec") is None:
            best[r["det_uid"]] = r          # nothing to key on; keep as-is
            continue
        key = (round(r["ra"], 4), round(r["dec"], 4), r.get("science_filefracday"))
        cur = best.get(key)
        if cur is None or (cur.get("quadrant_ambiguous") and
                           not r.get("quadrant_ambiguous")):
            best[key] = r
    return list(best.values())


def _build_views(con, tracks, thumb_index=None):
    """Cross-exposure questions a per-exposure shard cannot answer.

    Each is one query and one small file, fetched on demand. Adding a view later is
    one more query here - no schema change, no UI rework.

    thumb_index is {exposure_id: (channels, {row_id: rel})}, built by the exposure
    pass, so a view row carries the same cutout as its per-exposure twin.
    """
    thumb_index = thumb_index or {}
    def det_rows(where, params=(), limit=None):
        # Built by _det_json, the SAME function the per-exposure shards use, so a row
        # opened from a view and the same row opened from the map render an identical
        # drawer. Duplicating the field list here is how the two drifted before: a view
        # row had no `thumb`, so arriving via Candidates showed no cutout at all.
        out = []
        for r in con.execute(
            # the track_members join is what lets _realness see a linked detection; a
            # track member listed in top_snr would otherwise read "unevaluated"
            "SELECT d.*, e.ut_date, e.science_filefracday, e.quadrant_ambiguous,"
            "       m.track_uid, t.honesty_class"
            " FROM detections d JOIN exposures e USING(exposure_id)"
            " LEFT JOIN track_members m"
            "   ON m.exposure_id=d.exposure_id AND m.row_id=d.row_id"
            " LEFT JOIN tracks t ON t.track_uid=m.track_uid"
                f" WHERE {where} ORDER BY d.snr DESC"
                + ("" if limit is None else f" LIMIT {int(limit)}"), params):
            ch, tmap = thumb_index.get(r["exposure_id"], ((), {}))
            trel = tmap.get(r["row_id"])
            row = _det_json(r, trel, ch if trel else ())
            row.update({
                "exposure_id": r["exposure_id"], "ut_date": r["ut_date"],
                "science_filefracday": r["science_filefracday"],
                # quadrant_ambiguous travels WITH the detection: without it a card
                # cannot warn that the exposure's (ccdid,qid) label is untrustworthy,
                # and the caveat only surfaced once a reader drilled all the way in.
                "quadrant_ambiguous": bool(r["quadrant_ambiguous"]),
            })
            out.append(row)
        return out

    views = {}
    # SHORT-routed AND actually novelty-checked. A row whose SkyBoT lookup never
    # completed has UNKNOWN novelty, so listing it here would quietly assert the one
    # thing nobody checked - that it is not already a catalogued object. It goes to
    # views/unchecked.json instead, and the two files stay disjoint (verify asserts it).
    views["candidates"] = _view(
        _dedupe_physical(det_rows("d.streak_route='SHORT_NEA_CANDIDATE'"
                                  " AND COALESCE(d.novelty_status,'') != 'unchecked'")),
        "Unconfirmed fast-mover candidates, one row per physical detection. Per-detection "
        "SHORT rows are dominated by artifacts until several of them link into a track. "
        "NOT discoveries.")
    views["verdicts"] = _view(
        det_rows("d.braai_pass=1"),
        "Stationary classifications of real point sources. This is a type call, not a "
        "detection of anything new.")
    # the ONLY view that caps a survey-wide query, so it is the only one that can claim
    # completeness it does not have. Report the real pre-cap total, not the slice length.
    n_all = con.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
    views["top_snr"] = _view(
        det_rows("1=1", limit=VIEW_ROW_CAP),
        "The brightest detections across the whole survey, any branch.",
        total=n_all)
    # SkyBoT was unreachable for these: novelty is UNKNOWN. Its own file so it can
    # never be counted into candidates.
    views["unchecked"] = _view(
        det_rows("d.novelty_status='unchecked'"),
        "Novelty was NOT checked for these (SkyBoT unreachable). Unknown, not novel.")

    views["recoveries"] = _view(
        [t for t in tracks if t["honesty_class"] == "recovery"],
        "Linked tracks matching a catalogued object. Found blind by the pipeline - "
        "this is a validation of the detector, not a discovery.")
    views["tracks_unconfirmed"] = _view(
        [t for t in tracks if t["honesty_class"] == "unconfirmed"],
        "Linked tracks with no catalogue match. Unvetted and unconfirmed.")

    nights = [{
        "ut_date": r["ut_date"], "n_exposures": r["n"], "n_detections": r["d"],
        "n_short": r["s"], "n_braai_pass": r["b"],
        "n_tracks": con.execute("SELECT COUNT(*) FROM tracks WHERE ut_date=?",
                                (r["ut_date"],)).fetchone()[0],
    } for r in con.execute(
        "SELECT ut_date, COUNT(*) n, SUM(n_detections) d, SUM(n_short) s,"
        " SUM(n_braai_pass) b FROM exposures WHERE ut_date IS NOT NULL"
        " GROUP BY ut_date ORDER BY ut_date")]
    views["nights"] = _view(nights, "Per-night rollup of everything harvested.")

    # PER SKY CELL - the grouping the exposure shards cannot express. A ZTF grid cell
    # (field, ccdid, qid) is ONE patch of sky, and its exposures are repeat visits to
    # it: the same pixels, different times, all reduced against that cell's shared deep
    # reference. That stack of epochs is what "N images aligned on top of each other"
    # means concretely, and every detection in the cell is a difference BETWEEN those
    # epochs. index.json is keyed per exposure and so scatters a cell's visits across
    # N unrelated entries; this reassembles them.
    #
    # n_epochs is the honest count of what was REDUCED, not of what ZTF observed - a
    # cell with 56 visits that night still shows the 10 that were harvested.
    cells = {}
    for r in con.execute(
            "SELECT e.*, "
            "  (SELECT COUNT(*) FROM detections d WHERE d.exposure_id=e.exposure_id"
            "     AND d.streak_route='SHORT_NEA_CANDIDATE') AS n_short_d"
            " FROM exposures e ORDER BY e.field, e.ccdid, e.qid, e.obsjd"):
        key = f"{r['field']:06d}_c{r['ccdid']:02d}_q{r['qid']}_f{r['fid']}"
        c = cells.setdefault(key, {
            "cell_key": key, "field": r["field"], "ccdid": r["ccdid"],
            "qid": r["qid"], "fid": r["fid"], "ra": _f(r["ra_center"]),
            "dec": _f(r["dec_center"]), "corners": json.loads(r["corners_json"] or "null"),
            "ut_dates": [], "epochs": [], "n_detections": 0, "n_short": 0,
            "ref_key": r["ref_key"], "n_reference": r["n_reference"],
            "quadrant_ambiguous": bool(r["quadrant_ambiguous"]),
        })
        c["epochs"].append({
            "exposure_id": r["exposure_id"],
            "science_filefracday": r["science_filefracday"],
            "obsjd": _f(r["obsjd"], 6), "ut_date": r["ut_date"],
            "n_detections": r["n_detections"] or 0,
            "n_short": r["n_short_d"] or 0,
            "stage_mask": r["stage_mask"],
        })
        c["n_detections"] += r["n_detections"] or 0
        c["n_short"] += r["n_short_d"] or 0
        if r["ut_date"] and r["ut_date"] not in c["ut_dates"]:
            c["ut_dates"].append(r["ut_date"])
    for c in cells.values():
        c["n_epochs"] = len(c["epochs"])
        jd = [e["obsjd"] for e in c["epochs"] if e["obsjd"] is not None]
        # hours, because an intra-night cadence measured in days is all zeroes
        c["span_hours"] = round((max(jd) - min(jd)) * 24.0, 3) if len(jd) > 1 else 0.0
    views["cells"] = _view(
        sorted(cells.values(), key=lambda c: (-c["n_epochs"], c["cell_key"])),
        "One row per sky cell: the repeat visits to a single patch that were reduced "
        "against a shared reference. Detections are differences BETWEEN these epochs.")
    return views


def _build_summary(con, tracks, index, n_shown, n_total, built, views):
    # READ the number off the view instead of recomputing it. Two independent queries
    # that must agree will eventually disagree: this one already drifted twice - once
    # by not excluding 'unchecked', then again by not deduplicating the quadrant
    # collision, each time promising rows the view did not contain. One source of truth.
    n_short = views["candidates"]["truncated"]["total"]
    n_short_unchecked = con.execute(
        "SELECT COUNT(*) FROM detections WHERE streak_route='SHORT_NEA_CANDIDATE'"
        " AND novelty_status='unchecked'").fetchone()[0]
    n_unchecked = con.execute(
        "SELECT COUNT(*) FROM detections WHERE novelty_status='unchecked'").fetchone()[0]
    by_class = {}
    for t in tracks:
        by_class[t["honesty_class"]] = by_class.get(t["honesty_class"], 0) + 1
    return {
        "schema": SCHEMA, "built": built,
        "n_exposures": len(index),
        "n_nights": len({e["ut_date"] for e in index if e["ut_date"]}),
        "n_detections_total": n_total,
        "n_detections_shown": n_shown,
        "recoveries": {"count": by_class.get("recovery", 0),
                       "note": "blind-found objects that ARE in the catalogue"},
        "candidates": {"per_detection": n_short,
                       "tracks_unconfirmed": by_class.get("unconfirmed", 0),
                       "note": "unvetted; per-detection SHORT rows are dominated by "
                               "artifacts until linked"},
        "verdicts": {"braai_pass": con.execute(
            "SELECT COUNT(*) FROM detections WHERE braai_pass=1").fetchone()[0],
            "typed": con.execute(
                "SELECT COUNT(*) FROM detections WHERE pathB_type IS NOT NULL"
                " OR pathA_type IS NOT NULL").fetchone()[0]},
        "novelty_unchecked": {"detections": n_unchecked,
                              "short_routed": n_short_unchecked,
                              "tracks": by_class.get("unchecked", 0),
                              "note": "SkyBoT unreachable; novelty is UNKNOWN, "
                                      "not novel"},
        # a literal, machine-checkable field. --verify asserts it.
        "discoveries": 0,
        "statement": STATEMENT,
    }


# ---------------------------------------------------------------- thumbnails


def _stamp_source(exposure):
    """ExposureStamps for one exposure, or None if its difference is unresolvable.

    A `quadrant_ambiguous` legacy run carries a WRONG (ccdid, qid) label, so the strict
    resolve finds nothing; ExposureStamps then identifies the frame by which footprint
    actually CONTAINS the region centre this run recorded from its own diff's WCS (see
    det_stamps.diff_by_footprint). That is a proof, not the relaxed glob that caused
    the original bug - and when it cannot prove a unique frame it returns None, so the
    drawer says "no cutout" rather than showing a different patch of sky.
    """
    from det_stamps import ExposureStamps
    try:
        src = ExposureStamps(exposure["science_filefracday"], exposure["field"],
                             exposure["ccdid"], exposure["qid"],
                             ra=exposure["ra_center"], dec=exposure["dec_center"])
    except Exception:
        return None
    if not src.usable:
        src.close()
        return None
    return src


def _exposure_thumbs(exposure, dets, out_root):
    """Render a stamp per detection for ONE exposure. -> ({row_id: rel}, channels).

    Per EXPOSURE, not per detection, because the expensive part is opening the images
    (a full fpack decompress of the difference) and every detection in the exposure
    cuts from the same three planes. The channel list is likewise a property of the
    exposure, so it is resolved once and handed to every row.

    Resumable: an exposure whose PNGs are all already on disk never opens a FITS file
    at all, so re-running the export after adding one night costs only that night.
    """
    eid = exposure["exposure_id"]
    rel_dir = os.path.join("thumbs", str(eid))
    abs_dir = os.path.join(out_root, rel_dir)
    meta_path = os.path.join(abs_dir, "channels.json")

    wanted = {d["row_id"]: os.path.join(rel_dir, f"{d['row_id']}.png") for d in dets}
    if os.path.exists(meta_path):
        try:
            with open(meta_path) as f:
                channels = json.load(f)["channels"]
            have = {r: rel for r, rel in wanted.items()
                    if os.path.exists(os.path.join(out_root, rel))}
            if len(have) == len(wanted):
                return have, channels
        except (ValueError, KeyError, OSError):
            pass

    src = _stamp_source(exposure)
    if src is None:
        return {}, []

    try:
        import numpy as np  # noqa: F401  (imported for the PIL round-trip below)
        from PIL import Image
    except ImportError:
        src.close()
        return {}, []

    os.makedirs(abs_dir, exist_ok=True)
    out, channels = {}, src.channels
    try:
        for d in dets:
            rel = wanted[d["row_id"]]
            dst = os.path.join(out_root, rel)
            if os.path.exists(dst):
                out[d["row_id"]] = rel
                continue
            if d["ra"] is None or d["dec"] is None:
                continue
            try:
                arr, got = src.stamp_rgb(d["ra"], d["dec"])
            except Exception:
                continue
            if arr is None:
                continue
            if got != channels:
                # a source near the edge of the reference footprint lost a plane; the
                # strip would be a different width than the caption promises
                continue
            Image.fromarray(arr).save(dst, "PNG", optimize=True)
            out[d["row_id"]] = rel
    finally:
        src.close()

    if out:
        _write(meta_path, {"channels": channels, "n": len(out)})
    return out, channels


def _now():
    import datetime
    return datetime.datetime.now().isoformat(timespec="seconds")


# ---------------------------------------------------------------- verify


def verify(out_root=OUT_ROOT):
    """Structural + honesty checks on the exported tree. Returns a list of problems."""
    problems = []

    def load(rel):
        p = os.path.join(out_root, rel)
        if not os.path.exists(p):
            problems.append(f"missing {rel}")
            return None
        with open(p) as f:
            return json.load(f)

    index = load("index.json")
    summary = load("summary.json")
    tracks = load("tracks.json")
    if index is None or summary is None or tracks is None:
        return problems

    seen_uids = set()
    for e in index["exposures"]:
        ex = load(os.path.join("exposures", f"{e['id']}.json"))
        if ex is None:
            continue
        for d in ex["detections"]:
            seen_uids.add(d["det_uid"])
            if d["thumb"] and not os.path.exists(os.path.join(out_root, d["thumb"])):
                problems.append(f"thumb missing on disk: {d['thumb']}")
            if d["novelty_status"] not in (None, "known", "novel", "unchecked"):
                problems.append(f"bad novelty_status {d['novelty_status']}")

    for t in tracks["tracks"]:
        if t["honesty_class"] not in ("recovery", "unconfirmed", "unchecked"):
            problems.append(f"bad honesty_class {t['honesty_class']}")
        if t["novelty_status"] == "unchecked" and t["honesty_class"] != "unchecked":
            problems.append(f"{t['track_uid']}: unchecked novelty read as "
                            f"{t['honesty_class']}")
        for m in t["members"]:
            if m["det_uid"] not in seen_uids:
                problems.append(f"track member not exported: {m['det_uid']}")

    if summary.get("discoveries") != 0:
        problems.append("summary.discoveries must be 0")

    # an 'unchecked' row must not also appear in the candidates view
    vd = os.path.join(out_root, "views")
    if os.path.isdir(vd):
        unchecked = {r["det_uid"] for r in
                     json.load(open(os.path.join(vd, "unchecked.json")))["rows"]}
        cand = {r["det_uid"] for r in
                json.load(open(os.path.join(vd, "candidates.json")))["rows"]}
        overlap = unchecked & cand
        if overlap:
            problems.append(f"{len(overlap)} unchecked row(s) counted as candidates")

    # A view that hit its cap must SAY so. shown == total == cap is the signature of a
    # slice reported as a complete answer, which lets the page claim it is showing
    # everything over what is really the top VIEW_ROW_CAP of tens of thousands.
    for name in sorted(os.listdir(vd)) if os.path.isdir(vd) else []:
        if not name.endswith(".json"):
            continue
        v = json.load(open(os.path.join(vd, name)))
        t = v.get("truncated", {})
        if t.get("shown") == VIEW_ROW_CAP and t.get("total") == VIEW_ROW_CAP:
            problems.append(
                f"views/{name}: shown==total=={VIEW_ROW_CAP} (the cap) - a capped view "
                f"is claiming to be complete; pass the real pre-cap total to _view()")

    # every candidate must be ONE physical detection: same sky position at the same
    # exposure time under two quadrant labels is the pre-da4698f collision, counted twice.
    if os.path.exists(os.path.join(vd, "candidates.json")):
        seen, dup = set(), 0
        for r in json.load(open(os.path.join(vd, "candidates.json")))["rows"]:
            if r.get("ra") is None:
                continue
            k = (round(r["ra"], 4), round(r["dec"], 4), r.get("science_filefracday"))
            dup += k in seen
            seen.add(k)
        if dup:
            problems.append(f"candidates.json: {dup} duplicate physical detection(s) "
                            f"(same ra/dec + filefracday under two exposures)")

    # "discovery" may appear ONLY in a NEGATED claim ("not a discovery", "no novel
    # discoveries") or as the literal zero counter. Any other use would be the page
    # asserting a discovery, which this pipeline has never made.
    import re
    negation = re.compile(r"\b(no|not|never|zero|without)\b", re.I)
    for root, _, files in os.walk(out_root):
        for fn in files:
            if not fn.endswith(".json"):
                continue
            txt = open(os.path.join(root, fn)).read()
            for m in re.finditer(r"discover\w*", txt, re.I):
                if txt[m.start():m.end() + 3] == 'discoveries":0':
                    continue                       # the machine-checkable counter
                before = txt[max(0, m.start() - 60):m.start()]
                if negation.search(before):
                    continue                       # an explicit denial, which is fine
                ctx = txt[max(0, m.start() - 40):m.end() + 40]
                problems.append(f"{fn}: unguarded '{m.group()}' in: ...{ctx}...")
    return problems


def main():
    p = argparse.ArgumentParser()
    # --verify BUILDS then checks, which is how everyone reads it. It used to SKIP the
    # build, so `build_survey.py --verify` in a pipeline script cheerfully validated a
    # stale export and printed "verify: OK" while exporting nothing - the failure looks
    # exactly like success. --verify-only is the (rarer) check-what-is-on-disk case.
    p.add_argument("--verify", action="store_true",
                   help="build, then check the result (recommended)")
    p.add_argument("--verify-only", action="store_true",
                   help="check the EXISTING export without rebuilding it")
    p.add_argument("--no-thumbs", action="store_true")
    p.add_argument("--limit", type=int)
    p.add_argument("--out", default=OUT_ROOT)
    a = p.parse_args()

    if not a.verify_only:
        stats = build(out_root=a.out, limit=a.limit, thumbs=not a.no_thumbs)
        print(f"exported {stats['exposures']} exposures, "
              f"{stats['detections_shown']}/{stats['detections_total']} detections, "
              f"{stats['tracks']} tracks, {stats['thumbs']} thumbs")
        print("views:", ", ".join(stats["views"]))

    if not (a.verify or a.verify_only):
        return 0
    problems = verify(out_root=a.out)
    if problems:
        print(f"\nVERIFY FAILED ({len(problems)}):")
        for q in problems[:20]:
            print("  -", q)
        return 1
    print("verify: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
