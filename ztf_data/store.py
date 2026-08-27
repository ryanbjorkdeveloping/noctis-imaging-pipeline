"""The survey results store: one SQLite file indexing every harvested exposure.

THE only module that knows the schema, exactly as paths.py is the only module that
knows the directory layout. Everything else calls functions here.

Why SQLite and not more flat files: before this, results lived in one directory per
exposure plus a single global sweep_candidates.ecsv that every later sweep CLOBBERED.
There was no way to ask "what have I processed?", no dedupe, and no resume - run_field
re-queries IRSA and re-detects on every call. A harvester that runs for hours across
hundreds of targets needs all three, and needs to survive being killed.

Why SQLite and not a server: it is stdlib, it is one file that sits next to the data it
indexes, and it is READ ONCE at build time by ui/build_survey.py. The browser never sees
it - it gets static JSON exported from it. No backend exists anywhere in this design.

CONCURRENCY RULE (load-bearing): the harvester's pool workers NEVER open this database.
They produce files in their own run dir and return plain dicts; the PARENT process owns
the only connection and commits one transaction per completed target. A DB write is
microseconds against a ~35 s worker, so multi-process writes would buy nothing and cost
busy-retry logic under spawn.
"""

import datetime
import json
import os
import sqlite3
import subprocess

import config  # noqa: F401  LOAD-BEARING: before paths.py reaches ztfquery.io

from ztf_data.paths import OWN_ROOT

SCHEMA_VERSION = 1
DB_PATH = OWN_ROOT / "survey.db"

# A target that keeps failing must not burn the queue forever. Retryable failures
# (an IRSA outage) deliberately do NOT count against this - see record_failure.
MAX_ATTEMPTS = 3

# ---------------------------------------------------------------- stage bitfield
# What has actually been computed for a target. The harvester asks for a want_mask and
# skips any target whose stage_mask already satisfies it, so adding a branch later
# re-runs ONLY the missing stages instead of redoing detection.
STAGE_DETECT = 1 << 0   # run_field: catalog written
STAGE_STAMPS = 1 << 1   # (3,63,63) cutouts written
STAGE_STREAK = 1 << 2   # DeepStreaks routes + streak_pa + obsjd
STAGE_NOVELTY = 1 << 3  # SkyBoT per-detection (may be partial -> 'unchecked' rows)
STAGE_BRAAI = 1 << 4
STAGE_TYPE_B = 1 << 5   # ALeRCE type
STAGE_TYPE_A = 1 << 6   # local TF1 stamp CNN
STAGE_VERDICT = 1 << 7  # merge_verdict applied

STAGE_NAMES = {
    STAGE_DETECT: "detect", STAGE_STAMPS: "stamps", STAGE_STREAK: "streak",
    STAGE_NOVELTY: "novelty", STAGE_BRAAI: "braai", STAGE_TYPE_B: "typeB",
    STAGE_TYPE_A: "typeA", STAGE_VERDICT: "verdict",
}

# the branch presets the CLI exposes. motion implies streaks_only; stationary forces
# stamps (and therefore the reference download + template stack).
# A branch mask must list only stages that branch can actually DELIVER. claim() re-offers
# any target whose stage_mask does not cover want_mask, so a mask demanding a stage the
# branch structurally never sets is a LIVELOCK: the target is claimed, fully re-run,
# re-ingested identically, and offered again on the next batch, forever.
#
# That is exactly what 'motion' did by asking for STAGE_VERDICT. merge_verdict reads
# braai_pass, so _harvest_one only sets STAGE_VERDICT when STAGE_BRAAI ran - a
# motion-only run never sets it, by design, because a verdict without braai would label
# every detection 'bogus' when the branch simply never executed. Observed 2026-08-12:
# 124 completions, exposures stuck at 75, the same ~35 targets cycling for an hour at
# full CPU with nothing to show for it and no error anywhere.
#
# CORRECTION (found live, 2026-08-17): the paragraph above assumed 'stationary' runs
# braai and can honestly produce STAGE_VERDICT. It cannot - _harvest_one (read in full)
# never sets STAGE_BRAAI on ANY code path; that bit is set ONLY by ztf_data.classify's
# separate ALeRCE-enrichment pass, entirely outside `harvest run`. So 'stationary' asking
# for STAGE_BRAAI|STAGE_VERDICT was the identical livelock this comment already diagnosed
# for 'motion', just moved rather than fixed: observed live at attempts=9-12 on targets
# with 1000+ real detections already ingested, repeatedly reclaimed and flipped to
# 'skipped' once MAX_ATTEMPTS was crossed - real data mislabeled by an unsatisfiable mask,
# not a processing failure. 'stationary' only legitimately DELIVERS the catalog + the
# local (3,63,63) stamp triplets; braai/type/verdict are `classify`'s job, requested via
# the separate "type" mask below, never through `harvest run`.
BRANCH_MASKS = {
    "motion": STAGE_DETECT | STAGE_STREAK | STAGE_NOVELTY,
    "stationary": STAGE_DETECT | STAGE_STAMPS,
    "type": STAGE_TYPE_B,
    "pathA": STAGE_TYPE_A,
}


def stage_list(mask):
    """Bitfield -> the names, for manifests and exported JSON."""
    return [name for bit, name in sorted(STAGE_NAMES.items()) if mask & bit]


SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

-- One row per (grid cell, science epoch, detection region). THE resume key.
CREATE TABLE IF NOT EXISTS targets (
  target_id   INTEGER PRIMARY KEY,
  target_key  TEXT    NOT NULL UNIQUE,
  field       INTEGER NOT NULL,
  ccdid       INTEGER NOT NULL,
  qid         INTEGER NOT NULL,
  fid         INTEGER NOT NULL,
  science_filefracday INTEGER NOT NULL,
  full_frame  INTEGER NOT NULL,
  -- enqueue-time metadata, free from night.enumerate_night (no extra query)
  ut_date     TEXT,          -- the LINKING GROUP key: link.py is intra-night ONLY
  obsjd       REAL,
  ra          REAL,
  dec         REAL,
  seeing      REAL,
  maglimit    REAL,
  infobits    INTEGER,
  status      TEXT NOT NULL DEFAULT 'pending'
              CHECK (status IN ('pending','running','done','failed','skipped')),
  stage_mask  INTEGER NOT NULL DEFAULT 0,
  want_mask   INTEGER NOT NULL DEFAULT 0,
  attempts    INTEGER NOT NULL DEFAULT 0,
  last_error  TEXT,
  code_version TEXT,
  enqueued_at TEXT,
  started_at  TEXT,
  finished_at TEXT
);
CREATE INDEX IF NOT EXISTS ix_targets_status ON targets(status, target_id);
CREATE INDEX IF NOT EXISTS ix_targets_cell   ON targets(field, ccdid, qid, fid);
CREATE INDEX IF NOT EXISTS ix_targets_night  ON targets(ut_date);

-- A completed target, 1:1 with a run dir. Separate from `targets` so the queue stays
-- small and hot, and so results are never lost by a queue reset.
CREATE TABLE IF NOT EXISTS exposures (
  exposure_id INTEGER PRIMARY KEY,
  target_id   INTEGER NOT NULL UNIQUE REFERENCES targets(target_id),
  target_key  TEXT    NOT NULL UNIQUE,
  field INTEGER, ccdid INTEGER, qid INTEGER, fid INTEGER,
  science_filefracday INTEGER,
  full_frame INTEGER,
  ut_date TEXT, obsjd REAL,
  run_dir     TEXT NOT NULL,
  ref_key     TEXT,
  n_reference INTEGER, gap_days REAL, science_infobits INTEGER,
  region_h INTEGER, region_w INTEGER,
  corners_json TEXT,                 -- [[ra,dec] x4], CCW from pixel (0,0)
  ra_center REAL, dec_center REAL,
  -- 'wcs'  = read off the diff's WCS, exact
  -- 'catalog_bbox' = the detections' ra/dec extent, because the diff was evicted.
  --   Close but NOT the region outline; the map must not present it as surveyed area.
  footprint_source TEXT,
  -- Legacy only. Runs made before the resolve_local ccd/qid fix (da4698f) resolved one
  -- visit's diff for EVERY quadrant of that field, so N run dirs hold one identical
  -- catalog. Import keeps one and records how many dirs claimed it. The detections'
  -- ra/dec are still correct (they come from the diff's own WCS); it is the (ccdid,qid)
  -- LABEL that cannot be trusted, so this row is one real exposure of unknown quadrant.
  quadrant_ambiguous INTEGER DEFAULT 0,
  quadrant_claimed_by INTEGER DEFAULT 1,
  -- denormalized on purpose: the UI index is ONE query, no GROUP BY joins
  n_detections        INTEGER DEFAULT 0,
  n_streak_candidates INTEGER DEFAULT 0,
  n_short             INTEGER DEFAULT 0,
  n_long              INTEGER DEFAULT 0,
  n_braai_scored      INTEGER DEFAULT 0,
  n_braai_pass        INTEGER DEFAULT 0,
  n_typed             INTEGER DEFAULT 0,
  n_track_members     INTEGER DEFAULT 0,
  stage_mask INTEGER NOT NULL DEFAULT 0,
  seconds     REAL,
  code_version TEXT,
  created_at  TEXT
);
CREATE INDEX IF NOT EXISTS ix_exp_night ON exposures(ut_date, obsjd);
CREATE INDEX IF NOT EXISTS ix_exp_cell  ON exposures(field, ccdid, qid, fid);

-- PK is (exposure_id, row_id): row_id ('src_p1_000') RESTARTS PER RUN and is NOT
-- globally unique. det_uid is the derived global key used by the UI, thumbnails,
-- and every cross-run join.
CREATE TABLE IF NOT EXISTS detections (
  exposure_id INTEGER NOT NULL REFERENCES exposures(exposure_id) ON DELETE CASCADE,
  row_id      TEXT    NOT NULL,
  det_uid     TEXT    NOT NULL UNIQUE,
  ra REAL, dec REAL,
  x_centroid REAL, y_centroid REAL,
  snr REAL, segment_flux REAL,
  elongation REAL, orientation REAL,
  sign INTEGER, on_edge INTEGER,
  -- motion branch
  streak_route TEXT,
  rb_pass INTEGER, kd_pass INTEGER, sl_short INTEGER,
  streak_pa REAL,
  -- per-detection novelty. THREE-STATE, never collapsed to a bool: an outage
  -- ('unchecked') must never be readable as a discovery ('novel').
  novelty_status TEXT
    CHECK (novelty_status IN ('known','novel','unchecked') OR novelty_status IS NULL),
  nearest_known TEXT, nearest_sep_arcsec REAL, nearest_motion REAL,
  -- stationary branch
  braai_p_real REAL, braai_pass INTEGER,
  braai_stamp_source TEXT,           -- 'native' | 'cutout'
  braai_oid TEXT,
  pathB_matched INTEGER, pathB_oid TEXT, pathB_type TEXT,
  pathB_top_class TEXT, pathB_prob REAL,
  pathA_type TEXT, pathA_calibrated INTEGER,
  final_verdict TEXT,
  type_path TEXT,                    -- 'alerce' | 'localCNN' | ''
  stamp_rel   TEXT,                  -- 'cutouts/src_p1_000.npy', relative to run_dir
  thumb_rel   TEXT,
  PRIMARY KEY (exposure_id, row_id)
);
CREATE INDEX IF NOT EXISTS ix_det_route   ON detections(streak_route)
  WHERE streak_route IS NOT NULL;
CREATE INDEX IF NOT EXISTS ix_det_verdict ON detections(final_verdict)
  WHERE final_verdict IS NOT NULL;
CREATE INDEX IF NOT EXISTS ix_det_snr     ON detections(exposure_id, snr DESC);
CREATE INDEX IF NOT EXISTS ix_det_radec   ON detections(dec, ra);

CREATE TABLE IF NOT EXISTS tracks (
  track_uid   TEXT PRIMARY KEY,
  ut_date     TEXT NOT NULL,          -- linking group; NEVER cross-night
  link_run    TEXT NOT NULL,
  n_det INTEGER, n_fields INTEGER, arc_min REAL,
  rate_arcsec_hr REAL, motion_pa_deg REAL, rms_arcsec REAL,
  obsjd_mid REAL, ra REAL, dec REAL,
  novelty_status TEXT CHECK (novelty_status IN ('known','novel','unchecked')),
  nearest_known TEXT, nearest_sep_arcsec REAL, nearest_motion REAL, rate_ratio REAL,
  -- the honesty classifier, computed ONCE here and never re-derived in JS:
  --   'recovery'    = known         (blind-found but catalogued -> a validation)
  --   'unconfirmed' = novel         (NOT a discovery: unvetted, usually artifacts)
  --   'unchecked'   = SkyBoT unreachable
  honesty_class TEXT NOT NULL
    CHECK (honesty_class IN ('recovery','unconfirmed','unchecked')),
  created_at TEXT
);
CREATE INDEX IF NOT EXISTS ix_tracks_night ON tracks(ut_date);

CREATE TABLE IF NOT EXISTS track_members (
  track_uid   TEXT NOT NULL REFERENCES tracks(track_uid) ON DELETE CASCADE,
  exposure_id INTEGER NOT NULL,
  row_id      TEXT NOT NULL,
  det_uid     TEXT NOT NULL,
  obsjd REAL, ra REAL, dec REAL,
  resid_arcsec REAL,
  PRIMARY KEY (track_uid, det_uid)
);
CREATE INDEX IF NOT EXISTS ix_tm_det ON track_members(exposure_id, row_id);

-- Kills the repeat-ALeRCE-call problem: the same static source is otherwise
-- re-resolved once per epoch of a grid cell (10x waste on a 10-epoch cell).
CREATE TABLE IF NOT EXISTS alerce_cache (
  coord_key   TEXT PRIMARY KEY,       -- f"{ra:.4f}_{dec:.4f}" ~ 0.36" bins at dec 0
  oid         TEXT,                   -- NULL with resolved=1 means a REAL "no match"
  resolved    INTEGER NOT NULL,       -- 0 = the call FAILED; never read as "no match"
  stamp_ok    INTEGER,
  top_class   TEXT, top_prob REAL,
  fetched_at  TEXT
);

CREATE TABLE IF NOT EXISTS templates (
  ref_key TEXT PRIMARY KEY,
  field INTEGER, ccdid INTEGER, qid INTEGER, fid INTEGER,
  n_ref INTEGER, ref_ffds_json TEXT,
  path TEXT, bytes INTEGER, built_at TEXT, seconds REAL
);
"""


# ---------------------------------------------------------------- connection


def connect(path=DB_PATH, readonly=False, check_same_thread=True):
    """Open the store. WAL so a kill -9 mid-write is survivable.

    check_same_thread=False is for a caller that serializes every access itself (a
    lock around each query/commit) and wants to share ONE connection across worker
    threads — e.g. classify.py's --workers path, where the actual work (ALeRCE HTTP
    calls) is the part worth parallelizing and the DB touches are brief and locked.
    WAL already permits concurrent readers; this only lifts sqlite3's OWN same-thread
    guard, it does not remove the need for the caller's lock.
    """
    path = str(path)
    if readonly:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30.0,
                              check_same_thread=check_same_thread)
    else:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        con = sqlite3.connect(path, timeout=30.0, check_same_thread=check_same_thread)
    con.row_factory = sqlite3.Row
    if not readonly:
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA synchronous=NORMAL")
    con.execute("PRAGMA foreign_keys=ON")
    return con


def init(con):
    """Create the schema (idempotent) and stamp the version."""
    con.executescript(SCHEMA)
    with con:
        con.execute("INSERT OR IGNORE INTO meta(key,value) VALUES('schema_version',?)",
                    (str(SCHEMA_VERSION),))
        con.execute("INSERT OR IGNORE INTO meta(key,value) VALUES('created',?)", (_now(),))
    found = int(con.execute("SELECT value FROM meta WHERE key='schema_version'")
                .fetchone()[0])
    if found != SCHEMA_VERSION:
        raise RuntimeError(
            f"survey.db is schema v{found}, this code speaks v{SCHEMA_VERSION}"
        )
    return con


def _now():
    return datetime.datetime.now().isoformat(timespec="seconds")


def git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


# ---------------------------------------------------------------- keys


def target_key(field, ccdid, qid, fid, science_filefracday, full_frame=False):
    """The resume key. Mirrors paths.run_dir's identity, including the REGION.

    The region belongs in the key for the same reason it is in the run dir: the same
    grid+epoch detected full-frame (~950 rows) and central-1000 (114) are different
    catalogs, and collapsing them would make one silently overwrite the other.
    """
    region = "F" if full_frame else "C"
    return f"{int(field):06d}_{int(ccdid):02d}_{int(qid)}_{int(fid)}_{int(science_filefracday)}_{region}"


def det_uid(tkey, row_id):
    """Globally unique detection id. row_id alone is NOT unique across exposures."""
    return f"{tkey}:{row_id}"


# ---------------------------------------------------------------- queue


def enqueue(con, targets, want_mask, full_frame=True):
    """INSERT OR IGNORE targets. Idempotent: re-enqueueing a night adds nothing.

    `targets` are dicts from night.night_targets (the 5 run_field keys), optionally
    carrying ut_date/obsjd/ra/dec/seeing/maglimit/infobits from the enumerator table.
    Returns the number of rows NEWLY added.
    """
    before = con.execute("SELECT COUNT(*) FROM targets").fetchone()[0]
    rows = []
    for t in targets:
        ff = int(t.get("full_frame", full_frame))
        rows.append((
            target_key(t["field"], t["ccdid"], t["qid"], t["fid"],
                       t["science_filefracday"], bool(ff)),
            int(t["field"]), int(t["ccdid"]), int(t["qid"]), int(t["fid"]),
            int(t["science_filefracday"]), ff,
            t.get("ut_date"), _f(t.get("obsjd")), _f(t.get("ra")), _f(t.get("dec")),
            _f(t.get("seeing")), _f(t.get("maglimit")), _i(t.get("infobits")),
            int(want_mask), _now(),
        ))
    with con:
        con.executemany(
            "INSERT OR IGNORE INTO targets "
            "(target_key,field,ccdid,qid,fid,science_filefracday,full_frame,"
            " ut_date,obsjd,ra,dec,seeing,maglimit,infobits,want_mask,enqueued_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
        # widen the ask on targets that already exist: this is what makes
        # "harvest motion now, add stationary later" re-run only the missing stages.
        con.executemany(
            "UPDATE targets SET want_mask = want_mask | ? WHERE target_key = ?",
            [(int(want_mask), r[0]) for r in rows])
    return con.execute("SELECT COUNT(*) FROM targets").fetchone()[0] - before


def reset_running(con):
    """Startup recovery: a kill -9 leaves rows claimed. Safe because exactly one
    harvester process may run at a time (enforced by the lockfile in harvest.py)."""
    with con:
        cur = con.execute(
            "UPDATE targets SET status='pending' WHERE status='running'")
    return cur.rowcount


def claim(con, want_mask, limit=32, code_version=None):
    """Select eligible targets, mark them running, return them as plain dicts.

    Eligible = never done, previously failed (under the attempt cap), or done but
    NOT yet satisfying want_mask. A target whose stage_mask already covers want_mask
    is skipped WITHOUT an IRSA round-trip - that is the whole point of the store.

    quadrant_ambiguous exposures are NEVER re-claimed. They come from run dirs written
    before the resolve_local ccdid/qid fix (da4698f), where one visit's diff was
    recorded under up to 10 quadrant labels. Their DETECTIONS are real - ra/dec comes
    from the diff's own WCS - but the (ccdid,qid) LABEL is untrustworthy, and it is the
    label that resolves a file. So re-running them cannot succeed: it raises
    EpochSelectionError, burns an attempt, and flips an exposure that already holds
    good ingested data to 'failed'. Filling their missing stages needs a re-download
    under the correct quadrant, not a retry of a known-bad key.
    """
    rows = con.execute(
        "SELECT t.* FROM targets t "
        "  LEFT JOIN exposures e ON e.target_key = t.target_key "
        " WHERE COALESCE(e.quadrant_ambiguous, 0) = 0 "
        "   AND ((t.status IN ('pending','failed') AND t.attempts < ?) "
        "    OR (t.status='done' AND (t.stage_mask & ?) != ?)) "
        " ORDER BY t.ut_date IS NULL, t.ut_date, t.obsjd, t.target_id "
        " LIMIT ?", (MAX_ATTEMPTS, int(want_mask), int(want_mask), int(limit))
    ).fetchall()
    keys = [r["target_key"] for r in rows]
    if keys:
        with con:
            con.executemany(
                "UPDATE targets SET status='running', started_at=?, attempts=attempts+1,"
                " code_version=? WHERE target_key=?",
                [(_now(), code_version, k) for k in keys])
    return [dict(r) for r in rows]


def record_failure(con, tkey, error, retryable=False):
    """Mark a target failed. A RETRYABLE failure (IRSA outage) does not consume an
    attempt - otherwise a ten-minute blip permanently skips every target it touched."""
    with con:
        if retryable:
            con.execute(
                "UPDATE targets SET status='pending', attempts=MAX(attempts-1,0), "
                "last_error=?, finished_at=? WHERE target_key=?",
                (str(error)[:4000], _now(), tkey))
        else:
            con.execute(
                "UPDATE targets SET status=CASE WHEN attempts >= ? THEN 'skipped' "
                "ELSE 'failed' END, last_error=?, finished_at=? WHERE target_key=?",
                (MAX_ATTEMPTS, str(error)[:4000], _now(), tkey))


# ---------------------------------------------------------------- ingest


def ingest(con, res, code_version=None):
    """One completed target -> exposures + detections + counts, atomically.

    `res` is _harvest_one's plain dict. A kill inside this transaction is rolled back
    by WAL; a kill between targets loses at most that target's DB row, and its run dir
    is already on disk (run_field is deterministic, so a re-run reproduces it).
    """
    tkey = res["target_key"]
    trow = con.execute("SELECT * FROM targets WHERE target_key=?", (tkey,)).fetchone()
    if trow is None:
        raise KeyError(f"ingest for an unknown target: {tkey}")

    with con:
        con.execute(
            "INSERT INTO exposures (target_id,target_key,field,ccdid,qid,fid,"
            " science_filefracday,full_frame,ut_date,obsjd,run_dir,ref_key,n_reference,"
            " gap_days,science_infobits,region_h,region_w,corners_json,ra_center,"
            " dec_center,footprint_source,quadrant_ambiguous,quadrant_claimed_by,"
            " stage_mask,seconds,code_version,created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(target_key) DO UPDATE SET "
            " run_dir=excluded.run_dir, ref_key=excluded.ref_key,"
            " obsjd=excluded.obsjd, corners_json=excluded.corners_json,"
            " ra_center=excluded.ra_center, dec_center=excluded.dec_center,"
            " footprint_source=excluded.footprint_source,"
            " region_h=excluded.region_h, region_w=excluded.region_w,"
            " stage_mask=exposures.stage_mask | excluded.stage_mask,"
            " seconds=excluded.seconds, code_version=excluded.code_version",
            (trow["target_id"], tkey, trow["field"], trow["ccdid"], trow["qid"],
             trow["fid"], trow["science_filefracday"], trow["full_frame"],
             trow["ut_date"], _f(res.get("obsjd")) or trow["obsjd"], res.get("run_dir"),
             res.get("ref_key"), _i(res.get("n_reference")), _f(res.get("gap_days")),
             _i(res.get("science_infobits")),
             _i((res.get("region_shape") or [None, None])[0]),
             _i((res.get("region_shape") or [None, None])[1]),
             json.dumps(res.get("corners")) if res.get("corners") else None,
             _f(res.get("ra_center")), _f(res.get("dec_center")),
             res.get("footprint_source", "wcs" if res.get("corners") else None),
             int(bool(res.get("quadrant_ambiguous", 0))),
             int(res.get("quadrant_claimed_by", 1) or 1),
             int(res.get("stage_mask", 0)), _f(res.get("seconds")), code_version, _now()))

        eid = con.execute("SELECT exposure_id FROM exposures WHERE target_key=?",
                          (tkey,)).fetchone()[0]
        dets = res.get("detections") or []
        if dets:
            con.executemany(
                "INSERT INTO detections (exposure_id,row_id,det_uid,ra,dec,x_centroid,"
                " y_centroid,snr,segment_flux,elongation,orientation,sign,on_edge,"
                " streak_route,rb_pass,kd_pass,sl_short,streak_pa,novelty_status,"
                " nearest_known,nearest_sep_arcsec,nearest_motion,braai_p_real,"
                " braai_pass,braai_stamp_source,braai_oid,pathB_matched,pathB_oid,"
                " pathB_type,pathB_top_class,pathB_prob,pathA_type,pathA_calibrated,"
                " final_verdict,type_path,stamp_rel) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,"
                "?,?,?,?,?) "
                "ON CONFLICT(exposure_id,row_id) DO UPDATE SET " + _DET_UPSERT_SET,
                [_det_row(eid, tkey, d) for d in dets])
        _update_counts(con, eid)
        con.execute(
            "UPDATE targets SET status='done', stage_mask=stage_mask|?, "
            "last_error=NULL, code_version=?, finished_at=? WHERE target_key=?",
            (int(res.get("stage_mask", 0)), code_version, _now(), tkey))
    return eid


_DET_COLUMNS = [
    "ra", "dec", "x_centroid", "y_centroid", "snr", "segment_flux", "elongation",
    "orientation", "sign", "on_edge", "streak_route", "rb_pass", "kd_pass", "sl_short",
    "streak_pa", "novelty_status", "nearest_known", "nearest_sep_arcsec",
    "nearest_motion", "braai_p_real", "braai_pass", "braai_stamp_source", "braai_oid",
    "pathB_matched", "pathB_oid", "pathB_type", "pathB_top_class", "pathB_prob",
    "pathA_type", "pathA_calibrated", "final_verdict", "type_path", "stamp_rel",
]
# On re-ingest, COALESCE keeps an earlier branch's answer whenever THIS pass did not
# compute that branch. Load-bearing for the wide-then-deep flow the whole design rests
# on: a motion-only result dict carries no braai keys, so a plain `excluded.col` would
# overwrite an existing braai score with NULL and silently destroy earlier work.
# The tradeoff is deliberate and stated: a value can be added or corrected, never
# nulled back out by a later pass. Re-detection changes (x_centroid, snr, ...) still
# land, because those columns are always present in every result.
_DET_UPSERT_SET = ",".join(
    f"{c}=COALESCE(excluded.{c}, detections.{c})" for c in _DET_COLUMNS
)


def _det_row(eid, tkey, d):
    vals = [eid, d["row_id"], det_uid(tkey, d["row_id"])]
    for c in _DET_COLUMNS:
        v = d.get(c)
        if isinstance(v, bool):
            v = int(v)
        vals.append(v)
    return tuple(vals)


def _update_counts(con, eid):
    """Recompute the denormalized per-exposure counters the UI index reads."""
    con.execute(
        "UPDATE exposures SET "
        " n_detections=(SELECT COUNT(*) FROM detections WHERE exposure_id=:e),"
        " n_streak_candidates=(SELECT COUNT(*) FROM detections WHERE exposure_id=:e"
        "   AND streak_route IN ('SHORT_NEA_CANDIDATE','LONG_SATELLITE')),"
        " n_short=(SELECT COUNT(*) FROM detections WHERE exposure_id=:e"
        "   AND streak_route='SHORT_NEA_CANDIDATE'),"
        " n_long=(SELECT COUNT(*) FROM detections WHERE exposure_id=:e"
        "   AND streak_route='LONG_SATELLITE'),"
        " n_braai_scored=(SELECT COUNT(*) FROM detections WHERE exposure_id=:e"
        "   AND braai_p_real IS NOT NULL),"
        " n_braai_pass=(SELECT COUNT(*) FROM detections WHERE exposure_id=:e"
        "   AND braai_pass=1),"
        " n_typed=(SELECT COUNT(*) FROM detections WHERE exposure_id=:e"
        "   AND (pathB_type IS NOT NULL OR pathA_type IS NOT NULL)),"
        " n_track_members=(SELECT COUNT(*) FROM track_members m WHERE m.exposure_id=:e)"
        " WHERE exposure_id=:e", {"e": eid})


# ---------------------------------------------------------------- tracks


def honesty_class(novelty_status):
    """The ONE place the three public categories are named. Never 'discovery'.

    recovery    - found blind by the pipeline, but the object IS catalogued.
    unconfirmed - no catalogue match. NOT a discovery: unvetted and usually an artifact.
    unchecked   - SkyBoT could not be reached, so novelty is UNKNOWN.
    """
    return {"known": "recovery", "novel": "unconfirmed"}.get(
        novelty_status, "unchecked")


def upsert_track(con, track, members, ut_date, link_run):
    """One linked track + its member detections. Members must share ut_date."""
    status = track.get("novelty_status") or "unchecked"
    with con:
        con.execute(
            "INSERT INTO tracks (track_uid,ut_date,link_run,n_det,n_fields,arc_min,"
            " rate_arcsec_hr,motion_pa_deg,rms_arcsec,obsjd_mid,ra,dec,novelty_status,"
            " nearest_known,nearest_sep_arcsec,nearest_motion,rate_ratio,honesty_class,"
            " created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(track_uid) DO UPDATE SET "
            " n_det=excluded.n_det, rate_arcsec_hr=excluded.rate_arcsec_hr,"
            " motion_pa_deg=excluded.motion_pa_deg, rms_arcsec=excluded.rms_arcsec,"
            " novelty_status=excluded.novelty_status,"
            " honesty_class=excluded.honesty_class",
            (track["track_uid"], ut_date, link_run, _i(track.get("n_det")),
             _i(track.get("n_fields")), _f(track.get("arc_min")),
             _f(track.get("rate_arcsec_hr")), _f(track.get("motion_pa_deg")),
             _f(track.get("rms_arcsec")), _f(track.get("obsjd_mid")),
             _f(track.get("ra")), _f(track.get("dec")), status,
             track.get("nearest_known"), _f(track.get("nearest_sep_arcsec")),
             _f(track.get("nearest_motion")), _f(track.get("rate_ratio")),
             honesty_class(status), _now()))
        con.execute("DELETE FROM track_members WHERE track_uid=?", (track["track_uid"],))
        con.executemany(
            "INSERT INTO track_members (track_uid,exposure_id,row_id,det_uid,obsjd,"
            " ra,dec,resid_arcsec) VALUES (?,?,?,?,?,?,?,?)",
            [(track["track_uid"], m["exposure_id"], m["row_id"], m["det_uid"],
              _f(m.get("obsjd")), _f(m.get("ra")), _f(m.get("dec")),
              _f(m.get("resid_arcsec"))) for m in members])
        for eid in {m["exposure_id"] for m in members}:
            _update_counts(con, eid)


# ---------------------------------------------------------------- caches


def alerce_coord_key(ra, dec):
    """0.36" bins at the equator - comfortably inside the 2-5" match radius."""
    return f"{float(ra):.4f}_{float(dec):.4f}"


def alerce_get(con, ra, dec):
    """Cached ALeRCE answer, or None if never asked. A row with resolved=0 means the
    CALL FAILED and is returned as such - never as 'no match'."""
    row = con.execute("SELECT * FROM alerce_cache WHERE coord_key=?",
                      (alerce_coord_key(ra, dec),)).fetchone()
    return dict(row) if row else None


def alerce_put(con, ra, dec, oid=None, resolved=True, stamp_ok=None,
               top_class=None, top_prob=None):
    with con:
        con.execute(
            "INSERT INTO alerce_cache (coord_key,oid,resolved,stamp_ok,top_class,"
            " top_prob,fetched_at) VALUES (?,?,?,?,?,?,?) "
            "ON CONFLICT(coord_key) DO UPDATE SET oid=excluded.oid,"
            " resolved=excluded.resolved, stamp_ok=excluded.stamp_ok,"
            " top_class=excluded.top_class, top_prob=excluded.top_prob,"
            " fetched_at=excluded.fetched_at",
            (alerce_coord_key(ra, dec), oid, int(bool(resolved)),
             _i(stamp_ok), top_class, _f(top_prob), _now()))


def record_template(con, ref_key, field, ccdid, qid, fid, ref_ffds, path, seconds=None):
    with con:
        con.execute(
            "INSERT OR REPLACE INTO templates (ref_key,field,ccdid,qid,fid,n_ref,"
            " ref_ffds_json,path,bytes,built_at,seconds) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (ref_key, field, ccdid, qid, fid, len(ref_ffds),
             json.dumps([int(x) for x in ref_ffds]), str(path),
             os.path.getsize(path) if path and os.path.exists(path) else None,
             _now(), _f(seconds)))


# ---------------------------------------------------------------- reporting


def status_counts(con):
    q = con.execute("SELECT status, COUNT(*) n FROM targets GROUP BY status").fetchall()
    out = {r["status"]: r["n"] for r in q}
    out["exposures"] = con.execute("SELECT COUNT(*) FROM exposures").fetchone()[0]
    out["detections"] = con.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
    out["tracks"] = con.execute("SELECT COUNT(*) FROM tracks").fetchone()[0]
    return out


def integrity(con):
    """Structural checks. Returns {check: value}; every count must be 0."""
    checks = {
        "integrity_check": con.execute("PRAGMA integrity_check").fetchone()[0],
        "foreign_key_violations": len(con.execute("PRAGMA foreign_key_check").fetchall()),
        "orphan_detections": con.execute(
            "SELECT COUNT(*) FROM detections d LEFT JOIN exposures e"
            " USING(exposure_id) WHERE e.exposure_id IS NULL").fetchone()[0],
        "orphan_track_members": con.execute(
            "SELECT COUNT(*) FROM track_members m LEFT JOIN detections d"
            " ON m.exposure_id=d.exposure_id AND m.row_id=d.row_id"
            " WHERE d.row_id IS NULL").fetchone()[0],
        # link.py is INTRA-NIGHT by design and has no max-dt gate, so a cross-night
        # member means the harvester grouped wrongly and the track is fiction.
        "cross_night_track_members": con.execute(
            "SELECT COUNT(*) FROM tracks t JOIN track_members m USING(track_uid)"
            " JOIN exposures e ON e.exposure_id=m.exposure_id"
            " WHERE e.ut_date IS NOT t.ut_date").fetchone()[0],
        # an outage must never be readable as a discovery
        "unchecked_marked_recovery": con.execute(
            "SELECT COUNT(*) FROM tracks WHERE novelty_status='unchecked'"
            " AND honesty_class <> 'unchecked'").fetchone()[0],
        "exposures_missing_detections": con.execute(
            "SELECT COUNT(*) FROM exposures WHERE n_detections > 0 AND exposure_id"
            " NOT IN (SELECT DISTINCT exposure_id FROM detections)").fetchone()[0],
    }
    return checks


def _f(v):
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def _i(v):
    try:
        return None if v is None else int(v)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    con = init(connect())
    print(f"survey.db at {DB_PATH}")
    print("status:", status_counts(con))
    for k, v in integrity(con).items():
        print(f"  {k}: {v}")
