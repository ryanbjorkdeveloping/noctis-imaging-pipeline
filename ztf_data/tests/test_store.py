"""Unit tests for the survey store. Temp DB, no network, no pipeline.

The load-bearing one is `row_id` collision: stamp basenames like 'src_p1_000' RESTART
at every run, so two exposures genuinely both contain one. Anything keyed on row_id
alone silently merges unrelated detections across the survey - det_uid exists to make
that impossible, and this asserts it rather than trusting the naming convention.

The rest guard the invariants an unattended harvester leans on: idempotent enqueue,
stage_mask skip logic, retryable-vs-fatal failure accounting, the three-state novelty
CHECK, and the honesty mapping that must never turn an outage into a recovery.
"""

import os
import sqlite3
import tempfile

from ztf_data import store
from ztf_data.store import (
    STAGE_DETECT, STAGE_STAMPS, STAGE_STREAK, STAGE_BRAAI, STAGE_VERDICT,
    MAX_ATTEMPTS,
)

MOTION = STAGE_DETECT | STAGE_STREAK | STAGE_VERDICT
STATIONARY = MOTION | STAGE_STAMPS | STAGE_BRAAI


def _target(ffd, field=518, ccdid=1, qid=1, fid=2, ut_date="2019-02-08", obsjd=2458522.5):
    return {"field": field, "ccdid": ccdid, "qid": qid, "fid": fid,
            "science_filefracday": ffd, "ut_date": ut_date, "obsjd": obsjd,
            "ra": 139.2, "dec": 9.3, "seeing": 2.1, "maglimit": 20.1, "infobits": 0}


def _result(tkey, rows, stage_mask=MOTION, **kw):
    res = {"target_key": tkey, "stage_mask": stage_mask, "seconds": 1.0,
           "run_dir": f"/tmp/{tkey}", "obsjd": 2458522.5, "ra_center": 139.2,
           "dec_center": 9.3, "corners": [[1, 2], [3, 4], [5, 6], [7, 8]],
           "region_shape": [2952, 2944], "ref_key": "deadbeef", "detections": rows}
    res.update(kw)
    return res


def _det(row_id, **kw):
    d = {"row_id": row_id, "ra": 139.1, "dec": 9.4, "snr": 20.0, "sign": 1,
         "elongation": 2.0, "on_edge": False}
    d.update(kw)
    return d


def test_store():
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "survey.db")
        con = store.init(store.connect(db))

        # ---------------------------------------------------------- enqueue
        n = store.enqueue(con, [_target(1), _target(2)], MOTION)
        assert n == 2, n
        # idempotent: re-enqueueing the same night must add nothing
        assert store.enqueue(con, [_target(1), _target(2)], MOTION) == 0
        # but it DOES widen the ask - this is the wide-then-deep flow
        assert store.enqueue(con, [_target(1)], STATIONARY) == 0
        got = con.execute("SELECT want_mask FROM targets WHERE science_filefracday=1"
                          ).fetchone()[0]
        assert got & STAGE_BRAAI, "re-enqueueing with a wider mask must widen want_mask"

        # the detection REGION is part of the identity, not just the manifest
        assert store.enqueue(con, [dict(_target(1), full_frame=0)], MOTION) == 1, \
            "a central-region run of the same epoch is a DIFFERENT target"

        # ---------------------------------------------------------- claim / skip
        claimed = store.claim(con, MOTION, limit=10)
        assert len(claimed) == 3
        assert all(c["status"] == "pending" for c in claimed), "claim returns pre-state"
        assert con.execute("SELECT COUNT(*) FROM targets WHERE status='running'"
                           ).fetchone()[0] == 3

        # ---------------------------------------------------------- row_id collision
        k1 = store.target_key(518, 1, 1, 2, 1, True)
        k2 = store.target_key(518, 1, 1, 2, 2, True)
        store.ingest(con, _result(k1, [_det("src_p1_000"), _det("src_m1_000")]))
        store.ingest(con, _result(k2, [_det("src_p1_000")]))
        uids = [r[0] for r in con.execute("SELECT det_uid FROM detections").fetchall()]
        assert len(uids) == 3 and len(set(uids)) == 3, (
            f"row_id collides across exposures; det_uid must disambiguate: {uids}")
        assert k1 != k2 and all(u.startswith((k1, k2)) for u in uids)

        # counts are denormalized at ingest, not computed by the UI
        row = con.execute("SELECT n_detections FROM exposures WHERE target_key=?",
                          (k1,)).fetchone()
        assert row[0] == 2, row[0]

        # ---------------------------------------------------------- skip logic
        # both are done with MOTION, so a MOTION harvest must not re-claim them...
        pending = [c["target_key"] for c in store.claim(con, MOTION, limit=10)]
        assert k1 not in pending and k2 not in pending, \
            "a target already satisfying want_mask must be skipped (no IRSA round-trip)"
        # ...but asking for STATIONARY re-selects them for the MISSING stages only
        again = [c["target_key"] for c in store.claim(con, STATIONARY, limit=10)]
        assert k1 in again and k2 in again, "a wider want_mask must re-select"

        # ------------------------------------------- wide-then-deep must not destroy
        # The chosen seed shape harvests motion first and deepens later, so a target
        # gets ingested MORE THAN ONCE with different branches. A later motion-only
        # pass carries no braai keys; if the upsert wrote them as NULL it would
        # silently erase the stationary work. COALESCE is what prevents that.
        store.ingest(con, _result(
            k2, [_det("src_p1_000", braai_p_real=0.91, braai_pass=True,
                      braai_stamp_source="native", final_verdict="VS")],
            stage_mask=STATIONARY))
        store.ingest(con, _result(          # motion-only re-run, no braai fields at all
            k2, [_det("src_p1_000", snr=25.0, streak_route="SHORT_NEA_CANDIDATE")],
            stage_mask=MOTION))
        row = con.execute(
            "SELECT braai_p_real,braai_pass,final_verdict,snr,streak_route"
            " FROM detections WHERE det_uid=?", (store.det_uid(k2, "src_p1_000"),)
        ).fetchone()
        assert row["braai_p_real"] == 0.91, \
            f"a motion re-run erased the braai score: {row['braai_p_real']}"
        assert row["braai_pass"] == 1 and row["final_verdict"] == "VS"
        assert row["snr"] == 25.0, "fresh values must still land"
        assert row["streak_route"] == "SHORT_NEA_CANDIDATE"
        # and the exposure's stage_mask accumulates rather than being replaced
        sm = con.execute("SELECT stage_mask FROM exposures WHERE target_key=?",
                         (k2,)).fetchone()[0]
        assert sm & STAGE_BRAAI and sm & STAGE_STREAK, sm

        # ---------------------------------------------------------- failures
        store.record_failure(con, k1, "boom", retryable=False)
        a1 = con.execute("SELECT attempts,status FROM targets WHERE target_key=?",
                         (k1,)).fetchone()
        assert a1["status"] in ("failed", "skipped")
        before = a1["attempts"]
        # a RETRYABLE failure (IRSA outage) must not consume an attempt, or a blip
        # permanently skips good sky
        store.claim(con, STATIONARY, limit=10)
        store.record_failure(con, k1, "IRSA 502", retryable=True)
        a2 = con.execute("SELECT attempts,status FROM targets WHERE target_key=?",
                         (k1,)).fetchone()
        assert a2["status"] == "pending", a2["status"]
        assert a2["attempts"] <= before, (
            f"retryable failure consumed an attempt: {before} -> {a2['attempts']}")

        # the attempt cap eventually stops a genuinely bad target
        for _ in range(MAX_ATTEMPTS + 2):
            store.claim(con, STATIONARY, limit=10)
            store.record_failure(con, k1, "boom", retryable=False)
        assert con.execute("SELECT status FROM targets WHERE target_key=?",
                           (k1,)).fetchone()[0] == "skipped"

        # ---------------------------------------------------------- three-state novelty
        eid = con.execute("SELECT exposure_id FROM exposures WHERE target_key=?",
                          (k2,)).fetchone()[0]
        try:
            with con:
                con.execute("UPDATE detections SET novelty_status='maybe'"
                            " WHERE exposure_id=?", (eid,))
            raise AssertionError("novelty_status must be constrained to three states")
        except sqlite3.IntegrityError:
            pass

        # honesty mapping: an outage is NEVER a recovery
        assert store.honesty_class("known") == "recovery"
        assert store.honesty_class("novel") == "unconfirmed"
        assert store.honesty_class("unchecked") == "unchecked"
        assert store.honesty_class(None) == "unchecked"

        # ---------------------------------------------------------- tracks + cascade
        det = con.execute("SELECT det_uid,row_id FROM detections WHERE exposure_id=?",
                          (eid,)).fetchone()
        store.upsert_track(
            con,
            {"track_uid": "2019-02-08:t0", "n_det": 3, "rate_arcsec_hr": 6900.0,
             "novelty_status": "known", "nearest_known": "2019 BE5", "rate_ratio": 1.04},
            [{"exposure_id": eid, "row_id": det["row_id"], "det_uid": det["det_uid"],
              "obsjd": 2458522.5, "ra": 139.1, "dec": 9.4, "resid_arcsec": 5.0}],
            ut_date="2019-02-08", link_run="t")
        assert con.execute("SELECT honesty_class FROM tracks").fetchone()[0] == "recovery"

        checks = store.integrity(con)
        assert checks["integrity_check"] == "ok", checks
        assert all(v == 0 for k, v in checks.items() if k != "integrity_check"), checks

        # deleting an exposure must take its detections AND track members with it
        with con:
            con.execute("DELETE FROM exposures WHERE exposure_id=?", (eid,))
        assert con.execute("SELECT COUNT(*) FROM detections WHERE exposure_id=?",
                           (eid,)).fetchone()[0] == 0, "ON DELETE CASCADE not firing"

        # ---------------------------------------------------------- alerce cache
        assert store.alerce_get(con, 1.0, 2.0) is None, "never asked -> None"
        store.alerce_put(con, 1.0, 2.0, oid=None, resolved=True)
        hit = store.alerce_get(con, 1.0, 2.0)
        assert hit["resolved"] == 1 and hit["oid"] is None, \
            "'asked, no match' must be distinguishable from 'never asked'"
        store.alerce_put(con, 3.0, 4.0, resolved=False)
        assert store.alerce_get(con, 3.0, 4.0)["resolved"] == 0, \
            "a FAILED call must never be readable as 'no match'"

        print("OK: det_uid disambiguates colliding row_ids; skip/retry/CHECK/cascade hold")


if __name__ == "__main__":
    test_store()
