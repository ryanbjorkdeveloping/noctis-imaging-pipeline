"""Harvester queue/ingest/resume logic, with run_field faked. No network, no pipeline.

run_field is monkeypatched on purpose rather than run for real: this test is about the
HARVESTER (claim -> ingest -> skip -> recover -> link per night), and faking the one
seam makes it deterministic, offline, and fast. The real pipeline path is covered by
test_regression_2020 and test_full_frame_stamps.

The two assertions worth the file:
  1. A second run over satisfied targets claims nothing AND never calls run_field.
     If the skip regresses, an unattended harvester silently re-downloads and
     re-detects everything it already has - the exact cost the store exists to avoid.
     Enforced by making the fake raise on any call during the second pass.
  2. Linking is grouped per UT night. link_tracklets has NO maximum time separation
     (only dt>0 and a rate window), so a multi-night candidate set fabricates tracks.
"""

import os
import tempfile

import numpy as np
from astropy.table import Table

import ztf_data.harvest as harvest
from ztf_data import store
from ztf_data.run_field import FieldResult
from ztf_data.store import STAGE_DETECT, STAGE_STREAK, STAGE_BRAAI

N_DET = 5
WANT = STAGE_DETECT


def _fake_field_result(root, tgt, full_frame=True, n=N_DET):
    """Write a real catalog.ecsv so _rows_from_catalog exercises its real code path."""
    tkey = store.target_key(tgt["field"], tgt["ccdid"], tgt["qid"], tgt["fid"],
                            tgt["science_filefracday"], full_frame)
    rdir = os.path.join(root, tkey)
    cuts = os.path.join(rdir, "cutouts")
    os.makedirs(cuts, exist_ok=True)
    cat = Table({
        "x_centroid": np.arange(n, dtype=float) + 10.0,
        "y_centroid": np.arange(n, dtype=float) + 20.0,
        "ra": 139.0 + np.arange(n) * 0.001,
        "dec": 9.0 + np.arange(n) * 0.001,
        "snr": np.linspace(5.0, 50.0, n),
        "segment_flux": np.linspace(100.0, 900.0, n),
        "elongation": np.linspace(1.0, 4.0, n),
        "orientation": np.zeros(n),
        "sign": np.where(np.arange(n) % 2 == 0, 1, -1),
        "on_edge": np.zeros(n, dtype=bool),
        # row_id collides across exposures BY DESIGN - that is what det_uid handles
        "stamp_path": [os.path.join(cuts, f"src_p1_{i:03d}.npy") for i in range(n)],
    })
    cpath = os.path.join(rdir, "catalog.ecsv")
    cat.write(cpath, format="ascii.ecsv", overwrite=True)
    return FieldResult(
        run_dir=rdir, catalog_path=cpath, cutouts_dir=cuts,
        diff_path=os.path.join(rdir, "diff.fits"), template_path=None,
        science_filefracday=int(tgt["science_filefracday"]), n_detections=n,
        full_frame=full_frame, obsjd=float(tgt.get("obsjd", 2458522.5)),
        ra_center=139.0, dec_center=9.0,
        corners=[[139.4, 8.7], [138.6, 8.7], [138.6, 9.4], [139.4, 9.4]],
        ref_key="cafebabe", science_infobits=0,
    )


def _targets(n=4, date="2019-02-08"):
    return [{"field": 518, "ccdid": 1, "qid": 1, "fid": 2,
             "science_filefracday": 20190208200000 + i, "ut_date": date,
             "obsjd": 2458522.5 + i * 0.01, "ra": 139.0, "dec": 9.0,
             "seeing": 2.1, "maglimit": 20.0, "infobits": 0, "full_frame": 1}
            for i in range(n)]


def test_harvest_smoke():
    with tempfile.TemporaryDirectory() as tmp:
        con = store.init(store.connect(os.path.join(tmp, "survey.db")))
        real_run_field = harvest.run_field
        calls = {"n": 0}

        def fake_run_field(**kw):
            calls["n"] += 1
            tgt = dict(kw)
            tgt.setdefault("obsjd", 2458522.5)
            return _fake_field_result(tmp, tgt, full_frame=kw.get("full_frame", True))

        harvest.run_field = fake_run_field
        try:
            tgts = _targets(4)
            assert harvest.store.enqueue(con, tgts, WANT) == 4

            # ---------------------------------------------- first pass
            c = harvest.run_batch(con, n_workers=1, batch=10, want_mask=WANT,
                                  streaks_only=True, full_frame=True)
            assert c == {"claimed": 4, "done": 4, "failed": 0, "retryable": 0}, c
            assert calls["n"] == 4, calls
            assert con.execute("SELECT COUNT(*) FROM exposures").fetchone()[0] == 4
            assert con.execute("SELECT COUNT(*) FROM detections").fetchone()[0] == 4 * N_DET

            # row_id collides across all 4 exposures; det_uid must not
            uids = [r[0] for r in con.execute("SELECT det_uid FROM detections")]
            rids = [r[0] for r in con.execute("SELECT row_id FROM detections")]
            assert len(set(rids)) == N_DET, "row_ids are supposed to collide here"
            assert len(set(uids)) == 4 * N_DET, "det_uid failed to disambiguate"

            # a motion-only pass must NOT invent verdicts: braai never ran, and
            # merge_verdict would otherwise label every row 'bogus'
            verdicts = [r[0] for r in con.execute(
                "SELECT DISTINCT final_verdict FROM detections")]
            assert verdicts == [None], (
                f"motion-only run fabricated verdicts without braai: {verdicts}")

            # ---------------------------------------------- resume: skip, no work
            calls["n"] = 0

            def exploding_run_field(**kw):
                raise AssertionError("run_field called for an already-satisfied target")

            harvest.run_field = exploding_run_field
            c2 = harvest.run_batch(con, n_workers=1, batch=10, want_mask=WANT,
                                   streaks_only=True, full_frame=True)
            assert c2["claimed"] == 0, f"satisfied targets were re-claimed: {c2}"
            assert calls["n"] == 0

            # ...but a WIDER want_mask re-selects them (the wide-then-deep flow)
            harvest.run_field = fake_run_field
            c3 = harvest.run_batch(con, n_workers=1, batch=10,
                                   want_mask=WANT | STAGE_BRAAI,
                                   streaks_only=True, full_frame=True)
            assert c3["claimed"] == 4, c3

            # ---------------------------------------------- kill recovery
            with con:
                con.execute("UPDATE targets SET status='running'")
            assert store.reset_running(con) == 4
            assert con.execute("SELECT COUNT(*) FROM targets WHERE status='running'"
                               ).fetchone()[0] == 0

            # ---------------------------------------------- per-night linking
            _plant_track(con, "2019-02-08", 2458522.5)
            _plant_track(con, "2019-02-09", 2458523.5)
            n_tracks = harvest.link_pending_nights(con)
            assert n_tracks == 2, f"expected one track per night, got {n_tracks}"

            nights = [r[0] for r in con.execute("SELECT ut_date FROM tracks ORDER BY 1")]
            assert nights == ["2019-02-08", "2019-02-09"], nights
            checks = store.integrity(con)
            assert checks["cross_night_track_members"] == 0, checks
            assert checks["integrity_check"] == "ok"
            assert all(v == 0 for k, v in checks.items() if k != "integrity_check"), checks

            # every member resolves to a real detection, and to the right night
            bad = con.execute(
                "SELECT COUNT(*) FROM track_members m JOIN tracks t USING(track_uid)"
                " JOIN exposures e ON e.exposure_id=m.exposure_id"
                " WHERE e.ut_date <> t.ut_date").fetchone()[0]
            assert bad == 0

            print(f"OK: 4 exposures, {4 * N_DET} detections, skip is free, "
                  f"{n_tracks} tracks grouped per night")
        finally:
            harvest.run_field = real_run_field


def _plant_track(con, date, jd0):
    """Three collinear, constant-rate SHORT detections on one night -> one track.

    ~6900"/hr due east at dec 0 puts it inside link's [1000, 20000] rate window with a
    steady 90 deg position angle, which is what the PA gate requires.
    """
    for i in range(3):
        tgt = {"field": 900 + i, "ccdid": 1, "qid": 1, "fid": 2,
               "science_filefracday": int(date.replace("-", "")) * 1000 + i,
               "ut_date": date, "obsjd": jd0 + i * 0.01, "full_frame": 1}
        store.enqueue(con, [tgt], STAGE_DETECT | STAGE_STREAK)
        tkey = store.target_key(tgt["field"], 1, 1, 2, tgt["science_filefracday"], True)
        store.claim(con, STAGE_DETECT | STAGE_STREAK, limit=50)
        store.ingest(con, {
            "target_key": tkey, "stage_mask": STAGE_DETECT | STAGE_STREAK,
            "run_dir": "/tmp/x", "obsjd": jd0 + i * 0.01, "ra_center": 0.0,
            "dec_center": 0.0, "corners": None, "region_shape": [10, 10],
            "seconds": 0.1, "ref_key": None,
            "detections": [{
                "row_id": "src_p1_000", "ra": 100.0 + i * 0.46, "dec": 0.0,
                "snr": 30.0, "sign": 1, "elongation": 4.0, "on_edge": False,
                "streak_route": "SHORT_NEA_CANDIDATE", "streak_pa": 90.0,
            }],
        })


if __name__ == "__main__":
    test_harvest_smoke()
