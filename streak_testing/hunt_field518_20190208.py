"""First real blind discovery hunt: one high-cadence quadrant, many time-visits.

Field 518 / c01 / q1 was imaged 54x on 2019-02-08 at a steady 4.3-min cadence - the
dense temporal sampling the linker needs (ZTF's usual ~2 visits/night can never reach
MIN_DETECTIONS=3). We sweep the first N visits of this ONE 0.85-deg quadrant: a fast NEA
crossing the patch would land in ~6 consecutive exposures -> a linkable track. This is
UNVETTED sky (we do not know what, if anything, is there); the honest expected result is
0 tracks or a KNOWN mover, and either way it exercises the corrected pipeline + linker on
real non-BE5 survey data.

MUST be a guarded __main__ script: n_workers>1 uses spawn, and spawned children re-import
the driver at module level - an unguarded parallel sweep call would recursively spawn.

Run from project root:  .venv/bin/python streak_testing/hunt_field518_20190208.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # sets $ZTFDATA before ztfquery

from ztf_data.night import enumerate_night
from ztf_data.sweep import sweep_targets

FIELD, CCD, QID, FID = 518, 1, 1, 2
DATE = "2019-02-08"
N_VISITS = 16
N_WORKERS = 4


def main():
    # the 54 visits of this one quadrant, earliest first
    t = enumerate_night(DATE, fields=[FIELD], fid=FID)
    sub = t[[(int(r["ccdid"]), int(r["qid"])) == (CCD, QID) for r in t]]
    sub.sort("obsjd")
    ffds = [int(x) for x in sub["science_filefracday"]][:N_VISITS]
    targets = [dict(field=FIELD, ccdid=CCD, qid=QID, fid=FID, science_filefracday=f)
               for f in ffds]
    print(f"hunting {len(targets)} visits of field {FIELD}/c{CCD:02d}/q{QID} "
          f"({DATE}), n_workers={N_WORKERS}, full_frame")

    cands, tracks = sweep_targets(targets, streaks_only=True, full_frame=True,
                                  n_workers=N_WORKERS)

    print("\n=== HUNT RESULT ===")
    print("streak candidates:", len(cands))
    print("tracks           :", len(tracks))
    for r in tracks:
        print(f"  rate={r['rate_arcsec_hr']:.0f}\"/hr pa={r['motion_pa_deg']:.1f} "
              f"n_det={r['n_detections']} status={r['novelty_status']} "
              f"is_novel={r['is_novel']} nearest={r['nearest_known']}")
    novel = [r for r in tracks if r["is_novel"]]
    print(f"\nNOVEL tracks (discovery candidates): {len(novel)}")


if __name__ == "__main__":
    main()
