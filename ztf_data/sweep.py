"""Multi-field motion sweep: run_field + streak scoring across many (field, epoch)
targets -> one aggregated table of fast-asteroid (streak) candidates.

The payoff of generalizing run_field: point it at many fields, tap each output for
streaks, collect the SHORT_NEA candidates. force=False + the per-run template cache
make re-runs cheap; a per-target failure is logged, not fatal.
"""

import os
import traceback

from astropy.io import fits
from astropy.table import Table, vstack

from ztf_data.run_field import run_field
from ztf_data.streaks import streaks_from_run
from ztf_data.novelty import annotate_novelty, annotate_track_novelty
from ztf_data.link import link_from_candidates
from ztf_data.night import enumerate_night, night_targets
from ztf_data.paths import OWN_ROOT

STREAK_ROUTES = ("SHORT_NEA_CANDIDATE", "LONG_SATELLITE")


def _obsjd(diff_path):
    """The science exposure's OBSJD, from the diff header (HDU 1 for .fz)."""
    hdu = 1 if str(diff_path).endswith(".fz") else 0
    return float(fits.getheader(str(diff_path), hdu)["OBSJD"])


def _cap_worker_threads(n_workers, n_cores=None):
    """Cap per-process math/TF threads to ~cores/n_workers so N concurrent scorers don't
    each grab ALL cores (oversubscription -> thrash -> ~2x instead of ~3x). Set in the
    PARENT before the pool starts so the spawned workers AND their .venv_streaks scoring
    subprocess (which inherits the worker's env) both see it. setdefault, so an explicit
    caller override wins."""
    n = n_cores or os.cpu_count() or 4
    t = str(max(1, n // max(1, n_workers)))
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
                "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS"):
        os.environ.setdefault(var, t)


def _sweep_one(args):
    """One target's run_field + streak scoring — the CPU-heavy, parallelizable unit.
    Module-level + picklable args so it survives macOS 'spawn' in a ProcessPool.

    scratch_dir=None is load-bearing under parallelism: streaks_from_run then writes the
    .venv_streaks handoff files into the target's OWN run_dir (unique per field/ccd/qid/ffd),
    so concurrent workers never clobber each other's stamps/results (the silent-corruption race).
    Returns (streaks_table_or_None, label, n_detections, error_or_None) - never raises, so one
    bad target can't kill the pool.
    """
    tgt, run_kwargs = args
    label = f"field {tgt.get('field')} / {tgt.get('science_filefracday') or tgt.get('science_year')}"
    try:
        fr = run_field(**tgt, **run_kwargs)
        cands = streaks_from_run(fr, scratch_dir=None)
        streaks = cands[[r in STREAK_ROUTES for r in cands["route"]]] if len(cands) else cands
        if len(streaks):
            annotate_novelty(streaks, _obsjd(fr.diff_path))
        streaks["field"] = tgt.get("field")
        streaks["science_filefracday"] = fr.science_filefracday
        return streaks, label, fr.n_detections, None
    except Exception:
        return None, label, 0, traceback.format_exc()


def sweep_targets(targets, out_path=None, scratch_dir=None, n_workers=1, **run_kwargs):
    """Run the full pipeline + streak scoring over a list of targets.

    targets: list of dicts, each {field, ccdid, qid, fid, science_year OR
    science_filefracday}. Returns (candidates, tracks): a table of streak candidates
    (route in STREAK_ROUTES) across all fields with field/science_filefracday provenance,
    plus the linked tracks table. A per-target exception is caught + logged so one bad
    field can't kill the sweep.

    n_workers>1 scores targets in parallel (ProcessPoolExecutor). The bottleneck is CPU
    CNN scoring (~30 s/target), so this is the scaling lever; RAM caps it (~1.8 GB/worker,
    so ~4 on 16 GB). Each worker uses its OWN run_dir scratch (no shared-file race), threads
    are capped so scorers don't oversubscribe, and linking stays serial after aggregation.
    """
    if out_path is None:
        out_path = os.path.join(OWN_ROOT, "sweep_candidates.ecsv")

    args = [(tgt, run_kwargs) for tgt in targets]
    if n_workers and n_workers > 1:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor
        _cap_worker_threads(n_workers)
        ctx = multiprocessing.get_context("spawn")  # macOS default; explicit for safety
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex:
            results = list(ex.map(_sweep_one, args))   # order preserved -> [i/N] labels stay right
    else:
        results = [_sweep_one(a) for a in args]

    per_field = []
    for i, (streaks, label, n_det, err) in enumerate(results, 1):
        if err is not None:
            print(f"[{i}/{len(targets)}] {label}: FAILED\n{err}")
            continue
        if len(streaks):
            per_field.append(streaks)
        print(f"[{i}/{len(targets)}] {label}: {n_det} detections, "
              f"{len(streaks)} streak candidate(s)")

    result = vstack(per_field) if per_field else Table()
    result.write(out_path, format="ascii.ecsv", overwrite=True)
    print(f"\nsweep complete: {len(result)} streak candidate(s) across "
          f"{len(targets)} target(s) -> {out_path}")

    # LINK the per-detection candidates into tracks - the discovery-grade output. One
    # row per MOVING OBJECT with a measured motion vector, so novelty matches position
    # AND rate (far more specific than the per-detection nearest-fast-mover check).
    # Needs the time/direction columns streaks_from_run emits; older runs lack them.
    tracks = Table()
    if len(result) and {"obsjd", "streak_pa"} <= set(result.colnames):
        tracks = link_from_candidates(result)
        if len(tracks):
            annotate_track_novelty(tracks)
        tracks_path = os.path.join(os.path.dirname(out_path), "tracks.ecsv")
        tracks.write(tracks_path, format="ascii.ecsv", overwrite=True)
        n_novel = int(sum(tracks["is_novel"])) if len(tracks) else 0
        print(f"linked: {len(tracks)} track(s) ({n_novel} novel) -> {tracks_path}")
    return result, tracks


def sweep_night(date, ra_range=None, dec_range=None, fields=None, fid=2,
                ut_hour_range=None, max_targets=200, streaks_only=True, full_frame=True,
                n_workers=1, out_path=None, scratch_dir=None):
    """Discovery in one call: enumerate a night's exposures in a sky region, sweep them
    for streaks, link into tracks. Returns (candidates, tracks); tracks.ecsv carries
    is_novel. This is the hand-list-free front end - the caller names a NIGHT and a BOX,
    not a target list.

    max_targets caps how many exposures are swept (they run SERIALLY at ~45 s each, so a
    full night+region of ~1000s would take hours). It is a GUARD, not a science cut: raise
    it deliberately, or narrow the region/field list. Exposures are swept EARLIEST-FIRST
    (enumerate_night sorts by obsjd), so a capped run still covers one contiguous time
    slice - the shape a mover actually traverses - rather than a scattered sample.

    full_frame + streaks_only default True: the discovery configuration (8.7x the sky per
    exposure, and the diff-only path that skips the 1.6 GB reference download per field).
    """
    table = enumerate_night(date, ra_range=ra_range, dec_range=dec_range,
                            fields=fields, fid=fid, ut_hour_range=ut_hour_range)
    if max_targets is not None and len(table) > max_targets:
        print(f"night has {len(table)} exposures in region; capping to the first "
              f"{max_targets} by time (raise max_targets or narrow the region to cover more)")
        table = table[:max_targets]
    targets = night_targets(table)
    print(f"sweeping {len(targets)} exposure(s) for {date} (n_workers={n_workers})...")
    return sweep_targets(targets, out_path=out_path, scratch_dir=scratch_dir,
                         n_workers=n_workers, streaks_only=streaks_only, full_frame=full_frame)


if __name__ == "__main__":
    import json
    import sys

    # `python -m ztf_data.sweep night 2019-01-31 120 127 13 17` -> sweep_night; or a
    # targets JSON path -> sweep_targets; or no arg -> the cached 2020 one-target demo.
    if len(sys.argv) > 2 and sys.argv[1] == "night":
        date = sys.argv[2]
        box = [float(x) for x in sys.argv[3:7]] if len(sys.argv) >= 7 else [None] * 4
        sweep_night(date, ra_range=box[:2] if box[0] is not None else None,
                    dec_range=box[2:] if box[2] is not None else None)
    elif len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            targets = json.load(f)
        sweep_targets(targets)
    else:
        targets = [dict(field=468, ccdid=3, qid=2, fid=2,
                        science_filefracday=20200518187454)]
        sweep_targets(targets)
