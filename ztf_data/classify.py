"""STATIONARY CLASSIFICATION — the missing "what object is this?" pass over the harvest.

The harvest so far is MOTION-only: every detection went through the streak cascade
(STAGE_STREAK), but STAGE_BRAAI/STAGE_TYPE_B have never run on a single one of them —
`_harvest_one` imports and checks those bits but nothing ever sets them (confirmed by
reading it; this is a real, not-yet-built gap, not a flag flip). This module is that
missing stage, run as a separate ENRICHMENT pass over detections already in the DB —
it does NOT re-run run_field, and it does not touch exposures/targets bookkeeping.

WHY NATIVE-ONLY (no fallback to our own cutout): a motion-only harvest never wrote the
(3,63,63) stamp triplets (STAGE_STAMPS unset — see paths.stamps_written), so there is no
local stamp to fall back to the way pipeline.run_braai_gate_native does for the old
single-field run. On top of that, CLAUDE.md's own investigation found our homemade
cutouts carry a registration dipole that tanks braai recall to ~0.05-0.14 versus ~0.68-
0.96 on ZTF's native ALeRCE stamps — so native-only is not just what we have data for,
it is also the MORE ACCURATE route. No ALeRCE match -> every classification field stays
NULL ("not run"), never scored as bogus — the same three-state discipline novelty.py
uses for an unreachable SkyBoT.

BOUNDED PILOT BY DESIGN: classifying all ~289k harvested detections at ALeRCE's network
rate (roughly 1-3 network round trips per candidate) would run well over a day. Per-cell
top-N (default 20) picks the brightest DISTINCT sources per sky cell — deduplicated by
rounded position, so one star seen across 16 epochs costs one lookup, not sixteen — and
propagates the result to every epoch's row of that same source. That is a couple-hour
run that proves the pipeline end to end; widen with --per-cell for fuller coverage later.
"""

import argparse
import contextlib
import os
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

# every native ALeRCE stamp HDUList triggers astropy's FITS-header VerifyWarning for a
# HIERARCH-worthy keyword (STAMP_TYPE) — harmless, but at 47k candidates it drowns out
# the progress lines in a redirected log (they use Python's line-buffered print(), while
# the warnings module writes straight to stderr, so the warnings "win" the race to disk).
warnings.filterwarnings("ignore", category=Warning, module="astropy")

# LOAD-BEARING: the `alerce` client's session.request() calls never pass timeout=, so a
# single stalled TCP connection blocks that worker thread FOREVER (observed twice live:
# a classify run frozen for 2h20m with 0 CPU seconds consumed while stuck on one request).
# Patch requests.Session.request globally to default a timeout when the caller didn't set
# one, so a hung connection raises (caught by classify_one's own try/except, counted as an
# error) instead of wedging a worker permanently. Must run before Alerce() is constructed.
import requests  # noqa: E402
_orig_session_request = requests.Session.request
def _session_request_with_timeout(self, *args, **kwargs):
    kwargs.setdefault("timeout", 30)
    return _orig_session_request(self, *args, **kwargs)
requests.Session.request = _session_request_with_timeout

import config  # noqa: F401  LOAD-BEARING: before ztfquery.io is touched (see ztf_data/__init__.py)

from ztf_data import store

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

BRAAI_MODELS = os.path.join(PROJECT_ROOT, "braai", "models")

# how tightly to bin (ra, dec) when collapsing repeat-epoch detections of the same
# physical source; 3 decimal places is ~1.3" at the equator, comfortably inside the
# stamp_classifier's own match radius, so it never straddles two real objects.
COORD_BIN = 3


def pick_candidates(con, per_cell=20, min_snr=5.0):
    """The N brightest DISTINCT sources per sky cell, not yet classified.

    'Distinct' = deduplicated by rounded (ra, dec): a bright star seen in every one of
    a cell's 16 epochs is one candidate, not sixteen. `streak_route IS NULL` restricts
    to point-source-shaped detections (the streak branch already typed the elongated
    ones as mover/cosmic-ray/bogus — a different question from "what star/galaxy/AGN
    is this"). Returns one representative row per group; classify_and_store propagates
    the result to every det_uid sharing that group.
    """
    rows = con.execute(
        "SELECT d.det_uid, d.row_id, d.ra, d.dec, d.snr, d.exposure_id, "
        " e.field, e.ccdid, e.qid, e.fid "
        "FROM detections d JOIN exposures e ON e.exposure_id = d.exposure_id "
        "WHERE d.streak_route IS NULL AND d.braai_p_real IS NULL AND d.snr >= ? "
        "ORDER BY d.snr DESC",
        (min_snr,),
    ).fetchall()

    by_cell = {}
    for r in rows:
        cell = (r["field"], r["ccdid"], r["qid"], r["fid"])
        key = (round(r["ra"], COORD_BIN), round(r["dec"], COORD_BIN))
        groups = by_cell.setdefault(cell, {})
        groups.setdefault(key, r)          # first hit wins — rows are SNR-sorted already

    picked = []
    for cell, groups in sorted(by_cell.items()):
        best = sorted(groups.values(), key=lambda r: -r["snr"])[:per_cell]
        picked.extend(best)
    return picked


def classify_one(con, client, model, ra, dec, radius_arcsec=3.0, lock=None):
    """Resolve (ra, dec) against ALeRCE; on a match, score braai on the object's own
    native stamp and look up its type. Cached per rounded position across runs via
    store.alerce_get/alerce_put so re-running this script (or a wider --per-cell pass
    later) does not re-pay the network cost for a source already resolved.

    lock=None (the default, serial --workers 1 path) means every call below runs
    unguarded, same as before. When called from a worker thread (--workers > 1), pass
    a shared threading.Lock: it guards ONLY the brief DB touches and the braai forward
    pass, never the ALeRCE HTTP calls themselves — those are the actual bottleneck and
    the whole point of running several at once, so they must NOT be serialized.
    """
    from ztf_classification.stamp_classifier import classify_detection, apply_threshold
    from ztf_classification.braai_realbogus import make_triplet, braai_score

    guard = lock if lock is not None else contextlib.nullcontext()

    out = {"pathB_matched": 0, "pathB_oid": None, "braai_p_real": None,
           "braai_pass": None, "braai_stamp_source": None, "braai_oid": None,
           "pathB_type": None, "pathB_top_class": None, "pathB_prob": None,
           "final_verdict": None, "type_path": None}

    with guard:
        cached = store.alerce_get(con, ra, dec)
    if cached and cached["resolved"]:
        if cached["oid"] is None:
            return out                                  # cached "no match" — not run
        typed = {"matched": True, "oid": cached["oid"],
                 "top_class": cached["top_class"], "top_prob": cached["top_prob"]}
    else:
        try:
            typed = classify_detection(client, ra, dec, radius_arcsec=radius_arcsec)  # NETWORK
        except Exception:
            with guard:
                store.alerce_put(con, ra, dec, resolved=False)
            return out
        with guard:
            store.alerce_put(con, ra, dec, oid=typed.get("oid"),
                             resolved=True, stamp_ok=bool(typed.get("matched")),
                             top_class=typed.get("top_class"), top_prob=typed.get("top_prob"))

    if not typed.get("matched"):
        return out

    out["pathB_matched"] = 1
    out["pathB_oid"] = typed["oid"]
    out["pathB_top_class"] = typed["top_class"]
    out["pathB_prob"] = typed["top_prob"]
    out["pathB_type"] = apply_threshold(typed["top_class"], typed["top_prob"])
    out["type_path"] = "alerce"
    out["final_verdict"] = out["pathB_type"]

    try:
        hdul = client.get_stamps(typed["oid"], format="HDUList")                      # NETWORK
        triplet = make_triplet(hdul[0].data, hdul[1].data, hdul[2].data)
        with guard:                            # braai's Keras model is not asserted thread-safe;
            p_real = float(braai_score(model, triplet))  # this call is cheap, so serializing it
        out["braai_p_real"] = p_real                      # costs ~nothing next to the network wait
        out["braai_pass"] = p_real >= 0.5
        out["braai_stamp_source"] = "native"
        out["braai_oid"] = typed["oid"]
        # braai disagreeing with a matched, typed object is worth surfacing plainly —
        # the type stands (it is ALeRCE's own answer), the verdict notes the doubt.
        if not out["braai_pass"]:
            out["final_verdict"] = out["pathB_type"] + " (braai: low confidence real)"
    except Exception:
        pass                                            # type still stands without braai

    return out


def store_result(con, cell, key, result, lock=None):
    """Propagate one classification to EVERY detection sharing this cell + rounded
    position — the object identity is a property of the SOURCE, not one noisy epoch's
    measurement, so all of that star's rows across every visit get the same answer."""
    field, ccdid, qid, fid = cell
    ra_bin, dec_bin = key
    guard = lock if lock is not None else contextlib.nullcontext()
    with guard, con:
        con.execute(
            "UPDATE detections SET "
            " pathB_matched=?, pathB_oid=?, pathB_type=?, pathB_top_class=?, pathB_prob=?,"
            " braai_p_real=?, braai_pass=?, braai_stamp_source=?, braai_oid=?,"
            " final_verdict=?, type_path=? "
            "WHERE exposure_id IN ("
            "  SELECT exposure_id FROM exposures WHERE field=? AND ccdid=? AND qid=? AND fid=?"
            ") AND ROUND(ra, ?) = ? AND ROUND(dec, ?) = ?",
            (result["pathB_matched"], result["pathB_oid"], result["pathB_type"],
             result["pathB_top_class"], result["pathB_prob"], result["braai_p_real"],
             result["braai_pass"], result["braai_stamp_source"], result["braai_oid"],
             result["final_verdict"], result["type_path"],
             field, ccdid, qid, fid, COORD_BIN, ra_bin, COORD_BIN, dec_bin))
        con.execute(
            "UPDATE exposures SET stage_mask = stage_mask | ? "
            "WHERE field=? AND ccdid=? AND qid=? AND fid=?",
            (store.STAGE_BRAAI | store.STAGE_TYPE_B, field, ccdid, qid, fid))
        con.execute(
            "UPDATE targets SET stage_mask = stage_mask | ? "
            "WHERE field=? AND ccdid=? AND qid=? AND fid=?",
            (store.STAGE_BRAAI | store.STAGE_TYPE_B, field, ccdid, qid, fid))


def run(per_cell=20, min_snr=5.0, limit=None, sleep_s=0.2, dry_run=False, workers=1):
    con = store.connect(store.DB_PATH, check_same_thread=(workers <= 1))
    candidates = pick_candidates(con, per_cell=per_cell, min_snr=min_snr)
    if limit:
        candidates = candidates[:limit]

    n_cells = len({(r["field"], r["ccdid"], r["qid"], r["fid"]) for r in candidates})
    print(f"{len(candidates)} distinct sources across {n_cells} cells "
          f"(top {per_cell}/cell, snr>={min_snr}, workers={workers})")

    if dry_run:
        for r in candidates[:20]:
            print(f"  {r['field']:06d}/c{r['ccdid']:02d}/q{r['qid']}  "
                  f"ra={r['ra']:.5f} dec={r['dec']:.5f} snr={r['snr']:.1f}")
        if len(candidates) > 20:
            print(f"  ... and {len(candidates) - 20} more")
        print("dry run — no network calls made, nothing written")
        return

    from alerce.core import Alerce
    from ztf_classification.braai_realbogus import load_braai
    model = load_braai(BRAAI_MODELS)

    counts = {"matched": 0, "unmatched": 0, "braai_scored": 0, "error": 0}
    t0 = time.time()

    def note(i, total):
        elapsed = time.time() - t0
        rate = i / elapsed if elapsed else 0
        eta_min = (total - i) / rate / 60 if rate else 0
        print(f"  [{i}/{total}] matched={counts['matched']} "
              f"unmatched={counts['unmatched']} braai={counts['braai_scored']} "
              f"errors={counts['error']} — {rate:.2f}/s, ETA {eta_min:.0f} min")

    def tally(result):
        if result["pathB_matched"]:
            counts["matched"] += 1
            if result["braai_p_real"] is not None:
                counts["braai_scored"] += 1
        else:
            counts["unmatched"] += 1

    if workers <= 1:
        client = Alerce()
        for i, r in enumerate(candidates, 1):
            cell = (r["field"], r["ccdid"], r["qid"], r["fid"])
            key = (round(r["ra"], COORD_BIN), round(r["dec"], COORD_BIN))
            try:
                result = classify_one(con, client, model, r["ra"], r["dec"])
                store_result(con, cell, key, result)
                tally(result)
            except Exception as e:
                counts["error"] += 1
                print(f"  [{i}/{len(candidates)}] error at {r['ra']:.5f},{r['dec']:.5f}: {e}")
            if i % 10 == 0 or i == len(candidates):
                note(i, len(candidates))
            if sleep_s:
                time.sleep(sleep_s)
    else:
        # ALeRCE's HTTP round trip (~1-1.5s) is the entire bottleneck — braai inference
        # is a small local CNN forward pass and negligible next to it — so this
        # parallelizes network WAIT, not compute. `lock` serializes only the brief DB
        # touches and the braai call itself (see classify_one's docstring); the network
        # calls run fully concurrently across worker threads. `sleep_s` becomes a
        # PER-THREAD pace, so total request rate scales roughly as workers/sleep_s —
        # a deliberate politeness knob, not a bug: raise --workers to go faster, but
        # each extra worker is that much more concurrent load on ALeRCE's servers.
        lock = threading.Lock()
        local = threading.local()

        def worker_client():
            if not hasattr(local, "client"):
                local.client = Alerce()
            return local.client

        def task(r):
            result = classify_one(con, worker_client(), model, r["ra"], r["dec"], lock=lock)
            if sleep_s:
                time.sleep(sleep_s)
            return r, result

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(task, r) for r in candidates]
            for i, fut in enumerate(as_completed(futures), 1):
                try:
                    r, result = fut.result()
                    cell = (r["field"], r["ccdid"], r["qid"], r["fid"])
                    key = (round(r["ra"], COORD_BIN), round(r["dec"], COORD_BIN))
                    store_result(con, cell, key, result, lock=lock)
                    tally(result)
                except Exception as e:
                    counts["error"] += 1
                    print(f"  [{i}/{len(candidates)}] error: {e}")
                if i % 10 == 0 or i == len(candidates):
                    note(i, len(candidates))

    print(f"done in {(time.time() - t0)/60:.1f} min — {counts}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--per-cell", type=int, default=20,
                   help="brightest distinct sources to classify per sky cell (default 20)")
    p.add_argument("--min-snr", type=float, default=5.0)
    p.add_argument("--limit", type=int, default=None, help="cap total candidates (safety)")
    p.add_argument("--sleep", type=float, default=0.2,
                   help="seconds a worker pauses between its own ALeRCE calls (politeness)")
    p.add_argument("--workers", type=int, default=1,
                   help="concurrent ALeRCE lookups (network-bound, so this genuinely helps; "
                        "each extra worker is more concurrent load on ALeRCE's servers)")
    p.add_argument("--dry-run", action="store_true", help="print the selection, call nothing")
    a = p.parse_args()
    run(per_cell=a.per_cell, min_snr=a.min_snr, limit=a.limit, sleep_s=a.sleep,
        dry_run=a.dry_run, workers=a.workers)
