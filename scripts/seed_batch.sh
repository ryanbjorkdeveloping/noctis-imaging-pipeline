#!/usr/bin/env bash
# Seed batch: 10 sky cells x 10 epochs each, all branches, into the survey store.
#
#   ./seed_batch.sh [UT_DATE] [N_CELLS] [N_EPOCHS] [WORKERS]
#   ./seed_batch.sh 2019-02-08 10 10 4
#
# WHAT THE SHAPE MEANS, and why it is not arbitrary:
#
#   10 epochs of ONE cell = 10 exposures of the same patch of sky, all reprojected
#   onto that cell's shared deep reference template. That is the "10 images aligned
#   on top of each other" - the pipeline differences each science epoch against the
#   stack and detects what CHANGED between them.
#
#   10 cells = 10 different patches of sky, each ~0.83 deg on a side.
#
#   Cells are chosen by VISIT COUNT that night, not at random. Two reasons, both
#   load-bearing:
#     - link.MIN_DETECTIONS=3 is unreachable on ZTF's usual ~2 visits/night, so only
#       a high-cadence cell can ever produce a linked track (a moving object).
#     - the deep template amortizes per CELL, so 10 epochs of one cell cost ONE
#       5-10 min stack, while 10 scattered cells cost ten of them plus 10x the
#       reference download. Cadence-first is the difference between feasible and not.
#
# RESUMABLE: every step is idempotent. Ctrl-C and re-run - the store skips whatever
# is already satisfied, with zero IRSA round-trips for finished targets.
set -euo pipefail
cd "$(dirname "$0")/.."   # scripts/ -> repo root

DATE="${1:-2019-02-08}"
CELLS="${2:-10}"
EPOCHS="${3:-10}"
WORKERS="${4:-4}"
PY=.venv/bin/python

echo "=== seed batch: ${CELLS} cells x ${EPOCHS} epochs on ${DATE}, ${WORKERS} workers ==="

# ---- preflight -------------------------------------------------------------
# This batch DOWNLOADS, so it needs IRSA. Check before enqueueing: IRSA answers an
# outage with a 200-OK HTML error page that parses into a bogus DataFrame, so the
# failure mode is a confusing KeyError deep in the run, not a connection error.
# epochs.py raises IRSAUnavailable for exactly this - fail here instead, loudly.
echo "-- preflight: IRSA reachability"
if ! $PY - <<'EOF'
import sys
import config  # noqa: F401  LOAD-BEARING: sets $ZTFDATA before ztfquery imports
from ztf_data.epochs import query_grid, IRSAUnavailable
try:
    _, e = query_grid(468, 3, 2, 2)
    print(f"   IRSA OK - {len(e)} epochs for the probe cell")
except IRSAUnavailable as exc:
    print(f"   IRSA UNAVAILABLE: {exc}")
    sys.exit(1)
except Exception as exc:
    print(f"   IRSA probe failed: {type(exc).__name__}: {exc}")
    sys.exit(1)
EOF
then
    cat <<'MSG'

   IRSA is not answering, so nothing new can be downloaded. This is an outage, not
   a bug on our side - every guard behaved correctly.

   What still works right now, with no network at all:
       .venv/bin/python -m ztf_data.harvest enqueue --local --fid 2
       .venv/bin/python -m ztf_data.harvest run --offline --branches motion --workers 4
   That reduces every frame already in ztfdata/sci/ and is how the current survey
   store was built. Re-run this script when IRSA recovers.
MSG
    exit 1
fi

# ---- the batch -------------------------------------------------------------
echo "-- init store (idempotent)"
$PY -m ztf_data.harvest init

# --spread: one cell per DISTINCT field. Without it, ranking by visit count returns
# whole quadrants of a single field (a ZTF field reads out as 64 quadrants that share
# a visit count), which is ten distinct patches tiled into one contiguous block - not
# ten different areas of sky.
echo "-- enqueue ${CELLS}x${EPOCHS} on ${DATE} (spread across fields)"
$PY -m ztf_data.harvest enqueue --date "$DATE" --cadence "${CELLS}x${EPOCHS}" \
    --spread --branches motion,stationary

# WIDE THEN DEEP - two passes over the SAME targets, which is what the stage_mask is
# for. Pass 1 reads the official diff alone: no reference download, no template stack,
# ~35 s/exposure, and it populates the survey map with real detections almost at once.
# Pass 2 revisits those same targets with a wider want_mask and runs ONLY the missing
# stationary stages. Ordering it this way means an interrupted seed batch still leaves
# a complete, browsable motion survey rather than a half-built deep one.
#
# max-batches is generous on purpose: run_batch claims `--batch` at a time and stops
# when nothing is claimable, so an over-estimate costs nothing while an under-estimate
# silently leaves the queue half-done.
echo "-- pass 1/2: motion (diff only - fast, no reference downloads)"
$PY -m ztf_data.harvest run --branches motion \
    --workers "$WORKERS" --batch 20 --max-batches 40

echo "-- pass 2/2: stationary (downloads references, stacks ONE template per cell)"
$PY -m ztf_data.harvest run --branches motion,stationary \
    --workers "$WORKERS" --batch 20 --max-batches 40

echo "-- link tracklets, per UT night"
$PY -m ztf_data.harvest link

echo "-- integrity"
$PY -m ztf_data.harvest status --check

echo "-- export static JSON for the UI"
$PY ui/build_survey.py --verify

echo
echo "=== done. serve with ./serve.sh  ->  http://localhost:8765/ui/survey.html ==="
