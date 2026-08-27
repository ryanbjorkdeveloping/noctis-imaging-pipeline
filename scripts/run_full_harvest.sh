#!/usr/bin/env bash
# Full-coverage orchestration run: finish everything already queued, expand coverage
# across several new dates/fields, classify everything, link, and rebuild every UI
# export. Logs to run_full_harvest.log. Safe to re-run - every step is idempotent
# (harvest claim/COALESCE-upsert, classify's ALeRCE cache).
#
# MOTION-FIRST BY DESIGN (2026-08-17): the stationary/deep-reference branch is hitting a
# reference-sciimg download wall RIGHT NOW (verified: same fields/years downloaded fine
# in past sessions, so this is transient/rate-limit-shaped, not a permanent data gap).
# Burning every target's 3-attempt budget against a live external issue would be wasteful
# and would wrongly exhaust retries that should be spent once the condition clears. So
# this script does motion + classify (which need no reference downloads at all - classify
# reads native ALeRCE stamps over the network, not local sciimg) as the guaranteed-useful
# core, and makes the stationary/deep-reference pass ONE bounded, best-effort attempt at
# the end that never blocks anything else and is safe to re-run later on its own via:
#   .venv/bin/python -m ztf_data.harvest run --branches motion,stationary --workers 2 \
#     --batch 20 --max-batches 200
set -uo pipefail
cd "$(dirname "$0")/.."   # scripts/ -> repo root
PY=.venv/bin/python
WORKERS=2
CLASSIFY_WORKERS=4

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "=== preflight: IRSA reachability ==="
if ! $PY - <<'EOF'
import sys
import config  # noqa: F401
from ztf_data.epochs import query_grid, IRSAUnavailable
try:
    _, e = query_grid(468, 3, 2, 2)
    print(f"IRSA OK - {len(e)} epochs for probe cell")
except IRSAUnavailable as exc:
    print(f"IRSA UNAVAILABLE: {exc}"); sys.exit(1)
except Exception as exc:
    print(f"IRSA probe failed: {type(exc).__name__}: {exc}"); sys.exit(1)
EOF
then
    log "IRSA is down. Aborting expansion; will still finish local/cached work."
    IRSA_OK=0
else
    IRSA_OK=1
fi

log "=== step 0: init store (idempotent) ==="
$PY -m ztf_data.harvest init

log "=== step 1: finish whatever motion work is already queued ==="
$PY -m ztf_data.harvest run --branches motion --workers $WORKERS --batch 20 --max-batches 40

if [ "$IRSA_OK" = "1" ]; then
    log "=== step 2: expand coverage - 6 new dates spread across different sky regions ==="
    for DATE in 2018-06-15 2019-09-20 2020-11-10 2021-03-05 2022-08-22 2023-12-01; do
        log "-- enqueue 15x12 spread on $DATE (motion)"
        $PY -m ztf_data.harvest enqueue --date "$DATE" --cadence 15x12 --spread --branches motion || log "   enqueue failed for $DATE (continuing)"
    done

    log "=== step 3: motion pass over everything newly enqueued (diff-only, fast) ==="
    $PY -m ztf_data.harvest run --branches motion --workers $WORKERS --batch 20 --max-batches 200
else
    log "=== step 2-3 skipped: IRSA unavailable, no new downloads possible ==="
fi

log "=== step 4: link tracklets per UT night ==="
$PY -m ztf_data.harvest link

log "=== step 5: classify EVERY point-source candidate, uncapped (object typing, no local downloads needed) ==="
$PY -m ztf_data.classify --per-cell 1000000 --min-snr 3.0 --workers $CLASSIFY_WORKERS --sleep 0.15

log "=== step 6: integrity check ==="
$PY -m ztf_data.harvest status --check

log "=== step 7: ONE bounded, best-effort stationary attempt (deep reference per cell) ==="
log "    (does not block anything above; safe/cheap to re-run alone later if this still fails)"
$PY -m ztf_data.harvest run --branches motion,stationary --workers $WORKERS --batch 10 --max-batches 5 || log "    stationary pass hit errors (expected if the download wall is still up) - continuing"

log "=== step 8: rebuild UI exports (uses whatever stationary data exists) ==="
$PY ui/build_survey.py --verify
$PY ui/build_cell_stacks.py
$PY ui/build_cell_fits.py

log "=== step 9: final status ==="
$PY -m ztf_data.harvest status

log "=== DONE ==="
