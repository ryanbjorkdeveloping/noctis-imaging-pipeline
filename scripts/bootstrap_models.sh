#!/usr/bin/env bash
# Fetch the pretrained third-party models the pipeline runs on.
#
#   ./scripts/bootstrap_models.sh [braai|deepstreaks|lc|all]
#
# These are NOT vendored into this repo. They are separate MIT-licensed projects
# with their own git history, and cloning them (rather than copying the files in)
# is what keeps their authorship and licensing intact. See README > Attribution.
#
# NO TRAINING happens here or anywhere in this project — every model is used as
# published by its authors.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
WHAT="${1:-all}"

clone() {   # clone <dir> <url> <approx-size> <what-it-is>
    local dir="$1" url="$2" size="$3" desc="$4"
    if [ -d "$dir/.git" ]; then
        echo "  ✓ $dir already present — skipping"
        return 0
    fi
    if [ -e "$dir" ]; then
        echo "  ! $dir exists but is not a git clone. Move it aside and re-run." >&2
        return 1
    fi
    echo "  → cloning $dir ($desc, ~$size)"
    git clone --depth 1 "$url" "$dir"
}

verify() {  # verify <label> <path...>
    local label="$1"; shift
    local missing=0
    for p in "$@"; do
        [ -e "$p" ] || { echo "  ✗ MISSING: $p" >&2; missing=1; }
    done
    [ "$missing" -eq 0 ] && echo "  ✓ $label verified"
    return "$missing"
}

if [ "$WHAT" = "braai" ] || [ "$WHAT" = "all" ]; then
    echo "== braai (real/bogus CNN, Duev et al. 2019) =="
    clone braai https://github.com/dmitryduev/braai.git 34MB "VGG6 real/bogus"
    verify "braai weights" "$ROOT/braai/models/braai_d6_m9.weights.h5"
fi

if [ "$WHAT" = "deepstreaks" ] || [ "$WHAT" = "all" ]; then
    echo "== DeepStreaks (streak/fast-mover cascade, Duev et al. 2019) =="
    echo "   NOTE: ~600MB — this is the big one."
    clone DeepStreaks https://github.com/dmitryduev/DeepStreaks.git 600MB "9 pretrained CNNs"
    verify "DeepStreaks models" \
        "$ROOT/DeepStreaks/service/models" \
        "$ROOT/DeepStreaks/service/code/config.json"
    n=$(find "$ROOT/DeepStreaks/service/models" -name '*.weights.h5' 2>/dev/null | wc -l | tr -d ' ')
    echo "  → found $n weight files (expected 9: 3 families x 3 architectures)"
fi

if [ "$WHAT" = "lc" ] || [ "$WHAT" = "all" ]; then
    echo "== lc_classifier (ALeRCE light-curve RF) — OPTIONAL, for lc_testing/ only =="
    clone lc_classifier https://github.com/alercebroker/lc_classifier.git 400MB "hierarchical RF"
    echo "  → then: .venv_lc/bin/pip install -e lc_classifier/ --no-deps"
    echo "  → model pickles download on first use via .download_model()"
fi

cat <<'NOTE'

Not fetched — no public weights exist:
  stamp_classifier_model/  The ALeRCE 2020 stamp-classifier TF1 checkpoint used by
                           stamp_testing/pathA_*.py. It was vendored from the
                           alercebroker/pipeline monorepo; the current public
                           release has no runnable checkpoint. Those Path A
                           scripts will not run. The Path B route
                           (ztf_classification/stamp_classifier.py, which reads
                           ALeRCE's hosted classifier over HTTP) works and is what
                           the main pipeline uses.
NOTE
echo "done."
