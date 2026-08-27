"""
Run the DeepStreaks cascade on the synthetic set.

FIRST it resolves the sl polarity empirically (does sl>0.5 mean short or long?)
by printing mean sl scores on known-short vs known-long streaks. THEN it scores
everything through rb/kd/sl and prints per-category routing.

    .venv_streaks/bin/python streak_testing/test_cascade_synthetic.py
"""
import os, sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from ztf_classification.deepstreaks import load_deepstreaks, score_families, ARCHS
from streak_testing.synthetic_streaks import make_dataset


def ensemble_mean(scores, fam, mask):
    """Mean score of family `fam` (averaged over its 3 archs) for images in mask."""
    per_arch = np.stack([scores[fam][a] for a in ARCHS])   # (3, N)
    ens = per_arch.mean(axis=0)                             # (N,) ensemble mean
    return ens[mask].mean()


def main():
    imgs, lbls = make_dataset(n_per_class=12, seed=0)
    lbls = np.array(lbls)
    print(f"scoring {len(imgs)} synthetic images through 9 models...\n")

    models = load_deepstreaks()
    scores = score_families(models, imgs)

    # --- STEP 1: resolve sl polarity ---
    short_mask = lbls == "short"
    long_mask = lbls == "long"
    sl_short = ensemble_mean(scores, "sl", short_mask)
    sl_long = ensemble_mean(scores, "sl", long_mask)
    print("=== sl polarity probe (mean ensemble sl score) ===")
    print(f"  known-SHORT streaks: sl = {sl_short:.3f}")
    print(f"  known-LONG  streaks: sl = {sl_long:.3f}")
    meaning = "sl>0.5 => LONG" if sl_long > sl_short else "sl>0.5 => SHORT"
    print(f"  => interpretation: {meaning}\n")

    # --- STEP 2: mean rb / kd / sl per category ---
    print("=== mean ensemble scores per category ===")
    print(f"{'category':10s} {'rb':>6s} {'kd':>6s} {'sl':>6s}")
    for cat in ["short", "long", "point", "noise"]:
        m = lbls == cat
        rb = ensemble_mean(scores, "rb", m)
        kd = ensemble_mean(scores, "kd", m)
        sl = ensemble_mean(scores, "sl", m)
        print(f"{cat:10s} {rb:6.3f} {kd:6.3f} {sl:6.3f}")


if __name__ == "__main__":
    main()