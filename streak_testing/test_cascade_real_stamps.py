"""
Phase 2 (real-data check) — score the ~24 GENUINE ZTF streak stamps from
DeepStreaks' own reals_zoo.png montage through the full rb/kd/sl cascade.

These are all REAL streaks (no bogus in the montage), so this validates that
the cascade correctly ACCEPTS real streaks (recall on reals) — the DeepStreaks
equivalent of validating braai on native ZTF stamps.

    .venv_streaks/bin/python streak_testing/test_cascade_real_stamps.py
"""
import os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from PIL import Image
from ztf_classification.deepstreaks import (
    load_deepstreaks, score_families, cascade_decision, ARCHS,
)

ZOO = os.path.join(PROJECT_ROOT, "DeepStreaks", "doc", "reals_zoo.png")
NCOLS, NROWS = 8, 3
MARGIN = 4          # px trimmed off each cell edge to avoid the grid lines


def slice_stamps():
    """Cut the montage into individual 144x144 grayscale stamps (real streaks)."""
    zoo = Image.open(ZOO).convert("L")
    W, H = zoo.size
    cw, ch = W // NCOLS, H // NROWS
    stamps = []
    for r in range(NROWS):
        for c in range(NCOLS):
            box = (c * cw + MARGIN, r * ch + MARGIN,
                   (c + 1) * cw - MARGIN, (r + 1) * ch - MARGIN)
            cell = zoo.crop(box).resize((144, 144), Image.BILINEAR)
            stamps.append(np.array(cell, dtype=np.float32) / 255.0)
    return np.stack(stamps)[..., np.newaxis]   # (N,144,144,1)


def main():
    stamps = slice_stamps()
    print(f"sliced {len(stamps)} real streak stamps from reals_zoo.png\n")

    models = load_deepstreaks()
    scores = score_families(models, stamps)

    # per-stamp cascade decision + tally
    routes = {}
    print(f"{'#':>2s}  {'rb':>5s} {'kd':>5s} {'sl':>5s}  route")
    for i in range(len(stamps)):
        d = cascade_decision(scores, i)
        # ensemble-max score per family, just to show the numbers
        def famscore(fam):
            return max(scores[fam][a][i] for a in ARCHS)
        print(f"{i:2d}  {famscore('rb'):5.2f} {famscore('kd'):5.2f} {famscore('sl'):5.2f}  {d['route']}")
        routes[d["route"]] = routes.get(d["route"], 0) + 1

    print("\n=== summary (these are ALL real streaks) ===")
    for route, n in sorted(routes.items(), key=lambda kv: -kv[1]):
        print(f"  {route:22s} {n}")
    # recall = fraction that pass rb (the real/bogus gate accepts them as real)
    rb_pass = sum(1 for i in range(len(stamps))
                  if any(scores["rb"][a][i] > 0.5 for a in ARCHS))
    print(f"\nrb recall on real streaks: {rb_pass}/{len(stamps)} = {rb_pass/len(stamps):.2f}")


if __name__ == "__main__":
    main()
