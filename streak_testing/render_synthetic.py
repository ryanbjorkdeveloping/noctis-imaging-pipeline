"""
render one of each synthetic category to a PNG so we can
confirm the streaks/points/noise actually look right before trusting scores.

    .venv_streaks/bin/python streak_testing/render_synthetic.py
"""
import os, sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")               # headless: write a file, don't open a window
import matplotlib.pyplot as plt
from streak_testing.synthetic_streaks import make_dataset

CATEGORIES = ["short", "long", "point", "noise"]
OUT = os.path.join(PROJECT_ROOT, "ztfdata", "synthetic_streaks_preview.png")


def main():
    imgs, lbls = make_dataset(n_per_class=3, seed=1)

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    for ax, cat in zip(axes, CATEGORIES):
        # grab the first image of this category
        idx = lbls.index(cat)
        ax.imshow(imgs[idx, :, :, 0], cmap="gray", origin="lower")
        ax.set_title(cat)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(OUT, dpi=100)
    print(f"saved preview -> {OUT}")


if __name__ == "__main__":
    main()
