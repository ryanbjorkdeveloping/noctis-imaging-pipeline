"""
synthetic streak generator for the DeepStreaks smoke test.

Produces 144x144 grayscale images with KNOWN ground truth so any wrong routing is 
a wiring/preprocessing bug, not a model limitation. Four categories:

- short streak
- long streak
- point source
- pure noise

Output arrays are float32 in [0,1], shape (144,144,1) - the exact input.

Run test using: .venv_streaks/bin/python streak_testing/synthetic_streaks.py

"""

import numpy as np 

SIZE = 144

# a 144x144 LIGHT-GRAY (~0.6) background of faint Gaussian noise in [0,1].
# Real ZTF _scimref stamps are DARK streaks on a gray sky (difference-like),
# so the background must be gray, not black — this is the polarity fix.
def _blank(noise=0.05, level=0.82):
    img = np.random.normal(level, noise, (SIZE, SIZE)).astype(np.float32)
    return np.clip(img, 0, 1)

# draw a birght line of given length/angle through the image center. 
# We paint every pixel within `width` pixels of the line segment.
def _draw_streak(img, length, angle_deg, width=1.5, brightness=1.0):
    cx, cy = SIZE / 2, SIZE / 2
    ang = np.deg2rad(angle_deg)
    dx, dy = np.cos(ang), np.sin(ang)

    # sample points along the segment, from -length/2 to +length/2 about center
    for t in np.linspace(-length / 2, length / 2, int(length * 3)):
        px, py = cx + t * dx, cy + t * dy
        # stamp a small soft blob at (px,py) so the line has finite width
        x0, x1 = int(px - width - 1), int(px + width + 2)
        y0, y1 = int(py - width - 1), int(py + width + 2)
        for xi in range(max(0, x0), min(SIZE, x1)):
            for yi in range(max(0, y0), min(SIZE, y1)):
                r2 = (xi - px)**2 + (yi - py)**2
                # paint a DARK streak on the gray sky: drive the pixel DOWN toward
                # (1 - brightness), i.e. dark, instead of up toward white
                img[yi, xi] = min(img[yi, xi], 1.0 - brightness * np.exp(-r2 / (2 * width ** 2)))
    return img

# a compact Guassian blob at center - a star, not a streak.
def _point_source(brightness=1.0, sigma=2.5):
    img = _blank()
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    cx, cy = SIZE / 2, SIZE / 2
    blob = brightness * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
    return np.clip(img - blob, 0, 1)    # SUBTRACT: a dark point on the gray sky

# build the labeled synthetic set. Returns (images, labels) where 
# images is (N, 144, 144, 1) float32 and labels is a list of category strings.def
def make_dataset(n_per_class=12, seed=0):
    rng = np.random.default_rng(seed)
    images, labels = [], []

    for _ in range(n_per_class):
        # SHORT streak: ~25-40px, random angle. brightness=1.0 => streak core
        # reaches TRUE BLACK (0.0), matching the full-range real _scimref stamps.
        img = _blank()
        _draw_streak(
            img,
            length=rng.uniform(25, 40),
            angle_deg =rng.uniform(0,180),
            brightness=1.0
        )
        images.append(img); labels.append("short")

        # LONG streak: ~100-135 px (spans most of the frame)
        img = _blank()
        _draw_streak(img, length=rng.uniform(100, 135), angle_deg=rng.uniform(0, 180),
                     brightness=1.0)
        images.append(img); labels.append("long")

        # POINT source
        images.append(_point_source(brightness=1.0))
        labels.append("point")

        # NOISE / blank
        images.append(_blank(noise=rng.uniform(0.03, 0.08)))
        labels.append("noise")

    images = np.clip(np.stack(images), 0, 1).astype(np.float32)
    images = images[..., np.newaxis]          # (N,144,144) -> (N,144,144,1)
    return images, labels

# Example: make a small test set and print per-class counts
if __name__ == "__main__":
    imgs, lbls = make_dataset(n_per_class=3)
    print(f"generated {len(imgs)} images, shape {imgs.shape}, dtype {imgs.dtype}")
    print(f"value range: [{imgs.min():.3f}, {imgs.max():.3f}]  (should be ~[0,1])")
    from collections import Counter
    print("by category:", dict(Counter(lbls)))
