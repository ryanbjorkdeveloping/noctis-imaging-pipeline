import os
import sys

import cv2
import numpy as np

# Make the project root importable so `utilities` resolves when this file is run
# as a script (whose own directory — not the root — is what lands on sys.path).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utilities.image_io import run_cli  # noqa: E402


class DetectStarFeatures:
    """Score a close-up image on whether its main object is a star.

    Two feature groups, scored and weighted:
      Group 1 (weighted heavier) — main blob features:
          brightness, emission color, surface texture
      Group 2 (weighted lighter) — edge flares:
          bright plasma protrusions outside the disk

    If the combined weighted score clears a threshold, the object is
    labeled "Star". Otherwise no label is attached.
    """

    # Star decision: limb flares only. Empirically, on close-up imagery the
    # bright plasma flares at a star's limb are the ONLY appearance feature that
    # separates a star from a reflective planet/asteroid. Brightness, color and
    # texture are all confounded — a sunlit reddish cratered planet is bright,
    # warm-colored and "textured" too — so they cannot decide star-vs-planet.
    # We therefore label Star only on a clear flare signature, which no
    # reflective body can fake.
    #
    # HONEST LIMITATION: a real star with weak/absent visible flares (a quiet
    # close-up) is, by appearance alone, indistinguishable from a planet and
    # will be missed here. Separating that case needs motion/kinematics or a
    # trained model, not these OpenCV appearance features.
    FLARE_STAR_THRESHOLD = 0.30

    # Minimum disk radius (px) to score — rejects nebula specks / background
    # stars. Roundness floor for a candidate to count as a real disk.
    MIN_RADIUS = 25
    MIN_ROUNDNESS = 0.6

    def __init__(self, image_path):
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)

        # Per-object "current disk" state the feature methods read from. Set by
        # _set_current_disk() as the scoring loop steps through each candidate.
        self.disk_mask = None
        self.bbox = None
        self.disk_area = 0.0
        self.roundness = 0.0

        # Find every candidate disk (star + planets) once.
        self.candidates = self._detect_candidate_disks()

    def _detect_candidate_disks(self):
        """Find every round disk in the image: bright blobs AND Hough circles.

        No single detector catches everything here: a bright fuzzy-edged star
        is found by brightness thresholding, while darker planets are found by
        Hough circle detection. We gather both, dedupe overlaps, and keep the
        round ones above MIN_RADIUS. Each candidate is (cx, cy, radius).
        """
        candidates = []

        # Source 1 — bright contour blobs. A luminous star's core only reads as
        # a clean round disk at a HIGH threshold; at low thresholds it fuses
        # with its surrounding glow into a low-roundness mega-blob. So scan a
        # few high thresholds and keep whatever isolates as round.
        for t in (220, 200, 180):
            _, thresh = cv2.threshold(self.gray, t, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for c in contours:
                area = cv2.contourArea(c)
                (cx, cy), radius = cv2.minEnclosingCircle(c)
                if radius < self.MIN_RADIUS:
                    continue
                roundness = area / (np.pi * radius * radius) if radius else 0.0
                if roundness >= self.MIN_ROUNDNESS:
                    # The high threshold finds only the bright core; grow out to
                    # the full glowing disk so the flare annulus lands on the
                    # real limb, not inside the disk body.
                    candidates.append(self._grow_to_full_disk(cx, cy, radius))

        # Source 2 — Hough circles (catches darker round planets, and a bright
        # full-frame star whose clipped contour fails the bright pass). Grow
        # each so a found core snaps out to its full disk for correct flare
        # sampling; grow is a no-op when no larger round contour encloses it.
        blur = cv2.medianBlur(self.gray, 5)
        circles = cv2.HoughCircles(
            blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=120,
            param1=120, param2=50,
            minRadius=self.MIN_RADIUS, maxRadius=350,
        )
        if circles is not None:
            for cx, cy, radius in circles[0]:
                candidates.append(
                    self._grow_to_full_disk(float(cx), float(cy), float(radius))
                )

        return self._dedupe_circles(candidates)

    def _grow_to_full_disk(self, cx, cy, radius):
        """Expand a bright-core circle out to the full glowing disk.

        The high-threshold pass captures only a luminous star's bright core. We
        find the low-threshold (40) contour that contains this core's center and
        snap to its enclosing circle when that circle is larger and still round.
        This keeps the limb (and its flares) inside the scored disk. Falls back
        to the original circle if no better contour is found.
        """
        _, thresh = cv2.threshold(self.gray, 40, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for c in contours:
            if cv2.pointPolygonTest(c, (cx, cy), False) < 0:
                continue  # this contour doesn't contain the bright core
            area = cv2.contourArea(c)
            (fx, fy), fr = cv2.minEnclosingCircle(c)
            roundness = area / (np.pi * fr * fr) if fr else 0.0
            if fr > radius and roundness >= self.MIN_ROUNDNESS:
                return (fx, fy, fr)
        return (cx, cy, radius)

    @staticmethod
    def _dedupe_circles(circles):
        """Drop near-duplicate circles (same object found by both sources).

        Two circles whose centers are closer than the larger radius are treated
        as the same object; the larger one is kept.
        """
        kept = []
        for cx, cy, r in sorted(circles, key=lambda c: -c[2]):
            duplicate = False
            for kx, ky, kr in kept:
                dist = np.hypot(cx - kx, cy - ky)
                if dist < kr:  # center falls inside an already-kept disk
                    duplicate = True
                    break
            if not duplicate:
                kept.append((cx, cy, r))
        return kept

    def _set_current_disk(self, cx, cy, radius):
        """Point the feature methods at one candidate disk.

        Builds the filled circular mask, bbox, area, and roundness for the
        given circle so the existing detect_* methods score this object.
        """
        cx, cy, radius = int(cx), int(cy), int(radius)
        mask = np.zeros_like(self.gray)
        cv2.circle(mask, (cx, cy), radius, 255, -1)

        self.disk_mask = mask
        self.bbox = (cx - radius, cy - radius, 2 * radius, 2 * radius)
        self.disk_area = float(np.pi * radius * radius)
        # A Hough/enclosing circle is round by construction.
        self.roundness = 1.0

    # ------------------------------------------------------------------
    # Group 1 — main blob features (weighted heavier)
    # ------------------------------------------------------------------

    def detect_blob_brightness(self):
        """Score self-luminosity: a star is bright and often saturated.

        Combines mean brightness inside the disk with the fraction of
        near-saturated pixels. A reflective planet is far dimmer.
        """
        if self.disk_mask is None:
            return 0.0

        disk_pixels = self.gray[self.disk_mask == 255]
        if disk_pixels.size == 0:
            return 0.0

        # Use a high percentile, not the mean: a real star can have large dark
        # patches (coronal holes / sunspots) that drag the average down even
        # though the disk is genuinely self-luminous. The brightest regions are
        # what reveal emission.
        bright_level = np.percentile(disk_pixels, 95) / 255.0
        bright_fraction = np.mean(disk_pixels > 150)

        score = 0.6 * bright_level + 0.4 * bright_fraction
        return float(np.clip(score, 0.0, 1.0))

    def detect_blob_color(self):
        """Score emission color: one dominant, highly-saturated hue.

        A star glows a single temperature-driven color (warm red/orange or
        hot blue-white) across the whole disk. We reward a tightly
        concentrated hue with high color saturation — temperature-agnostic,
        so it works for red and blue stars alike.
        """
        if self.disk_mask is None:
            return 0.0

        hue = self.hsv[:, :, 0][self.disk_mask == 255]
        sat = self.hsv[:, :, 1][self.disk_mask == 255]
        if hue.size == 0:
            return 0.0

        # Hue concentration: what fraction of pixels fall in the dominant
        # hue band. Histogram over OpenCV's 0-179 hue range, 18 bins (10 deg).
        hist = cv2.calcHist([hue.astype(np.uint8)], [0], None, [18], [0, 180])
        hue_concentration = float(hist.max() / hist.sum())

        mean_saturation = sat.mean() / 255.0

        score = 0.6 * hue_concentration + 0.4 * mean_saturation
        return float(np.clip(score, 0.0, 1.0))

    def detect_surface_texture(self):
        """Score surface granulation via high-frequency texture.

        A star's convection cells give the disk a mottled texture, which
        shows up as high Laplacian variance. A smooth/flat disk scores low.
        """
        if self.disk_mask is None:
            return 0.0

        # Measure texture over the disk INTERIOR only. Eroding the mask pulls
        # the sample away from the bright limb edge, so the score reflects
        # genuine surface granulation rather than the disk's outer boundary.
        # (The close-up gate already rejects sparse dot-fields, which would
        # otherwise inflate Laplacian variance with point-to-black edges.)
        kernel = np.ones((15, 15), np.uint8)
        interior = cv2.erode(self.disk_mask, kernel, iterations=1)

        laplacian = cv2.Laplacian(self.gray, cv2.CV_64F)
        texture = laplacian[interior == 255]
        if texture.size == 0:
            return 0.0

        variance = texture.var()
        # Measured interior Laplacian variance on a real granular star disk is
        # ~50, so ~60 marks a "clearly textured" level for an 8-bit image.
        score = variance / 60.0
        return float(np.clip(score, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Group 2 — edge flares (weighted lighter)
    # ------------------------------------------------------------------

    def detect_edge_flares(self):
        """Score bright plasma flares/prominences at the disk's limb.

        Flares and prominences sit on the bright outer edge of the disk, not
        in empty space beyond it (a tight bounding box already hugs the glowing
        limb). So we look in the outer annulus of the disk itself for very
        bright spots. Planets never show this; stars do.
        """
        if self.disk_mask is None:
            return 0.0

        x, y, w, h = self.bbox
        cx, cy = x + w // 2, y + h // 2
        radius = max(w, h) // 2

        # Sample the annulus at the BRIGHT-CORE limb (0.62-0.78 R), not the
        # outer edge of the grown disk. The disk is grown out to its full glow,
        # which buries a star's flares well inside the geometric edge; measuring
        # the outer rim then reads black for stars and bright-background for
        # planets. At the core limb, a star's plasma flares show as bright spots
        # while reflective planets/asteroids have nothing bright in this band —
        # this is what actually separates them (star ~0.015 fraction here vs
        # planets/asteroids <=0.003-0.014).
        annulus = np.zeros_like(self.gray)
        cv2.circle(annulus, (cx, cy), int(radius * 0.78), 255, -1)
        cv2.circle(annulus, (cx, cy), int(radius * 0.62), 0, -1)

        annulus_pixels = self.gray[annulus == 255]
        if annulus_pixels.size == 0:
            return 0.0

        # Flares are the very brightest pixels along the core limb.
        flare_fraction = np.mean(annulus_pixels > 230)
        # Scaled so a real-but-small flare patch clears the star threshold while
        # planet/asteroid limbs (near-zero in this band) stay below it.
        score = flare_fraction * 20.0
        return float(np.clip(score, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _score_current_disk(self):
        """Score the currently-set disk and decide star / not via limb flares.

        Labels Star only when the limb-flare score clears FLARE_STAR_THRESHOLD —
        the one appearance feature that separates a self-luminous star from a
        reflective body. The other features are still computed and returned for
        transparency/inspection, but they do not drive the decision (they are
        confounded between stars and sunlit planets). Assumes _set_current_disk().
        """
        features = {
            "brightness": self.detect_blob_brightness(),
            "color": self.detect_blob_color(),
            "texture": self.detect_surface_texture(),
            "edge_flares": self.detect_edge_flares(),
        }

        flare_score = features["edge_flares"]
        is_star = flare_score >= self.FLARE_STAR_THRESHOLD
        label = "Star" if is_star else None

        # Confidence: the PRESENCE of a clear flare is the star signal, not how
        # much of the limb it covers (a compact flare is just as conclusive as a
        # full ring — planets don't flare at all). So once flare clears the
        # threshold, report high confidence in an 0.85-1.0 band rather than the
        # raw (often small) flare fraction. Below threshold, report the raw low
        # value so non-stars stay honestly low.
        if is_star:
            span = 1.0 - self.FLARE_STAR_THRESHOLD
            confidence = 0.85 + 0.15 * (flare_score - self.FLARE_STAR_THRESHOLD) / span
        else:
            confidence = flare_score

        return {
            "features": features,
            "combined_score": float(np.clip(confidence, 0.0, 1.0)),
            "label": label,
        }

    def scoring_all(self):
        """Score every candidate disk in the image.

        Returns a list of per-object results, each carrying its (cx, cy, radius)
        circle plus the feature scores, combined score, and label.
        """
        results = []
        for cx, cy, radius in self.candidates:
            self._set_current_disk(cx, cy, radius)
            result = self._score_current_disk()
            result["circle"] = (int(cx), int(cy), int(radius))
            results.append(result)
        return results

    def draw_results(self, results, output_dir="output_images"):
        """Annotate every scored object on one copy of the image and save it.

        Each object gets a box around its disk (green if labeled "Star", red
        otherwise) plus its score. Writes to output_dir using the input
        filename with a "_star_detected" suffix. Returns the output path.
        """
        output = self.image.copy()

        for result in results:
            is_star = result["label"] == "Star"
            color = (0, 255, 0) if is_star else (0, 0, 255)  # BGR green / red
            cx, cy, radius = result["circle"]
            x, y = cx - radius, cy - radius
            cv2.rectangle(output, (x, y), (cx + radius, cy + radius), color, 2)

            label_text = "Star" if is_star else "Not star"
            score_text = f"{label_text} {result['combined_score']:.2f}"
            cv2.putText(output, score_text, (x, max(y - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        os.makedirs(output_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(self.image_path))[0]
        output_path = os.path.join(output_dir, f"{stem}_star_detected.jpg")
        cv2.imwrite(output_path, output)
        return output_path


def process_image(path):
    """Score every object in one image, print results, and save the output."""
    detector = DetectStarFeatures(path)
    results = detector.scoring_all()

    print(f"\nImage: {path}  ({len(results)} object(s) found)")
    for i, result in enumerate(results):
        cx, cy, r = result["circle"]
        print(f"  Object {i} @ ({cx},{cy}) r={r}: "
              f"score={result['combined_score']:.3f} -> {result['label']}")

    output_path = detector.draw_results(results)
    print(f"  Saved annotated result to: {output_path}")
    return results


if __name__ == "__main__":
    run_cli(process_image)
