import os
import sys

import cv2
import numpy as np

# Make the project root importable so `utilities` resolves when this file is run
# as a script (whose own directory — not the root — is what lands on sys.path).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utilities.image_io import run_cli  # noqa: E402


class DetectAsteroidFeatures:
    """Score a close-up image on whether each object reads as an asteroid.

    Standalone counterpart to DetectStarFeatures / DetectPlanetFeatures. This
    file answers ONE question per object: "how asteroid-like is this?" and, if
    the score clears a threshold, labels it "Asteroid" with a confidence; below
    threshold no label/highlight is attached.

    An asteroid is characterized (visually, in a single still) by the four
    signals the user named, weighted by how well each one actually SEPARATES an
    asteroid from the other bodies in this project's imagery (planets, the Sun,
    background stars):

      1. detect_color_being_gray (the cleanest) — most asteroids are bare gray
         rock with very low color saturation. Planets are colored (blue/red/
         orange) and the Sun is a strong orange, so low saturation excludes
         them with a wide margin (measured Vesta saturation ~0.00 vs Sun ~0.96).
      2. detect_rocky_surface    — the rocky, pitted terrain gives the disk
         interior very high high-frequency texture variance (Vesta ~825 vs a
         smooth Sun disk ~54). A strong, clean discriminator on this imagery.
      3. detect_rugged_edge      — an asteroid is a lumpy rock, not a perfect
         sphere, so its limb deviates from a circle more than a planet's clean
         round limb. Measured as relative radial-distance variance of the
         silhouette. NOTE: this is a comparatively THIN signal — Vesta is a
         fairly round asteroid (limb roughness ~0.08 vs a true circle ~0.00 and
         the Sun's flaring limb ~0.06) — so it is weighted modestly, not as the
         lead feature. It still earns its keep on genuinely jagged rocks.
      4. detect_craters          — small circular crater rims dotting the
         interior, counted via Hough. A positive asteroid cue, but solar
         granulation also produces Hough circles, so it is corroborating only
         and the lightest-weighted.

    Combined into a single asteroid_score in [0, 1]:
        asteroid_score = W_GRAY    * color_being_gray
                       + W_ROCKY   * rocky_surface
                       + W_RUGGED  * rugged_edge
                       + W_CRATERS * craters
    """

    # Feature weights, ordered by how cleanly each separates an asteroid from a
    # planet/star on this project's imagery (measured on the samples). Gray color
    # and rocky surface are the two wide-margin discriminators and lead. Rugged
    # edge is real but thin on a near-spherical asteroid, so it is modest. Craters
    # corroborate but are noisy (solar granulation fakes them), so lightest.
    W_GRAY = 0.40
    W_ROCKY = 0.30
    W_RUGGED = 0.18
    W_CRATERS = 0.12

    # An object is labeled "Asteroid" only when its combined score clears this.
    # Calibrated so the cratered, gray, irregular Vesta sample clears it while
    # round/colored planets and the flaring Sun fall below.
    ASTEROID_THRESHOLD = 0.55

    # Minimum object radius (px) to score — rejects nebula specks / background
    # stars. We do NOT gate on roundness here (the planet detector does): an
    # asteroid is deliberately allowed to be irregular, so a roundness floor
    # would reject the very objects we want.
    MIN_RADIUS = 25

    # Rugged-edge scoring. We measure the silhouette's relative limb roughness:
    # the standard deviation of the radial distance from the centroid to each
    # boundary point, divided by the mean radius. A perfect circle is ~0.00; the
    # Sun's flaring limb ~0.06; the (fairly round) Vesta sample ~0.08; a jagged
    # rock is higher still. ROUGH_SMOOTH is the value at/below which the limb is
    # "smooth" (ruggedness 0); ROUGH_JAGGED at/above which it is "fully rugged"
    # (ruggedness 1). The window is deliberately set so a flaring star limb
    # (~0.06) reads as smooth while Vesta's bumpier limb (~0.08) starts to score.
    ROUGH_SMOOTH = 0.065
    ROUGH_JAGGED = 0.20

    # Footprint solidity (fill of its bounding circle) required to be a real,
    # compact body. A cratered asteroid or a lone disk fills ~0.80; a sprawling
    # multi-object scene whose objects merge into one threshold blob fills far
    # less (Sun-scene ~0.35, observatory/Milky-Way scene ~0.53), so a 0.6 floor
    # rejects those merged-scene blobs while keeping genuine bodies.
    MIN_FILL = 0.6

    # Gray-color scoring. Mean HSV saturation (0-1) at/below GRAY_SAT_LOW reads
    # as fully gray (score 1); at/above GRAY_SAT_HIGH as fully colored (score 0).
    GRAY_SAT_LOW = 0.12
    GRAY_SAT_HIGH = 0.40

    # Crater scoring. Craters are detected as small circles inside the body via
    # Hough; their count is normalized by this to [0, 1]. A handful of clear
    # crater rims is already a strong asteroid signal.
    CRATER_COUNT_NORM = 8.0

    # Rocky-surface scoring. Interior Laplacian variance normalized by this; a
    # pitted rock measures high, a smooth planet low.
    TEXTURE_NORM = 120.0

    def __init__(self, image_path):
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)

        # Per-object "current object" state the feature methods read from. Set by
        # _set_current_object() as the scoring loop steps through each candidate.
        self.mask = None          # filled footprint mask of the object
        self.bbox = None
        self.center = None
        self.radius = 0
        self.roughness = 0.0      # measured limb roughness (carried in)

        # Find every candidate object once.
        self.candidates = self._detect_candidate_objects()

    # ------------------------------------------------------------------
    # Candidate detection (object finding, roundness-agnostic)
    # ------------------------------------------------------------------

    def _detect_candidate_objects(self):
        """Find every solid object sitting against the dark sky.

        Unlike the star/planet detectors, we do NOT use Hough circles or a
        roundness gate to FIND objects — an asteroid is irregular and would be
        rejected by both. Instead we threshold the object away from the black
        background and take every sufficiently-large, sufficiently-solid external
        contour. Each candidate carries the contour itself plus its measured limb
        roughness, so detect_rugged_edge can score the real boundary shape (not a
        fitted circle).

        Returns a list of (contour, cx, cy, radius, roughness).
        """
        # Separate any lit object from black space. A low threshold captures the
        # full body including its darker (shadowed) limb, so the outline reflects
        # the true silhouette rather than only the brightest patch.
        _, thresh = cv2.threshold(self.gray, 30, 255, cv2.THRESH_BINARY)
        # Close small gaps (crater shadows reaching the limb) so a single body
        # stays one contour instead of fragmenting.
        kernel = np.ones((7, 7), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        candidates = []
        for c in contours:
            (cx, cy), radius = cv2.minEnclosingCircle(c)
            if radius < self.MIN_RADIUS:
                continue
            # Ignore wispy nebula/glow smears and merged multi-object scenes: a
            # real compact body fills most of its bounding circle, a diffuse or
            # merged blob does not.
            fill = cv2.contourArea(c) / (np.pi * radius * radius) if radius else 0.0
            if fill < self.MIN_FILL:
                continue
            roughness = self._limb_roughness(c, cx, cy)
            candidates.append((c, float(cx), float(cy), float(radius), roughness))

        return self._dedupe(candidates)

    @staticmethod
    def _limb_roughness(contour, cx, cy):
        """Relative limb roughness: radial-distance std / mean from the center.

        Walk the silhouette's boundary points and measure each point's distance
        to the object center; a perfect circle keeps that distance constant
        (roughness ~0.00), while a lumpy rock's limb wanders in and out
        (roughness rises). Dividing by the mean radius makes it scale-invariant.
        This captures genuine boundary bumpiness — unlike a global circularity,
        which on a near-spherical asteroid like Vesta barely differs from a disk.
        """
        pts = contour.reshape(-1, 2).astype(np.float64)
        d = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
        mean = d.mean()
        return float(d.std() / mean) if mean else 0.0

    @staticmethod
    def _dedupe(candidates):
        """Drop near-duplicate objects, keeping the largest at each location."""
        kept = []
        for cand in sorted(candidates, key=lambda c: -c[3]):
            _, cx, cy, r, _ = cand
            duplicate = False
            for _, kx, ky, kr, _ in kept:
                if np.hypot(cx - kx, cy - ky) < kr:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(cand)
        return kept

    def _set_current_object(self, contour, cx, cy, radius, roughness):
        """Point the feature methods at one candidate object.

        Builds a filled footprint mask from the actual contour (not a circle, so
        the lumpy silhouette is sampled faithfully), plus bbox/center/radius and
        the carried-in limb roughness.
        """
        cx, cy, radius = int(cx), int(cy), int(radius)
        mask = np.zeros_like(self.gray)
        cv2.drawContours(mask, [contour], -1, 255, -1)

        x, y, w, h = cv2.boundingRect(contour)
        self.mask = mask
        self.bbox = (x, y, w, h)
        self.center = (cx, cy)
        self.radius = radius
        self.roughness = float(roughness)

    # ------------------------------------------------------------------
    # Asteroid features (each returns [0, 1])
    # ------------------------------------------------------------------

    def detect_rugged_edge(self):
        """Score outline ruggedness: high for a lumpy rock, low for a sphere.

        Maps the silhouette's relative limb roughness onto [0, 1]: at/below
        ROUGH_SMOOTH the limb is a clean round edge (score 0); at/above
        ROUGH_JAGGED it is fully rugged (score 1); linear in between. A planet or
        star presents a smooth round limb (low roughness) while a rocky asteroid
        wanders off the circle. This is a real but thin signal on a near-
        spherical asteroid, so it is weighted modestly (see W_RUGGED).
        """
        if self.mask is None:
            return 0.0
        span = self.ROUGH_JAGGED - self.ROUGH_SMOOTH
        score = (self.roughness - self.ROUGH_SMOOTH) / span if span else 0.0
        return float(np.clip(score, 0.0, 1.0))

    def detect_color_being_gray(self):
        """Score how gray (desaturated) the object is: high for bare rock.

        Most asteroids are gray rock with low color saturation; planets are
        colored and the Sun is a strong orange. We take mean HSV saturation over
        the footprint interior and map it inversely: at/below GRAY_SAT_LOW it is
        fully gray (score 1), at/above GRAY_SAT_HIGH fully colored (score 0).
        """
        if self.mask is None:
            return 0.0
        sat = self.hsv[:, :, 1][self.mask == 255]
        if sat.size == 0:
            return 0.0
        mean_sat = sat.mean() / 255.0
        span = self.GRAY_SAT_HIGH - self.GRAY_SAT_LOW
        score = (self.GRAY_SAT_HIGH - mean_sat) / span if span else 0.0
        return float(np.clip(score, 0.0, 1.0))

    def detect_craters(self):
        """Score crater presence: high when many small round pits dot the surface.

        Craters appear as small circular rims inside the body. We run Hough on
        the disk interior to count them and normalize by CRATER_COUNT_NORM. Only
        circles whose centers fall inside the (eroded) footprint count, so limb
        and background structure are excluded. A smooth planet yields ~none.
        """
        if self.mask is None:
            return 0.0

        # Restrict crater search to the body interior (away from the limb edge).
        kernel = np.ones((9, 9), np.uint8)
        interior = cv2.erode(self.mask, kernel, iterations=1)
        if cv2.countNonZero(interior) == 0:
            return 0.0

        body = cv2.bitwise_and(self.gray, self.gray, mask=interior)
        blurred = cv2.medianBlur(body, 5)

        # Crater radii scale with the body; bound them to a sensible fraction.
        min_r = max(4, int(self.radius * 0.04))
        max_r = max(min_r + 1, int(self.radius * 0.30))
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=max(8, min_r * 2),
            param1=100, param2=28,
            minRadius=min_r, maxRadius=max_r,
        )
        if circles is None:
            return 0.0

        count = 0
        for cx, cy, _ in circles[0]:
            xi, yi = int(cx), int(cy)
            if 0 <= yi < interior.shape[0] and 0 <= xi < interior.shape[1]:
                if interior[yi, xi] == 255:
                    count += 1

        return float(np.clip(count / self.CRATER_COUNT_NORM, 0.0, 1.0))

    def detect_rocky_surface(self):
        """Score rocky/pitted surface via high-frequency interior texture.

        Rocky terrain and crater shadows give the disk interior high Laplacian
        variance. Sampled over the ERODED footprint so the bright limb edge does
        not inflate the measure — the score reflects genuine surface roughness,
        not the boundary. A smooth planet scores low.
        """
        if self.mask is None:
            return 0.0
        kernel = np.ones((15, 15), np.uint8)
        interior = cv2.erode(self.mask, kernel, iterations=1)
        laplacian = cv2.Laplacian(self.gray, cv2.CV_64F)
        texture = laplacian[interior == 255]
        if texture.size == 0:
            return 0.0
        score = texture.var() / self.TEXTURE_NORM
        return float(np.clip(score, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def scoring(self):
        """Score the currently-set object for asteroid-ness and decide the label.

        Combines the four weighted features into asteroid_score and labels
        "Asteroid" iff it clears ASTEROID_THRESHOLD. Below threshold, label is
        None so no highlight is drawn. Assumes _set_current_object() has been
        called for the object being scored.
        """
        features = {
            "rugged_edge": self.detect_rugged_edge(),
            "color_being_gray": self.detect_color_being_gray(),
            "craters": self.detect_craters(),
            "rocky_surface": self.detect_rocky_surface(),
        }

        asteroid_score = (
            self.W_RUGGED * features["rugged_edge"]
            + self.W_GRAY * features["color_being_gray"]
            + self.W_CRATERS * features["craters"]
            + self.W_ROCKY * features["rocky_surface"]
        )
        asteroid_score = float(np.clip(asteroid_score, 0.0, 1.0))

        is_asteroid = asteroid_score >= self.ASTEROID_THRESHOLD
        return {
            "features": features,
            "asteroid_score": asteroid_score,
            "label": "Asteroid" if is_asteroid else None,
        }

    def scoring_all(self):
        """Score every candidate object in the image.

        Returns a list of per-object results, each carrying its (cx, cy, radius)
        circle and bbox plus the feature sub-scores, asteroid_score, and label.
        """
        results = []
        for contour, cx, cy, radius, roughness in self.candidates:
            self._set_current_object(contour, cx, cy, radius, roughness)
            result = self.scoring()
            result["circle"] = (int(cx), int(cy), int(radius))
            result["bbox"] = self.bbox
            results.append(result)
        return results

    def draw_results(self, results, output_dir="output_images"):
        """Annotate asteroid detections on one copy of the image and save it.

        Only objects labeled "Asteroid" are highlighted (cyan-gray box) with
        their confidence; non-asteroids get no highlight, per the requirement.
        Writes to output_dir using the input filename with an "_asteroid_detected"
        suffix. Returns the output path.
        """
        output = self.image.copy()

        for result in results:
            if result["label"] != "Asteroid":
                continue  # no highlight for non-asteroids
            color = (200, 200, 60)  # BGR — cyan-gray box for asteroid
            x, y, w, h = result["bbox"]
            cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
            score_text = f"Asteroid {result['asteroid_score']:.2f}"
            cv2.putText(output, score_text, (x, max(y - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        os.makedirs(output_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(self.image_path))[0]
        output_path = os.path.join(output_dir, f"{stem}_asteroid_detected.jpg")
        cv2.imwrite(output_path, output)
        return output_path


def process_image(path):
    """Score every object in one image, print results, and save the output."""
    detector = DetectAsteroidFeatures(path)
    results = detector.scoring_all()

    asteroids = [r for r in results if r["label"] == "Asteroid"]
    print(f"\nImage: {path}  ({len(results)} object(s) found, "
          f"{len(asteroids)} asteroid(s))")
    for i, result in enumerate(results):
        cx, cy, r = result["circle"]
        f = result["features"]
        label = result["label"] or "not asteroid"
        print(f"  Object {i} @ ({cx},{cy}) r={r}: "
              f"score={result['asteroid_score']:.3f} -> {label}")
        print(f"      rugged={f['rugged_edge']:.3f} "
              f"gray={f['color_being_gray']:.3f} "
              f"craters={f['craters']:.3f} "
              f"rocky={f['rocky_surface']:.3f}")

    output_path = detector.draw_results(results)
    print(f"  Saved annotated result to: {output_path}")
    return results


if __name__ == "__main__":
    run_cli(process_image)
