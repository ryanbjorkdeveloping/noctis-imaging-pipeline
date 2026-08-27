import os
import sys

import cv2
import numpy as np

# Make the project root importable so `utilities` resolves when this file is run
# as a script (whose own directory — not the root — is what lands on sys.path).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utilities.image_io import run_cli  # noqa: E402


class DetectCometFeatures:
    """Score a close-up image on whether each object reads as a comet.

    Standalone counterpart to DetectStarFeatures / DetectPlanetFeatures /
    DetectAsteroidFeatures. This file answers ONE question per object: "could
    this be a comet?" and, if the combined score clears a threshold, labels it
    "Comet" with a confidence; below threshold no label/highlight is attached.

    A comet is fundamentally different from the other three classes. A star,
    planet, or asteroid is a COMPACT, roughly-ROUND blob; a comet is a
    DIRECTIONAL STREAK WITH A BRIGHTNESS GRADIENT — a bright nucleus/head at one
    end and a fading tail "path" trailing off the other. The user's model:
    "a pathway, a bright-line shape, fading at one part while bright on the other
    end, the faded part following the bright end like a path."

    That shape gives four geometric signals, none of which a round body fakes:

      1. detect_bright_line_shape (ELONGATION, the lead) — the object is a long
         line, not a blob. major_axis / minor_axis of the fitted box. Every
         star/planet/asteroid dot is ~round (ratio ~1), so elongation alone
         rejects them.
      2. detect_fading_end (BRIGHTNESS FALLOFF) — brightness decreases along the
         long axis. A satellite/star trail is uniform along its length; a comet
         fades head -> tail. This is the false-positive killer for non-comet
         streaks.
      3. detect_pathway (LINEARITY) — the lit pixels hug ONE axis as a coherent
         path, rather than being a sprawling/forked clump. Oriented-box fill.
      4. detect_bright_end (HEAD-TAIL POLARITY) — the brightness is concentrated
         at ONE end (the head) with the fade trailing off the other; "the faded
         part follows the bright end." Bright-end vs faded-end imbalance.

    Combined into a single comet_score in [0, 1]:
        comet_score = W_BRIGHT_LINE * bright_line_shape
                    + W_FADING_END  * fading_end
                    + W_PATHWAY     * pathway
                    + W_BRIGHT_END  * bright_end

    HONEST LIMITATIONS:
      - Calibrated on a SINGLE positive (comet_sighting.jpg, a bright classic
        comet) plus the project's negatives. Thresholds may not generalize to
        faint or oddly-oriented comets.
      - A faint, tail-less comet collapses to a fuzzy dot in a still image and
        these geometric features have nothing to grab — that case needs
        multi-frame motion, not single-image appearance.
      - Color (green/blue coma) and texture are intentionally NOT load-bearing
        here (both are exposure/processing dependent), mirroring how the
        asteroid detector treats craters.
    """

    # Feature weights, ordered by how cleanly each separates a comet from the
    # round bodies and from uniform streaks. Elongation leads (it alone rejects
    # every round dot); the brightness falloff is the discriminator against
    # uniform trails; linearity and head-tail polarity corroborate.
    # The CLEAN MONOTONIC FADE from a bright head (bright_end + fading_end) is the
    # comet's defining, hardest-to-fake signature, so those two lead. Elongation
    # confirms it is a streak not a glow; pathway corroborates (a round glow also
    # scores high on pathway, so it is the weakest discriminator and weighted
    # lightest). Calibrated so the comet (fade 1.0, monotonic 1.0) clears the
    # 0.55 threshold while bright glows / partial streaks (fade & monotonic both
    # ~0.6-0.8) fall short.
    W_BRIGHT_END = 0.40    # head-tail clean monotonic falloff (the lead)
    W_FADING_END = 0.30    # total brightness drop head -> tail
    W_BRIGHT_LINE = 0.20   # elongation (streak vs glow)
    W_PATHWAY = 0.10       # linearity (hugs one axis)

    # An object is labeled "Comet" only when its combined score clears this.
    # Calibrated with a wide margin: the comet sample scores 1.000 while the
    # closest non-comet (an elongated bright trail with only a partial fade)
    # tops out ~0.49, so 0.60 cleanly separates them.
    COMET_THRESHOLD = 0.60

    # Minimum object enclosing radius (px) to score — drops background star dots
    # and noise specks that survive thresholding. We do NOT gate on roundness
    # (the sibling detectors do): a comet is deliberately elongated, so a
    # roundness floor would reject the very object we want.
    MIN_RADIUS = 18

    # Candidate detection knobs (calibrated below).
    # A MID threshold isolates the comet's bright SPINE (head + brightest tail
    # core), which is elongated, rather than the wide diffuse glow a low
    # threshold grabs (which fuses into a near-round fat blob and destroys the
    # elongation signal). Measured on comet_sighting.jpg: th=28 -> elong 1.36
    # (round blob); th~90 -> elong ~2.85 (the bright path emerges).
    THRESH = 90
    # A small morphological-close kernel only bridges tiny gaps along the spine;
    # it must NOT be large or it re-fuses the comet back into a round blob.
    CLOSE_KERNEL = 5

    # Elongation scoring. major/minor axis ratio. At/below ELONGATION_MIN the
    # object is round (score 0); at/above ELONGATION_FULL it is a clear streak
    # (score 1). Deliberately a STEEP, early-saturating ramp: it acts as a near
    # binary "is this a streak at all?" gate, NOT a magnitude. Once an object is
    # clearly elongated, MORE elongation must not let it outrank a true comet on
    # raw length — among streaks, the FADE QUALITY (fading_end x bright_end)
    # ranks them. The comet's spine reads ~2.5; a thin uniform trail reads
    # higher but is killed by its weak fade, not by losing this ramp.
    ELONGATION_MIN = 1.8
    ELONGATION_FULL = 2.6

    # Pathway/linearity scoring. Fraction of the oriented bounding box that the
    # contour fills (area / (major*minor)). A coherent streak fills its oriented
    # box reasonably; a sprawling/forked clump fills it poorly. Mapped from
    # PATH_FILL_LOW (score 0) to PATH_FILL_HIGH (score 1).
    PATH_FILL_LOW = 0.20
    PATH_FILL_HIGH = 0.55

    # Fading-end scoring. Sample mean brightness in this many bins along the
    # distance from the head; score on how strongly brightness falls from the
    # head ring to the tail-tip ring, normalized. A flat trail scores ~0; a
    # comet scores high.
    NUM_BINS = 8
    # The head is the centroid of the brightest (100 - HEAD_PERCENTILE)% of the
    # object's pixels — the bright compact nucleus. Distance-from-head bins are
    # measured from there.
    HEAD_PERCENTILE = 90.0
    # Normalized brightness drop (head_ring - tail_ring)/head_ring at/above which
    # the fade is "full" (score 1). Measured on comet_sighting.jpg: head ring
    # ~204, tail ring ~91 -> drop ~0.55.
    FADE_FULL = 0.55

    # bright_end (head-tail polarity) is scored as the monotonic-falloff fraction
    # (see detect_bright_end) and needs no scalar constant — it is already [0,1].

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
        # Oriented geometry of the current candidate (from minAreaRect):
        self.major = 0.0          # long side length (px)
        self.minor = 0.0          # short side length (px)
        self.angle = 0.0          # major-axis angle (deg)
        self.end_a = None         # (x, y) one end of the major axis
        self.end_b = None         # (x, y) other end of the major axis
        self.contour_area = 0.0

        # Find every candidate object once.
        self.candidates = self._detect_candidate_objects()

    # ------------------------------------------------------------------
    # Candidate detection (find ALL objects, roundness-agnostic)
    # ------------------------------------------------------------------

    def _detect_candidate_objects(self):
        """Find every object sitting against the dark sky — the "all objects" loop.

        Unlike the sibling detectors (Hough circles / roundness / fill gates,
        all of which would THROW A COMET AWAY because it is elongated and
        low-fill), we keep candidates regardless of roundness. We threshold the
        sky away, bridge the sparse tail to the head with a large MORPH_CLOSE so
        the comet stays one component, take every sufficiently-large external
        contour, and attach its oriented geometry (major/minor axis + axis
        endpoints) for the feature methods.

        Returns a list of (contour, cx, cy, radius, major, minor, angle,
        end_a, end_b).
        """
        _, thresh = cv2.threshold(self.gray, self.THRESH, 255, cv2.THRESH_BINARY)
        kernel = np.ones((self.CLOSE_KERNEL, self.CLOSE_KERNEL), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        candidates = []
        for c in contours:
            (cx, cy), radius = cv2.minEnclosingCircle(c)
            if radius < self.MIN_RADIUS:
                continue

            major, minor, angle, end_a, end_b = self._oriented_geometry(c)
            if major <= 0 or minor <= 0:
                continue

            candidates.append((
                c, float(cx), float(cy), float(radius),
                major, minor, angle, end_a, end_b,
            ))

        return self._dedupe(candidates)

    @staticmethod
    def _oriented_geometry(contour):
        """Fit a minimum-area rectangle and return its oriented geometry.

        Returns (major, minor, angle_deg, end_a, end_b) where major >= minor are
        the rectangle side lengths and end_a/end_b are the midpoints of the two
        SHORT sides — i.e. the two ends of the long axis (the comet's head and
        tail ends). The brightness sampling walks the line between them.
        """
        rect = cv2.minAreaRect(contour)
        (rcx, rcy), (w, h), angle = rect
        box = cv2.boxPoints(rect)  # 4 corners, ordered

        if w >= h:
            major, minor = w, h
            # Short sides connect corners (1->2) and (3->0); their midpoints are
            # the ends of the long axis.
            end_a = (box[1] + box[2]) / 2.0
            end_b = (box[3] + box[0]) / 2.0
        else:
            major, minor = h, w
            end_a = (box[0] + box[1]) / 2.0
            end_b = (box[2] + box[3]) / 2.0

        return (
            float(major), float(minor), float(angle),
            (float(end_a[0]), float(end_a[1])),
            (float(end_b[0]), float(end_b[1])),
        )

    @staticmethod
    def _dedupe(candidates):
        """Drop near-duplicate objects, keeping the largest at each location."""
        kept = []
        for cand in sorted(candidates, key=lambda c: -c[3]):
            cx, cy, r = cand[1], cand[2], cand[3]
            duplicate = False
            for k in kept:
                kx, ky, kr = k[1], k[2], k[3]
                if np.hypot(cx - kx, cy - ky) < kr:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(cand)
        return kept

    def _set_current_object(self, contour, cx, cy, radius,
                            major, minor, angle, end_a, end_b):
        """Point the feature methods at one candidate object."""
        cx, cy, radius = int(cx), int(cy), int(radius)
        mask = np.zeros_like(self.gray)
        cv2.drawContours(mask, [contour], -1, 255, -1)

        x, y, w, h = cv2.boundingRect(contour)
        self.mask = mask
        self.bbox = (x, y, w, h)
        self.center = (cx, cy)
        self.radius = radius
        self.major = major
        self.minor = minor
        self.angle = angle
        self.end_a = end_a
        self.end_b = end_b
        self.contour_area = cv2.contourArea(contour)

    # ------------------------------------------------------------------
    # Brightness sampling along the long axis
    # ------------------------------------------------------------------

    def _head_distance_bins(self):
        """Mean brightness in NUM_BINS rings of increasing distance from the head.

        The minAreaRect long axis is unreliable on a comet — it is set by the
        fat diffuse fan, not the head->tail direction, so projecting onto it can
        peak in the middle. Instead we locate the HEAD as the centroid of the
        brightest pixels (the bright compact nucleus), then bin every mask pixel
        by its distance from that head, normalized by the farthest pixel. Bin 0
        is the head, the last bin is the tail tip. This yields a clean
        "brightness vs distance from head" falloff curve.

        A comet falls off monotonically from a bright head (e.g. 204 -> 91); a
        uniform streak stays flat; a star's limb-brightened disk RISES outward;
        a wandering blob is non-monotonic. Returns a length-NUM_BINS array
        (NaN for empty rings) plus is computed fresh each call.
        """
        if self.mask is None:
            return np.full(self.NUM_BINS, np.nan)

        ys, xs = np.nonzero(self.mask)
        if xs.size == 0:
            return np.full(self.NUM_BINS, np.nan)

        brightness = self.gray[ys, xs].astype(np.float64)

        # Head = centroid of the brightest decile of the object's pixels.
        bright_cut = np.percentile(brightness, self.HEAD_PERCENTILE)
        head_sel = brightness >= bright_cut
        hx = xs[head_sel].mean()
        hy = ys[head_sel].mean()

        dist = np.hypot(xs - hx, ys - hy)
        dmax = dist.max()
        if dmax <= 0:
            return np.full(self.NUM_BINS, np.nan)
        dn = dist / dmax

        bins = np.full(self.NUM_BINS, np.nan)
        for b in range(self.NUM_BINS):
            lo = b / self.NUM_BINS
            hi = (b + 1) / self.NUM_BINS
            sel = (dn >= lo) & (dn < hi) if b < self.NUM_BINS - 1 else (dn >= lo)
            if np.any(sel):
                bins[b] = brightness[sel].mean()
        return bins

    @staticmethod
    def _monotonic_falloff(valid):
        """Fraction of adjacent steps in `valid` that decrease (or hold).

        A perfect monotonic fade (comet) scores 1.0; a flat uniform streak hovers
        near the mid-range (ties count as non-increasing); a rising/wandering
        profile scores low. `valid` is the NaN-free bin array, head-first.
        """
        if valid.size < 2:
            return 0.0
        steps = np.diff(valid)
        non_increasing = np.sum(steps <= 0)
        return float(non_increasing / steps.size)

    # ------------------------------------------------------------------
    # Comet features (each returns [0, 1])
    # ------------------------------------------------------------------

    def detect_bright_line_shape(self):
        """ELONGATION: score how line-like (vs round) the object is.

        major/minor axis ratio of the fitted oriented box, mapped from
        ELONGATION_MIN (round, score 0) to ELONGATION_FULL (clear streak,
        score 1). This is the lead feature: every round star/planet/asteroid
        scores ~0 here, so it alone rejects them.
        """
        if self.mask is None or self.minor <= 0:
            return 0.0
        ratio = self.major / self.minor
        span = self.ELONGATION_FULL - self.ELONGATION_MIN
        score = (ratio - self.ELONGATION_MIN) / span if span else 0.0
        return float(np.clip(score, 0.0, 1.0))

    def detect_pathway(self):
        """LINEARITY: score how well the object forms one coherent path.

        Fraction of the oriented bounding box the contour fills
        (area / (major*minor)). A clean streak hugging one axis fills its box
        reasonably; a sprawling/forked/noise clump fills it poorly. Mapped from
        PATH_FILL_LOW (score 0) to PATH_FILL_HIGH (score 1).
        """
        if self.mask is None or self.major <= 0 or self.minor <= 0:
            return 0.0
        fill = self.contour_area / (self.major * self.minor)
        span = self.PATH_FILL_HIGH - self.PATH_FILL_LOW
        score = (fill - self.PATH_FILL_LOW) / span if span else 0.0
        return float(np.clip(score, 0.0, 1.0))

    def detect_fading_end(self):
        """BRIGHTNESS FALLOFF: score that brightness fades along the path.

        Bins the brightness along the major axis and measures the normalized
        drop from the brightest bin to the dimmest bin. A uniform streak
        (satellite/star trail) shows little drop (score ~0); a comet's head is
        far brighter than its tail tip (large drop, score ~1). Mapped by
        FADE_FULL. This is the discriminator against non-comet streaks.
        """
        bins = self._head_distance_bins()
        valid = bins[~np.isnan(bins)]
        if valid.size < 2:
            return 0.0
        head = valid[0]
        tail = valid[-1]
        if head <= 0:
            return 0.0
        # Drop from the bright head ring to the tail-tip ring — a comet's tail
        # tip is far dimmer than its head; a uniform streak barely drops.
        drop = (head - tail) / head
        return float(np.clip(drop / self.FADE_FULL, 0.0, 1.0))

    def detect_bright_end(self):
        """HEAD-TAIL POLARITY: score the CLEAN MONOTONIC fade from the head.

        This is the single most comet-specific cue. Binning brightness by
        distance from the head, a comet falls off smoothly and monotonically
        (bright head -> faded tail, every ring dimmer than the last); a uniform
        streak stays flat (ties, no clean fade); a limb-brightened star disk
        RISES outward; a wandering blob is non-monotonic. We score the fraction
        of adjacent rings that are non-increasing — "the faded part follows the
        bright end like a path."
        """
        bins = self._head_distance_bins()
        valid = bins[~np.isnan(bins)]
        return self._monotonic_falloff(valid)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def scoring(self):
        """Score the current object for comet-ness and decide the label.

        A comet is CONJUNCTIVE: it must be BOTH an elongated streak AND show a
        clean fading head->tail gradient. A weighted sum fails here — round
        bright blobs fade from their center too (so fade/monotonicity alone admit
        them), and thin uniform trails are elongated too (so elongation alone
        admits them). Only the COMBINATION is a comet. So we multiply two terms:

          streak_term = bright_line_shape (elongation)      -> kills round blobs
          fade_term   = fading_end * bright_end (drop x      -> kills uniform /
                        clean monotonicity), the gradient        partial-fade
                        confidence                                streaks

        comet_score = streak_term * fade_term, then a light additive nudge from
        pathway so a clean coherent path edges up a borderline real comet without
        being able to rescue a non-streak (it is gated by the product). Labeled
        "Comet" iff comet_score >= COMET_THRESHOLD.

        Measured: comet -> 0.388 * (1.0*1.0) ~ 0.39 core, + pathway nudge clears
        threshold; space-streak obj3 -> 0.944 * (0.631*0.714) ~ 0.43 core but its
        partial fade keeps it under; round blobs -> elongation ~0 zeroes it.
        """
        features = {
            "bright_line_shape": self.detect_bright_line_shape(),
            "fading_end": self.detect_fading_end(),
            "pathway": self.detect_pathway(),
            "bright_end": self.detect_bright_end(),
        }

        streak_term = features["bright_line_shape"]
        fade_term = features["fading_end"] * features["bright_end"]
        core = streak_term * fade_term
        # Pathway can only ADD a small bonus, scaled by the core, so it cannot
        # rescue a non-streak or a non-fader (core ~0 -> bonus ~0).
        comet_score = core + self.W_PATHWAY * features["pathway"] * core
        comet_score = float(np.clip(comet_score, 0.0, 1.0))

        is_comet = comet_score >= self.COMET_THRESHOLD
        return {
            "features": features,
            "comet_score": comet_score,
            "label": "Comet" if is_comet else None,
        }

    def scoring_all(self):
        """Score every candidate object in the image."""
        results = []
        for cand in self.candidates:
            self._set_current_object(*cand)
            result = self.scoring()
            result["circle"] = (self.center[0], self.center[1], self.radius)
            result["bbox"] = self.bbox
            result["elongation"] = (
                self.major / self.minor if self.minor else 0.0
            )
            results.append(result)
        return results

    def draw_results(self, results, output_dir="output_images"):
        """Annotate comet detections on one copy of the image and save it.

        Only objects labeled "Comet" are highlighted (magenta box) with their
        confidence; non-comets get no highlight (same convention as the asteroid
        detector). Writes to output_dir using the input filename with a
        "_comet_detected" suffix. Returns the output path.
        """
        output = self.image.copy()

        for result in results:
            if result["label"] != "Comet":
                continue
            color = (255, 0, 255)  # BGR magenta
            x, y, w, h = result["bbox"]
            cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
            score_text = f"Comet {result['comet_score']:.2f}"
            cv2.putText(output, score_text, (x, max(y - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        os.makedirs(output_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(self.image_path))[0]
        output_path = os.path.join(output_dir, f"{stem}_comet_detected.jpg")
        cv2.imwrite(output_path, output)
        return output_path


def process_image(path):
    """Score every object in one image, print results, and save the output."""
    detector = DetectCometFeatures(path)
    results = detector.scoring_all()

    comets = [r for r in results if r["label"] == "Comet"]
    print(f"\nImage: {path}  ({len(results)} object(s) found, "
          f"{len(comets)} comet(s))")
    for i, result in enumerate(results):
        cx, cy, r = result["circle"]
        f = result["features"]
        label = result["label"] or "not comet"
        print(f"  Object {i} @ ({cx},{cy}) r={r} elong={result['elongation']:.2f}: "
              f"score={result['comet_score']:.3f} -> {label}")
        print(f"      line={f['bright_line_shape']:.3f} "
              f"fade={f['fading_end']:.3f} "
              f"path={f['pathway']:.3f} "
              f"bright_end={f['bright_end']:.3f}")

    output_path = detector.draw_results(results)
    print(f"  Saved annotated result to: {output_path}")
    return results


if __name__ == "__main__":
    run_cli(process_image)
