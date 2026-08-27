import os
import sys

import cv2
import numpy as np

# Make the project root importable so `utilities` resolves when this file is run
# as a script (whose own directory — not the root — is what lands on sys.path).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utilities.image_io import run_cli  # noqa: E402


class DetectPlanetFeatures:
    """Score a close-up image on how strongly each object reads as a planet.

    Standalone counterpart to DetectStarFeatures. This file answers ONE
    question per object: "how planet-like is this?" It does NOT compare
    classes, vote, or decide star-vs-planet — that is handled elsewhere.

    A planet is characterized (visually, in a single still) by two signals:
      1. Round sphere      — it is a round disk (graded roundness).
      2. Sharp, flare-free  — its limb is a clean edge with no plasma flares
         edge                (the visual inverse of a star's flaring limb).

    These are combined into a single planet_score in [0, 1]:
        planet_score = PLANET_EDGE_WEIGHT  * edge_sharpness
                     + PLANET_ROUND_WEIGHT * roundness_score
    No decision threshold is baked in here — the raw score (and its two
    sub-scores) are reported per object for later thresholding/tuning.
    """

    # Planet score weights. Edge sharpness carries more weight: it is the real
    # discriminator (a planet has a crisp, flare-free limb; a star flares),
    # whereas roundness adds less separation because every candidate is already
    # roundness-gated below. Roundness still earns its keep by penalizing lumpy
    # asteroid-like blobs that slip through the gate.
    PLANET_EDGE_WEIGHT = 0.65
    PLANET_ROUND_WEIGHT = 0.35

    # An object is labeled "Planet" only when its combined planet_score clears
    # this. The sibling detectors (asteroid 0.55, comet 0.60) each carry a real
    # decision threshold; this gives the planet detector the same kind of verdict
    # so the connector can compare threshold-cleared winners across classes
    # rather than treating an always-on "Planet" tag as a detection. The two
    # sub-scores are already normalized [0, 1]: a clean flare-free round disk
    # scores high on edge_sharpness and clears 0.55, while a merged-blob leak
    # (low roundness, no clean limb) falls below it.
    PLANET_THRESHOLD = 0.55

    # Minimum disk radius (px) to score — rejects nebula specks / background
    # stars. Roundness floor for a candidate to count as a real disk.
    #
    # Roundness is measured as CIRCULARITY (4*pi*area / perimeter^2, on a
    # simplified contour — see _circularity), which is 1.0 for a perfect circle
    # and collapses toward 0 for a jagged/spiky outline: a rough boundary
    # inflates the perimeter sharply while barely changing area. This rejects a
    # comet (jagged ~0.10) while passing a clean disk (measured ~0.94 on the
    # sample planet/asteroid/star images). NOTE: the previous fill-ratio metric
    # (area / minEnclosingCircle^2) could NOT tell them apart — a jagged shape
    # still fills most of its bounding circle (~0.63), so a comet sailed past the
    # gate and read as fully round.
    #
    # MIN_ROUNDNESS is 0.5 (was 0.6 for the old metric): clean disks measure
    # ~0.94 and a jagged comet ~0.10, so 0.5 separates them with wide margin.
    #
    # The Hough pass uses a DIFFERENT roundness metric — ring edge-support
    # (_ring_roundness), on its own scale — because a phase-lit planet's bright
    # contour is a crescent and circularity can't see the full disk. Real
    # planets score ~0.97-1.0 of the ring; a comet only ~0.33-0.67 (its spiky
    # boundary intersects the fitted circle intermittently). MIN_RING_ROUNDNESS
    # is 0.85, comfortably between the two populations.
    MIN_RADIUS = 25
    MIN_ROUNDNESS = 0.5
    MIN_RING_ROUNDNESS = 0.85

    # NOTE: a local-contrast "blob pass" was trialed here to recover the planets
    # Hough misses in a busy nebula frame, but it was REVERTED. The colored nebula
    # in `multiple_planets_and_a_star.jpg` has soft round glowy patches that mimic
    # a planet, and NO cheap feature (limb-roundness, color saturation, edge
    # brightness-drop, interior solidity — all measured) separates those patches
    # from real planets, so it produced empty-space false positives. Project rule:
    # a MISSED planet is acceptable, a FALSE POSITIVE is not. Detection therefore
    # stays on the Hough + bright-contour passes, which find real planets with zero
    # false positives — at the cost of missing dim / frame-clipped / ringed planets.

    # Cap on how much _grow_to_full_disk may expand a bright core. A real disk's
    # full glow is only ~1.5x its bright core; a ~3x jump means the core fused
    # with a star's glow + nebula into a frame-spanning blob (which would then
    # deduplicate away every other planet). 2.0 sits between the two.
    GROW_MAX_RADIUS_RATIO = 2.0

    # Lumpy-body suppression (rejects a cratered asteroid's craters being read as
    # separate planets — see _suppress_inside_lumpy_body). A "dominant body" must
    # fill at least this fraction of the smaller image dimension in radius to
    # count as the frame's subject; candidates smaller than CHILD_MAX of that
    # body's radius and inside it are treated as surface features and dropped.
    LUMPY_BODY_MIN_RADIUS_FRAC = 0.35
    LUMPY_BODY_CHILD_MAX_RADIUS_FRAC = 0.9
    # The body must also be SOLID — fill at least this fraction of its bounding
    # circle — so a sprawling star-glow + nebula smear (fill ~0.36) is not
    # mistaken for a compact cratered asteroid (fill ~0.81) and does not suppress
    # the planets floating within it.
    LUMPY_BODY_MIN_FILL = 0.6

    def __init__(self, image_path):
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Per-object "current disk" state the feature methods read from. Set by
        # _set_current_disk() as the scoring loop steps through each candidate.
        self.disk_mask = None
        self.bbox = None
        self.disk_area = 0.0
        self.roundness = 0.0

        # Find every candidate disk (planets, plus any star/asteroid) once.
        self.candidates = self._detect_candidate_disks()

    # ------------------------------------------------------------------
    # Candidate detection (class-agnostic disk finding)
    # ------------------------------------------------------------------

    def _detect_candidate_disks(self):
        """Find every round disk in the image: bright blobs AND Hough circles.

        No single detector catches everything: a bright fuzzy-edged object is
        found by brightness thresholding, while darker round planets are found
        by Hough circle detection. We gather both, dedupe overlaps, and keep
        the round ones above MIN_RADIUS. Each candidate is
        (cx, cy, radius, roundness) — the measured roundness is carried through
        so the planet score can grade it (the star detector discards it).
        """
        candidates = []

        # Source 1 — bright contour blobs. A luminous object's core only reads
        # as a clean round disk at a HIGH threshold; at low thresholds it fuses
        # with surrounding glow into a low-roundness mega-blob. So scan a few
        # high thresholds and keep whatever isolates as round.
        for t in (220, 200, 180):
            _, thresh = cv2.threshold(self.gray, t, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for c in contours:
                (cx, cy), radius = cv2.minEnclosingCircle(c)
                if radius < self.MIN_RADIUS:
                    continue
                roundness = self._circularity(c)
                if roundness >= self.MIN_ROUNDNESS:
                    # Grow the bright core out to the full glowing disk so the
                    # limb annulus lands on the real edge, not inside the body.
                    # The bright contour already gave us this object's real
                    # circularity, so keep it if grow can't improve the geometry.
                    candidates.append(self._grow_to_full_disk(cx, cy, radius, roundness))

        # Source 2 — Hough circles (catches darker / partially-lit round planets
        # that the bright pass misses). Hough only FITS a circle to a roughly-
        # circular gradient pattern — it does NOT verify the object is actually
        # round, so a jagged comet can produce a Hough hit. A Hough hit also
        # carries no roundness of its own. We CANNOT measure roundness from the
        # bright contour here: a phase-lit planet's lit region is a crescent
        # (low circularity) even though the planet is a full sphere. Instead we
        # verify the FULL fitted circle via ring edge-support (_ring_roundness):
        # a real planet has a continuous smooth limb all the way around (lit +
        # dark side), while a comet's spiky boundary only intersects the circle
        # intermittently. Hits that fail the roundness floor (a comet) are dropped.
        blur = cv2.medianBlur(self.gray, 5)
        circles = cv2.HoughCircles(
            blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=120,
            param1=120, param2=50,
            minRadius=self.MIN_RADIUS, maxRadius=350,
        )
        if circles is not None:
            for cx, cy, radius in circles[0]:
                ring = self._ring_roundness(float(cx), float(cy), float(radius))
                if ring >= self.MIN_RING_ROUNDNESS:
                    candidates.append((float(cx), float(cy), float(radius), ring))

        candidates = self._suppress_inside_lumpy_body(candidates)
        return self._dedupe_circles(candidates)

    def _suppress_inside_lumpy_body(self, candidates):
        """Drop candidates that are sub-features inside one large, non-round body.

        A heavily-cratered asteroid (e.g. Vesta) is a single lumpy object, but
        Hough fits a circle to each crater RIM, and a crisp rim scores high
        ring-roundness on its own small radius — so the craters survive as
        spurious "planets." We reject them by recognizing the parent: the
        dominant object's footprint is LARGE, SOLID, and its own limb is NOT
        round (ring-roundness below the gate, ~0.49 for Vesta). When such a body
        exists, every candidate sitting well inside it is a surface feature, not
        a free-floating disk, and is dropped. A round dominant body (the Sun's
        core, or a real planet) passes the ring gate, so it is NOT treated as a
        lumpy body and its candidates are kept.

        The SOLID requirement is what stops this from misfiring on a scene where
        the dominant "footprint" is really a star's glow + nebula band merged
        together: that smear is large and non-round too, but it is sprawling, not
        solid (it fills only ~0.36 of its bounding circle vs ~0.81 for a real
        asteroid), so it must NOT suppress the planets floating within it.
        """
        fp = self._dominant_footprint()
        if fp is None:
            return candidates
        fx, fy, fr, fill = fp
        h, w = self.gray.shape

        # Only act on a body large enough to plausibly be the frame's subject.
        if fr < self.LUMPY_BODY_MIN_RADIUS_FRAC * min(h, w):
            return candidates
        # ...that is SOLID (a compact body, not a sprawling glow/nebula smear)...
        if fill < self.LUMPY_BODY_MIN_FILL:
            return candidates
        # ...and only if that body's own limb is NOT round (a real lumpy body,
        # not a round disk whose bright core we want to keep).
        if self._ring_roundness(fx, fy, fr) >= self.MIN_RING_ROUNDNESS:
            return candidates

        kept = []
        for cx, cy, r, roundness in candidates:
            inside = (np.hypot(cx - fx, cy - fy) < fr * 0.9
                      and r < fr * self.LUMPY_BODY_CHILD_MAX_RADIUS_FRAC)
            if not inside:
                kept.append((cx, cy, r, roundness))
        return kept

    def _dominant_footprint(self):
        """Footprint (cx, cy, radius, fill) of the largest object vs the dark sky.

        Threshold at a low brightness to separate any object from black space,
        then take the largest external contour's enclosing circle. `fill` is the
        contour area divided by the enclosing circle's area — how solidly the
        footprint fills its bounding circle. A compact body (a cratered asteroid)
        nearly fills it (~0.8); a sprawling glow/nebula smear with tendrils and
        gaps fills only a fraction (~0.36), which is how the two are told apart.
        Returns None if nothing is found.
        """
        _, thresh = cv2.threshold(self.gray, 30, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None
        big = max(contours, key=cv2.contourArea)
        (fx, fy), fr = cv2.minEnclosingCircle(big)
        fill = cv2.contourArea(big) / (np.pi * fr * fr) if fr else 0.0
        return (fx, fy, fr, fill)

    @staticmethod
    def _circularity(contour):
        """Circularity (4*pi*area / perimeter^2): 1.0 = perfect circle, ~0 = jagged.

        Unlike a min-enclosing-circle fill ratio, this collapses for a spiky or
        rough outline because the perimeter grows much faster than the area —
        which is exactly how a comet is told apart from a smooth disk.

        The contour is first simplified with approxPolyDP. A raw pixel contour
        has a staircase perimeter that deflates circularity even for a true
        circle (a clean disk measured ~0.57 raw vs ~0.94 simplified); the
        simplification strips that pixel-step noise so the metric reflects the
        real boundary shape — a genuine spiky comet outline still scores low.
        """
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            return 0.0
        approx = cv2.approxPolyDP(contour, 0.01 * perimeter, True)
        area = cv2.contourArea(approx)
        per = cv2.arcLength(approx, True)
        if per == 0:
            return 0.0
        return float(4.0 * np.pi * area / (per * per))

    def _ring_roundness(self, cx, cy, radius, n_sectors=180, band=0.12):
        """Fraction of the fitted circle's circumference that has a real edge.

        Measures roundness on the FULL disk boundary (not the bright crescent):
        walk n_sectors angular steps around the circle and, in each, look for a
        strong image gradient within a thin annulus (radius +/- band). A true
        sphere — even one only partly sunlit — has a continuous limb all the way
        around, so nearly every sector finds an edge (~0.97-1.0). A comet's spiky
        boundary only crosses the fitted circle here and there (~0.33-0.67), so
        it scores low. Returns the supported fraction in [0, 1].
        """
        gx = cv2.Sobel(self.gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(self.gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        h, w = self.gray.shape

        radii = np.linspace(radius * (1 - band), radius * (1 + band), 5)
        hits = 0
        for i in range(n_sectors):
            ang = 2.0 * np.pi * i / n_sectors
            cos_a, sin_a = np.cos(ang), np.sin(ang)
            best = 0.0
            for rr in radii:
                x = int(cx + rr * cos_a)
                y = int(cy + rr * sin_a)
                if 0 <= x < w and 0 <= y < h:
                    best = max(best, mag[y, x])
            if best > 40:  # a strong edge is present in this sector
                hits += 1
        return hits / n_sectors

    def _grow_to_full_disk(self, cx, cy, radius, measured_roundness=None):
        """Grow a candidate out to its full glowing disk and measure roundness.

        Find the low-threshold (40) contour that contains this center; when a
        larger round one exists, snap out to it (keeping the limb inside the
        scored disk) and use ITS circularity as the candidate's roundness.

        The grow is REFUSED if the low-threshold contour is more than
        GROW_MAX_RADIUS_RATIO larger than the bright core. A real disk's full
        glow is only modestly bigger than its bright core (~1.5x); a jump of
        ~3x means the core has fused with a star's glow + nebula into a frame-
        spanning blob. Snapping to that blob would make one mega-candidate that
        deduplicates away every other planet in the frame — which is exactly the
        bug it prevents.

        Fallback when no acceptable larger round contour is found depends on the
        caller:
          - measured_roundness given (bright-contour pass): the candidate already
            has its own measured circularity, so keep the original circle with
            that roundness. This preserves per-object detection in busy images
            where the threshold-40 contour merges many objects into one blob.
          - measured_roundness None (Hough pass): a Hough hit has NO roundness of
            its own and Hough does not verify roundness, so without a real round
            contour to back it we return None and drop it (this is what discards
            a jagged comet that produced a spurious Hough circle).
        """
        _, thresh = cv2.threshold(self.gray, 40, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for c in contours:
            if cv2.pointPolygonTest(c, (cx, cy), False) < 0:
                continue  # this contour doesn't contain the candidate center
            (fx, fy), fr = cv2.minEnclosingCircle(c)
            if fr <= radius:
                continue  # not larger than the candidate — not a fuller disk
            if fr > radius * self.GROW_MAX_RADIUS_RATIO:
                continue  # too much bigger — a fused glow/nebula blob, not a disk
            roundness = self._circularity(c)
            if roundness >= self.MIN_ROUNDNESS:
                return (fx, fy, fr, roundness)

        if measured_roundness is not None:
            return (cx, cy, radius, measured_roundness)
        return None

    @staticmethod
    def _dedupe_circles(circles):
        """Drop near-duplicate circles (same object found by >1 source).

        Two circles are the SAME object only when they are concentric AND
        comparable in size — i.e. the smaller center sits within the SMALLER
        radius of the pair. The old rule (center within the LARGER radius) made a
        frame-spanning glow/Sun blob swallow every small planet floating on top of
        it, erasing real detections in a busy multi-planet frame. Keying the
        overlap test on the smaller radius keeps a tight, distinct planet that
        merely overlaps a much larger blob. Each circle carries its measured
        roundness as a 4th element, preserved through dedup.
        """
        kept = []
        for cx, cy, r, roundness in sorted(circles, key=lambda c: -c[2]):
            duplicate = False
            for kx, ky, kr, _ in kept:
                dist = np.hypot(cx - kx, cy - ky)
                if dist < min(r, kr):  # concentric, comparable-size overlap
                    duplicate = True
                    break
            if not duplicate:
                kept.append((cx, cy, r, roundness))
        return kept

    def _set_current_disk(self, cx, cy, radius, roundness):
        """Point the feature methods at one candidate disk.

        Builds the filled circular mask, bbox, area, and stores the candidate's
        measured roundness so detect_roundness_score() can grade it.
        """
        cx, cy, radius = int(cx), int(cy), int(radius)
        mask = np.zeros_like(self.gray)
        cv2.circle(mask, (cx, cy), radius, 255, -1)

        self.disk_mask = mask
        self.bbox = (cx - radius, cy - radius, 2 * radius, 2 * radius)
        self.disk_area = float(np.pi * radius * radius)
        self.roundness = float(roundness)

    # ------------------------------------------------------------------
    # Planet features
    # ------------------------------------------------------------------

    def detect_edge_sharpness(self):
        """Score a sharp, flare-free limb: high for planets, low for stars.

        This is the inverse of the star detector's flare measure. We sample the
        bright-core limb annulus (0.62-0.78 R) and take the fraction of
        very-bright (>230) pixels = flare_fraction. A star's plasma flares show
        as bright spots there; a planet's clean limb has nothing bright in this
        band. edge_sharpness = 1 - clip(flare_fraction * 20, 0, 1), so a flaring
        object scores near 0 and a crisp flare-free limb scores near 1. The 20x
        scaling matches the star detector so both files agree on "a flare".
        """
        if self.disk_mask is None:
            return 0.0

        x, y, w, h = self.bbox
        cx, cy = x + w // 2, y + h // 2
        radius = max(w, h) // 2

        annulus = np.zeros_like(self.gray)
        cv2.circle(annulus, (cx, cy), int(radius * 0.78), 255, -1)
        cv2.circle(annulus, (cx, cy), int(radius * 0.62), 0, -1)

        annulus_pixels = self.gray[annulus == 255]
        if annulus_pixels.size == 0:
            return 0.0

        flare_fraction = np.mean(annulus_pixels > 230)
        flare_score = float(np.clip(flare_fraction * 20.0, 0.0, 1.0))
        return 1.0 - flare_score

    def detect_roundness_score(self):
        """Grade how round the disk is, mapped to [0, 1].

        The candidate's measured roundness (circularity for bright-pass disks,
        ring edge-support for Hough disks — both 1.0 for a perfect circle) is
        mapped linearly from the MIN_ROUNDNESS floor up to 1.0 onto [0, 1]. A
        clean spherical disk scores near 1; a candidate that only just clears
        the roundness gate scores near 0.
        """
        if self.disk_mask is None:
            return 0.0

        span = 1.0 - self.MIN_ROUNDNESS
        score = (self.roundness - self.MIN_ROUNDNESS) / span if span else 0.0
        return float(np.clip(score, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _score_current_disk(self):
        """Score the currently-set disk for planet-ness. Assumes _set_current_disk().

        Returns the raw planet_score and its two sub-scores, and labels "Planet"
        iff the score clears PLANET_THRESHOLD (None below it). This is a real
        per-class verdict, consistent with the asteroid/comet detectors, so the
        connector can trust each detector's own label.
        """
        features = {
            "edge_sharpness": self.detect_edge_sharpness(),
            "roundness_score": self.detect_roundness_score(),
        }

        planet_score = float(np.clip(
            self.PLANET_EDGE_WEIGHT * features["edge_sharpness"]
            + self.PLANET_ROUND_WEIGHT * features["roundness_score"],
            0.0, 1.0,
        ))

        is_planet = planet_score >= self.PLANET_THRESHOLD
        return {
            "features": features,
            "planet_score": planet_score,
            "label": "Planet" if is_planet else None,
        }

    def scoring_all(self):
        """Score every candidate disk in the image.

        Returns a list of per-object results, each carrying its (cx, cy, radius)
        circle plus the feature sub-scores, planet_score, and label.
        """
        results = []
        for cx, cy, radius, roundness in self.candidates:
            self._set_current_disk(cx, cy, radius, roundness)
            result = self._score_current_disk()
            result["circle"] = (int(cx), int(cy), int(radius))
            results.append(result)
        return results

    def draw_results(self, results, output_dir="output_images"):
        """Annotate planet detections on one copy of the image and save it.

        Only objects labeled "Planet" (score cleared PLANET_THRESHOLD) are boxed,
        matching the asteroid/comet convention; below-threshold candidates get no
        highlight. Writes to output_dir using the input filename with a
        "_planet_detected" suffix. Returns the output path.
        """
        output = self.image.copy()

        for result in results:
            if result["label"] != "Planet":
                continue  # no highlight for below-threshold candidates
            color = (255, 128, 0)  # BGR — blue/orange box for planet candidates
            cx, cy, radius = result["circle"]
            x, y = cx - radius, cy - radius
            cv2.rectangle(output, (x, y), (cx + radius, cy + radius), color, 2)

            score_text = f"Planet {result['planet_score']:.2f}"
            cv2.putText(output, score_text, (x, max(y - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        os.makedirs(output_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(self.image_path))[0]
        output_path = os.path.join(output_dir, f"{stem}_planet_detected.jpg")
        cv2.imwrite(output_path, output)
        return output_path


def process_image(path):
    """Score every object in one image, print results, and save the output."""
    detector = DetectPlanetFeatures(path)
    results = detector.scoring_all()

    print(f"\nImage: {path}  ({len(results)} object(s) found)")
    for i, result in enumerate(results):
        cx, cy, r = result["circle"]
        f = result["features"]
        label = result["label"] or "not planet"
        print(f"  Object {i} @ ({cx},{cy}) r={r}: "
              f"planet_score={result['planet_score']:.3f} -> {label} "
              f"(edge={f['edge_sharpness']:.3f}, round={f['roundness_score']:.3f})")

    output_path = detector.draw_results(results)
    print(f"  Saved annotated result to: {output_path}")
    return results


if __name__ == "__main__":
    run_cli(process_image)
