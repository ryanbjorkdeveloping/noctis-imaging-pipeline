"""Connector: identify a close-up object by running ALL four class detectors.

The four detectors in `close_up_object_detection/` each answer ONE question —
"is this object a star / planet / asteroid / comet?" — independently. This
connector ties them together: it runs all four over an image, gathers every
detection, and decides what the image's object actually IS by picking the
highest-confidence detection that CLEARED its class's decision threshold.

Strategy (per-class best, then pick):
  * Each detector finds its own objects and scores them on its own [0, 1] scale.
    A detection's own `label` is non-None only when that detector's threshold was
    met (star flare >= 0.30, planet >= 0.55, asteroid >= 0.55, comet >= 0.60), so
    the connector does NOT re-implement thresholds — it trusts each detector's
    verdict and just compares the cleared winners' scores.
  * The single highest-scoring CLEARED detection becomes the image's prediction.
  * If NO detector clears its threshold, the object is still highlighted but
    labeled the generic "Object" (we found something, but no class is confident).

KNOWN LIMITATION — cross-class score scales differ (the star detector reports a
0.85-1.0 confidence band on a pass, while planet/asteroid/comet report raw
weighted scores), so comparing raw scores across classes is only roughly fair.
On the project's current one-object-per-image samples each image has a single
dominant true class that clears only its OWN threshold, so contests are rare. If
images later contain multiple competing real objects, the scores should be
normalized to a common scale before comparison.

CLI:  python connectors/id_all_close_samples.py [target]
      target = a folder (batch every image) or a single image; default input_images.
"""

import os
import sys

import cv2

# Make the project root importable so `utilities` and `close_up_object_detection`
# resolve when this file is run as a script (its own dir, not the root, is what
# lands on sys.path).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utilities.image_io import run_cli  # noqa: E402
from close_up_object_detection.detect_star_features import DetectStarFeatures  # noqa: E402
from close_up_object_detection.detect_planet_features import DetectPlanetFeatures  # noqa: E402
from close_up_object_detection.detect_asteroid_features import DetectAsteroidFeatures  # noqa: E402
from close_up_object_detection.detect_comet_features import DetectCometFeatures  # noqa: E402
from close_up_object_detection.detect_galaxy_features import DetectGalaxyFeatures #noqa: E402

class IdAllCloseSamples:
    """Run all four close-up detectors on one image and identify its object."""

    # (detector class, result key holding that detector's [0, 1] score, class label)
    DETECTORS = [
        (DetectStarFeatures, "combined_score", "Star"),
        (DetectPlanetFeatures, "planet_score", "Planet"),
        (DetectAsteroidFeatures, "asteroid_score", "Asteroid"),
        (DetectCometFeatures, "comet_score", "Comet"),
        (DetectGalaxyFeatures, "galaxy_score", "Galaxy"),
    ]

    # BGR draw colors per class, plus the neutral color for an unidentified
    # ("Object") detection that no detector was confident enough to claim.
    CLASS_COLORS = {
        "Star": (0, 255, 0),        # green
        "Planet": (255, 128, 0),    # blue/orange
        "Asteroid": (200, 200, 60), # cyan-gray
        "Comet": (255, 0, 255),     # magenta
        "Galaxy": (255, 255, 0),
        "Object": (0, 215, 255),    # amber — unidentified
    }

    def __init__(self, image_path):
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

    def extract_close_up_detection_scripts(self):
        """Run all four detectors and collect every detection they produce.

        Each detector is instantiated on this image and asked for `scoring_all()`.
        Returns a flat list of detection dicts, one per (detector, object), each
        carrying:
          class    — the class label that detector scores for (Star/Planet/...)
          score    — that detector's own [0, 1] score for the object
          cleared  — True iff the detector's own label is set (threshold met)
          bbox     — (x, y, w, h) for drawing (derived from circle if absent)
        """
        detections = []
        for detector_cls, score_key, class_label in self.DETECTORS:
            detector = detector_cls(self.image_path)
            for result in detector.scoring_all():
                detections.append({
                    "class": class_label,
                    "score": float(result.get(score_key, 0.0)),
                    "cleared": result.get("label") is not None,
                    "bbox": self._result_bbox(result),
                })
        return detections

    @staticmethod
    def _result_bbox(result):
        """Get an (x, y, w, h) box from a detection result.

        Asteroid/comet results already carry a `bbox`; star/planet results carry
        a `circle` (cx, cy, radius) which we convert to a square box.
        """
        if result.get("bbox") is not None:
            return tuple(result["bbox"])
        cx, cy, r = result["circle"]
        return (cx - r, cy - r, 2 * r, 2 * r)

    def determine_highest_scoring_caliber(self, detections=None):
        """Pick the image's object: the best detection that cleared a threshold.

        Among detections that CLEARED their class threshold, the highest-scoring
        one wins and gives the image its predicted class. If none cleared, fall
        back to the highest raw score but label it the generic "Object" (something
        is there, but no class is confident). Returns the winning detection dict
        with a final "prediction" field, or None if no object was detected at all.
        """
        if detections is None:
            detections = self.extract_close_up_detection_scripts()
        if not detections:
            return None

        cleared = [d for d in detections if d["cleared"]]
        if cleared:
            winner = max(cleared, key=lambda d: d["score"])
            winner = dict(winner, prediction=winner["class"])
        else:
            # Nothing crossed a threshold — highlight the strongest candidate but
            # do not commit to a class.
            winner = max(detections, key=lambda d: d["score"])
            winner = dict(winner, prediction="Object")
        return winner

    def identify_all_objects(self, detections=None):
        """Every distinct object the image contains, each with its predicted class.

        Unlike `determine_highest_scoring_caliber` (one winner per image), this
        returns ALL objects so a multi-body scene (e.g. a star with planets around
        it) is fully labeled. Each cleared detection becomes an identified object
        tagged with its class. Where two detectors claim the SAME physical object
        (overlapping boxes — e.g. star and planet both firing on one disk), the
        higher score wins that object, so each object is reported once. Returns a
        list of detection dicts each with a `prediction` field.
        """
        if detections is None:
            detections = self.extract_close_up_detection_scripts()

        cleared = [d for d in detections if d["cleared"]]
        # Resolve same-object contests across detectors: largest score first, drop
        # any later detection whose box center sits inside an already-claimed box.
        identified = []
        for d in sorted(cleared, key=lambda d: -d["score"]):
            x, y, w, h = d["bbox"]
            cx, cy = x + w / 2, y + h / 2
            claimed = False
            for kept in identified:
                kx, ky, kw, kh = kept["bbox"]
                if kx <= cx <= kx + kw and ky <= cy <= ky + kh:
                    claimed = True
                    break
            if not claimed:
                identified.append(dict(d, prediction=d["class"]))
        return identified

    def draw_results(self, objects, output_dir="output_images"):
        """Box every identified object on the image, colored by class, and save.

        Each object gets a box in its class color with a "<Class> <score>" label.
        Writes to output_dir using the input filename with an "_identified" suffix.
        Returns the output path.
        """
        output = self.image.copy()
        for obj in objects:
            prediction = obj["prediction"]
            color = self.CLASS_COLORS.get(prediction, self.CLASS_COLORS["Object"])
            x, y, w, h = obj["bbox"]
            cv2.rectangle(output, (x, y), (x + w, y + h), color, 3)
            text = f"{prediction} {obj['score']:.2f}"
            cv2.putText(output, text, (x, max(y - 10, 18)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        os.makedirs(output_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(self.image_path))[0]
        output_path = os.path.join(output_dir, f"{stem}_identified.jpg")
        cv2.imwrite(output_path, output)
        return output_path


def process_image(path):
    """Identify EVERY object in one image, print the verdicts, and save the output."""
    identifier = IdAllCloseSamples(path)
    detections = identifier.extract_close_up_detection_scripts()
    objects = identifier.identify_all_objects(detections)
    winner = identifier.determine_highest_scoring_caliber(detections)

    print(f"\nImage: {path}  ({len(detections)} detection(s) across 4 detectors)")
    if not objects:
        # Nothing cleared a threshold. If anything was detected at all, highlight
        # the strongest as a generic "Object"; otherwise report nothing.
        if winner is not None:
            print(f"  No class confident — highlighting strongest as Object "
                  f"({winner['score']:.3f} as {winner['class']}).")
            objects = [winner]  # winner already carries prediction="Object"
        else:
            print("  No object detected.")
    else:
        counts = {}
        for o in objects:
            counts[o["prediction"]] = counts.get(o["prediction"], 0) + 1
        summary = ", ".join(f"{n} {cls}" for cls, n in counts.items())
        print(f"  Identified {len(objects)} object(s): {summary}")
        print(f"  Top prediction: {winner['prediction']} ({winner['score']:.3f})")

    output_path = identifier.draw_results(objects)
    print(f"  Saved identified result to: {output_path}")
    return objects


if __name__ == "__main__":
    run_cli(process_image)
