"""
Score the local ALeRCE stamp CNN (type) against the cross-matched answer key.

Two comparisons (per the label's source):
  - vs ALeRCE-sourced labels  = agreement with ALeRCE's OWN (newer) stamp CNN
    -> a DRIFT metric (same model lineage as our 2020 checkpoint, NOT independent).
  - vs SIMBAD-sourced labels  = independent ground truth (the honest accuracy number,
    but a small sample).
Only detections labeled with one of the 4 astro classes (VS/AGN/SN/asteroid) are
scored; 'other'(galaxy) and 'unmatched' are out-of-scope for a 4-class CNN.

Run:  python stamp_testing/score_stamp_vs_labels.py
"""

import os, sys
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import tensorflow as tf
tf.compat.v1.disable_v2_behavior()

from astropy.table import Table
from ztf_classification.stamp_classifier import center_crop
from ztf_classification.pathA_metadata import (
    read_fits_header_fields, load_psfcat, build_metadata, normalize_metadata,
)

CLASSES = ["AGN", "SN", "VS", "asteroid", "bogus"]
ASTRO = ["AGN", "SN", "VS", "asteroid"]

LABELS = os.path.join(PROJECT_ROOT, "ztfdata", "labels",
                      "ztf_20180322273264_labeled_set.ecsv")
SCI = os.path.join(PROJECT_ROOT, "ztfdata", "sci", "2018", "0322", "273264",
                   "ztf_20180322273264_000468_zr_c03_o_q2_sciimg.fits")
PSFCAT = os.path.join(PROJECT_ROOT, "ztfdata", "sci", "2018", "0322", "273264",
                      "ztf_20180322273264_000468_zr_c03_o_q2_psfcat.fits")
CUTOUTS = os.path.join(PROJECT_ROOT, "ztfdata", "ztf_own_pipeline_data_testing_small_scale", "cutouts")
CKPT = os.path.join(
    PROJECT_ROOT, "stamp_classifier_model", "results", "staging_model",
    "DeepHits_EntropyRegBeta0.5000_batch64_lr0.00100_droprate0.5000_inputsize21_filtersize5_0_20200708-160759",
    "checkpoints")
NORM = CKPT.replace("/checkpoints", "") + "/feature_norm_stats.pkl"
# our (3,63,63) triplet -> (21,21,3) model cube, per-image normalized to [0,1].
# (identical to pathA_our_detections.build_image)
def build_image(npy_path):
    sci, ref, diff = np.load(npy_path)
    cube = np.stack(
        [np.nan_to_num(center_crop(x.astype(np.float32))) for x in (sci, ref, diff)],
        axis=-1,
    )
    imgs = cube[np.newaxis, ...]
    imgs = imgs - np.nanmin(imgs, axis=(1, 2))[:, None, None, :]
    imgs = imgs / np.nanmax(imgs, axis=(1, 2))[:, None, None, :]
    return np.nan_to_num(imgs[0]).astype(np.float32)


# feed one detection through the frozen graph -> drop-bogus argmax type string.
def predict_type(sess, g, handle_ph, train_flag, out, cube, meta_vec):
    rots = np.stack([np.rot90(cube, k) for k in range(4)], axis=0).astype(np.float32)
    lbl = np.zeros((1,), dtype=np.int64)
    img_ds  = tf.compat.v1.data.Dataset.from_tensor_slices(rots).batch(4)
    meta_ds = tf.compat.v1.data.Dataset.from_tensor_slices(meta_vec).batch(1)
    lbl_ds  = tf.compat.v1.data.Dataset.from_tensor_slices(lbl).batch(1)
    ds = tf.compat.v1.data.Dataset.zip((img_ds, meta_ds, lbl_ds))
    it = tf.compat.v1.data.make_initializable_iterator(ds)
    sess.run(it.initializer)
    h = sess.run(it.string_handle())
    probs = sess.run(out, feed_dict={handle_ph: h, train_flag: False})[0]
    # drop bogus (braai already gates real/bogus), argmax over the 4 astro classes
    astro = {c: float(p) for c, p in zip(CLASSES, probs) if c != "bogus"}
    return max(astro, key=astro.get)

def main():
    from sklearn.metrics import confusion_matrix, classification_report

    tab = Table.read(LABELS)
    # only score detections whose label is one of the 4 astro classes
    rows = [r for r in tab if str(r["label"]) in ASTRO]
    print(f"{len(tab)} detections; {len(rows)} labeled with an astro class (scored)\n")

    # independent ground truth: query SIMBAD directly at each scored detection's
    # position (bypasses the merge priority, which let ALeRCE outrank SIMBAD)
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from ztf_classification.crossmatch_labels import label_stationary_simbad
    row_coords = SkyCoord(ra=[r["ra"] for r in rows] * u.deg,
                          dec=[r["dec"] for r in rows] * u.deg)
    simbad_types = label_stationary_simbad(row_coords)   # {row_index: type}

    hdr_fields = read_fits_header_fields(SCI)
    psfcat = load_psfcat(PSFCAT)

    g = tf.Graph()
    with g.as_default():
        saver = tf.compat.v1.train.import_meta_graph(CKPT + "/model.meta")
        sess = tf.compat.v1.Session()
        saver.restore(sess, CKPT + "/model")
        handle_ph  = g.get_tensor_by_name("iterators/Placeholder_6:0")
        train_flag = g.get_tensor_by_name("inputs/training_flag:0")
        out        = g.get_tensor_by_name("outputs/Softmax:0")
        print("MODEL LOADED\n")

        y_true, y_pred, sources, y_simbad = [], [], [], []
        for i, r in enumerate(rows):
            name = os.path.basename(str(r["stamp_path"])).replace(".npy", "")
            cube = build_image(os.path.join(CUTOUTS, name + ".npy"))
            meta = build_metadata(float(r["ra"]), float(r["dec"]), float(r["sign"]),
                                  float(r["segment_flux"]), float(r["snr"]),
                                  hdr_fields, psfcat)
            meta_vec = normalize_metadata(meta, NORM)
            pred = predict_type(sess, g, handle_ph, train_flag, out, cube, meta_vec)
            y_true.append(str(r["label"]))
            y_pred.append(pred)
            sources.append(str(r["label_source"]))
            y_simbad.append(simbad_types.get(i))   # SIMBAD type or None

    y_true = np.array(y_true); y_pred = np.array(y_pred); sources = np.array(sources)

    def report(mask, title):
        yt, yp = y_true[mask], y_pred[mask]
        if len(yt) == 0:
            print(f"=== {title}: no detections ===\n"); return
        acc = (yt == yp).mean()
        print(f"=== {title}  (n={len(yt)}, accuracy {acc:.3f}) ===")
        print("confusion matrix (rows=truth, cols=pred), labels=" + str(ASTRO))
        print(confusion_matrix(yt, yp, labels=ASTRO))
        print(classification_report(yt, yp, labels=ASTRO, zero_division=0))

    # PART 1: agreement with ALeRCE's own (newer) CNN -- a DRIFT metric, same lineage
    report(sources == "ztf_alerce", "vs ALeRCE labels (agreement/drift, NOT independent)")
    # PART 2: independent ground truth from the merged label (usually empty -- ALeRCE outranks)
    report(sources == "simbad", "vs SIMBAD (merged label) INDEPENDENT")

    # PART 3: independent SIMBAD truth, queried DIRECTLY for every scored detection
    # (not merged away), restricted to detections SIMBAD typed as an astro class
    y_simbad = np.array([s if s is not None else "" for s in y_simbad])
    astro_mask = np.isin(y_simbad, ASTRO)
    if astro_mask.any():
        yt, yp = y_simbad[astro_mask], y_pred[astro_mask]
        acc = (yt == yp).mean()
        print(f"=== vs SIMBAD DIRECT (independent, n={len(yt)}, accuracy {acc:.3f}) ===")
        print(confusion_matrix(yt, yp, labels=ASTRO))
        print(classification_report(yt, yp, labels=ASTRO, zero_division=0))
    else:
        print("=== vs SIMBAD DIRECT: SIMBAD typed 0 detections as an astro class ===")
        print("    (SIMBAD mostly returns 'star'/'G' here, which aren't in the CNN's 4 classes)\n")

    print("SCOPE: PART 1 shares model lineage with ALeRCE (drift, not accuracy);")
    print("       PART 2 is independent but tiny. The local 2020 CNN is uncalibrated")
    print("       + fed proxy metadata -> trust relative ranking, not absolute type.")


if __name__ == "__main__":
    main()
