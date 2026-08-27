"""
The missing test: run OUR LOCAL 2020 stamp CNN on a batch of labeled ALeRCE
objects (native stamps + real alert metadata) and score agreement vs ALeRCE's
type. This measures whether the OLD local model still matches good labels, on
objects that ARE in the 4 classes (AGN/SN/VS/asteroid) -- unlike our stationary
field, which had zero objects in those classes.

Honest scope: labels come from ALeRCE's NEWER model, so this is local-2020 vs
ALeRCE-current AGREEMENT (a drift metric), on the right kind of data this time.

Runs in the MAIN venv (TF2 + local CNN + alerce). Slow (get_stamps + get_avro +
a graph run per object).  Run:  python stamp_testing/test_stamp_local_vs_labels.py
"""
import os, sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import io, pickle, math
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
import ephem
import fastavro
from alerce.core import Alerce
from ztf_classification.stamp_classifier import center_crop, find_objects_of_class

ASTRO = ["AGN", "SN", "VS", "asteroid"]
N_PER_CLASS = 5

# all 23 metadata fields IN THE MODEL'S EXACT ORDER (position is load-bearing)
FEATURES = ["sgscore1","distpsnr1","sgscore2","distpsnr2","sgscore3","distpsnr3",
            "isdiffpos","fwhm","magpsf","sigmapsf","ra","dec","diffmaglim","classtar",
            "ndethist","ncovhist","ecl_lat","ecl_long","gal_lat","gal_long",
            "non_detections","chinr","sharpnr"]

# the model's per-field CLIP bounds (tames the ZTF -999 "no data" sentinels)
CLIP = {"sgscore1":[-1,None],"distpsnr1":[-1,None],"sgscore2":[-1,None],"distpsnr2":[-1,None],
        "sgscore3":[-1,None],"distpsnr3":[-1,None],"fwhm":[None,10],"ndethist":[None,20],
        "ncovhist":[None,3000],"chinr":[-1,15],"sharpnr":[-1,1.5],"non_detections":[None,2000]}

CKPT = os.path.join(
    PROJECT_ROOT, "stamp_classifier_model", "results", "staging_model",
    "DeepHits_EntropyRegBeta0.5000_batch64_lr0.00100_droprate0.5000_inputsize21_filtersize5_0_20200708-160759",
    "checkpoints",
)


def predict_local(sess, g, handle_ph, train_flag, out, client, oid):
    """Run the local 2020 stamp CNN on one ALeRCE object -> drop-bogus argmax class."""
    # fetch the native (sci, ref, diff) stamp
    hdul = client.get_stamps(oid, format="HDUList")
    sci, ref, diff = hdul[0].data, hdul[1].data, hdul[2].data

    # build the (21,21,3) cube: center-crop 63->21, scrub NaNs
    cube = np.stack(
        [np.nan_to_num(center_crop(x.astype(np.float32))) for x in (sci, ref, diff)],
        axis=-1,
    )

    # IMAGE NORMALIZATION (load-bearing): per-image, per-channel min-max to [0,1]
    imgs = cube[np.newaxis, ...]                                   # (1,21,21,3)
    imgs = imgs - np.nanmin(imgs, axis=(1, 2))[:, None, None, :]
    imgs = imgs / np.nanmax(imgs, axis=(1, 2))[:, None, None, :]
    cube = np.nan_to_num(imgs[0]).astype(np.float32)

    # feed the image as its 4 rotations (cyclic pooling expects a multiple of 4)
    rots = np.stack([np.rot90(cube, k) for k in range(4)], axis=0).astype(np.float32)

    # assemble the 23-field metadata from the ZTF alert packet
    alert = list(fastavro.reader(io.BytesIO(client.get_avro(oid))))[0]
    cand = alert["candidate"]

    eq = ephem.Equatorial(math.radians(cand["ra"]), math.radians(cand["dec"]))
    ecl, gal = ephem.Ecliptic(eq), ephem.Galactic(eq)
    derived = {
        "ecl_lat": float(ecl.lat), "ecl_long": math.degrees(ecl.lon),
        "gal_lat": float(gal.lat), "gal_long": math.degrees(gal.lon),
        "non_detections": 0.0,
        "isdiffpos": 1.0 if cand["isdiffpos"] == "t" else -1.0,
    }

    def get_field(name):
        if name in derived:  return derived[name]
        return float(cand[name])

    raw_meta = np.array([get_field(f) for f in FEATURES], dtype=np.float32)

    # clip per the model's dict (tames -999 sentinels), THEN z-score normalize
    for i, f in enumerate(FEATURES):
        if f in CLIP:
            lo, hi = CLIP[f]
            raw_meta[i] = np.clip(raw_meta[i],
                                  lo if lo is not None else -np.inf,
                                  hi if hi is not None else np.inf)

    NORM = pickle.load(open(CKPT.replace("/checkpoints", "") + "/feature_norm_stats.pkl", "rb"))
    meta = raw_meta.copy()
    for i, f in enumerate(FEATURES):
        if f in NORM:
            meta[i] = (raw_meta[i] - NORM[f]["mean"]) / NORM[f]["std"]
    meta = meta.reshape(1, 23).astype(np.float32)

    # feed the graph (image = 4 rotations, metadata = 1 row) and predict
    lbl = np.zeros((1,), dtype=np.int64)
    img_ds  = tf.compat.v1.data.Dataset.from_tensor_slices(rots).batch(4)
    meta_ds = tf.compat.v1.data.Dataset.from_tensor_slices(meta).batch(1)
    lbl_ds  = tf.compat.v1.data.Dataset.from_tensor_slices(lbl).batch(1)
    ds = tf.compat.v1.data.Dataset.zip((img_ds, meta_ds, lbl_ds))
    it = tf.compat.v1.data.make_initializable_iterator(ds)
    sess.run(it.initializer)
    h = sess.run(it.string_handle())

    probs = sess.run(out, feed_dict={handle_ph: h, train_flag: False})[0]

    # drop bogus (braai already gates real/bogus), renormalize, argmax
    classes = ["AGN", "SN", "VS", "asteroid", "bogus"]
    astro = {c: float(p) for c, p in zip(classes, probs) if c != "bogus"}
    total = sum(astro.values())
    astro = {c: p / total for c, p in astro.items()}
    return max(astro, key=astro.get)


def main():
    from sklearn.metrics import confusion_matrix, classification_report

    # load the frozen graph once
    g = tf.Graph()
    with g.as_default():
        saver = tf.compat.v1.train.import_meta_graph(CKPT + "/model.meta")
        sess = tf.compat.v1.Session()
        saver.restore(sess, CKPT + "/model")
        handle_ph  = g.get_tensor_by_name("iterators/Placeholder_6:0")
        train_flag = g.get_tensor_by_name("inputs/training_flag:0")
        out        = g.get_tensor_by_name("outputs/Softmax:0")
        print("MODEL LOADED — frozen graph restored\n")

        client = Alerce()
        y_true, y_pred = [], []
        for cls in ASTRO:
            oids = find_objects_of_class(client, cls, N_PER_CLASS)
            for oid in oids:
                try:
                    pred = predict_local(sess, g, handle_ph, train_flag, out, client, oid)
                except Exception as e:
                    print(f"  {oid} ({cls}): skipped ({e})"); continue
                y_true.append(cls); y_pred.append(pred)
                print(f"  {oid}: true={cls:9s} local_pred={pred}")

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    if len(y_true) == 0:
        print("\nNo objects scored (ALeRCE returned none / all skipped)."); return
    acc = (y_true == y_pred).mean()
    print(f"\n=== LOCAL 2020 stamp CNN vs ALeRCE labels ({len(y_true)} objects, acc {acc:.3f}) ===")
    print("confusion (rows=true, cols=pred), labels=" + str(ASTRO))
    print(confusion_matrix(y_true, y_pred, labels=ASTRO))
    print(classification_report(y_true, y_pred, labels=ASTRO, zero_division=0))
    print("\nSCOPE: labels are ALeRCE's NEWER model, so this is local-2020-vs-ALeRCE")
    print("       AGREEMENT (drift), on objects genuinely in the 4 classes.")


if __name__ == "__main__":
    main()
