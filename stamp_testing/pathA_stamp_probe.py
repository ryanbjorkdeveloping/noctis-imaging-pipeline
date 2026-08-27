import os, sys

# project root on path to reuse center_crop function
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
from alerce.core import Alerce
from ztf_classification.stamp_classifier import center_crop

# metadata assembly deps
import io, pickle, math
import ephem
import fastavro

# all 23 metadata fields IN THE MODEL'S EXACT ORDER (position is load-bearing)
FEATURES = ["sgscore1","distpsnr1","sgscore2","distpsnr2","sgscore3","distpsnr3",
            "isdiffpos","fwhm","magpsf","sigmapsf","ra","dec","diffmaglim","classtar",
            "ndethist","ncovhist","ecl_lat","ecl_long","gal_lat","gal_long",
            "non_detections","chinr","sharpnr"]

# the model's per-field CLIP bounds (from stamp_clf.py); "max"/"min" = no bound that side.
# This is what tames the ZTF -999 "no data" sentinels before normalization.
CLIP = {"sgscore1":[-1,None],"distpsnr1":[-1,None],"sgscore2":[-1,None],"distpsnr2":[-1,None],
        "sgscore3":[-1,None],"distpsnr3":[-1,None],"fwhm":[None,10],"ndethist":[None,20],
        "ncovhist":[None,3000],"chinr":[-1,15],"sharpnr":[-1,1.5],"non_detections":[None,2000]}

# 1. the pretrained checkpoint (ALeRCE 2021 stamp classifier, vendored under
#    stamp_classifier_model/ — gitignored like braai/; ~7MB TF1 checkpoint).
CKPT = os.path.join(
    PROJECT_ROOT, "stamp_classifier_model", "results", "staging_model",
    "DeepHits_EntropyRegBeta0.5000_batch64_lr0.00100_droprate0.5000_inputsize21_filtersize5_0_20200708-160759",
    "checkpoints",
)

# 2. load the frozen graph + weights (replaces StampClassifier, which won't instantiate)
g = tf.Graph()
with g.as_default():
    saver = tf.compat.v1.train.import_meta_graph(CKPT + "/model.meta")
    sess = tf.compat.v1.Session()
    saver.restore(sess, CKPT + "/model")
    handle_ph  = g.get_tensor_by_name("iterators/Placeholder_6:0")   # string-handle feed
    train_flag = g.get_tensor_by_name("inputs/training_flag:0")
    out        = g.get_tensor_by_name("outputs/Softmax:0")
    print("MODEL LOADED — frozen graph restored")

    # 3. fetch a known asteroid's real stamps (ALeRCE: asteroid 1.0)
    client = Alerce()
    oid = "ZTF17aaaacet"
    hdul = client.get_stamps(oid, format="HDUList")
    sci, ref, diff = hdul[0].data, hdul[1].data, hdul[2].data

    # build the (21,21,3) cube: center-crop 63->21, scrub NaNs
    cube = np.stack(
        [np.nan_to_num(center_crop(x.astype(np.float32))) for x in (sci, ref, diff)],
        axis=-1,
    )

    # --- IMAGE NORMALIZATION (the load-bearing step): per-image, per-channel min-max to [0,1] ---
    # The model was trained on normalized images; feeding raw pixel counts reads as "bogus".
    imgs = cube[np.newaxis, ...]                                   # (1,21,21,3)
    imgs = imgs - np.nanmin(imgs, axis=(1, 2))[:, None, None, :]
    imgs = imgs / np.nanmax(imgs, axis=(1, 2))[:, None, None, :]
    cube = np.nan_to_num(imgs[0]).astype(np.float32)

    # feed the image as its 4 rotations (cyclic-pooling expects a multiple of 4)
    rots = np.stack([np.rot90(cube, k) for k in range(4)], axis=0).astype(np.float32)

    # 4. assemble the real 23-field metadata from the ZTF alert packet
    alert = list(fastavro.reader(io.BytesIO(client.get_avro(oid))))[0]
    cand = alert["candidate"]

    # 4 coord fields + non_detections + isdiffpos aren't directly in `cand`:
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
        return float(cand[name])                                  # the 18 avro fields

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

    # 5. feed the graph (image 4-rotations + 1 metadata row) and predict
    lbl = np.zeros((1,), dtype=np.int64)
    img_ds  = tf.compat.v1.data.Dataset.from_tensor_slices(rots).batch(4)
    meta_ds = tf.compat.v1.data.Dataset.from_tensor_slices(meta).batch(1)
    lbl_ds  = tf.compat.v1.data.Dataset.from_tensor_slices(lbl).batch(1)
    ds = tf.compat.v1.data.Dataset.zip((img_ds, meta_ds, lbl_ds))
    it = tf.compat.v1.data.make_initializable_iterator(ds)
    sess.run(it.initializer)
    h = sess.run(it.string_handle())

    probs = sess.run(out, feed_dict={handle_ph: h, train_flag: False})[0]

    classes = ["AGN", "SN", "VS", "asteroid", "bogus"]
    print(f"\n--- ALeRCE stamp classifier (local, oid={oid}) ---")
    for c, p in zip(classes, probs):
        print(f"  {c:9s} : {p:.4f}")

    # 6. THE VERDICT: drop bogus (braai already gated real/bogus), renormalize, argmax
    astro = {c: float(p) for c, p in zip(classes, probs) if c != "bogus"}
    total = sum(astro.values())
    astro = {c: p / total for c, p in astro.items()}
    top = max(astro, key=astro.get)
    print("\nafter dropping bogus + renormalizing (Path B rule):")
    for c in ["AGN", "SN", "VS", "asteroid"]:
        print(f"  {c:9s} : {astro[c]:.4f}")
    print(f"\n>>> VERDICT: {top}  (P={astro[top]:.3f})")
    print(f"    ALeRCE says: asteroid  ->  {'MATCH' if top == 'asteroid' else 'MISMATCH'}")
