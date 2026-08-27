"""
Path A-2 — type OUR OWN Stage-3 detections with the local ALeRCE stamp CNN.
Image  <- our Stage-3 (3,63,63) .npy triplet   (not ALeRCE's get_stamps)
Metadata <- pathA_metadata.build_metadata + normalize_metadata  (not get_avro)
The graph load + cyclic feed + drop-bogus argmax are reused from pathA_stamp_probe.py.
Run:  python stamp_testing/pathA_our_detections.py
"""

import os, sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np 
import tensorflow as tf 
tf.compat.v1.disable_v2_behavior()

from astropy.table import Table
from ztf_classification.stamp_classifier import center_crop
from ztf_classification.pathA_metadata import(
    read_fits_header_fields, load_psfcat, build_metadata, normalize_metadata,
)

CLASSES = ["AGN", "SN", "VS", "asteroid", "bogus"]

CAT = os.path.join(
    PROJECT_ROOT, 
    "ztfdata", 
    "catalog",
    "ztf_20180322273264_000468_zr_c03_o_q2_stage3_catalog.ecsv"
)

SCI = os.path.join(
    PROJECT_ROOT,
    "ztfdata",
    "sci",
    "2018",
    "0322",
    "273264",
    "ztf_20180322273264_000468_zr_c03_o_q2_sciimg.fits"
)

PSFCAT = os.path.join(
    PROJECT_ROOT,
    "ztfdata",
    "sci",
    "2018",
    "0322",
    "273264",
    "ztf_20180322273264_000468_zr_c03_o_q2_psfcat.fits"
)

CUTOUTS = os.path.join(PROJECT_ROOT, "ztfdata", "ztf_own_pipeline_data_testing_small_scale", "cutouts")

CKPT = os.path.join(
    PROJECT_ROOT,
    "stamp_classifier_model",
    "results",
    "staging_model",
    "DeepHits_EntropyRegBeta0.5000_batch64_lr0.00100_droprate0.5000_inputsize21_filtersize5_0_20200708-160759",
    "checkpoints",
)

NORM = CKPT.replace("/checkpoints", "") + "/feature_norm_stats.pkl"

SURVIVORS = {"src_p1_003", "src_p1_024", "src_m1_009"}


# image builder

#our (3,63,63) triplet -> (21, 21, 3) model cube, image-normalized to [0,1].
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

#main setup and graph restore

def main():
    cat = Table.read(CAT)
    rows = [r for r in cat
            if os.path.basename(str(r["stamp_path"])).replace(".npy", "") in SURVIVORS]
    print(f"Loaded catalog: {len(cat)} sources; {len(rows)} braai-passed survivors\n")

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
        print("MODEL LOADED — frozen graph restored\n")

        for r in rows:
            name = os.path.basename(str(r["stamp_path"])).replace(".npy", "")

            #IMAGE half: our .npy triplet -> (21,21,3), fed as its 4 rotations
            cube = build_image(os.path.join(CUTOUTS, name + ".npy"))
            rots = np.stack([np.rot90(cube,k) for k in range(4)], axis=0).astype(np.float32)

            #METADATA half: build_metadata + normalize_metadata
            #our 23-field vector, built + normalized locally
            meta = build_metadata(float(r["ra"]), float(r["dec"]), float(r["sign"]),
                                  float(r["segment_flux"]), float(r["snr"]),
                                  hdr_fields, psfcat)
            meta_vec = normalize_metadata(meta, NORM)

            # feed the graph via its tf.data iterator (image=4 rows, metadata=1 row)
            lbl = np.zeros((1,), dtype=np.int64)
            img_ds  = tf.compat.v1.data.Dataset.from_tensor_slices(rots).batch(4)
            meta_ds = tf.compat.v1.data.Dataset.from_tensor_slices(meta_vec).batch(1)
            lbl_ds  = tf.compat.v1.data.Dataset.from_tensor_slices(lbl).batch(1)
            ds = tf.compat.v1.data.Dataset.zip((img_ds, meta_ds, lbl_ds))
            it = tf.compat.v1.data.make_initializable_iterator(ds)
            sess.run(it.initializer)
            h = sess.run(it.string_handle())

            probs = sess.run(out, feed_dict={handle_ph: h, train_flag: False})[0]

            # VERDICT: drop bogus (braai already has that shit), renormalize, argmax
            astro = {c: float(p) for c, p in zip(CLASSES, probs) if c != "bogus"}
            total = sum(astro.values())
            astro = {c: p / total for c, p in astro.items()}
            top = max(astro, key=astro.get)
            
            sgn = "+" if float(r["sign"]) > 0 else "-"
            print(f"=== {name}  (isdiffpos {sgn}, dist-to-star {meta['distpsnr1']:.2f}\", "
                  f"sgscore {meta['sgscore1']:.2f}) ===")
            print("  5-class softmax:  " +
                  "  ".join(f"{c}={p:.3f}" for c, p in zip(CLASSES, probs)))
            print("  after drop-bogus: " +
                  "  ".join(f"{c}={astro[c]:.3f}" for c in ["AGN", "SN", "VS", "asteroid"]))
            print(f"  >>> TYPE: {top}  (P={astro[top]:.3f})\n")

    print("SCOPE: the 2020 checkpoint discriminates but is NOT calibrated to ALeRCE's")
    print("       scale; our metadata is approximated (PS1-proxy sgscore, psfcat chinr).")
    print("       Trust the relative drop-bogus ranking, not absolute P.\n")


if __name__ == "__main__":
    main()