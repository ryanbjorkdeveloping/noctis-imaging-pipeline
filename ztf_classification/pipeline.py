"""
Stage-4 full pipeline — orchestration layer.

Connects the four already-built classifiers (braai real/bogus, the stamp-type
classifiers, and the DeepStreaks motion branch) into ONE end-to-end run over our
Stage-3 detections, emitting a single verdict per detection. Orchestration only —
no new models, no training. See notebook/stage4_pipeline.ipynb for the tutored
rung-by-rung build.

braai / stamp Path A+B run in the main .venv. The DeepStreaks streak branch runs
in .venv_streaks via a subprocess (Keras-version wall) — see run_streak_subprocess.
"""

import sys, os
import numpy as np

# pipeline.py lives inside ztf_classification/, so the project root is one level
# up. Put it on sys.path FIRST so `from ztf_classification.<x> import ...` resolves
# even when this module is imported standalone (not just from the notebook).
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ztf_classification.braai_realbogus import load_braai, make_triplet, braai_score

from ztf_classification.stamp_classifier import center_crop

# The frozen graph's output order (softmax slots). 'bogus' is dropped at argmax
# because braai already answered real/bogus upstream.
PATHA_CLASSES = ["AGN", "SN", "VS", "asteroid", "bogus"]

# NOTE: Path A needs TF1 graph mode (tf.compat.v1.disable_v2_behavior), but braai
# needs TF2 eager mode. We must NOT disable v2 at import — that would break braai.
# The switch is deferred into run_pathA_type, so braai (Rungs 1-2) runs eager, and
# only once Path A opens its frozen graph do we flip to TF1 mode. Because the switch
# is process-global and one-way, run all braai scoring BEFORE calling run_pathA_type.


def run_braai_gate(catalog, cutouts_dir, model):
    """Score every detection with braai. Returns a list of dicts, one per row.

    Each dict is that detection's growing record — later rungs (type, streak,
    verdict) add fields to it, joined by row_id.
    """
    results = []
    for row in catalog:
        # resolve the cutout by basename (the catalog's stamp_path is stale)
        stamp_path = os.path.join(cutouts_dir, os.path.basename(str(row["stamp_path"])))
        row_id = os.path.splitext(os.path.basename(stamp_path))[0]   # e.g. "src_p1_003"

        npy = np.load(stamp_path)                       # (3, 63, 63)
        triplet = make_triplet(npy[0], npy[1], npy[2])  # sci, ref, diff
        p_real = braai_score(model, triplet)

        results.append({
            "row_id": row_id,
            "ra": float(row["ra"]),
            "dec": float(row["dec"]),
            "elongation": float(row["elongation"]),
            "sign": int(row["sign"]),
            "segment_flux": float(row["segment_flux"]),
            "snr": float(row["snr"]),
            "on_edge": bool(row["on_edge"]),
            "stamp_path": stamp_path,
            "braai_p_real": p_real,
            "braai_pass": p_real >= 0.5,
        })
    return results


def run_braai_gate_native(catalog, cutouts_dir, model, client, radius_arcsec=5.0):
    """Like run_braai_gate, but score braai on ZTF's OWN native (sci,ref,diff)
    stamp when the detection matches an ALeRCE object — our homemade cutouts carry
    a registration DIPOLE that braai correctly rejects (own-stamp recall ~0.11),
    while ZTF's native stamps score the SAME objects ~0.78 (Stage-2c Rung G). Falls
    back to the homemade cutout when there's no ALeRCE match (a truly novel object).

    Same result-dict shape as run_braai_gate (so every downstream rung is unchanged),
    plus two audit fields: braai_stamp_source ("native"/"cutout") and braai_oid.
    """
    from ztf_classification.stamp_classifier import resolve_oid_for_coords

    results = []
    for row in catalog:
        stamp_path = os.path.join(cutouts_dir, os.path.basename(str(row["stamp_path"])))
        row_id = os.path.splitext(os.path.basename(stamp_path))[0]
        ra, dec = float(row["ra"]), float(row["dec"])

        # try ZTF's native stamp first (dipole-free); fall back to our cutout
        source, oid = "cutout", None
        triplet = None
        oid = resolve_oid_for_coords(client, ra, dec, radius_arcsec=radius_arcsec)
        if oid is not None:
            try:
                hdul = client.get_stamps(oid, format="HDUList")
                triplet = make_triplet(hdul[0].data, hdul[1].data, hdul[2].data)
                source = "native"
            except Exception:
                triplet = None                       # stamp fetch failed → fall back
        if triplet is None:
            npy = np.load(stamp_path)
            triplet = make_triplet(npy[0], npy[1], npy[2])
            source = "cutout"

        p_real = braai_score(model, triplet)
        results.append({
            "row_id": row_id,
            "ra": ra, "dec": dec,
            "elongation": float(row["elongation"]),
            "sign": int(row["sign"]),
            "segment_flux": float(row["segment_flux"]),
            "snr": float(row["snr"]),
            "on_edge": bool(row["on_edge"]),
            "stamp_path": stamp_path,
            "braai_p_real": p_real,
            "braai_pass": p_real >= 0.5,
            "braai_stamp_source": source,
            "braai_oid": oid,
        })
    return results


def run_pathB_type(results, client, radius_arcsec=2.0):
    """For each braai passer, ask ALeRCE for a type (Path B).
    Mutates each passer dict in place, adding pathB_* fields."""
    from ztf_classification.stamp_classifier import classify_detection, apply_threshold

    for r in results:
        if not r["braai_pass"]:
            continue
        out = classify_detection(client, r["ra"], r["dec"], radius_arcsec=radius_arcsec)
        r["pathB_matched"] = out["matched"]
        r["pathB_oid"] = out.get("oid")
        if out["matched"]:
            r["pathB_type"] = apply_threshold(out["top_class"], out["top_prob"])
            r["pathB_top_class"] = out["top_class"]
            r["pathB_prob"] = out["top_prob"]
        else:
            r["pathB_type"] = None            # NO-MATCH → Path A will handle it in Rung 3

    return results

# --- Path A helpers (lifted from stamp_testing/score_stamp_vs_labels.py) --------

def build_image(npy_path):
    """Our (3,63,63) triplet -> the model's (21,21,3) cube.
    Center-crop each channel 63->21, stack channel-last, min-max normalize [0,1]."""
    sci, ref, diff = np.load(npy_path)
    cube = np.stack(
        [np.nan_to_num(center_crop(x.astype(np.float32))) for x in (sci, ref, diff)],
        axis=-1,
    )
    imgs = cube[np.newaxis, ...]
    imgs = imgs - np.nanmin(imgs, axis=(1, 2))[:, None, None, :]
    imgs = imgs / np.nanmax(imgs, axis=(1, 2))[:, None, None, :]
    return np.nan_to_num(imgs[0]).astype(np.float32)


def predict_type(sess, g, handle_ph, train_flag, out, cube, meta_vec):
    """Feed one detection through the frozen graph -> drop-bogus argmax type.
    The model wants the image as its 4 rotations (cyclic pooling); metadata is
    one row. We fetch the softmax, drop 'bogus' (braai already gated it), argmax.
    Called only from run_pathA_type, i.e. after TF1 mode is switched on."""
    import tensorflow as tf
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
    astro = {c: float(p) for c, p in zip(PATHA_CLASSES, probs) if c != "bogus"}
    return max(astro, key=astro.get)

def run_pathA_type(results, hdr_fields, psfcat, ckpt, norm_path, cutouts_dir):
    """Fallback typing via the local 2020 stamp CNN, for braai passers Path B did
    NOT confidently type (NO-MATCH or 'uncertain'). Mutates dicts in place, adding
    pathA_type + pathA_calibrated. Opens the frozen TF1 graph ONCE, loops, closes.

    NB: disable_v2_behavior() is process-global and one-way — braai will not work
    again in this kernel after this runs, so call this AFTER all braai scoring."""
    import tensorflow as tf
    tf.compat.v1.disable_v2_behavior()   # flip to TF1 graph mode — braai is done by now
    from ztf_classification.pathA_metadata import build_metadata, normalize_metadata

    todo = [r for r in results
            if r["braai_pass"]
            and (not r.get("pathB_matched") or r.get("pathB_type") == "uncertain")]
    if not todo:
        print("Path A: nothing to do (Path B typed every passer).")
        return results

    g = tf.Graph()
    with g.as_default():
        saver = tf.compat.v1.train.import_meta_graph(ckpt + "/model.meta")
        sess = tf.compat.v1.Session()
        saver.restore(sess, ckpt + "/model")
        handle_ph  = g.get_tensor_by_name("iterators/Placeholder_6:0")
        train_flag = g.get_tensor_by_name("inputs/training_flag:0")
        out        = g.get_tensor_by_name("outputs/Softmax:0")
        print(f"Path A: model loaded, typing {len(todo)} detection(s)\n")

        for r in todo:
            cube = build_image(r["stamp_path"])
            meta = build_metadata(r["ra"], r["dec"], float(r["sign"]),
                                  r["segment_flux"], r["snr"], hdr_fields, psfcat)
            meta_vec = normalize_metadata(meta, norm_path)
            r["pathA_type"] = predict_type(sess, g, handle_ph, train_flag,
                                           out, cube, meta_vec)
            r["pathA_calibrated"] = False   # 2020 checkpoint: trust ranking, not absolute
    return results

def diff_gray_scale(diff_data):
    """Robust (median, noise-sigma) of the WHOLE diff, computed ONCE.
    The diff background is ~0 with noise; MAD*1.4826 ≈ the Gaussian sigma. Every
    stamp is scaled by this SHARED pair so the sky always lands at the same gray
    and only real sources deviate — a per-window min/max stretch (the earlier bug)
    manufactured a fake dynamic range per stamp and broke DeepStreaks' polarity."""
    d = np.nan_to_num(diff_data.astype(np.float32))
    med = float(np.median(d))
    sigma = 1.4826 * float(np.median(np.abs(d - med))) or 1.0
    return med, sigma


def cut_streak_stamp(diff_data, x, y, med, sigma, size=144, gray=0.82, span=12.0):
    """Cut a `size`×`size` window from the full diff at (x,y) and return it as a
    DARK-on-light-gray float32 stamp in [0,1] — matching DeepStreaks' training
    distribution (real ZTF _scimref streaks are dark-on-gray, NOT bright-on-black).

    Scaling uses the SHARED (med, sigma) from diff_gray_scale, so the noise sky
    always lands at `gray` and only real (bright) sources go dark. `span` sets how
    many sigma map to the full swing to black. Edge windows are gray-padded so
    every stamp is uniformly size×size.

    `span=12` (was 3.5): injection-calibrated. 3.5 mapped ±2σ diff NOISE to near-black,
    producing salt-and-pepper stamps the DeepStreaks `rb` gate calls bogus — so a REAL
    injected streak scored rb=0.000 (the branch could reject noise but never ACCEPT a
    streak; it was only ever validated in the negative direction on a streak-free field).
    span=12 keeps the noise sky gray so only the bright streak goes dark: an injected
    streak flips to rb=1.000/SHORT while the 60 real non-streak elongated detections on
    field 468 still all route BOGUS (0 false positives). See ztf_data/tests/test_streak_injection.py."""
    half = size // 2
    xi, yi = int(round(x)), int(round(y))
    H, W = diff_data.shape

    # gray canvas; copy in the valid overlap of the requested window
    stamp = np.full((size, size), gray, dtype=np.float32)
    y0, y1 = yi - half, yi + half            # source window in full-frame coords
    x0, x1 = xi - half, xi + half
    sy0, sx0 = max(0, -y0), max(0, -x0)      # where the valid region lands in the stamp
    fy0, fy1 = max(0, y0), min(H, y1)        # clipped window in the full frame
    fx0, fx1 = max(0, x0), min(W, x1)
    window = np.nan_to_num(diff_data[fy0:fy1, fx0:fx1].astype(np.float32))

    # FIXED scale: (pixel - sky median) in units of noise-sigma, then invert to
    # dark-on-gray. sky ≈ 0 sigma -> stays gray; bright source -> positive -> dark.
    dev = (window - med) / (sigma * span)     # ~0 for sky, ->1 for a strong source
    inv = np.clip(gray - dev, 0.0, 1.0)       # bright source -> dark; sky -> gray

    stamp[sy0:sy0 + inv.shape[0], sx0:sx0 + inv.shape[1]] = inv
    return stamp

def build_streak_batch(results, catalog, diff_data, elong_thresh=1.5, size=144):
    """Select elongated detections (elongation > thresh), cut a 144×144 dark-on-gray
    stamp for each from the full diff. Returns (stamps (M,144,144,1), row_ids list)."""
    med, sigma = diff_gray_scale(diff_data)   # ONE shared scale for the whole diff
    # map row_id -> catalog row for the full-frame centroid (x/y_centroid)
    by_id = {os.path.splitext(os.path.basename(str(r["stamp_path"])))[0]: r for r in catalog}
    stamps, ids = [], []
    for r in results:
        if r["elongation"] <= elong_thresh:
            continue
        if r["on_edge"]:
            continue                          # half-off-frame -> half-padded slab, skip
        crow = by_id[r["row_id"]]
        stamp = cut_streak_stamp(diff_data, float(crow["x_centroid"]), float(crow["y_centroid"]),
                                 med, sigma, size)
        stamps.append(stamp)
        ids.append(r["row_id"])
    batch = np.stack(stamps)[..., np.newaxis].astype(np.float32)  # (M,144,144,1)
    return batch, ids


_STREAK_PY = os.path.join(PROJECT_ROOT, ".venv_streaks", "bin", "python")
# module-level so ONE persistent server is reused across every call in this process
# (and, under a ProcessPool sweep, one per worker — module globals are per-process).
# None = not yet started; False = start failed / died -> stay on the one-shot fallback.
_STREAK_SERVER = None


def _write_streak_job(streak_batch, streak_ids, scratch_dir):
    """Write the (stamps, index) handoff files; return their paths + the results path."""
    import json
    os.makedirs(scratch_dir, exist_ok=True)
    stamps_npy = os.path.join(scratch_dir, "elongated_stamps.npy")
    index_json = os.path.join(scratch_dir, "elongated_index.json")
    results_json = os.path.join(scratch_dir, "streak_results.json")
    np.save(stamps_npy, streak_batch)
    with open(index_json, "w") as f:
        json.dump([{"row_id": rid} for rid in streak_ids], f)
    return stamps_npy, index_json, results_json


def _start_streak_server():
    """Launch the persistent .venv_streaks scorer and block until it reports 'ready'
    (the one-time ~9 s model load). Returns the Popen, or None if it failed to start —
    the caller then stays on the one-shot fallback. Registered for atexit shutdown so a
    sweep never leaks .venv_streaks processes."""
    import json, subprocess, atexit
    script = os.path.join(PROJECT_ROOT, "streak_testing", "streak_score_server.py")
    if not os.path.exists(_STREAK_PY):
        return None
    try:
        proc = subprocess.Popen([_STREAK_PY, script],
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                text=True, bufsize=1)
        line = proc.stdout.readline()                 # blocks through model load; '' on early death
        if not line or json.loads(line).get("status") != "ready":
            proc.kill()
            return None
    except Exception:
        return None
    atexit.register(lambda: (proc.stdin.close(), proc.wait()) if proc.poll() is None else None)
    return proc


def run_streak_subprocess(streak_batch, streak_ids, scratch_dir):
    """Hand the streak stamps to DeepStreaks running in .venv_streaks (a separate
    process — Keras-version wall: DeepStreaks is Keras 2.15, this venv is Keras 3,
    which cannot coexist). Returns {row_id: dict}.

    Uses a PERSISTENT server (loads the 9 models once, scores every subsequent batch on
    the warm models) so a multi-target sweep pays the ~9 s load once, not per target. If
    the server can't start or dies mid-sweep it transparently falls back to the original
    one-shot spawn (streak_score_batch.py) — same result either way, just slower."""
    import json
    global _STREAK_SERVER

    stamps_npy, index_json, results_json = _write_streak_job(
        streak_batch, streak_ids, scratch_dir)

    if _STREAK_SERVER is None:                         # first call: try to start one
        _STREAK_SERVER = _start_streak_server() or False

    if _STREAK_SERVER:                                 # warm path
        try:
            job = json.dumps({"stamps": stamps_npy, "index": index_json,
                              "out": results_json})
            _STREAK_SERVER.stdin.write(job + "\n")
            _STREAK_SERVER.stdin.flush()
            reply = _STREAK_SERVER.stdout.readline()   # '' if the server died
            if reply and json.loads(reply).get("status") == "ok":
                with open(results_json) as f:
                    return {d["row_id"]: d for d in json.load(f)}
            _STREAK_SERVER = False                     # err/EOF -> drop to one-shot for good
        except Exception:
            _STREAK_SERVER = False

    return _run_streak_oneshot(stamps_npy, index_json, results_json)


def _run_streak_oneshot(stamps_npy, index_json, results_json):
    """The original per-call spawn: launch streak_score_batch.py with the .venv_streaks
    interpreter, reloading the models each time. The correctness fallback for when the
    persistent server is unavailable."""
    import json, subprocess
    script = os.path.join(PROJECT_ROOT, "streak_testing", "streak_score_batch.py")
    subprocess.run([_STREAK_PY, script, "--stamps", stamps_npy,
                    "--index", index_json, "--out", results_json], check=True)
    with open(results_json) as f:
        return {d["row_id"]: d for d in json.load(f)}


def merge_verdict(r):
    """Collapse one detection's branch outputs into a single final_verdict string.
    Priority: braai bogus > streak (for elongated reals) > point-source type."""
    braai_real = r.get("braai_pass", False)
    streak = r.get("streak_route")            # None if not elongated

    if not braai_real:
        # braai gated it bogus — but DeepStreaks has its own rb gate, so surface
        # a streak call rather than letting braai's point-source rb veto it.
        if streak in ("SHORT_NEA_CANDIDATE", "LONG_SATELLITE"):
            return f"bogus (point-source) / streak: {streak}"
        return "bogus"

    # braai REAL: an elongated streak call outranks a point-source type
    if streak == "SHORT_NEA_CANDIDATE":
        return "fast_asteroid_candidate"
    if streak == "LONG_SATELLITE":
        return "satellite"

    # not a streak (or streak branch rejected it) -> the point-source type.
    # Path B wins when it gave a confident (non-uncertain) type; otherwise fall
    # through to Path A (the local CNN typed exactly the ones Path B was unsure of).
    pb = r.get("pathB_type")
    if pb not in (None, "uncertain"):
        return pb
    pa = r.get("pathA_type")
    if pa:
        return pa
    return "uncertain"

def build_verdict_table(results, out_path):
    """Collapse every detection to a final_verdict and write the audit .ecsv."""
    from astropy.table import Table

    rows = []
    for r in results:
        r["final_verdict"] = merge_verdict(r)
        # which path actually supplied the reported type (mirrors merge_verdict):
        # Path B if it was confident, else Path A if it typed the leftover.
        pb = r.get("pathB_type")
        if pb not in (None, "uncertain"):
            psrc_type, tpath, calibrated = pb, "alerce", True
        elif r.get("pathA_type"):
            psrc_type, tpath, calibrated = r["pathA_type"], "localCNN", False
        else:
            psrc_type, tpath, calibrated = "", "", False
        rows.append({
            "row_id": r["row_id"],
            "ra": r["ra"], "dec": r["dec"],
            "elongation": r["elongation"],
            "braai_p_real": round(r["braai_p_real"], 4),
            "braai_pass": r["braai_pass"],
            "point_source_type": psrc_type,
            "type_path": tpath,
            "type_calibrated": calibrated,
            "streak_route": r.get("streak_route") or "",
            "final_verdict": r["final_verdict"],
        })
    tab = Table(rows)
    tab.write(out_path, format="ascii.ecsv", overwrite=True)
    return tab
