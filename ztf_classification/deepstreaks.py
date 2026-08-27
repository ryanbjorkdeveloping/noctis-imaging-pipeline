"""
DeepStreaks motion/streak branch — core inference library (Phase 1).

Loads the pretrained DeepStreaks CNN families (rb / kd / sl) and runs the
3-gate cascade on 144x144x1 streak cutouts. Pretrained ONLY — no training.

Runs in the isolated .venv_streaks (TF 2.15 / Keras 2.15), NOT the main venv:
    .venv_streaks/bin/python streak_testing/<script>.py

Version wall (same as braai): the models were serialized with Keras 2.2.4, so
model_from_json loads the architecture natively here, but load_weights() fails
on the old HDF5 group layout -> we load weights MANUALLY BY NAME via h5py.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import json
import h5py
import numpy as np 
from keras.models import model_from_json

# project root -> the vendored DeepStreaks clone
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "DeepStreaks", "service", "models")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "DeepStreaks", "service", "code", "config.json")

# the 3 families, each an ensemble of 3 architectures. "pass" = >=1 member > 0.5. 
FAMILIES = ("rb", "kd", "sl")
ARCHS = ("vgg6", "resnet50", "densenet121")
THRESHOLD = 0.5

# load a single model: architecture from JSON, weights MANUALLY via h5py.
#
# Why manual: the .weights.h5 files were written by Keras 2.2.4, and modern
# h5py-3 can't read their attribute strings the way Keras' own load_weights
# expects (fails with KeyError: 'vars' doesn't exist), even under tf-keras.
# Downgrading h5py<3 won't build on arm64/py3.11, so we parse the H5 ourselves.
#
# The load-bearing subtlety (cost hours to find): weight ORDER must come from
# each layer group's `weight_names` attribute — the authoritative order Keras
# saved. A naive sorted() puts BatchNorm 'beta' before 'gamma' (alphabetical)
# and silently corrupts every BN layer in ResNet/DenseNet. We also resolve each
# dataset by walking the group recursively (DenseNet nests deeper than 1 level).
def _load_one(base_name):
    base = os.path.join(MODELS_DIR, base_name)
    with open(base + ".architecture.json") as f:
        model = model_from_json(f.read())

    by_name = {layer.name: layer for layer in model.layers}
    with h5py.File(base + ".weights.h5", "r") as f:
        top_names = [n.decode() if isinstance(n, bytes) else n
                     for n in f.attrs["layer_names"]]
        for ln in top_names:
            grp = f[ln]
            weight_names = grp.attrs.get("weight_names")
            if weight_names is None or len(weight_names) == 0:
                continue                      # weightless layer (pool/dropout/flatten)
            weight_names = [w.decode() if isinstance(w, bytes) else w
                            for w in weight_names]

            # collect every dataset in this layer group, keyed by basename
            found = {}
            def _collect(g):
                for k in g.keys():
                    item = g[k]
                    if isinstance(item, h5py.Dataset):
                        found[k] = item[()]
                    else:
                        _collect(item)
            _collect(grp)

            # set weights in the file's authoritative order (kernel/bias, or
            # gamma/beta/moving_mean/moving_variance for BatchNorm)
            ordered = [found[w.split("/")[-1]] for w in weight_names]
            by_name[ln].set_weights(ordered)
    return model

#Load all 9 models -> nested dict {family: {arch: model}}. Uses the 'models'
# block of config.json to map family+arch -> the on-disk base name.
def load_deepstreaks():
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)["models"]

    models = {fam: {} for fam in FAMILIES}
    for fam in FAMILIES:
        for arch in ARCHS:
            base_name = cfg[f"{fam}_{arch}"]
            models[fam][arch] = _load_one(base_name)
    return models

# Score a batch of (N,144,144,1) images through every model. 
# Returns {family: {arch: (N,) scores}}. 
def score_families(models, batch):
    out = {}
    for fam in FAMILIES:
        out[fam] = {}
        for arch in ARCHS:
            preds = models[fam][arch].predict(batch, verbose=0)
            out[fam][arch] = preds.reshape(-1) # (N, 1) -> (N,)
    return out

# apply the DeepStreaks cascade to image i using the ensemble "OR" rule. 
# returns a dict: per-family pass/fail + a final routing label. 
def cascade_decision(scores, i):
    def family_pass(fam):
        return any(scores[fam][arch][i] > THRESHOLD for arch in ARCHS)

    rb_pass = family_pass("rb")
    kd_pass = family_pass("kd")
    # sl POLARITY (resolved empirically on the synthetic smoke test): the ensemble
    # scores known-SHORT streaks ~1.0 and known-LONG streaks ~0.55, so sl_pass
    # (>0.5) means SHORT (= a fast asteroid), NOT long. Do not flip this without
    # re-running the sl polarity probe in test_cascade_synthetic.py.
    sl_short = family_pass("sl")

    if not rb_pass:
        route = "BOGUS"
    elif not kd_pass:
        route = "COSMIC_RAY"
    elif sl_short:
        route = "SHORT_NEA_CANDIDATE"
    else:
        route = "LONG_SATELLITE"

    return {
        "rb_pass": rb_pass,
        "kd_pass": kd_pass,
        "sl_short": sl_short,
        "route": route,
    }