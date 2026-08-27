# Stage 4 — Real/Bogus Gate (braai) — Build Spec

> Hand this file to the agent. It sets up a **real/bogus detection gate** for a ZTF
> time-series sky-image pipeline using the **pretrained `braai` CNN**. Object-type
> classification (asteroid / SN / etc.) is a **later** phase and out of scope here.

---

## 1. Objective & scope

Build a working real/bogus gate that takes a ZTF-style image triplet (science,
reference, difference) and returns `P(real)` in `[0, 1]`.

- **Model:** pretrained `braai` (dmitryduev/braai). **Do NOT train a model.** braai is
  used as-is for inference.
- **In scope now:** environment setup, data access, inference, threshold selection,
  evaluation, and bridging to the project's own Stage-1–3 cutouts.
- **Deferred (do not build now):** a custom CNN in the ALeRCE mold + benchmarking it
  against braai. See §9.

Mental model — three slots, wired in series:

```
[ data + labels ]        [ model ]              [ threshold + evaluate ]
 alerce client   ---->    pretrained braai  ---->  cutoff on P(real), scored vs labels
 (ZTF triplets)          (no training)
```

`alerce` is a **data/label source**, braai is the **model**. They are not chained into
each other; the faucet feeds the model.

---

## 2. Prerequisites

- Python 3.9–3.11
- Packages: `tensorflow`, `numpy`, `astropy`, `alerce`, `matplotlib`, `scikit-learn`
- Existing Stage-1–3 codebase already produces per-candidate cutouts + a feature table
  (used only in §7 Step 5).

```bash
git clone https://github.com/dmitryduev/braai.git   # provides models/ + example notebooks
pip install tensorflow numpy astropy alerce matplotlib scikit-learn
```

**Model files:** inspect `braai/models/`. Each model is a pair:
`{name}.architecture.json` + `{name}.weights.h5`. Use the **latest** pair available
(e.g. `d6_m9`). The normalization must match the model — see §8.

---

## 3. Core inference module

Create `braai_realbogus.py`. Preprocessing and model-loading are copied from braai's own
`nb/braai_run.ipynb` so inputs match braai's training distribution.

```python
import os
import numpy as np
from tensorflow.keras.models import model_from_json


def load_braai(models_dir, base_name="d6_m9"):
    """Load braai from its architecture.json + weights.h5 pair."""
    with open(os.path.join(models_dir, f"{base_name}.architecture.json")) as f:
        model = model_from_json(f.read())
    model.load_weights(os.path.join(models_dir, f"{base_name}.weights.h5"))
    return model


def make_triplet(science, template, difference, l2_normalize=True):
    """Inputs: three 2D ZTF cutouts. Output: (63,63,3) float array.
    Channels ordered science, template(reference), difference."""
    channels = []
    for data in (science, template, difference):
        data = np.nan_to_num(data.astype(np.float32))       # NaNs -> 0
        if l2_normalize:                                     # per-cutout L2 norm
            norm = np.linalg.norm(data)
            if norm > 0:
                data = data / norm
        h, w = data.shape                                    # pad up to 63x63
        if (h, w) != (63, 63):
            data = np.pad(data, [(0, 63 - h), (0, 63 - w)],
                          mode="constant", constant_values=1e-9)
        channels.append(data)
    return np.stack(channels, axis=-1)                       # (63, 63, 3)


def braai_score(model, triplet):
    """Return P(real) in [0,1]. Closer to 1 = real, closer to 0 = bogus."""
    return float(model.predict(np.expand_dims(triplet, 0), verbose=0)[0, 0])
```

---

## 4. Data access — native ZTF triplets via ALeRCE

The `alerce` client returns ZTF's own alert cutouts (braai's exact training
distribution) and doubles as a weak-label source (its 5-class stamp classifier).

```python
from alerce.core import Alerce
client = Alerce()

# Native ZTF science/template/difference cutouts as a 3-HDU FITS HDUList
hdul = client.get_stamps("ZTF18abee782", survey="ztf", format="HDUList")
science, template, difference = hdul[0].data, hdul[1].data, hdul[2].data
```

**Verify HDU order once** by plotting the three cutouts: science + template should look
like the star field; difference should be mostly noise with a central blob. If order is
swapped, fix indexing before scoring.

---

## 5. Task list (execute in order; each has an acceptance criterion)

**Step 1 — First score.** Wire §3 + §4 together and score a handful of ZTF object IDs.
- ✅ Done when a `P(real)` score prints for each object.

**Step 2 — Sanity check vs ZTF's own rb.** For the same objects, fetch ZTF's built-in
real/bogus score via `client.query_detections(oid)` (the `rb` field) and compare.
- ✅ Done when obviously-real sources score high on both braai and `rb`. They need not
  match exactly (different models); a strong disagreement on obvious cases means a
  preprocessing bug.

**Step 3 — Batch scoring + threshold.** Score a larger, mixed set (obvious reals +
obvious artifacts). Plot the score distribution. Choose a real/bogus cutoff.
- Start at `P(real) >= 0.5`; tune using §6.
- ✅ Done when a chosen threshold cleanly separates the eyeballed reals from artifacts.

**Step 4 — Evaluation.** Assemble a labeled evaluation set (§6) and measure braai at the
chosen threshold.
- ✅ Done when precision, recall, and PR-AUC are reported (NOT raw accuracy — see §8).

**Step 5 — Bridge to own Stage-1–3 cutouts.** Feed the project's own difference-imaging
cutouts through `make_triplet` + braai.
- Match cutout size to 63×63, apply the same L2 normalization and sci/ref/diff channel
  order. Expect some domain shift (own difference imaging ≠ ZTF's).
- ✅ Done when braai produces sensible scores on a hand-labeled subset of the project's
  own candidates, evaluated as in Step 4.

---

## 6. Labels for evaluation

braai **inference needs no labels** (triplets in → scores out). Labels are needed only to
*evaluate* braai (and later to train a custom model).

- **Weak labels:** ALeRCE's class per object (collapse its 4 astrophysical classes →
  `real`, keep `bogus`). Treat as noisy.
- **Gold set:** hand-vet a few hundred triplets by eye. This is the trustworthy set for
  final metrics. Keep it drawn representatively (mostly bogus, reflecting reality).

---

## 7. Evaluation metrics

Class imbalance is extreme (most candidates are bogus), so **do not report raw accuracy**
— a model calling everything bogus would look ~99% accurate and be useless. Report:

- Precision and recall at the chosen threshold
- Precision–recall AUC (and ROC-AUC)
- Confusion matrix at the threshold
- The purity/completeness tradeoff as the threshold moves

---

## 8. Constraints & known gotchas

- **Normalization must match the model.** L2 norm (`l2_normalize=True`) is correct for
  models **newer than `d6_m7`**. For `d6_m7` or older, use `tf.keras.utils.normalize`
  instead. Mismatch = "runs but scores are garbage."
- **Channel order** is science, template(reference), difference. Confirm before scoring.
- **TensorFlow/Keras version.** Loading braai's older `.json` + `.h5` usually works in
  TF2; if `model_from_json`/`load_weights` throws, it's a Keras-version mismatch, not a
  code bug.
- **Domain shift.** braai "just works" on native ZTF alert stamps. On the project's own
  Stage-1–3 cutouts, preprocessing must be matched and some accuracy drop is expected.
- **ALeRCE labels are model predictions**, not ground truth — weak supervision only.
- **Higher score = more real.** braai output is `P(real)`.

---

## 9. Deferred — do NOT build now

A custom CNN in the ALeRCE-stamp-classifier mold (rotation-invariant CNN on the triplet +
metadata), benchmarked against braai. This is the "learn neural-net internals" phase and
is explicitly out of scope. braai occupies the model slot for now.

---

## 10. Definition of done

- `braai_realbogus.py` loads a pretrained braai model and scores native ZTF triplets.
- A real/bogus threshold is selected and justified against a labeled set.
- braai is evaluated with precision/recall/PR-AUC (not accuracy) on a hand-vetted gold set.
- The same path runs on the project's own Stage-1–3 cutouts with matched preprocessing.
