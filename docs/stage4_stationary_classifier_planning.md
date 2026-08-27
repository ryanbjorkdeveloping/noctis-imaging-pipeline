# Stage 4 (Stationary Branch) — Object-Type Classification — Build Spec

> Hand this to your coding-editor AI tutor. It builds the **stationary classification
> branch** that runs *after* the braai real/bogus gate (already built — see the braai
> spec). Two pretrained ALeRCE models: the **stamp classifier** (Phase A, now) and the
> **light-curve classifier** (Phase B, later). **Do NOT train a custom model.**

---

## 1. Objective & scope

Take the real detections braai passed and assign each an **object type**.

- **Phase A — Stamp classifier (build now):** single-image type from the first detection.
  Runs on the same sci/ref/diff triplet braai uses. Pretrained CNN.
- **Phase B — Light-curve classifier (build later):** better-informed type from the
  brightness-vs-time behavior, once ≥6 epochs exist. Pretrained hierarchical random forest.
- **Out of scope:** the moving/streak branch (DeepStreaks) and any custom model training.

---

## 2. Mental model (how the pieces relate)

```
[ braai ]  --real-->  [ Stamp classifier ]        early call, 1 image
 gate                 (CNN, 5 classes)             (drop its 'bogus' class)
                            |
                            |  ... object re-observed on later nights ...
                            v
                      [ Light-curve classifier ]   later call, >=6 epochs
                      (hierarchical random forest, richer taxonomy)
```

Critical wiring facts (do not violate):
- The stamp and light-curve classifiers are **independent**. The light-curve classifier
  does **not** take the stamp classifier's output as input; it reads the light curve.
- They **coexist**: stamp = early call, light-curve = later, better-informed call. They are
  **never merged into one weighted score.** Do not average them.
- A **threshold** applies *within one classifier's own probabilities* (braai for real/bogus;
  each type classifier for its own classes). There is no master threshold fusing models.

---

## 3. Prerequisites & repositories

```bash
# Data + labels (ZTF stamps, classifications, light curves)
pip install alerce            # repo: alercebroker/alerce_client

# Stamp classifier — pretrained, runnable tutorial notebook
#   repo:   github.com/alercebroker/usecases
#   notebook: notebooks/ADASS_XXXII_Nov2022_tutorial/ALeRCE_ML_Stamp_Classifier.ipynb
#   (Colab-runnable; loads the pretrained model. Originally TensorFlow 1.14 — see gotchas.)

# Light-curve classifier — pretrained hierarchical random forest
git clone https://github.com/alercebroker/lc_classifier.git
cd lc_classifier
python -m pip install -r requirements.txt
python -m pip install -e .    # also: matplotlib, tqdm for the examples

# General stack
pip install astropy numpy pandas scikit-learn matplotlib
```

Upstream input: the triplets braai already scored and passed (from the braai spec).

---

## 4. PHASE A — Stamp classifier (build now)

**Inputs the model needs:**
1. A `21×21×3` image cube (science, reference, difference).
2. A vector of **alert metadata features** (coordinates, magnitudes, star/galaxy score,
   distances to nearest sources, seeing, etc.). The authoritative field list is Table 1 of
   Carrasco-Davis et al. 2021, defined by the ZTF avro alert schema:
   https://zwickytransientfacility.github.io/ztf-avro-alert/

**Task A1 — Get the pretrained model.** Open the `usecases` stamp-classifier notebook, run
it end-to-end once on its example so a known-good inference works before touching your data.
- ✅ Done when the notebook classifies its sample alerts into the 5 classes.

**Task A2 — Preprocessing bridge (braai 63×63 → stamp 21×21).** braai's triplets are 63×63;
the stamp classifier expects 21×21. Build a bridge that, for each braai-passed detection:
crops/resizes the sci/ref/diff cutouts to 21×21, applies the stamp classifier's own
normalization (match the notebook exactly), and assembles the metadata feature vector from
the same ZTF alert fields.
- ✅ Done when one of your detections produces a valid `(21,21,3)` cube + metadata vector.

**Task A3 — Run inference.** Feed cube + metadata to the pretrained model → softmax over
5 classes (SN, AGN, variable star, asteroid, bogus).
- Because braai already gated real/bogus, **drop the `bogus` probability** and take the
  argmax over the 4 astrophysical classes (or renormalize those 4).
- ✅ Done when each braai-passed detection gets an astrophysical class + probabilities.

**Task A4 — Thresholds + evaluation.** Choose a per-class confidence cutoff. Evaluate against
labels: ALeRCE's own classifications as weak labels (via the `alerce` client) plus a few
hundred you hand-vet as a gold set. Report precision/recall per class and a confusion matrix
(not raw accuracy).
- ✅ Done when per-class precision/recall are reported on a labeled set.

---

## 5. PHASE B — Light-curve classifier (build later)

Only start once objects have accumulated epochs. **Requires ≥6 detections in g OR ≥6 in r.**

**Task B1 — Assemble light curves.** For each object, gather its multi-epoch photometry
(time, magnitude, band, errors). Pull via the `alerce` client (object detections / light
curves) or from your own repeated-epoch measurements. Add colors from ZTF and AllWISE.
- ✅ Done when objects with ≥6 g or ≥6 r detections have a light-curve table.

**Task B2 — Format + feature extraction.** Convert light curves into the Pandas dataframe
format `lc_classifier` expects (it ships transform helpers). Run its feature extractors
(period, parametric decay, autoregressive, statistical features, colors).
- ✅ Done when a feature dataframe is produced for the objects.

**Task B3 — Run the hierarchical classifier.** Use `lc_classifier`'s hierarchical model
(a balanced set of 4 random forests). Output is two-level: top level = periodic / stochastic
/ transient; bottom level = the finer subclasses.
- ✅ Done when objects receive a two-level taxonomy label with probabilities.

**Task B4 — Reconcile with the stamp result.** Store both classifications per object; treat
the light-curve result as the trusted, better-informed call once available. **Do not average
the two.**
- ✅ Done when each object shows an early (stamp) and later (light-curve) classification,
  independently.

---

## 6. Constraints & known gotchas

- **Input size:** stamp classifier = `21×21×3`, braai = `63×63×3`. The crop/normalize bridge
  (Task A2) is mandatory; skipping it gives garbage classes.
- **Metadata is required**, not optional — the stamp classifier fuses image features with
  metadata *late* (image → CNN → features, metadata normalized separately, then concatenated).
  Provide the metadata vector or the model underperforms.
- **Drop the bogus class** in Phase A — braai already handles real/bogus. Keep the 4
  astrophysical classes.
- **TensorFlow version:** the stamp classifier was built on TF 1.14. Prefer running via the
  provided notebook / its pinned environment; a modern TF2 env may need compat handling.
- **Light-curve classifier needs history:** it cannot run until ≥6 g or ≥6 r detections exist.
  Single-epoch objects are stamp-classifier-only.
- **`lc_classifier` data format:** it expects a specific Pandas structure — use its transform
  tools rather than hand-building dataframes.
- **Never merge the two classifiers** into one weighted score, and never build one master
  threshold across them. Thresholds act inside a single classifier's probabilities.
- **Accuracy expectation:** single-epoch stamp = fast, coarse first cut; light-curve = finer
  but harder (top-level ~0.96 precision/recall, fine subclasses drop toward ~0.57). Set
  expectations accordingly.

---

## 7. Definition of done

- **Phase A:** braai-passed detections receive an astrophysical type from the pretrained
  stamp classifier (bogus dropped), with per-class thresholds and an evaluation on a labeled set.
- **Phase B (later):** objects with ≥6 g or ≥6 r detections receive a two-level light-curve
  taxonomy label from `lc_classifier`, stored alongside — not merged with — the stamp result.