# Stage 4 (Stationary Branch, Phase B) — Light-Curve Classifier — Claude Code Spec

> **For Claude Code.** Set up and configure the **ALeRCE light-curve classifier** — the
> multi-epoch type classifier that runs *after* the braai real/bogus gate and *after* the
> single-epoch stamp classifier. It is pretrained: a Balanced Hierarchical Random Forest.
> **Do not train a new model.** The library's API evolves, so **read the live repo and run
> its example notebook before adapting anything** (see Task 1).

---

## 1. Context — where this fits

Pipeline so far (already built / specced separately):

```
braai (real/bogus gate)  →  stamp classifier (early type, 1 image)  →  THIS: light-curve classifier
```

- The stamp classifier gives an **early** type from a single image.
- The light-curve classifier gives a **later, better-informed** type once an object has
  enough observation history. It reads the object's **brightness-vs-time behavior**, not a
  single image.
- **They are independent and coexist.** The light-curve classifier does **not** take the
  stamp classifier's output as input, and the two results are **never merged/averaged** into
  one score. Store both per object; treat the light-curve call as the trusted one once it exists.

**Hard prerequisite:** the light-curve classifier needs light curves with **≥6 detections in
the g band OR ≥6 in the r band**. Objects with fewer stay stamp-classifier-only.

---

## 2. Repositories & install

- **Model + features:** `github.com/alercebroker/lc_classifier` (branch `main`)
- **Reference notebook (READ + RUN FIRST):**
  `lc_classifier/examples/feature_extraction_from_ztf_data.ipynb`
- **Light-curve data source:** `alerce` client (`pip install alerce`, repo
  `alercebroker/alerce_client`) — to pull ZTF detections / light curves.

```bash
git clone https://github.com/alercebroker/lc_classifier.git
cd lc_classifier
python -m pip install -r requirements.txt
python -m pip install -e .        # plus matplotlib, tqdm for the examples
pip install alerce astropy pandas numpy scikit-learn
```

---

## 3. What the classifier consumes and produces

**Inputs:**
- Variability **features** computed from the ZTF light curve (period, parametric decay,
  autoregressive, statistical, etc. — the library's own extractors).
- **Colors** from AllWISE and ZTF photometry.

**Model:** Balanced Hierarchical Random Forest = **4 random forests** in a two-level tree:
1. Top level: **periodic / stochastic / transient**
2. Transient → SNIa, SNIbc, SNII, SLSN
3. Stochastic → blazar, QSO, AGN, YSO (and CV/Nova)
4. Periodic → LPV, Cepheid, RRL, DSCT, EB, Periodic-Other

**Output:** a two-level label (top class + subclass) with probabilities. **15 classes total.**

---

## 4. Task list (execute in order; each has an acceptance criterion)

**Task 1 — Run the reference notebook first.** Open and run
`examples/feature_extraction_from_ztf_data.ipynb` end-to-end on its sample data. Confirm the
current API for: (a) the expected input dataframe schema, (b) the feature-extraction call,
(c) loading and applying the pretrained hierarchical classifier.
- ✅ Done when the notebook produces features and a classification on its example, unmodified.

**Task 2 — Assemble light curves for our objects.** For each object that passed braai (and
was blob-detected as a stationary source), gather its multi-epoch photometry: time (MJD),
magnitude, band (g/r), and errors. Pull via the `alerce` client (object light curves /
detections) or from our own repeated-epoch measurements. Attach AllWISE + ZTF colors.
- Apply the ≥6-detections-in-g-or-r cut before proceeding.
- ✅ Done when qualifying objects have a light-curve table with colors.

**Task 3 — Transform to the library's format.** Convert our light curves into the exact
Pandas dataframe structure `lc_classifier` expects (the repo ships transform helpers — use
them, do not hand-build the schema). Note the library's built-in preprocessing already:
drops duplicate observations, discards noisy detections, discards bogus, requires >5
detections, and drops nan/inf values — align our data so it survives these filters.
- ✅ Done when our dataframe passes the library's preprocessing without dropping valid objects.

**Task 4 — Extract features.** Run the library's feature extractors over the formatted light
curves to produce the feature dataframe the classifier consumes.
- ✅ Done when a complete feature dataframe is produced for the qualifying objects.

**Task 5 — Run the hierarchical classifier.** Apply the pretrained Balanced Hierarchical
Random Forest to the features. Capture the two-level output (top class + subclass +
probabilities).
- ✅ Done when each qualifying object gets a two-level taxonomy label with probabilities.

**Task 6 — Store alongside the stamp result.** Persist both classifications per object (early
stamp call + later light-curve call), independently. Do **not** merge them.
- ✅ Done when each object record shows both results side by side, unmerged.

---

## 5. Constraints & known gotchas

- **API evolves — follow the repo, not memory.** `lc_classifier` has changed across
  versions. The `examples/` notebooks are authoritative for the current call signatures.
- **≥6 g or ≥6 r detections required.** Single- or few-epoch objects cannot be classified
  here; they remain stamp-classifier-only. This is the whole reason this is the *later* phase.
- **Data format is strict.** Use the library's transform tools to build the expected Pandas
  dataframe; hand-rolled schemas will fail its preprocessing.
- **Colors matter.** The model uses AllWISE + ZTF colors alongside light-curve features —
  don't omit them.
- **Feature-extractor versioning.** ALeRCE tracks multiple feature/classifier versions;
  compute features with the version the pretrained model expects (match the notebook).
- **Coexist, never merge.** No weighted combination of stamp + light-curve outputs, and no
  master threshold across models. Thresholds act within a single classifier's probabilities.
- **Accuracy expectation.** Top-level (periodic/stochastic/transient) is strong (~0.96
  precision, ~0.99 recall); the 15 fine subclasses are much harder (dropping toward ~0.57
  precision). Report per-level metrics; don't expect subclass-level accuracy to match the top.

---

## 6. Definition of done

- The reference notebook runs unmodified on its sample (known-good baseline).
- Our qualifying objects (≥6 g or ≥6 r detections) are transformed to the library's format,
  features are extracted, and the pretrained hierarchical classifier assigns a two-level
  taxonomy label with probabilities.
- Each object stores both the stamp (early) and light-curve (later) classifications
  independently — not merged.