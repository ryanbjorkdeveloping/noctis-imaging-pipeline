# Stage 4 (Moving/Streak Branch) — DeepStreaks — Claude Code Spec

> **For Claude Code.** Set up the **moving/streak classification branch** using the
> **pretrained DeepStreaks CNNs**. **Do not train any model.** The goal of Phase 1 is a
> **smoke test**: prove the models load and the gates behave correctly on inputs whose truth
> we control. Only then move to real data.
>
> **Read this first — the critical constraint:** unlike the stationary branch, DeepStreaks has
> **no public labeled dataset** to validate against. Its inputs came from ZTF's *internal*
> Streak pipeline (IPAC, inside the ZTF Solar-System framework), which is **not a public data
> product**. Do **not** attempt to download a "ZTF streak dataset" or an ALeRCE-style labeled
> streak stream — it does not exist. We validate with **synthetic streaks we generate
> ourselves**, then feed the models streaks detected in our own images.

---

## 1. Context — where this fits

This branch is **completely independent of braai and the stamp classifier**.

```
POINT-SOURCE BRANCH (built):   detections --> braai (real/bogus) --> stamp classifier --> light-curve classifier
STREAK BRANCH   (this spec):   streak cutouts --> DeepStreaks (its OWN real/bogus gate + type gates)
```

- DeepStreaks is fed by a **different detector** (streak/elongated-source finding), not the
  point-source difference-image alert path.
- It has its **own real/bogus gate** (`rb`). It does **not** consume braai's output. Never wire
  braai into DeepStreaks — the inputs are different shapes (point sources vs. streaks).

---

## 2. What DeepStreaks is

- **Repo (pretrained models included, ~280 MB):** `github.com/dmitryduev/DeepStreaks`
- **Input:** a single **144×144×1 grayscale** streak cutout. (Note: NOT a sci/ref/diff triplet,
  NOT 21×21, NOT 63×63.)
- **Implementation:** Keras with a TensorFlow backend.
- **Architecture:** three *families* of **binary** CNN classifiers (each family is an ensemble;
  a streak passes a family if **at least one member scores > 0.5**):
  - **`rb`** — bogus (0) vs. **real streak** (1). NOTE: *all* streak-like objects count as real
    here, including fast-moving-object streaks, long satellite streaks, **and cosmic rays**.
  - **`kd`** — **keep vs. ditch** (filters cosmic rays / artifacts).
  - **`sl`** — **short vs. long** streak. **Short = fast asteroid. Long = satellite.**
- **Decision logic:** a streak must pass **all three** families (thresholds
  `threshold_rb=0.5`, `threshold_kd=0.5`, `threshold_sl=0.5`) to be a plausible **NEA candidate**.
- **Reference performance:** 96–98% true-positive rate, <1% false-positive rate.

```
streak cutout (144x144x1)
      -> rb  ---no--> bogus (discard)
      -> kd  ---no--> cosmic ray (discard)
      -> sl  --long-> SATELLITE (identified, then filtered out)
             --short-> NEA / fast-asteroid CANDIDATE
```

---

## 3. Install

```bash
git clone https://github.com/dmitryduev/DeepStreaks.git   # includes pretrained models (~280 MB)
pip install tensorflow numpy astropy scikit-image matplotlib
```

Inspect the repo's model directory and its `deepstreaks.py` for the exact model filenames,
loading pattern, and preprocessing. **Match the repo's own preprocessing exactly** — the same
class of bug as braai's normalization mismatch (runs fine, scores are garbage).

---

## 4. PHASE 1 — Synthetic smoke test (build this first)

**Purpose:** verify the implementation, not the science. We control the truth by construction,
so any wrong behavior is a wiring bug, not a model limitation.

**Task 1.1 — Load the pretrained models.** Load all three families (`rb`, `kd`, `sl`) from the
repo's pretrained weights. Report how many members each family has.
- ✅ Done when all three families load and their expected input shape prints as `(144,144,1)`.

**Task 1.2 — Generate a synthetic test set.** Write a generator producing 144×144 grayscale
images with known ground truth, in these categories:
- **Short streaks** (a bright line, modest length, random angle) → expect: real, keep, SHORT.
- **Long streaks** (a bright line spanning most of the frame) → expect: real, keep, LONG.
- **Point sources** (a compact Gaussian blob) → expect: NOT a real streak (fail `rb`).
- **Pure noise / blank** → expect: NOT a real streak (fail `rb`).
Add realistic Gaussian noise and vary brightness/angle/length. A few dozen of each is plenty.
- ✅ Done when a labeled synthetic set exists and sample images render sensibly when plotted.

**Task 1.3 — Run the cascade.** Score every synthetic image through `rb` → `kd` → `sl`, applying
the 0.5 thresholds and the "at least one family member passes" rule.
- ✅ Done when each image gets three scores and a final routing decision.

**Task 1.4 — Assert expected behavior.** Check the routing against the known truth:
- Short streaks → pass rb, pass kd, classified SHORT (NEA candidate).
- Long streaks → pass rb, pass kd, classified LONG (satellite; filtered out).
- Point sources and noise → fail `rb`.
- ✅ **Smoke test passes** when these hold for the clear-cut synthetic cases. If they don't,
  the bug is in preprocessing (scaling/normalization/input shape), not the model.

---

## 5. PHASE 2 — Streaks from our own Stage-1–3 images (after Phase 1 passes)

We have no access to ZTF's internal streak stream, so we build our **own streak detector** as
the front-end, then feed DeepStreaks.

**Task 2.1 — Detect elongated sources.** Run source detection on our Stage-1–3 **difference
images** and select **elongated** candidates. Our Stage-3 feature table already carries
`elongation` and `ellipticity` — filter on these (a streak is a high-elongation source, unlike
the compact point sources braai handles).
- ✅ Done when a list of elongated candidate positions is produced.

**Task 2.2 — Cut 144×144×1 stamps.** Extract a 144×144 grayscale cutout centered on each
elongated candidate. Must be large enough to contain the whole trail. Apply the repo's
preprocessing/normalization.
- ✅ Done when candidates yield valid `(144,144,1)` arrays.

**Task 2.3 — Run the cascade + review.** Score through rb/kd/sl. Visually inspect the ones that
pass all three (short-streak / NEA candidates) and the long-streak (satellite) rejects.
- ✅ Done when candidates are routed and a human has eyeballed the survivors.

**Expect a domain-shift accuracy drop.** DeepStreaks was trained on ZTF's own streak-pipeline
cutouts; our detection front-end and preprocessing differ. Phase 1 proves the wiring; Phase 2
performance on our data is an open question, not a guarantee.

---

## 6. Constraints & known gotchas

- **No public labeled streak dataset exists.** Do not search for one. Validate with synthetics
  (Phase 1), then our own detections (Phase 2).
- **Input is 144×144×1 grayscale** — a single channel, not a triplet. Do not reuse braai's or the
  stamp classifier's preprocessing.
- **Preprocessing must match the repo.** Follow `deepstreaks.py` exactly (scaling/normalization).
  Mismatch = plausible-looking but meaningless scores.
- **`rb` is permissive by design** — cosmic rays pass it. That's what `kd` is for. Don't "fix" it.
- **Satellites are a by-product, not the target.** DeepStreaks hunts fast asteroids; satellites
  are identified as LONG streaks and filtered out. If satellites matter to our project, capture
  the LONG class explicitly rather than discarding it.
- **This branch finds *fast/streaking* asteroids only.** Slow main-belt asteroids stay point-like
  and never streak — they require cross-match + cross-night linking (a different method, Stage 6),
  not DeepStreaks.
- **Never chain braai → DeepStreaks.** Separate detectors, separate gates, separate input shapes.
- **TensorFlow/Keras version.** The models are older Keras; expect possible compat handling on a
  modern TF2 install (same class of issue as braai).

---

## 7. Definition of done

- **Phase 1:** all three pretrained families load; synthetic short streaks, long streaks, point
  sources, and noise route exactly as expected through rb/kd/sl. (Implementation verified.)
- **Phase 2:** elongated sources detected in our own difference images are cut to 144×144×1,
  scored through the cascade, and routed into NEA candidates (short) vs. satellites (long) vs.
  rejects — with human visual review of the survivors.