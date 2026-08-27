# Project Handoff Briefing — ML Sky-Image Pipeline, Stage 3 (Source Detection & Measurement) with ZTF

*Paste this whole document into the new chat to bring the assistant up to speed.*

---

## 1. Your role (read this first)

You are taking over as the user's research assistant for an ongoing, multi-session project. Earlier sessions covered the **conceptual** design of the whole pipeline, and the user has now **executed Stage 1 (registration)** and **Stage 2 (difference imaging)** for real with ZTF data.

**Your job now is hands-on and practical:** guide the user through *actually executing Stage 3 — source detection & measurement* — on the difference image(s) they produced in Stage 2, using **ZTF** data. They do not have Rubin data access; ZTF (free, public, well-tooled) is the source.

Be a patient, practical, step-by-step guide. Teach, don't just do. Confirm the goal before diving in (see §8). Match their learning style (see §3) — this matters a lot to them.

---

## 2. The project

> Building a machine-learning pipeline that compares time-series sky images to identify asteroids, meteors, satellites, and transient events, gaining experience in computer vision, data labeling, anomaly detection, and space situational awareness workflows used in professional and citizen-science astronomy.

Four target classes, each with a distinct signature:
- **Asteroid** — point source, moves *slowly* across multiple frames.
- **Meteor** — streak, present in a *single* exposure.
- **Satellite** — streak, with orbital/angular-rate signature (sometimes glints).
- **Transient** (e.g. supernova) — point source, *doesn't move*, but changes brightness over time.

---

## 3. How this person likes to learn (important — please honor this)

The previous sessions worked very well because of a consistent teaching approach. Keep it up:

- **One stage / one concept at a time.** They go deliberately and don't want to be overwhelmed. Let them set the pace and signal when to move on.
- **They take structured notes.** End substantial explanations with a **notes-ready summary** (a clean blockquote they can paste into their notes) and, where useful, a **"litmus test"** for telling this thing apart from adjacent concepts. They respond extremely positively to both.
- **Visuals help them.** They are a visual learner; diagrams of concepts/architecture/flows land well. Use them where they genuinely aid understanding (don't force them).
- **Validate what's right, then gently correct what's off.** They often paraphrase back their understanding to check it. Affirm the correct parts explicitly and fix misconceptions kindly and precisely — never condescending. Watch for the recurring **"figure/ground" flip** (confusing the *fixed reference* with the *thing being detected/measured*).
- **Build on prior stages.** Tie new material back to what they already locked in (especially Stages 1 and 2).
- **Honesty over hype.** They appreciate caveats, "here's what's actually hard," and clear statements of what is/isn't standard terminology.

---

## 4. The pipeline mental model (status in brackets)

Stage numbering is *ours* — a teaching scaffold, **not** universal terminology; say so if a paper slices it differently. Each stage has **one central hard problem**.

- **Stage 0 — Pre-processing** *(upstream; not covered in depth).*
- **Stage 1 — Registration / alignment** ✅ **DONE.** Aligned multiple epochs onto a common pixel grid using fixed Gaia stars; built the trustworthy reference frame.
- **Stage 2 — Difference imaging** ✅ **DONE.** Subtracted the aligned reference template from each science image (PSF + photometric matching) so the static sky cancels and only new/moved/changed objects remain.
- **Stage 3 — Source detection & measurement** ⬅️ **ACTIVE (this chat).** *Central problem: measuring trustworthy features on faint, marginal, few-pixel blobs + setting the detection threshold (completeness vs. purity).* Convert the difference image (pixels) into a **catalog of candidate detections** (data). Deliberately inclusive — detects everything above threshold and does NOT judge real vs. bogus.
- **Stage 4 — Classification (real/bogus + type).** Real/bogus gate, then sort into the four classes. Central problem: extreme class imbalance + reliable labels (human scanning, known-object cross-match, synthetic "fakes"). Anomaly detection alongside.
- **Stage 5 — Alert distribution & brokering.** Package classified detections into standardized alerts; stream to brokers.
- **Stages 6–8 (mapped only):** linking & orbit determination → follow-up/confirmation → cataloging/archiving (the pipeline is a **cycle**).

---

## 5. Where we are now

Stages 1 and 2 are **complete**: the user has clean, PSF-matched **difference image(s)** on the common pixel grid (static sky ≈ noise; real changes appear as positive/negative residual sources; some artifacts remain, as expected).

**Active task: Stage 3 — detect and measure the sources in the difference image(s), producing a candidate catalog + cutout triplets.**

---

## 6. Stage 3 recap (the concept they already hold)

- **The transformation:** difference image (pixels) → a structured **catalog of candidate detections** (one row per blob, with measured numbers). Everything downstream operates on this list, not on pixels.
- **Sub-steps:** (1) estimate the local background & noise; (2) threshold-detect blobs N-sigma above noise; (3) deblend/segment merged sources; (4) centroid → precise position → RA/Dec via the Stage-1 WCS; (5) photometry (flux/mag + uncertainty); (6) measure shape/morphology features; (7) save science/reference/difference **cutout "triplets"** per detection.
- **Deliberately inclusive:** it detects *everything* above threshold — real movers, transients, AND artifacts — and does **not** decide real vs. bogus. That judgment is Stage 4's job.
- **Why it matters for the project:** the shape features feed Stage 4's classifier; the cutout triplets are the images the CNN trains on and the things a human labels. This stage is where the **computer-vision, data-labeling, and anomaly-detection** pillars physically live.
- **The central difficulty:** squeezing *trustworthy* position/shape numbers out of faint, few-pixel blobs barely above the noise — and choosing the detection threshold (the completeness-vs-purity dial).

---

## 7. Practical tools & how to go about it with ZTF

Introduce one step at a time. Inputs are the Stage-2 difference image(s) plus the aligned science + reference frames (for building cutout triplets).

**Environment.** Continue the Stage-1/2 env: `astropy`, `numpy`, `photutils`, `sep` (SExtractor-in-Python). No new heavy installs.

**Sub-steps → tools:**

1. **Background & noise map.** `photutils.background.Background2D` (or `sep.Background`) to get a background and an **RMS/noise map** — you need local noise to define "significant."
2. **Detect (BOTH signs).** Segmentation: `photutils.segmentation.detect_sources` / `SourceFinder` (which also deblends), or `sep.extract(data, thresh, err=rms)`. For star-like sources, `photutils.detection.DAOStarFinder` is an option. **Run detection on the difference image *and* on its negative** (`-diff`) — negative sources are real faders/vanishers and the negative lobes of dipoles.
3. **Deblend.** `photutils.segmentation.deblend_sources` (or `SourceFinder`, which combines detect+deblend); SEP deblends via its detection params.
4. **Measure.** `photutils.segmentation.SourceCatalog` gives centroids, photometry, and morphology in one object; pass `wcs=` to get RA/Dec directly, and `mask=` to exclude bad pixels. (`aperture_photometry` / the PSF-photometry module are alternatives.)
5. **Shape features.** From `SourceCatalog`: elongation, ellipticity, orientation, FWHM/semi-axes, etc. — these are the **Stage-4 features**. (Elongation is what will separate streaks (meteors/satellites) from point sources (asteroids/transients) later.)
6. **Cutout triplets.** For each detection, use `astropy.nddata.Cutout2D` to save aligned **science / reference / difference** stamps (e.g. 21×21 or 63×63 px) centered on it. **This is the real ML deliverable** — decide stamp size/format now.

**Difference-image gotchas (important):**
- **Correlated noise.** The Stage-1 resampling + Stage-2 subtraction correlate neighboring-pixel noise, so a naive per-pixel RMS *underestimates* the true noise and a nominal "5σ" over-detects. Calibrate the threshold **empirically** (inspect false-detection rates), and if you used a ZOGY/PyZOGY output, prefer its noise-propagated / matched-filter (S_corr) image for thresholding.
- **Mask** saturated stars, bad pixels, and frame edges — they generate false residuals.

**Validate against ZTF (excellent ground truth + a bridge to Stage 4):**
- A **ZTF alert** *is* a Stage-3/Stage-4 output: each alert packet bundles a difference-image detection's measured features **and** its science/reference/difference cutout triplet already. Reading a ZTF alert (`ztfquery.alert.AlertReader`) for the same field lets them see exactly what a professional Stage-3 catalog row + triplet looks like. ZTF DR also provides difference-image PSF catalogs.

**Output:** a candidate catalog (an `astropy` table: position, RA/Dec, flux, S/N, shape features, sign) + saved cutout triplets → hand to **Stage 4**.

---

## 8. Decision to confirm with the user up front

Same pattern as Stages 1–2 (ZTF already does this internally):

- **(A) Build it themselves (recommended for the project's goals):** run their own detection + measurement on their Stage-2 difference image, and validate against ZTF alerts/catalogs. Best for learning the CV mechanics.
- **(B) Lean on ZTF products:** use ZTF's difference-image catalogs / alert packets directly to move faster toward the ML stages.

Also worth asking: **how permissive to set the detection threshold** (start inclusive — completeness over purity at this stage), and **segmentation vs. point-source finder** (segmentation is more general because it handles elongated streaks, not just round sources). Likely (A), inclusive threshold, segmentation-based — but ask, don't assume.

---

## 9. Honest caveats to keep in mind

- **Do NOT judge real vs. bogus here.** Stage 3 only *detects and measures*. Resist filtering out "junk" — that's Stage 4's job, and throwing things away now can lose real faint movers forever.
- **The threshold is the key knob:** too low = a flood of noise spikes; too high = miss faint asteroids/transients. Start permissive; it's the completeness-vs-purity tradeoff.
- **Difference-image noise is correlated**, so nominal N-sigma over-detects — calibrate empirically rather than trusting the raw pixel RMS.
- **Detect both positive and negative** sources.
- **Shape features get unreliable near the threshold** (few noisy pixels) — this is the central difficulty of the stage, not a bug to eliminate.
- **Cutout triplets are the deliverable that matters** for the ML/labeling goals — lock down their size and format early and keep them consistent.
- **Keep variables fixed:** same filter / CCD / quadrant as before; modest cutouts; a handful of epochs first.
- **Numbering is a teaching scaffold,** not official terminology.
- Honor the learning style in §3: notes-ready summaries, litmus tests, validate-then-correct, watch the figure/ground flip, one step at a time.
