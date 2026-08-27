# Project Handoff Briefing — ML Sky-Image Pipeline, Stage 2 (Difference Imaging) with ZTF

*Paste this whole document into the new chat to bring the assistant up to speed.*

---

## 1. Your role (read this first)

You are taking over as the user's research assistant for an ongoing, multi-session project. Earlier sessions covered the **conceptual** design of the whole pipeline (stages 1–5, plus a map of later stages), and the user has now **finished executing Stage 1** (registration/alignment) for real with ZTF data.

**Your job now is hands-on and practical:** guide the user through *actually executing Stage 2 — difference imaging (image subtraction)* — on the aligned image stack they produced in Stage 1, using **ZTF** data. They do not have Rubin data access; ZTF (free, public, well-tooled) is the source.

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
- **Validate what's right, then gently correct what's off.** They often paraphrase back their understanding to check it. Affirm the correct parts explicitly and fix misconceptions kindly and precisely — never condescending. A recurring pattern to watch for is a **"figure/ground" flip** (confusing the *fixed reference* with the *thing being tracked/measured*).
- **Build on prior stages.** Tie new material back to what they already locked in (especially Stage 1).
- **Honesty over hype.** They appreciate caveats, "here's what's actually hard," and clear statements of what is/isn't standard terminology.

---

## 4. The pipeline mental model (status in brackets)

The stage numbering is *ours* — a teaching scaffold, **not** universal terminology; say so if a paper slices it differently. Each stage has **one central hard problem**.

- **Stage 0 — Pre-processing** *(upstream; not yet covered in depth).* Raw readout → bias/dark/flat → cosmic-ray cleanup → reference-template building.
- **Stage 1 — Registration / alignment** ✅ **DONE.** Lined up multiple images of the same sky patch onto a common pixel grid using fixed Gaia stars as anchors. *Built the trustworthy reference frame; does not detect/track movers.*
- **Stage 2 — Difference imaging** ⬅️ **ACTIVE (this chat).** *Central problem: matching the two images' blur (PSF) and brightness scale.* Subtract the aligned reference template from each aligned science image so the static sky cancels to ~zero and only new/moved/changed objects remain. Separates **static vs. changed**, NOT known vs. unknown.
- **Stage 3 — Source detection & measurement.** Turn the difference image into a catalog of candidate detections (position, brightness, shape) + cutout triplets. Deliberately inclusive; doesn't judge real vs. bogus.
- **Stage 4 — Classification (real/bogus + type).** Real/bogus gate, then sort into the four classes. Central problem: extreme class imbalance + getting reliable labels (human scanning, known-object cross-match, synthetic "fakes"). Anomaly detection alongside.
- **Stage 5 — Alert distribution & brokering.** Package classified detections into standardized alerts and stream to brokers (ALeRCE, ANTARES, Lasair, Fink, etc.). Central problem: scale + latency.
- **Stages 6–8 (mapped only):** 6 = linking & orbit determination (asteroid finale; HelioLinC3D, THOR, find_orb); 7 = follow-up & confirmation; 8 = cataloging & archiving (the pipeline is a **cycle** — today's catalog becomes tomorrow's reference data).

---

## 5. Where we are now

Stage 1 is **complete and verified**: the user has a clean, aligned, common-grid stack of ZTF single-exposure science images plus the aligned reference template (same filter, same CCD/quadrant), confirmed by fixed stars landing on identical pixels (a quick test-subtraction showed stars roughly cancelling).

**Active task: Stage 2 — perform difference imaging on that aligned stack.**

---

## 6. Stage 2 recap (the concept they already hold)

- **Operation:** aligned science image **−** aligned reference template, pixel by pixel → static sky cancels → only new/moved/changed objects survive as residual point sources.
- **The hard part is NOT the subtraction** — it's first making unchanged stars actually cancel, which requires:
  - **PSF matching:** the blur differs between frames (seeing varies). Always **convolve the *sharper* image to match the *blurrier* one** — you degrade to match, never sharpen.
  - **Photometric scaling:** rescale flux so a steady star reads identically in both before subtracting.
- **Output:** a difference image where *positive* = appeared/brightened, *negative* = faded/vanished, and *dipoles* = either real fast-movers or (more often) imperfect alignment/PSF matching.
- **Honest framing:** a difference image is never perfectly empty — artifacts always remain, which is exactly why Stages 3 (detect) and 4 (real/bogus) exist downstream.
- **Callback to Stage 1:** their resampling step slightly **correlated** neighboring-pixel noise. ZOGY assumes *uncorrelated* noise, so flag this when choosing/tuning a method.

---

## 7. Practical tools & how to go about it with ZTF

A learner-friendly arc (introduce one step at a time):

1. **Environment.** Continue the Stage-1 env: `astropy`, `numpy`, `scipy`, `photutils`, `sep`, plus one subtraction package (below). Inputs are the aligned science frames + aligned reference from Stage 1 (already on a common grid — a big head start).

2. **Naive subtraction first (pedagogically valuable).** Just compute `science − reference` on the common grid and look at it. It will look *bad* — rings/dipoles around every star from PSF + flux mismatch. This makes the central problem of Stage 2 visible and motivates everything next. Don't skip this.

3. **PSF-matched subtraction — pick a tool:**
   - **`ois`** (`pip install ois`) — pure-Python **Alard–Lupton "Optimal Image Subtraction"** (kernel matching). Minimal setup, ideal for the *first real* PSF-matched subtraction. (Same author as `astroalign`, which they may have used in Stage 1.)
   - **`PyZOGY`** (`pip install git+https://github.com/dguevel/PyZOGY.git`) — Python **ZOGY** ("proper" subtraction). Needs the science + reference images **and their PSF models** as FITS. Closer to modern survey practice; there's a fully-worked example at `github.com/griffin-h/image_subtraction`, and it's used in current LSST-prep work (e.g. SLIDE, Dong et al. 2025).
   - **`properimage`** — another Python proper-subtraction option with a readthedocs tutorial.
   - **HOTPANTS** (Becker) — the classic C kernel implementation; powerful but needs compiling (CFITSIO). The reference tool; optional for a learner.
   - **SFFT** — GPU-accelerated, fast, modern; more advanced, optional.
   - **Recommended path:** start with **`ois`** (first clean PSF-matched result, almost no install pain) → then **`PyZOGY`** to experience the ZOGY proper-difference approach and PSF handling.

4. **PSF estimation (a real sub-step for ZOGY/PyZOGY).** PyZOGY needs a PSF model for *both* science and reference. Build one with `photutils` `EPSFBuilder`, or PSFEx. Budget time here — **PSF quality drives subtraction quality.**

5. **Inspect & validate.** After matching, the static sky should look like ~noise; real changes appear as clean positive/negative point sources; dipoles flag imperfect matching. Mask bad pixels, saturated stars, and frame edges (they fake residuals).

6. **Cross-check against ZTF's own difference image (great ground truth).** ZTF's pipeline already produces a PSF-matched difference product (e.g. `scimrefdiffimg.fits`, retrievable via `ztfquery`/IRSA — confirm the exact product string in the docs). Comparing the user's homemade difference image to ZTF's official one is an excellent reality check on their subtraction.

7. **Output.** A clean difference image (or set) on the common grid → hand to **Stage 3** (source detection & measurement).

---

## 8. Decision to confirm with the user up front

Just like Stage 1 (where ZTF's WCS was pre-solved), ZTF *already produces difference images*. Clarify intent before starting:

- **(A) Build it themselves (recommended for the project's goals):** run their own PSF-matched subtraction on the aligned sci + reference, and validate against ZTF's difference product. Best for actually learning the computer-vision mechanics.
- **(B) Lean on ZTF products:** pull ZTF's ready-made difference images to move quickly toward Stage 3 / the ML stages.

Also ask **which algorithm family to try first** — kernel-matching (`ois`/HOTPANTS) vs. ZOGY-style (`PyZOGY`/`properimage`). Likely (A) with the `ois` → `PyZOGY` arc, but ask, don't assume.

---

## 9. Honest caveats to keep in mind

- **A perfect zero is impossible.** Expect leftover artifacts — that's normal and is exactly what motivates Stages 3–4. Don't let them chase a flawless difference image.
- **Convolution direction:** always degrade the *sharper* image to match the *blurrier* one; never sharpen.
- **Masking matters:** bad pixels, saturated stars, and edges produce false residuals — mask them.
- **PSF quality is everything** for ZOGY/PyZOGY: a poor PSF model = dipoles everywhere.
- **Noise correlation** introduced by Stage-1 resampling slightly violates ZOGY's independent-noise assumption; relevant when interpreting ZOGY outputs.
- **Keep variables fixed:** same filter / same CCD / same quadrant as Stage 1; modest cutouts; a handful of epochs before scaling.
- **Install friction:** HOTPANTS (compile) and SFFT (CUDA) can be fiddly; the Python-native `ois`/`PyZOGY` avoid most of it.
- **Numbering is a teaching scaffold,** not official terminology.
- Honor the learning style in §3: notes-ready summaries, litmus tests, validate-then-correct, watch the figure/ground flip, one step at a time.

---

## 10. Where Stage 2 landed (COMPLETE — executed & validated)

Stage 2 was executed end-to-end with the recommended path (build it ourselves; `ois` → `PyZOGY`)
across two notebooks: `notebook/stage2_subtraction.ipynb` (the 4 rungs) and
`notebook/stage2_validation.ipynb` (validate vs ZTF). Decisions taken: persistence = save Stage-1
aligned frames to `ztfdata/aligned/` as FITS; template = the sharpest single aligned frame (idx 2,
SEEING 1.752); work on a central 1000×1000 cutout.

**Results (residual std on the central patch):** naïve **208.7** → `ois` **136.5 (−34.6%)** →
ZOGY **344** (on ZOGY's *whitened* scale, not comparable to raw counts; its gain loop didn't
converge and the Stage-1 noise-correlation caveat showed up as mottling/ringing — exactly as §6/§9
predicted) → ZTF's **official** difference **10.9** (the gold standard).

**Verdict:** our approach is **validated** — our homemade subtraction cancels the static sky and
surfaces the same real residual (the saturated star) in the same place as ZTF's pipeline. The ~12×
residual-noise gap to ZTF is **template depth, not algorithm**: ZTF uses a deep co-add reference
(near-noiseless), we used one noisy science frame. The big learning wins: difference imaging is a
**toolbox** (pick the method that fits your data + goal — the winner is data-dependent, not the
fanciest); **judge by the image, not a cross-scale metric**; and the **reference template's depth**
drives residual noise as much as the algorithm. See `CLAUDE.md` → "Stage 2 — ZTF Difference Imaging"
for the full per-rung detail and gotchas. **Next: Stage 3 (source detection on the difference).**
