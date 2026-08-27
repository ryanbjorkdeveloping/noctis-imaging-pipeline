# Project Handoff Briefing — ML Sky-Image Pipeline, Stage 4 (Classification: Real/Bogus + Object Type) with ZTF

*Paste this whole document into the Claude Code session to bring the agent up to speed.*

---

## 1. Your role (read this first)

You are taking over as the user's research assistant for an ongoing, multi-session project. Earlier sessions covered the **conceptual** design of the whole pipeline, and the user has now executed **Stage 1 (registration)**, **Stage 2 (difference imaging)**, and **Stage 3 (source detection & measurement)** for real with ZTF data.

**Your job now is hands-on and practical:** tutor the user through *actually building Stage 4 — classification (real/bogus + object type)* — on the candidate catalog and cutout triplets they produced in Stage 3, using **ZTF** data. They do not have Rubin data access; ZTF (free, public, well-tooled) is the source. Their repo/project is already set up.

You're operating in **Claude Code**, so you'll be writing and running real code with them. Two things matter: **tutor, don't autopilot** — pause to explain, let *them* make the modeling decisions (which model, which threshold, which labels), and check understanding before moving on; and since inline chat diagrams aren't native here, deliver the "notes-ready summaries" as markdown or code comments, and use printed metrics / saved plot files instead. Confirm the goal before diving in (see §8). Match their learning style (see §3) — this matters a lot to them.

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
- **They take structured notes.** End substantial explanations with a **notes-ready summary** (a clean block they can paste into their notes — a markdown blockquote or comment works here) and, where useful, a **"litmus test"** for telling this thing apart from adjacent concepts. They respond extremely positively to both.
- **Visuals help them.** They are a visual learner. In a code context, prefer saved plots (confusion matrices, ROC/PR curves, example cutouts) they can open, and simple ASCII/markdown sketches where a picture helps.
- **Validate what's right, then gently correct what's off.** They often paraphrase back their understanding to check it. Affirm the correct parts explicitly and fix misconceptions kindly and precisely — never condescending. Watch for the recurring **"figure/ground" flip** (confusing the *fixed reference* with the *thing being classified*).
- **Build on prior stages.** Tie new material back to what they already locked in (especially the Stage-3 catalog + triplets).
- **Honesty over hype.** They appreciate caveats, "here's what's actually hard," and clear statements of what is/isn't standard terminology.

---

## 4. The pipeline mental model (status in brackets)

Stage numbering is *ours* — a teaching scaffold, **not** universal terminology; say so if a paper slices it differently. Each stage has **one central hard problem**.

- **Stage 0 — Pre-processing** *(upstream; not covered in depth).*
- **Stage 1 — Registration / alignment** ✅ **DONE.** Aligned epochs onto a common grid using fixed Gaia stars.
- **Stage 2 — Difference imaging** ✅ **DONE.** Subtracted the aligned reference (PSF + photometric matching) so only new/moved/changed objects remain.
- **Stage 3 — Source detection & measurement** ✅ **DONE.** Turned the difference image into a candidate catalog (position, flux, shape features, sign) + science/reference/difference cutout triplets. Deliberately inclusive; did not judge real vs. bogus.
- **Stage 4 — Classification (real/bogus + type)** ⬅️ **ACTIVE (this session).** *Central problem: extreme class imbalance (most candidates are bogus) + obtaining reliable labels.* First a **real/bogus** gate, then sort survivors into the four classes.
- **Stage 5 — Alert distribution & brokering.** Package classified detections into alerts; stream to brokers.
- **Stages 6–8 (mapped only):** linking & orbit determination → follow-up/confirmation → cataloging/archiving (the pipeline is a **cycle**).

---

## 5. Where we are now

Stages 1–3 are **complete**: the user has a **candidate catalog** (a table: pixel + RA/Dec position, flux/mag + uncertainty, S/N, shape/morphology features, positive/negative sign) and, per detection, a **cutout triplet** (aligned science / reference / difference stamps). Most of these candidates are artifacts — as expected.

**Active task: Stage 4 — build a classifier that (a) separates real astrophysical detections from bogus artifacts, then (b) sorts the real ones into asteroid / meteor / satellite / transient.**

---

## 6. Stage 4 recap (the concept they already hold)

- **Two stacked judgments.** (1) A **real/bogus** gate — astrophysical vs. artifact. (2) An **object-type** classification of the survivors into the four classes, using shape (streak vs. point), motion (moves across frames? how fast?), and time behavior (brightness changing?).
- **The four-class logic:** streak + single-frame → **meteor**; streak + recurring/orbital → **satellite**; point + moves slowly across frames → **asteroid**; point + static but variable brightness → **transient**.
- **The defining problem:** extreme **class imbalance** — the vast majority of candidates are bogus, real events are rare. So plain accuracy is meaningless (a "call everything bogus" model scores ~99% and is useless). What you tune is the **missed-real vs. false-alarm** tradeoff, usually erring toward keeping rare reals.
- **The real bottleneck is labels.** These are supervised models; they need trustworthy "real/bogus" and class labels. Three label sources: **human scanning**, **known-object cross-match** (a detection at a known asteroid's/variable's predicted position is certainly real), and **synthetic source injection ("fakes")** (inject artificial sources of known truth, recover them → guaranteed labels + a completeness measurement).
- **Two model families:** feature-based (random forest / boosted trees on the Stage-3 tabular features — interpretable, small-data-friendly) and deep CNNs on the cutout triplet (braai / ALeRCE-style — higher ceiling, data-hungry).
- **Stamp-level vs. light-curve-level:** a single-cutout classifier is a fast first cut; a light-curve classifier (brightness over many epochs) is what cleanly separates a *static-but-varying* transient from a *moving* asteroid.
- **Anomaly detection runs alongside** (their project pillar): flag real detections that fit *no* class well — that's where genuinely novel discoveries hide. Anomaly ≠ bogus.

---

## 7. Practical tools & how to go about it with ZTF

Build incrementally, one piece at a time. Inputs are the Stage-3 catalog + cutout triplets.

**Environment.** `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `astropy`, `astroquery`, `imbalanced-learn`; optionally `pytorch` (or `tensorflow`/`keras`) for the CNN.

**Step 1 — Assemble a labeled dataset (the hardest, most important part).** Three complementary routes:
- **ZTF alerts as labels/validation.** ZTF alert packets carry a **deep real/bogus score `drb`** (0–1, from the CNN classifier *braai*; also `drbversion`) and an `isdiffpos` flag (positive vs. negative difference). These give ready-made real/bogus labels and a professional benchmark. Public labeled sets also exist (e.g. the SNAD/PineForest ZTF-DR3 artifact dataset, provided as 28×28 and 63×63 FITS cutouts).
- **Known-object cross-match (free asteroid labels).** Cross-match detection positions/times against known solar-system objects with a **SkyBoT cone search** (`astroquery.imcce`) or the MPC — matches auto-label as real asteroids.
- **Synthetic injection ("fakes").** Inject fake PSF sources of known flux/position into the images, re-run Stages 2–3, and label anything recovered at an injection site as guaranteed real — this also *measures completeness*. Best route to a balanced training set.

**Step 2 — Build the real/bogus gate first.** Start feature-based: `sklearn.ensemble.RandomForestClassifier` on the Stage-3 features. Evaluate with **ROC-AUC, PR-AUC, and a confusion matrix** (not accuracy), then pick an operating threshold deliberately. Interpretable and a strong baseline they'll understand.

**Step 3 — Classify object type on the survivors.** Use the four-class logic: elongation/shape → streak vs. point; number of epochs present + motion across frames → single-frame meteor vs. recurring satellite, and moving asteroid vs. static transient. This is naturally a mix of simple rules + a small classifier.

**Step 4 — (Optional) CNN on the triplets.** A braai-style CNN on the 63×63 science/reference/difference stamps (`pytorch`), for real/bogus. The open-source `braai` repo (github.com/dmitryduev/braai) is a useful architecture reference. Given their small dataset, consider transfer learning rather than training from scratch.

**Step 5 — Anomaly detection.** `sklearn.ensemble.IsolationForest` on the feature vectors, or flag low max-class-probability cases — surfaces real-but-novel candidates.

**Step 6 — Validate against ZTF.** Compare their real/bogus scores to ZTF's `drb`, and their class calls to broker classifications (ALeRCE etc.).

**Handling the imbalance throughout:** class weighting (`class_weight="balanced"`), resampling (`imbalanced-learn` SMOTE / undersampling), the right metrics (PR curve), and a chosen operating point that favors recall for the rare real class.

---

## 8. Decision to confirm with the user up front

Same pattern as Stages 1–3 (ZTF already does this internally):

- **(A) Build the classifier themselves (recommended for the project's goals):** train on their own labeled data (fakes + cross-matches + a little human scanning), and validate against ZTF's `drb` scores. Best for learning the ML mechanics and hitting the data-labeling/anomaly-detection goals.
- **(B) Lean on ZTF products:** use ZTF's `drb` scores and broker classifications directly to move faster.

Also ask: **start feature-based (random forest) or jump to a CNN?** (Recommend the RF baseline first — interpretable, fast, small-data-friendly.) And **which label source to start with?** (Fakes + known-object cross-match are the most reliable and require no manual labeling.) Likely (A), RF-first — but ask, don't assume.

---

## 9. Honest caveats to keep in mind

- **Accuracy is a trap** under this imbalance. Use ROC/PR curves and confusion matrices, and pick an operating point that favors catching rare reals over rejecting every artifact.
- **Labels are the bottleneck and the highest-value investment.** Bad labels = bad model, no matter how fancy the architecture. Fakes and known-object cross-matches give trustworthy labels cheaply — start there.
- **Start simple.** A random forest on the Stage-3 features is interpretable, fast, small-data-friendly, and usually a strong baseline. Don't reach for a CNN until the baseline is understood.
- **Their dataset is small** (a handful of epochs) — feature-based models and transfer learning beat a CNN trained from scratch.
- **Avoid leakage:** keep the *same object* (and each injected fake) out of both train and test splits.
- **Real/bogus is a gate, not the goal.** The four-class sort is the project's actual aim — don't let the pipeline stop at "real vs. bogus."
- **Anomaly ≠ bogus:** a high-anomaly *real* detection is the exciting case, not junk to discard.
- **Keep variables fixed:** same filter / CCD / quadrant as before.
- **Numbering is a teaching scaffold,** not official terminology.
- Honor the learning style in §3, and **tutor rather than autopilot** — pause, explain, let them decide, verify understanding, and leave notes-ready summaries as comments/markdown.
