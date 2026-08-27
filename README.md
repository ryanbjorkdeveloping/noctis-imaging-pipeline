# noctis-imaging-pipeline

A difference-imaging pipeline for the **Zwicky Transient Facility** (ZTF) public archive:
it downloads survey images of a patch of sky, aligns them, subtracts a deep reference to
find what *changed*, measures every change, and classifies each one — stationary source
(variable star / AGN / supernova) or moving object (asteroid / satellite), real or artifact.

It runs on a laptop. CPU only, no GPU anywhere.

> **This pipeline has made no discoveries.** It has *recovered* known objects — the 2019 BE5
> near-Earth asteroid, found blind and then confirmed against JPL Horizons — and it produces
> *candidates* and *verdicts*. Nothing here is an alert feed. That distinction is enforced in
> the code, not just the docs: novelty is a three-state value (`known` / `novel` / `unchecked`),
> and "I could not ask the catalogue" never collapses into "I asked and found nothing".

---

## What it does, stage by stage

| Stage | Module | What happens |
|---|---|---|
| 1 · Acquire | `ztf_data/epochs.py`, `fetch.py` | Query IRSA for one ZTF grid cell `(field, ccd, quadrant, filter)`; pick a science epoch and reference epochs; download |
| 1 · Align | `ztf_data/align.py` | Reproject every epoch onto one pixel grid using the WCS already solved in each header |
| 2 · Reference | `ztf_data/template.py` | Median-stack ~40 sharp epochs from *other years* into a deep reference |
| 2 · Subtract | — | Science − reference. Static sky cancels; only changes survive |
| 3 · Detect | `ztf_data/detect.py` | Segment everything ≥3σ, both signs. Deliberately inclusive — a missed faint source is gone forever, a false one is cheap |
| 3 · Measure | `ztf_data/measure.py` | Position, flux, S/N, shape → catalog + `(3,63,63)` sci/ref/diff cutouts |
| 4 · Real/bogus | `ztf_classification/braai_realbogus.py` | Pretrained **braai** CNN scores each cutout |
| 4 · Type | `ztf_classification/stamp_classifier.py` | Object class via ALeRCE's stamp classifier |
| 4 · Motion | `ztf_classification/deepstreaks.py` | Pretrained **DeepStreaks** 3-gate cascade on elongated sources: real streak? cosmic ray? short (asteroid) or long (satellite)? |
| 4 · Link | `ztf_data/link.py` | Fuse detections across exposures *and fields* into one track with a measured motion vector |
| 4 · Novelty | `ztf_data/novelty.py` | Cross-match the track against SkyBoT — known object, or not in any catalogue? |

**No model is trained here.** braai, DeepStreaks and the ALeRCE classifiers are used exactly as
published by their authors. What this project builds is the machinery *around* them: the data
reduction that produces their inputs, and the linking and cross-matching that turn isolated
detections into objects.

Above all of that, `ztf_data/harvest.py` is a **resumable, parallel survey harvester** — point it
at a night and a region of sky, interrupt it whenever, re-run it, and it picks up where it left off.
Results land in a SQLite store (`ztf_data/store.py`).

---

## Install

Python **3.11**. Three virtual environments, and the split is not optional — see the note below.

```bash
git clone https://github.com/ryanbjorkdeveloping/noctis-imaging-pipeline.git
cd noctis-imaging-pipeline

# 1. main environment
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 2. two packages that are NOT installable from PyPI
.venv/bin/pip install --no-build-isolation git+https://github.com/toros-astro/ois
.venv/bin/pip install git+https://github.com/dguevel/PyZOGY

# 3. pretrained models (~640MB; DeepStreaks is most of it)
./scripts/bootstrap_models.sh

# 4. motion branch runs in its own environment
python3.11 -m venv .venv_streaks
.venv_streaks/bin/pip install -r requirements-streaks.txt

# 5. tell it where to put data
cp .env.example .env    # then edit ZTFDATA, or just: export ZTFDATA=/path/to/data
```

`ois` needs `--no-build-isolation` (its PyPI tarball will not build against NumPy 2).
PyZOGY prints a benign `Invalid script entry point` error on install — the library still imports.

**Why three environments.** The DeepStreaks models were serialized with Keras 2.2.4 and cannot be
loaded by Keras 3, which braai needs. TF 2.15 is the last Keras-2 release with arm64/Python-3.11
wheels. `ztf_classification/pipeline.py` bridges the two by driving `.venv_streaks/bin/python` as a
subprocess with a file-based handoff. `.venv_lc` is a third, optional environment for the light-curve
classifier, which pins `scikit-learn<=1.2.2` and NumPy 1.x. Merging any of them breaks the others.

### Data access

Images come from **IRSA** at NASA/IPAC. You need a free IRSA account; `ztfquery` prompts once on the
first download and caches the credentials encrypted under `~/.ztfquery` — never in this repo.

`$ZTFDATA` is the only environment variable the pipeline reads. Point it at a disk with room:
a single quadrant-image is ~40MB, and a deep reference stack pulls ~40 of them.

---

## Quickstart

Run one exposure end to end and get a catalog out:

```python
import config                      # MUST be first — sets $ZTFDATA before ztfquery loads
from ztf_data.run_field import run_field

res = run_field(field=468, ccdid=3, qid=2, fid=2,
                science_filefracday=20200518187454,
                streaks_only=True)   # motion path: no reference download
print(res.catalog_path)              # → an .ecsv of every detection
```

That exact call is the project's regression fixture: it must produce **114 detections**.

Hunt for fast movers across a night, with no hand-typed target list:

```python
from ztf_data.sweep import sweep_night
candidates, tracks = sweep_night("2019-01-31", ra_range=(120, 127), dec_range=(13, 17))
```

Or drive the harvester from the shell:

```bash
.venv/bin/python -m ztf_data.harvest init
.venv/bin/python -m ztf_data.harvest enqueue --date 2019-02-08 --cadence 5x10 --branches motion
.venv/bin/python -m ztf_data.harvest run --workers 2 --branches motion
.venv/bin/python -m ztf_data.harvest link
.venv/bin/python -m ztf_data.harvest status
```

**Use `--workers 2` on 16GB of RAM** at full frame — each worker peaks around 1.8GB.

Then turn your own store into browsable JSON:

```bash
.venv/bin/python exporters/build_survey.py --verify
```

---

## Layout

```
config.py                 loads .env → $ZTFDATA. Must stay at the repo root:
                          ztf_data/__init__.py imports it so no submodule can skip it
ztf_data/                 Stages 1–3 + motion + the survey harvester and store
  tests/                  the regression contract
ztf_classification/       Stage 4 inference (braai, DeepStreaks, stamp classifier)
streak_testing/           motion-branch validation + the .venv_streaks subprocess entry points
stamp_testing/            classifier validation and scoring
braai_testing/            real/bogus validation
lc_testing/               light-curve classifier (optional, needs .venv_lc)
exporters/                survey.db → static JSON/PNG
scripts/                  bootstrap_models.sh, seed_batch.sh, run_full_harvest.sh
notebooks/                the stage-by-stage derivation, outputs stripped
docs/                     design briefings for each stage
close_up_object_detection/  see below — a separate, earlier project
```

`streak_testing/`, `ztf_data/` and `ztf_classification/` **cannot be renamed**:
`ztf_classification/pipeline.py` builds literal subprocess paths into `streak_testing/`, and roughly
thirty files resolve their imports relative to the repo root by name.

### The OpenCV track

`close_up_object_detection/`, `connectors/`, `utilities/`, `main.py` and `opencv_testing_*.py` are an
**earlier, separate project**: classical OpenCV shape/brightness detectors that classify a single
close-up photograph of one object (star, planet, asteroid, comet, galaxy). They share no code with the
ZTF pipeline in either direction. They are here because they are part of the same body of work, not
because they are part of the pipeline. Point them at your own images:

```bash
.venv/bin/python main.py path/to/image.jpg
```

---

## Tests

Three run offline with no data at all — these are the ones a fresh clone can use:

```bash
.venv/bin/python -m ztf_data.tests.test_store
.venv/bin/python -m ztf_data.tests.test_harvest_smoke
.venv/bin/python -m ztf_data.tests.test_quadrant_resolve
```

Run them **as modules from the repo root**. Invoking by file path puts `ztf_data/tests/` on
`sys.path` and `import ztf_data` fails.

Five more encode real regressions but need downloaded fixtures: `test_regression_2020` (the
114-detection lock), `test_streak_injection` (a synthetic streak must be detected *and* routed as a
fast mover — the positive control), `test_link_be5` (the cross-field linker), `test_full_frame_stamps`,
`test_irsa_guard`.

---

## Known limits

Stated plainly, because they shape what the results mean:

- **Our own cutouts carry a registration dipole.** braai scores ~0.68–0.96 on ZTF's own native alert
  stamps but only ~0.05–0.14 on stamps we cut ourselves, because bright stars do not cancel to
  sub-pixel precision in our subtraction. Where a detection matches a catalogued ZTF object, the
  pipeline uses ZTF's native stamp instead. For a genuinely novel source, no such stamp exists, and
  the dipole-limited score is what you get. Beating it means sub-pixel registration and PSF matching.
- **The motion branch has a brightness floor.** End-to-end recovery works for bright fast movers
  (2019 BE5 at V=15.1). A fainter one (2018 VJ10, V≈17.6) *is* detected and measured correctly, but
  the pretrained DeepStreaks real/bogus gate declines it. That is the published model's own learned
  purity, not a bug in this code — but it bounds what you will recover, somewhere between V=15 and 17.6.
- **Linking is intra-night only.** Motion is treated as linear on the sky, which holds for one night.
  Linking tracklets across nights into an orbit is a different problem (HelioLinC/THOR territory).
- **Path A is not runnable.** `stamp_testing/pathA_*.py` needs an ALeRCE TF1 checkpoint with no public
  release. The Path B route, which the main pipeline actually uses, works.
- **Scale.** ~35–65s per quadrant-image plus download. One full ZTF night (~25,000 quadrants) is
  roughly 2.5 days of continuous running. The archive is ~10^8 quadrant-images. "All of ZTF" is not
  reachable this way and is not the goal; a bounded, resumable harvest of chosen sky is.

---

## Attribution

This pipeline uses these pretrained models, unmodified, all MIT-licensed. If you use it, cite them:

- **braai** — real/bogus classification. Duev et al. 2019, *MNRAS* 489(3), 3582–3590
  ([arXiv:1907.11259](https://arxiv.org/abs/1907.11259)) · [dmitryduev/braai](https://github.com/dmitryduev/braai)
- **DeepStreaks** — streak detection. Duev et al. 2019, *MNRAS* 486(3), 4158
  ([arXiv:1904.05920](https://arxiv.org/abs/1904.05920)) · [dmitryduev/DeepStreaks](https://github.com/dmitryduev/DeepStreaks)
- **ALeRCE** — stamp and light-curve classifiers. Carrasco-Davis et al. 2021; Sánchez-Sáez et al. 2021.
  [alercebroker](https://github.com/alercebroker)

Data comes from the **Zwicky Transient Facility**, a project led by Caltech Optical Observatories,
funded by NSF grant AST-1440341, served publicly by **IRSA** at IPAC/Caltech. Cross-matches use
**SkyBoT** (IMCCE), **SIMBAD** (CDS), **JPL Horizons** and **Pan-STARRS** (MAST).

Difference imaging uses [`ois`](https://github.com/toros-astro/ois) (Alard–Lupton) and
[`PyZOGY`](https://github.com/dguevel/PyZOGY) (ZOGY).

## License

MIT — see [LICENSE](LICENSE). Fork it, change it, break it.
