""" The one place that knows the ztfdata/ layout. Pure - no I/O beyond mkdir. """

from pathlib import Path

from ztfquery.io import LOCALSOURCE

# our own pipeline's outputs, kept separate from ZTF's own products in ztfdata/
OWN_ROOT = Path(LOCALSOURCE) / "ztf_own_pipeline_data_testing_small_scale"

# one directory per (grid, science epoch, DETECTION REGION) - a sweep can't collide.
# The region is part of the key, not just the manifest: the same grid+epoch detected
# full-frame (~950 rows) vs central-1000 (114) are different catalogs, and without
# the suffix the second run would silently overwrite the first's catalog + cutouts.
# Central runs keep their bare filefracday dir, so existing runs stay valid.
def run_dir(field, ccdid, qid, fid, science_filefracday, full_frame=False):
    grid = f"f{field:06d}_c{ccdid:02d}_q{qid}_f{fid}"
    leaf = f"{science_filefracday}_full" if full_frame else str(science_filefracday)
    return OWN_ROOT / "runs" / grid / leaf

#the artifacts a run produces. Values are paths, not open files.
def run_paths (rdir, create=False):
    paths = {
        "catalog": rdir / "catalog.ecsv",
        "cutouts": rdir / "cutouts",
        "template": rdir / "deep_template.fits",
        "diff": rdir / "diff.fits",
        "manifest": rdir / "manifest.json",
    }
    if create:
        rdir.mkdir(parents=True, exist_ok=True)
        paths["cutouts"].mkdir(exist_ok=True)
    return paths


# The template belongs to the GRID CELL + REFERENCE SET, not to one science epoch.
# Kept in run_paths above for backward compatibility with runs already on disk, but
# new runs write HERE: a deep template costs 5-10 min and ~9 GB peak RSS to stack, and
# the per-run location made every epoch of the same cell rebuild an identical file.
# Keyed on the CONTENT of the reference set (see run_field.ref_key) rather than on the
# inputs that chose it, so a change in selection logic can never silently reuse a stale
# stack - two different ref lists are two different keys, by construction.
def template_path(field, ccdid, qid, fid, ref_key, create=False):
    grid = f"f{field:06d}_c{ccdid:02d}_q{qid}_f{fid}"
    tdir = OWN_ROOT / "templates" / grid
    if create:
        tdir.mkdir(parents=True, exist_ok=True)
    return tdir / f"{ref_key}.fits"
