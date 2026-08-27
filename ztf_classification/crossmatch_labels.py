"""
Build a LABELED evaluation set for our Stage-3 detections by cross-matching
each detection's RA/Dec against authoritative external catalogs. This is the
answer key we score braai (real/bogus) and the stamp classifier (type) against.

Three independent sources of truth, each a "rung":
  A. SkyBoT (IMCCE)  -> known ASTEROIDS at this exact epoch (movers).
  B. SIMBAD          -> known STATIONARY objects (VS / AGN / SN / star / galaxy).
  C. ZTF alert rb    -> ZTF's OWN pipeline verdict (real/bogus + class) = gold standard.

Each rung returns {detection_row_index: label}. build_labeled_set() merges them
into one table. A detection that matches NOTHING is itself informative (likely
bogus, or a real-but-uncatalogued source) -- recorded as label "unmatched".

Import-safe: only defines functions. The `client`/epoch/coords are passed IN.
"""


import numpy as np 
from astropy.coordinates import SkyCoord 
from astropy.time import Time
import astropy.units as u

# how close (arcsec) a catalog object must sit to a detection to count as the same thing
MATCH_TOL_ARCSEC = 5.0


# Rung A: SkyBoT: known asteroids (movers) at this exact epoch
"""
Cross-match our detections against SkyBoT's known solar-system objects.

det_coords   : SkyCoord array of our detections (from cat["ra"], cat["dec"])
field_center : SkyCoord of the field center (SkyBoT searches a cone)
epoch        : astropy Time of the exposure (asteroids MOVE, so time matters)

Returns {det_index: "asteroid"} for each detection within tol of a known
asteroid. Empty dict = no asteroids in this field/epoch (a NORMAL result).
"""

def label_asteroids_skybot(det_coords, field_center, epoch, rad_deg=0.15, tol_arcsec=MATCH_TOL_ARCSEC):
    from astroquery.imcce import Skybot

    try:
        sbt = Skybot.cone_search(field_center, rad=rad_deg*u.deg, epoch=epoch)
    except RuntimeError:
        #SkyBoT raises RuntimeError when the cone is empty -- not an error for us
        return {}
    if sbt is None or len(sbt) ==0:
        return {}


    ast_coords = SkyCoord(ra=sbt["RA"], dec=sbt["DEC"])
    # for each asteroid, which of OUR detections is nearest, and how far
    idx, sep2d, _ = ast_coords.match_to_catalog_sky(det_coords)

    labels = {}
    for i, sep in zip(idx, sep2d):
        if sep < tol_arcsec * u.arcsec:
            labels[int(i)] = "asteroid"
    return labels    

#RUNG B: SIMBAD: knwon stationary objects (type labels our detections)
# Map SIMBAD's object-type vocabulary onto OUR 4 stamp-classifier classes.
# SIMBAD 'otype' is a short code; we bucket the common ones. Anything we don't
# recognize -> "other" (a real object, just not one of our 4 classes).

SIMBAD_OTYPE_MAP = {
    "QSO": "AGN", "AGN": "AGN", "Sy1": "AGN", "Sy2": "AGN", "Bla": "AGN",
    "SN*": "SN", "SN": "SN",
    "V*": "VS", "RR*": "VS", "Ce*": "VS", "Mi*": "VS", "Er*": "VS", "Ro*": "VS",
    "*": "star", "PM*": "star", "HB*": "star",
    "G": "other", "GiG": "other", "GiC": "other", "IG": "other",
}

def _simbad_class(otype):
    """ Bucket a raw SIMBAD otype into one of our 4 stamp classes or 'other'. """
    if otype is None:
        return None
    otype = otype.strip()
    if otype in SIMBAD_OTYPE_MAP:
        return SIMBAD_OTYPE_MAP[otype]

    # a variable-star otype often ends in "*" with a prefix we didn't list
    if otype.endswith("*"):
        return "VS" if otype != "*" else "star"
    return "other"

def label_stationary_simbad(det_coords, tol_arcsec=MATCH_TOL_ARCSEC):
    """
    Cross-match each detection against SIMBAD (per-detection cone search).

    Returns {det_index: class} where class is AGN / SN / VS / star / other,
    for every detection with a SIMBAD object within tol. No match -> not in dict.
    """
    from astroquery.simbad import Simbad

    sim = Simbad()
    sim.add_votable_fields("otype")   # ask SIMBAD to include the object-type code

    labels = {}
    for i, c in enumerate(det_coords):
        try:
            res = sim.query_region(c, radius=tol_arcsec * u.arcsec)
        except Exception:
            res = None
        if res is None or len(res) == 0:
            continue
        # the query returns objects sorted by distance; take the nearest (row 0)
        otype = str(res["otype"][0]) if "otype" in res.colnames else None
        cls = _simbad_class(otype)
        if cls is not None:
            labels[i] = cls
    return labels

# ---------------------------------------------------------------------------
# Rung C -- ZTF alert rb + type via ALeRCE (the GOLD STANDARD: grades braai too)
# ---------------------------------------------------------------------------
# ALeRCE is the only queryable path to ZTF's alert stream. We reuse the exact
# cone-search + probability read from stamp_classifier.py. For each detection we
# pull TWO ZTF-pipeline truths: rb (real/bogus, grades braai) and type (grades
# the stamp classifier). NO-MATCH is a normal result -- ALeRCE only catalogs
# objects that triggered an alert, so faint/archival sources may be absent.
def label_ztf_alerce(det_coords, client, radius_arcsec=MATCH_TOL_ARCSEC):
    """
    Cross-match each detection against ALeRCE (ZTF's broker).

    Returns {det_index: {"oid": str, "ztf_rb": float|None, "ztf_type": str|None}}
    for every detection that resolves to an ALeRCE object. No match -> absent.
    """
    from ztf_classification.stamp_classifier import (
        resolve_oid_for_coords, fetch_stamp_probabilities, classify_astrophysical,
    )

    labels = {}
    for i, c in enumerate(det_coords):
        ra, dec = c.ra.deg, c.dec.deg
        oid = resolve_oid_for_coords(client, ra, dec, radius_arcsec=radius_arcsec)
        if oid is None:
            continue

        # rb (real/bogus) = the max rb over this object's detections -> grades braai
        rb = None
        try:
            dets = client.query_detections(oid, format="pandas")
            if dets is not None and len(dets) and "rb" in dets.columns:
                rb = float(dets["rb"].max())
        except Exception:
            rb = None

        # type = ALeRCE stamp-classifier top astro class -> grades the stamp CNN
        ztf_type = None
        probs = fetch_stamp_probabilities(client, oid)
        if probs is not None:
            top_class, _, _ = classify_astrophysical(probs)
            ztf_type = top_class

        labels[i] = {"oid": oid, "ztf_rb": rb, "ztf_type": ztf_type}
    return labels

# Rung D -- merge all three rungs into ONE labeled table (the answer key)

def build_labeled_set(cat, field_center, epoch, client, tol_arcsec=MATCH_TOL_ARCSEC):
    """
    Run all three cross-match rungs on the Stage-3 catalog and merge into one
    row-per-detection labeled table. Priority for the single 'label' column:
    SkyBoT asteroid  >  ZTF/ALeRCE type  >  SIMBAD type  >  "unmatched".
    Also keeps the raw ZTF rb (grades braai) and the source of each label.
    
    cat          : the Stage-3 catalog Table (needs 'ra','dec' columns)
    field_center : SkyCoord for the SkyBoT cone
    epoch        : astropy Time of the exposure
    client       : an alerce.core.Alerce() instance

    Returns a new astropy Table: the input catalog + columns
    label, label_source, ztf_rb, ztf_oid.
    """

    from astropy.table import Table

    det_coords = SkyCoord(ra=cat["ra"] * u.deg, dec=cat["dec"] * u.deg)

    ast   = label_asteroids_skybot(det_coords, field_center, epoch, tol_arcsec=tol_arcsec)
    sim   = label_stationary_simbad(det_coords, tol_arcsec=tol_arcsec)
    ztf   = label_ztf_alerce(det_coords, client, radius_arcsec=tol_arcsec)

    labels, sources, rbs, oids = [], [], [], []
    for i in range(len(cat)):
        z = ztf.get(i)
        if i in ast:                        # 1. asteroid = highest authority
            labels.append(ast[i]);           sources.append("skybot")
        elif z is not None and z["ztf_type"]:  # 2. ZTF/ALeRCE type
            labels.append(z["ztf_type"]);    sources.append("ztf_alerce")
        elif i in sim:                      # 3. SIMBAD stationary type
            labels.append(sim[i]);           sources.append("simbad")
        else:                               # 4. nothing matched
            labels.append("unmatched");      sources.append("none")
        # ztf rb / oid carried regardless of which source won the label
        rbs.append(z["ztf_rb"] if z else np.nan)
        oids.append(z["oid"] if z else "")

    out = cat.copy()
    out["label"] = labels
    out["label_source"] = sources
    out["ztf_rb"] = rbs
    out["ztf_oid"] = oids
    return out

