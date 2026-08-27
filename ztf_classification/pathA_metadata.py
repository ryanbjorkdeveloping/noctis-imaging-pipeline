"""
build the 23-field stamp-classifier metadata for OUR OWN Stage-3
detections (which have no ZTF alert, so every field is computed locally).

Tier 1 = the trivial fields (coords + sign). Later tiers add photometry
(MAGZP), a psfcat cross-match, and PS1 sgscore.

Tier 2 = turn raw flux into calibrated magnitude fields.

Tier 3 = the local cross-match (chinr, sharpnr, distpsnr1/2/3) and final fields (sgscore + classtar + history)
"""

import csv
import io
import math
import ephem

import numpy as np
from astropy.io import fits

from astropy.coordinates import SkyCoord
import astropy.units as u

import requests

import os
import pickle

PS1_URL = "https://catalogs.mast.stsci.edu/api/v0.1/panstarrs/dr2/mean.csv"

# The 23 fields in the model's exact order
FEATURES = ["sgscore1","distpsnr1","sgscore2","distpsnr2","sgscore3","distpsnr3",
            "isdiffpos","fwhm","magpsf","sigmapsf","ra","dec","diffmaglim","classtar",
            "ndethist","ncovhist","ecl_lat","ecl_long","gal_lat","gal_long",
            "non_detections","chinr","sharpnr"]

# the model's per-field CLIP bounds (from stamp_clf.py FEATURES_CLIPPING_DICT).
# None = no bound on that side. This tames the -999 "no data" sentinels pre-normalize.
CLIP = {"sgscore1": [-1, None], "distpsnr1": [-1, None], "sgscore2": [-1, None],
        "distpsnr2": [-1, None], "sgscore3": [-1, None], "distpsnr3": [-1, None],
        "fwhm": [None, 10], "ndethist": [None, 20], "ncovhist": [None, 3000],
        "chinr": [-1, 15], "sharpnr": [-1, 1.5], "non_detections": [None, 2000]}


# Tier 1 Functionality

#set up coordinate fields with metadata ra and ec
# the 4 coordinate fields: ecliptic + galactic lat/long from ra/dec.
def coordinate_fields(ra, dec):
    eq = ephem.Equatorial(math.radians(ra), math.radians(dec))
    ecl = ephem.Ecliptic(eq)
    gal = ephem.Galactic(eq)
    return {
        "ecl_lat": float(ecl.lat),
        "ecl_long": math.degrees(ecl.lon),
        "gal_lat": float(gal.lat),
        "gal_long": math.degrees(gal.lon),
    }

# returns dictionary of fields we can fill from Tier 1 alone.
def build_metadata_tier1(ra, dec, sign): 
    meta = {
        "ra": float(ra),
        "dec": float(dec),
        "isdiffpos": 1.0 if sign > 0 else -1.0
    }
    meta.update(coordinate_fields(ra, dec))
    return meta    

# Tier 2 Functionality

# Read the per-exposure calibration values for the science FITS header.
def read_fits_header_fields (sci_fits_path):
    hdr = fits.open(sci_fits_path)[0].header
    return {
        "magzp": float(hdr["MAGZP"]),
        "fwhm": float(hdr["SEEING"]),
        "diffmaglim": float(hdr["MAGLIM"]),  
    }   

#turn raw flux into calibrated fields. what tier 2 does
def photometry_fields(segment_flux, snr, hdr_fields):
    magzp = hdr_fields["magzp"]
    flux = float(segment_flux)
    magpsf = magzp - 2.5 * np.log10(flux)
    sigmapsf = 1.08 / snr
    return {
        "magpsf": float(magpsf),
        "sigmapsf": float(sigmapsf),
        "diffmaglim": hdr_fields["diffmaglim"],
        "fwhm": hdr_fields["fwhm"],
    }

# Tier 3 configuration

# load ZTF's PSF star catalog as (SkyCoord array, chi array, sharp array). 
def load_psfcat(psfcat_path):
    tab = fits.open(psfcat_path)[1].data
    coords = SkyCoord(ra=tab["ra"]*u.deg, dec=tab["dec"]*u.deg)
    return coords, tab["chi"], tab["sharp"]
    
# distances to the 3 nearest psfcat stars + the nearest star's chi/sharp.set
def crossmatch_fields(ra, dec, psfcat):
    coords, chi, sharp = psfcat
    det = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)

    # angular separation from this detection to every catalog star, in arcsec
    seps = det.separation(coords).arcsec
    order = np.argsort(seps)
    nearest3 = order[:3]

    return {
        "distpsnr1": float(seps[nearest3[0]]),
        "distpsnr2": float(seps[nearest3[1]]),
        "distpsnr3": float(seps[nearest3[2]]),
        # chinr/sharpnr = the shape params of the SINGLE nearest star
        "chinr":  float(chi[nearest3[0]]),
        "sharpnr": float(sharp[nearest3[0]]),
    }

# Approximate ALeRCE's sgscore via Pan-STARRS iPSFMag - iKronMag.
#    Point source (star): PSF~Kron -> diff~0 -> score~1.  Galaxy: Kron brighter -> score~0.
#    Returns a [0,1] star score, or 0.5 (neutral) if PS1 has no usable source here."""

def ps1_star_score(ra, dec, radius_deg=0.0006):
    try:
        params = {"ra": ra, "dec": dec, "radius": radius_deg, "nDetections.gte": 1}
        r = requests.get(PS1_URL, params=params, timeout=20)
        rows = list(csv.DictReader(io.StringIO(r.text)))   # parse by COLUMN NAME, not position
        if not rows:
            return 0.5
        psf = float(rows[0]["iMeanPSFMag"])
        kron = float(rows[0]["iMeanKronMag"])
        if psf < -900 or kron < -900:                       # -999 = PS1 "no measurement"
            return 0.5
        diff = psf - kron                                   # ~0 star, >0.05 galaxy
        return float(max(0.0, min(1.0, 1.0 - diff / 0.1)))  # map to [0,1] star score
    except Exception:
        return 0.5

#sgscore for the source and neighbors, classtar approx, history constants
def crossmatch_extr_fields(ra, dec):
    sg = ps1_star_score(ra, dec)
    return{
        "sgscore1": sg,
        "sgscore2": sg,
        "sgscore3": sg,
        "classtar": sg,
        "ndethist": 1.0,
        "ncovhist": 1.0,
        "non_detections": 0.0,
    }

# actual scoring process for the metadata

#merge all tiers into one field --> value dictionary (the full 23 fields,raw)
def build_metadata(ra, dec, sign, segment_flux, snr, hdr_fields, psfcat):
    meta = {}
    meta.update(build_metadata_tier1(ra, dec, sign))
    meta.update(photometry_fields(segment_flux, snr, hdr_fields))
    meta.update(crossmatch_fields(ra, dec, psfcat))
    meta.update(crossmatch_extr_fields(ra, dec))
    return meta


# order the features array, clip, z-score.
def normalize_metadata(meta, norm_stats_path):
    raw = np.array([meta[f] for f in FEATURES], dtype=np.float32)

    #1. clip per the model's dictionary (tame -999 sentinels)
    for i, f in enumerate(FEATURES):
        if f in CLIP:
            lo, hi = CLIP[f]
            raw[i] = np.clip(raw[i], lo if lo is not None else -np.inf, hi if hi is not None else np.inf)

    #2. z-score per field: (x - mean) / std
    norm = pickle.load(open(norm_stats_path, "rb"))
    out = raw.copy()
    for i, f in enumerate(FEATURES):
        if f in norm:
            m, s = norm[f]["mean"], norm[f]["std"]
            out[i] = (raw[i] - m) / s
    return out.reshape(1, 23).astype(np.float32)