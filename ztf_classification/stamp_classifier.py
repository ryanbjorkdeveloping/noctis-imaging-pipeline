import numpy as np

# The photometry metadata fields we pull from query_detections, in a FIXED order.
# Order is load-bearing: the model expects "slot 0 = magpsf", etc., every time.
META_FIELDS = ["magpsf", "sigmapsf", "magap", "fid", "distnr", "isdiffpos", "ra", "dec"]

# The four real astrophysical classes. We deliberately DROP 'bogus' — braai
# already answered real/bogus upstream, so asking it again here is redundant.
ASTRO_CLASSES = ["SN", "AGN", "VS", "asteroid"]


def fetch_stamp_probabilities(client, oid):
    """Return the stamp classifier's 5-class softmax for one object as a dict.
    Returns None if the object isn't found (aged out of ALeRCE, bad id, etc.)."""
    try:
        # query_probabilities returns EVERY classifier's rows stacked together.
        df = client.query_probabilities(oid, format="pandas")
    except Exception as e:
        print(f" {oid}: query failed ({e})")
        return None

    # Keep only the stamp classifier's rows (drop lc_classifier, etc.).
    stamp = df[df["classifier_name"] == "stamp_classifier"]
    if len(stamp) == 0:
        print(f"{oid}: no stamp classifier found")
        return None

    # Collapse the two columns into a {class_name: probability} dict.
    return dict(zip(stamp["class_name"], stamp["probability"]))


def find_objects_of_class(client, class_name, n=2, min_prob=0.8):
    """Ask ALeRCE for `n` object IDs it confidently calls `class_name`
    (probability >= min_prob). The inverse of fetch_stamp_probabilities:
    class -> list of IDs, instead of ID -> class."""
    df = client.query_objects(
        classifier="stamp_classifier",
        class_name=class_name,
        probability=min_prob,
        page_size=n,
        format="pandas",
    )
    return df["oid"].tolist()


def center_crop(image, size=21):
    """Return the central size×size square of a larger image.
    ALeRCE stamps are 63×63 but the stamp classifier wants 21×21; the object
    is centered, so the middle is what we keep. We CROP (not resize) to preserve
    the original sharp pixels — resizing would blur the point source."""
    h, w = image.shape
    top = (h - size) // 2               # (63-21)//2 = 21 → start 21 px from top
    left = (w - size) // 2
    return image[top:top + size, left:left + size]


def fetch_cube(client, oid, size=21):
    """Fetch an object's sci/ref/diff stamps and return a clean (21,21,3) cube."""
    hdul = client.get_stamps(oid, format="HDUList")

    science = hdul[0].data              # HDU 0 = science image
    reference = hdul[1].data            # HDU 1 = reference / template
    difference = hdul[2].data           # HDU 2 = difference image

    channels = []
    for img in (science, reference, difference):
        # Cast to float32 (model dtype) AND replace NaN pixels with 0.
        # Diff images have NaN at edges/masked regions; NaN is contagious in
        # math (NaN + x = NaN), so one bad pixel would poison the whole output.
        img = np.nan_to_num(img.astype(np.float32))
        # Cut the 63×63 stamp down to the model's 21×21 window.
        img = center_crop(img, size)
        channels.append(img)

    # Stack the three 21×21 images along a NEW last axis → (21, 21, 3),
    # channel-last, order science/reference/difference (same as braai).
    return np.stack(channels, axis=-1)


def assemble_metadata_vector(client, oid):
    """Build a fixed-order float32 metadata vector from the object's first
    detection. Returns None if there are no detections; missing fields -> 0.0."""
    det = client.query_detections(oid, format="pandas")
    if len(det) == 0:
        return None

    first = det.iloc[0]                 # earliest detection = the "early call"

    values = []
    for field in META_FIELDS:           # iterate in the FIXED order
        if field in det.columns:
            values.append(float(first[field]))
        else:
            values.append(0.0)          # sentinel for a missing field
    return np.array(values, dtype=np.float32)


def classify_astrophysical(probs):
    """Turn the 5-class softmax into a final call: drop 'bogus', renormalize
    the 4 astrophysical classes back to sum=1, and argmax.
    Returns (top_class, top_prob, renormed_dict)."""
    # Keep only the 4 astrophysical classes (this is where 'bogus' is dropped).
    astro = {cls: probs[cls] for cls in ASTRO_CLASSES}

    # Renormalize: after removing 'bogus' the 4 sum to <1, so divide by their
    # total to restore a proper probability distribution (needed for thresholds).
    total = sum(astro.values())
    astro = {cls: p / total for cls, p in astro.items()}

    # Argmax: the most likely astrophysical class.
    top_class = max(astro, key=astro.get)
    return top_class, astro[top_class], astro

def resolve_oid_for_coords(client, ra, dec, radius_arcsec=2.0):
    # client.query_objects(ra=ra, dec=dec, radius=radius_arcsec, format="pandas")
    # → if empty, return None ; else return the first row's "oid"

    df = client.query_objects(ra=ra, dec=dec, radius=radius_arcsec, format="pandas")
    
    if len(df) == 0:
        return None
    
    return df.iloc[0]["oid"]

def classify_detection(client, ra, dec, radius_arcsec=2.0):
    # resolve_oid_for_coords → if None, return {"matched": False, "oid": None}
    # else fetch_stamp_probabilities → if None, {"matched": False, "oid": oid}
    # else classify_astrophysical → return {"matched": True, "oid", "top_class", "top_prob", "astro"}

    oid = resolve_oid_for_coords(client, ra, dec, radius_arcsec)
    
    if oid is None:
        return {"matched": False, "oid": None}
    
    probs = fetch_stamp_probabilities(client, oid)
    
    if probs is None:
        return {"matched": False, "oid": oid}
    
    top_class, top_prob, astro = classify_astrophysical(probs)
    
    return {"matched": True, "oid": oid, "top_class": top_class, "top_prob": top_prob, "astro": astro}

def apply_threshold(top_class, top_prob, threshold=0.5):
    # return top_class if top_prob >= threshold else "uncertain"

    if top_prob >= threshold:
        return top_class
    
    return "uncertain"