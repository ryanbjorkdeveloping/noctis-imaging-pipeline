"""
Weak-label sourcing for the stamp-classifier evaluation

AUTO-FETCH: ask ALeRCE for objects it confidently types as each class, and use
those as the gold set. No hardcoded oids.

HONESTY CAVEAT: both the 'prediction' (from classify_astrophysical) and the
'label' (the class we asked ALeRCE for) come from ALeRCE's same server. So this
validates OUR decision-rule plumbing (drop-bogus / renormalize / threshold),
NOT independent classifier accuracy.
"""

from ztf_classification.stamp_classifier import find_objects_of_class, ASTRO_CLASSES

def fetch_examples(client, n_per_class=3, min_prob=0.8):
    """Return {class_name: [oid, ...]} — a few confident ALeRCE objects per class.
    Loops find_objects_of_class over SN / AGN / VS / asteroid."""
    # build a dict: for each cls in ASTRO_CLASSES, call find_objects_of_class(...)
    # and store the returned oid list under that class key. Return the dict.
    
    examples = {}

    for cls in ASTRO_CLASSES:
        oids = find_objects_of_class(client, cls, n=n_per_class, min_prob=min_prob)
        examples[cls] = oids
    
    return examples