"""
Run the ALeRCE light-curve clasifier on our eval set
Preprocess -> extract features -> classify. Run from the sandbox:

.venv_lc/bin/python lc_testing/run_lc_classifier.py
"""

import os, sys
import pandas as pd
from lc_classifier.features import ZTFLightcurvePreprocessor, ZTFFeatureExtractor
from lc_classifier.classifier.models import HierarchicalRandomForest
import pandas as pd

PR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL = os.path.join(PR, "ztfdata", "lc_eval")

# load our eval set
detections = pd.read_parquet(os.path.join(EVAL, "lightcurves.parquet")).set_index("oid")
objects = pd.read_csv(os.path.join(EVAL, "objects.csv")).set_index("oid")
print(f"Loaded {objects.index.nunique()} objects and {len(detections)} detections")

# preprocess (transform to the library's clean format)
preprocessor = ZTFLightcurvePreprocessor(stream=False)
detections = preprocessor.preprocess(detections, objects=objects)
print(f"after preprocess: {detections.index.nunique()} objects survived, {len(detections)} total detections")

# extract features (SLOW - P4J period-finding runs on every curve)
feature_extractor = ZTFFeatureExtractor(bands=(1,2), stream=False)
non_detections = pd.DataFrame()
features = feature_extractor.compute_features(
    detections=detections,
    non_detections=non_detections,
)
print(f"features extracted: {features.shape} (objects x features)")
print("sample feature columns:", list(features.columns)[:8])

# classify with the pretrained Hierarchical Random Forest
model = HierarchicalRandomForest()
model.download_model()
model.load_model(model.MODEL_PICKLE_PATH)
print("model loaded")

predictions = model.predict(features)
print("\nPREDICTIONS:")
print(predictions)

#grade predictions vs the known ALeRCE class
# map any fine class up to its top-level branch (periodic / stochastic / transient)

TOP = {
    "LPV": "Periodic", "RRL": "Periodic", "E": "Periodic", "CEP": "Periodic",
    "DSCT": "Periodic", "Periodic-Other": "Periodic",
    "QSO": "Stochastic", "AGN": "Stochastic", "YSO": "Stochastic",
    "Blazar": "Stochastic", "CV/Nova": "Stochastic",
    "SNIa": "Transient", "SNII": "Transient", "SNIbc": "Transient", "SLSN": "Transient",
}

truth = objects["true_class"]
pred = predictions["classALeRCE"]
graded = truth.to_frame("true").join(pred.rename("pred")).dropna()
graded["true_top"] = graded["true"].map(TOP)
graded["pred_top"] = graded["pred"].map(TOP)

fine_acc = (graded["true"] == graded["pred"]).mean()
top_acc = (graded["true_top"] == graded["pred_top"]).mean()

print(f"\n=== GRADING ({len(graded)} objects) ===")
print(f"top-level accuracy (Periodic/Stochastic/Transient): {top_acc:.3f}")
print(f"fine-class accuracy (exact subclass):               {fine_acc:.3f}")
print("\nper-object:")
print(graded[["true", "pred", "true_top", "pred_top"]].to_string())
