"""
Score a batch of 144x144x1 streak stamps through the DeepStreaks cascade and
write per-stamp routes to JSON. Runs INSIDE .venv_streaks (Keras 2.15) — the
main-venv pipeline calls this as a subprocess (Keras-version wall).

    .venv_streaks/bin/python streak_testing/streak_score_batch.py \
        --stamps <in.npy> --index <in.json> --out <out.json>
"""
import os, sys, json, argparse
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from ztf_classification.deepstreaks import load_deepstreaks, score_families, cascade_decision

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stamps", required=True)   # (M,144,144,1) .npy
    ap.add_argument("--index",  required=True)   # [{"row_id": ...}, ...] json, same order
    ap.add_argument("--out",    required=True)   # results json to write
    args = ap.parse_args()

    batch = np.load(args.stamps).astype(np.float32)
    with open(args.index) as f:
        index = json.load(f)

    models = load_deepstreaks()
    scores = score_families(models, batch)

    out = []
    for i, entry in enumerate(index):
        d = cascade_decision(scores, i)          # {rb_pass, kd_pass, sl_short, route}
        # cascade_decision returns numpy bools; json.dump can't serialize np.bool_.
        d = {k: (bool(v) if isinstance(v, np.bool_) else v) for k, v in d.items()}
        d["row_id"] = entry["row_id"]
        out.append(d)

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"scored {len(out)} stamps → {args.out}")

if __name__ == "__main__":
    main()
