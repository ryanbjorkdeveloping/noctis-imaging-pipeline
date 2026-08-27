"""
Persistent DeepStreaks scorer — loads the 9 models ONCE, then services many
score-jobs over stdin/stdout. Runs INSIDE .venv_streaks (Keras 2.15); the
main-venv pipeline drives it (Keras-version wall — see run_streak_subprocess).

Why this exists: the one-shot streak_score_batch.py reloads all 9 models (~9 s:
6 s import + 3 s load) on EVERY invocation, and a sweep calls it once per target.
A wide night-sweep of ~80 exposures therefore burns ~12 min just reloading the
same models. This server pays that cost ONCE and scores every subsequent batch
on the warm models — the named "next lever" for scaling the sweep.

Protocol (newline-delimited JSON, one message per line, stdout flushed each time):
  server -> parent:  {"status": "ready"}                    once, after models load
  parent -> server:  {"stamps": p, "index": p, "out": p}    a job (paths to .npy / .json / .json)
  server -> parent:  {"status": "ok",  "out": p, "n": M}    job done, results written to `out`
  server -> parent:  {"status": "err", "msg": "..."}        job failed (server stays alive)
  parent closes stdin (EOF)  ->  server exits 0

The result JSON written to `out` is byte-for-byte the same shape streak_score_batch.py
produces, so the persistent and one-shot paths are interchangeable.
"""
import os
import sys
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from ztf_classification.deepstreaks import load_deepstreaks, score_families, cascade_decision


def _emit(obj):
    """One JSON line to stdout, flushed — the parent blocks on readline()."""
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def _score_job(models, stamps_path, index_path, out_path):
    """Score one batch exactly as streak_score_batch.py does, write `out_path`."""
    batch = np.load(stamps_path).astype(np.float32)
    with open(index_path) as f:
        index = json.load(f)

    scores = score_families(models, batch)
    out = []
    for i, entry in enumerate(index):
        d = cascade_decision(scores, i)          # {rb_pass, kd_pass, sl_short, route}
        # cascade_decision returns numpy bools; json.dump can't serialize np.bool_.
        d = {k: (bool(v) if isinstance(v, np.bool_) else v) for k, v in d.items()}
        d["row_id"] = entry["row_id"]
        out.append(d)

    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    return len(out)


def main():
    models = load_deepstreaks()              # the ~9 s cost, paid exactly once
    _emit({"status": "ready"})

    for line in sys.stdin:                    # blocks until a job or EOF
        line = line.strip()
        if not line:
            continue
        try:
            job = json.loads(line)
            n = _score_job(models, job["stamps"], job["index"], job["out"])
            _emit({"status": "ok", "out": job["out"], "n": n})
        except Exception as exc:              # a bad batch must NOT kill the server
            _emit({"status": "err", "msg": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    main()
