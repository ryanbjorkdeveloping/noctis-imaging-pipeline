"""

smoke test: load all 9 pretrained DeepStreaks models and confirm each
family has 3 members and the input shape is (144, 144, 1).

Runs in the isolated venv:  .venv_streaks/bin/python streak_testing/test_load_models.py

"""

import os, sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from ztf_classification.deepstreaks import load_deepstreaks, FAMILIES, ARCHS

def main():
    print("loading DeepStreaks models (rb / kd / sl, 3 archs each)...\n")
    models = load_deepstreaks()

    for fam in FAMILIES:
        members = models[fam]
        print(f"Family '{fam}': {len(members)} members")
        for arch in ARCHS:
            shape = models[fam][arch].input_shape[1:] # drop the batch dim
            print (f"  {arch:12s}: input_shape={shape}")
        print()

    # check to see if it's done or not
    all_ok = all(len(models[fam]) == 3 for fam in FAMILIES)
    a_shape = models["rb"]["vgg6"].input_shape[1:]
    print(f"all 3 families have 3 members: {all_ok}")
    print(f"input shape is (144,144,1):   {a_shape == (144, 144, 1)}")


if __name__ == "__main__":
    main()