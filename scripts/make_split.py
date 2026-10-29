"""
Create a new stratified patient-level split. NOT needed for the main results:
splits/patient_split.json is the frozen split used for every reported number
(including the MAG-Net baseline). Use this only for a different dataset.

    python scripts/make_split.py --data_dir data/raw --out splits/my_split.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from btmtl.data import find_data_dir, load_records, make_split  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} exists - refusing to overwrite a split file.")
    js = make_split(load_records(find_data_dir(a.data_dir)), seed=a.seed)
    with open(a.out, "w") as f:
        json.dump(js, f, indent=2)
    print({k: v["n_slices"] for k, v in js["summary"].items()}, "->", a.out)


if __name__ == "__main__":
    main()
