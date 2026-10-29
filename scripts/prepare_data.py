"""
Convert the Figshare brain-tumour dataset (Cheng et al., 2017; 3,064 .mat files)
into images/, masks/ and labels.csv.

    python scripts/prepare_data.py --raw_dir /path/to/figshare_mat --out_dir data/raw

Only needed if you do not already have the prepared dataset (on Kaggle the
prepared copy is the 'bt-data' dataset). Image ids are numbered exactly as in
the earlier pipeline (img_{index:05d}.png over the sorted list of .mat files),
so per-slice results stay pairable with baselines/.

cjdata fields: label (1 meningioma, 2 glioma, 3 pituitary), PID, image,
tumorMask (or tumorBorder). Files are MATLAB v7.3 (HDF5) -> h5py; older files -> scipy.
"""

import argparse
import csv
import glob
import os

import cv2
import numpy as np

NAMES = {1: "meningioma", 2: "glioma", 3: "pituitary"}


def _h5_str(ds):
    arr = np.array(ds)
    if arr.dtype.kind in "ui" and arr.size > 1:
        return "".join(chr(int(c)) for c in arr.flatten())
    return str(arr.flatten()[0]) if arr.size else ""


def _mask_from_border(border, shape):
    m = np.zeros(shape, np.uint8)
    pts = np.asarray(border).reshape(-1, 2).astype(np.int32)
    if len(pts) >= 3:
        cv2.fillPoly(m, [pts], 1)
    return m


def read_mat(path):
    try:
        import h5py
        with h5py.File(path, "r") as f:
            cj = f["cjdata"]
            label = int(np.array(cj["label"]).item())
            image = np.array(cj["image"]).T
            mask = (np.array(cj["tumorMask"]).T if "tumorMask" in cj
                    else _mask_from_border(np.array(cj["tumorBorder"]).squeeze(), image.shape))
            pid = _h5_str(cj["PID"]) if "PID" in cj else None
            return label, image, mask, pid
    except OSError:
        from scipy.io import loadmat
        cj = loadmat(path)["cjdata"][0, 0]
        image = np.array(cj["image"])
        mask = (np.array(cj["tumorMask"]) if "tumorMask" in cj.dtype.names
                else _mask_from_border(np.array(cj["tumorBorder"]).squeeze(), image.shape))
        pid = str(np.array(cj["PID"]).flatten()[0]) if "PID" in cj.dtype.names else None
        return int(cj["label"][0, 0]), image, mask, pid


def to_uint8(img):
    img = img.astype(np.float32)
    lo, hi = img.min(), img.max()
    return np.zeros(img.shape, np.uint8) if hi - lo < 1e-6 else ((img - lo) / (hi - lo) * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True)
    ap.add_argument("--out_dir", default="data/raw")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(a.out_dir, "masks"), exist_ok=True)

    files = sorted(glob.glob(os.path.join(a.raw_dir, "**", "*.mat"), recursive=True))
    if not files:
        raise SystemExit(f"No .mat files under {a.raw_dir}")
    rows, skipped = [], 0
    for i, p in enumerate(files):
        if os.path.basename(p).lower() == "cvind.mat":
            continue                       # fold indices, not a slice
        try:
            label, image, mask, pid = read_mat(p)
        except Exception as e:             # noqa: BLE001
            print(f"skip {p}: {e}")
            skipped += 1
            continue
        if label not in NAMES:
            skipped += 1
            continue
        iid = f"img_{i:05d}.png"
        cv2.imwrite(os.path.join(a.out_dir, "images", iid), to_uint8(image))
        cv2.imwrite(os.path.join(a.out_dir, "masks", iid), (mask > 0).astype(np.uint8) * 255)
        rows.append((iid, NAMES[label], (pid or f"unknown_{i:05d}").strip()))
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(files)}")
    with open(os.path.join(a.out_dir, "labels.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "label", "patient_id"])
        w.writerows(rows)
    print(f"done: {len(rows)} slices, {len({r[2] for r in rows})} patients, {skipped} skipped -> {a.out_dir}")


if __name__ == "__main__":
    main()
