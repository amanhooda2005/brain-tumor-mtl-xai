"""
Data loading, the patient-level split, augmentation and the Dataset.

Expected layout (produced by scripts/prepare_data.py from the Figshare .mat files):

    <data_dir>/images/img_00001.png   grayscale slice, uint8
    <data_dir>/masks/img_00001.png    tumour mask, 0 / 255
    <data_dir>/labels.csv             image_id,label,patient_id

The image ids follow the same numbering as the earlier results (and as the
MAG-Net baseline in baselines/), so per-slice results can be paired by image_id.
"""

import csv
import glob
import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from . import CLASSES

CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
# Figshare numeric codes, accepted in case labels.csv stores numbers
FIGSHARE_CODE = {"1": "meningioma", "2": "glioma", "3": "pituitary"}

# ImageNet mean/std averaged over RGB: the encoder's first conv is built for
# 1 channel by summing the RGB filters, so this is the matching normalisation.
MEAN, STD = 0.449, 0.226


# --------------------------------------------------------------------------- #
# records and data_dir discovery
# --------------------------------------------------------------------------- #
def find_data_dir(data_dir=None):
    """Return a folder containing labels.csv + images/ + masks/.

    If data_dir is None or wrong, search /kaggle/input (Kaggle) and ./data.
    """
    def ok(d):
        return (d and os.path.isfile(os.path.join(d, "labels.csv"))
                and os.path.isdir(os.path.join(d, "images"))
                and os.path.isdir(os.path.join(d, "masks")))

    if ok(data_dir):
        return data_dir
    for root in ("/kaggle/input", "data", "."):
        for p in sorted(glob.glob(os.path.join(root, "**", "labels.csv"), recursive=True)):
            d = os.path.dirname(p)
            if ok(d):
                if data_dir:
                    print(f"[data] '{data_dir}' not usable, found dataset at {d}")
                return d
    raise FileNotFoundError(
        f"No dataset found (looked at '{data_dir}', /kaggle/input, ./data). "
        "Expected labels.csv, images/ and masks/. Run scripts/prepare_data.py first.")


def load_records(data_dir):
    """List of dicts: image_id, label (int), label_name, patient_id (str)."""
    recs = []
    with open(os.path.join(data_dir, "labels.csv"), newline="") as f:
        for row in csv.DictReader(f):
            name = str(row["label"]).strip().lower()
            name = FIGSHARE_CODE.get(name, name)
            if name not in CLASS_TO_IDX:
                raise ValueError(f"Unknown label '{row['label']}' for {row['image_id']}")
            recs.append(dict(image_id=row["image_id"], label=CLASS_TO_IDX[name],
                             label_name=name, patient_id=str(row["patient_id"]).strip()))
    if not recs:
        raise ValueError("labels.csv is empty")
    return recs


# --------------------------------------------------------------------------- #
# patient-level split
# --------------------------------------------------------------------------- #
def _assignment_from_json(js):
    """Accept several split-file layouts and return {patient_id: split}."""
    if "assignment" in js and isinstance(js["assignment"], dict):
        return {str(k): v for k, v in js["assignment"].items()}
    for key in ("patients", "splits", "split"):
        if key in js and isinstance(js[key], dict):
            js = js[key]
            break
    if all(k in js for k in ("train", "val", "test")) and isinstance(js["train"], list):
        out = {}
        for s in ("train", "val", "test"):
            for p in js[s]:
                if str(p) in out and out[str(p)] != s:
                    raise ValueError(f"Patient {p} is listed in both '{out[str(p)]}' and '{s}'")
                out[str(p)] = s
        return out
    if all(v in ("train", "val", "test") for v in js.values()):
        return {str(k): v for k, v in js.items()}
    raise ValueError("Unrecognised split-file format")


def load_split(split_file, records, strict=True):
    """Return {'train': [record idx], 'val': [...], 'test': [...]} from a frozen split.

    Checks: every patient in the data has an assignment, each patient is in
    exactly one split (true by construction of the dict), and the slice
    counts match the summary stored in the split file.
    """
    with open(split_file) as f:
        js = json.load(f)
    assign = _assignment_from_json(js)
    out = {"train": [], "val": [], "test": []}
    missing = set()
    for i, r in enumerate(records):
        s = assign.get(r["patient_id"])
        if s is None:
            missing.add(r["patient_id"])
            continue
        out[s].append(i)
    if missing:
        msg = f"{len(missing)} patients in labels.csv are not in the split file"
        if strict:
            raise ValueError(msg + " - wrong dataset or wrong split file?")
        print("[split] WARNING:", msg)

    pats = {s: {records[i]["patient_id"] for i in idx} for s, idx in out.items()}
    leak = (pats["train"] & pats["val"]) | (pats["train"] & pats["test"]) | (pats["val"] & pats["test"])
    if leak:
        raise RuntimeError(f"Patient leakage across splits: {sorted(leak)[:5]}")

    summary = js.get("summary", {})
    for s in ("train", "val", "test"):
        line = f"[split] {s:5s}: {len(pats[s]):3d} patients, {len(out[s]):4d} slices"
        exp = summary.get(s, {}).get("n_slices")
        if exp is not None:
            line += "  (matches split file)" if exp == len(out[s]) else f"  (MISMATCH: split file says {exp})"
            if exp != len(out[s]) and strict:
                raise ValueError(line)
        print(line)
    print("[split] leakage check: PASS (no patient appears in two splits)")
    return out


def make_split(records, seed=42, train_frac=0.70, val_frac=0.15):
    """Stratified patient-level split (stratified on each patient's label)."""
    from sklearn.model_selection import train_test_split
    plabel = {}
    for r in records:
        plabel.setdefault(r["patient_id"], Counter())[r["label"]] += 1
    pids = sorted(plabel)
    y = [plabel[p].most_common(1)[0][0] for p in pids]
    tr, rest, _, y_rest = train_test_split(pids, y, train_size=train_frac, stratify=y, random_state=seed)
    rel_val = val_frac / (1 - train_frac)
    va, te = train_test_split(rest, train_size=rel_val, stratify=y_rest, random_state=seed)
    assign = {**{p: "train" for p in tr}, **{p: "val" for p in va}, **{p: "test" for p in te}}
    summary = {}
    for s in ("train", "val", "test"):
        rs = [r for r in records if assign[r["patient_id"]] == s]
        cc = Counter(r["label_name"] for r in rs)
        summary[s] = {"n_patients": sum(1 for v in assign.values() if v == s),
                      "n_slices": len(rs), "class_counts": dict(cc)}
    return {"format": "brain_tumor_patient_split_v1", "n_patients": len(pids),
            "assignment": assign, "split_seed": seed, "train_frac": train_frac,
            "val_frac": val_frac, "stratify": True, "summary": summary}


# --------------------------------------------------------------------------- #
# image cache
# --------------------------------------------------------------------------- #
def _read_pair(data_dir, image_id, size):
    img = cv2.imread(os.path.join(data_dir, "images", image_id), cv2.IMREAD_GRAYSCALE)
    msk = cv2.imread(os.path.join(data_dir, "masks", image_id), cv2.IMREAD_GRAYSCALE)
    if img is None or msk is None:
        raise FileNotFoundError(f"Could not read {image_id}")
    if img.shape != (size, size):
        interp = cv2.INTER_AREA if img.shape[0] > size else cv2.INTER_CUBIC
        img = cv2.resize(img, (size, size), interpolation=interp)
    msk = (msk > 127).astype(np.uint8)
    if msk.shape != (size, size):
        msk = cv2.resize(msk, (size, size), interpolation=cv2.INTER_NEAREST)
    return img, msk


def load_arrays(data_dir, records, indices, size, workers=8):
    """Decode and resize the selected slices once and keep them in RAM (uint8)."""
    imgs = np.zeros((len(indices), size, size), np.uint8)
    msks = np.zeros((len(indices), size, size), np.uint8)

    def job(k):
        imgs[k], msks[k] = _read_pair(data_dir, records[indices[k]]["image_id"], size)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(job, range(len(indices))))
    labels = np.array([records[i]["label"] for i in indices], np.int64)
    return imgs, msks, labels


def load_masks_at(data_dir, records, indices, size):
    """Ground-truth masks resized (nearest) to `size` - used for evaluation."""
    out = np.zeros((len(indices), size, size), np.uint8)
    for k, i in enumerate(indices):
        m = cv2.imread(os.path.join(data_dir, "masks", records[i]["image_id"]), cv2.IMREAD_GRAYSCALE)
        m = (m > 127).astype(np.uint8)
        out[k] = cv2.resize(m, (size, size), interpolation=cv2.INTER_NEAREST) if m.shape != (size, size) else m
    return out


# --------------------------------------------------------------------------- #
# augmentation and dataset
# --------------------------------------------------------------------------- #
def get_transforms(train):
    """Geometric + intensity augmentation (albumentations 2.x API). Size is fixed by the cache."""
    import albumentations as A
    if not train:
        return None
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Affine(scale=(0.85, 1.15), translate_percent=(-0.06, 0.06), rotate=(-15, 15),
                 border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0, p=0.7),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        A.OneOf([A.GaussNoise(std_range=(0.01, 0.04)),
                 A.GaussianBlur(blur_limit=(3, 5))], p=0.25),
    ])


class SliceDataset(Dataset):
    def __init__(self, imgs, msks, labels, transform=None):
        self.imgs, self.msks, self.labels, self.tf = imgs, msks, labels, transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, k):
        img, msk = self.imgs[k], self.msks[k]
        if self.tf is not None:
            o = self.tf(image=img, mask=msk)
            img, msk = o["image"], o["mask"]
        x = (img.astype(np.float32) / 255.0 - MEAN) / STD
        return (torch.from_numpy(x)[None],
                torch.from_numpy(msk.astype(np.float32))[None],
                int(self.labels[k]), k)


def class_weights(labels, n_classes=len(CLASSES)):
    """Inverse-frequency weights normalised to mean 1."""
    cnt = np.bincount(labels, minlength=n_classes).astype(np.float64)
    w = cnt.sum() / (n_classes * np.maximum(cnt, 1))
    return torch.tensor(w / w.mean(), dtype=torch.float32)


def patients_of(records, indices):
    return [records[i]["patient_id"] for i in indices]


def describe(records, indices):
    c = Counter(records[i]["label_name"] for i in indices)
    p = len({records[i]["patient_id"] for i in indices})
    return f"{len(indices)} slices / {p} patients / " + ", ".join(f"{k} {c[k]}" for k in CLASSES)
