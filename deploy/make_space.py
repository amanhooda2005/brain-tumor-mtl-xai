"""
Build a folder that can be uploaded as a Hugging Face Space (Docker, free CPU).

    python deploy/make_space.py                              # uses runs/seed*/best.pth and data/raw
    python deploy/make_space.py --ckpts a.pth b.pth c.pth --data_dir path/to/data/raw

Output: ./hf_space/ containing the app, the package, the three models (stored in
half precision to halve their size) and the 469 test-set slices for the gallery.
Then follow deploy/DEPLOY.md to upload it.
"""

import argparse
import csv
import glob
import os
import shutil
import sys

import cv2
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from btmtl.data import load_records, load_split  # noqa: E402

DOCKERFILE = """FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
WORKDIR /home/user/app
RUN pip install --no-cache-dir --user torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt
COPY --chown=user . .
EXPOSE 7860
CMD ["python", "demo/app.py", "--host", "0.0.0.0", "--port", "7860", "--no_browser"]
"""

REQUIREMENTS = """segmentation-models-pytorch==0.5.0
opencv-python-headless>=4.8
numpy>=1.24
fastapi>=0.110
uvicorn>=0.27
python-multipart>=0.0.9
h5py>=3.9
scipy>=1.10
"""

README = """---
title: Brain Tumour MRI Analyser
emoji: 🧠
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: cc-by-4.0
short_description: Tumour mask, tumour type and measured explanations from MRI
---

# Brain Tumour MRI Analyser

Live demo of an explainable multi-task model for brain tumour MRI (BCSE497J Project-I, VIT Vellore):
a shared ImageNet-pretrained ResNet-34 encoder with a U-Net decoder for the tumour mask and a
segmentation-guided head for the tumour type (glioma, meningioma, pituitary), plus Grad-CAM and
HiResCAM explanations scored against the expert mask.

Test set: 469 slices from 33 patients not used in training. Three training runs:
accuracy 98.22 ± 0.44 %, Dice 0.833 ± 0.006 (MAG-Net on the same slices: 94.88 %, 0.760).

Research prototype, not for clinical use. Data: Figshare brain tumour dataset (J. Cheng, 2017), CC BY 4.0.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="*", default=None)
    ap.add_argument("--data_dir", default=os.path.join(ROOT, "data", "raw"))
    ap.add_argument("--split_file", default=os.path.join(ROOT, "splits", "patient_split.json"))
    ap.add_argument("--out", default=os.path.join(os.getcwd(), "hf_space"))
    ap.add_argument("--image_size", type=int, default=384, help="stored size of gallery images")
    a = ap.parse_args()

    ckpts = a.ckpts or sorted(glob.glob(os.path.join(ROOT, "runs", "seed*", "best.pth")))
    if not ckpts:
        raise SystemExit("No checkpoints found (runs/seed*/best.pth). Pass --ckpts.")
    if os.path.exists(a.out):
        shutil.rmtree(a.out)
    os.makedirs(a.out)

    # code
    for d in ("btmtl", "splits"):
        shutil.copytree(os.path.join(ROOT, d), os.path.join(a.out, d),
                        ignore=shutil.ignore_patterns("__pycache__"))
    os.makedirs(os.path.join(a.out, "scripts"))
    shutil.copy(os.path.join(ROOT, "scripts", "prepare_data.py"), os.path.join(a.out, "scripts"))
    shutil.copytree(os.path.join(ROOT, "demo"), os.path.join(a.out, "demo"),
                    ignore=shutil.ignore_patterns("__pycache__", "sample_data", "checkpoints", ".upload*"))
    ev = os.path.join(ROOT, "results", "eval.json")
    if os.path.isfile(ev):
        os.makedirs(os.path.join(a.out, "results"))
        shutil.copy(ev, os.path.join(a.out, "results"))

    # models in half precision
    total = 0
    for k, p in enumerate(ckpts):
        ck = torch.load(p, map_location="cpu", weights_only=False)
        sd = {n: (t.half() if t.is_floating_point() else t) for n, t in ck["model"].items()}
        dst = os.path.join(a.out, "runs", f"seed{k}", "best.pth")
        os.makedirs(os.path.dirname(dst))
        torch.save(dict(model=sd, config=ck["config"]), dst)
        total += os.path.getsize(dst)
        print(f"model {k}: {os.path.relpath(p, ROOT)} -> {os.path.getsize(dst) / 2**20:.0f} MB")

    # test-split gallery
    n = 0
    if os.path.isfile(os.path.join(a.data_dir, "labels.csv")):
        recs = load_records(a.data_dir)
        test = load_split(a.split_file, recs)["test"]
        dd = os.path.join(a.out, "data", "raw")
        os.makedirs(os.path.join(dd, "images")); os.makedirs(os.path.join(dd, "masks"))
        with open(os.path.join(dd, "labels.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["image_id", "label", "patient_id"])
            for i in test:
                r = recs[i]
                img = cv2.imread(os.path.join(a.data_dir, "images", r["image_id"]), cv2.IMREAD_GRAYSCALE)
                if img.shape[0] > a.image_size:
                    img = cv2.resize(img, (a.image_size, a.image_size), interpolation=cv2.INTER_AREA)
                cv2.imwrite(os.path.join(dd, "images", r["image_id"]), img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
                shutil.copy(os.path.join(a.data_dir, "masks", r["image_id"]), os.path.join(dd, "masks", r["image_id"]))
                w.writerow([r["image_id"], r["label_name"], r["patient_id"]])
                n += 1
        print(f"gallery: {n} test slices")
    else:
        print(f"no dataset at {a.data_dir} - the Space will work with uploads only")

    for name, text in (("Dockerfile", DOCKERFILE), ("requirements.txt", REQUIREMENTS), ("README.md", README)):
        with open(os.path.join(a.out, name), "w") as f:
            f.write(text)

    size = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(a.out) for f in fs)
    print(f"\nready: {a.out}  ({size / 2**20:.0f} MB, {len(ckpts)} models, {n} gallery slices)")
    print("next:  follow deploy/DEPLOY.md (hf auth login, then hf upload)")


if __name__ == "__main__":
    main()
