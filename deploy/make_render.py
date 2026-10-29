"""
Build a small, self-contained folder for Render's free Docker plan (512 MB RAM).

    python deploy/make_render.py                       # uses runs/seed0/best.pth and data/raw
    python deploy/make_render.py --ckpt runs/seed0/best.pth --data_dir data/raw

What it does
  * exports ONE trained model to four TorchScript files (fp32, identical numbers),
    so the server needs only PyTorch, not segmentation_models_pytorch / timm
    (that saves ~190 MB of memory)
  * copies the web app and the 469 test-set slices for the gallery
  * writes a Dockerfile that installs the CPU-only PyTorch build
  * checks that the exported model gives the same output as the original

Output: ./render_app/  ->  push it to a GitHub repository, then create a Render
web service from that repository (see deploy/DEPLOY_RENDER.md).
"""

import argparse
import csv
import glob
import json
import os
import shutil
import sys

import cv2
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from btmtl.data import load_records, load_split  # noqa: E402
from btmtl.model import ProposedMTNet, count_params, load_checkpoint  # noqa: E402


class Stem(nn.Module):
    """ResNet encoder up to the 1/16 stage (layer3)."""

    def __init__(self, enc):
        super().__init__()
        self.conv1, self.bn1, self.relu, self.maxpool = enc.conv1, enc.bn1, enc.relu, enc.maxpool
        self.layer1, self.layer2, self.layer3 = enc.layer1, enc.layer2, enc.layer3

    def forward(self, x):
        f1 = self.relu(self.bn1(self.conv1(x)))
        f2 = self.layer1(self.maxpool(f1))
        f3 = self.layer2(f2)
        f4 = self.layer3(f3)
        return x, f1, f2, f3, f4


class Decoder(nn.Module):
    def __init__(self, decoder, seg_head):
        super().__init__()
        self.decoder, self.seg_head = decoder, seg_head

    def forward(self, f0, f1, f2, f3, f4, f5):
        return self.seg_head(self.decoder([f0, f1, f2, f3, f4, f5]))


class Head(nn.Module):
    _attn_pool = staticmethod(ProposedMTNet._attn_pool)
    classify = ProposedMTNet.classify

    def __init__(self, cls_head, seg_guided):
        super().__init__()
        self.cls_head, self.seg_guided = cls_head, seg_guided

    def forward(self, f4, f5, seg):
        return self.classify(f4, f5, seg)


DOCKERFILE = """FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MALLOC_ARENA_MAX=2 LITE_DIR=lite LITE_PRECISION=bf16
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
RUN pip install torch=={torch} --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 10000
CMD ["sh", "-c", "python demo/app.py --host 0.0.0.0 --port ${{PORT:-10000}} --no_browser"]
"""

REQUIREMENTS = """numpy>=1.24
opencv-python-headless>=4.8
fastapi>=0.110
uvicorn>=0.27
python-multipart>=0.0.9
h5py>=3.9
"""

RENDER_YAML = """services:
  - type: web
    name: brain-tumour-mri
    runtime: docker
    plan: free
    dockerfilePath: ./Dockerfile
    envVars:
      - key: LITE_PRECISION
        value: bf16
    healthCheckPath: /api/status
"""

DOCKERIGNORE = """**/__pycache__
*.pyc
.git
"""

README = """# Brain Tumour MRI Analyser (web version)

Live demo of an explainable multi-task model for brain tumour MRI (BCSE497J Project-I, VIT Vellore).
One trained model (seed {seed}) runs on a shared CPU; the full project uses a three-model ensemble.

Research prototype, not for clinical use. Data: Figshare brain tumour dataset (J. Cheng, 2017), CC BY 4.0.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "runs", "seed0", "best.pth"))
    ap.add_argument("--data_dir", default=os.path.join(ROOT, "data", "raw"))
    ap.add_argument("--split_file", default=os.path.join(ROOT, "splits", "patient_split.json"))
    ap.add_argument("--out", default=os.path.join(os.getcwd(), "render_app"))
    ap.add_argument("--image_size", type=int, default=384, help="stored size of gallery images")
    a = ap.parse_args()

    if not os.path.isfile(a.ckpt):
        hits = sorted(glob.glob(os.path.join(ROOT, "runs", "seed*", "best.pth")))
        if not hits:
            raise SystemExit(f"Checkpoint not found: {a.ckpt}")
        a.ckpt = hits[0]
    if os.path.exists(a.out):
        shutil.rmtree(a.out)
    os.makedirs(os.path.join(a.out, "lite"))

    # ---- export the model -------------------------------------------------- #
    torch.manual_seed(0)
    model, cfg = load_checkpoint(a.ckpt, torch.device("cpu"))
    model.float().eval()
    size = cfg.get("img_size", 384)
    parts = dict(stem=Stem(model.encoder).eval(),
                 layer4=model.encoder.layer4.eval(),
                 decoder=Decoder(model.decoder, model.seg_head).eval(),
                 head=Head(model.cls_head, cfg.get("seg_guided", True)).eval())
    x = torch.randn(1, 1, size, size)
    with torch.no_grad():
        feats = parts["stem"](x)
        f5 = parts["layer4"](feats[4])
        seg = parts["decoder"](*feats, f5)
        traced = dict(stem=torch.jit.trace(parts["stem"], (x,)),
                      layer4=torch.jit.trace(parts["layer4"], (feats[4],)),
                      decoder=torch.jit.trace(parts["decoder"], (*feats, f5)),
                      head=torch.jit.trace(parts["head"], (feats[4], f5, seg)))
    for name, t in traced.items():
        t.save(os.path.join(a.out, "lite", f"{name}.pt"))

    # ---- check: exported parts == original model ---------------------------- #
    x = torch.randn(2, 1, size, size)
    with torch.no_grad():
        seg_ref, log_ref = model(x)
        loaded = {n: torch.jit.load(os.path.join(a.out, "lite", f"{n}.pt")) for n in traced}
        fs = loaded["stem"](x)
        f5 = loaded["layer4"](fs[4])
        seg_new = loaded["decoder"](*fs, f5)
        log_new = loaded["head"](fs[4], f5, seg_new)
    d_seg = (seg_ref - seg_new).abs().max().item()
    d_log = (log_ref - log_new).abs().max().item()
    print(f"export check: max difference mask {d_seg:.2e}, class scores {d_log:.2e}")
    if d_seg > 1e-3 or d_log > 1e-3:
        raise SystemExit("Exported model does not match the original - stopping.")

    seed = os.path.basename(os.path.dirname(a.ckpt))
    with open(os.path.join(a.out, "lite", "meta.json"), "w") as f:
        json.dump(dict(config={k: v for k, v in cfg.items() if isinstance(v, (int, float, str, bool, list))},
                       params=count_params(model), source=os.path.relpath(a.ckpt, ROOT),
                       torch=torch.__version__), f, indent=2)
    for n in traced:
        print(f"  lite/{n}.pt  {os.path.getsize(os.path.join(a.out, 'lite', n + '.pt')) / 2**20:.0f} MB")

    # ---- code (no model.py: the server never imports segmentation_models_pytorch) ---- #
    os.makedirs(os.path.join(a.out, "btmtl"))
    for f in ("__init__.py", "data.py", "gradcam.py", "metrics.py"):
        shutil.copy(os.path.join(ROOT, "btmtl", f), os.path.join(a.out, "btmtl", f))
    shutil.copytree(os.path.join(ROOT, "splits"), os.path.join(a.out, "splits"))
    os.makedirs(os.path.join(a.out, "scripts"))
    shutil.copy(os.path.join(ROOT, "scripts", "prepare_data.py"), os.path.join(a.out, "scripts"))
    shutil.copytree(os.path.join(ROOT, "demo"), os.path.join(a.out, "demo"),
                    ignore=shutil.ignore_patterns("__pycache__", "sample_data", "checkpoints", ".upload*"))
    ev = os.path.join(ROOT, "results", "eval.json")
    if os.path.isfile(ev):
        os.makedirs(os.path.join(a.out, "results"))
        shutil.copy(ev, os.path.join(a.out, "results"))

    # ---- test-split gallery ------------------------------------------------- #
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
        print(f"no dataset at {a.data_dir} - the web app will work with uploads only")

    tv = torch.__version__.split("+")[0]
    for name, text in (("Dockerfile", DOCKERFILE.format(torch=tv)), ("requirements.txt", REQUIREMENTS),
                       ("render.yaml", RENDER_YAML), (".dockerignore", DOCKERIGNORE),
                       ("README.md", README.format(seed=seed.replace("seed", "")))):
        with open(os.path.join(a.out, name), "w") as f:
            f.write(text)

    total = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(a.out) for f in fs)
    biggest = max((os.path.getsize(os.path.join(dp, f)), f) for dp, _, fs in os.walk(a.out) for f in fs)
    print(f"\nready: {a.out}  ({total / 2**20:.0f} MB; largest file {biggest[1]} {biggest[0] / 2**20:.0f} MB, "
          f"GitHub limit 100 MB)")
    print("next:  follow deploy/DEPLOY_RENDER.md")


if __name__ == "__main__":
    main()
