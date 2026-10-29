"""
Live demo server for the proposed multi-task model.

    python demo/app.py                       # from the project folder
    python demo/app.py --ckpts runs/seed0/best.pth runs/seed1/best.pth runs/seed2/best.pth --data_dir data/raw

Opens http://127.0.0.1:8000 in the browser. Works fully offline once the
packages are installed: no CDN, no internet, no external fonts.

What it does for one MRI slice:
  * tumour mask (U-Net decoder) and tumour type with probabilities (classification head)
  * Grad-CAM and HiResCAM heatmaps for the predicted class
  * if the expert mask is known (test-set slice or .mat upload): Dice / IoU of the
    predicted mask and the faithfulness scores of both heatmaps
"""

import argparse
import base64
import csv
import glob
import json
import os
import sys
import tempfile
import threading
import time
import webbrowser

import cv2
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from btmtl import CLASSES  # noqa: E402
from btmtl.data import MEAN, STD, load_records, load_split  # noqa: E402
from btmtl.gradcam import faithfulness_row  # noqa: E402

try:
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.responses import FileResponse, JSONResponse, Response
    import uvicorn
except ImportError:
    raise SystemExit("Missing packages. Run:  pip install fastapi uvicorn python-multipart")

EVAL = 256          # resolution used for every reported metric (same as the MAG-Net baseline)
SHOW = 384          # resolution of the images sent to the browser
MAX_UPLOAD = 20 * 2**20  # bytes
NICE = {"glioma": "Glioma", "meningioma": "Meningioma", "pituitary": "Pituitary tumour"}


# --------------------------------------------------------------------------- #
# model back-ends: full checkpoints (needs segmentation_models_pytorch) or the
# lightweight TorchScript export made by deploy/make_render.py (needs only torch)
# --------------------------------------------------------------------------- #
class FullModel:
    def __init__(self, path, device):
        from btmtl.model import count_params, load_checkpoint
        self.net, self.cfg = load_checkpoint(path, device)
        self.n_params = count_params(self.net)

    def __call__(self, x):
        return self.net(x)

    def cam_inputs(self, x, cls_idx):
        """Return (gradient, activation) at the 1/16-scale encoder stage."""
        with torch.enable_grad():
            _, logits, feats = self.net(x, return_features=True)
            f = feats[-2]
            (g,) = torch.autograd.grad(logits[0, cls_idx], f)
        return g, f.detach()


class LiteModel:
    """The same network split into four TorchScript parts.

    stem: x -> features up to the 1/16 stage, layer4: 1/16 -> 1/32,
    decoder: all features -> mask logits, head: (F4, F5, mask logits) -> class logits.
    The class score depends on F4 only through layer4 and the head (the mask is
    detached inside the head), so the gradient taken here equals the full model's.
    """

    def __init__(self, folder, device):
        import warnings
        warnings.filterwarnings("ignore", category=FutureWarning, module="torch.jit")
        meta = json.load(open(os.path.join(folder, "meta.json")))
        load = lambda n: torch.jit.load(os.path.join(folder, n), map_location=device).eval()  # noqa: E731
        self.stem, self.layer4, self.decoder, self.head = (load(n) for n in ("stem.pt", "layer4.pt", "decoder.pt", "head.pt"))
        for part in (self.stem, self.layer4, self.decoder, self.head):
            for prm in part.parameters():
                prm.requires_grad_(False)
        # LITE_PRECISION=bf16 halves memory (for 512 MB hosts); the small head stays fp32
        self.precision = os.environ.get("LITE_PRECISION", "fp32").lower()
        self.dtype = torch.bfloat16 if self.precision == "bf16" else torch.float32
        if self.dtype != torch.float32:
            self.stem, self.layer4, self.decoder = (m.to(self.dtype) for m in (self.stem, self.layer4, self.decoder))
        self.cfg = meta["config"]
        self.n_params = meta["params"]

    def _features(self, x):
        f0, f1, f2, f3, f4 = self.stem(x.to(self.dtype))
        f5 = self.layer4(f4)
        return f0, f1, f2, f3, f4, f5

    def __call__(self, x):
        with torch.no_grad():
            f0, f1, f2, f3, f4, f5 = self._features(x)
            seg = self.decoder(f0, f1, f2, f3, f4, f5)
            return seg.float(), self.head(f4, f5, seg).float()

    def cam_inputs(self, x, cls_idx):
        with torch.no_grad():
            f0, f1, f2, f3, f4, f5 = self._features(x)
            seg = self.decoder(f0, f1, f2, f3, f4, f5)
        with torch.enable_grad():
            f4 = f4.clone().requires_grad_(True)
            logits = self.head(f4, self.layer4(f4), seg)
            (g,) = torch.autograd.grad(logits[0, cls_idx], f4)
        return g.float(), f4.detach().float()


# --------------------------------------------------------------------------- #
# state
# --------------------------------------------------------------------------- #
class State:
    models, cfg, device, img_size, lite = [], {}, None, 384, False
    ckpts, data_dir, records, test_idx, by_id = [], None, [], [], {}
    lock = threading.Lock()


S = State()


def pick_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "mps" and not torch.backends.mps.is_available():
        print("[demo] MPS not available, using CPU")
        return torch.device("cpu")
    return torch.device(name)


def find_ckpts(given):
    if given:
        return given
    for pat in ("runs/seed*/best.pth", "demo/checkpoints/*.pth", "checkpoints/*.pth"):
        hits = sorted(glob.glob(os.path.join(ROOT, pat)))
        if hits:
            return hits
    return []


def find_data(given):
    cands = [given] if given else []
    cands += [os.path.join(ROOT, "data", "raw"), os.path.join(ROOT, "data"), os.path.join(HERE, "sample_data")]
    for d in cands:
        if d and os.path.isfile(os.path.join(d, "labels.csv")) and os.path.isdir(os.path.join(d, "images")):
            return d
    return None


def load(args):
    S.device = pick_device(args.device)
    if S.device.type == "cpu":
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 0)) or max(1, os.cpu_count() or 1))
    lite = args.lite_dir and os.path.isfile(os.path.join(args.lite_dir, "meta.json"))
    if lite:
        S.lite = True
        S.models = [LiteModel(args.lite_dir, S.device)]
        S.ckpts = [os.path.join(args.lite_dir, "stem.pt")]
        print(f"[demo] lightweight model from {args.lite_dir} on {S.device} ({S.models[0].precision})")
    else:
        S.ckpts = find_ckpts(args.ckpts)
        if not S.ckpts:
            raise SystemExit(
                "No checkpoints found. Put the trained models at runs/seed0/best.pth, runs/seed1/best.pth, "
                "runs/seed2/best.pth (or demo/checkpoints/*.pth), or pass --ckpts.")
        S.models = [FullModel(p, S.device) for p in S.ckpts]
        print(f"[demo] {len(S.models)} model(s) on {S.device}: {', '.join(os.path.relpath(p, ROOT) for p in S.ckpts)}")
    S.cfg = S.models[0].cfg
    S.img_size = S.cfg.get("img_size", 384)

    S.data_dir = find_data(args.data_dir)
    if S.data_dir:
        S.records = load_records(S.data_dir)
        S.by_id = {r["image_id"]: i for i, r in enumerate(S.records)}
        try:
            import contextlib, io
            with contextlib.redirect_stdout(io.StringIO()):
                split = load_split(args.split_file, S.records, strict=False)
            S.test_idx = split["test"]
            n_pat = len({S.records[i]["patient_id"] for i in S.test_idx})
            print(f"[demo] test split: {len(S.test_idx)} slices from {n_pat} patients (leakage check passed)")
        except Exception as e:  # noqa: BLE001
            print(f"[demo] split file not usable ({e}); gallery shows all slices")
            S.test_idx = list(range(len(S.records)))
        print(f"[demo] dataset {S.data_dir}: {len(S.test_idx)} test slices in the gallery")
    else:
        print("[demo] no dataset folder found - gallery disabled, uploads still work")

    with torch.no_grad():   # warm-up so the first live request is fast
        x = torch.zeros(1, 1, S.img_size, S.img_size, device=S.device)
        for m in S.models:
            m(x)


# --------------------------------------------------------------------------- #
# image helpers
# --------------------------------------------------------------------------- #
def to_u8(img):
    img = img.astype(np.float32)
    lo, hi = float(img.min()), float(img.max())
    return np.zeros(img.shape, np.uint8) if hi - lo < 1e-6 else ((img - lo) / (hi - lo) * 255).astype(np.uint8)


def resize(img, size, mask=False):
    if img.shape[:2] == (size, size):
        return img
    if mask:
        return cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA if img.shape[0] > size else cv2.INTER_CUBIC)


def png_b64(rgb_or_gray):
    arr = rgb_or_gray
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", arr)
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


def contour(rgb, mask, color, thick=2):
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(rgb, cs, -1, color, thick)
    return rgb


def mask_overlay(gray, pred, gt):
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB).astype(np.float32)
    cyan = np.array([46, 196, 214], np.float32)
    m = pred.astype(bool)
    rgb[m] = rgb[m] * 0.45 + cyan * 0.55
    rgb = rgb.astype(np.uint8)
    contour(rgb, pred, (46, 196, 214), 2)
    if gt is not None:
        contour(rgb, gt, (255, 255, 255), 2)
    return rgb


def heat_overlay(gray, heat, gt):
    hm = cv2.applyColorMap((np.clip(heat, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)[..., ::-1]
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    out = (rgb * 0.45 + hm * 0.55).astype(np.uint8)
    if gt is not None:
        contour(out, gt, (255, 255, 255), 2)
    return out


def read_upload(name, data):
    """Return (gray uint8, mask uint8 or None, true label or None)."""
    ext = os.path.splitext(name.lower())[1]
    if ext == ".mat":
        fd, tmp = tempfile.mkstemp(suffix=".mat")
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        try:
            sys.path.insert(0, os.path.join(ROOT, "scripts"))
            from prepare_data import NAMES, read_mat
            label, image, mask, _ = read_mat(tmp)
        finally:
            os.remove(tmp)
        return to_u8(image), (mask > 0).astype(np.uint8), NAMES.get(label)
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise HTTPException(400, f"Could not read {name}. Use a PNG, JPG, TIFF, BMP or Figshare .mat file.")
    if img.ndim == 3:
        img = cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        img = to_u8(img)
    return img, None, None


# --------------------------------------------------------------------------- #
# inference
# --------------------------------------------------------------------------- #
def cams(model, x, cls_idx):
    g, f = model.cam_inputs(x, cls_idx)
    out = {}
    for name, cam in (("gradcam", (g.mean((2, 3), keepdim=True) * f).sum(1, keepdim=True)),
                      ("hirescam", (g * f).sum(1, keepdim=True))):
        cam = F.interpolate(F.relu(cam).float(), size=(EVAL, EVAL), mode="bilinear", align_corners=False)[0, 0]
        cam = cam.detach().cpu().numpy()
        lo, hi = cam.min(), cam.max()
        out[name] = (cam - lo) / (hi - lo) if hi - lo > 1e-8 else np.zeros_like(cam)
    return out, f.shape[-1]


def analyse(gray, gt=None, true_label=None, ensemble=True, tta=True, image_id=None):
    t0 = time.perf_counter()
    members = S.models if ensemble else S.models[:1]
    img = resize(gray, S.img_size)
    x = ((torch.from_numpy(img).float() / 255.0 - MEAN) / STD)[None, None].to(S.device)
    with S.lock, torch.no_grad():
        seg = torch.zeros(1, 1, S.img_size, S.img_size, device=S.device)
        prob = torch.zeros(1, len(CLASSES), device=S.device)
        n = 0
        for m in members:
            s, c = m(x)
            seg += torch.sigmoid(s); prob += torch.softmax(c, 1); n += 1
            if tta:
                s2, c2 = m(torch.flip(x, dims=[3]))
                seg += torch.flip(torch.sigmoid(s2), dims=[3]); prob += torch.softmax(c2, 1); n += 1
        seg, prob = (seg / n)[0, 0], (prob / n)[0].cpu().numpy()
    t_pred = time.perf_counter() - t0
    k = int(prob.argmax())

    t1 = time.perf_counter()
    with S.lock:
        heat = {"gradcam": 0, "hirescam": 0}
        for m in members:
            h, grid = cams(m, x, k)
            for key in heat:
                heat[key] = heat[key] + h[key]
    for key in heat:
        h = heat[key] / len(members)
        lo, hi = h.min(), h.max()
        heat[key] = (h - lo) / (hi - lo) if hi - lo > 1e-8 else np.zeros_like(h)
    t_cam = time.perf_counter() - t1

    seg_eval = F.interpolate(seg[None, None], size=(EVAL, EVAL), mode="bilinear", align_corners=False)[0, 0].cpu().numpy() > 0.5
    seg_show = cv2.resize(seg_eval.astype(np.uint8), (SHOW, SHOW), interpolation=cv2.INTER_NEAREST)
    gray_show = resize(gray, SHOW)
    gt_eval = resize(gt, EVAL, mask=True) if gt is not None else None
    gt_show = resize(gt, SHOW, mask=True) if gt is not None else None

    out = dict(
        image_id=image_id,
        prediction=CLASSES[k], prediction_name=NICE[CLASSES[k]],
        probabilities={c: float(p) for c, p in zip(CLASSES, prob)},
        true_label=true_label, correct=(true_label == CLASSES[k]) if true_label else None,
        tumour_area_pct=float(seg_eval.mean() * 100),
        setting=f"{len(members)} model{'s' if len(members) > 1 else ''}{' + flip' if tta else ''}",
        time_ms=dict(prediction=round(t_pred * 1000), explanation=round(t_cam * 1000)),
        images=dict(
            input=png_b64(gray_show),
            mask=png_b64(mask_overlay(gray_show, seg_show, gt_show)),
            gradcam=png_b64(heat_overlay(gray_show, cv2.resize(heat["gradcam"], (SHOW, SHOW)), gt_show)),
            hirescam=png_b64(heat_overlay(gray_show, cv2.resize(heat["hirescam"], (SHOW, SHOW)), gt_show)),
        ),
        has_expert_mask=gt is not None,
    )
    if gt_eval is not None:
        p, g = seg_eval, gt_eval.astype(bool)
        inter, ps, gs = (p & g).sum(), p.sum(), g.sum()
        out["segmentation"] = dict(dice=float(2 * inter / (ps + gs)) if ps + gs else 1.0,
                                   iou=float(inter / (ps + gs - inter)) if ps + gs - inter else 1.0,
                                   tumour_pct_expert=float(g.mean() * 100))
        out["faithfulness"] = {}
        for key in heat:
            r = faithfulness_row(heat[key], gt_eval, grid)
            out["faithfulness"][key] = dict(energy=r["cam_energy"], pointing=bool(r["cam_pointing"]),
                                            iou=r["cam_iou"], lift=r["cam_energy_lift"])
    return out


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #
app = FastAPI(title="Brain tumour MTL demo")


@app.get("/")
def index():
    return FileResponse(os.path.join(HERE, "static", "index.html"))


@app.get("/static/{name}")
def static(name):
    p = os.path.join(HERE, "static", os.path.basename(name))
    if not os.path.isfile(p):
        raise HTTPException(404)
    return FileResponse(p)


@app.get("/api/status")
def status():
    return dict(device=str(S.device), n_models=len(S.models), encoder=S.cfg.get("encoder"),
                img_size=S.img_size, params_per_model=S.models[0].n_params, lite=S.lite,
                precision=getattr(S.models[0], "precision", "fp32"),
                checkpoints=[os.path.relpath(p, ROOT) for p in S.ckpts],
                dataset=bool(S.data_dir), n_test=len(S.test_idx), classes=list(CLASSES))


@app.get("/api/samples")
def samples(cls: str = "all", limit: int = 48, seed: int = 0):
    if not S.data_dir:
        return dict(items=[])
    idx = [i for i in S.test_idx if cls == "all" or S.records[i]["label_name"] == cls]
    rng = np.random.default_rng(seed)
    if seed and len(idx) > limit:
        idx = sorted(rng.choice(idx, size=limit, replace=False).tolist())
    items = [dict(image_id=S.records[i]["image_id"], label=S.records[i]["label_name"],
                  patient_id=S.records[i]["patient_id"]) for i in idx[:limit]]
    return dict(items=items, total=len(idx))


_thumbs = {}


@app.get("/api/thumb/{image_id}")
def thumb(image_id: str):
    if image_id not in S.by_id:
        raise HTTPException(404)
    if image_id not in _thumbs:
        img = cv2.imread(os.path.join(S.data_dir, "images", image_id), cv2.IMREAD_GRAYSCALE)
        ok, buf = cv2.imencode(".jpg", cv2.resize(img, (144, 144), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 85])
        _thumbs[image_id] = buf.tobytes()
    return Response(_thumbs[image_id], media_type="image/jpeg")


@app.post("/api/predict_sample")
def predict_sample(image_id: str = Form(...), ensemble: bool = Form(True), tta: bool = Form(True)):
    if image_id not in S.by_id:
        raise HTTPException(404, f"{image_id} is not in the dataset")
    r = S.records[S.by_id[image_id]]
    gray = cv2.imread(os.path.join(S.data_dir, "images", image_id), cv2.IMREAD_GRAYSCALE)
    gt = (cv2.imread(os.path.join(S.data_dir, "masks", image_id), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
    out = analyse(gray, gt, r["label_name"], ensemble, tta, image_id)
    out["patient_id"] = r["patient_id"]
    out["in_test_split"] = S.by_id[image_id] in set(S.test_idx)
    return out


@app.post("/api/predict")
async def predict(file: UploadFile = File(...), mask: UploadFile = File(None),
                  ensemble: bool = Form(True), tta: bool = Form(True)):
    data = await file.read()
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, f"File is larger than {MAX_UPLOAD // 2**20} MB.")
    gray, gt, label = read_upload(file.filename, data)
    if mask is not None and mask.filename:
        m = cv2.imdecode(np.frombuffer(await mask.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise HTTPException(400, f"Could not read the mask {mask.filename}.")
        gt = (cv2.resize(m, gray.shape[::-1], interpolation=cv2.INTER_NEAREST) > 127).astype(np.uint8)
    out = analyse(gray, gt, label, ensemble, tta, file.filename)
    return out


@app.get("/api/results")
def results():
    with open(os.path.join(HERE, "results_summary.json")) as f:
        res = json.load(f)
    ev = os.path.join(ROOT, "results", "eval.json")
    if os.path.isfile(ev):
        with open(ev) as f:
            e = json.load(f)
        res["local_eval"] = dict(selected=e.get("selected_configuration"),
                                 accuracy=e["classification"]["accuracy"],
                                 dice=e["segmentation"]["dice"]["mean"])
    return res


@app.exception_handler(Exception)
async def errors(request, exc):
    if isinstance(exc, HTTPException):
        return JSONResponse({"error": exc.detail}, status_code=exc.status_code)
    return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=500)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpts", nargs="*", default=None)
    ap.add_argument("--lite_dir", default=os.environ.get("LITE_DIR"),
                    help="folder with the TorchScript export from deploy/make_render.py")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--split_file", default=os.path.join(ROOT, "splits", "patient_split.json"))
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    ap.add_argument("--no_browser", action="store_true")
    a = ap.parse_args()
    load(a)
    url = f"http://{'127.0.0.1' if a.host == '0.0.0.0' else a.host}:{a.port}"
    print(f"[demo] ready -> {url}")
    if not a.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
