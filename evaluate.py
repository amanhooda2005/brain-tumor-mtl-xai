"""
Evaluate one or more trained checkpoints on the validation and test splits.

    python evaluate.py --ckpts runs/seed0/best.pth runs/seed1/best.pth runs/seed2/best.pth --out results

Inference configurations scored:
    seed_k          each checkpoint alone
    seed_k+tta      with horizontal-flip test-time augmentation
    ensemble        mean of all checkpoints' probabilities
    ensemble+tta    ensemble with flip TTA

The configuration to report is chosen AUTOMATICALLY on the VALIDATION split
(highest 0.5*Dice + 0.5*macro-F1). Its test numbers are the result. All
configurations' val and test numbers are still written to eval.json.

Segmentation metrics are computed at --eval_size (default 256, the resolution
used for the earlier results and the MAG-Net baseline), so the per-slice CSV can
be paired with baselines/eval_magnet_per_slice.csv by image_id. Dice at the
original image resolution is reported as well (dice_native).
"""

import argparse
import csv
import os
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from btmtl import CLASSES
from btmtl.data import MEAN, STD, find_data_dir, load_arrays, load_masks_at, load_records, load_split
from btmtl.gradcam import SWEEP, faithfulness_row, gradcam_batch, overlay
from btmtl.metrics import (cls_metrics, cluster_bootstrap_groupdiff, patient_ci, seg_per_slice,
                           summarize, wilson_ci)
from btmtl.model import count_params, load_checkpoint
from btmtl.utils import get_device, save_json


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--split_file", default="splits/patient_split.json")
    ap.add_argument("--out", default="results")
    ap.add_argument("--eval_size", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--cam_stage", type=int, default=-2, help="-2 = 1/16 stage, -1 = 1/32 stage")
    ap.add_argument("--force_config", default=None,
                    help="override the val-selected configuration (not recommended)")
    ap.add_argument("--no_cam", action="store_true")
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--limit", type=int, default=0, help="debug only")
    return ap.parse_args()


def to_tensor(imgs_u8):
    return (torch.from_numpy(imgs_u8).float()[:, None] / 255.0 - MEAN) / STD


@torch.no_grad()
def predict(model, imgs, device, tta, bs):
    amp = device.type == "cuda"
    segs, clss = [], []
    for i in range(0, len(imgs), bs):
        x = to_tensor(imgs[i:i + bs]).to(device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            s, c = model(x)
            s, c = torch.sigmoid(s.float()), torch.softmax(c.float(), 1)
            if tta:
                s2, c2 = model(torch.flip(x, dims=[3]))
                s = (s + torch.sigmoid(torch.flip(s2.float(), dims=[3]))) / 2
                c = (c + torch.softmax(c2.float(), 1)) / 2
        segs.append(s[:, 0].half().cpu())
        clss.append(c.cpu())
    return torch.cat(segs), torch.cat(clss).numpy()


def resize_probs(seg, size, bs=64):
    out = []
    for i in range(0, len(seg), bs):
        out.append(F.interpolate(seg[i:i + bs, None].float(), size=(size, size), mode="bilinear",
                                 align_corners=False)[:, 0].numpy())
    return np.concatenate(out)


def score_config(seg, cls, gt_eval, labels, eval_size):
    pred = resize_probs(seg, eval_size) > 0.5
    s = seg_per_slice(pred, gt_eval)
    c = cls_metrics(labels, cls.argmax(1))
    dice = float(s["dice"].mean())
    return dict(dice=dice, iou=float(s["iou"].mean()), accuracy=c["accuracy"], f1=c["f1"],
                score=0.5 * dice + 0.5 * c["f1"]), s, c, pred


def native_dice(seg, data_dir, recs, idx):
    out = []
    for k, i in enumerate(idx):
        m = cv2.imread(os.path.join(data_dir, "masks", recs[i]["image_id"]), cv2.IMREAD_GRAYSCALE) > 127
        p = F.interpolate(seg[k][None, None].float(), size=m.shape, mode="bilinear",
                          align_corners=False)[0, 0].numpy() > 0.5
        den = p.sum() + m.sum()
        out.append(1.0 if den == 0 else 2 * np.logical_and(p, m).sum() / den)
    return np.array(out)


def latency(models, img_size, device, tta, n=50):
    x1 = torch.randn(1, 1, img_size, img_size, device=device)
    amp = device.type == "cuda"
    if device.type != "cuda":
        n = 5

    @torch.no_grad()
    def run(x):
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            for m in models:
                m(x)
                if tta:
                    m(torch.flip(x, dims=[3]))

    for _ in range(5):
        run(x1)
    if amp:
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        run(x1)
    if amp:
        torch.cuda.synchronize()
    return 1000 * (time.perf_counter() - t) / n


def figures(out, rows, cm, imgs, gt, pred, heat, eval_size):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[fig] matplotlib missing - figures skipped")
        return
    os.makedirs(out, exist_ok=True)
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(cm, cmap="Blues")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(3), CLASSES)
    ax.set_yticks(range(3), CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Test confusion matrix")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "confusion_matrix.png"), dpi=200)
    plt.close(fig)

    if heat is None:
        return
    corr = np.array([r["correct"] for r in rows], bool)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for mask, name in ((corr, "correct"), (~corr, "incorrect")):
        if mask.any():
            ax.plot(SWEEP, [np.mean([rows[i][f"cam_iou@{t}"] for i in np.where(mask)[0]]) for t in SWEEP],
                    marker="o", label=f"{name} (n={mask.sum()})")
    ax.set_xlabel("Grad-CAM threshold τ")
    ax.set_ylabel("Mean heatmap–tumour IoU")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "threshold_sweep.png"), dpi=200)
    plt.close(fig)

    order_c = [i for i in np.argsort([-r["cam_energy"] for r in rows]) if corr[i]][:4]
    order_w = list(np.where(~corr)[0][:4])
    pick = order_c + order_w
    fig, axes = plt.subplots(len(pick), 4, figsize=(10, 2.6 * len(pick)), squeeze=False)
    for r, i in enumerate(pick):
        im = cv2.resize(imgs[i], (eval_size, eval_size), interpolation=cv2.INTER_AREA)
        row = rows[i]
        axes[r, 0].imshow(im, cmap="gray")
        axes[r, 0].set_title(f"{row['image_id']}\ntrue {row['true_label']}", fontsize=8)
        axes[r, 1].imshow(gt[i], cmap="gray")
        axes[r, 1].set_title("ground-truth mask", fontsize=8)
        axes[r, 2].imshow(pred[i], cmap="gray")
        axes[r, 2].set_title(f"predicted mask (Dice {row['dice']:.2f})", fontsize=8)
        axes[r, 3].imshow(overlay(im, heat[i]))
        axes[r, 3].contour(gt[i], levels=[0.5], colors="w", linewidths=0.8)
        axes[r, 3].set_title(f"Grad-CAM, pred {row['pred_label']}\nIoU {row['cam_iou']:.2f}", fontsize=8)
        for c in range(4):
            axes[r, c].axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "gradcam_examples.png"), dpi=150)
    plt.close(fig)


def main():
    a = parse()
    device = get_device()
    os.makedirs(a.out, exist_ok=True)

    models, cfgs = zip(*[load_checkpoint(p, device) for p in a.ckpts])
    sizes = {c["img_size"] for c in cfgs}
    if len(sizes) != 1:
        raise SystemExit(f"Checkpoints use different img_size {sizes}; evaluate them separately.")
    img_size = sizes.pop()
    print(f"[eval] {len(models)} checkpoint(s), img_size {img_size}, eval_size {a.eval_size}, device {device}")

    data_dir = find_data_dir(a.data_dir or cfgs[0].get("data_dir"))
    recs = load_records(data_dir)
    split = load_split(a.split_file, recs, strict=not a.limit)
    if a.limit:
        split = {k: v[: a.limit] for k, v in split.items()}

    D = {}
    for s in ("val", "test"):
        imgs, _, labels = load_arrays(data_dir, recs, split[s], img_size)
        D[s] = dict(imgs=imgs, labels=labels, gt=load_masks_at(data_dir, recs, split[s], a.eval_size),
                    groups=[recs[i]["patient_id"] for i in split[s]])

    # ---- raw predictions -------------------------------------------------- #
    P = {}
    for k, m in enumerate(models):
        for tta in (False, True):
            for s in ("val", "test"):
                P[(k, tta, s)] = predict(m, D[s]["imgs"], device, tta, a.batch_size)

    configs = [(f"seed_{k}", [k], False) for k in range(len(models))]
    configs += [(f"seed_{k}+tta", [k], True) for k in range(len(models))]
    if len(models) > 1:
        configs += [("ensemble", list(range(len(models))), False),
                    ("ensemble+tta", list(range(len(models))), True)]

    def combine(members, tta, s):
        seg = sum(P[(k, tta, s)][0].float() for k in members) / len(members)
        cls = sum(P[(k, tta, s)][1] for k in members) / len(members)
        return seg, cls

    table = {}
    print(f"\n{'configuration':16s} | {'val Dice':>8s} {'val acc':>8s} {'val F1':>8s} {'val score':>9s} "
          f"| {'test Dice':>9s} {'test acc':>8s} {'test F1':>8s}")
    for name, mem, tta in configs:
        table[name] = {}
        for s in ("val", "test"):
            seg, cls = combine(mem, tta, s)
            table[name][s] = score_config(seg, cls, D[s]["gt"], D[s]["labels"], a.eval_size)[0]
        v, t = table[name]["val"], table[name]["test"]
        print(f"{name:16s} | {v['dice']:8.4f} {v['accuracy']:8.4f} {v['f1']:8.4f} {v['score']:9.4f} "
              f"| {t['dice']:9.4f} {t['accuracy']:8.4f} {t['f1']:8.4f}")

    selected = max(configs, key=lambda c: table[c[0]]["val"]["score"])
    if a.force_config:
        selected = next(c for c in configs if c[0] == a.force_config)
        print(f"\n[eval] WARNING: configuration forced to {a.force_config} (not val-selected)")
    name, mem, tta = selected
    print(f"\n[eval] selected on VALIDATION: {name}")

    # ---- full analysis of the selected configuration on TEST ------------- #
    T = D["test"]
    seg, cls = combine(mem, tta, "test")
    summ, s, c, pred = score_config(seg, cls, T["gt"], T["labels"], a.eval_size)
    yp = cls.argmax(1)
    dn = native_dice(seg, data_dir, recs, split["test"])

    rows = []
    for k, i in enumerate(split["test"]):
        r = recs[i]
        rows.append(dict(index=k, image_id=r["image_id"], patient_id=r["patient_id"],
                         true_label=r["label_name"], dice=float(s["dice"][k]), iou=float(s["iou"][k]),
                         sensitivity=float(s["sensitivity"][k]), tumour_pixels=float(s["tumour_pixels"][k]),
                         pred_label=CLASSES[yp[k]], correct=int(yp[k] == r["label"]),
                         dice_native=float(dn[k]),
                         **{f"prob_{cn}": float(cls[k, j]) for j, cn in enumerate(CLASSES)}))

    heat = None
    if not a.no_cam:
        print("[eval] Grad-CAM faithfulness ...")
        heat = np.zeros((len(rows), a.eval_size, a.eval_size), np.float32)
        grid, cb = None, max(1, a.batch_size // 2)
        for kk in mem:
            for i in range(0, len(rows), cb):
                x = to_tensor(T["imgs"][i:i + cb]).to(device)
                tgt = torch.as_tensor(yp[i:i + x.shape[0]], device=device)
                h, grid = gradcam_batch(models[kk], x, tgt, a.eval_size, a.cam_stage)
                heat[i:i + x.shape[0]] += h
        heat /= len(mem)
        lo, hi = heat.min((1, 2), keepdims=True), heat.max((1, 2), keepdims=True)
        heat = np.where(hi - lo > 1e-8, (heat - lo) / (hi - lo + 1e-12), 0).astype(np.float32)
        for k, r in enumerate(rows):
            r["cam_pred"] = int(yp[k])
            r.update(faithfulness_row(heat[k], T["gt"][k], grid))

    with open(os.path.join(a.out, "eval_per_slice.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    g = T["groups"]
    n_correct = int(sum(r["correct"] for r in rows))
    seg_block = {m: summarize([r[m] for r in rows], g, a.n_boot)
                 for m in ("dice", "iou", "sensitivity", "dice_native")}
    seg_block["specificity_note"] = ("not reported: background dominates every slice, so "
                                     "specificity is ~0.99 even for an empty prediction")
    acc_ci = patient_ci([r["correct"] for r in rows], g, a.n_boot)
    cls_block = dict(c, accuracy_ci95_wilson=wilson_ci(n_correct, len(rows)),
                     accuracy_ci95_patient=acc_ci, n_errors=len(rows) - n_correct,
                     n_patients_with_errors=len({r["patient_id"] for r in rows if not r["correct"]}))

    cam_block = None
    if heat is not None:
        corr = [bool(r["correct"]) for r in rows]
        cam_block = dict(target_stage=a.cam_stage, grid=grid, threshold=0.5,
                         ensemble_note="heatmaps averaged over ensemble members, no TTA",
                         all={m: summarize([r[m] for r in rows], g, a.n_boot) for m in
                              ("cam_iou", "cam_dice", "cam_pointing", "cam_energy", "cam_ceiling", "cam_random")},
                         sweep={str(t): float(np.mean([r[f"cam_iou@{t}"] for r in rows])) for t in SWEEP},
                         correct_vs_incorrect={})
        for m in ["cam_iou", "cam_energy", "cam_pointing"] + [f"cam_iou@{t}" for t in SWEEP]:
            cam_block["correct_vs_incorrect"][m] = cluster_bootstrap_groupdiff(
                [r[m] for r in rows], corr, g, a.n_boot)

    sel_models = [models[k] for k in mem]
    eff = dict(params_per_model=count_params(models[mem[0]]),
               params_total=int(sum(count_params(m) for m in sel_models)),
               n_models=len(mem), tta=tta, img_size=img_size, device=str(device),
               ms_per_slice_batch1=latency(sel_models, img_size, device, tta))

    result = dict(selected_configuration=name, selected_on="validation score 0.5*Dice + 0.5*macro-F1",
                  checkpoints=list(a.ckpts), encoder=cfgs[0]["encoder"], seg_guided=cfgs[0].get("seg_guided"),
                  eval_size=a.eval_size, data_dir=data_dir, split_file=a.split_file,
                  test_split=dict(n_slices=len(rows), n_patients=len(set(g))),
                  segmentation=seg_block, classification=cls_block,
                  explainability_faithfulness=cam_block, efficiency=eff, all_configurations=table)
    save_json(result, os.path.join(a.out, "eval.json"))
    figures(os.path.join(a.out, "figures"), rows, c["confusion_matrix"], T["imgs"], T["gt"], pred, heat,
            a.eval_size)

    d = seg_block["dice"]
    print(f"\n===== TEST ({name}; {len(rows)} slices, {len(set(g))} patients) =====")
    print(f"Dice        {d['mean']:.4f}  patient-CI [{d['ci95_patient'][0]:.4f}, {d['ci95_patient'][1]:.4f}]"
          f"   (native res {seg_block['dice_native']['mean']:.4f})")
    print(f"IoU         {seg_block['iou']['mean']:.4f}")
    print(f"Accuracy    {c['accuracy']:.4f}  patient-CI [{acc_ci[0]:.4f}, {acc_ci[1]:.4f}]  "
          f"({len(rows) - n_correct} errors from {cls_block['n_patients_with_errors']} patients)")
    print(f"Macro-F1    {c['f1']:.4f}")
    if cam_block:
        al = cam_block["all"]
        print(f"Grad-CAM    IoU@0.5 {al['cam_iou']['mean']:.4f} | pointing {al['cam_pointing']['mean']:.4f} | "
              f"energy {al['cam_energy']['mean']:.4f} | random floor {al['cam_random']['mean']:.4f} | "
              f"ceiling {al['cam_ceiling']['mean']:.4f} (grid {grid})")
    print(f"Efficiency  {eff['params_total'] / 1e6:.2f}M params, {eff['ms_per_slice_batch1']:.1f} ms/slice on {device}")
    print(f"MAG-Net on the same split: Dice 0.7604, accuracy 0.9488, macro-F1 0.9467")
    print(f"\nwritten: {a.out}/eval.json, {a.out}/eval_per_slice.csv, {a.out}/figures/")
    print(f"next:    python compare.py --new {a.out}/eval_per_slice.csv")


if __name__ == "__main__":
    main()
