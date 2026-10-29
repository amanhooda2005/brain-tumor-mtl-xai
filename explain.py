"""
Explanation study on the trained models - NO retraining.

    python explain.py --ckpts runs/seed0/best.pth runs/seed1/best.pth runs/seed2/best.pth

Candidates: {Grad-CAM, HiResCAM} x {1/16 stage, 1/32 stage}.
The candidate is chosen on the VALIDATION split by the pre-specified criterion
(mean CAM energy inside the tumour, threshold-free); its test numbers are the result.
All candidates' val and test numbers are written to results_xai/eval_xai.json.

The class being explained is the class predicted by the configuration selected in
evaluate.py (read from results/eval.json), so classification and segmentation
columns are unchanged; only the cam_* columns are recomputed.

Then:  python compare.py --new results_xai/eval_per_slice.csv --out results_xai/comparison
"""

import argparse
import csv
import json
import os

import cv2
import numpy as np
import torch

from btmtl import CLASSES
from btmtl.data import find_data_dir, load_arrays, load_masks_at, load_records, load_split
from btmtl.gradcam import METHODS, SWEEP, faithfulness_row, gradcam_batch, overlay
from btmtl.metrics import cluster_bootstrap_diff, cluster_bootstrap_groupdiff, summarize
from btmtl.model import load_checkpoint
from btmtl.utils import get_device, save_json
from evaluate import predict, to_tensor

STAGES = {-2: "1/16", -1: "1/32"}
SHOW = ("cam_energy", "cam_energy_lift", "cam_pointing", "cam_iou", "cam_random", "cam_ceiling")


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--split_file", default="splits/patient_split.json")
    ap.add_argument("--results", default="results", help="folder written by evaluate.py")
    ap.add_argument("--out", default="results_xai")
    ap.add_argument("--eval_size", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--criterion", default="cam_energy", help="val metric used to choose the candidate")
    ap.add_argument("--n_boot", type=int, default=10000)
    return ap.parse_args()


def heatmaps(models, imgs, targets, size, stage, method, device, bs):
    heat = np.zeros((len(imgs), size, size), np.float32)
    grid = None
    for m in models:
        for i in range(0, len(imgs), bs):
            x = to_tensor(imgs[i:i + bs]).to(device)
            t = torch.as_tensor(targets[i:i + len(x)], device=device)
            h, grid = gradcam_batch(m, x, t, size, stage, method)
            heat[i:i + len(x)] += h
    heat /= len(models)
    lo, hi = heat.min((1, 2), keepdims=True), heat.max((1, 2), keepdims=True)
    return np.where(hi - lo > 1e-8, (heat - lo) / (hi - lo + 1e-12), 0).astype(np.float32), grid


def main():
    a = parse()
    device = get_device()
    os.makedirs(os.path.join(a.out, "figures"), exist_ok=True)

    ev = json.load(open(os.path.join(a.results, "eval.json")))
    cfg_name = ev["selected_configuration"]
    tta = cfg_name.endswith("+tta")
    loaded = [load_checkpoint(p, device) for p in a.ckpts]
    if cfg_name.startswith("ensemble"):
        members = [m for m, _ in loaded]
    else:
        members = [loaded[int(cfg_name.split("_")[1].split("+")[0])][0]]
    img_size = loaded[0][1]["img_size"]
    print(f"[xai] explaining configuration '{cfg_name}' ({len(members)} model(s)); predictions with TTA={tta}")

    data_dir = find_data_dir(a.data_dir or loaded[0][1].get("data_dir"))
    recs = load_records(data_dir)
    split = load_split(a.split_file, recs)

    # test predictions: exactly those written by evaluate.py
    with open(os.path.join(a.results, "eval_per_slice.csv"), newline="") as f:
        test_rows = list(csv.DictReader(f))
    test_ids = [recs[i]["image_id"] for i in split["test"]]
    if [r["image_id"] for r in test_rows] != test_ids:
        raise SystemExit("results/eval_per_slice.csv does not match the test split order.")

    D = {}
    for s in ("val", "test"):
        imgs, _, labels = load_arrays(data_dir, recs, split[s], img_size)
        D[s] = dict(imgs=imgs, labels=labels, gt=load_masks_at(data_dir, recs, split[s], a.eval_size),
                    groups=[recs[i]["patient_id"] for i in split[s]])
    D["test"]["pred"] = np.array([CLASSES.index(r["pred_label"]) for r in test_rows])
    cls = sum(predict(m, D["val"]["imgs"], device, tta, 32)[1] for m in members) / len(members)
    D["val"]["pred"] = cls.argmax(1)
    print(f"[xai] val accuracy of explained configuration: {(D['val']['pred'] == D['val']['labels']).mean():.4f}")

    cands = [(m, s) for m in METHODS for s in STAGES]
    res, keep = {}, {}
    for method, stage in cands:
        key = f"{method}@{STAGES[stage]}"
        res[key] = {}
        for s in ("val", "test"):
            heat, grid = heatmaps(members, D[s]["imgs"], D[s]["pred"], a.eval_size, stage, method,
                                  device, a.batch_size)
            rows = [faithfulness_row(heat[k], D[s]["gt"][k], grid) for k in range(len(heat))]
            res[key][s] = {m: float(np.mean([r[m] for r in rows])) for m in SHOW}
            res[key][s]["grid"] = grid
            if s == "test":
                keep[key] = (heat, rows)
        print(f"[xai] {key:16s} val energy {res[key]['val']['cam_energy']:.4f} | "
              f"test energy {res[key]['test']['cam_energy']:.4f}")

    sel = max(res, key=lambda k: res[k]["val"][a.criterion])
    base = f"gradcam@{STAGES[-2]}"
    print(f"\n{'candidate':16s} |   val: energy   lift  point  IoU@.5 |  test: energy   lift  point  IoU@.5")
    for k in res:
        v, t = res[k]["val"], res[k]["test"]
        mark = "  <- selected on val" if k == sel else ("  (original)" if k == base else "")
        print(f"{k:16s} | {v['cam_energy']:12.4f} {v['cam_energy_lift']:6.2f} {v['cam_pointing']:6.3f} "
              f"{v['cam_iou']:7.4f} | {t['cam_energy']:12.4f} {t['cam_energy_lift']:6.2f} "
              f"{t['cam_pointing']:6.3f} {t['cam_iou']:7.4f}{mark}")

    heat, rows = keep[sel]
    g = D["test"]["groups"]
    corr = [r["correct"] == "1" for r in test_rows]
    out_rows = []
    for r, cr in zip(test_rows, rows):
        nr = {k: v for k, v in r.items() if not k.startswith("cam_")}
        nr["cam_pred"] = CLASSES.index(r["pred_label"])
        nr.update(cr)
        out_rows.append(nr)
    with open(os.path.join(a.out, "eval_per_slice.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    paired = {}
    if sel != base:
        _, brows = keep[base]
        for m in ("cam_energy", "cam_energy_lift", "cam_pointing", "cam_iou"):
            paired[m] = cluster_bootstrap_diff([r[m] for r in rows], [r[m] for r in brows], g, a.n_boot)
    within = {m: cluster_bootstrap_groupdiff([r[m] for r in rows], corr, g, a.n_boot)
              for m in ["cam_energy", "cam_pointing", "cam_iou"] + [f"cam_iou@{t}" for t in SWEEP]}

    save_json(dict(explained_configuration=cfg_name, selection_criterion=f"val mean {a.criterion}",
                   selected=sel, original=base, candidates=res,
                   selected_test={m: summarize([r[m] for r in rows], g, a.n_boot) for m in SHOW},
                   selected_vs_original_test=paired, correct_vs_incorrect_test=within,
                   n_errors=int(sum(not c for c in corr)),
                   note=("tumours cover ~1.6% of each slice on average, so a uniform heatmap has "
                         "cam_energy ~0.016 and cam_energy_lift 1.0")),
              os.path.join(a.out, "eval_xai.json"))

    if paired:
        print(f"\n=== test: {sel} vs {base} (same model, same slices; patient-cluster bootstrap) ===")
        for m, r in paired.items():
            print(f"{m:16s} diff {r['diff']:+.4f}  CI [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]  p {r['p']:.4f}")

    # figure: original Grad-CAM vs selected method
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        bheat = keep[base][0]
        order = np.argsort([-float(r["tumour_pixels"]) for r in test_rows])
        pick = list(order[::max(1, len(order) // 6)][:6])
        fig, ax = plt.subplots(len(pick), 3, figsize=(8, 2.7 * len(pick)), squeeze=False)
        for i, k in enumerate(pick):
            im = cv2.resize(D["test"]["imgs"][k], (a.eval_size, a.eval_size), interpolation=cv2.INTER_AREA)
            ax[i, 0].imshow(im, cmap="gray")
            ax[i, 0].set_title(f"{test_rows[k]['image_id']} ({test_rows[k]['true_label']})", fontsize=8)
            for j, (h, nm) in enumerate(((bheat[k], base), (heat[k], sel)), start=1):
                ax[i, j].imshow(overlay(im, h))
                ax[i, j].contour(D["test"]["gt"][k], levels=[0.5], colors="w", linewidths=0.8)
                e = (h * D["test"]["gt"][k]).sum() / max(h.sum(), 1e-8)
                ax[i, j].set_title(f"{nm}  energy {e:.2f}", fontsize=8)
            for j in range(3):
                ax[i, j].axis("off")
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, "figures", "cam_methods.png"), dpi=150)
        plt.close(fig)
    except ImportError:
        pass

    st = res[sel]["test"]
    print(f"\n[xai] selected on VALIDATION: {sel}")
    print(f"[xai] test: energy {st['cam_energy']:.4f} (lift {st['cam_energy_lift']:.2f}x uniform) | "
          f"pointing {st['cam_pointing']:.4f} | IoU@0.5 {st['cam_iou']:.4f} | random floor {st['cam_random']:.4f}")
    print(f"written: {a.out}/eval_xai.json, {a.out}/eval_per_slice.csv, {a.out}/figures/cam_methods.png")
    print(f"next:    python compare.py --new {a.out}/eval_per_slice.csv --out {a.out}/comparison")


if __name__ == "__main__":
    main()
