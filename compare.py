"""
Paired statistical comparison of the new model against the base-paper model
(MAG-Net, re-implemented and evaluated earlier on the SAME patient split) and
against the previous version of the multi-task model.

    python compare.py --new results/eval_per_slice.csv

Slices are paired by image_id. Confidence intervals and p-values come from a
patient-cluster bootstrap (resampling the 33 test patients, not the 469 slices),
accuracy is also tested with an exact McNemar test, and p-values are Holm-corrected
across the metrics of each comparison.
"""

import argparse
import csv
import os

import numpy as np

from btmtl.metrics import cluster_bootstrap_diff, cluster_bootstrap_groupdiff, holm, mcnemar_exact
from btmtl.utils import save_json

METRICS = [("correct", "Accuracy"), ("dice", "Dice"), ("iou", "IoU"),
           ("cam_iou", "CAM IoU@0.5"), ("cam_energy", "CAM energy in tumour"),
           ("cam_pointing", "Pointing game")]


def read(path):
    with open(path, newline="") as f:
        return {r["image_id"]: r for r in csv.DictReader(f)}


def fmt_p(p):
    return "<0.001" if p < 1e-3 else f"{p:.3f}"


def compare(new, base, base_name, n_boot):
    if set(new) != set(base):
        raise SystemExit(f"{base_name}: the CSVs cover different slices ({len(new)} vs {len(base)}). "
                         "Both must be evaluated on the same test split.")
    ids = sorted(new)
    bad = [i for i in ids if new[i]["true_label"] != base[i]["true_label"]]
    if bad:
        raise SystemExit(f"{base_name}: true labels differ for {len(bad)} slices - different datasets?")
    groups = [new[i]["patient_id"] for i in ids]
    rows = []
    for col, label in METRICS:
        if col not in new[ids[0]] or col not in base[ids[0]]:
            continue
        a = np.array([float(new[i][col]) for i in ids])
        b = np.array([float(base[i][col]) for i in ids])
        r = dict(metric=label, column=col, new=float(a.mean()), baseline=float(b.mean()),
                 **cluster_bootstrap_diff(a, b, groups, n_boot))
        if col == "correct":
            r["mcnemar"] = mcnemar_exact(a, b)
        rows.append(r)
    for r, ph in zip(rows, holm([r["p"] for r in rows])):
        r["p_holm"] = ph
    return dict(baseline=base_name, n_slices=len(ids), n_patients=len(set(groups)), rows=rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--new", default="results/eval_per_slice.csv")
    ap.add_argument("--new_name", default="Proposed MTL (v6)")
    ap.add_argument("--baseline", nargs="+", default=[
        "baselines/eval_magnet_per_slice.csv:MAG-Net (base paper, same split)",
        "baselines/eval_multitask_v5_per_slice.csv:Previous multi-task (v5)"])
    ap.add_argument("--new_cam", default="Grad-CAM", help="explanation method used for --new (for the notes)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--n_boot", type=int, default=10000)
    a = ap.parse_args()
    out = a.out or os.path.join(os.path.dirname(a.new) or ".", "comparison")
    os.makedirs(out, exist_ok=True)

    new = read(a.new)
    if a.new_cam != "Grad-CAM":
        print(f"[compare] note: new CAM = {a.new_cam}, baseline CAM = Grad-CAM -> CAM rows mix model and method")
    report = dict(new=a.new, new_name=a.new_name, comparisons=[])
    md = [f"# {a.new_name}: paired comparison on the test split\n",
          "Patient-cluster bootstrap (10,000 resamples of patients); p-values Holm-corrected per comparison.\n",
          f"CAM rows: new model uses {a.new_cam}; baselines use Grad-CAM at their bottleneck."
          + ("" if a.new_cam == "Grad-CAM" else " CAM differences therefore mix model and explanation method.") + "\n"]
    for spec in a.baseline:
        path, _, name = spec.partition(":")
        name = name or os.path.basename(path)
        if not os.path.exists(path):
            print(f"[compare] skipping missing {path}")
            continue
        res = compare(new, read(path), name, a.n_boot)
        report["comparisons"].append(res)
        md.append(f"\n## vs {name}  ({res['n_slices']} slices, {res['n_patients']} patients)\n")
        md.append("| Metric | New | Baseline | Difference | 95% CI | p (Holm) |")
        md.append("|---|---|---|---|---|---|")
        print(f"\n=== {a.new_name}  vs  {name} ===")
        for r in res["rows"]:
            scale = 100 if r["column"] in ("correct", "cam_pointing") else 1
            unit = " pts" if scale == 100 else ""
            nf = (lambda v: f"{100 * v:.2f}%") if scale == 100 else (lambda v: f"{v:.4f}")
            extra = ""
            if "mcnemar" in r:
                mc = r["mcnemar"]
                extra = f" (McNemar {mc['a_only_correct']} vs {mc['b_only_correct']}, p={fmt_p(mc['p'])})"
            line = (f"| {r['metric']} | {nf(r['new'])} | {nf(r['baseline'])} | "
                    f"{scale * r['diff']:+.{2 if scale == 100 else 4}f}{unit}{extra} | "
                    f"[{scale * r['ci95'][0]:+.{2 if scale == 100 else 4}f}, {scale * r['ci95'][1]:+.{2 if scale == 100 else 4}f}] | "
                    f"{fmt_p(r['p_holm'])} |")
            md.append(line)
            sig = "significant" if r["p_holm"] < 0.05 else "not significant"
            print(f"{r['metric']:28s} new {nf(r['new']):>8s}  base {nf(r['baseline']):>8s}  "
                  f"diff {scale * r['diff']:+.4f}{unit}  CI [{scale * r['ci95'][0]:+.4f}, {scale * r['ci95'][1]:+.4f}]  "
                  f"p_holm {fmt_p(r['p_holm'])}  ({sig}){extra}")

    # ---- within the new model: are correct predictions explained better? ----
    ids = sorted(new)
    if "cam_iou" in new[ids[0]]:
        groups = [new[i]["patient_id"] for i in ids]
        corr = [new[i]["correct"] == "1" for i in ids]
        cols = ["cam_iou", "cam_energy", "cam_pointing"] + [c for c in new[ids[0]] if c.startswith("cam_iou@")]
        within = {}
        md.append("\n## Within the new model: correct vs incorrect predictions\n")
        md.append("| Measure | Correct | Incorrect | Difference | 95% CI (patient) | p |")
        md.append("|---|---|---|---|---|---|")
        print(f"\n=== {a.new_name}: Grad-CAM faithfulness, correct vs incorrect ===")
        for col in cols:
            r = cluster_bootstrap_groupdiff([float(new[i][col]) for i in ids], corr, groups, a.n_boot)
            within[col] = r
            if r is None:
                print(f"{col}: needs at least one correct and one incorrect slice")
                continue
            md.append(f"| {col} | {r['group1_mean']:.4f} | {r['group2_mean']:.4f} | {r['diff']:+.4f} | "
                      f"[{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}] | {fmt_p(r['p'])} |")
            print(f"{col:14s} correct {r['group1_mean']:.4f}  incorrect {r['group2_mean']:.4f}  "
                  f"diff {r['diff']:+.4f}  CI [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]  p {fmt_p(r['p'])}")
        n_err = sum(not c for c in corr)
        md.append(f"\n{n_err} incorrect slices from "
                  f"{len({groups[k] for k, c in enumerate(corr) if not c})} patients. "
                  "With few errors this test has little power; report it with that caveat.\n")
        report["within_new_correct_vs_incorrect"] = within

    save_json(report, os.path.join(out, "comparison.json"))
    with open(os.path.join(out, "comparison.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"\nwritten: {out}/comparison.json, {out}/comparison.md")


if __name__ == "__main__":
    main()
