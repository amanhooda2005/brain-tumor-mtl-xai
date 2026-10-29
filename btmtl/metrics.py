"""Per-slice metrics, classification metrics and the statistics used in the report.

Two kinds of confidence interval are produced:
  * slice bootstrap   - resamples slices (what most papers report)
  * patient bootstrap - resamples patients; slices from one patient are not
                        independent, so this is the honest interval.
"""

import math

import numpy as np

from . import CLASSES


# --------------------------------------------------------------------------- #
# segmentation
# --------------------------------------------------------------------------- #
def seg_per_slice(pred, gt):
    """pred, gt: (N, H, W) bool/0-1 arrays -> dict of per-slice arrays."""
    p = pred.reshape(len(pred), -1).astype(bool)
    g = gt.reshape(len(gt), -1).astype(bool)
    tp = (p & g).sum(1).astype(np.float64)
    ps, gs = p.sum(1).astype(np.float64), g.sum(1).astype(np.float64)
    union = ps + gs - tp
    both_empty = (ps + gs) == 0
    dice = np.where(both_empty, 1.0, 2 * tp / np.maximum(ps + gs, 1))
    iou = np.where(both_empty, 1.0, tp / np.maximum(union, 1))
    sens = np.where(gs == 0, np.nan, tp / np.maximum(gs, 1))
    tn = p.shape[1] - union
    spec = tn / np.maximum(tn + (ps - tp), 1)
    return dict(dice=dice, iou=iou, sensitivity=sens, specificity=spec, tumour_pixels=gs)


# --------------------------------------------------------------------------- #
# classification
# --------------------------------------------------------------------------- #
def cls_metrics(y_true, y_pred, n_classes=len(CLASSES)):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    cm = np.zeros((n_classes, n_classes), int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    per = {}
    precs, recs, f1s = [], [], []
    for c in range(n_classes):
        tp = cm[c, c]
        pr = tp / cm[:, c].sum() if cm[:, c].sum() else 0.0
        rc = tp / cm[c, :].sum() if cm[c, :].sum() else 0.0
        f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
        per[CLASSES[c]] = dict(precision=float(pr), recall=float(rc), f1=float(f1),
                               support=int(cm[c, :].sum()))
        precs.append(pr); recs.append(rc); f1s.append(f1)
    return dict(accuracy=float((y_true == y_pred).mean()),
                precision=float(np.mean(precs)), recall=float(np.mean(recs)),
                f1=float(np.mean(f1s)), per_class=per,
                confusion_matrix=cm.tolist(), confusion_matrix_labels=list(CLASSES),
                majority_class_baseline_accuracy=float(np.bincount(y_true).max() / len(y_true)))


# --------------------------------------------------------------------------- #
# confidence intervals
# --------------------------------------------------------------------------- #
def slice_ci(values, n_boot=10000, seed=0):
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    b = v[idx].mean(1)
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def patient_ci(values, groups, n_boot=10000, seed=0):
    v = np.asarray(values, float)
    g = np.asarray(groups)
    ok = ~np.isnan(v)
    v, g = v[ok], g[ok]
    uniq, inv = np.unique(g, return_inverse=True)
    s = np.bincount(inv, weights=v, minlength=len(uniq))
    c = np.bincount(inv, minlength=len(uniq)).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    b = s[idx].sum(1) / c[idx].sum(1)
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def summarize(values, groups, n_boot=10000):
    v = np.asarray(values, float)
    return dict(mean=float(np.nanmean(v)), std=float(np.nanstd(v)), n=int((~np.isnan(v)).sum()),
                ci95_slice=slice_ci(v, n_boot), ci95_patient=patient_ci(v, groups, n_boot))


# --------------------------------------------------------------------------- #
# paired / grouped tests
# --------------------------------------------------------------------------- #
def cluster_bootstrap_diff(a, b, groups, n_boot=10000, seed=0):
    """Paired mean(a - b) with a patient-cluster bootstrap CI and two-sided p."""
    d = np.asarray(a, float) - np.asarray(b, float)
    ok = ~np.isnan(d)
    d, g = d[ok], np.asarray(groups)[ok]
    uniq, inv = np.unique(g, return_inverse=True)
    s = np.bincount(inv, weights=d, minlength=len(uniq))
    c = np.bincount(inv, minlength=len(uniq)).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    boot = s[idx].sum(1) / c[idx].sum(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
    return dict(diff=float(d.mean()), ci95=[float(lo), float(hi)], p=float(min(p, 1.0)),
                n_slices=int(len(d)), n_patients=int(len(uniq)))


def cluster_bootstrap_groupdiff(values, in_group1, groups, n_boot=10000, seed=0):
    """Unpaired mean(values | group1) - mean(values | group2), resampling patients."""
    v = np.asarray(values, float)
    g1 = np.asarray(in_group1, bool)
    ok = ~np.isnan(v)
    v, g1, g = v[ok], g1[ok], np.asarray(groups)[ok]
    if g1.all() or (~g1).all():
        return None
    uniq, inv = np.unique(g, return_inverse=True)
    k = len(uniq)
    s1 = np.bincount(inv, weights=v * g1, minlength=k)
    c1 = np.bincount(inv, weights=g1.astype(float), minlength=k)
    s2 = np.bincount(inv, weights=v * ~g1, minlength=k)
    c2 = np.bincount(inv, weights=(~g1).astype(float), minlength=k)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, k, size=(n_boot, k))
    n1, n2 = c1[idx].sum(1), c2[idx].sum(1)
    ok = (n1 > 0) & (n2 > 0)
    boot = s1[idx].sum(1)[ok] / n1[ok] - s2[idx].sum(1)[ok] / n2[ok]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
    return dict(group1_mean=float(v[g1].mean()), group2_mean=float(v[~g1].mean()),
                diff=float(v[g1].mean() - v[~g1].mean()), ci95=[float(lo), float(hi)],
                p=float(min(p, 1.0)), n_group1=int(g1.sum()), n_group2=int((~g1).sum()),
                n_patients_group2=int((c2 > 0).sum()))


def mcnemar_exact(a_correct, b_correct):
    a, b = np.asarray(a_correct, int), np.asarray(b_correct, int)
    n01 = int(((a == 1) & (b == 0)).sum())   # A right, B wrong
    n10 = int(((a == 0) & (b == 1)).sum())   # A wrong, B right
    n = n01 + n10
    if n == 0:
        return dict(a_only_correct=0, b_only_correct=0, p=1.0)
    k = min(n01, n10)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return dict(a_only_correct=n01, b_only_correct=n10, p=float(min(1.0, 2 * tail)))


def holm(pvals):
    m = len(pvals)
    order = np.argsort(pvals)
    adj, running = np.empty(m), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj.tolist()


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [float(c - h), float(c + h)]
