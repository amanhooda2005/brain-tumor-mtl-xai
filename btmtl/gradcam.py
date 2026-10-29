"""
Grad-CAM for the classification branch and the faithfulness scores that test
whether the explanation points at the tumour.

Grad-CAM (Selvaraju et al., 2017) on encoder feature map A (C x h x w):
    alpha_c = mean_{i,j} d y_class / d A_c,ij
    CAM     = ReLU( sum_c alpha_c * A_c )   -> upsampled, min-max normalised to [0, 1]

Default target is the 1/16-resolution encoder stage (24 x 24 for a 384 input):
the classification head reads it directly and, through the last stage, indirectly.

Faithfulness (per slice, against the ground-truth tumour mask):
    cam_iou / cam_dice  heatmap > tau vs mask, tau = 0.5, plus a sweep of tau
    cam_pointing        is the heatmap's peak inside the tumour (pointing game)
    cam_energy          fraction of heatmap mass inside the tumour (threshold-free)
    cam_ceiling         best IoU any map at the CAM's native grid could reach
    cam_random          expected IoU of a random map with the same area (floor)

HiResCAM (Draelos & Carin, 2020) keeps the gradient's spatial pattern instead of
averaging it away:
    CAM = ReLU( sum_c (d y_class / d A_c) * A_c )      (element-wise)
For a head that pools features under the predicted tumour mask, Grad-CAM's
spatial average discards exactly that localisation; HiResCAM does not.

Grad-CAM is computed in batches: in eval mode every sample's score depends
only on its own features, so the gradients do not mix across the batch.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F

SWEEP = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95)


METHODS = ("gradcam", "hirescam")


def gradcam_batch(model, x, target, out_size, stage=-2, method="gradcam"):
    """Return (heatmaps [B, out, out] in [0, 1], native grid size)."""
    model.eval()
    with torch.enable_grad():
        _, logits, feats = model(x, return_features=True)
        f = feats[stage]
        score = logits.float().gather(1, target.view(-1, 1)).sum()
        (g,) = torch.autograd.grad(score, f)
    if method == "gradcam":
        cam = (g.mean((2, 3), keepdim=True) * f).sum(1, keepdim=True)
    elif method == "hirescam":
        cam = (g * f).sum(1, keepdim=True)
    else:
        raise ValueError(f"unknown CAM method {method}")
    cam = F.relu(cam).float()
    grid = cam.shape[-1]
    cam = F.interpolate(cam, size=(out_size, out_size), mode="bilinear", align_corners=False)[:, 0]
    lo = cam.amin((1, 2), keepdim=True)
    hi = cam.amax((1, 2), keepdim=True)
    cam = torch.where(hi - lo > 1e-8, (cam - lo) / (hi - lo + 1e-12), torch.zeros_like(cam))
    return cam.detach().cpu().numpy().astype(np.float32), grid


def _iou(a, b):
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 0.0


def _dice(a, b):
    s = a.sum() + b.sum()
    return float(2 * np.logical_and(a, b).sum() / s) if s else 0.0


def resolution_ceiling(gt, grid):
    """IoU of the best blocky mask a grid x grid map can express."""
    g = gt.astype(bool)
    if not g.any():
        return 0.0
    s = g.shape[0]
    cells = np.zeros((grid, grid), bool)
    edges = np.linspace(0, s, grid + 1).round().astype(int)
    for i in range(grid):
        for j in range(grid):
            cells[i, j] = g[edges[i]:edges[i + 1], edges[j]:edges[j + 1]].any()
    coarse = np.zeros_like(g)
    for i in range(grid):
        for j in range(grid):
            if cells[i, j]:
                coarse[edges[i]:edges[i + 1], edges[j]:edges[j + 1]] = True
    return _iou(coarse, g)


def faithfulness_row(heat, gt, grid, tau=0.5):
    g = gt.astype(bool)
    hb = heat > tau
    a, gf = hb.mean(), g.mean()
    row = dict(cam_iou=_iou(hb, g), cam_dice=_dice(hb, g),
               cam_pointing=float(g.flat[int(np.argmax(heat))]) if g.any() else 0.0,
               cam_energy=float((heat * g).sum() / heat.sum()) if heat.sum() > 0 else 0.0,
               cam_ceiling=resolution_ceiling(g, grid),
               cam_random=float(a * gf / (a + gf - a * gf)) if (a + gf) > 0 else 0.0,
               cam_area=float(a),
               cam_energy_lift=float(((heat * g).sum() / heat.sum()) / gf) if heat.sum() > 0 and gf > 0 else 0.0)
    for t in SWEEP:
        row[f"cam_iou@{t}"] = _iou(heat > t, g)
    return row


def overlay(gray_u8, heat, alpha=0.45):
    rgb = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2RGB)
    hm = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)[..., ::-1]
    return (rgb * (1 - alpha) + hm * alpha).astype(np.uint8)
