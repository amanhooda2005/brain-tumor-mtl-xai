"""Unit tests for the core modules (run: python -m pytest -q tests/).

They use small synthetic inputs, so they run on a CPU in under a minute and need
neither the dataset nor trained checkpoints.
"""
import json
import math
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from btmtl.data import load_split, _assignment_from_json          # noqa: E402
from btmtl.gradcam import gradcam_batch, faithfulness_row, resolution_ceiling  # noqa: E402
from btmtl.losses import MultiTaskLoss, soft_dice_loss            # noqa: E402
from btmtl.metrics import (cls_metrics, cluster_bootstrap_diff, holm,  # noqa: E402
                           mcnemar_exact, patient_ci, seg_per_slice)
from btmtl.model import ProposedMTNet, count_params               # noqa: E402
from btmtl.utils import EMA                                       # noqa: E402


# ------------------------------------------------------------------ metrics
def test_TC01_dice_identical_masks_is_one():
    m = np.zeros((1, 32, 32), bool); m[0, 8:16, 8:16] = True
    r = seg_per_slice(m, m)
    assert r["dice"][0] == 1.0 and r["iou"][0] == 1.0


def test_TC02_dice_disjoint_masks_is_zero():
    a = np.zeros((1, 32, 32), bool); a[0, :8, :8] = True
    b = np.zeros((1, 32, 32), bool); b[0, 20:, 20:] = True
    assert seg_per_slice(a, b)["dice"][0] == 0.0


def test_TC03_dice_known_overlap():
    a = np.zeros((1, 10, 10), bool); a[0, :, :4] = True      # 40 px
    b = np.zeros((1, 10, 10), bool); b[0, :, 2:6] = True     # 40 px, 20 shared
    r = seg_per_slice(a, b)
    assert math.isclose(r["dice"][0], 0.5) and math.isclose(r["iou"][0], 20 / 60)


def test_TC04_dice_is_per_slice_not_pooled():
    big = np.zeros((32, 32), bool); big[:, :16] = True
    small = np.zeros((32, 32), bool); small[0, 0] = True
    pred = np.stack([big, np.zeros_like(small)])
    gt = np.stack([big, small])
    d = seg_per_slice(pred, gt)["dice"]
    assert d.tolist() == [1.0, 0.0]          # pooled Dice would hide the missed slice


def test_TC05_classification_metrics_and_confusion():
    r = cls_metrics([0, 0, 1, 2, 2], [0, 1, 1, 2, 0])
    assert math.isclose(r["accuracy"], 0.6)
    assert r["confusion_matrix"] == [[1, 1, 0], [0, 1, 0], [1, 0, 1]]


def test_TC06_mcnemar_matches_hand_calculation():
    # 21 slices only A right, 6 only B right (run 2 vs MAG-Net): exact two-sided p
    a = [1] * 21 + [0] * 6 + [1] * 100
    b = [0] * 21 + [1] * 6 + [1] * 100
    r = mcnemar_exact(a, b)
    p = 2 * sum(math.comb(27, i) for i in range(7)) / 2 ** 27
    assert r["a_only_correct"] == 21 and r["b_only_correct"] == 6
    assert math.isclose(r["p"], p) and r["p"] < 0.01


def test_TC07_holm_correction():
    adj = holm([0.01, 0.04, 0.03])
    assert np.allclose(adj, [0.03, 0.06, 0.06])


def test_TC08_patient_bootstrap_wider_than_slice_bootstrap_for_clustered_data():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(10), 30)
    values = np.repeat(rng.normal(0, 1, 10), 30) + rng.normal(0, 0.05, 300)
    lo, hi = patient_ci(values, groups, n_boot=2000)
    se_slice = values.std() / math.sqrt(len(values))
    assert (hi - lo) > 4 * 1.96 * se_slice      # resampling patients reflects the clustering


def test_TC09_paired_cluster_bootstrap_detects_shift():
    rng = np.random.default_rng(1)
    groups = np.repeat(np.arange(20), 10)
    a = rng.normal(0.8, 0.05, 200)
    r = cluster_bootstrap_diff(a, a - 0.07, groups, n_boot=2000)
    assert math.isclose(r["diff"], 0.07, abs_tol=1e-9) and r["ci95"][0] > 0 and r["p"] < 0.01


# ------------------------------------------------------------------ split
def _records(pids):
    return [dict(image_id=f"img_{i}", label=0, label_name="glioma", patient_id=p)
            for i, p in enumerate(pids)]


def test_TC10_split_loads_and_counts_match(tmp_path):
    recs = _records(["a", "a", "b", "c", "c", "c"])
    js = {"assignment": {"a": "train", "b": "val", "c": "test"},
          "summary": {"train": {"n_slices": 2}, "val": {"n_slices": 1}, "test": {"n_slices": 3}}}
    f = tmp_path / "split.json"; f.write_text(json.dumps(js))
    out = load_split(str(f), recs)
    assert out == {"train": [0, 1], "val": [2], "test": [3, 4, 5]}


def test_TC11_split_count_mismatch_stops(tmp_path):
    recs = _records(["a", "a", "b", "c"])
    js = {"assignment": {"a": "train", "b": "val", "c": "test"},
          "summary": {"train": {"n_slices": 3}}}
    f = tmp_path / "split.json"; f.write_text(json.dumps(js))
    with pytest.raises(ValueError):
        load_split(str(f), recs)


def test_TC12_unassigned_patient_stops(tmp_path):
    recs = _records(["a", "b", "z"])
    js = {"assignment": {"a": "train", "b": "test"}}
    f = tmp_path / "split.json"; f.write_text(json.dumps(js))
    with pytest.raises(ValueError):
        load_split(str(f), recs)


def test_TC13_list_layout_with_leak_is_rejected():
    # a patient listed under two splits must stop the program, not be silently reassigned
    with pytest.raises(ValueError):
        _assignment_from_json({"train": ["a", "b"], "val": ["c"], "test": ["a"]})


# ------------------------------------------------------------------ model
@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return ProposedMTNet(weights=None).eval()


def test_TC14_parameter_count(model):
    assert count_params(model) == 25_595_739


def test_TC15_output_shapes(model):
    with torch.no_grad():
        seg, logits, feats = model(torch.randn(2, 1, 384, 384), return_features=True)
    assert tuple(seg.shape) == (2, 1, 384, 384) and tuple(logits.shape) == (2, 3)
    assert tuple(feats[-2].shape) == (2, 256, 24, 24) and tuple(feats[-1].shape) == (2, 512, 12, 12)


def test_TC16_mask_is_detached_from_classification_loss(model):
    x = torch.randn(1, 1, 128, 128)
    model.zero_grad()
    feats = model.encoder(x)
    dec = model.decoder(feats)
    seg = model.seg_head(dec)
    logits = model.classify(feats[-2].detach(), feats[-1].detach(), seg)
    logits.sum().backward()
    assert all(p.grad is None or float(p.grad.abs().sum()) == 0 for p in model.seg_head.parameters())


def test_TC17_empty_mask_falls_back_to_average_pool():
    f = torch.randn(1, 8, 6, 6)
    p = torch.zeros(1, 1, 24, 24)               # nothing predicted as tumour
    pooled = ProposedMTNet._attn_pool(f, p)
    assert torch.allclose(pooled, f.mean((2, 3)), atol=1e-6)


def test_TC18_attention_pool_reads_only_masked_region():
    f = torch.zeros(1, 1, 4, 4); f[0, 0, 0, 0] = 10.0
    p = torch.zeros(1, 1, 4, 4); p[0, 0, 0, 0] = 1.0
    assert ProposedMTNet._attn_pool(f, p).item() > 9.9


# ------------------------------------------------------------------ losses
def test_TC19_soft_dice_loss_bounds():
    t = torch.zeros(1, 1, 8, 8); t[..., 2:6, 2:6] = 1
    good = soft_dice_loss(torch.where(t > 0, 20.0, -20.0), t)
    bad = soft_dice_loss(torch.where(t > 0, -20.0, 20.0), t)
    assert good.item() < 0.02 and bad.item() > 0.95


def test_TC20_multitask_loss_is_sum_of_parts():
    crit = MultiTaskLoss()
    seg = torch.randn(2, 1, 16, 16); mask = (torch.rand(2, 1, 16, 16) > 0.7).float()
    cls = torch.randn(2, 3); y = torch.tensor([0, 2])
    total, ls, lc = crit(seg, cls, mask, y)
    assert torch.isclose(total, ls + lc)


# ------------------------------------------------------------------ EMA
def test_TC21_ema_moves_towards_model_weights():
    lin = torch.nn.Linear(2, 1)
    ema = EMA(lin, decay=0.9)
    with torch.no_grad():
        lin.weight.add_(1.0)
    before = ema.ema.weight.clone()
    ema.update(lin)
    after = ema.ema.weight
    assert torch.all((after - before).abs() > 0) and torch.all((lin.weight - after).abs() > 0)


# ------------------------------------------------------------------ CAM and faithfulness
def test_TC22_cam_shapes_and_range(model):
    x = torch.randn(2, 1, 384, 384)
    t = torch.tensor([0, 1])
    for method in ("gradcam", "hirescam"):
        h, grid = gradcam_batch(model, x, t, out_size=256, stage=-2, method=method)
        assert h.shape == (2, 256, 256) and grid == 24
        assert h.min() >= 0 and h.max() <= 1


def test_TC23_faithfulness_perfect_heatmap():
    g = np.zeros((64, 64), bool); g[16:32, 16:32] = True
    r = faithfulness_row(g.astype(np.float32), g, grid=64)
    assert r["cam_energy"] == 1.0 and r["cam_pointing"] == 1.0 and r["cam_iou"] == 1.0


def test_TC24_uniform_heatmap_energy_equals_tumour_area():
    g = np.zeros((64, 64), bool); g[:8, :8] = True
    r = faithfulness_row(np.full((64, 64), 0.7, np.float32), g, grid=64)
    assert math.isclose(r["cam_energy"], g.mean(), rel_tol=1e-6)
    assert math.isclose(r["cam_energy_lift"], 1.0, rel_tol=1e-6)


def test_TC25_resolution_ceiling():
    g = np.zeros((256, 256), bool); g[100:110, 100:110] = True
    c24 = resolution_ceiling(g, 24)
    assert 0 < c24 < 1 and resolution_ceiling(g, 256) == 1.0
