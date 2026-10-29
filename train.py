"""
Train the proposed multi-task model on the frozen patient-level split.

    python train.py --seed 0 --out runs/seed0

Model selection uses the VALIDATION split only (score = 0.5 * Dice + 0.5 * macro-F1,
computed on the EMA weights). The test split is never touched here.
"""

import argparse
import csv
import math
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from btmtl import CLASSES
from btmtl.data import (SliceDataset, class_weights, describe, find_data_dir, get_transforms,
                        load_arrays, load_records, load_split)
from btmtl.losses import MultiTaskLoss
from btmtl.metrics import cls_metrics, seg_per_slice
from btmtl.model import ProposedMTNet, count_params
from btmtl.utils import EMA, get_device, save_json, seed_everything, worker_init


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", default=None, help="folder with labels.csv, images/, masks/ (auto-detected on Kaggle)")
    ap.add_argument("--split_file", default="splits/patient_split.json")
    ap.add_argument("--out", default="runs/seed0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--weights", default="imagenet", help="'imagenet' or 'none'")
    ap.add_argument("--img_size", type=int, default=384)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=3e-4, help="decoder + heads")
    ap.add_argument("--encoder_lr", type=float, default=1e-4, help="pretrained encoder")
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--warmup_epochs", type=int, default=3)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--w_seg", type=float, default=1.0)
    ap.add_argument("--w_cls", type=float, default=1.0)
    ap.add_argument("--weighting", choices=["fixed", "uncertainty"], default="fixed")
    ap.add_argument("--cls_dropout", type=float, default=0.3)
    ap.add_argument("--ema_decay", type=float, default=0.998, help="0 disables EMA")
    ap.add_argument("--no_seg_guided", action="store_true", help="ablation: plain pooling head")
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="debug: use only N slices per split")
    return ap.parse_args()


@torch.no_grad()
def validate(model, loader, device, amp):
    model.eval()
    preds, gts, yt, yp = [], [], [], []
    for x, m, y, _ in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            seg, logits = model(x)
        preds.append((seg.float() > 0).cpu().numpy()[:, 0])
        gts.append(m.numpy()[:, 0] > 0.5)
        yp.append(logits.argmax(1).cpu().numpy())
        yt.append(y.numpy())
    s = seg_per_slice(np.concatenate(preds), np.concatenate(gts))
    c = cls_metrics(np.concatenate(yt), np.concatenate(yp))
    dice = float(s["dice"].mean())
    return dict(dice=dice, iou=float(s["iou"].mean()), acc=c["accuracy"], f1=c["f1"],
                score=0.5 * dice + 0.5 * c["f1"])


def main():
    a = parse()
    seed_everything(a.seed)
    device = get_device()
    amp = device.type == "cuda" and not a.no_amp
    torch.backends.cudnn.benchmark = True
    os.makedirs(a.out, exist_ok=True)

    data_dir = find_data_dir(a.data_dir)
    recs = load_records(data_dir)
    split = load_split(a.split_file, recs, strict=not a.limit)
    if a.limit:
        split = {k: v[: a.limit] for k, v in split.items()}
    print(f"[data] {data_dir}")
    for s in ("train", "val"):
        print(f"[data] {s}: {describe(recs, split[s])}")

    t0 = time.time()
    tr = load_arrays(data_dir, recs, split["train"], a.img_size)
    va = load_arrays(data_dir, recs, split["val"], a.img_size)
    print(f"[data] cached {len(tr[2]) + len(va[2])} slices at {a.img_size}px in {time.time() - t0:.0f}s")

    g = torch.Generator().manual_seed(a.seed)
    dl_tr = DataLoader(SliceDataset(*tr, get_transforms(True)), batch_size=a.batch_size, shuffle=True,
                       drop_last=True, num_workers=a.workers, pin_memory=True, generator=g,
                       worker_init_fn=worker_init, persistent_workers=a.workers > 0)
    dl_va = DataLoader(SliceDataset(*va), batch_size=a.batch_size * 2, shuffle=False,
                       num_workers=a.workers, pin_memory=True, persistent_workers=a.workers > 0)

    cfg = dict(vars(a), data_dir=data_dir, seg_guided=not a.no_seg_guided, classes=list(CLASSES),
               model="ProposedMTNet")
    weights = None if a.weights.lower() == "none" else a.weights
    try:
        model = ProposedMTNet(a.encoder, weights, cls_dropout=a.cls_dropout,
                              seg_guided=cfg["seg_guided"]).to(device)
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"Could not download the {weights} weights for {a.encoder} ({type(e).__name__}). "
                         "On Kaggle turn Internet ON in the notebook settings (needs a phone-verified "
                         "account). Training without pretrained weights (--weights none) is much weaker.")
    cfg["params"] = count_params(model)
    print(f"[model] ProposedMTNet / {a.encoder} ({weights}) / seg-guided={cfg['seg_guided']} / "
          f"{cfg['params'] / 1e6:.2f}M params")

    cw = class_weights(tr[2]).to(device)
    crit = MultiTaskLoss(cw, a.label_smoothing, a.w_seg, a.w_cls, a.weighting).to(device)

    def groups(params, lr):
        dec = [p for p in params if p.requires_grad and p.ndim > 1]
        nodec = [p for p in params if p.requires_grad and p.ndim <= 1]
        return [dict(params=dec, lr=lr, weight_decay=a.weight_decay),
                dict(params=nodec, lr=lr, weight_decay=0.0)]

    enc = list(model.encoder.parameters())
    enc_ids = {id(p) for p in enc}
    rest = [p for p in model.parameters() if id(p) not in enc_ids] + list(crit.parameters())
    opt = torch.optim.AdamW(groups(enc, a.encoder_lr) + groups(rest, a.lr))

    steps = len(dl_tr)
    total, warm = a.epochs * steps, a.warmup_epochs * steps

    def lr_lambda(i):
        if i < warm:
            return (i + 1) / warm
        t = (i - warm) / max(1, total - warm)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * t))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    ema = EMA(model, a.ema_decay) if a.ema_decay > 0 else None
    save_json(cfg, os.path.join(a.out, "run_config.json"))

    hist_path = os.path.join(a.out, "history.csv")
    fields = ["epoch", "lr", "loss", "loss_seg", "loss_cls", "train_acc",
              "val_dice", "val_iou", "val_acc", "val_f1", "val_score", "sec"]
    with open(hist_path, "w", newline="") as f:
        csv.writer(f).writerow(fields)

    best, best_ep, bad = -1.0, 0, 0
    for ep in range(1, a.epochs + 1):
        model.train()
        t0 = time.time()
        tot = np.zeros(3)
        correct = n = 0
        for x, m, y, _ in dl_tr:
            x, m, y = x.to(device, non_blocking=True), m.to(device, non_blocking=True), y.to(device)
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                seg, logits = model(x)
            loss, ls, lc = crit(seg, logits, m, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            if ema:
                ema.update(model)
            tot += [loss.item(), ls.item(), lc.item()]
            correct += (logits.argmax(1) == y).sum().item()
            n += len(y)
        tot /= steps
        net = ema.ema if ema else model
        v = validate(net, dl_va, device, amp)
        sec = time.time() - t0
        row = [ep, opt.param_groups[2]["lr"], *tot.round(4), round(correct / n, 4),
               *(round(v[k], 4) for k in ("dice", "iou", "acc", "f1", "score")), round(sec, 1)]
        with open(hist_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

        tag = ""
        if v["score"] > best:
            best, best_ep, bad = v["score"], ep, 0
            torch.save(dict(model=net.state_dict(), config=cfg, epoch=ep, val=v),
                       os.path.join(a.out, "best.pth"))
            tag = "  * saved"
        elif ep > a.warmup_epochs:
            bad += 1
        print(f"ep {ep:3d} | loss {tot[0]:.4f} (seg {tot[1]:.4f} cls {tot[2]:.4f}) "
              f"train-acc {correct / n:.4f} | val dice {v['dice']:.4f} acc {v['acc']:.4f} "
              f"f1 {v['f1']:.4f} score {v['score']:.4f} | {sec:.0f}s{tag}", flush=True)
        if bad >= a.patience:
            print(f"[train] early stop: no val improvement for {a.patience} epochs")
            break

    save_json(dict(best_epoch=best_ep, best_val_score=best), os.path.join(a.out, "best_summary.json"))
    print(f"[train] done. best val score {best:.4f} at epoch {best_ep} -> {a.out}/best.pth")


if __name__ == "__main__":
    main()
