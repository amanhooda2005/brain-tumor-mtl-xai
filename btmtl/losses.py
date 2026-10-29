import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_dice_loss(logits, target, eps=1.0):
    """1 - per-image soft Dice, averaged over the batch."""
    p = torch.sigmoid(logits.float())
    t = target.float()
    inter = (p * t).sum((1, 2, 3))
    den = p.sum((1, 2, 3)) + t.sum((1, 2, 3))
    return (1 - (2 * inter + eps) / (den + eps)).mean()


class MultiTaskLoss(nn.Module):
    """seg: BCE + soft Dice;  cls: class-weighted CE with label smoothing.

    weighting='fixed'       total = w_seg * L_seg + w_cls * L_cls
    weighting='uncertainty' learned homoscedastic-uncertainty weights (Kendall et al., 2018)
    """

    def __init__(self, class_weights=None, label_smoothing=0.1, w_seg=1.0, w_cls=1.0,
                 weighting="fixed"):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
        self.w_seg, self.w_cls, self.weighting = w_seg, w_cls, weighting
        if weighting == "uncertainty":
            self.log_vars = nn.Parameter(torch.zeros(2))

    def forward(self, seg_logits, cls_logits, mask, label):
        l_seg = F.binary_cross_entropy_with_logits(seg_logits.float(), mask) + soft_dice_loss(seg_logits, mask)
        l_cls = self.ce(cls_logits.float(), label)
        if self.weighting == "uncertainty":
            s = self.log_vars
            total = torch.exp(-s[0]) * l_seg + torch.exp(-s[1]) * l_cls + s.sum()
        else:
            total = self.w_seg * l_seg + self.w_cls * l_cls
        return total, l_seg.detach(), l_cls.detach()
