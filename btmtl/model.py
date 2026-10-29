"""
The proposed model: a shared-encoder multi-task U-Net with a
segmentation-guided classification head.

    input (1 x H x W)
      |
    pretrained encoder (ResNet-34 / EfficientNet / ... from segmentation_models_pytorch)
      |-- features at 1/2, 1/4, 1/8, 1/16 (F4), 1/32 (F5)
      |
      +--> U-Net decoder with scSE attention --> segmentation logits (1 x H x W)
      |
      +--> classification head:
             F5: global-avg pool, global-max pool, mask-attention pool
             F4: global-avg pool, mask-attention pool
             mask-attention pool = sum(F * p) / sum(p), p = predicted tumour
             probability (detached) resized to the feature map
             -> concat -> Linear-BN-ReLU-Dropout-Linear -> 3 logits

Why the attention pool: plain global pooling averages tumour and healthy
tissue together. Pooling under the predicted mask lets the segmentation
branch tell the classifier where to look, which is the point of training the
two tasks jointly. The probability map is detached so the classification loss
cannot distort the mask to suit itself; a small floor (1e-4) makes the pool
fall back to an ordinary average when the mask is empty.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


class ProposedMTNet(nn.Module):
    def __init__(self, encoder="resnet34", weights="imagenet", num_classes=3,
                 decoder_channels=(256, 128, 64, 32, 16), cls_hidden=512,
                 cls_dropout=0.3, seg_guided=True):
        super().__init__()
        base = smp.Unet(encoder_name=encoder, encoder_weights=weights, in_channels=1,
                        classes=1, decoder_channels=decoder_channels,
                        decoder_attention_type="scse")
        self.encoder = base.encoder
        self.decoder = base.decoder
        self.seg_head = base.segmentation_head
        self.seg_guided = seg_guided

        ch = self.encoder.out_channels
        c5, c4 = ch[-1], ch[-2]
        n_in = (3 * c5 + 2 * c4) if seg_guided else (2 * c5 + c4)
        self.cls_head = nn.Sequential(
            nn.Linear(n_in, cls_hidden), nn.BatchNorm1d(cls_hidden), nn.ReLU(inplace=True),
            nn.Dropout(cls_dropout), nn.Linear(cls_hidden, num_classes))

    @staticmethod
    def _attn_pool(f, p):
        a = F.interpolate(p, size=f.shape[-2:], mode="area") + 1e-4
        return (f * a).sum((2, 3)) / a.sum((2, 3))

    def classify(self, f4, f5, seg_logits):
        f4, f5 = f4.float(), f5.float()          # pool in fp32 (safe under AMP)
        parts = [f5.mean((2, 3)), f5.amax((2, 3)), f4.mean((2, 3))]
        if self.seg_guided:
            p = torch.sigmoid(seg_logits.detach().float())
            parts += [self._attn_pool(f5, p), self._attn_pool(f4, p)]
        return self.cls_head(torch.cat(parts, 1))

    def forward(self, x, return_features=False):
        feats = self.encoder(x)
        seg = self.seg_head(self.decoder(feats))
        logits = self.classify(feats[-2], feats[-1], seg)
        if return_features:
            return seg, logits, feats
        return seg, logits


def build_model(cfg, weights=None):
    """Build from a run config dict (weights=None when loading a checkpoint)."""
    return ProposedMTNet(encoder=cfg.get("encoder", "resnet34"),
                         weights=weights,
                         cls_dropout=cfg.get("cls_dropout", 0.3),
                         seg_guided=cfg.get("seg_guided", True))


def load_checkpoint(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(ck["config"], weights=None)
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), ck["config"]


def count_params(model):
    return sum(p.numel() for p in model.parameters())
