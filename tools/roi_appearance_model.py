"""Fresh small-patch appearance encoders; no pretrained vision or language weights."""

from __future__ import annotations

import numpy as np
import torch
from tiny_rgb_model import RmsNorm
from tiny_rgb_mps_safe import PortableGelu
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18, resnet34

STREAMS = ("face_features", "left_hand_features", "right_hand_features")
SIDE = 64
SEED = 20261016


def pixel_tensor(patches):
    if patches.dtype != np.uint8 or patches.ndim != 5 or patches.shape[1:] != (3, SIDE, SIDE, 3):
        raise ValueError("ROI appearance accepts three already-cropped64px uint8 RGB regions")
    return (
        torch.from_numpy(patches.copy())
        .permute(0, 1, 4, 2, 3)
        .contiguous()
        .float()
        .div_(127.5)
        .sub_(1)
    )


def encoder(constructor):
    model = constructor(weights=None, norm_layer=lambda n: RmsNorm(n, channel_first=True))
    model.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
    nn.init.kaiming_normal_(model.conv1.weight, mode="fan_out", nonlinearity="relu")
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, 384)

    def replace(module):
        for name, child in tuple(module.named_children()):
            if isinstance(child, nn.ReLU):
                setattr(module, name, PortableGelu())
            else:
                replace(child)

    replace(model)
    return model


class RoiAppearance(nn.Module):
    """Share fresh34-layer hand CNN; use a separate fresh18-layer face CNN.

    ResNet-shaped residual blocks have a3x3 stride1 stem, no initial max pool,
    explicit channel RMS and portable smooth activations, not stock ResNet weights.
    Body coordinates and language/sequence models are outside this appearance pilot.
    """

    def __init__(self):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(SEED)
            self.hands = encoder(resnet34)
            self.face = encoder(resnet18)

    def trainable_parameters(self):
        return {name: p for name, p in self.named_parameters() if p.requires_grad}

    def forward(self, pixels):
        if (
            pixels.ndim != 5
            or pixels.shape[1:] != (3, 3, SIDE, SIDE)
            or pixels.dtype != torch.float32
        ):
            raise ValueError(
                "fresh ROI network requires normalized FP32 face/left/right64px patches"
            )
        count = len(pixels)
        if count < 1 or not torch.isfinite(pixels).all():
            raise ValueError("fresh ROI network needs finite real image inputs")
        hands = self.hands(torch.cat([pixels[:, 1], pixels[:, 2]], dim=0))
        return {
            "face_features": self.face(pixels[:, 0]),
            "left_hand_features": hands[:count],
            "right_hand_features": hands[count:],
        }


def feature_loss(predictions, targets):
    if set(predictions) != set(STREAMS):
        raise ValueError("ROI appearance feature streams differ")
    components, usable = {}, {}
    for name in STREAMS:
        predicted, target = predictions[name], targets[name].detach().to(predictions[name].device)
        if (
            predicted.ndim != 2
            or predicted.shape[-1] != 384
            or predicted.shape != target.shape
            or not torch.isfinite(predicted).all()
            or not torch.isfinite(target).all()
        ):
            raise ValueError("ROI feature targets or outputs have invalid shape/finite values")
        valid = target.square().sum(-1) > 1e-12
        usable[name] = int(valid.sum())
        components[name] = (
            (1 - F.cosine_similarity(predicted[valid], target[valid], dim=-1)).mean()
            if valid.any()
            else predicted.sum() * 0
        )
    if not any(usable.values()):
        raise ValueError("ROI teacher has no usable appearance target")
    return {"loss": sum(components.values()) / 3, "components": components, "usable_frames": usable}


def temporal_gap(predictions, targets):
    """Same-frame versus half-clip-shifted hand targets on identical valid coverage."""
    gaps = []
    for name in STREAMS[1:]:
        predicted, target = predictions[name], targets[name].to(predictions[name].device)
        if len(target) < 2 or predicted.shape != target.shape:
            raise ValueError("ROI temporal diagnostic requires a complete multi-frame clip")
        shifted = target.roll(len(target) // 2, dims=0)
        valid = (target.square().sum(-1) > 1e-12) & (shifted.square().sum(-1) > 1e-12)
        if valid.any():
            correct = F.cosine_similarity(predicted[valid], target[valid], dim=-1)
            wrong = F.cosine_similarity(predicted[valid], shifted[valid], dim=-1)
            gaps.append((correct - wrong).mean())
    if not gaps:
        raise ValueError("ROI temporal diagnostic has no common valid hand targets")
    return float(torch.stack(gaps).mean())
