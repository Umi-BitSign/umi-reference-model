"""Candidate-only native-input DINO adapter; no model loading, downloads or service access."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

STREAMS = ("face_features", "left_hand_features", "right_hand_features")
PATCH_SIDE = 14


def checked_side(side):
    if type(side) is not int or not 28 <= side <= 224 or side % PATCH_SIDE:
        raise ValueError("native DINO input side must be28-224 and a multiple of14;64 is invalid")
    return side


def input_tensor(patches, *, side=112):
    """Normalize already-native RGB crops without resize, padding or hidden upsampling."""
    checked_side(side)
    if (
        not isinstance(patches, np.ndarray)
        or patches.dtype != np.uint8
        or patches.ndim != 5
        or patches.shape[1:] != (3, side, side, 3)
        or len(patches) < 1
    ):
        raise ValueError(
            "small DINO input requires complete uint8 RGB face/left/right native crops"
        )
    pixels = torch.from_numpy(patches.copy()).permute(0, 1, 4, 2, 3).contiguous().float().div_(255)
    mean = pixels.new_tensor([0.485, 0.456, 0.406])[None, None, :, None, None]
    std = pixels.new_tensor([0.229, 0.224, 0.225])[None, None, :, None, None]
    return (pixels - mean) / std


class SmallInputDino(nn.Module):
    """Wrap two newly loaded candidate backbones; leave original weight files untouched.

    Caller supplies its own private hand/face instances. Left/right hands share
    the hand backbone, as in C5. The unused mask token is frozen because ordinary
    unmasked inference does not use it. Actual DINO numerical/quality/speed
    qualification is required separately before training or selecting this model.
    """

    def __init__(self, hands, face, *, side=112):
        super().__init__()
        if hands is face:
            raise ValueError("small DINO requires separate private hand and face backbones")
        self.side = checked_side(side)
        self.hands, self.face = hands, face
        for model in (hands, face):
            if (
                model.patch_embed.patch_size != (14, 14)
                or model.patch_embed.proj.kernel_size != (14, 14)
                or model.patch_embed.proj.stride != (14, 14)
                or model.patch_embed.proj.padding != (0, 0)
                or model.patch_embed.proj.dilation != (1, 1)
                or model.patch_embed.proj.groups != 1
                or model.patch_embed.proj.in_channels != 3
                or model.patch_embed.proj.out_channels != 384
                or not isinstance(model.mask_token, nn.Parameter)
                or model.mask_token.shape != (1, 384)
            ):
                raise ValueError("small DINO needs the original14px/384-wide backbone contract")
            model.requires_grad_(True).eval()
            model.mask_token.requires_grad_(False)
            model.patch_embed.proj.register_forward_pre_hook(self._first_layer)

    def _first_layer(self, module, args):
        value = args[0]
        if (
            value.ndim != 4
            or value.shape[1:] != (3, self.side, self.side)
            or value.dtype != torch.float32
        ):
            raise ValueError("DINO patch projection received a hidden resize/pad or wrong surface")

    def forward(self, pixels):
        if (
            pixels.ndim != 5
            or pixels.shape[1:] != (3, 3, self.side, self.side)
            or pixels.dtype != torch.float32
            or len(pixels) < 1
            or not torch.isfinite(pixels.reshape(-1)).all()
        ):
            raise ValueError("small DINO requires finite FP32 native face/hand inputs")
        count = len(pixels)
        hands = self.hands(torch.cat([pixels[:, 1], pixels[:, 2]], dim=0))
        result = {
            "face_features": self.face(pixels[:, 0]),
            "left_hand_features": hands[:count],
            "right_hand_features": hands[count:],
        }
        if any(
            value.shape != (count, 384) or not torch.isfinite(value).all()
            for value in result.values()
        ):
            raise ValueError("small DINO changed the original feature width or frame count")
        return result

    def trainable_parameters(self):
        return {name: value for name, value in self.named_parameters() if value.requires_grad}
