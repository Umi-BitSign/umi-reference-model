"""Express the original nonoverlapping14px patch convolution as one linear map."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class LinearPatchProjection(nn.Module):
    """Reuse exact original parameters and state keys without resize or padding.

    This is a candidate Metal-backward workaround. It needs real C5 CPU/Metal
    qualification before use; isolated algebra tests are insufficient.
    """

    def __init__(self, original):
        super().__init__()
        if (
            not isinstance(original, nn.Conv2d)
            or original.weight.shape != (384, 3, 14, 14)
            or original.bias is None
            or original.bias.shape != (384,)
            or original.kernel_size != (14, 14)
            or original.stride != (14, 14)
            or original.padding != (0, 0)
            or original.dilation != (1, 1)
            or original.groups != 1
            or original.padding_mode != "zeros"
        ):
            raise ValueError("linear DINO patches require the original14px/stride14 projection")
        self.weight, self.bias = original.weight, original.bias
        for name in (
            "kernel_size",
            "stride",
            "padding",
            "dilation",
            "groups",
            "in_channels",
            "out_channels",
        ):
            setattr(self, name, getattr(original, name))

    def forward(self, pixels):
        if (
            pixels.ndim != 4
            or pixels.shape[1] != 3
            or pixels.dtype != torch.float32
            or min(pixels.shape[2:]) < 14
            or any(side % 14 for side in pixels.shape[2:])
        ):
            raise ValueError("linear DINO patches cannot resize or pad incomplete14px grids")
        patches = F.unfold(pixels, kernel_size=14, stride=14).transpose(1, 2).contiguous()
        values = F.linear(patches, self.weight.flatten(1), self.bias)
        return (
            values.transpose(1, 2)
            .contiguous()
            .reshape(len(pixels), 384, pixels.shape[2] // 14, pixels.shape[3] // 14)
        )
