"""Preserve native exact-GELU forward values while avoiding the Metal backward kernel."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class _NativeForwardExactGelu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        ctx.save_for_backward(value)
        return F.gelu(value, approximate="none")

    @staticmethod
    def backward(ctx, gradient):
        (value,) = ctx.saved_tensors
        normal_cdf = 0.5 * (1.0 + torch.erf(value * math.sqrt(0.5)))
        normal_pdf = torch.exp(-0.5 * value.square()) / math.sqrt(2.0 * math.pi)
        return gradient * (normal_cdf + value * normal_pdf)


class NativeForwardExactGelu(nn.Module):
    def forward(self, value):
        return _NativeForwardExactGelu.apply(value)


def replace_capacity_gelu(model):
    """Keep all parameters/state keys and exact activation semantics."""
    replacements = 0
    for name, child in tuple(model.named_children()):
        if isinstance(child, nn.GELU):
            if child.approximate != "none":
                raise ValueError("capacity GELU requires exact native semantics")
            setattr(model, name, NativeForwardExactGelu())
            replacements += 1
        else:
            replacements += replace_capacity_gelu(child)
    return replacements
