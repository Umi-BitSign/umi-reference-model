"""Equivalent float32 Fairseq exact-GELU arithmetic for Metal gradient qualification."""

from __future__ import annotations

import math

import torch
from torch import nn


class PortableLayerNorm(nn.Module):
    """Keep native LayerNorm's centering, epsilon and exact learned affine parameters."""

    def __init__(self, original):
        super().__init__()
        self.normalized_shape, self.eps = original.normalized_shape, original.eps
        self.weight, self.bias = original.weight, original.bias

    def forward(self, value):
        if value.dtype != torch.float32:
            raise ValueError("qualified winner LayerNorm requires float32")
        axes = tuple(range(value.ndim - len(self.normalized_shape), value.ndim))
        centered = value - value.mean(dim=axes, keepdim=True)
        result = centered * torch.rsqrt(centered.square().mean(dim=axes, keepdim=True) + self.eps)
        if self.weight is not None:
            result = result * self.weight
        if self.bias is not None:
            result = result + self.bias
        return result


def replace_native_layer_norms(model):
    replaced = 0
    for name, child in tuple(model.named_children()):
        if isinstance(child, nn.LayerNorm):
            setattr(model, name, PortableLayerNorm(child))
            replaced += 1
        else:
            replaced += replace_native_layer_norms(child)
    return replaced


def portable_fairseq_gelu(value):
    original_dtype = value.dtype
    floats = value.float()
    return (0.5 * floats * (1.0 + torch.erf(floats / math.sqrt(2.0)))).to(original_dtype)


def replace_fairseq_exact_gelu(model):
    replaced = 0
    for module in model.modules():
        function = getattr(module, "activation_fn", None)
        if (
            getattr(function, "__module__", None) == "fairseq.modules.gelu"
            and getattr(function, "__name__", None) == "gelu"
        ):
            module.activation_fn = portable_fairseq_gelu
            replaced += 1
    return replaced
