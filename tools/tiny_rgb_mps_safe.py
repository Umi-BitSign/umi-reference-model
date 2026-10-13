"""Equivalent GELU arithmetic for diagnosing the pinned Metal backward implementation."""

from __future__ import annotations

import math

import torch
from torch import nn


class PortableGelu(nn.Module):
    def forward(self, value):
        return value * 0.5 * (1.0 + torch.erf(value / math.sqrt(2.0)))


def replace_native_gelu(model: nn.Module) -> int:
    """Preserve all weights and topology; replace exact GELUs with the same formula."""
    replaced = 0
    for name, child in tuple(model.named_children()):
        if isinstance(child, nn.GELU):
            if child.approximate != "none":
                raise ValueError("portable RGB GELU requires the exact original activation")
            setattr(model, name, PortableGelu())
            replaced += 1
        else:
            replaced += replace_native_gelu(child)
    return replaced
