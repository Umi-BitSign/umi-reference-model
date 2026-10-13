"""Select and restore the SHuBERT/ByT5 decoder's full cross-attention weights."""

from __future__ import annotations

import torch
from torch import nn


def cross_attention_parameters(model: nn.Module) -> dict[str, nn.Parameter]:
    expected = {}
    for block_number, block in enumerate(model.decoder.block):
        attention = block.layer[1].EncDecAttention
        for projection in ("q", "k", "v", "o"):
            layer = getattr(attention, projection)
            if not isinstance(layer, nn.Linear) or layer.bias is not None:
                raise TypeError("cross-attention projection is not a bias-free linear layer")
            name = f"decoder.block.{block_number}.layer.1.EncDecAttention.{projection}.weight"
            expected[name] = layer.weight
    if not expected:
        raise ValueError("decoder has no cross-attention projection weights")
    return expected


def train_cross_attention(model: nn.Module) -> list[nn.Parameter]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    selected = cross_attention_parameters(model)
    for parameter in selected.values():
        parameter.requires_grad_(True)
    return list(selected.values())


def load_cross_attention(model: nn.Module, tensors: dict[str, torch.Tensor]) -> None:
    selected = cross_attention_parameters(model)
    if set(tensors) != set(selected):
        raise ValueError("cross-attention tensors do not match the frozen model topology")
    with torch.no_grad():
        for name, parameter in selected.items():
            weight = tensors[name]
            if weight.shape != parameter.shape or weight.dtype != parameter.dtype:
                raise ValueError(f"cross-attention tensor shape or dtype differs: {name}")
            parameter.copy_(weight)
