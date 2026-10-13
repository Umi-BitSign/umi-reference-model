"""Full cross-attention training and restore guards."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from shubert_cross_attention import (
    cross_attention_parameters,
    load_cross_attention,
    train_cross_attention,
)


class _ToyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        attention = nn.Module()
        attention.EncDecAttention = nn.Module()
        for projection in ("q", "k", "v", "o"):
            setattr(attention.EncDecAttention, projection, nn.Linear(4, 4, bias=False))
        self.layer = nn.ModuleList([nn.Linear(4, 4), attention])


class _ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(4, 4)
        self.decoder = nn.Module()
        self.decoder.block = nn.ModuleList([_ToyBlock(), _ToyBlock()])


def test_only_decoder_cross_attention_becomes_trainable():
    model = _ToyModel()
    selected = train_cross_attention(model)
    names = cross_attention_parameters(model)
    assert len(selected) == 8
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert set(names) == trainable
    assert all(".layer.1.EncDecAttention." in name for name in names)


def test_restore_requires_exact_topology_shape_and_dtype():
    model = _ToyModel()
    original = cross_attention_parameters(model)
    tensors = {name: torch.ones_like(parameter) for name, parameter in original.items()}
    load_cross_attention(model, tensors)
    assert all(torch.equal(parameter, tensors[name]) for name, parameter in original.items())
    with pytest.raises(ValueError, match="topology"):
        load_cross_attention(model, dict(list(tensors.items())[:-1]))
    bad = {**tensors, next(iter(tensors)): torch.zeros(3, 4)}
    with pytest.raises(ValueError, match="shape"):
        load_cross_attention(model, bad)
