"""Visual LoRA changes only planned SHuBERT and decoder parameters."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from shubert_visual_lora import (
    install_visual_lora,
    load_visual_adapter,
    load_visual_adapter_with_new_layers,
    visual_adapter_state,
)


def _model():
    model = nn.Module()
    model.encoder = nn.Module()
    model.encoder.adapter = nn.Module()
    adapter = model.encoder.adapter
    adapter.layer_weights = nn.Parameter(torch.ones(3))
    adapter.final_layer = nn.Linear(4, 4)
    adapter.signhubert_adapter = nn.Module()
    adapter.signhubert_adapter.signhubert = nn.Module()
    visual_encoder = nn.Module()
    adapter.signhubert_adapter.signhubert.encoder = visual_encoder
    visual_encoder.layers = nn.ModuleList()
    for _ in range(3):
        layer = nn.Module()
        layer.self_attn = nn.Module()
        layer.self_attn.q_proj = nn.Linear(4, 4)
        layer.self_attn.v_proj = nn.Linear(4, 4)
        visual_encoder.layers.append(layer)
    model.decoder = nn.Module()
    model.decoder.block = nn.ModuleList()
    for _ in range(2):
        block = nn.Module()
        block.layer = nn.ModuleList([nn.Identity(), nn.Module()])
        block.layer[1].EncDecAttention = nn.Module()
        block.layer[1].EncDecAttention.q = nn.Linear(4, 4)
        block.layer[1].EncDecAttention.v = nn.Linear(4, 4)
        model.decoder.block.append(block)
    return model


def test_selects_visual_top_layers_projection_and_decoder_cross_attention():
    model = _model()
    selected = install_visual_lora(model, rank=2, alpha=4, visual_last_layers=1)
    assert all(parameter.requires_grad for parameter in selected.values())
    assert not model.encoder.adapter.signhubert_adapter.signhubert.encoder.layers[
        0
    ].self_attn.q_proj.weight.requires_grad
    assert not model.encoder.adapter.signhubert_adapter.signhubert.encoder.layers[
        1
    ].self_attn.q_proj.weight.requires_grad
    assert any("layers.2.self_attn.q_proj.lora_a" in name for name in selected)
    assert any("decoder.block.0.layer.1.EncDecAttention.q.lora_a" in name for name in selected)
    assert "encoder.adapter.layer_weights" in selected
    assert "encoder.adapter.final_layer.weight" in selected
    assert set(selected) == set(visual_adapter_state(model))


def test_load_requires_exact_finite_tensors():
    model = _model()
    install_visual_lora(model, rank=2, alpha=4, visual_last_layers=1)
    state = visual_adapter_state(model)
    load_visual_adapter(model, state)
    missing = dict(state)
    missing.pop(next(iter(missing)))
    with pytest.raises(ValueError, match="topology"):
        load_visual_adapter(model, missing)
    bad = {name: value.clone() for name, value in state.items()}
    bad[next(iter(bad))].flatten()[0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        load_visual_adapter(model, bad)


def test_rejects_invalid_layer_count():
    with pytest.raises(ValueError, match="exceeds"):
        install_visual_lora(_model(), rank=2, alpha=4, visual_last_layers=4)


def test_expanded_loader_preserves_old_tensors_and_zero_output_new_layers():
    source = _model()
    install_visual_lora(source, rank=2, alpha=4, visual_last_layers=1)
    source_state = visual_adapter_state(source)
    target = _model()
    install_visual_lora(target, rank=2, alpha=4, visual_last_layers=3)
    missing = load_visual_adapter_with_new_layers(target, source_state)
    target_state = visual_adapter_state(target)
    assert all(torch.equal(target_state[name], value) for name, value in source_state.items())
    assert len(missing) == 8
    assert all(
        torch.count_nonzero(target_state[name]).item() == 0
        for name in missing
        if name.endswith(".lora_b.weight")
    )
