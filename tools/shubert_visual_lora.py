"""Small trainable visual-encoder and translation cross-attention adapter."""

from __future__ import annotations

import torch
from shubert_lora import LoRALinear
from torch import nn


def install_visual_lora(
    model: nn.Module, rank: int, alpha: float, visual_last_layers: int
) -> dict[str, nn.Parameter]:
    if rank < 1 or alpha <= 0 or visual_last_layers < 1:
        raise ValueError("visual LoRA rank, alpha and layer count must be positive")
    adapter = model.encoder.adapter
    visual_layers = adapter.signhubert_adapter.signhubert.encoder.layers
    if not isinstance(visual_layers, nn.ModuleList) or visual_last_layers > len(
        visual_layers
    ):
        raise ValueError("visual LoRA layer count exceeds the SHuBERT encoder")
    if not isinstance(adapter.layer_weights, nn.Parameter) or not isinstance(
        adapter.final_layer, nn.Linear
    ):
        raise ValueError("visual adapter lacks its layer weights or final projection")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for layer in visual_layers[-visual_last_layers:]:
        attention = layer.self_attn
        for name in ("q_proj", "v_proj"):
            base = getattr(attention, name)
            setattr(attention, name, LoRALinear(base, rank, alpha))
    for block in model.decoder.block:
        attention = block.layer[1].EncDecAttention
        for name in ("q", "v"):
            base = getattr(attention, name)
            setattr(attention, name, LoRALinear(base, rank, alpha))
    adapter.layer_weights.requires_grad_(True)
    for parameter in adapter.final_layer.parameters():
        parameter.requires_grad_(True)
    selected = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if not selected or not any("signhubert" in name for name in selected):
        raise RuntimeError("visual LoRA installed no trainable SHuBERT parameters")
    return selected


def visual_adapter_state(model: nn.Module) -> dict[str, torch.Tensor]:
    selected = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if not selected:
        raise ValueError("visual adapter has no trainable parameters")
    return {
        name: parameter.detach().cpu().contiguous().clone()
        for name, parameter in selected.items()
    }


def load_visual_adapter(model: nn.Module, weights: dict[str, torch.Tensor]) -> None:
    selected = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if not selected or set(weights) != set(selected):
        raise ValueError("visual adapter tensors differ from the installed topology")
    with torch.no_grad():
        for name, parameter in selected.items():
            weight = weights[name]
            if weight.shape != parameter.shape or weight.dtype != parameter.dtype:
                raise ValueError(f"visual adapter tensor has wrong shape or dtype: {name}")
            if not torch.isfinite(weight).all():
                raise ValueError(f"visual adapter tensor is non-finite: {name}")
            parameter.copy_(weight)


def load_visual_adapter_with_new_layers(
    model: nn.Module, weights: dict[str, torch.Tensor]
) -> set[str]:
    """Load an adapter into a wider visual-layer topology with zero-output new LoRA."""
    selected = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if not weights or not set(weights) < set(selected):
        raise ValueError("expanded visual adapter needs a strict source subset")
    missing = set(selected) - set(weights)
    if any("signhubert" not in name or ".lora_" not in name for name in missing):
        raise ValueError("expanded visual adapter may add only SHuBERT LoRA tensors")
    missing_a = {name for name in missing if name.endswith(".lora_a.weight")}
    missing_b = {name for name in missing if name.endswith(".lora_b.weight")}
    if (
        not missing_a
        or {name.replace(".lora_a.weight", ".lora_b.weight") for name in missing_a}
        != missing_b
        or any(torch.count_nonzero(selected[name]).item() for name in missing_b)
    ):
        raise ValueError("expanded visual adapter new layers are not zero-output LoRA")
    with torch.no_grad():
        for name, weight in weights.items():
            parameter = selected[name]
            if (
                weight.shape != parameter.shape
                or weight.dtype != parameter.dtype
                or not torch.isfinite(weight).all()
            ):
                raise ValueError(f"expanded visual adapter tensor differs: {name}")
            parameter.copy_(weight)
    return missing
