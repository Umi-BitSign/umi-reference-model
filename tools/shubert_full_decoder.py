"""Select and serialize the complete ByT5 translation decoder."""

from __future__ import annotations

import torch
from torch import nn


def full_decoder_parameters(model: nn.Module) -> dict[str, nn.Parameter]:
    required = ("decoder", "decoder_emb", "lm_head", "encoder")
    if any(not isinstance(getattr(model, name, None), nn.Module) for name in required):
        raise ValueError("translation model lacks a decoder, embedding, head or encoder")
    decoder_ids = {
        id(parameter)
        for module in (model.decoder, model.decoder_emb, model.lm_head)
        for parameter in module.parameters()
    }
    encoder_ids = {id(parameter) for parameter in model.encoder.parameters()}
    if not decoder_ids or decoder_ids & encoder_ids:
        raise ValueError("decoder parameters are empty or shared with the visual encoder")
    parameters = {
        name: parameter
        for name, parameter in model.named_parameters()
        if id(parameter) in decoder_ids
    }
    if {id(parameter) for parameter in parameters.values()} != decoder_ids:
        raise ValueError("decoder parameters are missing from the model state")
    return parameters


def train_full_decoder(model: nn.Module) -> dict[str, nn.Parameter]:
    selected = full_decoder_parameters(model)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in selected.values():
        parameter.requires_grad_(True)
    return selected


def full_decoder_state(model: nn.Module) -> dict[str, torch.Tensor]:
    selected = full_decoder_parameters(model)
    if any(not parameter.requires_grad for parameter in selected.values()):
        raise ValueError("full decoder was not marked trainable")
    return {
        name: parameter.detach().cpu().contiguous().clone() for name, parameter in selected.items()
    }


def load_full_decoder(model: nn.Module, weights: dict[str, torch.Tensor]) -> None:
    selected = full_decoder_parameters(model)
    if set(weights) != set(selected):
        raise ValueError("full decoder checkpoint has missing or unexpected tensors")
    with torch.no_grad():
        for name, parameter in selected.items():
            weight = weights[name]
            if weight.shape != parameter.shape or weight.dtype != parameter.dtype:
                raise ValueError(f"full decoder tensor has wrong shape or dtype: {name}")
            if not torch.isfinite(weight).all():
                raise ValueError(f"full decoder tensor is non-finite: {name}")
            parameter.copy_(weight)
