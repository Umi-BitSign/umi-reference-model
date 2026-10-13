"""Small, mergeable LoRA adapters for the frozen SHuBERT/ByT5 decoder."""

from __future__ import annotations

import math

import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        if rank < 1 or alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive")
        if not isinstance(base, nn.Linear):
            raise TypeError("LoRA can only wrap a linear layer")
        self.base = base
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        self.scale = alpha / rank
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.base(value) + self.lora_b(self.lora_a(value)) * self.scale

    @property
    def weight(self) -> torch.Tensor:
        # Fairseq's fused attention reads projection weights instead of calling
        # their forward methods. Return the same effective map on either path.
        return self.base.weight + (self.lora_b.weight @ self.lora_a.weight) * self.scale

    @property
    def bias(self) -> nn.Parameter | None:
        return self.base.bias

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def merged(self) -> nn.Linear:
        with torch.no_grad():
            update = self.lora_b.weight @ self.lora_a.weight
            self.base.weight.add_(update.to(self.base.weight.dtype), alpha=self.scale)
        return self.base


def install_lora(
    model: nn.Module, rank: int, alpha: float, targets: str = "cross_qv"
) -> list[nn.Parameter]:
    if targets not in {"cross_qv", "decoder_attention", "decoder_all"}:
        raise ValueError("unsupported LoRA target set")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable = []
    for block in model.decoder.block:
        attention_targets = [(block.layer[1].EncDecAttention, ("q", "v"))]
        if targets in {"decoder_attention", "decoder_all"}:
            attention_targets = [
                (block.layer[0].SelfAttention, ("q", "v")),
                (block.layer[1].EncDecAttention, ("q", "k", "v", "o")),
            ]
        if targets == "decoder_all":
            feed_forward = block.layer[2].DenseReluDense
            attention_targets.append((feed_forward, ("wi_0", "wi_1", "wo")))
        for parent, names in attention_targets:
            for name in names:
                layer = LoRALinear(getattr(parent, name), rank, alpha)
                setattr(parent, name, layer)
                trainable.extend((layer.lora_a.weight, layer.lora_b.weight))
    if not trainable:
        raise RuntimeError("no decoder cross-attention layers were adapted")
    return trainable


def adapter_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().contiguous()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def load_adapter(model: nn.Module, tensors: dict[str, torch.Tensor]) -> None:
    expected = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    if set(tensors) != expected:
        raise ValueError("adapter tensors do not match the installed LoRA topology")
    missing, unexpected = model.load_state_dict(tensors, strict=False)
    if unexpected or expected.intersection(missing):
        raise RuntimeError("adapter tensors did not load into the expected LoRA weights")


def merge_lora(model: nn.Module) -> None:
    merged_count = 0
    for parent in list(model.modules()):
        for name, layer in list(parent.named_children()):
            if isinstance(layer, LoRALinear):
                setattr(parent, name, layer.merged())
                merged_count += 1
    if merged_count == 0:
        raise RuntimeError("no LoRA layers were available to merge")
