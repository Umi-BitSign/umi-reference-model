"""Learn video/text attention queries and keys with the original decoder fixed.

This is an unqualified architecture. It supplies no corpus, optimizer, trained
model, exact linguistic boundaries or quality claim.
"""

from __future__ import annotations

import contextlib
import math

import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutput


class AlignmentProjection(nn.Module):
    def __init__(self, base, *, rank=8, seed=20261019):
        super().__init__()
        if not isinstance(base, nn.Linear) or base.bias is not None:
            raise ValueError("alignment projection requires an original bias-free linear map")
        if type(rank) is not int or not 1 <= rank <= 64:
            raise ValueError("alignment rank must be an integer in1..64")
        self.base = base.requires_grad_(False).eval()
        self.rank, self.enabled = rank, True
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.input_projection = nn.Linear(base.in_features, rank, bias=False)
            self.output_projection = nn.Linear(rank, base.out_features, bias=False)
            nn.init.zeros_(self.output_projection.weight)
        self.scaling = 1 / math.sqrt(rank)

    @property
    def weight(self):
        return self.base.weight

    def forward(self, hidden):
        original = self.base(hidden)
        if not self.enabled or (
            not torch.is_grad_enabled()
            and torch.count_nonzero(self.output_projection.weight).item() == 0
        ):
            return original
        return original + self.scaling * self.output_projection(self.input_projection(hidden))


class CrossAttentionAlignment(nn.Module):
    """All decoder cross-attention q/k maps; every original value stays frozen."""

    def __init__(self, language, *, rank=8, seed=20261019):
        super().__init__()
        if any(isinstance(module, AlignmentProjection) for module in language.modules()):
            raise ValueError("alignment adapters cannot be installed twice")
        self.language = language.requires_grad_(False).eval()
        if hasattr(language, "frozen_winner_weights"):
            language.frozen_winner_weights = True
        self.rank, self.seed = rank, seed
        for index, block in enumerate(language.decoder.block):
            attention = block.layer[1].EncDecAttention
            for offset, name in enumerate(("q", "k")):
                setattr(
                    attention,
                    name,
                    AlignmentProjection(
                        getattr(attention, name), rank=rank, seed=seed + 2 * index + offset
                    ),
                )
        self.train(False)
        self.require_scope()

    def train(self, mode=True):
        # Keep the original numerical/dropout path fixed, including during fitting.
        super().train(False)
        return self

    def projections(self):
        return {
            name: module
            for name, module in self.language.named_modules()
            if isinstance(module, AlignmentProjection)
        }

    def trainable_parameters(self):
        return {
            name: parameter
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        }

    def frozen_state(self):
        """Map unchanged original values back to their pre-installation names."""
        original = {}
        for name, value in self.language.state_dict().items():
            if ".input_projection." in name or ".output_projection." in name:
                continue
            original[name.replace(".base.", ".")] = value
        return original

    def require_scope(self):
        projections = self.projections()
        expected = {
            "language." + name + "." + suffix
            for name in projections
            for suffix in ("input_projection.weight", "output_projection.weight")
        }
        selected = self.trainable_parameters()
        if len(projections) != 2 * len(self.language.decoder.block) or set(selected) != expected:
            raise ValueError(
                "alignment scope must contain only all decoder cross-attention q/k adapters"
            )
        if any(p.device.type != "cpu" or p.dtype != torch.float32 for p in self.parameters()):
            raise ValueError("alignment requires finite CPU FP32 values")
        if any(not torch.isfinite(p).all() for p in selected.values()):
            raise ValueError("alignment requires finite CPU FP32 adapter values")
        if any(module.training for module in self.modules()):
            raise ValueError("alignment original decoder must retain evaluation mode")
        if any(p.grad is not None for p in self.parameters() if not p.requires_grad):
            raise ValueError("alignment original decoder received gradients")

    def configuration(self):
        return {
            "schema": "umi-studio-decoder-cross-attention-alignment/1",
            "rank": self.rank,
            "seed": self.seed,
            "scaling": "one_over_sqrt_rank",
            "decoder_layers": len(self.language.decoder.block),
            "projections": {
                name: {
                    "input_width": module.base.in_features,
                    "output_width": module.base.out_features,
                }
                for name, module in self.projections().items()
            },
            "trainable_parameters": sum(p.numel() for p in self.trainable_parameters().values()),
            "trainable_scope": "low-rank residuals of all decoder cross-attention query/key maps",
            "original_value_self_attention_feedforward_embeddings_and_head": "frozen",
            "source_positions": "all original positions; original supplied mask",
            "initial_inference": "bitwise original decoder maps with zero output factors",
            "exact_linguistic_spans_supervised": False,
        }

    @contextlib.contextmanager
    def original_decoder(self):
        previous = {name: module.enabled for name, module in self.projections().items()}
        try:
            for module in self.projections().values():
                module.enabled = False
            yield
        finally:
            for name, module in self.projections().items():
                module.enabled = previous[name]

    def forward(self, memory, mask, labels):
        self.require_scope()
        if (
            memory.ndim != 3
            or memory.shape[0] != 1
            or memory.shape[1] < 1
            or memory.shape[2] != self.language.config.d_model
            or memory.requires_grad
            or memory.device.type != "cpu"
            or memory.dtype != torch.float32
            or not torch.isfinite(memory).all()
            or mask.shape != memory.shape[:2]
            or mask.dtype != torch.bool
            or not mask.all()
            or labels.ndim != 2
            or labels.shape[0] != 1
            or labels.shape[1] < 1
            or labels.dtype != torch.long
            or labels.device.type != "cpu"
        ):
            raise ValueError(
                "alignment requires complete detached original memory, full mask and target"
            )
        return self.language(
            encoder_outputs=BaseModelOutput(last_hidden_state=memory),
            attention_mask=mask.long(),
            labels=labels,
            use_cache=False,
            return_dict=True,
        )
