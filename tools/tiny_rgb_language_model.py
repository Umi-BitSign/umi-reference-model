"""Randomly initialized compact RGB vision with a supplied pretrained language decoder."""

from __future__ import annotations

import torch
from fresh_lowres_model import discard_unused_text_encoder
from tiny_rgb_model import RmsNorm, TinyRgbConfig, TinyRgbTranslator
from torch import nn
from transformers.modeling_outputs import BaseModelOutput


class ScratchRgbVision(TinyRgbTranslator):
    """Reuse only the scratch prototype's image and temporal layers."""

    def __init__(self, config: TinyRgbConfig | None = None):
        super().__init__(config)
        del self.embedding
        del self.decoder
        del self.decoder_norm

    def forward(self, video, frame_mask):
        return self.encode(video, frame_mask)


class TinyRgbLanguageTranslator(nn.Module):
    def __init__(self, language_model: nn.Module, visual_config: TinyRgbConfig | None = None):
        super().__init__()
        self.vision = ScratchRgbVision(visual_config)
        width = language_model.config.d_model
        self.visual_projection = nn.Linear(self.vision.config.width, width)
        nn.init.normal_(self.visual_projection.weight, std=0.02)
        nn.init.zeros_(self.visual_projection.bias)
        self.memory_norm = RmsNorm(width)
        original_language_parameters = sum(p.numel() for p in language_model.parameters())
        discard_unused_text_encoder(language_model)
        # Shared embeddings survive encoder removal; count unique parameters actually released.
        self.removed_language_encoder_parameters = original_language_parameters - sum(
            p.numel() for p in language_model.parameters()
        )
        # Never apply a recursive initializer after attaching the pretrained language component.
        self.language_model = language_model

    def encode(self, video, frame_mask):
        memory, mask = self.vision(video, frame_mask)
        return self.memory_norm(self.visual_projection(memory)), mask

    def forward(self, video, frame_mask, labels):
        memory, mask = self.encode(video, frame_mask)
        return self.language_model(
            encoder_outputs=BaseModelOutput(last_hidden_state=memory),
            attention_mask=mask.long(),
            labels=labels,
            use_cache=False,
            return_dict=True,
        )

    @torch.no_grad()
    def generate(self, video, frame_mask, **kwargs):
        memory, mask = self.encode(video, frame_mask)
        return self.language_model.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=memory),
            attention_mask=mask.long(),
            **kwargs,
        )

    def parameter_counts(self) -> dict[str, int]:
        return {
            "scratch_visual_encoder": sum(p.numel() for p in self.vision.parameters()),
            "scratch_visual_language_bridge": sum(
                p.numel() for p in self.visual_projection.parameters()
            )
            + sum(p.numel() for p in self.memory_norm.parameters()),
            "pretrained_language_decoder_and_shared_embeddings": sum(
                p.numel() for p in self.language_model.parameters()
            ),
            "total": sum(p.numel() for p in self.parameters()),
            "removed_unused_text_encoder": self.removed_language_encoder_parameters,
        }
