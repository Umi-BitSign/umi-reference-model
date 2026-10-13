"""Fresh translator over generic face, hand, and pose feature streams."""

from __future__ import annotations

import math

import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutput

STREAM_WIDTHS = (384, 384, 384, 14)


class FreshStreamTranslator(nn.Module):
    """Project generic visual streams, model time, and decode with generic T5."""

    def __init__(
        self,
        language_model: nn.Module,
        *,
        temporal_width: int = 256,
        temporal_layers: int = 3,
        temporal_heads: int = 8,
        feedforward_width: int = 1024,
        max_frames: int = 256,
        dropout: float = 0.1,
        use_language_encoder: bool = True,
    ) -> None:
        super().__init__()
        if temporal_width % temporal_heads or max_frames < 1:
            raise ValueError("invalid fresh stream topology")
        projected_widths = (96, 96, 96, 32)
        # Generic hand streams can be exactly constant when no hand is detected.
        # Affine LayerNorm backward is numerically unstable for those rows on MPS;
        # fixed normalization is sufficient because the following projections learn.
        self.stream_norms = nn.ModuleList(
            nn.LayerNorm(width, elementwise_affine=False) for width in STREAM_WIDTHS
        )
        self.stream_projections = nn.ModuleList(
            nn.Linear(source, target)
            for source, target in zip(STREAM_WIDTHS, projected_widths, strict=True)
        )
        self.frame_projection = nn.Sequential(
            nn.GELU(), nn.Linear(sum(projected_widths), temporal_width)
        )
        self.temporal_positions = nn.Parameter(torch.empty(max_frames, temporal_width))
        layer = nn.TransformerEncoderLayer(
            d_model=temporal_width,
            nhead=temporal_heads,
            dim_feedforward=feedforward_width,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            layer, num_layers=temporal_layers, norm=nn.LayerNorm(temporal_width)
        )
        self.language_projection = nn.Linear(
            temporal_width, language_model.config.d_model, bias=False
        )
        self.language_model = language_model
        self.max_frames = max_frames
        self.use_language_encoder = use_language_encoder
        nn.init.normal_(self.temporal_positions, mean=0.0, std=temporal_width**-0.5)

    def encode_visual(
        self, streams: torch.Tensor, frame_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if streams.ndim != 3 or streams.shape[-1] != sum(STREAM_WIDTHS):
            raise ValueError("streams must have shape batch x time x 1166")
        if frame_mask.shape != streams.shape[:2] or frame_mask.dtype != torch.bool:
            raise ValueError("frame mask must be boolean with shape batch x time")
        if streams.shape[1] > self.max_frames or not torch.all(frame_mask.any(dim=1)):
            raise ValueError("clip is empty or exceeds the frame limit")
        pieces = torch.split(streams, STREAM_WIDTHS, dim=-1)
        hidden = torch.cat(
            [projection(norm(piece)) for piece, norm, projection in zip(
                pieces, self.stream_norms, self.stream_projections, strict=True
            )],
            dim=-1,
        )
        hidden = self.frame_projection(hidden)
        hidden = hidden * math.sqrt(hidden.shape[-1])
        hidden = hidden + self.temporal_positions[: hidden.shape[1]]
        hidden = self.temporal_encoder(hidden, src_key_padding_mask=~frame_mask)
        return hidden, frame_mask

    def encode(
        self, streams: torch.Tensor, frame_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden, frame_mask = self.encode_visual(streams, frame_mask)
        hidden = self.language_projection(hidden)
        if self.use_language_encoder:
            hidden = self.language_model.encoder(
                inputs_embeds=hidden,
                attention_mask=frame_mask,
                return_dict=True,
            ).last_hidden_state
        return hidden, frame_mask

    def forward(self, streams: torch.Tensor, frame_mask: torch.Tensor, labels: torch.Tensor):
        memory, memory_mask = self.encode(streams, frame_mask)
        return self.language_model(
            encoder_outputs=BaseModelOutput(last_hidden_state=memory),
            attention_mask=memory_mask,
            labels=labels,
            use_cache=False,
            return_dict=True,
        )

    @torch.inference_mode()
    def generate(self, streams: torch.Tensor, frame_mask: torch.Tensor, **options):
        memory, memory_mask = self.encode(streams, frame_mask)
        return self.language_model.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=memory),
            attention_mask=memory_mask,
            **options,
        )


def configure_decoder_training(model: FreshStreamTranslator) -> dict[str, nn.Parameter]:
    """Train every new visual layer and the generic T5 decoder; freeze its encoder."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    modules = (
        model.stream_norms,
        model.stream_projections,
        model.frame_projection,
        model.temporal_encoder,
        model.language_projection,
        model.language_model.decoder,
        model.language_model.lm_head,
    )
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    model.temporal_positions.requires_grad_(True)
    trainable = {
        name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    if (
        not any(name.startswith("stream_projections.") for name in trainable)
        or not any(name.startswith("temporal_encoder.") for name in trainable)
        or not any(name.startswith("language_model.decoder.") for name in trainable)
        or any(name.startswith("language_model.encoder.") for name in trainable)
    ):
        raise ValueError("fresh stream trainable scope is invalid")
    return trainable


def configure_visual_pretraining(model: FreshStreamTranslator) -> dict[str, nn.Parameter]:
    """Train only the compact visual-temporal student layers."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in (model.stream_projections, model.frame_projection, model.temporal_encoder):
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    model.temporal_positions.requires_grad_(True)
    trainable = {
        name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    if not trainable or any(name.startswith("language_model.") for name in trainable):
        raise ValueError("fresh stream visual pretraining scope is invalid")
    return trainable


def configure_alignment_training(model: FreshStreamTranslator) -> dict[str, nn.Parameter]:
    """Train the new visual path and cross-attention without changing T5 language layers."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in (
        model.stream_projections,
        model.frame_projection,
        model.temporal_encoder,
        model.language_projection,
    ):
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    model.temporal_positions.requires_grad_(True)
    for block in model.language_model.decoder.block:
        attention = block.layer[1].EncDecAttention
        for projection in (attention.q, attention.k, attention.v, attention.o):
            for parameter in projection.parameters():
                parameter.requires_grad_(True)
    trainable = {
        name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    if (
        not any(name.startswith("stream_projections.") for name in trainable)
        or not any("EncDecAttention" in name for name in trainable)
        or any("SelfAttention" in name for name in trainable)
        or any("DenseReluDense" in name for name in trainable)
        or any(name.startswith("language_model.shared.") for name in trainable)
    ):
        raise ValueError("fresh stream alignment scope is invalid")
    return trainable

