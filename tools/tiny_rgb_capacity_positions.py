"""Capacity-only vision with backend-identical nonlearned temporal positions."""

from __future__ import annotations

import torch
from tiny_rgb_language_model import ScratchRgbVision
from tiny_rgb_model import positions


class CanonicalPositionsVision(ScratchRgbVision):
    def __init__(self, config):
        super().__init__(config)
        self.register_buffer(
            "fixed_positions",
            positions(config.max_frames * config.spatial_queries, config.width, "cpu"),
            persistent=False,
        )

    def encode(self, video, frame_mask):
        # Same pixel stem, pooling, masks and attention as ScratchRgbVision; only
        # the nonlearned constants are computed once on CPU instead of per backend.
        if video.ndim != 5 or video.shape[2] != 3 or not video.is_floating_point():
            raise ValueError("capacity RGB input must be floating-point BTCHW pixels")
        batch, time, channels, height, width = video.shape
        if (
            frame_mask.shape != (batch, time)
            or frame_mask.dtype != torch.bool
            or not frame_mask.any(1).all()
            or time > self.config.max_frames
            or min(height, width) < 16
        ):
            raise ValueError("capacity RGB frame mask or input dimensions differ")
        chunks = []
        for images in video.reshape(batch * time, channels, height, width).split(
            self.config.frame_chunk
        ):
            features = self.stem(images)
            h, w = features.shape[-2:]
            y, x = torch.meshgrid(
                torch.linspace(-1, 1, h, device=video.device),
                torch.linspace(-1, 1, w, device=video.device),
                indexing="ij",
            )
            xy = torch.stack((x, y), -1).reshape(-1, 2)
            memory = self.visual_projection(features.flatten(2).transpose(1, 2)) + self.coordinates(
                xy
            )
            query = self.queries[None].expand(len(images), -1, -1)
            chunks.append(query + self.spatial_pool(query, self.spatial_norm(memory)))
        hidden = torch.cat(chunks).reshape(
            batch, time * self.config.spatial_queries, self.config.width
        )
        mask = frame_mask.repeat_interleave(self.config.spatial_queries, dim=1)
        hidden = hidden + self.fixed_positions[: hidden.shape[1]]
        for layer in self.encoder:
            hidden = layer(hidden, mask)
        return self.encoder_norm(hidden), mask
