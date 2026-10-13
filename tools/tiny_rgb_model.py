"""A small raw-RGB translator whose visual and language weights all start from scratch."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

PAD, BOS, EOS, VOCABULARY = 0, 1, 2, 259


def encode_text(text: str) -> list[int]:
    return [value + 3 for value in text.encode("utf-8")] + [EOS]


def decode_text(tokens: list[int]) -> str:
    values = []
    for token in tokens:
        if token == EOS:
            break
        if 3 <= token < VOCABULARY:
            values.append(token - 3)
    return bytes(values).decode("utf-8", errors="replace")


def resize_rgb_clip(frames: np.ndarray, short_side: int) -> torch.Tensor:
    """Downscale uint8 THWC RGB on CPU before any learned operation; never force a square."""
    import cv2

    if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8:
        raise ValueError("raw RGB clip must be a uint8 THWC array")
    if min(frames.shape[:3]) < 1 or short_side < 32:
        raise ValueError("raw RGB dimensions and short-side cap must be positive")
    height, width = frames.shape[1:3]
    scale = min(1.0, short_side / min(height, width))
    output = max(1, round(width * scale)), max(1, round(height * scale))
    resized = np.stack(
        [cv2.resize(frame, output, interpolation=cv2.INTER_AREA) for frame in frames]
    )
    return torch.from_numpy(resized).permute(0, 3, 1, 2).float().div_(127.5).sub_(1.0)


@dataclass(frozen=True)
class TinyRgbConfig:
    width: int = 192
    heads: int = 4
    encoder_layers: int = 2
    decoder_layers: int = 3
    spatial_queries: int = 4
    max_frames: int = 512
    max_text: int = 512
    dropout: float = 0.1
    frame_chunk: int = 16


class RmsNorm(nn.Module):
    """Explicit float32 RMS arithmetic, shared by spatial, temporal and language blocks."""

    def __init__(self, width: int, channel_first: bool = False):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(width))
        self.channel_first = channel_first

    def forward(self, value):
        axis = 1 if self.channel_first else -1
        normalized = value * torch.rsqrt(value.float().square().mean(axis, keepdim=True) + 1e-4)
        scale = self.scale[None, :, None, None] if self.channel_first else self.scale
        return normalized * scale


class Attention(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float):
        super().__init__()
        self.heads = heads
        self.q, self.k, self.v, self.out = [nn.Linear(width, width, bias=False) for _ in range(4)]
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, memory, memory_mask=None, causal=False):
        batch, time, width = query.shape

        def split(value):
            return value.reshape(batch, -1, self.heads, width // self.heads).transpose(1, 2)

        q, k, v = split(self.q(query)), split(self.k(memory)), split(self.v(memory))
        scores = q @ k.transpose(-1, -2) / math.sqrt(width // self.heads)
        if memory_mask is not None:
            scores = scores.masked_fill(~memory_mask[:, None, None, :], -1e4)
        if causal:
            allowed = torch.ones(
                time, memory.shape[1], dtype=torch.bool, device=query.device
            ).tril()
            scores = scores.masked_fill(~allowed, -1e4)
        attended = self.dropout(scores.softmax(-1)) @ v
        return self.out(attended.transpose(1, 2).reshape(batch, time, width))


class Block(nn.Module):
    def __init__(self, config: TinyRgbConfig, decoder: bool = False):
        super().__init__()
        width = config.width
        self.self_norm, self.ff_norm = RmsNorm(width), RmsNorm(width)
        self.attention = Attention(width, config.heads, config.dropout)
        self.cross = Attention(width, config.heads, config.dropout) if decoder else None
        self.cross_norm = RmsNorm(width) if decoder else None
        self.ff = nn.Sequential(
            nn.Linear(width, width * 3),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(width * 3, width),
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden, mask, memory=None, memory_mask=None):
        normalized = self.self_norm(hidden)
        hidden = hidden + self.dropout(
            self.attention(normalized, normalized, mask, causal=self.cross is not None)
        )
        if self.cross is not None:
            hidden = hidden + self.dropout(self.cross(self.cross_norm(hidden), memory, memory_mask))
        return hidden + self.dropout(self.ff(self.ff_norm(hidden)))


def positions(length: int, width: int, device) -> torch.Tensor:
    rates = torch.exp(torch.arange(0, width, 2, device=device).float() * (-math.log(10000) / width))
    phase = torch.arange(length, device=device).float()[:, None] * rates
    return torch.stack((phase.sin(), phase.cos()), dim=-1).flatten(1)


class TinyRgbTranslator(nn.Module):
    def __init__(self, config: TinyRgbConfig | None = None):
        super().__init__()
        config = config or TinyRgbConfig()
        if (
            min(
                config.width,
                config.heads,
                config.encoder_layers,
                config.decoder_layers,
                config.spatial_queries,
                config.max_frames,
                config.max_text,
                config.frame_chunk,
            )
            < 1
            or config.width % config.heads
            or config.width % 2
            or not 0 <= config.dropout < 1
        ):
            raise ValueError("invalid tiny RGB architecture")
        self.config = config
        layers = []
        incoming = 3
        for channels in (16, 32, 64, 96):
            layers.extend(
                [
                    nn.Conv2d(incoming, channels, 3, stride=2, padding=1),
                    RmsNorm(channels, channel_first=True),
                    nn.GELU(),
                ]
            )
            incoming = channels
        self.stem = nn.Sequential(*layers)
        self.visual_projection = nn.Linear(incoming, config.width)
        self.coordinates = nn.Linear(2, config.width, bias=False)
        self.queries = nn.Parameter(torch.randn(config.spatial_queries, config.width) * 0.02)
        self.spatial_norm = RmsNorm(config.width)
        self.spatial_pool = Attention(config.width, config.heads, config.dropout)
        self.encoder = nn.ModuleList([Block(config) for _ in range(config.encoder_layers)])
        self.encoder_norm = RmsNorm(config.width)
        self.embedding = nn.Embedding(VOCABULARY, config.width, padding_idx=PAD)
        self.decoder = nn.ModuleList(
            [Block(config, decoder=True) for _ in range(config.decoder_layers)]
        )
        self.decoder_norm = RmsNorm(config.width)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        if isinstance(module, nn.Linear | nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
            if isinstance(module, nn.Embedding) and module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()

    def encode(self, video, frame_mask):
        if video.ndim != 5 or video.shape[2] != 3 or not video.is_floating_point():
            raise ValueError("tiny RGB input must be floating-point BTCHW pixels")
        batch, time, channels, height, width = video.shape
        if (
            frame_mask.shape != (batch, time)
            or frame_mask.dtype != torch.bool
            or not frame_mask.any(1).all()
            or time > self.config.max_frames
            or min(height, width) < 16
        ):
            raise ValueError("tiny RGB frame mask or input dimensions differ")
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
        hidden = hidden + positions(hidden.shape[1], self.config.width, video.device)
        for layer in self.encoder:
            hidden = layer(hidden, mask)
        return self.encoder_norm(hidden), mask

    def decode(self, input_ids, memory, memory_mask):
        if input_ids.ndim != 2 or input_ids.shape[1] > self.config.max_text:
            raise ValueError("tiny RGB text sequence exceeds its declared limit")
        hidden = self.embedding(input_ids) + positions(
            input_ids.shape[1], self.config.width, input_ids.device
        )
        for layer in self.decoder:
            hidden = layer(hidden, input_ids.ne(PAD), memory, memory_mask)
        return F.linear(self.decoder_norm(hidden), self.embedding.weight)

    def forward(self, video, frame_mask, targets):
        if targets.ndim != 2 or targets.shape[0] != video.shape[0] or targets.shape[1] < 1:
            raise ValueError("tiny RGB targets must be a nonempty batch of byte-token sequences")
        memory, mask = self.encode(video, frame_mask)
        shifted = torch.full_like(targets, PAD)
        shifted[:, 0] = BOS
        shifted[:, 1:] = targets[:, :-1]
        return self.decode(shifted, memory, mask)

    @torch.no_grad()
    def generate(self, video, frame_mask, maximum_tokens: int = 384):
        if not 1 <= maximum_tokens < self.config.max_text:
            raise ValueError("tiny RGB generation limit exceeds its declared context")
        memory, mask = self.encode(video, frame_mask)
        tokens = torch.full((len(video), 1), BOS, device=video.device, dtype=torch.long)
        done = torch.zeros(len(video), device=video.device, dtype=torch.bool)
        for _ in range(maximum_tokens):
            next_token = self.decode(tokens, memory, mask)[:, -1].argmax(-1)
            next_token = torch.where(done, EOS, next_token)
            tokens = torch.cat((tokens, next_token[:, None]), dim=1)
            done |= next_token.eq(EOS)
            if done.all():
                break
        return tokens[:, 1:]
