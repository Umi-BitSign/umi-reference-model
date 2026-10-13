"""Landmark-to-ByT5 model for an independently trained FSboard specialist."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from transformers import T5ForConditionalGeneration


def normalized_features(
    landmarks: np.ndarray, *, frame_stride: int = 2, max_frames: int = 256
) -> np.ndarray:
    """Standardize observed coordinates and retain missing points as zeros."""
    if (
        landmarks.ndim != 3
        or landmarks.shape[1:] != (543, 3)
        or landmarks.dtype != np.float32
        or frame_stride < 1
        or max_frames < 1
        or np.isinf(landmarks).any()
    ):
        raise ValueError("unexpected FSboard landmarks or sampling parameters")
    sampled = landmarks[::frame_stride]
    sampled = sampled[~np.isnan(sampled).all(axis=(1, 2))]
    if not 0 < len(sampled) <= max_frames:
        raise ValueError("FSboard clip has no usable frames or exceeds the frame limit")
    normalized = np.empty_like(sampled)
    for axis in range(3):
        values = sampled[:, :, axis]
        observed = values[np.isfinite(values)]
        center = float(observed.mean()) if observed.size else (0.0 if axis == 2 else 0.5)
        spread = float(observed.std()) if observed.size else 0.0
        if not np.isfinite(spread) or spread < 1e-6:
            spread = 1.0
        normalized[:, :, axis] = np.where(np.isfinite(values), (values - center) / spread, 0)
    return np.ascontiguousarray(normalized.reshape(len(sampled), 1629))


class FSboardByT5(nn.Module):
    def __init__(self, language_model: T5ForConditionalGeneration) -> None:
        super().__init__()
        self.projection = nn.Linear(1629, language_model.config.d_model)
        self.language_model = language_model

    def forward(
        self, features: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor
    ):
        if features.ndim != 3 or features.shape[-1] != 1629:
            raise ValueError("FSboard feature tensor must have 1,629 columns")
        embeddings = self.projection(features)
        return self.language_model(
            inputs_embeds=embeddings, attention_mask=attention_mask, labels=labels
        )
