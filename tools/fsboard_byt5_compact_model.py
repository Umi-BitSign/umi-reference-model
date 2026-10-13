"""ByT5 projection for the compact 83-point FSboard landmark experiment."""

from __future__ import annotations

import torch
from torch import nn
from transformers import T5ForConditionalGeneration


class FSboardByT5Compact(nn.Module):
    def __init__(self, language_model: T5ForConditionalGeneration) -> None:
        super().__init__()
        self.projection = nn.Linear(249, language_model.config.d_model)
        self.language_model = language_model

    def forward(
        self, features: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor
    ):
        if features.ndim != 3 or features.shape[-1] != 249:
            raise ValueError("compact FSboard feature tensor must have 249 columns")
        return self.language_model(
            inputs_embeds=self.projection(features), attention_mask=attention_mask, labels=labels
        )
