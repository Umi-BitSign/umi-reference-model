"""Custom small-pixel appearance with the winner's full frozen sequence backbone."""

from __future__ import annotations

import torch
from tiny_rgb_capacity import visual_config
from tiny_rgb_capacity_activations import replace_capacity_gelu
from tiny_rgb_capacity_positions import CanonicalPositionsVision
from tiny_rgb_language_model import ScratchRgbVision
from tiny_rgb_visual_teacher import SpatialTeacherHeads
from torch import nn
from torch.nn import functional as F
from transformers.modeling_outputs import BaseModelOutput


def build_front_end(capacity="large", seed=317):
    torch.manual_seed(seed)
    base = ScratchRgbVision(visual_config(capacity))
    vision = CanonicalPositionsVision(base.config)
    vision.load_state_dict(base.state_dict(), strict=True)
    replace_capacity_gelu(vision)
    return vision, SpatialTeacherHeads(vision.config.width)


class RawFeatureWinnerStudent(nn.Module):
    """Predict native streams, preserving SignHuBERT and all winner T5 blocks.

    The 503M encoder is a learned sequence adapter plus eighteen T5 blocks,
    rather than the DINO image front-end. Both encoder and decoder stay fixed.
    """

    def __init__(self, semantic_encoder, language, *, capacity="large"):
        super().__init__()
        self.vision, self.feature_heads = build_front_end(capacity)
        self.semantic_encoder = semantic_encoder.requires_grad_(False).eval()
        self.language = language.requires_grad_(False).eval()
        if hasattr(language, "frozen_winner_weights"):
            language.frozen_winner_weights = True
        self.capacity = capacity

    def train(self, mode=True):
        super().train(mode)
        self.semantic_encoder.eval()
        self.language.eval()
        return self

    def trainable_parameters(self):
        return {name: value for name, value in self.named_parameters() if value.requires_grad}

    def predict_features(self, video, frame_mask):
        hidden, token_mask = self.vision(video, frame_mask)
        features, mask = self.feature_heads(hidden, token_mask)
        if not torch.equal(mask, frame_mask):
            raise ValueError("feature winner student cannot change real frame coverage")
        return features, mask

    def semantic_memory(self, features, frame_mask):
        if frame_mask.ndim != 2 or frame_mask.shape[0] != 1 or frame_mask.dtype != torch.bool:
            raise ValueError("native feature winner currently requires one prefix-masked clip")
        frames = int(frame_mask.sum())
        if frames < 1 or not frame_mask[:, :frames].all() or frame_mask[:, frames:].any():
            raise ValueError("native feature winner cannot invent/drop interior frames")
        # SignHuBERT's native adapter ignores padding masks. Remove only trailing
        # padding before it, then restore masked decoder positions differentiably.
        memory = self.semantic_encoder(
            **{name: value[:, :frames] for name, value in features.items()},
            return_dict=True,
        ).last_hidden_state
        if memory.shape[:2] != (1, frames) or memory.shape[-1] != self.language.config.d_model:
            raise ValueError("native winner semantic timeline/width differs")
        return F.pad(memory, (0, 0, 0, frame_mask.shape[1] - frames))

    def encode(self, video, frame_mask):
        features, mask = self.predict_features(video, frame_mask)
        return self.semantic_memory(features, mask), mask

    def forward(self, video, frame_mask, labels):
        features, mask = self.predict_features(video, frame_mask)
        memory = self.semantic_memory(features, mask)
        output = self.language(
            encoder_outputs=BaseModelOutput(last_hidden_state=memory),
            attention_mask=mask.long(),
            labels=labels,
            use_cache=False,
            return_dict=True,
        )
        return {"language": output, "memory": memory, "frame_mask": mask, "features": features}

    @torch.no_grad()
    def generate(self, video, frame_mask, **kwargs):
        memory, mask = self.encode(video, frame_mask)
        return self.language.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=memory),
            attention_mask=mask.long(),
            **kwargs,
        )
