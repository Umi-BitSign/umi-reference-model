"""Differentiate native112px vision through unchanged C5 sequence/language weights."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from c5_small_input_vision import STREAMS, input_tensor
from tiny_rgb_feature_winner_student import RawFeatureWinnerStudent
from torch import nn
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class Native112Clip:
    pixels: torch.Tensor
    pose: torch.Tensor

    def to(self, device):
        return Native112Clip(self.pixels.to(device), self.pose.to(device))


def clip_input(values):
    return Native112Clip(
        input_tensor(values["pixels"], side=112),
        torch.from_numpy(values["pose_features"].copy()),
    )


class Native112SemanticStudent(RawFeatureWinnerStudent):
    """Retain the caller's visual gradient scope, including frozen unused mask tokens.

    Both fixed C5 components retain eval mode. Complete pose and pixel timelines
    stay aligned through differentiable visual chunks. This interface requires
    separate real-model/full-risk numerical qualification before optimization.
    """

    def __init__(self, vision, semantic_encoder, language, *, chunk_frames=2):
        nn.Module.__init__(self)
        if type(chunk_frames) is not int or chunk_frames < 1 or vision.side != 112:
            raise ValueError("native112 learning requires112px vision and positive chunk size")
        masks = []
        for name in ("hands", "face"):
            token = getattr(getattr(vision, name, None), "mask_token", None)
            if not isinstance(token, nn.Parameter) or token.requires_grad:
                raise ValueError("native112 learning must retain frozen unused mask tokens")
            masks.append(token)
        if any(
            not parameter.requires_grad
            for parameter in vision.parameters()
            if not any(parameter is token for token in masks)
        ):
            raise ValueError("native112 learning cannot silently freeze functional visual tensors")
        # The64px parent's global requires_grad_(True) would unfreeze both masks.
        self.vision = vision
        self.semantic_encoder = semantic_encoder.requires_grad_(False).eval()
        self.language = language.requires_grad_(False).eval()
        if hasattr(language, "frozen_winner_weights"):
            language.frozen_winner_weights = True
        self.chunk_frames = chunk_frames

    def predict_features(self, video, frame_mask):
        if (
            not isinstance(video, Native112Clip)
            or video.pixels.ndim != 5
            or video.pixels.shape[1:] != (3, 3, 112, 112)
            or video.pixels.dtype != torch.float32
            or video.pose.shape != (len(video.pixels), 14)
            or video.pose.dtype != torch.float32
            or video.pose.requires_grad
            or video.pose.device != video.pixels.device
            or frame_mask.shape != (1, len(video.pixels))
            or frame_mask.dtype != torch.bool
            or frame_mask.device != video.pixels.device
            or len(video.pixels) < 1
            or not frame_mask.all()
            or not torch.isfinite(video.pose).all()
            or not torch.isfinite(video.pixels.reshape(-1)).all()
        ):
            raise ValueError("native112 learning requires complete finite pixels and fixed14D pose")
        parts = {name: [] for name in STREAMS}
        for start in range(0, len(video.pixels), self.chunk_frames):
            pixels = video.pixels[start : start + self.chunk_frames]
            features = (
                checkpoint(self.vision, pixels, use_reentrant=False)
                if torch.is_grad_enabled()
                else self.vision(pixels)
            )
            if set(features) != set(STREAMS) or any(
                value.shape != (len(pixels), 384) or not torch.isfinite(value).all()
                for value in features.values()
            ):
                raise ValueError("native112 visual chunks changed feature width or frame coverage")
            for name in STREAMS:
                parts[name].append(features[name])
        return {
            **{name: torch.cat(parts[name], dim=0)[None] for name in STREAMS},
            "pose_features": video.pose[None],
        }, frame_mask
