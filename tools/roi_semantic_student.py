"""Differentiate actual translation risk through64px vision and fixed C5 backbones."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from roi_appearance_model import STREAMS, pixel_tensor
from tiny_rgb_feature_winner_student import RawFeatureWinnerStudent
from torch import nn
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class RoiClip:
    pixels: torch.Tensor
    pose: torch.Tensor

    def to(self, device):
        return RoiClip(self.pixels.to(device), self.pose.to(device))


def clip_input(values):
    return RoiClip(pixel_tensor(values["pixels"]), torch.from_numpy(values["pose_features"].copy()))


class RoiSemanticStudent(RawFeatureWinnerStudent):
    """Only the existing ROI vision is trainable; pose is always video-derived.

    Reuse the qualified native sequence adapter and decoder interface. Chunk
    checkpointing recomputes visual activations instead of detaching features.
    The complete clip enters the sequence encoder, with no temporal truncation.
    """

    def __init__(self, vision, semantic_encoder, language, *, chunk_frames=8):
        nn.Module.__init__(self)
        if type(chunk_frames) is not int or chunk_frames < 1:
            raise ValueError("ROI semantic chunks require positive real frame counts")
        self.vision = vision.requires_grad_(True)
        self.semantic_encoder = semantic_encoder.requires_grad_(False).eval()
        self.language = language.requires_grad_(False).eval()
        if hasattr(language, "frozen_winner_weights"):
            language.frozen_winner_weights = True
        self.chunk_frames = chunk_frames

    def predict_features(self, video, frame_mask):
        if (
            not isinstance(video, RoiClip)
            or video.pixels.ndim != 5
            or video.pixels.shape[1:] != (3, 3, 64, 64)
            or video.pixels.dtype != torch.float32
            or video.pose.shape != (len(video.pixels), 14)
            or video.pose.dtype != torch.float32
            or video.pose.requires_grad
            or frame_mask.shape != (1, len(video.pixels))
            or frame_mask.dtype != torch.bool
            or len(video.pixels) < 1
            or not frame_mask.all()
            or not torch.isfinite(video.pose).all()
        ):
            raise ValueError("ROI learning requires one complete real clip and fixed14D pose")
        parts = {name: [] for name in STREAMS}
        for start in range(0, len(video.pixels), self.chunk_frames):
            pixels = video.pixels[start : start + self.chunk_frames]
            features = (
                checkpoint(self.vision, pixels, use_reentrant=False)
                if torch.is_grad_enabled()
                else self.vision(pixels)
            )
            for name in STREAMS:
                parts[name].append(features[name])
        return {
            **{name: torch.cat(parts[name], dim=0)[None] for name in STREAMS},
            "pose_features": video.pose[None],
        }, frame_mask


def visual_gradients(model, *, expected_tensors=116):
    values = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            if parameter.grad is not None:
                raise RuntimeError("ROI risk sent gradients into a frozen C5 parameter")
            continue
        if (
            not name.startswith("vision.")
            or parameter.grad is None
            or not torch.isfinite(parameter.grad).all()
            or float(parameter.grad.norm()) <= 0
        ):
            raise RuntimeError(f"ROI lacks finite nonzero semantic gradient: {name}")
        values[name] = parameter.grad.detach().cpu().clone()
    if len(values) != expected_tensors:
        raise RuntimeError("ROI semantic optimizer topology changed")
    return values
