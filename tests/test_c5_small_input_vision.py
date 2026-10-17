from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from c5_small_input_vision import SmallInputDino, checked_side, input_tensor


class Backbone(nn.Module):
    """A tiny projection fixture, not a real DINO quality/numerical qualification."""

    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Module()
        self.patch_embed.patch_size = (14, 14)
        self.patch_embed.proj = nn.Conv2d(3, 384, 14, stride=14)
        self.mask_token = nn.Parameter(torch.zeros(1, 384))
        self.received = []

    def forward(self, pixels):
        self.received.append(tuple(pixels.shape))
        return self.patch_embed.proj(pixels).mean(dim=(2, 3))


@pytest.mark.parametrize("side", [28, 56, 70, 84, 112, 224])
def test_valid_native_sides_preserve_real14px_patch_geometry(side):
    assert checked_side(side) == side


@pytest.mark.parametrize("side", [True, 0, 14, 64, 111, 225, 112.0])
def test_invalid_native_sides_cannot_silently_pad_crop_or_upscale(side):
    with pytest.raises(ValueError):
        checked_side(side)


def test_RGB_normalization_preserves_color_order_and_already_small_surface():
    pixels = np.zeros((2, 3, 112, 112, 3), dtype=np.uint8)
    pixels[..., 0] = 255
    values = input_tensor(pixels)
    assert values.shape == (2, 3, 3, 112, 112)
    assert values.dtype == torch.float32
    assert torch.allclose(
        values[0, 0, :, 0, 0], torch.tensor([(1 - 0.485) / 0.229, -0.456 / 0.224, -0.406 / 0.225])
    )
    assert np.all(pixels[..., 0] == 255)


def test_two_backbones_share_hands_keep_feature_contract_and_pass_only_real_small_pixels():
    hands, face = Backbone(), Backbone()
    candidate = SmallInputDino(hands, face)
    result = candidate(input_tensor(np.full((2, 3, 112, 112, 3), 160, dtype=np.uint8)))
    assert hands.received == [(4, 3, 112, 112)]
    assert face.received == [(2, 3, 112, 112)]
    assert all(value.shape == (2, 384) for value in result.values())
    sum(value.square().sum() for value in result.values()).backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in candidate.trainable_parameters().values()
    )
    assert hands.mask_token.grad is None and face.mask_token.grad is None
    assert not hands.mask_token.requires_grad and not face.mask_token.requires_grad
    assert not any(name.endswith("mask_token") for name in candidate.trainable_parameters())


def test_actual_first_projection_rejects_hidden_resize_inside_a_backbone():
    class ResizingBackbone(Backbone):
        def forward(self, pixels):
            return super().forward(nn.functional.interpolate(pixels, size=(224, 224)))

    model = SmallInputDino(ResizingBackbone(), Backbone())
    with pytest.raises(ValueError, match="hidden"):
        model(input_tensor(np.zeros((1, 3, 112, 112, 3), dtype=np.uint8)))


@pytest.mark.parametrize("mutation", ["shape", "dtype", "nonfinite"])
def test_forward_rejects_wrong_surface_dtype_and_nonfinite_pixels(mutation):
    model = SmallInputDino(Backbone(), Backbone())
    value = torch.zeros(1, 3, 3, 112, 112)
    if mutation == "shape":
        value = value[..., :64, :64]
    elif mutation == "dtype":
        value = value.to(torch.uint8)
    else:
        value[0, 0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        model(value)


def test_source_crops_must_already_have_the_declared_native_size():
    with pytest.raises(ValueError, match="native"):
        input_tensor(np.zeros((1, 3, 64, 64, 3), dtype=np.uint8))


def test_backbone_contract_cannot_change_feature_width_or_patch_surface():
    model = Backbone()
    model.patch_embed.patch_size = (16, 16)
    with pytest.raises(ValueError, match="contract"):
        SmallInputDino(model, Backbone())


def test_hand_and_face_cannot_alias_the_same_backbone():
    model = Backbone()
    with pytest.raises(ValueError, match="separate"):
        SmallInputDino(model, model)
