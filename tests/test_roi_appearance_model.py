from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from roi_appearance_model import STREAMS, RoiAppearance, feature_loss, pixel_tensor, temporal_gap


@pytest.fixture(scope="module")
def model():
    torch.set_num_threads(2)
    return RoiAppearance().eval()


def test_fresh_visual_capacity_grows_without_growing_input_surface(model):
    assert sum(p.numel() for p in model.parameters()) > 32_000_000
    assert model.hands.conv1.kernel_size == (3, 3)
    assert model.hands.conv1.stride == (1, 1)
    assert isinstance(model.hands.maxpool, nn.Identity)
    assert not any(isinstance(m, nn.BatchNorm2d | nn.ReLU) for m in model.modules())
    assert len(list(model.buffers())) == 0
    assert all(p.requires_grad for p in model.parameters())


def test_region_order_and_shared_hand_mapping_are_exact(model):
    pixels = torch.rand(1, 3, 3, 64, 64)
    with torch.no_grad():
        result = model(pixels)
        swapped = model(pixels[:, [0, 2, 1]])
    assert set(result) == set(STREAMS)
    assert all(value.shape == (1, 384) for value in result.values())
    torch.testing.assert_close(result["face_features"], swapped["face_features"])
    torch.testing.assert_close(result["left_hand_features"], swapped["right_hand_features"])
    torch.testing.assert_close(result["right_hand_features"], swapped["left_hand_features"])


def test_appearance_loss_updates_real_pixels_and_fresh_visual_weights_not_targets(model):
    model.zero_grad(set_to_none=True)
    pixels = torch.rand(1, 3, 3, 64, 64, requires_grad=True)
    target = {name: torch.rand(1, 384, requires_grad=True) for name in STREAMS}
    loss = feature_loss(model(pixels), target)["loss"]
    loss.backward()
    assert pixels.grad is not None and pixels.grad.norm() > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert all(t.grad is None for t in target.values())
    model.zero_grad(set_to_none=True)


def test_pixels_are_resized_before_the_network_not_in_a_hidden_resize_layer(model):
    source = np.full((1, 3, 64, 64, 3), 255, dtype=np.uint8)
    pixels = pixel_tensor(source)
    assert pixels.shape == (1, 3, 3, 64, 64)
    assert torch.equal(pixels, torch.ones_like(pixels))
    with pytest.raises(ValueError):
        pixel_tensor(np.zeros((1, 3, 128, 128, 3), dtype=np.uint8))
    with pytest.raises(ValueError):
        model(torch.zeros(1, 3, 3, 128, 128))


def test_temporal_control_distinguishes_aligned_features_from_a_constant_predictor():
    target = {name: torch.eye(4, 384) for name in STREAMS}
    assert temporal_gap(target, target) == pytest.approx(1)
    constant = {
        name: value.mean(0, keepdim=True).expand_as(value) for name, value in target.items()
    }
    assert temporal_gap(constant, target) == pytest.approx(0)


def test_zero_absent_target_is_not_used_as_a_label():
    target = {name: torch.zeros(1, 384) for name in STREAMS}
    with pytest.raises(ValueError, match="no usable"):
        feature_loss(target, target)


def test_input_validity_uses_metal_supported_flat_reduction(model, monkeypatch):
    actual = torch.isfinite
    dimensions = []

    def checked(value):
        dimensions.append(value.ndim)
        return actual(value)

    monkeypatch.setattr(torch, "isfinite", checked)
    pixels = torch.zeros(1, 3, 3, 64, 64)
    pixels[0, 2, 0, 63, 63] = float("nan")
    with pytest.raises(ValueError, match="finite real image"):
        model(pixels)
    assert dimensions == [1]
