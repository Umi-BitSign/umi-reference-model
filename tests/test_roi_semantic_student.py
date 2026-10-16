from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from roi_appearance_model import STREAMS
from roi_semantic_student import RoiClip, RoiSemanticStudent, clip_input, visual_gradients
from terra_quality_training import minimum_risk, sequence_log_probability
from terra_visual_learning import backward_candidate_risk


class Vision(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(9, 1152)

    def forward(self, pixels):
        features = self.projection(pixels.mean(dim=(-1, -2)).reshape(len(pixels), 9))
        return dict(zip(STREAMS, features.chunk(3, dim=-1), strict=True))


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(1166, 8)

    def forward(self, return_dict, **features):
        return SimpleNamespace(
            last_hidden_state=self.projection(
                torch.cat([features[name] for name in (*STREAMS, "pose_features")], dim=-1)
            )
        )


class Language(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(d_model=8)
        self.output = nn.Linear(8, 16)

    def forward(self, *, encoder_outputs, labels, **kwargs):
        pooled = encoder_outputs.last_hidden_state.mean(1, keepdim=True)
        logits = self.output(pooled).expand(1, labels.shape[1], 16)
        return SimpleNamespace(
            logits=logits, loss=F.cross_entropy(logits.reshape(-1, 16), labels.reshape(-1))
        )


def student():
    torch.manual_seed(100)
    return RoiSemanticStudent(Vision(), Encoder(), Language(), chunk_frames=2).eval()


def clip(frames=5):
    return RoiClip(torch.randn(frames, 3, 3, 64, 64), torch.randn(frames, 14))


def test_only_vision_is_trainable_even_when_parent_is_training():
    model = student().train()
    assert set(model.trainable_parameters()) == {
        "vision.projection.weight",
        "vision.projection.bias",
    }
    assert model.semantic_encoder.training is False
    assert model.language.training is False


def test_complete_timeline_and_video_pose_survive_differentiable_chunks():
    model, video = student(), clip()
    features, mask = model.predict_features(video, torch.ones(1, 5, dtype=torch.bool))
    assert all(features[n].shape == (1, 5, 384) for n in STREAMS)
    assert torch.equal(features["pose_features"][0], video.pose)
    assert mask.sum() == 5
    sum(features[n].square().mean() for n in STREAMS).backward()
    assert len(visual_gradients(model, expected_tensors=2)) == 2


@pytest.mark.parametrize(
    "kind", ("interior_mask", "trailing_mask", "pose_gradient", "pose_nan", "wrong_pixels")
)
def test_incomplete_or_invalid_inputs_are_rejected(kind):
    model, video = student(), clip()
    mask = torch.ones(1, 5, dtype=torch.bool)
    if kind == "interior_mask":
        mask[0, 1] = False
    elif kind == "trailing_mask":
        mask[0, -1] = False
    elif kind == "pose_gradient":
        video.pose.requires_grad_(True)
    elif kind == "pose_nan":
        video.pose[0, 0] = float("nan")
    else:
        video = RoiClip(video.pixels[:, :, :, :32], video.pose)
    with pytest.raises(ValueError):
        model.predict_features(video, mask)


def test_pure_terra_gradient_equals_full_expected_risk_autograd():
    model, video = student(), clip()
    original = copy.deepcopy(model.state_dict())
    mask = torch.ones(1, 5, dtype=torch.bool)
    labels = [torch.tensor([[2, 3, 1]]), torch.tensor([[4, 5, 1]])]
    probability = torch.stack(
        [
            sequence_log_probability(model(video, mask, value)["language"].logits, value)
            for value in labels
        ]
    )
    minimum_risk(probability, [10, 0])["loss"].backward()
    expected = visual_gradients(model, expected_tensors=2)
    model.zero_grad(set_to_none=True)
    measured = backward_candidate_risk(
        model, video, mask, labels, [10, 0], alpha=0.005, caption_weight=0
    )
    actual = visual_gradients(model, expected_tensors=2)
    assert measured["terra_gradient_coefficient_l1"] > 0
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=1e-4, atol=1e-8)
    for name, value in model.state_dict().items():
        assert torch.equal(value, original[name])


def test_pixel_conversion_and_transfer_preserve_values_and_pose():
    values = {
        "pixels": np.full((2, 3, 64, 64, 3), 255, dtype=np.uint8),
        "pose_features": np.arange(28, dtype=np.float32).reshape(2, 14),
    }
    video = clip_input(values).to("cpu")
    assert torch.all(video.pixels == 1)
    assert np.array_equal(video.pose.numpy(), values["pose_features"])
    assert not video.pose.requires_grad


def test_frozen_gradient_is_an_error():
    model = student()
    for parameter in model.trainable_parameters().values():
        parameter.grad = torch.ones_like(parameter)
    model.language.output.bias.grad = torch.ones_like(model.language.output.bias)
    with pytest.raises(RuntimeError, match="frozen"):
        visual_gradients(model, expected_tensors=2)
