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
from c5_small_input_semantic import Native112Clip, Native112SemanticStudent, clip_input
from c5_small_input_vision import STREAMS
from roi_semantic_student import visual_gradients
from terra_quality_training import minimum_risk, sequence_log_probability
from terra_visual_learning import backward_candidate_risk


class Vision(nn.Module):
    """Small gradient fixture, not real DINO qualification."""

    def __init__(self):
        super().__init__()
        self.side = 112
        self.projection = nn.Linear(9, 1152)
        self.hands, self.face = nn.Module(), nn.Module()
        for model in (self.hands, self.face):
            model.mask_token = nn.Parameter(torch.zeros(1, 384), requires_grad=False)

    def forward(self, pixels):
        values = self.projection(pixels.mean((-1, -2)).reshape(len(pixels), 9))
        return dict(zip(STREAMS, values.chunk(3, dim=-1), strict=True))


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
        logits = self.output(encoder_outputs.last_hidden_state.mean(1, keepdim=True))
        logits = logits.expand(1, labels.shape[1], 16)
        return SimpleNamespace(
            logits=logits, loss=F.cross_entropy(logits.reshape(-1, 16), labels.reshape(-1))
        )


def model():
    torch.manual_seed(772)
    return Native112SemanticStudent(Vision(), Encoder(), Language()).eval()


def clip():
    return Native112Clip(torch.randn(5, 3, 3, 112, 112), torch.randn(5, 14))


def test_constructor_preserves_original_values_and_frozen_mask_scope_even_in_train_mode():
    vision, encoder, language = Vision(), Encoder(), Language()
    original = copy.deepcopy(vision.state_dict())
    student = Native112SemanticStudent(vision, encoder, language).train()
    assert set(student.trainable_parameters()) == {
        "vision.projection.weight",
        "vision.projection.bias",
    }
    assert student.semantic_encoder.training is False and student.language.training is False
    assert not vision.hands.mask_token.requires_grad and not vision.face.mask_token.requires_grad
    assert all(torch.equal(value, original[name]) for name, value in vision.state_dict().items())


@pytest.mark.parametrize(
    "mutation", ["mask", "functional_frozen", "side", "chunk_zero", "chunk_bool"]
)
def test_changed_visual_scope_or_surface_is_rejected(mutation):
    vision, chunk = Vision(), 2
    if mutation == "mask":
        vision.face.mask_token.requires_grad_(True)
    elif mutation == "functional_frozen":
        vision.projection.weight.requires_grad_(False)
    elif mutation == "side":
        vision.side = 64
    elif mutation == "chunk_zero":
        chunk = 0
    else:
        chunk = True
    with pytest.raises(ValueError):
        Native112SemanticStudent(vision, Encoder(), Language(), chunk_frames=chunk)


def test_complete_timeline_and_original_pose_survive_differentiable_chunks():
    student, video = model(), clip()
    mask = torch.ones(1, 5, dtype=torch.bool)
    features, actual_mask = student.predict_features(video, mask)
    assert actual_mask is mask
    assert all(features[name].shape == (1, 5, 384) for name in STREAMS)
    torch.testing.assert_close(features["pose_features"][0], video.pose)
    sum(features[name].square().mean() for name in STREAMS).backward()
    assert len(visual_gradients(student, expected_tensors=2)) == 2
    assert (
        student.vision.hands.mask_token.grad is None and student.vision.face.mask_token.grad is None
    )


@pytest.mark.parametrize(
    "mutation",
    ["interior_mask", "trailing_mask", "pose_grad", "pose_nan", "pixel_nan", "small_pixels"],
)
def test_partial_or_caption_trainable_pose_and_invalid_pixels_are_rejected(mutation):
    student, video = model(), clip()
    mask = torch.ones(1, 5, dtype=torch.bool)
    if mutation == "interior_mask":
        mask[0, 1] = False
    elif mutation == "trailing_mask":
        mask[0, -1] = False
    elif mutation == "pose_grad":
        video.pose.requires_grad_(True)
    elif mutation == "pose_nan":
        video.pose[0, 0] = torch.nan
    elif mutation == "pixel_nan":
        video.pixels[0, 0, 0, 0, 0] = torch.nan
    else:
        video = Native112Clip(video.pixels[..., :64, :64], video.pose)
    with pytest.raises(ValueError):
        student.predict_features(video, mask)


def test_actual_candidate_gradient_matches_full_expected_risk_without_weight_updates():
    student, video = model(), clip()
    original = copy.deepcopy(student.state_dict())
    mask = torch.ones(1, 5, dtype=torch.bool)
    labels = [torch.tensor([[2, 3, 1]]), torch.tensor([[4, 5, 1]])]
    probabilities = torch.stack(
        [
            sequence_log_probability(student(video, mask, value)["language"].logits, value)
            for value in labels
        ]
    )
    minimum_risk(probabilities, [10, 0])["loss"].backward()
    expected = visual_gradients(student, expected_tensors=2)
    student.zero_grad(set_to_none=True)
    measured = backward_candidate_risk(
        student, video, mask, labels, [10, 0], alpha=0.005, caption_weight=0
    )
    actual = visual_gradients(student, expected_tensors=2)
    assert measured["terra_gradient_coefficient_l1"] > 0
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=1e-4, atol=1e-8)
    assert all(torch.equal(value, original[name]) for name, value in student.state_dict().items())


def test_input_uses_C5_imagenet_normalization_and_copies_original_pose():
    values = {
        "pixels": np.full((2, 3, 112, 112, 3), 255, dtype=np.uint8),
        "pose_features": np.arange(28, dtype=np.float32).reshape(2, 14),
    }
    video = clip_input(values).to("cpu")
    torch.testing.assert_close(
        video.pixels[0, 0, :, 0, 0],
        torch.tensor([(1 - 0.485) / 0.229, (1 - 0.456) / 0.224, (1 - 0.406) / 0.225]),
    )
    np.testing.assert_array_equal(video.pose.numpy(), values["pose_features"])
    video.pose.zero_()
    assert values["pose_features"].sum() > 0 and not video.pose.requires_grad
