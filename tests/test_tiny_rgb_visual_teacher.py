from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from tiny_rgb_capacity import visual_config
from tiny_rgb_capacity_activations import replace_capacity_gelu
from tiny_rgb_capacity_positions import CanonicalPositionsVision
from tiny_rgb_visual_teacher import (
    STREAM_WIDTHS,
    SpatialTeacherHeads,
    verify_training_alignment,
    visual_teacher_loss,
)


def binding():
    return {
        "role": "training",
        "training_allowed": True,
        "source": "how2sign",
        "example_id": "fixture",
        "source_group": "recording",
        "video_sha256": "a" * 64,
        "frame_indices_sha256": "b" * 64,
        "duration_integrity_audit_sha256": "c" * 64,
        "sampled_frames": 3,
    }


def test_alignment_requires_protected_training_identity_and_exact_timeline():
    row = binding()
    verify_training_alignment(row, copy.deepcopy(row))
    for field, value in (
        ("role", "open_development"),
        ("training_allowed", False),
        ("example_id", "other"),
        ("source_group", "other"),
        ("video_sha256", "d" * 64),
        ("frame_indices_sha256", "d" * 64),
        ("sampled_frames", 2),
        ("sampled_frames", True),
        ("duration_integrity_audit_sha256", None),
    ):
        changed = {**row, field: value}
        with pytest.raises(ValueError):
            verify_training_alignment(row, changed)


def test_heads_preserve_four_token_geometry_and_only_train_student():
    torch.manual_seed(14)
    hidden = torch.randn(1, 12, 8, requires_grad=True)
    heads = SpatialTeacherHeads(8)
    mask = torch.tensor([[True] * 8 + [False] * 4])
    predictions, frame_mask = heads(hidden, mask)
    targets = {
        name: torch.randn(1, 3, width, requires_grad=True) for name, width in STREAM_WIDTHS.items()
    }
    loss = visual_teacher_loss(predictions, targets, frame_mask)
    assert loss["usable_frames"] == dict.fromkeys(STREAM_WIDTHS, 2)
    loss["loss"].backward()
    assert all(target.grad is None for target in targets.values())
    assert hidden.grad[:, :8].abs().sum() > 0
    assert hidden.grad[:, 8:].abs().sum() == 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in heads.parameters())


def test_padding_is_excluded_before_checks_and_loss_is_identical():
    torch.manual_seed(15)
    predictions = {name: torch.randn(1, 3, width) for name, width in STREAM_WIDTHS.items()}
    targets = {name: torch.randn_like(value) for name, value in predictions.items()}
    mask = torch.tensor([[True, True, False]])
    expected = visual_teacher_loss(predictions, targets, mask)["loss"]
    for values in (predictions, targets):
        for value in values.values():
            value[:, 2] = float("nan")
    actual = visual_teacher_loss(predictions, targets, mask)["loss"]
    assert torch.equal(expected, actual)
    targets["face_features"][:, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite valid"):
        visual_teacher_loss(predictions, targets, mask)


def test_missing_hands_are_masked_not_taught_as_zero_labels():
    predictions = {
        name: torch.randn(1, 2, width, requires_grad=True) for name, width in STREAM_WIDTHS.items()
    }
    targets = {name: torch.ones_like(value) for name, value in predictions.items()}
    for name in ("left_hand_features", "right_hand_features"):
        targets[name].zero_()
    result = visual_teacher_loss(predictions, targets, torch.ones(1, 2, dtype=torch.bool))
    assert result["usable_frames"]["left_hand_features"] == 0
    result["loss"].backward()
    assert predictions["left_hand_features"].grad.abs().sum() == 0
    targets["face_features"].zero_()
    with pytest.raises(ValueError, match="no usable"):
        visual_teacher_loss(predictions, targets, torch.ones(1, 2, dtype=torch.bool))


def test_bad_token_geometry_and_partial_frame_masks_are_rejected():
    heads = SpatialTeacherHeads(8)
    with pytest.raises(ValueError, match="four spatial tokens"):
        heads(torch.randn(1, 7, 8), torch.ones(1, 7, dtype=torch.bool))
    with pytest.raises(ValueError, match="part of a real frame"):
        heads(torch.randn(1, 8, 8), torch.tensor([[True, False] + [True] * 6]))


@pytest.mark.parametrize("weight", [-1.0, float("nan"), float("inf")])
def test_invalid_pose_weight_is_rejected(weight):
    values = {name: torch.zeros(1, 1, width) for name, width in STREAM_WIDTHS.items()}
    with pytest.raises(ValueError, match="coefficient"):
        visual_teacher_loss(values, values, torch.ones(1, 1, dtype=torch.bool), pose_weight=weight)


def test_empty_masks_and_misaligned_feature_targets_are_rejected():
    values = {name: torch.ones(1, 2, width) for name, width in STREAM_WIDTHS.items()}
    with pytest.raises(ValueError, match="frame mask"):
        visual_teacher_loss(values, values, torch.zeros(1, 2, dtype=torch.bool))
    targets = {**values, "face_features": torch.ones(1, 1, 384)}
    with pytest.raises(ValueError, match="shape mismatch"):
        visual_teacher_loss(values, targets, torch.ones(1, 2, dtype=torch.bool))


def test_building_teacher_heads_never_changes_visual_state_or_pixel_surface():
    vision = CanonicalPositionsVision(visual_config("small"))
    original = {name: value.clone() for name, value in vision.state_dict().items()}
    heads = SpatialTeacherHeads(vision.config.width)
    assert all(torch.equal(value, vision.state_dict()[name]) for name, value in original.items())
    assert sum(parameter.numel() for parameter in heads.parameters()) == 1166 * 193
    assert vision.config == visual_config("small")


def test_auxiliary_teacher_signal_reaches_scratch_rgb_pixels_and_stem():
    torch.manual_seed(16)
    vision = CanonicalPositionsVision(visual_config("small")).eval()
    replace_capacity_gelu(vision)
    video = torch.randn(1, 3, 3, 32, 48, requires_grad=True)
    hidden, mask = vision(video, torch.tensor([[True, True, False]]))
    heads = SpatialTeacherHeads(vision.config.width)
    predictions, frame_mask = heads(hidden, mask)
    targets = {name: torch.randn(1, 3, width) for name, width in STREAM_WIDTHS.items()}
    result = visual_teacher_loss(predictions, targets, frame_mask)
    result["loss"].backward()
    assert video.grad[:, :2].abs().sum() > 0
    assert video.grad[:, 2:].abs().sum() == 0
    assert vision.stem[0].weight.grad.abs().sum() > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in vision.parameters())
