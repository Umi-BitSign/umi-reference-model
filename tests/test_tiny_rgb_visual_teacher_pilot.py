from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from tiny_rgb_capacity import visual_config
from tiny_rgb_language_model import ScratchRgbVision
from tiny_rgb_visual_teacher import STREAM_WIDTHS, visual_teacher_loss
from train_tiny_rgb_visual_teacher import RECIPE, build_student, clip_average_diagnostic_loss


def test_fresh_medium_vision_matches_seeded_control_without_loading_language():
    first, second = build_student(), build_student()
    assert all(
        torch.equal(value, second.state_dict()[name]) for name, value in first.state_dict().items()
    )
    torch.manual_seed(RECIPE["seed"])
    base = ScratchRgbVision(visual_config("medium"))
    assert all(
        torch.equal(value, first["vision"].state_dict()[name])
        for name, value in base.state_dict().items()
    )
    assert len(list(first.parameters())) == 50
    assert sum(p.numel() for p in first.parameters()) == 4111966
    assert not any("language" in name for name, _ in first.named_parameters())
    assert RECIPE["caption_or_language_updates"] == 0


def test_teacher_loss_reaches_all_fifty_fresh_student_parameters():
    student = build_student().eval()
    video = torch.randn(1, 2, 3, 32, 48)
    hidden, mask = student["vision"](video, torch.ones(1, 2, dtype=torch.bool))
    predictions, frame_mask = student["heads"](hidden, mask)
    targets = {name: torch.randn(1, 2, width) for name, width in STREAM_WIDTHS.items()}
    visual_teacher_loss(predictions, targets, frame_mask)["loss"].backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
        for p in student.parameters()
    )


def test_correct_wrong_clip_average_control_uses_identical_measurement_with_unequal_lengths():
    torch.manual_seed(12)
    targets = {name: torch.randn(1, 3, width) for name, width in STREAM_WIDTHS.items()}
    average = torch.cat([value.mean(1, keepdim=True) for value in targets.values()], -1)
    features = average.expand(1, 7, -1).clone()
    features[:, 5:] = float("nan")
    mask = torch.tensor([[True] * 5 + [False] * 2])
    correct = clip_average_diagnostic_loss(features, mask, targets)
    wrong = clip_average_diagnostic_loss(-features, mask, targets)
    assert abs(float(correct)) < 1e-6
    assert float(wrong - correct) > 1.0
    with pytest.raises(ValueError, match="masked feature"):
        clip_average_diagnostic_loss(features, torch.zeros_like(mask), targets)
