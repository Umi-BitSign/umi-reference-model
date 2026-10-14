from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from profile_winner_student_memory import memory_statistics


def test_memory_profile_distinguishes_scale_mismatch_from_direction():
    target = torch.tensor([[[1.0, -1.0], [2.0, -2.0]]])
    before = target.clone()
    result = memory_statistics(target * 20, target)
    assert result["student_to_teacher_rms_ratio"] == pytest.approx(20)
    assert result["mean_frame_cosine"] == pytest.approx(1)
    assert result["least_squares_global_scale"] == pytest.approx(0.05)
    assert result["scaled_rmse"] == pytest.approx(0, abs=1e-6)
    assert not result["quality_verified"] and torch.equal(target, before)
    orthogonal = target.flip(-1) * torch.tensor([1, -1])
    result = memory_statistics(orthogonal, target)
    assert result["mean_frame_cosine"] == pytest.approx(0)
    assert result["least_squares_global_scale"] == 0
    assert result["scaled_rmse"] > 0


@pytest.mark.parametrize("kind", ["shape", "nonfinite", "zero"])
def test_memory_profile_rejects_incomparable_or_invalid_states(kind):
    student = torch.ones(1, 2, 3)
    target = torch.ones_like(student)
    if kind == "shape":
        target = target[:, :1]
    elif kind == "nonfinite":
        student[0, 0, 0] = float("nan")
    else:
        target.zero_()
    with pytest.raises(ValueError, match="memory diagnostic"):
        memory_statistics(student, target)
