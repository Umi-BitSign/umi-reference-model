import copy
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from c5_all_source_training import RECIPE as CONTROL
from c5_retention import RECIPE, teacher_kl
from run_c5_stem_projection_frozen_tuning import tuning_config


def test_only_retention_changes_matched_recipe():
    without_retention = {key: value for key, value in RECIPE.items() if key not in {"id", "retention"}}
    assert without_retention == {key: value for key, value in CONTROL.items() if key != "id"}
    plan = {"schema": "umi-studio-c5-frozen-replay-plan/1", "recipe": copy.deepcopy(RECIPE)}
    assert tuning_config(plan)[0] == RECIPE
    plan["recipe"]["retention"]["coefficient"] = 2.0
    with pytest.raises(ValueError, match="recipe changed"):
        tuning_config(plan)


def test_identical_logits_have_zero_retention():
    logits = torch.randn(1, 4, 8, requires_grad=True)
    labels = torch.tensor([[1, 2, 3, 4]])
    loss = teacher_kl(logits, logits.detach().clone(), labels, 2.0)
    assert loss.item() == 0.0
    loss.backward()
    assert torch.allclose(logits.grad, torch.zeros_like(logits.grad), atol=1e-7)


def test_retention_matches_kl_and_only_updates_student_valid_tokens():
    student = torch.tensor([[[2., 1., -1.], [-2., 3., 1.]]], requires_grad=True)
    teacher = torch.tensor([[[1., 2., -1.], [4., -3., 2.]]], requires_grad=True)
    labels = torch.tensor([[1, -100]])
    loss = teacher_kl(student, teacher, labels, 2.0)
    expected = torch.nn.functional.kl_div(
        torch.log_softmax(student[:, 0] / 2, -1),
        torch.softmax(teacher[:, 0].detach() / 2, -1),
        reduction="batchmean",
    ) * 4
    assert loss.item() == pytest.approx(expected.item(), abs=1e-6)
    loss.backward()
    assert teacher.grad is None
    assert torch.count_nonzero(student.grad[:, 0]) > 0
    assert torch.count_nonzero(student.grad[:, 1]) == 0


@pytest.mark.parametrize("temperature", [0., -1., float("nan"), float("inf")])
def test_invalid_temperature_rejected(temperature):
    logits = torch.zeros(1, 2, 3)
    with pytest.raises(ValueError, match="temperature"):
        teacher_kl(logits, logits, torch.zeros(1, 2), temperature)


def test_shape_and_empty_mask_rejected():
    logits = torch.zeros(1, 2, 3)
    with pytest.raises(ValueError, match="shape"):
        teacher_kl(logits, logits[:, :1], torch.zeros(1, 2), 2.)
    with pytest.raises(ValueError, match="nonignored"):
        teacher_kl(logits, logits, torch.full((1, 2), -100), 2.)
