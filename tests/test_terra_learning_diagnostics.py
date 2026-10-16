from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from terra_learning_diagnostics import (
    conditioning,
    gradient_balance,
    retrieval_cases,
    stratum,
    sum_equivalence,
)


def test_gradient_balance_distinguishes_signal_size_from_cancellation():
    semantic = {"p": torch.tensor([3.0, 4.0])}
    auxiliary = {"p": torch.tensor([-3.0, -4.0])}
    result = gradient_balance(semantic, auxiliary)
    assert result["semantic_gradient_norm"] == 5
    assert result["semantic_to_weighted_auxiliary_norm_ratio"] == 1
    assert result["gradient_cosine"] == -1
    assert result["combined_gradient_norm"] == 0


def test_tied_scores_with_zero_semantic_gradient_are_valid():
    result = gradient_balance({"p": torch.zeros(2)}, {"p": torch.tensor([1.0, 2.0])})
    assert result["semantic_gradient_is_zero"]
    assert result["gradient_cosine"] is None
    assert result["semantic_to_weighted_auxiliary_norm_ratio"] == 0


@pytest.mark.parametrize("kind", ["scope", "shape", "nonfinite"])
def test_invalid_gradient_components_are_rejected(kind):
    first, second = {"p": torch.ones(2)}, {"p": torch.ones(2)}
    if kind == "scope":
        second = {"other": torch.ones(2)}
    elif kind == "shape":
        second["p"] = torch.ones(3)
    else:
        first["p"][0] = math.nan
    with pytest.raises(ValueError):
        gradient_balance(first, second)


def test_components_must_reproduce_actual_full_loss_gradient():
    a, b = {"p": torch.tensor([1.0, 2.0])}, {"p": torch.tensor([2.0, 3.0])}
    assert sum_equivalence(a, b, {"p": torch.tensor([3.0, 5.0])}) == 0
    with pytest.raises(ValueError, match="actual full loss"):
        sum_equivalence(a, b, {"p": torch.tensor([1.0, 5.0])})


def test_conditioning_holds_caption_prior_fixed_within_each_row():
    result = conditioning([[0.1, 1.1], [0.9, 0.3]])
    assert result["mean_correct_video_CE"] == pytest.approx(0.2)
    assert result["mean_other_video_minus_correct_CE"] == pytest.approx(0.8)
    assert result["strict_correct_video_top1"] == 2
    assert result["other_videos_are_not_human_verified_semantic_negatives"]
    assert not result["sign_cue_reliance_or_generalization_proven"]


def test_conditioning_retrieval_does_not_count_ties_as_success():
    assert conditioning([[1.0, 1.0], [1.0, 1.0]])["strict_correct_video_top1"] == 0


@pytest.mark.parametrize(
    "matrix", [[], [[1.0]], [[1.0, 0.2]], [[1.0, -1.0], [0.3, 0.2]], [[1.0, math.nan], [0.2, 0.3]]]
)
def test_invalid_conditioning_matrices_are_rejected(matrix):
    with pytest.raises(ValueError):
        conditioning(matrix)


def records():
    result = []
    for kind, quality, count in (
        ("contrast", [10, 1], 8),
        ("tied", [10, 10], 5),
        ("singleton", [10], 5),
    ):
        for i in range(count):
            result.append(
                {
                    "quality": quality,
                    "data_row": {
                        "source": "source",
                        "example_id": f"{kind}{i}",
                        "source_group": f"{kind}group{i}",
                        "reference": f"{kind} caption {i}",
                        "role": "training",
                        "training_allowed": True,
                    },
                }
            )
    return result


def test_training_case_selection_is_deterministic_stratified_and_source_distinct():
    inputs = records()
    selected = retrieval_cases(inputs)
    assert selected == retrieval_cases(list(reversed(inputs)))
    assert [stratum(r) for r in selected].count("contrast") == 6
    assert [stratum(r) for r in selected].count("tied_pair") == 3
    assert [stratum(r) for r in selected].count("singleton") == 3
    assert len({r["data_row"]["source_group"] for r in selected}) == 12


def test_retrieval_case_selection_rejects_development_permissions():
    inputs = records()
    for r in inputs:
        r["data_row"]["training_allowed"] = False
    with pytest.raises(ValueError, match="development"):
        retrieval_cases(inputs)


def test_retrieval_selection_rejects_insufficient_distinct_groups():
    inputs = copy.deepcopy(records())
    for r in inputs:
        r["data_row"]["source_group"] = "same group"
    with pytest.raises(ValueError, match="insufficient"):
        retrieval_cases(inputs)
