"""Compare exact loss derivatives on tiny frozen modules; no real rewards."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from c5_monotone_feature_timing import WIDTHS
from c5_timing_risk import backward_timing_risk
from c5_timing_semantic_model import clip_input, gradient_report
from terra_quality_training import minimum_risk, sequence_log_probability
from terra_visual_learning import backward_candidate_risk
from test_c5_timing_semantic_model import student


@pytest.fixture
def sample():
    generator = np.random.default_rng(319)
    clip = clip_input(
        {
            "streams": {
                k: (generator.normal(size=(13, width)) * np.linspace(0.1, 2, 13)[:, None]).astype(
                    np.float32
                )
                for k, width in WIDTHS.items()
            },
            "valid_intervals": np.ones(12, dtype=np.bool_),
        }
    )
    mask = torch.ones(1, 13, dtype=torch.bool)
    labels = [torch.tensor([[2, 1]]), torch.tensor([[3, 4, 1]]), torch.tensor([[5, 1, -100]])]
    return clip, mask, labels, [10, 6, 0]


@pytest.mark.parametrize("nonzero", [False, True])
def test_one_encoder_gradient_matches_full_graph_and_original_streamed_method(sample, nonzero):
    clip, mask, labels, quality = sample
    model = student()
    if nonzero:
        with torch.no_grad():
            model.timing.rate_weights.copy_(torch.tensor([0.17, -0.22, 0.11, 0.06]))
    before = {k: v.clone() for k, v in model.state_dict().items()}
    logs = torch.stack(
        [
            sequence_log_probability(model(clip, mask, value)["language"].logits, value)
            for value in labels
        ]
    )
    expected = minimum_risk(logs, quality, alpha=0.005)
    expected["loss"].backward()
    gradient = model.timing.rate_weights.grad.clone()
    model.zero_grad(set_to_none=True)
    original = backward_candidate_risk(
        model, clip, mask, labels, quality, alpha=0.005, caption_weight=0
    )
    torch.testing.assert_close(model.timing.rate_weights.grad, gradient, rtol=1e-5, atol=1e-8)
    model.zero_grad(set_to_none=True)
    report = backward_timing_risk(model, clip, mask, labels, quality)
    torch.testing.assert_close(model.timing.rate_weights.grad, gradient, rtol=1e-5, atol=1e-8)
    assert report["loss"] == float(expected["loss"]) == original["loss"]
    assert report["candidate_probabilities"] == original["candidate_probabilities"]
    assert (
        report["candidate_sequence_log_probabilities"]
        == original["candidate_sequence_log_probabilities"]
    )
    assert report["memory_gradient_norm"] > 0 and report["differentiable_encoder_calls"] == 1
    assert gradient_report(model)["gradient_norm"] > 0
    assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())


def test_accumulation_divisor_preserves_prior_gradients_and_scales_new_contribution(sample):
    clip, mask, labels, quality = sample
    model = student()
    backward_timing_risk(model, clip, mask, labels, quality)
    original = model.timing.rate_weights.grad.clone()
    backward_timing_risk(model, clip, mask, labels, quality, divisor=4)
    torch.testing.assert_close(
        model.timing.rate_weights.grad, original * 1.25, rtol=1e-5, atol=1e-8
    )


def test_no_visible_support_preserves_all_frames_and_reports_zero_timing_gradient(sample):
    clip, mask, labels, quality = sample
    clip.valid_intervals.zero_()
    model = student()
    report = backward_timing_risk(model, clip, mask, labels, quality)
    assert not report["timing_memory_has_gradient_path"]
    assert report["memory_gradient_norm"] > 0
    assert gradient_report(model, require_signal=False)["gradient_norm"] == 0
    assert mask.shape == (1, 13) and mask.all()


def test_equal_actual_scores_produce_valid_zero_risk_gradient(sample):
    clip, mask, labels, _ = sample
    model = student()
    report = backward_timing_risk(model, clip, mask, labels, [10, 10, 10])
    assert report["loss"] == report["terra_gradient_coefficient_l1"] == 0
    assert gradient_report(model, require_signal=False)["gradient_norm"] == 0


@pytest.mark.parametrize(
    "change", ["divisor", "empty", "no_anchor", "float_score", "length", "label_dtype"]
)
def test_risk_rejects_ungraded_or_incomplete_inputs(sample, change):
    clip, mask, labels, quality = sample
    kwargs = {}
    if change == "divisor":
        kwargs["divisor"] = 0
    elif change == "empty":
        labels, quality = [], []
    elif change == "no_anchor":
        quality[0] = 9
    elif change == "float_score":
        quality[1] = 6.0
    elif change == "length":
        quality.pop()
    else:
        labels[0] = labels[0].float()
    with pytest.raises(ValueError):
        backward_timing_risk(student(), clip, mask, labels, quality, **kwargs)
