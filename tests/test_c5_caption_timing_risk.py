"""Exact independent full-graph reference on tiny frozen backbones; no real grades."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_c5_timing_risk as original_tests
from c5_caption_timing_risk import backward_caption_timing_risk
from c5_timing_semantic_model import gradient_report
from terra_quality_training import sequence_log_probability
from test_c5_timing_semantic_model import student

sample = original_tests.sample


@pytest.mark.parametrize("nonzero", [False, True])
@pytest.mark.parametrize("alpha,caption_weight", [(0.25, 0), (1.0, 0.1), (1.0, 1.0)])
def test_single_encoder_derivative_matches_independent_full_graph(
    sample, nonzero, alpha, caption_weight
):
    clip, mask, labels, quality = sample
    model = student()
    if nonzero:
        with torch.no_grad():
            model.timing.rate_weights.copy_(torch.tensor([0.17, -0.22, 0.11, 0.06]))
    before = {k: v.clone() for k, v in model.state_dict().items()}
    outputs = [model(clip, mask, value)["language"] for value in labels]
    logs = torch.stack(
        [
            sequence_log_probability(output.logits, value)
            for output, value in zip(outputs, labels, strict=True)
        ]
    )
    lengths = torch.tensor([int((value != -100).sum()) for value in labels])
    probabilities = torch.softmax(alpha * logs / lengths, dim=0)
    expected_risk = (probabilities * (10 - torch.tensor(quality))).sum()
    expected = expected_risk + caption_weight * outputs[0].loss
    expected.backward()
    gradient = model.timing.rate_weights.grad.clone()
    model.zero_grad(set_to_none=True)
    calls = []
    hook = model.semantic_encoder.register_forward_hook(lambda *args: calls.append(True))
    report = backward_caption_timing_risk(
        model, clip, mask, labels, quality, alpha=alpha, caption_weight=caption_weight
    )
    hook.remove()
    torch.testing.assert_close(model.timing.rate_weights.grad, gradient, rtol=1e-5, atol=1e-8)
    assert report["loss"] == pytest.approx(float(expected), abs=1e-6)
    assert report["candidate_probabilities"] == pytest.approx(
        probabilities.detach().tolist(), abs=1e-7
    )
    assert report["candidate_token_counts"] == [2, 3, 2]
    assert len(calls) == report["differentiable_encoder_calls"] == 1
    assert gradient_report(model)["frozen_backbone_gradients_absent"]
    assert all(torch.equal(value, before[name]) for name, value in model.state_dict().items())


def test_anchor_supervision_provides_signal_when_candidate_risk_is_flat(sample):
    clip, mask, labels, _ = sample
    model = student()
    report = backward_caption_timing_risk(
        model, clip, mask, labels, [10, 10, 10], caption_weight=0.1
    )
    assert report["expected_terra_error"] == report["terra_gradient_coefficient_l1"] == 0
    assert report["caption_loss"] > 0 and report["loss"] > 0
    assert gradient_report(model)["gradient_norm"] > 0


def test_divisor_scales_both_loss_components_and_preserves_accumulated_gradients(sample):
    clip, mask, labels, quality = sample
    model = student()
    backward_caption_timing_risk(model, clip, mask, labels, quality)
    first = model.timing.rate_weights.grad.clone()
    backward_caption_timing_risk(model, clip, mask, labels, quality, divisor=4)
    torch.testing.assert_close(model.timing.rate_weights.grad, first * 1.25, rtol=1e-5, atol=1e-8)


def test_no_support_is_a_complete_exposure_with_zero_timing_derivative(sample):
    clip, mask, labels, quality = sample
    clip.valid_intervals.zero_()
    model = student()
    report = backward_caption_timing_risk(model, clip, mask, labels, quality)
    assert report["memory_gradient_norm"] > 0 and not report["timing_memory_has_gradient_path"]
    assert gradient_report(model, require_signal=False)["gradient_norm"] == 0
    assert mask.shape == (1, 13) and mask.all()


@pytest.mark.parametrize(
    "change", ["alpha", "caption", "anchor", "score", "empty_label", "divisor"]
)
def test_bad_configuration_or_incomplete_anchor_is_rejected_before_encoding(
    sample, monkeypatch, change
):
    clip, mask, labels, quality = sample
    kwargs = {}
    if change == "alpha":
        kwargs["alpha"] = float("nan")
    elif change == "caption":
        kwargs["caption_weight"] = -0.1
    elif change == "anchor":
        quality[0] = 9
    elif change == "score":
        quality[1] = 6.0
    elif change == "empty_label":
        labels[0].fill_(-100)
    else:
        kwargs["divisor"] = 0
    model = student()
    monkeypatch.setattr(
        model, "encode", lambda *args: pytest.fail("encoded before input validation")
    )
    with pytest.raises(ValueError):
        backward_caption_timing_risk(model, clip, mask, labels, quality, **kwargs)
