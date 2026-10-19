"""Independent full autograd comparisons for every adapter parameter group."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_c5_temporal_memory_model as model_tests
from c5_temporal_memory_risk import backward_memory_risk, gradient_report
from terra_quality_training import sequence_log_probability

fixture = model_tests.fixture


def labels():
    return [torch.tensor([[2, 1, -100]]), torch.tensor([[3, 4, 1]]), torch.tensor([[5, 1]])]


@pytest.mark.parametrize("nonzero", [False, True])
@pytest.mark.parametrize("weights", [(0.25, 0, 0), (1.0, 0.1, 0.01), (1.0, 1.0, 1.0)])
def test_matches_independent_full_autograd_for_all24600_parameters(fixture, nonzero, weights):
    model, clip, mask = fixture
    alpha, caption, residual = weights
    if nonzero:
        with torch.no_grad():
            output = model.memory_adapter.output_projection.weight
            output.copy_(torch.sin(torch.arange(output.numel()).reshape_as(output)) * 0.03)
            model.memory_adapter.temporal_weights[:, 0].fill_(0.2)
            model.memory_adapter.temporal_weights[:, 2].fill_(-0.1)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    candidates, quality = labels(), [10, 7, 0]
    outputs = [model(clip, mask, value) for value in candidates]
    logs = torch.stack(
        [
            sequence_log_probability(output["language"].logits, value)
            for output, value in zip(outputs, candidates, strict=True)
        ]
    )
    counts = logs.new_tensor([2, 3, 2])
    probabilities = torch.softmax(alpha * logs / counts, dim=0)
    error = (probabilities * logs.new_tensor([0, 3, 10])).sum()
    loss = (
        error
        + caption * outputs[0]["language"].loss
        + residual * outputs[0]["normalized_residual_penalty"]
    )
    loss.backward()
    reference = {name: p.grad.clone() for name, p in model.trainable_parameters().items()}
    model.zero_grad(set_to_none=True)
    calls = []
    hook = model.semantic_encoder.register_forward_hook(lambda *args: calls.append(True))
    report = backward_memory_risk(
        model,
        clip,
        mask,
        candidates,
        quality,
        alpha=alpha,
        caption_weight=caption,
        residual_weight=residual,
    )
    hook.remove()
    for name, parameter in model.trainable_parameters().items():
        torch.testing.assert_close(parameter.grad, reference[name], rtol=1e-4, atol=1e-9)
    assert report["loss"] == pytest.approx(float(loss), abs=1e-6)
    assert report["candidate_probabilities"] == pytest.approx(
        probabilities.detach().tolist(), abs=1e-7
    )
    assert report["candidate_token_counts"] == [2, 3, 2]
    assert len(calls) == report["original_encoder_calls"] == report["adapter_backward_calls"] == 1
    assert report["differentiable_encoder_calls"] == 0
    assert gradient_report(model)["parameters"] == 24600
    assert all(torch.equal(value, before[name]) for name, value in model.state_dict().items())


def test_flat_risk_and_all_hidden_intervals_still_learn_from_the_caption(fixture):
    model, clip, mask = fixture
    clip.valid_intervals.zero_()
    report = backward_memory_risk(model, clip, mask, labels(), [10, 10, 10])
    assert report["expected_terra_error"] == report["terra_gradient_coefficient_l1"] == 0
    assert gradient_report(model)["gradient_norm"] > 0


def test_divisor_scales_penalty_and_decoder_gradients_without_erasing_accumulation(fixture):
    model, clip, mask = fixture
    with torch.no_grad():
        model.memory_adapter.output_projection.weight.fill_(0.02)
    backward_memory_risk(model, clip, mask, labels(), [10, 7, 0], residual_weight=0.5)
    gradients = {k: p.grad.clone() for k, p in model.trainable_parameters().items()}
    backward_memory_risk(model, clip, mask, labels(), [10, 7, 0], residual_weight=0.5, divisor=4)
    for name, parameter in model.trainable_parameters().items():
        torch.testing.assert_close(parameter.grad, gradients[name] * 1.25, rtol=1e-5, atol=1e-9)


@pytest.mark.parametrize(
    "change", ["alpha", "caption", "penalty", "anchor", "grade", "tokens", "divisor"]
)
def test_invalid_objective_or_labels_reject_before_encoder(fixture, monkeypatch, change):
    model, clip, mask = fixture
    candidates, quality, kwargs = labels(), [10, 7, 0], {}
    if change == "alpha":
        kwargs["alpha"] = float("nan")
    elif change == "caption":
        kwargs["caption_weight"] = -1
    elif change == "penalty":
        kwargs["residual_weight"] = float("inf")
    elif change == "anchor":
        quality[0] = 9
    elif change == "grade":
        quality[1] = 7.0
    elif change == "tokens":
        candidates[0].fill_(-100)
    else:
        kwargs["divisor"] = 0
    monkeypatch.setattr(
        model, "encode_details", lambda *args: pytest.fail("encoder before admission")
    )
    with pytest.raises(ValueError):
        backward_memory_risk(model, clip, mask, candidates, quality, **kwargs)


@pytest.mark.parametrize("change", ["absent", "nan", "zero", "backbone"])
def test_gradient_report_rejects_invalid_or_missing_scope(fixture, change):
    model, clip, mask = fixture
    if change != "absent":
        backward_memory_risk(model, clip, mask, labels(), [10, 7, 0])
    if change == "nan":
        model.memory_adapter.output_projection.weight.grad[0, 0] = float("nan")
    elif change == "zero":
        for parameter in model.trainable_parameters().values():
            parameter.grad.zero_()
    elif change == "backbone":
        model.language.scale.grad = torch.ones(())
    with pytest.raises(ValueError):
        gradient_report(model)
