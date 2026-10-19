"""Synthetic interpolation/gradient checks, with no C5 model or real cohort."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from c5_monotone_feature_timing import WIDTHS, MonotoneFeatureTiming


def streams(frames=13):
    generator = torch.Generator().manual_seed(719)
    scale = torch.linspace(0.2, 2, frames)[:, None]
    return {
        name: torch.randn(frames, width, generator=generator) * scale
        for name, width in WIDTHS.items()
    }


def configured():
    model = MonotoneFeatureTiming()
    with torch.no_grad():
        model.rate_weights.copy_(torch.tensor([0.17, -0.22, 0.11, 0.06]))
    return model


def test_initial_inference_identity_is_bitwise_exact_including_signed_zero_and_is_not_aliased():
    values = streams()
    values["pose_features"][0, 0] = -0.0
    original = {k: v.clone() for k, v in values.items()}
    model = MonotoneFeatureTiming().eval()
    with torch.no_grad():
        result = model(values, valid_intervals=torch.ones(12, dtype=torch.bool))
    assert sum(p.numel() for p in model.parameters()) == 4
    assert torch.equal(result["positions"], torch.arange(13, dtype=torch.float32))
    assert torch.equal(result["interval_rates"], torch.ones(12))
    for name in WIDTHS:
        assert torch.equal(
            result["streams"][name].view(torch.int32), original[name].view(torch.int32)
        )
        assert result["streams"][name].data_ptr() != values[name].data_ptr()
        assert torch.equal(values[name].view(torch.int32), original[name].view(torch.int32))


def test_all_streams_follow_same_monotone_positions_and_independent_numpy_interpolation():
    values = streams()
    result = configured()(values, valid_intervals=torch.ones(12, dtype=torch.bool))
    positions = result["positions"].detach().numpy()
    assert positions[0] == 0 and positions[-1] == 12 and np.all(np.diff(positions) > 0)
    assert not np.array_equal(positions, np.arange(13))
    for name in WIDTHS:
        source = values[name].numpy()
        expected = np.stack(
            [np.interp(positions, np.arange(13), source[:, i]) for i in range(source.shape[1])],
            axis=1,
        )
        np.testing.assert_allclose(
            result["streams"][name].detach().numpy(), expected, rtol=2e-5, atol=2e-6
        )


def test_visibility_gaps_fix_both_endpoints_and_isolate_component_normalization():
    values = streams()
    mask = torch.ones(12, dtype=torch.bool)
    mask[5] = False
    model = configured()
    result = model(values, valid_intervals=mask)
    assert result["positions"][5] == 5 and result["positions"][6] == 6
    changed = {k: v.clone() for k, v in values.items()}
    for value in changed.values():
        value[6:] *= 100
    second = model(changed, valid_intervals=mask)
    assert torch.equal(result["positions"][:6], second["positions"][:6])
    for name in WIDTHS:
        assert torch.equal(result["streams"][name][:6], second["streams"][name][:6])
    assert result["interval_rates"][5] == 1


@pytest.mark.parametrize("frames", [1, 2, 7])
def test_singletons_short_components_and_all_gaps_keep_identity(frames):
    values = streams(frames)
    result = configured()(values, valid_intervals=torch.zeros(frames - 1, dtype=torch.bool))
    assert torch.equal(result["positions"], torch.arange(frames, dtype=torch.float32))
    assert result["log_rate_smoothness_penalty"] == 0
    assert result["normalized_displacement_penalty"] == 0
    for name in WIDTHS:
        assert torch.equal(result["streams"][name], values[name])


def test_stationary_features_have_identity_at_nonzero_parameters():
    values = {k: v[:1].expand_as(v).clone() for k, v in streams().items()}
    result = configured()(values, valid_intervals=torch.ones(12, dtype=torch.bool))
    assert torch.equal(result["positions"], torch.arange(13, dtype=torch.float32))


def test_extreme_weights_keep_positive_bounded_rates_and_complete_coverage():
    model = configured()
    with torch.no_grad():
        model.rate_weights.mul_(10000)
    result = model(streams(), valid_intervals=torch.ones(12, dtype=torch.bool))
    assert (result["interval_rates"] >= 0.5).all() and (result["interval_rates"] <= 2).all()
    assert (result["positions"][1:] > result["positions"][:-1]).all()
    assert result["positions"][0] == 0 and result["positions"][-1] == 12


def objective(result):
    return (result["positions"] * torch.linspace(-1, 1, len(result["positions"]))).mean() + result[
        "streams"
    ]["face_features"][:, 0].mean()


def test_identity_training_path_retains_nonzero_semantic_surrogate_gradients():
    model = MonotoneFeatureTiming()
    result = model(streams(), valid_intervals=torch.ones(12, dtype=torch.bool))
    objective(result).backward()
    assert model.rate_weights.grad is not None and torch.isfinite(model.rate_weights.grad).all()
    assert (model.rate_weights.grad.abs() > 1e-6).all()


def test_nonzero_timing_gradient_matches_central_finite_differences_away_from_knots():
    model = configured()
    values, mask = streams(), torch.ones(12, dtype=torch.bool)
    result = model(values, valid_intervals=mask)
    interior = result["positions"][1:-1].detach()
    assert ((interior - interior.round()).abs() > 0.003).all()
    objective(result).backward()
    actual = model.rate_weights.grad.clone()
    estimates = []
    step = 0.001
    with torch.no_grad():
        for i in range(4):
            saved = model.rate_weights[i].item()
            model.rate_weights[i] = saved + step
            right = objective(model(values, valid_intervals=mask)).item()
            model.rate_weights[i] = saved - step
            left = objective(model(values, valid_intervals=mask)).item()
            model.rate_weights[i] = saved
            estimates.append((right - left) / (2 * step))
    torch.testing.assert_close(actual, torch.tensor(estimates), rtol=0.01, atol=0.0002)


def test_dictionary_order_does_not_change_feature_channels():
    values = streams()
    model, mask = configured(), torch.ones(12, dtype=torch.bool)
    a = model(values, valid_intervals=mask)
    b = model(dict(reversed(list(values.items()))), valid_intervals=mask)
    assert torch.equal(a["positions"], b["positions"])


@pytest.mark.parametrize(
    "change",
    ["dtype", "width", "missing", "nonfinite", "requires_grad", "mask_dtype", "mask_length"],
)
def test_invalid_or_trainable_backbone_inputs_are_rejected(change):
    values, mask = streams(), torch.ones(12, dtype=torch.bool)
    if change == "dtype":
        values["face_features"] = values["face_features"].double()
    elif change == "width":
        values["pose_features"] = values["pose_features"][:, :13]
    elif change == "missing":
        del values["left_hand_features"]
    elif change == "nonfinite":
        values["face_features"][0, 0] = float("nan")
    elif change == "requires_grad":
        values["face_features"].requires_grad_(True)
    elif change == "mask_dtype":
        mask = mask.float()
    else:
        mask = mask[:1]
    with pytest.raises(ValueError):
        configured()(values, valid_intervals=mask)


@pytest.mark.parametrize("bound", [0, -1, float("inf"), float("nan"), math.log(5), True])
def test_invalid_log_rate_bounds_are_rejected(bound):
    with pytest.raises(ValueError):
        MonotoneFeatureTiming(maximum_log_rate=bound)
