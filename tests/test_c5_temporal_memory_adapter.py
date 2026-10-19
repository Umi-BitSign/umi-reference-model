"""Independent local-context arithmetic and bounded, gap-aware identity tests."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from c5_temporal_memory_adapter import WIDTH, TemporalMemoryAdapter


@pytest.fixture
def memory():
    generator = np.random.default_rng(113)
    value = torch.from_numpy(generator.normal(size=(1, 7, WIDTH)).astype(np.float32))
    value[0, 0, 0] = -0.0
    return value, torch.tensor([True, True, False, True, True, True])


def nonzero(adapter):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(116)
        with torch.no_grad():
            adapter.output_projection.weight.normal_(0, 0.1)
            adapter.temporal_weights.copy_(torch.randn_like(adapter.temporal_weights))


def test_initial_memory_is_bitwise_identity_and_does_not_share_input_storage(memory):
    value, valid = memory
    adapter = TemporalMemoryAdapter().eval()
    with torch.no_grad():
        result = adapter(value, valid_intervals=valid)
    assert torch.equal(result["memory"].view(torch.int32), value.view(torch.int32))
    assert result["memory"].data_ptr() != value.data_ptr()
    assert result["normalized_residual_penalty"].item() == 0
    assert adapter.configuration()["trainable_parameters"] == 24600
    assert sum(p.numel() for p in adapter.parameters()) == 24600


def test_seed_is_reproducible_and_preserves_caller_rng_state():
    before = torch.random.get_rng_state().clone()
    first = TemporalMemoryAdapter(seed=3)
    assert torch.equal(torch.random.get_rng_state(), before)
    second = TemporalMemoryAdapter(seed=3)
    assert all(
        torch.equal(value, second.state_dict()[name]) for name, value in first.state_dict().items()
    )
    third = TemporalMemoryAdapter(seed=4)
    assert not torch.equal(first.input_projection.weight, third.input_projection.weight)


def test_initial_output_projection_has_real_gradient_without_upstream_memory_gradient(memory):
    value, valid = memory
    adapter = TemporalMemoryAdapter()
    result = adapter(value, valid_intervals=valid)
    target = torch.linspace(-1, 1, WIDTH)[None, None, :]
    (result["memory"] * target).mean().backward()
    assert result["memory"].requires_grad
    assert adapter.output_projection.weight.grad.norm() > 0
    assert torch.count_nonzero(adapter.input_projection.weight.grad) == 0
    assert torch.count_nonzero(adapter.temporal_weights.grad) == 0
    assert value.grad is None


def test_nonzero_forward_agrees_with_independent_numpy_reference(memory):
    value, valid = memory
    adapter = TemporalMemoryAdapter(rank=3)
    nonzero(adapter)
    actual = adapter(value, valid_intervals=valid)
    x = value.numpy().astype(np.float64)[0]
    norm = (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + 1e-5)
    pre = norm @ adapter.input_projection.weight.detach().numpy().T
    hidden = 0.5 * pre * (1 + np.vectorize(math.erf)(pre / math.sqrt(2)))
    weights = adapter.temporal_weights.detach().numpy()
    context = np.zeros_like(hidden)
    for t in range(len(x)):
        context[t] += weights[:, 1] * hidden[t]
        if t > 0 and valid[t - 1]:
            context[t] += weights[:, 0] * hidden[t - 1]
        if t + 1 < len(x) and valid[t]:
            context[t] += weights[:, 2] * hidden[t + 1]
    scale = np.maximum(np.sqrt(np.square(x).mean(-1, keepdims=True)), 1e-6)
    delta = 0.1 * scale * np.tanh(context @ adapter.output_projection.weight.detach().numpy().T)
    np.testing.assert_allclose(
        actual["memory"].detach().numpy()[0], x + delta, rtol=1e-6, atol=3e-7
    )
    assert float(actual["normalized_residual_penalty"]) == pytest.approx(
        float(np.square(delta / scale).mean()), rel=1e-6
    )


def test_unknown_interval_blocks_added_neighbor_mixing(memory):
    value, valid = memory
    adapter = TemporalMemoryAdapter()
    nonzero(adapter)
    changed = value.clone()
    changed[:, 3:] *= -7
    first = adapter(value, valid_intervals=valid)["memory"]
    second = adapter(changed, valid_intervals=valid)["memory"]
    assert torch.equal(first[:, :3], second[:, :3])
    open_gap = valid.clone()
    open_gap[2] = True
    assert not torch.equal(first[:, 2], adapter(changed, valid_intervals=open_gap)["memory"][:, 2])


def test_residual_is_bounded_for_every_original_frame_and_all_parameters_can_learn(memory):
    value, valid = memory
    adapter = TemporalMemoryAdapter()
    nonzero(adapter)
    result = adapter(value, valid_intervals=valid)
    scale = value.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
    assert ((result["memory"] - value).abs() <= 0.1 * scale + 3e-7).all()
    assert 0 <= float(result["normalized_residual_penalty"]) <= 0.01 + 1e-7
    result["memory"].square().mean().backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm() > 0
        for p in adapter.parameters()
    )


def test_single_frame_and_all_unknown_intervals_keep_complete_coverage(memory):
    value, _ = memory
    adapter = TemporalMemoryAdapter()
    nonzero(adapter)
    for x in (value, value[:, :1]):
        valid = torch.zeros(x.shape[1] - 1, dtype=torch.bool)
        actual = adapter(x, valid_intervals=valid)
        assert actual["memory"].shape == x.shape and torch.isfinite(actual["memory"]).all()


@pytest.mark.parametrize(
    "change", ["dtype", "shape", "upstream_gradient", "mask", "nonfinite", "parameter_dtype"]
)
def test_invalid_memory_or_parameter_scope_is_rejected_before_adaptation(memory, change):
    value, valid = memory
    adapter = TemporalMemoryAdapter()
    if change == "dtype":
        value = value.double()
    elif change == "shape":
        value = value[0]
    elif change == "upstream_gradient":
        value = value.clone().requires_grad_(True)
    elif change == "mask":
        valid = valid.float()
    elif change == "nonfinite":
        value = value.clone()
        value[0, 0, 0] = float("nan")
    else:
        adapter.double()
    with pytest.raises(ValueError):
        adapter(value, valid_intervals=valid)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rank": 0},
        {"rank": True},
        {"maximum_relative_residual": 0},
        {"maximum_relative_residual": float("nan")},
        {"seed": -1},
    ],
)
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        TemporalMemoryAdapter(**kwargs)
