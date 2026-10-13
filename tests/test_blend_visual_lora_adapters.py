"""Visual LoRA blends preserve the exact weighted delta."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from blend_visual_lora_adapters import blended_weights
from interpolate_visual_adapter import DIRECT_NAMES


def _adapter(a: torch.Tensor, b: torch.Tensor) -> dict[str, torch.Tensor]:
    values = {name: torch.full((2,), 3.0) for name in DIRECT_NAMES}
    values["layer.q.lora_a.weight"] = a
    values["layer.q.lora_b.weight"] = b
    return values


def test_blend_is_exact_weighted_sum_of_lora_deltas():
    first = _adapter(torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0], [4.0]]))
    second = _adapter(torch.tensor([[5.0, 6.0]]), torch.tensor([[7.0], [8.0]]))
    output = blended_weights(first, second, 0.25)
    actual = output["layer.q.lora_b.weight"] @ output["layer.q.lora_a.weight"]
    expected = 0.75 * (
        first["layer.q.lora_b.weight"] @ first["layer.q.lora_a.weight"]
    ) + 0.25 * (
        second["layer.q.lora_b.weight"] @ second["layer.q.lora_a.weight"]
    )
    assert torch.equal(actual, expected)
    assert all(torch.equal(output[name], first[name]) for name in DIRECT_NAMES)


def test_blend_rejects_mismatched_projection_or_endpoint_weight():
    first = _adapter(torch.ones(1, 2), torch.ones(2, 1))
    second = _adapter(torch.ones(1, 2), torch.ones(2, 1))
    second[next(iter(DIRECT_NAMES))] = torch.zeros(2)
    with pytest.raises(ValueError, match="projection"):
        blended_weights(first, second, 0.5)
    with pytest.raises(ValueError, match="fractional"):
        blended_weights(first, first, 1.0)


def test_blend_can_linearly_interpolate_direct_visual_parameters():
    first = _adapter(torch.ones(1, 2), torch.ones(2, 1))
    second = _adapter(torch.ones(1, 2), torch.ones(2, 1))
    for name in DIRECT_NAMES:
        first[name] = torch.full((2,), 2.0)
        second[name] = torch.full((2,), 6.0)
    output = blended_weights(first, second, 0.25, blend_direct=True)
    assert all(
        torch.equal(output[name], torch.full((2,), 3.0)) for name in DIRECT_NAMES
    )
