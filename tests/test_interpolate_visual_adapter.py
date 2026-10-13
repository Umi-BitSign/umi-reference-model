"""Half-strength interpolation preserves the baseline visual parameter reference."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from interpolate_visual_adapter import DIRECT_NAMES, interpolated_weights


def _case():
    baseline = {name: torch.full((2,), 2.0) for name in DIRECT_NAMES}
    trained = {name: torch.full((2,), 4.0) for name in DIRECT_NAMES}
    trained["layer.q.lora_a.weight"] = torch.tensor([[3.0]])
    trained["layer.q.lora_b.weight"] = torch.tensor([[6.0]])
    return trained, baseline


def test_midpoint_scales_lora_delta_once_and_interpolates_direct_parameters():
    trained, baseline = _case()
    output = interpolated_weights(trained, baseline, 0.5)
    assert all(torch.equal(output[name], torch.full((2,), 3.0)) for name in DIRECT_NAMES)
    assert torch.equal(output["layer.q.lora_a.weight"], torch.tensor([[3.0]]))
    assert torch.equal(output["layer.q.lora_b.weight"], torch.tensor([[3.0]]))
    assert torch.equal(trained["layer.q.lora_b.weight"], torch.tensor([[6.0]]))


def test_interpolation_rejects_missing_pair_and_nonfinite_weight():
    trained, baseline = _case()
    trained.pop("layer.q.lora_b.weight")
    with pytest.raises(ValueError, match="topology"):
        interpolated_weights(trained, baseline, 0.5)
    trained, baseline = _case()
    trained["encoder.adapter.final_layer.weight"][0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        interpolated_weights(trained, baseline, 0.5)
