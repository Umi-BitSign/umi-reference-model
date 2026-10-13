"""Projection ablation restores only the three baseline visual parameters."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from ablate_visual_projection import projection_frozen_weights
from interpolate_visual_adapter import DIRECT_NAMES


def test_restores_baseline_projection_but_keeps_trained_lora():
    baseline = {name: torch.full((2,), 2.0) for name in DIRECT_NAMES}
    trained = {name: torch.full((2,), 4.0) for name in DIRECT_NAMES}
    trained["layer.q.lora_a.weight"] = torch.tensor([[3.0]])
    trained["layer.q.lora_b.weight"] = torch.tensor([[6.0]])
    output = projection_frozen_weights(trained, baseline)
    assert all(torch.equal(output[name], baseline[name]) for name in DIRECT_NAMES)
    assert torch.equal(output["layer.q.lora_a.weight"], trained["layer.q.lora_a.weight"])
    assert torch.equal(output["layer.q.lora_b.weight"], trained["layer.q.lora_b.weight"])
    trained.pop("layer.q.lora_a.weight")
    with pytest.raises(ValueError, match="topology"):
        projection_frozen_weights(trained, baseline)
