"""The STEM projection-frozen trainer exposes only LoRA tensors to its optimizer."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from interpolate_visual_adapter import DIRECT_NAMES
from train_c5_stem_projection_frozen import freeze_direct_parameters


def test_freezes_all_direct_parameters():
    selected = {name: torch.nn.Parameter(torch.ones(1)) for name in DIRECT_NAMES}
    selected["layer.lora_a.weight"] = torch.nn.Parameter(torch.ones(1, 1))
    trainable = freeze_direct_parameters(selected)
    assert set(trainable) == {"layer.lora_a.weight"}
    assert all(not selected[name].requires_grad for name in DIRECT_NAMES)
    assert trainable["layer.lora_a.weight"].requires_grad
