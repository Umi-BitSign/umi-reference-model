from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from tiny_rgb_capacity_positions import CanonicalPositionsVision
from tiny_rgb_language_model import ScratchRgbVision
from tiny_rgb_model import TinyRgbConfig, positions


def test_cpu_positions_match_original_vision_without_new_parameters_or_state_keys():
    torch.manual_seed(317)
    config = TinyRgbConfig(width=32, heads=4, encoder_layers=1, decoder_layers=1, dropout=0.0)
    reference = ScratchRgbVision(config).eval()
    candidate = CanonicalPositionsVision(config).eval()
    candidate.load_state_dict(reference.state_dict(), strict=True)
    assert candidate.state_dict().keys() == reference.state_dict().keys()
    assert sum(p.numel() for p in candidate.parameters()) == sum(
        p.numel() for p in reference.parameters()
    )
    assert torch.equal(candidate.fixed_positions[:20], positions(20, 32, "cpu"))
    video = torch.randn(1, 5, 3, 32, 48)
    mask = torch.tensor([[True, True, True, False, False]])
    expected, expected_mask = reference(video, mask)
    actual, actual_mask = candidate(video, mask)
    assert torch.equal(actual, expected)
    assert torch.equal(actual_mask, expected_mask)


def test_fixed_positions_follow_device_without_becoming_learned_or_serialized():
    config = TinyRgbConfig(width=32, heads=4, encoder_layers=1, decoder_layers=1)
    model = CanonicalPositionsVision(config)
    assert not model.fixed_positions.requires_grad
    assert "fixed_positions" not in model.state_dict()
    if torch.backends.mps.is_available():
        before = model.fixed_positions.clone()
        model.to("mps")
        assert model.fixed_positions.device.type == "mps"
        assert torch.equal(model.fixed_positions.cpu(), before)
