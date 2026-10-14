from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from transformers import T5Config, T5ForConditionalGeneration

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from tiny_rgb_capacity import build_capacity_translator, visual_budget, visual_config
from tiny_rgb_model import TinyRgbConfig


def language():
    # Small random language fixture; no pretrained-quality or ASL claim.
    return T5ForConditionalGeneration(
        T5Config(
            vocab_size=32,
            d_model=32,
            d_ff=64,
            num_layers=1,
            num_decoder_layers=1,
            num_heads=4,
            d_kv=8,
            dropout_rate=0.0,
            pad_token_id=0,
            eos_token_id=1,
            decoder_start_token_id=0,
        )
    )


def test_capacity_presets_keep_input_limits_and_token_geometry():
    baseline = TinyRgbConfig()
    assert visual_config("small") == baseline
    for name in ("medium", "large"):
        config = visual_config(name)
        for field in (
            "spatial_queries",
            "frame_chunk",
            "max_frames",
            "max_text",
            "dropout",
            "decoder_layers",
        ):
            assert getattr(config, field) == getattr(baseline, field)


def test_measured_capacity_grows_without_growing_pixel_stem():
    budgets = [visual_budget(name) for name in ("small", "medium", "large")]
    assert budgets[0]["scratch_visual_parameters"] == 986384
    assert 3_500_000 < budgets[1]["scratch_visual_parameters"] < 4_500_000
    assert 16_000_000 < budgets[2]["scratch_visual_parameters"] < 18_000_000
    assert len({row["convolution_stem_parameters"] for row in budgets}) == 1
    for row in budgets:
        assert row["scratch_visual_parameters"] == sum(
            row[key]
            for key in ("convolution_stem_parameters", "spatial_parameters", "temporal_parameters")
        )
        assert row["spatial_tokens_per_frame"] == 4
        assert row["optimizer_updates"] == 0
        assert not row["metal_training_numerics_verified"]
        assert not row["held_out_quality_verified"]


@pytest.mark.parametrize("capacity", ["small", "medium", "large"])
def test_capacity_keeps_supplied_language_weights_and_pixel_conditioning(capacity):
    torch.manual_seed(20261014)
    text_model = language()
    before = {
        key: value.clone()
        for key, value in text_model.state_dict().items()
        if not key.startswith("encoder.")
    }
    model = build_capacity_translator(text_model, capacity).eval()
    assert all(
        torch.equal(value, model.language_model.state_dict()[key]) for key, value in before.items()
    )
    assert not list(model.language_model.encoder.parameters())
    assert not hasattr(model.vision, "decoder")
    assert not hasattr(model.vision, "embedding")
    assert not any(isinstance(module, torch.nn.GELU) for module in model.vision.modules())
    video = torch.randn(1, 3, 3, 32, 48)
    original = video.clone()
    mask = torch.tensor([[True, True, False]])
    labels = torch.tensor([[4, 5, 1]])
    memory, memory_mask = model.encode(video, mask)
    assert memory.shape == (1, 12, 32)
    assert torch.equal(memory_mask, mask.repeat_interleave(4, dim=1))
    assert torch.equal(video, original)
    output = model(video, mask, labels)
    assert torch.isfinite(output.loss)
    assert not torch.allclose(output.logits, model(-video, mask, labels).logits, atol=1e-6)
    output.loss.backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    assert model.vision.stem[0].weight.grad.abs().sum() > 0
    actual_bridge = model.parameter_counts()["scratch_visual_language_bridge"]
    assert (
        actual_bridge == visual_budget(capacity, 32)["scratch_bridge_parameters_at_language_width"]
    )


def test_unknown_capacity_and_nonpositive_language_width_are_rejected():
    with pytest.raises(ValueError, match="unknown visual capacity"):
        visual_config("unbounded")
    with pytest.raises(ValueError, match="positive"):
        visual_budget("small", 0)
