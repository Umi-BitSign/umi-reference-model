from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from tiny_rgb_model import (
    VOCABULARY,
    TinyRgbConfig,
    TinyRgbTranslator,
    decode_text,
    encode_text,
    resize_rgb_clip,
)


def tiny():
    torch.manual_seed(317)
    return TinyRgbTranslator(
        TinyRgbConfig(
            width=32,
            heads=4,
            encoder_layers=1,
            decoder_layers=1,
            max_frames=8,
            max_text=32,
            dropout=0.0,
        )
    )


def test_utf8_byte_codec_round_trip_and_stop_token():
    text = "ASL café"
    assert decode_text(encode_text(text) + encode_text("ignored")) == text


def test_resize_happens_before_network_preserves_aspect_and_never_upscales():
    rgb = np.zeros((2, 240, 420, 3), dtype=np.uint8)
    low, high = resize_rgb_clip(rgb, 128), resize_rgb_clip(rgb, 192)
    assert low.shape == (2, 3, 128, 224)
    assert high.shape == (2, 3, 192, 336)
    assert low.numel() / high.numel() == pytest.approx(4 / 9)
    assert resize_rgb_clip(rgb[:, :24, :40], 128).shape[-2:] == (24, 40)
    observed = []
    model = tiny()
    model.stem[0].register_forward_pre_hook(lambda _, args: observed.append(args[0].shape[-2:]))
    model.encode(low.unsqueeze(0), torch.ones(1, 2, dtype=torch.bool))
    assert observed == [torch.Size((128, 224))]


def test_all_parameters_trainable_no_pretrained_components_and_spatial_tokens_retained():
    model = tiny()
    assert all(parameter.requires_grad for parameter in model.parameters())
    video = torch.randn(1, 3, 3, 32, 56)
    memory, mask = model.encode(video, torch.tensor([[True, True, False]]))
    assert memory.shape == (1, 12, 32)
    assert mask.tolist() == [[True] * 8 + [False] * 4]


def test_finite_gradients_reach_every_visual_and_language_parameter():
    model = tiny()
    video = torch.randn(1, 2, 3, 32, 56)
    targets = torch.tensor([encode_text("test")])
    logits = model(video, torch.ones(1, 2, dtype=torch.bool), targets)
    assert logits.shape == (1, 5, VOCABULARY)
    torch.nn.functional.cross_entropy(logits.flatten(0, 1), targets.flatten()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.stem[0].weight.grad.abs().sum() > 0
    assert model.embedding.weight.grad.abs().sum() > 0


def test_raw_pixels_affect_predictions_and_causal_decoder_cannot_read_future_labels():
    model = tiny().eval()
    mask = torch.ones(1, 2, dtype=torch.bool)
    video = torch.randn(1, 2, 3, 32, 56)
    targets = torch.tensor([encode_text("test")])
    first = model(video, mask, targets)
    assert not torch.allclose(first, model(-video, mask, targets), atol=1e-6)
    modified = targets.clone()
    modified[:, 3] = encode_text("z")[0]
    second = model(video, mask, modified)
    assert torch.allclose(first[:, :4], second[:, :4], atol=1e-6)


def test_resolution_changes_visual_work_but_not_architecture_or_parameter_count():
    model = tiny().eval()
    total = sum(p.numel() for p in model.parameters())
    for height, width in ((32, 56), (48, 84)):
        output = model.encode(
            torch.randn(1, 2, 3, height, width), torch.ones(1, 2, dtype=torch.bool)
        )[0]
        assert output.shape == (1, 8, 32)
        assert sum(p.numel() for p in model.parameters()) == total


def test_generation_limits_and_input_role_shapes_are_checked():
    model = tiny().eval()
    video = torch.randn(1, 2, 3, 32, 56)
    mask = torch.ones(1, 2, dtype=torch.bool)
    tokens = model.generate(video, mask, maximum_tokens=4)
    assert tokens.shape[0] == 1 and 1 <= tokens.shape[1] <= 4
    with pytest.raises(ValueError, match="BTCHW pixels"):
        model.encode(torch.randn(1, 2, 768), mask)
    with pytest.raises(ValueError, match="frame mask"):
        model.encode(video, torch.zeros_like(mask))


def test_padded_frame_pixels_cannot_change_valid_predictions():
    model = tiny().eval()
    video = torch.randn(1, 2, 3, 32, 56)
    mask = torch.tensor([[True, False]])
    targets = torch.tensor([encode_text("test")])
    original = model(video, mask, targets)
    video[:, 1] *= 100
    assert torch.allclose(original, model(video, mask, targets), atol=1e-6)


@pytest.mark.parametrize(
    "config",
    [
        TinyRgbConfig(heads=0),
        TinyRgbConfig(width=0),
        TinyRgbConfig(width=31),
        TinyRgbConfig(encoder_layers=0),
        TinyRgbConfig(dropout=1.0),
    ],
)
def test_invalid_architecture_is_rejected(config):
    with pytest.raises(ValueError, match="invalid tiny RGB architecture"):
        TinyRgbTranslator(config)
