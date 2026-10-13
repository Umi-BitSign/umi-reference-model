from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import tiny_rgb_data
import tiny_rgb_full_training_tools
from tiny_rgb_fit import gradient_receipt
from tiny_rgb_full_training_tools import (
    batched_gradient_receipt,
    decode_pixel_delta,
    encode_pixel_delta,
    masked_length_buckets,
)


@pytest.mark.parametrize("cap", (32, 48, 192))
def test_fast_streaming_keeps_exact_rgb_pixels_full_frames_and_aspect(tmp_path, monkeypatch, cap):
    import cv2

    (tmp_path / "data").mkdir()
    video = tmp_path / "data/source.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 25, (112, 64))
    assert writer.isOpened()
    rng = np.random.default_rng(317)
    for _ in range(20):
        writer.write(rng.integers(0, 256, (64, 112, 3), dtype=np.uint8))
    writer.release()
    monkeypatch.setattr(tiny_rgb_data, "ROOT", tmp_path)
    monkeypatch.setattr(tiny_rgb_full_training_tools, "ROOT", tmp_path)
    original = tiny_rgb_data.decode_rgb_video("data/source.mp4", cap)
    optimized = tiny_rgb_full_training_tools.decode_rgb_video_fast("data/source.mp4", cap)
    assert np.array_equal(original[0], optimized[0])
    assert np.array_equal(original[1], optimized[1])
    assert original[2] == optimized[2]
    with pytest.raises(ValueError, match="do not truncate"):
        tiny_rgb_full_training_tools.decode_rgb_video_fast("data/source.mp4", cap, max_frames=4)


def model():
    result = torch.nn.Module()
    result.vision = torch.nn.Linear(4, 4)
    result.visual_projection = torch.nn.Linear(4, 4)
    result.language_model = torch.nn.Linear(4, 4)
    for parameter in result.parameters():
        parameter.grad = torch.full_like(parameter, 0.1)
    return result


def test_batched_gradient_validation_matches_original_per_parameter_guard():
    candidate = model()
    assert batched_gradient_receipt(candidate, torch) == gradient_receipt(candidate, torch)


@pytest.mark.parametrize("bad", ("nan", "inf", "missing", "zero_group"))
def test_batched_gradient_guard_still_rejects_invalid_or_absent_signal(bad):
    candidate = model()
    if bad == "nan":
        candidate.vision.weight.grad[0, 0] = float("nan")
    elif bad == "inf":
        candidate.vision.weight.grad[0, 0] = float("inf")
    elif bad == "missing":
        candidate.vision.weight.grad = None
    else:
        for parameter in candidate.language_model.parameters():
            parameter.grad.zero_()
    with pytest.raises(RuntimeError, match="full RGB"):
        batched_gradient_receipt(candidate, torch)


@pytest.mark.parametrize("kind", ("random", "constant", "wrapping", "strided"))
def test_temporal_pixel_delta_is_lossless_for_every_byte_and_preserves_shape(kind):
    pixels = np.random.default_rng(317).integers(0, 256, (8, 32, 56, 3), dtype=np.uint8)
    if kind == "constant":
        pixels[1:] = pixels[0]
    elif kind == "wrapping":
        pixels[::2] = 255
        pixels[1::2] = 0
    elif kind == "strided":
        pixels = pixels[:, :, ::2]
        assert not pixels.flags.c_contiguous
    restored = decode_pixel_delta(encode_pixel_delta(pixels))
    assert restored.dtype == np.uint8 and restored.shape == pixels.shape
    assert np.array_equal(restored, pixels)


def test_masked_buckets_preserve_all_real_pixels_aspect_and_targets():
    video = torch.randn(1, 17, 3, 32, 56)
    label = torch.tensor([[4, 5, 6, 7, 1]])
    padded, mask, target = masked_length_buckets(video, label, torch)
    assert padded.shape == (1, 32, 3, 32, 56)
    assert torch.equal(padded[:, :17], video)
    assert torch.count_nonzero(padded[:, 17:]) == 0
    assert mask[:, :17].all() and not mask[:, 17:].any()
    assert target.shape == (1, 8)
    assert torch.equal(target[:, :5], label) and (target[:, 5:] == -100).all()
    assert masked_length_buckets(video, None, torch)[2] is None


def test_masked_buckets_do_not_change_valid_logits_or_loss():
    from tiny_rgb_language_model import TinyRgbLanguageTranslator
    from tiny_rgb_model import TinyRgbConfig
    from tiny_rgb_mps_safe import replace_native_gelu
    from transformers import T5Config, T5ForConditionalGeneration

    torch.manual_seed(317)
    text = T5ForConditionalGeneration(
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
    candidate = TinyRgbLanguageTranslator(
        text, TinyRgbConfig(width=32, heads=4, encoder_layers=1, decoder_layers=1, dropout=0.0)
    ).eval()
    replace_native_gelu(candidate)
    video = torch.randn(1, 3, 3, 32, 56)
    label = torch.tensor([[4, 5, 1]])
    padded, mask, target = masked_length_buckets(video, label, torch)
    plain = candidate(video, torch.ones(1, 3, dtype=torch.bool), label)
    bucketed = candidate(padded, mask, target)
    assert torch.allclose(plain.logits, bucketed.logits[:, :3], atol=2e-5, rtol=2e-5)
    assert torch.allclose(plain.loss, bucketed.loss, atol=2e-5, rtol=2e-5)
    plain.loss.backward()
    before = {name: parameter.grad.clone() for name, parameter in candidate.named_parameters()}
    candidate.zero_grad(set_to_none=True)
    bucketed.loss.backward()
    for name, parameter in candidate.named_parameters():
        assert torch.allclose(before[name], parameter.grad, atol=2e-5, rtol=2e-5), name


@pytest.mark.parametrize("frames,targets", ((0, 3), (513, 3), (3, 0), (3, 257)))
def test_masked_buckets_refuse_truncation_or_empty_inputs(frames, targets):
    with pytest.raises(ValueError, match="cannot truncate"):
        masked_length_buckets(torch.zeros(1, frames, 3, 32, 56), torch.ones(1, targets), torch)
