from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from transformers import T5Config

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from c5_winner_language_decoder import WinnerLanguageDecoder
from tiny_rgb_visual_teacher import STREAM_WIDTHS
from tiny_rgb_winner_student import RawRgbWinnerStudent, student_objective


def language():
    return WinnerLanguageDecoder(
        T5Config(
            vocab_size=32,
            d_model=32,
            d_ff=64,
            num_layers=1,
            num_decoder_layers=1,
            num_heads=4,
            d_kv=8,
            decoder_start_token_id=0,
            eos_token_id=1,
            pad_token_id=0,
        )
    )


def test_raw_student_preserves_supplied_language_and_complete_frame_geometry():
    decoder = language()
    before = {name: value.clone() for name, value in decoder.state_dict().items()}
    student = RawRgbWinnerStudent(decoder).train()
    assert not decoder.training and all(not p.requires_grad for p in decoder.parameters())
    assert all(torch.equal(value, decoder.state_dict()[name]) for name, value in before.items())
    assert len(student.trainable_parameters()) == 53
    student.eval()
    video = torch.randn(1, 3, 3, 32, 48)
    original = video.clone()
    mask = torch.tensor([[True, True, False]])
    memory, actual = student.encode(video, mask)
    assert memory.shape == (1, 3, 32) and torch.equal(mask, actual)
    assert torch.equal(video, original)
    changed, _ = student.encode(-video, mask)
    assert not torch.equal(changed, memory)


def test_human_and_semantic_objective_updates_all_visual_bridge_heads_but_not_language_or_teacher():
    student = RawRgbWinnerStudent(language()).eval()
    mask = torch.tensor([[True, True, False]])
    result = student(torch.randn(1, 3, 3, 32, 48), mask, torch.tensor([[4, 7, 1]]))
    latent = torch.randn_like(result["memory"], requires_grad=True)
    targets = {
        name: torch.randn(1, 3, width, requires_grad=True) for name, width in STREAM_WIDTHS.items()
    }
    objective = student_objective(result, latent, targets)
    objective["loss"].backward()
    assert latent.grad is None and all(value.grad is None for value in targets.values())
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
        for p in student.trainable_parameters().values()
    )
    assert all(parameter.grad is None for parameter in student.language.parameters())
    with pytest.raises(ValueError, match="latent frame"):
        student_objective(result, latent[:, :1], targets)


def test_generation_uses_only_raw_student_and_frozen_language():
    student = RawRgbWinnerStudent(language()).eval()
    tokens = student.generate(
        torch.randn(1, 2, 3, 32, 48),
        torch.ones(1, 2, dtype=torch.bool),
        max_new_tokens=4,
        num_beams=2,
        do_sample=False,
    )
    assert tokens.shape[0] == 1
