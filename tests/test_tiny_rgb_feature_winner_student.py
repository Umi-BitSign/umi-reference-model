from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import T5Config
from transformers.modeling_outputs import BaseModelOutput

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from c5_winner_language_decoder import WinnerLanguageDecoder
from tiny_rgb_feature_winner_student import RawFeatureWinnerStudent
from tiny_rgb_visual_teacher import STREAM_WIDTHS
from tiny_rgb_winner_student import student_objective


class SemanticEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(sum(STREAM_WIDTHS.values()), 32)
        self.config = SimpleNamespace(d_model=32)
        self.lengths = []

    def forward(self, *, return_dict, **features):
        assert return_dict
        value = torch.cat([features[name] for name in STREAM_WIDTHS], -1)
        self.lengths.append(value.shape[1])
        return BaseModelOutput(last_hidden_state=self.projection(value))


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


def test_fresh_larger_appearance_keeps_both_supplied_backbones_unchanged():
    encoder, decoder = SemanticEncoder(), language()
    before = [
        {key: value.clone() for key, value in module.state_dict().items()}
        for module in (encoder, decoder)
    ]
    student = RawFeatureWinnerStudent(encoder, decoder).train()
    assert len(student.trainable_parameters()) == 90
    assert sum(value.numel() for value in student.trainable_parameters().values()) == 17526750
    assert student.vision.config.width == 512 and student.vision.config.encoder_layers == 6
    assert student.vision.config.frame_chunk == 16
    for module, state in zip((encoder, decoder), before, strict=True):
        assert not module.training and all(not p.requires_grad for p in module.parameters())
        assert all(torch.equal(value, module.state_dict()[key]) for key, value in state.items())


def test_gradients_flow_through_fixed_semantic_and_language_weights_to_custom_pixels():
    student = RawFeatureWinnerStudent(SemanticEncoder(), language(), capacity="small").eval()
    mask = torch.tensor([[True, True, False]])
    output = student(torch.randn(1, 3, 3, 32, 48), mask, torch.tensor([[4, 7, 1]]))
    targets = {name: torch.randn(1, 3, width) for name, width in STREAM_WIDTHS.items()}
    with torch.no_grad():
        latent = student.semantic_memory(targets, mask)
    student_objective(output, latent, targets)["loss"].backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
        for p in student.trainable_parameters().values()
    )
    assert all(
        p.grad is None
        for module in (student.semantic_encoder, student.language)
        for p in module.parameters()
    )
    assert student.semantic_encoder.lengths == [2, 2]
    assert output["memory"].shape == (1, 3, 32) and not output["memory"][:, 2:].any()


def test_native_sequence_encoder_never_sees_bucket_padding_and_preserves_real_outputs():
    student = RawFeatureWinnerStudent(SemanticEncoder(), language(), capacity="small").eval()
    video = torch.randn(1, 3, 3, 32, 48)
    with torch.no_grad():
        plain, _ = student.encode(video[:, :2], torch.ones(1, 2, dtype=torch.bool))
        padded, _ = student.encode(video, torch.tensor([[True, True, False]]))
    assert student.semantic_encoder.lengths == [2, 2]
    assert torch.allclose(plain, padded[:, :2], atol=1e-5, rtol=1e-5)


def test_raw_generation_needs_no_teacher_features_or_caption_reference():
    student = RawFeatureWinnerStudent(SemanticEncoder(), language(), capacity="small").eval()
    tokens = student.generate(
        torch.randn(1, 2, 3, 32, 48),
        torch.ones(1, 2, dtype=torch.bool),
        max_new_tokens=4,
        num_beams=2,
        do_sample=False,
    )
    assert tokens.shape[0] == 1 and student.semantic_encoder.lengths == [2]


def test_semantic_encoder_rejects_interior_gaps_or_ragged_multiclip_batch():
    student = RawFeatureWinnerStudent(SemanticEncoder(), language(), capacity="small")
    features = {name: torch.randn(1, 3, width) for name, width in STREAM_WIDTHS.items()}
    for mask in (torch.tensor([[True, False, True]]), torch.ones(2, 3, dtype=torch.bool)):
        with pytest.raises(ValueError, match="native feature winner"):
            student.semantic_memory(features, mask)
