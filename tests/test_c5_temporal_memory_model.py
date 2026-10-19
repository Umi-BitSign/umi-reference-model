"""Frozen-backbone adapter integration on diverse tiny memory; no real C5 model."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from c5_monotone_feature_timing import WIDTHS
from c5_temporal_memory_adapter import TemporalMemoryAdapter
from c5_temporal_memory_model import TemporalMemoryStudent, load_student
from c5_timing_semantic_model import clip_input
from test_c5_timing_semantic_model import TinyLanguage


class DiverseEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.7))
        channel = torch.linspace(-1, 1, 1536)
        channel[0] = 0
        self.register_buffer("channel", channel)

    def forward(self, *, return_dict, **features):
        assert return_dict
        combined = sum(value.mean(-1, keepdim=True) for value in features.values())
        return SimpleNamespace(last_hidden_state=combined * self.channel * self.scale)


@pytest.fixture
def fixture():
    generator = np.random.default_rng(173)
    values = {
        "streams": {
            name: (generator.normal(size=(7, width)) * np.linspace(0.2, 2, 7)[:, None]).astype(
                np.float32
            )
            for name, width in WIDTHS.items()
        },
        "valid_intervals": np.array([1, 1, 0, 1, 1, 1], dtype=np.bool_),
    }
    return (
        TemporalMemoryStudent(DiverseEncoder(), TinyLanguage()).eval(),
        clip_input(values),
        torch.ones((1, 7), dtype=torch.bool),
    )


def test_initial_full_memory_logits_and_loss_are_original_inference_identity(fixture):
    model, clip, mask = fixture
    labels = torch.tensor([[2, 3, 1]])
    with torch.no_grad():
        direct = model.semantic_memory({k: v[None] for k, v in clip.streams.items()}, mask)
        details = model.encode_details(clip, mask)
        from transformers.modeling_outputs import BaseModelOutput

        original = model.language(
            encoder_outputs=BaseModelOutput(last_hidden_state=direct),
            attention_mask=mask.long(),
            labels=labels,
            use_cache=False,
            return_dict=True,
        )
        adapted = model(clip, mask, labels)["language"]
    assert torch.equal(details["memory"].view(torch.int32), direct.view(torch.int32))
    assert torch.equal(adapted.logits.view(torch.int32), original.logits.view(torch.int32))
    assert torch.equal(adapted.loss.view(torch.int32), original.loss.view(torch.int32))
    assert not details["base_memory"].requires_grad and details["memory"].shape == (1, 7, 1536)


def test_caption_gradient_and_update_only_reach_the_declared24600_parameters(fixture):
    model, clip, mask = fixture
    model.train()
    assert not model.semantic_encoder.training and not model.language.training
    assert model.memory_adapter.training
    before = {k: v.clone() for k, v in model.state_dict().items()}
    selected = model.trainable_parameters()
    assert list(selected) == sorted(selected)
    assert sum(p.numel() for p in selected.values()) == 24600
    optimizer = torch.optim.SGD(selected.values(), lr=0.1)
    details = model(clip, mask, torch.tensor([[2, 3, 1]]))
    details["language"].loss.backward()
    assert details["memory"].requires_grad and not details["base_memory"].requires_grad
    assert model.memory_adapter.output_projection.weight.grad.norm() > 0
    assert all(
        p.grad is None
        for module in (model.semantic_encoder, model.language)
        for p in module.parameters()
    )
    optimizer.step()
    assert any(
        not torch.equal(v, before[k])
        for k, v in model.state_dict().items()
        if k.startswith("memory_adapter.")
    )
    assert all(
        torch.equal(v, before[k])
        for k, v in model.state_dict().items()
        if not k.startswith("memory_adapter.")
    )
    model.require_fixed()


def test_generation_preserves_original_complete_memory_and_all_generation_options(fixture):
    model, clip, mask = fixture
    options = {
        "num_beams": 5,
        "max_length": 2048,
        "length_penalty": 0.6,
        "no_repeat_ngram_size": 16,
        "do_sample": False,
        "early_stopping": True,
        "pad_token_id": 0,
        "eos_token_id": 1,
    }
    assert model.generate(clip, mask, **options).tolist() == [[1, 2, 3]]
    memory, actual_mask, actual_options = model.language.last_generate
    assert memory.shape == (1, 7, 1536) and torch.equal(actual_mask, mask.long())
    assert actual_options == options


@pytest.mark.parametrize("change", ["mask", "prefix_padding", "input_gradient", "backbone_scope"])
def test_no_original_frame_or_backbone_scope_can_be_silently_changed(fixture, change):
    model, clip, mask = fixture
    if change == "mask":
        mask = mask.long()
    elif change == "prefix_padding":
        mask[0, -1] = False
    elif change == "input_gradient":
        clip.streams["pose_features"].requires_grad_(True)
    else:
        model.language.scale.requires_grad_(True)
    with pytest.raises(ValueError):
        model.encode_details(clip, mask)


def test_original_encoder_remains_outside_graph_even_without_visible_hand_intervals(fixture):
    model, clip, mask = fixture
    clip.valid_intervals.zero_()
    details = model(clip, mask, torch.tensor([[2, 1]]))
    details["language"].loss.backward()
    assert details["memory"].shape == (1, 7, 1536)
    assert model.memory_adapter.output_projection.weight.grad.norm() > 0
    assert model.semantic_encoder.scale.grad is None


def test_incomplete_configuration_cannot_release_any_public_model_loading(monkeypatch):
    import evaluate_c5_words as words

    monkeypatch.setattr(
        words, "models", lambda *args: pytest.fail("C5 loaded before configuration admission")
    )
    configuration = TemporalMemoryAdapter().configuration()
    configuration["trainable_parameters"] += 1
    with pytest.raises(ValueError, match="configuration changed"):
        load_student({}, configuration)
