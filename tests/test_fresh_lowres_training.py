from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn
from transformers import T5Config, T5ForConditionalGeneration

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from fresh_lowres_model import (
    DiscardedTextEncoder,
    FreshLowResolutionTranslator,
    discard_unused_text_encoder,
)
from fresh_lowres_training import (
    accumulation_group,
    allowed_cumulative_state_keys,
    configure_alignment_stage,
    configure_decoder_stage,
    configure_encoder_alignment_stage,
    configure_scratch_stage,
    configure_visual_alignment_stage,
    ctc_alignment_is_feasible,
    ctc_alignment_loss,
    ctc_token_logits,
    ordered_examples,
    repeat_source_examples,
    semantic_alignment_loss,
    training_schedule,
)


def test_ctc_alignment_loss_is_finite_and_updates_video_memory() -> None:
    memory = torch.randn(2, 6, 8, requires_grad=True)
    mask = torch.tensor(
        [[True, True, True, True, True, True], [True, True, True, True, False, False]]
    )
    labels = torch.tensor([[3, 4, 5, 1], [6, 7, 1, 0]])
    embeddings = torch.randn(16, 8)
    loss = ctc_alignment_loss(
        memory,
        mask,
        labels,
        embeddings,
        pad_token_id=0,
        eos_token_id=1,
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert memory.grad is not None
    assert torch.isfinite(memory.grad).all()


def test_ctc_token_logits_use_cosine_similarity() -> None:
    memory = torch.tensor([[[2.0, 0.0], [0.0, 3.0]]])
    embeddings = torch.tensor([[4.0, 0.0], [0.0, 5.0]])
    logits = ctc_token_logits(memory, embeddings)
    assert logits.shape == (1, 2, 2)
    assert torch.equal(logits.argmax(dim=-1), torch.tensor([[0, 1]]))


def test_ctc_feasibility_accounts_for_repeated_tokens() -> None:
    mask = torch.ones(1, 4, dtype=torch.bool)
    assert ctc_alignment_is_feasible(
        mask,
        torch.tensor([[3, 4, 1]]),
        pad_token_id=0,
        eos_token_id=1,
    )
    assert not ctc_alignment_is_feasible(
        mask,
        torch.tensor([[3, 3, 4, 4, 1]]),
        pad_token_id=0,
        eos_token_id=1,
    )


def test_semantic_alignment_prefers_the_matching_text() -> None:
    memory = torch.tensor([[[2.0, 0.0], [4.0, 0.0]]], requires_grad=True)
    mask = torch.ones(1, 2, dtype=torch.bool)
    matching_first = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    matching_second = matching_first.flip(0)
    good = semantic_alignment_loss(memory, mask, matching_first)
    bad = semantic_alignment_loss(memory, mask, matching_second)
    assert good < bad
    good.backward()
    assert memory.grad is not None


def test_training_schedule_repeats_the_frozen_order() -> None:
    rows = [{"example_id": "a"}, {"example_id": "b"}]
    assert training_schedule(rows, 2) == [
        (1, 1, rows[0]),
        (1, 2, rows[1]),
        (2, 1, rows[0]),
        (2, 2, rows[1]),
    ]


def test_accumulation_group_handles_partial_final_batch() -> None:
    assert accumulation_group(1, 10, 4) == (1, 4, False)
    assert accumulation_group(4, 10, 4) == (1, 4, True)
    assert accumulation_group(8, 10, 4) == (2, 4, True)
    assert accumulation_group(9, 10, 4) == (3, 2, False)
    assert accumulation_group(10, 10, 4) == (3, 2, True)


class _FrameEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Linear(3, 8)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return self.layer(frames.mean(dim=(-2, -1)))


class _MobileFrameEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(3, 8),
            nn.Linear(8, 8),
            nn.Linear(8, 8),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return self.features(frames.mean(dim=(-2, -1)))


def _model(use_language_encoder: bool = False) -> FreshLowResolutionTranslator:
    config = T5Config(
        vocab_size=384,
        d_model=32,
        d_ff=64,
        num_layers=1,
        num_decoder_layers=1,
        num_heads=2,
        d_kv=16,
        decoder_start_token_id=0,
        pad_token_id=0,
        eos_token_id=1,
    )
    language_model = T5ForConditionalGeneration(config)
    if not use_language_encoder:
        assert discard_unused_text_encoder(language_model) > 0
    return FreshLowResolutionTranslator(
        language_model,
        _FrameEncoder(),
        8,
        temporal_width=16,
        temporal_layers=1,
        temporal_heads=2,
        temporal_feedforward_width=32,
        max_frames=8,
        dropout=0.0,
        use_language_encoder=use_language_encoder,
    )


def test_discarded_text_encoder_still_allows_visual_generation() -> None:
    model = _model().eval()
    tokens = model.generate(
        torch.rand(1, 3, 3, 8, 8),
        torch.ones(1, 3, dtype=torch.bool),
        max_new_tokens=2,
    )
    assert tokens.shape[0] == 1
    assert isinstance(model.language_model.encoder, DiscardedTextEncoder)


def test_alignment_stage_trains_new_layers_and_cross_attention_only() -> None:
    model = _model()
    trainable = configure_alignment_stage(model)
    assert trainable
    assert not any(name.startswith("frame_encoder.") for name in trainable)
    assert not any(name.startswith("language_model.shared.") for name in trainable)
    assert any("EncDecAttention.q.weight" in name for name in trainable)
    assert any(name.startswith("temporal_encoder.") for name in trainable)
    assert set(trainable) == {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }


def test_scratch_stage_trains_every_retained_parameter() -> None:
    model = _model()
    trainable = configure_scratch_stage(model)
    assert set(trainable) == dict(model.named_parameters()).keys()
    assert all(parameter.requires_grad for parameter in trainable.values())


def test_decoder_stage_trains_decoder_but_keeps_visual_backbone_frozen() -> None:
    model = _model()
    trainable = configure_decoder_stage(model)
    assert not any(name.startswith("frame_encoder.") for name in trainable)
    assert any(name.startswith("language_model.shared.") for name in trainable)
    assert any(name.startswith("language_model.decoder.") for name in trainable)
    assert model.language_model.lm_head.weight.requires_grad


def test_encoder_alignment_preserves_decoder_language_weights() -> None:
    model = _model(use_language_encoder=True)
    trainable = configure_encoder_alignment_stage(model)
    assert any(name.startswith("language_model.encoder.") for name in trainable)
    assert any("EncDecAttention" in name for name in trainable)
    assert not any(name.startswith("frame_encoder.") for name in trainable)
    assert not any(name.startswith("language_model.shared.") for name in trainable)
    assert not model.language_model.decoder.block[0].layer[0].SelfAttention.q.weight.requires_grad


def test_visual_alignment_trains_only_mobilenet_suffix_and_alignment() -> None:
    model = _model()
    model.frame_encoder = _MobileFrameEncoder()
    trainable = configure_visual_alignment_stage(model, trainable_blocks=2)
    visual_names = [name for name in trainable if name.startswith("frame_encoder.features.")]
    assert visual_names
    assert all(
        name.startswith(("frame_encoder.features.1.", "frame_encoder.features.2."))
        for name in visual_names
    )
    assert any("EncDecAttention" in name for name in trainable)
    assert not any(name.startswith("language_model.encoder.") for name in trainable)
    assert not model.language_model.decoder.block[0].layer[0].SelfAttention.q.weight.requires_grad


def test_cumulative_state_allows_exact_stage_unions_only() -> None:
    scopes = {
        "alignment": {"projection", "cross_attention"},
        "decoder": {"projection", "cross_attention", "decoder"},
        "encoder_alignment": {"projection", "cross_attention", "encoder"},
    }
    allowed = allowed_cumulative_state_keys(scopes)
    assert frozenset(scopes["decoder"]) in allowed
    assert frozenset(
        scopes["decoder"] | scopes["encoder_alignment"]
    ) in allowed
    assert frozenset({"projection", "decoder"}) not in allowed
    assert frozenset({"projection", "cross_attention", "frame_encoder"}) not in allowed


def test_cumulative_state_rejects_empty_scopes() -> None:
    try:
        allowed_cumulative_state_keys({"alignment": set()})
    except ValueError as error:
        assert str(error) == "training stages have no parameter scopes"
    else:
        raise AssertionError("empty training stage scopes were accepted")


def test_training_order_is_seeded_without_mutating_inventory() -> None:
    rows = [{"example_id": str(index)} for index in range(10)]
    original = list(rows)
    assert ordered_examples(rows, 17) == ordered_examples(rows, 17)
    assert ordered_examples(rows, 17) != ordered_examples(rows, 18)
    assert rows == original


def test_source_repeats_are_deterministic_and_interleaved() -> None:
    rows = [
        {"source": "how2sign", "example_id": "h1"},
        {"source": "fleurs", "example_id": "f1"},
        {"source": "how2sign", "example_id": "h2"},
        {"source": "fleurs", "example_id": "f2"},
    ]
    original = list(rows)
    expanded = repeat_source_examples(rows, {"fleurs": 3}, 18)
    assert expanded == repeat_source_examples(rows, {"fleurs": 3}, 18)
    assert len(expanded) == 8
    assert sum(row["source"] == "fleurs" for row in expanded) == 6
    assert expanded != [row for row in rows for _ in range(3 if row["source"] == "fleurs" else 1)]
    assert rows == original


def test_source_repeats_preserve_order_at_one() -> None:
    rows = [{"source": "fleurs", "example_id": "f1"}]
    assert repeat_source_examples(rows, {"fleurs": 1}, 18) == rows
