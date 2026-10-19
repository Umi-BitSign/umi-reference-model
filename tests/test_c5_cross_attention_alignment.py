"""Real tiny T5 checks; no public model, media, training corpus or judge."""

import copy
import sys
from pathlib import Path

import pytest
import torch
from transformers import T5Config, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from c5_cross_attention_alignment import CrossAttentionAlignment


def example():
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(563)
        config = T5Config(
            vocab_size=17,
            d_model=12,
            d_kv=4,
            num_heads=2,
            num_layers=1,
            num_decoder_layers=2,
            d_ff=24,
            dropout_rate=0,
            decoder_start_token_id=0,
            eos_token_id=1,
            pad_token_id=0,
        )
        language = T5ForConditionalGeneration(config).requires_grad_(False).eval()
        memory = torch.randn(1, 9, 12)
    return language, memory, torch.ones(1, 9, dtype=torch.bool), torch.tensor([[5, 6, 1]])


def test_zero_initializer_retains_real_t5_logits_generation_original_values_and_rng():
    language, memory, mask, labels = example()
    original = copy.deepcopy(language.state_dict())
    expected = language(
        encoder_outputs=BaseModelOutput(last_hidden_state=memory),
        attention_mask=mask.long(),
        labels=labels,
        use_cache=False,
    ).logits

    def generation():
        # Transformers expands/mutates BaseModelOutput during beam generation.
        return dict(
            encoder_outputs=BaseModelOutput(last_hidden_state=memory),
            attention_mask=mask.long(),
            max_new_tokens=8,
            num_beams=3,
        )

    expected_tokens = language.generate(**generation())
    rng = torch.random.get_rng_state().clone()
    model = CrossAttentionAlignment(language, rank=3)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert set(original) == set(model.frozen_state())
    assert all(torch.equal(v, model.frozen_state()[k]) for k, v in original.items())
    with torch.no_grad():
        assert torch.equal(model(memory, mask, labels).logits, expected)
        assert torch.equal(model.language.generate(**generation()), expected_tokens)
    assert len(model.trainable_parameters()) == 8
    assert model.configuration()["trainable_parameters"] == 240
    assert not model.training


def test_actual_gradient_reaches_all_output_factors_and_original_values_stay_frozen():
    language, memory, mask, labels = example()
    model = CrossAttentionAlignment(language, rank=3)
    original = {k: v.clone() for k, v in model.frozen_state().items()}
    model.train(True)
    assert not any(m.training for m in model.modules())
    model(memory, mask, labels).loss.backward()
    for name, parameter in model.trainable_parameters().items():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        if "output_projection" in name:
            assert parameter.grad.norm() > 0
    model.require_scope()
    assert all(torch.equal(v, model.frozen_state()[k]) for k, v in original.items())


def test_nonzero_adapter_changes_real_t5_and_original_context_restores_exactly_on_error():
    language, memory, mask, labels = example()
    base = language(
        encoder_outputs=BaseModelOutput(last_hidden_state=memory),
        attention_mask=mask.long(),
        labels=labels,
        use_cache=False,
    ).logits
    model = CrossAttentionAlignment(language, rank=3)
    with torch.no_grad():
        for module in model.projections().values():
            module.output_projection.weight.copy_(
                torch.linspace(-0.1, 0.1, module.output_projection.weight.numel()).reshape_as(
                    module.output_projection.weight
                )
            )
        assert not torch.equal(model(memory, mask, labels).logits, base)
        with pytest.raises(RuntimeError, match="test"):
            with model.original_decoder():
                assert torch.equal(model(memory, mask, labels).logits, base)
                with model.original_decoder():
                    assert torch.equal(model(memory, mask, labels).logits, base)
                raise RuntimeError("test")
        assert all(m.enabled for m in model.projections().values())


def test_refuses_partial_mask_or_learning_original_parameters():
    language, memory, mask, labels = example()
    model = CrossAttentionAlignment(language, rank=3)
    mask[0, 4] = False
    with pytest.raises(ValueError, match="complete detached original memory"):
        model(memory, mask, labels)
    model.language.shared.weight.requires_grad_(True)
    with pytest.raises(ValueError, match="scope"):
        model.require_scope()


def test_refuses_double_installation():
    language, *_ = example()
    CrossAttentionAlignment(language, rank=3)
    with pytest.raises(ValueError, match="installed twice"):
        CrossAttentionAlignment(language, rank=3)
