"""Synthetic tiny T5 gradients against a separate direct full-graph objective."""

import copy
import sys
from pathlib import Path

import pytest
import torch
from transformers import T5Config, T5ForConditionalGeneration

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from c5_cross_attention_alignment import CrossAttentionAlignment
from c5_cross_attention_risk import backward_alignment_risk
from c5_retention import teacher_kl
from terra_quality_training import minimum_risk, sequence_log_probability


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


@pytest.mark.parametrize("graded", [False, True])
@pytest.mark.parametrize("nonzero", [False, True])
def test_streamed_gradient_matches_independent_full_graph_and_retains_frozen_values(
    graded, nonzero
):
    language, memory, mask, anchor = example()
    model = CrossAttentionAlignment(language, rank=3)
    if nonzero:
        with torch.no_grad():
            for module in model.projections().values():
                module.output_projection.weight.copy_(
                    torch.linspace(-0.1, 0.1, module.output_projection.weight.numel()).reshape_as(
                        module.output_projection.weight
                    )
                )
    direct = copy.deepcopy(model)
    original = {k: v.clone() for k, v in model.frozen_state().items()}
    labels = [anchor, torch.tensor([[7, 5, 9, 1]]), torch.tensor([[3, 1]])] if graded else [anchor]
    quality = [10, 1, 4] if graded else None
    with torch.no_grad(), direct.original_decoder():
        teacher = direct(memory, mask, anchor).logits.detach()
    outputs = [direct(memory, mask, value) for value in labels]
    caption = outputs[0].loss
    retention = teacher_kl(outputs[0].logits, teacher, anchor, 1.0)
    expected = 0.1 * caption + 0.01 * retention
    if graded:
        scores = torch.stack(
            [
                sequence_log_probability(o.logits, v) / (v != -100).sum()
                for o, v in zip(outputs, labels, strict=True)
            ]
        )
        expected = expected + 1.7 * minimum_risk(scores, quality, alpha=1.0)["loss"]
    (expected / 3).backward()
    result = backward_alignment_risk(
        model, memory, mask, labels, quality, risk_weight=1.7, divisor=3
    )
    assert result["loss"] == pytest.approx(float(expected.detach()), rel=1e-6, abs=1e-7)
    assert result["real_graded_candidates_used"] is graded
    assert (result["expected_terra_error"] is not None) is graded
    for name, parameter in model.trainable_parameters().items():
        actual, reference = parameter.grad, direct.trainable_parameters()[name].grad
        assert actual is not None and torch.isfinite(actual).all()
        assert torch.allclose(actual, reference, rtol=0.005, atol=1e-7)
    assert any(p.grad.norm() > 0 for p in model.trainable_parameters().values())
    assert all(torch.equal(v, model.frozen_state()[k]) for k, v in original.items())
    model.require_scope()


def test_ungraded_data_never_receives_invented_risk_labels():
    language, memory, mask, labels = example()
    model = CrossAttentionAlignment(language, rank=3)
    with pytest.raises(ValueError, match="only its human caption"):
        backward_alignment_risk(model, memory, mask, [labels, labels], None)
    with pytest.raises(ValueError, match="complete real integer rewards"):
        backward_alignment_risk(model, memory, mask, [labels], [9])
