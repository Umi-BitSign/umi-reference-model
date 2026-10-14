from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from c5_winner_mps_safe import (
    PortableLayerNorm,
    portable_fairseq_gelu,
    replace_fairseq_exact_gelu,
    replace_native_layer_norms,
)


def test_exact_float32_forward_and_strided_backward_match_reference_on_cpu():
    value = torch.linspace(-7, 7, 192).reshape(2, 8, 12).requires_grad_()
    reference = torch.nn.functional.gelu(value)
    result = portable_fairseq_gelu(value)
    assert torch.allclose(reference, result, atol=5e-7, rtol=2e-6)
    gradient = torch.randn(2, 12, 8).transpose(1, 2)
    assert not gradient.is_contiguous()
    expected = torch.autograd.grad(reference, value, gradient)[0]
    actual = torch.autograd.grad(result, value, gradient)[0]
    assert torch.allclose(expected, actual, atol=5e-7, rtol=2e-6)


def test_only_fairseq_exact_activation_is_replaced_without_touching_weights():
    def gelu(value):
        return torch.nn.functional.gelu(value)

    gelu.__module__ = "fairseq.modules.gelu"
    candidate = torch.nn.Linear(4, 4)
    candidate.activation_fn = gelu
    before = {name: value.clone() for name, value in candidate.state_dict().items()}
    assert replace_fairseq_exact_gelu(candidate) == 1
    assert candidate.activation_fn is portable_fairseq_gelu
    assert all(torch.equal(value, before[name]) for name, value in candidate.state_dict().items())
    assert replace_fairseq_exact_gelu(candidate) == 0


def test_centered_layer_norm_keeps_parameters_outputs_and_strided_gradients():
    torch.manual_seed(317)
    reference = torch.nn.LayerNorm(32, eps=1e-5)
    with torch.no_grad():
        reference.weight.copy_(torch.randn(32))
        reference.bias.copy_(torch.randn(32))
    candidate = PortableLayerNorm(reference)
    assert candidate.weight is reference.weight and candidate.bias is reference.bias
    value = torch.randn(2, 7, 32, requires_grad=True)
    gradient = torch.randn(2, 32, 7).transpose(1, 2)
    expected = reference(value)
    actual = candidate(value)
    assert torch.allclose(expected, actual, atol=2e-6, rtol=2e-6)
    parameters = (value, reference.weight, reference.bias)
    before = torch.autograd.grad(expected, parameters, gradient)
    after = torch.autograd.grad(actual, parameters, gradient)
    assert all(
        torch.allclose(x, y, atol=2e-5, rtol=2e-5) for x, y in zip(before, after, strict=True)
    )


def test_replacing_layer_norm_preserves_every_state_key_and_value():
    candidate = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.LayerNorm(4))
    before = {name: value.clone() for name, value in candidate.state_dict().items()}
    assert replace_native_layer_norms(candidate) == 1
    assert set(candidate.state_dict()) == set(before)
    assert all(torch.equal(value, before[name]) for name, value in candidate.state_dict().items())
