from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from c5_winner_mps_safe import portable_fairseq_gelu, replace_fairseq_exact_gelu


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
