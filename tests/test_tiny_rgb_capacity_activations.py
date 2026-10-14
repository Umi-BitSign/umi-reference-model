from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from tiny_rgb_capacity_activations import NativeForwardExactGelu, replace_capacity_gelu


@pytest.mark.parametrize("shape", [(2, 3, 7), (2, 3, 5, 7)])
def test_native_forward_is_exact_and_strided_gradients_match(shape):
    torch.manual_seed(317)
    value = torch.randn(shape).transpose(-1, -2).requires_grad_()
    reference = value.detach().clone().requires_grad_()
    weight = torch.randn_like(value)
    output = NativeForwardExactGelu()(value)
    expected = F.gelu(reference, approximate="none")
    assert torch.equal(output, expected)
    (output * weight).sum().backward()
    (expected * weight).sum().backward()
    assert torch.allclose(value.grad, reference.grad, atol=2e-7, rtol=2e-5)


def test_double_precision_first_and_second_derivatives_are_correct():
    value = torch.tensor(
        [[-2.0, -0.7, 0.0], [0.5, 1.0, 3.0]], dtype=torch.float64, requires_grad=True
    )
    operation = NativeForwardExactGelu()
    assert torch.autograd.gradcheck(operation, (value,))
    assert torch.autograd.gradgradcheck(operation, (value,))


def test_replacement_keeps_weights_keys_and_native_forward():
    torch.manual_seed(317)
    model = nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 4), nn.GELU()).eval()
    inputs = torch.randn(3, 4)
    expected = model(inputs)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    assert replace_capacity_gelu(model) == 2
    assert torch.equal(model(inputs), expected)
    assert model.state_dict().keys() == before.keys()
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in before.items())


def test_approximate_activation_is_rejected():
    with pytest.raises(ValueError, match="exact"):
        replace_capacity_gelu(nn.Sequential(nn.GELU(approximate="tanh")))


def test_noncontiguous_double_second_derivatives_are_correct():
    value = (
        torch.tensor([[-2.0, -0.7, 0.0], [0.5, 1.0, 3.0]], dtype=torch.float64)
        .transpose(0, 1)
        .requires_grad_()
    )
    assert not value.is_contiguous()
    assert torch.autograd.gradgradcheck(NativeForwardExactGelu(), (value,))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="native Metal required")
def test_channels_last_metal_forward_and_backward_match_native_cpu():
    torch.manual_seed(317)
    value = torch.randn(2, 16, 8, 12).contiguous(memory_format=torch.channels_last)
    weight = torch.randn_like(value)
    cpu = value.clone().requires_grad_()
    reference = F.gelu(cpu, approximate="none")
    (reference * weight).sum().backward()
    metal = value.to("mps").requires_grad_()
    actual = NativeForwardExactGelu()(metal)
    (actual * weight.to("mps")).sum().backward()
    assert float((actual.detach().cpu() - reference.detach()).abs().max()) < 1e-6
    relative = (
        metal.grad.detach().cpu().double() - cpu.grad.double()
    ).norm() / cpu.grad.double().norm()
    assert float(relative) < 1e-5
