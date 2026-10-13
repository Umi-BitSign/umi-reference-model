from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from tiny_rgb_model import TinyRgbConfig, TinyRgbTranslator
from tiny_rgb_mps_safe import PortableGelu, replace_native_gelu


def test_portable_gelu_matches_exact_cpu_forward_and_strided_backward():
    torch.manual_seed(317)
    left = torch.randn(4, 16, 12, 22).requires_grad_()
    right = left.detach().clone().requires_grad_()
    upstream = torch.randn(4, 12 * 22, 16).transpose(1, 2).reshape(left.shape)
    assert not upstream.is_contiguous()
    original = torch.nn.GELU()(left)
    portable = PortableGelu()(right)
    original.backward(upstream)
    portable.backward(upstream)
    assert torch.allclose(original, portable, atol=1e-6, rtol=1e-6)
    assert torch.allclose(left.grad, right.grad, atol=1e-6, rtol=1e-6)


def test_replacing_gelu_preserves_every_weight_and_parameter_name():
    model = TinyRgbTranslator(TinyRgbConfig(width=32, encoder_layers=1, decoder_layers=1))
    before = {n: t.clone() for n, t in model.state_dict().items()}
    replaced = replace_native_gelu(model)
    assert replaced == 6
    assert set(before) == set(model.state_dict())
    assert all(torch.equal(t, model.state_dict()[n]) for n, t in before.items())
    assert not any(isinstance(m, torch.nn.GELU) for m in model.modules())
