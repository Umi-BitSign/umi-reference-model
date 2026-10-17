from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from c5_patch_projection import LinearPatchProjection
from probe_c5_patch_projection import POLICY, isolated


@pytest.mark.parametrize("side", [112, 224])
@pytest.mark.parametrize("layout", ["contiguous", "strided", "channels_last"])
def test_linear_patches_match_independent_native_convolution_forward_and_all_gradients(
    side, layout
):
    torch.manual_seed(771)
    original = nn.Conv2d(3, 384, 14, stride=14)
    native = copy.deepcopy(original).double()
    linear = LinearPatchProjection(copy.deepcopy(original))
    pixels = torch.randn(4, 3, side, side)
    if layout == "strided":
        pixels = pixels[::2]
    elif layout == "channels_last":
        pixels = pixels.contiguous(memory_format=torch.channels_last)
    first, second = pixels.double().requires_grad_(), pixels.detach().requires_grad_()
    expected, actual = native(first), linear(second)
    torch.testing.assert_close(actual.double(), expected, rtol=1e-5, atol=2e-6)
    cotangent = torch.randn_like(actual)
    (expected * cotangent.double()).sum().backward()
    (actual * cotangent).sum().backward()
    for key, parameter in native.named_parameters():
        measured = dict(linear.named_parameters())[key].grad.double()
        assert float((measured - parameter.grad).norm() / parameter.grad.norm()) < 1e-6
    assert float((second.grad.double() - first.grad).norm() / first.grad.norm()) < 1e-6


def test_projection_preserves_parameter_objects_values_state_keys_and_gradient_scope():
    native = nn.Conv2d(3, 384, 14, stride=14)
    native.bias.requires_grad_(False)
    rng = torch.get_rng_state().clone()
    linear = LinearPatchProjection(native)
    assert linear.weight is native.weight and linear.bias is native.bias
    assert linear.weight.requires_grad and not linear.bias.requires_grad
    assert linear.state_dict().keys() == native.state_dict().keys()
    assert torch.equal(torch.get_rng_state(), rng)
    for key, value in native.state_dict().items():
        assert torch.equal(linear.state_dict()[key], value)


@pytest.mark.parametrize("changed", ["stride", "padding", "groups", "bias"])
def test_changed_projection_contract_is_rejected(changed):
    kwargs = {"stride": 14}
    kwargs[changed] = {"stride": 7, "padding": 1, "groups": 3, "bias": False}[changed]
    with pytest.raises(ValueError):
        LinearPatchProjection(nn.Conv2d(3, 384, 14, **kwargs))


@pytest.mark.parametrize("variant", POLICY["variants"])
def test_actual_isolation_helper_uses_supplied_cotangent_without_changing_original_weights(variant):
    torch.manual_seed(774)
    original = nn.Conv2d(3, 384, 14, stride=14)
    before = copy.deepcopy(original.state_dict())
    reference = copy.deepcopy(original).double()
    pixels = torch.randn(2, 3, 112, 112)
    cotangent = torch.randn(2, 384, 8, 8)
    expected = reference(pixels.double())
    (expected * cotangent.double()).sum().backward()
    _, actual, gradients = isolated(original, pixels, cotangent, variant, "cpu")
    torch.testing.assert_close(actual.double(), expected, rtol=1e-5, atol=2e-6)
    for key, parameter in reference.named_parameters():
        assert (
            float((gradients[key].double() - parameter.grad).norm() / parameter.grad.norm()) < 1e-6
        )
    assert all(torch.equal(value, original.state_dict()[key]) for key, value in before.items())
    assert all(parameter.grad is None for parameter in original.parameters())


def test_unknown_variant_is_rejected_and_diagnostic_never_claims_full_qualification():
    with pytest.raises(ValueError, match="variant"):
        isolated(None, None, None, "undeclared", "cpu")
    assert POLICY["full_component_numerics_qualified"] is False
    assert POLICY["optimizer_updates"] == POLICY["new_Terra_requests"] == 0
