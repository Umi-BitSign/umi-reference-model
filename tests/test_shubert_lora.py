"""LoRA topology, checkpoint-key restoration and adapter loading."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from tools.shubert_lora import LoRALinear, adapter_state, install_lora, load_adapter, merge_lora


class _Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q = nn.Linear(4, 4, bias=False)
        self.k = nn.Linear(4, 4, bias=False)
        self.v = nn.Linear(4, 4, bias=False)
        self.o = nn.Linear(4, 4, bias=False)


class _SelfAttentionLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.SelfAttention = _Attention()


class _CrossAttentionLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.EncDecAttention = _Attention()


class _FeedForward(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.wi_0 = nn.Linear(4, 8, bias=False)
        self.wi_1 = nn.Linear(4, 8, bias=False)
        self.wo = nn.Linear(8, 4, bias=False)


class _FeedForwardLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.DenseReluDense = _FeedForward()


class _TinyDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        block = nn.Module()
        block.layer = nn.ModuleList(
            [_SelfAttentionLayer(), _CrossAttentionLayer(), _FeedForwardLayer()]
        )
        self.block = nn.ModuleList([block])


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.decoder = _TinyDecoder()


@pytest.mark.parametrize(
    ("targets", "expected_layers"),
    [("cross_qv", 2), ("decoder_attention", 6), ("decoder_all", 9)],
)
def test_lora_zero_initialization_and_merge(targets: str, expected_layers: int) -> None:
    torch.manual_seed(17)
    model = _TinyModel()
    original_keys = set(model.state_dict())
    trainable = install_lora(model, rank=2, alpha=4, targets=targets)
    assert len(trainable) == expected_layers * 2
    layer = model.decoder.block[0].layer[1].EncDecAttention.q
    sample = torch.randn(3, 4)
    assert torch.equal(layer.weight, layer.base.weight)
    assert layer.bias is None
    assert layer.in_features == 4 and layer.out_features == 4
    assert torch.allclose(layer(sample), layer.base(sample))
    if targets == "decoder_all":
        feed_forward = model.decoder.block[0].layer[2].DenseReluDense
        assert torch.equal(feed_forward.wo.weight, feed_forward.wo.base.weight)
    with torch.no_grad():
        layer.lora_b.weight.fill_(0.1)
    adapted = layer(sample).detach().clone()
    merge_lora(model)
    assert set(model.state_dict()) == original_keys
    merged = model.decoder.block[0].layer[1].EncDecAttention.q
    assert torch.allclose(adapted, merged(sample), atol=1e-6)


def test_adapter_round_trip_rejects_missing_tensor() -> None:
    torch.manual_seed(17)
    trained = _TinyModel()
    install_lora(trained, rank=2, alpha=4)
    with torch.no_grad():
        trained.decoder.block[0].layer[1].EncDecAttention.q.lora_b.weight.fill_(0.2)
    tensors = adapter_state(trained)
    torch.manual_seed(17)
    loaded = _TinyModel()
    install_lora(loaded, rank=2, alpha=4)
    load_adapter(loaded, tensors)
    assert all(torch.equal(tensors[name], adapter_state(loaded)[name]) for name in tensors)
    with pytest.raises(ValueError, match="do not match"):
        load_adapter(loaded, {name: value for name, value in tensors.items() if "q." not in name})


def test_functional_weight_path_matches_forward_and_receives_gradients() -> None:
    torch.manual_seed(23)
    base = nn.Linear(8, 8)
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    layer = LoRALinear(base, rank=2, alpha=4)
    layer.lora_b.weight.data.normal_(std=0.1)
    sample = torch.randn(4, 8)
    functional = torch.nn.functional.linear(sample, layer.weight, layer.bias)
    assert torch.allclose(functional, layer(sample), atol=1e-6)
    functional.square().sum().backward()
    assert layer.lora_a.weight.grad.norm() > 0
    assert layer.lora_b.weight.grad.norm() > 0
    assert base.weight.grad is None


def test_fairseq_fused_attention_uses_lora_and_matches_merged_weights() -> None:
    import sys
    from pathlib import Path

    vendor = Path(__file__).resolve().parents[1] / "artifacts/baseline-6fe8df59/vendor"
    sys.path.insert(0, str(vendor))
    fairseq = pytest.importorskip("fairseq.modules.multihead_attention")
    torch.manual_seed(29)
    attention = fairseq.MultiheadAttention(8, 2, self_attention=True).eval()
    for parameter in attention.parameters():
        parameter.requires_grad_(False)
    attention.q_proj = LoRALinear(attention.q_proj, rank=2, alpha=4)
    attention.v_proj = LoRALinear(attention.v_proj, rank=2, alpha=4)
    sample = torch.randn(5, 1, 8)
    original = attention(sample, sample, sample, need_weights=False)[0].detach()
    attention.q_proj.lora_b.weight.data.normal_(std=0.1)
    attention.v_proj.lora_b.weight.data.normal_(std=0.1)
    adapted = attention(sample, sample, sample, need_weights=False)[0]
    assert not torch.allclose(adapted, original)
    adapted.square().sum().backward()
    for projection in (attention.q_proj, attention.v_proj):
        assert projection.lora_a.weight.grad.norm() > 0
        assert projection.lora_b.weight.grad.norm() > 0
        assert projection.base.weight.grad is None
    attention.q_proj = attention.q_proj.merged()
    attention.v_proj = attention.v_proj.merged()
    merged = attention(sample, sample, sample, need_weights=False)[0]
    assert torch.allclose(adapted, merged, atol=1e-6)
