"""Synthetic frozen-backbone timing integration; no actual C5 quality claim."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from c5_monotone_feature_timing import WIDTHS
from c5_timing_semantic_model import TimingSemanticStudent, clip_input, gradient_report
from terra_quality_training import minimum_risk, sequence_log_probability
from terra_visual_learning import backward_candidate_risk


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.7))

    def forward(self, *, return_dict, **features):
        assert return_dict is True
        combined = sum(value.mean(-1, keepdim=True) for value in features.values())
        memory = combined.expand(-1, -1, 1536) * self.scale
        return SimpleNamespace(last_hidden_state=memory)


class TinyLanguage(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.3))
        self.config = SimpleNamespace(d_model=1536)
        self.frozen_winner_weights = False
        self.last_generate = None

    def forward(self, *, encoder_outputs, attention_mask, labels, use_cache, return_dict):
        assert use_cache is False and return_dict is True
        assert attention_mask.dtype == torch.int64 and attention_mask.all()
        memory = encoder_outputs.last_hidden_state
        weights = torch.linspace(0.1, 1, memory.shape[1])
        summary = (memory.mean(-1) * weights).mean() * self.scale
        logits = (summary * torch.linspace(-1, 1, 8))[None, None].expand(1, labels.shape[1], 8)
        loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=-100)
        return SimpleNamespace(loss=loss, logits=logits)

    def generate(self, *, encoder_outputs, attention_mask, **kwargs):
        self.last_generate = (encoder_outputs.last_hidden_state.clone(), attention_mask, kwargs)
        return torch.tensor([[1, 2, 3]])


@pytest.fixture
def sample():
    generator = np.random.default_rng(173)
    values = {
        "streams": {
            name: (generator.normal(size=(13, width)) * np.linspace(0.2, 2, 13)[:, None]).astype(
                np.float32
            )
            for name, width in WIDTHS.items()
        },
        "valid_intervals": np.ones(12, dtype=np.bool_),
    }
    values["streams"]["pose_features"][0, 0] = -0.0
    return values, clip_input(values), torch.ones((1, 13), dtype=torch.bool)


def student():
    return TimingSemanticStudent(TinyEncoder(), TinyLanguage())


def test_only_four_timing_parameters_train_and_backbones_retain_eval():
    model = student().train()
    selected = model.trainable_parameters()
    assert set(selected) == {"timing.rate_weights"}
    assert sum(p.numel() for p in selected.values()) == 4
    assert model.training and model.timing.training
    assert not model.semantic_encoder.training and not model.language.training
    assert model.language.frozen_winner_weights is True
    model.require_fixed()
    assert model.eval() is model and not model.timing.training


def test_initial_inference_features_and_memory_match_direct_frozen_path_bitwise(sample):
    _, clip, mask = sample
    model = student().eval()
    with torch.no_grad():
        features, actual_mask, timing = model.timed_features(clip, mask)
        direct_features = {k: v[None] for k, v in clip.streams.items()}
        expected = model.semantic_memory(direct_features, mask)
        memory, encoded_mask = model.encode(clip, mask)
    assert actual_mask is mask and encoded_mask is mask
    assert torch.equal(timing["positions"], torch.arange(13, dtype=torch.float32))
    for key in WIDTHS:
        assert torch.equal(features[key].view(torch.int32), direct_features[key].view(torch.int32))
    assert torch.equal(memory.view(torch.int32), expected.view(torch.int32))


def test_frozen_semantic_loss_updates_only_timing_and_reports_real_gradient(sample):
    _, clip, mask = sample
    model = student().train()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    optimizer = torch.optim.SGD(model.trainable_parameters().values(), lr=0.01)
    result = model(clip, mask, torch.tensor([[1, 2]]))
    result["language"].loss.backward()
    report = gradient_report(model)
    assert report["gradient_norm"] > 0
    assert len(report["gradient_values"]) == 4
    assert report["frozen_backbone_gradients_absent"]
    assert torch.count_nonzero(model.timing.rate_weights.grad) == 4
    optimizer.step()
    for key, value in model.state_dict().items():
        if key == "timing.rate_weights":
            assert not torch.equal(value, before[key])
        else:
            assert torch.equal(value, before[key])


def test_generate_preserves_all_native_generation_options(sample):
    _, clip, mask = sample
    model = student().eval()
    options = {
        "num_beams": 4,
        "max_new_tokens": 120,
        "length_penalty": 1.0,
        "do_sample": False,
        "early_stopping": True,
        "no_repeat_ngram_size": 0,
    }
    result = model.generate(clip, mask, **options)
    assert result.tolist() == [[1, 2, 3]]
    memory, actual_mask, actual_options = model.language.last_generate
    assert memory.shape == (1, 13, 1536) and not memory.requires_grad
    assert torch.equal(actual_mask, mask.long()) and actual_options == options


def test_clip_input_copies_values_without_alias_or_cast(sample):
    values, clip, _ = sample
    for name in WIDTHS:
        assert np.array_equal(
            clip.streams[name].numpy().view(np.int32), values["streams"][name].view(np.int32)
        )
        values["streams"][name][:] = 19
        assert not torch.all(clip.streams[name] == 19)
    values["valid_intervals"][:] = False
    assert clip.valid_intervals.all()


@pytest.mark.parametrize("change", ["double", "missing", "mask_float"])
def test_clip_input_rejects_silent_casts_or_incomplete_streams(sample, change):
    values, _, _ = sample
    if change == "double":
        values["streams"]["face_features"] = values["streams"]["face_features"].astype(np.float64)
    elif change == "missing":
        del values["streams"]["pose_features"]
    else:
        values["valid_intervals"] = values["valid_intervals"].astype(np.float32)
    with pytest.raises(ValueError):
        clip_input(values)


@pytest.mark.parametrize(
    "change",
    ["mask_float", "mask_shape", "mask_gap", "input_grad", "backbone_grad", "backbone_train"],
)
def test_forward_rejects_changed_coverage_or_backbone_contract(sample, change):
    _, clip, mask = sample
    model = student()
    if change == "mask_float":
        mask = mask.float()
    elif change == "mask_shape":
        mask = mask[0]
    elif change == "mask_gap":
        mask[0, 5] = False
    elif change == "input_grad":
        clip.streams["face_features"].requires_grad_(True)
    elif change == "backbone_grad":
        model.semantic_encoder.scale.requires_grad_(True)
    else:
        model.language.train()
    with pytest.raises(ValueError):
        model(clip, mask, torch.tensor([[1]]))


@pytest.mark.parametrize("change", ["absent", "nan", "zero", "backbone"])
def test_gradient_report_rejects_missing_invalid_or_backbone_signal(change):
    model = student()
    if change != "absent":
        model.timing.rate_weights.grad = torch.ones(4)
    if change == "nan":
        model.timing.rate_weights.grad[0] = float("nan")
    elif change == "zero":
        model.timing.rate_weights.grad.zero_()
    elif change == "backbone":
        model.language.scale.grad = torch.tensor(1.0)
    with pytest.raises(ValueError):
        gradient_report(model)


def test_no_support_case_retains_clip_and_permits_explicit_zero_signal(sample):
    _, clip, mask = sample
    clip.valid_intervals.zero_()
    model = student()
    result = model(clip, mask, torch.tensor([[1]]))
    loss = result["language"].loss + result["timing"]["log_rate_smoothness_penalty"]
    loss.backward()
    assert gradient_report(model, require_signal=False)["gradient_norm"] == 0
    assert result["memory"].shape == (1, 13, 1536)


def test_clip_to_accepts_only_cpu_and_never_casts_or_moves(sample):
    _, clip, _ = sample
    assert clip.to(torch.device("cpu")) is clip and clip.to("cpu") is clip
    with pytest.raises(ValueError, match="stay on CPU"):
        clip.to("mps")


def test_existing_streamed_semantic_risk_matches_full_autograd_through_frozen_model(sample):
    _, clip, mask = sample
    model = student()
    labels = [torch.tensor([[2, 1]]), torch.tensor([[3, 4, 1]]), torch.tensor([[5, 1]])]
    quality = [10, 7, 0]
    logs = torch.stack(
        [
            sequence_log_probability(model(clip, mask, value)["language"].logits, value)
            for value in labels
        ]
    )
    expected = minimum_risk(logs, quality, alpha=0.005)
    expected["loss"].backward()
    gradient = model.timing.rate_weights.grad.clone()
    model.zero_grad(set_to_none=True)
    report = backward_candidate_risk(
        model, clip, mask, labels, quality, alpha=0.005, caption_weight=0
    )
    torch.testing.assert_close(model.timing.rate_weights.grad, gradient, rtol=1e-5, atol=1e-8)
    assert report["expected_terra_error"] == float(expected["loss"])
    assert report["terra_gradient_coefficient_l1"] > 0
    assert gradient_report(model)["gradient_norm"] > 0
