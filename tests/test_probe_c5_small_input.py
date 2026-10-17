from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import probe_c5_small_input as probe


def parent():
    return {
        "reserved_STEM_participants": ["P4"],
        "ROI_diagnostic_participants": ["P19", "P0"],
        "records": [
            {
                "roi_role": "training",
                "roi_training_allowed": True,
                "original_row": {
                    "example_id": str(i),
                    "signer": f"P{i + 1}",
                    "role": "training",
                    "training_allowed": True,
                    "original_pair": {"header": {"sampled_frames": 20}},
                },
            }
            for i in range(2)
        ],
    }


@pytest.mark.parametrize("field", ["roi_role", "roi_training_allowed", "role", "training_allowed"])
def test_training_cases_require_both_real_training_permissions(field):
    plan = parent()
    assert probe.training_cases(plan) == plan["records"]
    row = plan["records"][0]
    target = row if field.startswith("roi_") else row["original_row"]
    target[field] = False if field.endswith("allowed") else "development"
    with pytest.raises(ValueError, match="training"):
        probe.training_cases(plan)


@pytest.mark.parametrize("signer", ["P4", "P19", "P0"])
def test_training_cases_exclude_all_reserved_and_auxiliary_signers(signer):
    plan = parent()
    plan["records"][0]["original_row"]["signer"] = signer
    with pytest.raises(ValueError, match="nonreserved"):
        probe.training_cases(plan)


def test_training_cases_reject_duplicate_media_and_single_frame_cases():
    plan = parent()
    plan["records"][1] = copy.deepcopy(plan["records"][0])
    with pytest.raises(ValueError, match="duplicated"):
        probe.training_cases(plan)
    plan = parent()
    plan["records"][0]["original_row"]["original_pair"]["header"]["sampled_frames"] = 1
    with pytest.raises(ValueError):
        probe.training_cases(plan)


def test_native_packing_preserves_the_declared_surface_count_and_BGR_to_RGB_order():
    pixels = np.zeros((112, 112, 3), dtype=np.uint8)
    pixels[..., 2] = 255
    result = probe.pack_rgb([pixels, pixels], 2, 112)
    assert result.shape == (2, 112, 112, 3)
    assert (result[..., 0] == 255).all() and (result[..., 2] == 0).all()
    assert result.flags.c_contiguous and result.strides[-1] == 1
    for values, frames, side in [([pixels], 2, 112), ([pixels], 1, 224)]:
        with pytest.raises(ValueError):
            probe.pack_rgb(values, frames, side)


class LinearVisual(nn.Module):
    """Analytical gradient fixture, never a real DINO numerical qualification."""

    def __init__(self):
        super().__init__()
        self.hands, self.face = nn.Module(), nn.Module()
        for model in (self.hands, self.face):
            model.mask_token = nn.Parameter(torch.zeros(1, 384), requires_grad=False)
        self.weight = nn.Parameter(torch.linspace(0.1, 0.5, 384))

    def trainable_parameters(self):
        return {"weight": self.weight}

    def forward(self, pixels):
        return {
            name: pixels[:, i, :1].expand(-1, 384) * self.weight
            for i, name in enumerate(probe.STREAMS)
        }


def test_actual_feature_probe_matches_independent_closed_form_gradient_without_updates():
    model = LinearVisual()
    pixels = torch.tensor([[[1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0]]])
    before = model.weight.detach().clone()
    loss, output, gradients = probe.feature_probe(model, pixels)
    expected_gradient = torch.linspace(-1, 1, 384) * pixels.mean() / 384
    torch.testing.assert_close(gradients["weight"], expected_gradient)
    torch.testing.assert_close(model.weight, before)
    assert output.shape == (6, 384) and np.isfinite(loss)
    assert model.hands.mask_token.grad is None and model.face.mask_token.grad is None
    # A prior unrelated gradient cannot contaminate a subsequent comparison.
    model.weight.grad.fill_(999)
    _, _, repeated = probe.feature_probe(model, pixels)
    torch.testing.assert_close(repeated["weight"], expected_gradient)


def test_frozen_policy_does_not_claim_training_or_translation_qualification():
    assert probe.POLICY["training_qualification_complete"] is False
    assert probe.POLICY["translation_quality_evaluated"] is False
    assert probe.POLICY["new_Terra_requests"] == probe.POLICY["optimizer_updates"] == 0
    assert probe.POLICY["native_side"] == 112 and probe.POLICY["control_side"] == 224
    assert probe.verified_comparison.__module__ == "tiny_rgb_capacity_protocol"
