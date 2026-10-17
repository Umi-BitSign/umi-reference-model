from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import evaluate_c5_small_input as evaluate
import probe_c5_linear_patch as probe
import probe_c5_small_input as original
from c5_patch_projection import LinearPatchProjection
from c5_small_input_vision import input_tensor
from tiny_rgb_capacity_protocol import verified_comparison


class Dino(nn.Module):
    """Small projection fixture, never actual DINO qualification."""

    def __init__(self):
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(1, 384))
        self.patch_embed = nn.Module()
        self.patch_embed.patch_size = (14, 14)
        self.patch_embed.proj = nn.Conv2d(3, 384, 14, stride=14)

    def forward(self, pixels):
        return self.patch_embed.proj(pixels).mean((-1, -2))


def test_private_factory_changes_only112_projection_and_preserves224_control_values():
    hands, face = Dino(), Dino()
    before = copy.deepcopy({"hands": hands.state_dict(), "face": face.state_dict()})
    model = probe.linear_model(hands, face)
    assert isinstance(model.hands.patch_embed.proj, LinearPatchProjection)
    assert isinstance(model.face.patch_embed.proj, LinearPatchProjection)
    assert not model.hands.mask_token.requires_grad and not model.face.mask_token.requires_grad
    for name in ("hands", "face"):
        assert all(
            torch.equal(value, getattr(model, name).state_dict()[key])
            for key, value in before[name].items()
        )
    control = probe.linear_model(Dino(), Dino(), side=224)
    assert isinstance(control.hands.patch_embed.proj, nn.Conv2d)
    with pytest.raises(ValueError, match="112"):
        probe.linear_model(Dino(), Dino(), side=98)


def test_native_CPU_reference_preserves_values_scope_rng_and_actual_fixture_gradients():
    torch.manual_seed(776)
    model = probe.linear_model(Dino(), Dino())
    initial = original.state_digest(model.state_dict())
    rng = torch.get_rng_state().clone()
    native = probe.native_reference(model)
    assert torch.equal(rng, torch.get_rng_state())
    assert original.state_digest(native.state_dict()) == initial
    assert native.trainable_parameters().keys() == model.trainable_parameters().keys()
    pixels = input_tensor(
        np.random.default_rng(778).integers(0, 256, (2, 3, 112, 112, 3), dtype=np.uint8)
    )
    expected = original.feature_probe(native, pixels)
    actual = original.feature_probe(model, pixels)
    comparison = probe.compare(
        expected[2], actual[2], expected[0], actual[0], expected[1], actual[1]
    )
    verified_comparison(comparison, 4, cpu_equivalence=True)
    assert original.state_digest(model.state_dict()) == initial
    with pytest.raises(ValueError, match="linear scope"):
        probe.native_reference(native)


def test_original_loop_bytecode_and_all_other_scientific_globals_are_unchanged(tmp_path):
    before = dict(original.run.__globals__)
    bound = probe.bound_run(tmp_path, {"expected_pretrained_state_sha256": "state"})
    assert bound.__code__ is original.run.__code__
    assert (
        bound.__defaults__ == original.run.__defaults__
        and bound.__closure__ == original.run.__closure__
    )
    changed = {key for key in before if before[key] is not bound.__globals__[key]}
    assert changed == {"checked", "SmallInputDino", "feature_probe", "write", "POLICY"}
    assert all(value is original.run.__globals__[key] for key, value in before.items())
    assert all(probe.POLICY[key] == value for key, value in original.POLICY.items())
    with pytest.raises(ValueError, match="CPU comparisons"):
        bound.__globals__["write"](
            tmp_path / "qualification.json", {"pretrained_state_sha256": "state"}
        )
    assert not (tmp_path / "qualification.json").exists()


def result_fixture(tmp_path):
    comparison = {
        "loss_absolute_difference": 0,
        "logit_maximum_absolute_difference": 0,
        "total_gradient_norm_relative_error": 0,
        "gradient_differences": [
            {"parameter": str(i), "relative_l2_error": 0, "finite": True} for i in range(350)
        ],
    }
    plan = {
        "policy": {"native_side": 112, "control_maximum_absolute_error": 1e-4},
        "expected_pretrained_state_sha256": "state",
        "records": [],
    }
    reports, native_reports = [], []
    for index, frames in enumerate((63, 40)):
        plan["records"].append(
            {
                "original_row": {
                    "source": "asl_stem_original",
                    "example_id": str(index),
                    "original_pair": {"header": {"sampled_frames": frames}},
                }
            }
        )
        report = {
            "source": "asl_stem_original",
            "example_id": str(index),
            "complete_crop_frames": frames,
            "numerical_sample_positions": [0, frames - 1],
            "both_sides_native_pixel_parity": True,
            "control224_maximum_absolute_error": 0,
            "comparison": copy.deepcopy(comparison),
        }
        native = {
            "case_index": index,
            "source": report["source"],
            "example_id": str(index),
            "numerical_sample_positions": [0, frames - 1],
            "comparison": copy.deepcopy(comparison),
        }
        reports.append(report)
        native_reports.append(native)
        (tmp_path / f"comparison-{index:02d}.json").write_text(json.dumps(report))
        (tmp_path / f"native-CPU-equivalence-{index:02d}.json").write_text(json.dumps(native))
    return {
        "complete": True,
        "plan_sha256": "bound",
        "component_numerics_passed": True,
        "translation_quality_evaluated": False,
        "optimizer_updates": 0,
        "new_Terra_requests": 0,
        "promotion_authorized": False,
        "trainable_parameters": 43260672,
        "trainable_tensors": 350,
        "pretrained_state_sha256": "state",
        "patch_projection": probe.POLICY["patch_projection"],
        "comparisons": reports,
        "native_CPU_equivalence": native_reports,
    }, plan


@pytest.mark.parametrize(
    "mutation",
    [
        None,
        "missing_native",
        "native_output",
        "native_gradient",
        "native_id",
        "state",
        "actual_file",
    ],
)
def test_acceptance_requires_both_actual_native_CPU_comparisons_and_original_gates(
    tmp_path, mutation
):
    result, plan = result_fixture(tmp_path)
    if mutation == "missing_native":
        result["native_CPU_equivalence"].pop()
    elif mutation in ("native_output", "native_gradient", "native_id"):
        report = result["native_CPU_equivalence"][1]
        if mutation == "native_output":
            report["comparison"]["logit_maximum_absolute_difference"] = 1.1e-4
        elif mutation == "native_gradient":
            report["comparison"]["gradient_differences"][-1]["relative_l2_error"] = 0.006
        else:
            report["example_id"] = "other"
        (tmp_path / "native-CPU-equivalence-01.json").write_text(json.dumps(report))
    elif mutation == "state":
        result["pretrained_state_sha256"] = "changed"
    elif mutation == "actual_file":
        (tmp_path / "comparison-01.json").write_text("{}")
    if mutation is None:
        probe.require_result(result, plan, "bound", tmp_path)
    else:
        with pytest.raises(ValueError):
            probe.require_result(result, plan, "bound", tmp_path)


def test_evaluation_uses_the_declared_projection_and_rejects_unknown_choices():
    native = evaluate.appearance_model({}, Dino(), Dino())
    assert isinstance(native.hands.patch_embed.proj, nn.Conv2d)
    linear = evaluate.appearance_model(
        {"patch_projection": probe.POLICY["patch_projection"]}, Dino(), Dino()
    )
    assert isinstance(linear.hands.patch_embed.proj, LinearPatchProjection)
    with pytest.raises(ValueError, match="projection"):
        evaluate.appearance_model({"patch_projection": "unknown"}, Dino(), Dino())


@pytest.mark.parametrize("changed", [False, True])
def test_evaluation_linear_route_rechecks_actual_extra_CPU_evidence(tmp_path, monkeypatch, changed):
    result, plan = result_fixture(tmp_path)
    qualification_plan = tmp_path / "plan.json"
    qualification_plan.write_text(
        json.dumps({"schema": "umi-studio-C5-linear-patch-component-qualification/1"})
    )
    result["plan_sha256"] = probe.sha256(qualification_plan)
    if changed:
        report = result["native_CPU_equivalence"][1]
        report["comparison"]["gradient_differences"][-1]["relative_l2_error"] = 0.006
        (tmp_path / "native-CPU-equivalence-01.json").write_text(json.dumps(report))
    qualification_result = tmp_path / "qualification.json"
    qualification_result.write_text(json.dumps(result))
    monkeypatch.setattr(probe, "checked", lambda path: plan)
    monkeypatch.setattr(evaluate, "terminal", lambda path: None)
    if changed:
        with pytest.raises(ValueError, match="gradient"):
            evaluate.component_evidence(qualification_plan, qualification_result)
    else:
        actual_plan, actual_result = evaluate.component_evidence(
            qualification_plan, qualification_result
        )
        assert actual_plan == plan and actual_result == result
