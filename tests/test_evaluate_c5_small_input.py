from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import evaluate_c5_small_input as evaluate
import evaluate_roi_short_translation as original


def qualified():
    comparison = {
        "loss_absolute_difference": 0,
        "logit_maximum_absolute_difference": 0,
        "total_gradient_norm_relative_error": 0,
        "gradient_differences": [
            {"parameter": str(i), "relative_l2_error": 0, "finite": True} for i in range(350)
        ],
    }
    report = {
        "source": "asl_stem_original",
        "example_id": "training-case",
        "complete_crop_frames": 63,
        "numerical_sample_positions": [0, 62],
        "both_sides_native_pixel_parity": True,
        "control224_maximum_absolute_error": 0,
        "comparison": comparison,
    }
    result = {
        "complete": True,
        "plan_sha256": "bound",
        "component_numerics_passed": True,
        "translation_quality_evaluated": False,
        "optimizer_updates": 0,
        "new_Terra_requests": 0,
        "promotion_authorized": False,
        "trainable_parameters": 43260672,
        "trainable_tensors": 350,
        "comparisons": [report],
    }
    plan = {
        "policy": {"native_side": 112, "control_maximum_absolute_error": 1e-4},
        "records": [
            {
                "original_row": {
                    "source": "asl_stem_original",
                    "example_id": "training-case",
                    "original_pair": {"header": {"sampled_frames": 63}},
                }
            }
        ],
    }
    return result, plan


@pytest.mark.parametrize(
    "field,value",
    [
        ("complete", False),
        ("plan_sha256", "unbound"),
        ("component_numerics_passed", False),
        ("translation_quality_evaluated", True),
        ("optimizer_updates", 1),
        ("new_Terra_requests", 1),
        ("promotion_authorized", True),
        ("trainable_parameters", 43261440),
        ("trainable_tensors", 352),
        ("comparisons", []),
    ],
)
def test_incomplete_or_changed_real_model_qualification_cannot_start_evaluation(field, value):
    result, plan = qualified()
    evaluate.require_qualification(result, plan, "bound")
    result[field] = value
    with pytest.raises(ValueError):
        evaluate.require_qualification(result, plan, "bound")


@pytest.mark.parametrize(
    "field,value",
    [
        ("source", "another_source"),
        ("example_id", "different_case"),
        ("complete_crop_frames", 62),
        ("numerical_sample_positions", [0, 61]),
        ("both_sides_native_pixel_parity", False),
        ("control224_maximum_absolute_error", 0.001),
        ("control224_maximum_absolute_error", float("nan")),
    ],
)
def test_qualification_must_bind_actual_frames_native_pixels_and_control_output(field, value):
    result, plan = qualified()
    result["comparisons"][0][field] = value
    with pytest.raises(ValueError):
        evaluate.require_qualification(result, plan, "bound")


def test_qualification_rechecks_every_gradient_with_the_original_thresholds():
    result, plan = qualified()
    result["comparisons"][0]["comparison"]["gradient_differences"][-1]["relative_l2_error"] = 0.006
    with pytest.raises(ValueError, match="gradient"):
        evaluate.require_qualification(result, plan, "bound")
    result, plan = qualified()
    result["comparisons"][0]["comparison"]["gradient_differences"].pop()
    with pytest.raises(ValueError, match="coverage"):
        evaluate.require_qualification(result, plan, "bound")


def native_targets(frames):
    return {
        **{name: np.full((frames, 384), 999, dtype=np.float32) for name in evaluate.STREAMS},
        "pose_features": np.arange(frames * 14, dtype=np.float32).reshape(frames, 14),
    }


def test_small_input_features_cover_every_frame_once_and_keep_only_the_native_pose(monkeypatch):
    received = []

    class Batch:
        def __init__(self, pixels):
            self.indices = pixels[:, 0, 0, 0, 0].copy()

        def to(self, device):
            assert device == "mps"
            return self

    def appearance(batch):
        received.extend(batch.indices.tolist())
        return {
            name: torch.from_numpy(batch.indices.copy()).float()[:, None].expand(-1, 384)
            for name in evaluate.STREAMS
        }

    monkeypatch.setattr(evaluate, "input_tensor", Batch)
    pixels = np.zeros((5, 3, 112, 112, 3), dtype=np.uint8)
    pixels[:, 0, 0, 0, 0] = np.arange(5)
    targets = native_targets(5)
    features = evaluate.predicted_streams(
        appearance, pixels, targets, SimpleNamespace(run=lambda f: f())
    )
    assert received == list(range(5))
    for name in evaluate.STREAMS:
        torch.testing.assert_close(features[name][:, 0], torch.arange(5, dtype=torch.float32))
    np.testing.assert_array_equal(features["pose_features"].numpy(), targets["pose_features"])
    features["pose_features"].zero_()
    assert targets["pose_features"].sum() > 0


@pytest.mark.parametrize("kind", ["short_pose", "nonfinite_pose", "wrong_dtype", "missing_pose"])
def test_invalid_pose_is_rejected_before_any_appearance_work(kind):
    targets = native_targets(3)
    if kind == "short_pose":
        targets["pose_features"] = targets["pose_features"][:2]
    elif kind == "nonfinite_pose":
        targets["pose_features"][0, 0] = float("nan")
    elif kind == "wrong_dtype":
        targets["pose_features"] = targets["pose_features"].astype(np.float64)
    else:
        targets.pop("pose_features")
    with pytest.raises(ValueError):
        evaluate.predicted_streams(
            None, np.zeros((3, 3, 112, 112, 3), dtype=np.uint8), targets, None
        )


def test_original_C5_generation_controls_and_diagnostic_WER_are_reused_unchanged():
    assert evaluate.decode is original.decode
    assert evaluate.require_control is original.require_control
    assert evaluate.prediction is original.prediction
    assert evaluate.POLICY["WER_is_diagnostic_only"] is True
    assert evaluate.POLICY["new_Terra_requests"] == evaluate.POLICY["optimizer_updates"] == 0
    assert evaluate.POLICY["promotion_authorized"] is False
    result, plan = qualified()
    changed = copy.deepcopy(plan)
    changed["policy"]["native_side"] = 224
    with pytest.raises(ValueError):
        evaluate.require_qualification(result, changed, "bound")


@pytest.mark.parametrize("mismatched_control", [False, True])
def test_all_controls_must_reproduce_before_any_candidate_pixels_are_read(
    tmp_path, monkeypatch, mismatched_control
):
    import c5_certification_budget
    import c5_terra_tail
    import tiny_rgb_winner_student
    import transformers

    events = []

    class Budget:
        def __init__(self, path):
            pass

        def duty_fraction(self):
            return 0.5

        def cpu_threads(self):
            return 2

    class Duty:
        def __init__(self, *args):
            pass

        def run(self, operation):
            return operation()

        def receipt(self):
            return {}

    class Vision(torch.nn.Module):
        def __init__(self, *args):
            super().__init__()

        def to(self, device):
            assert device == "mps"
            return self

    class Encoder(torch.nn.Module):
        def forward(self, **kwargs):
            return SimpleNamespace(last_hidden_state=torch.ones(1, 3, 1536))

    reference = {
        "bundle_root": "artifacts/fixture",
        "translator_checkpoint_sha256": "fixed",
        "generation": {"num_beams": 5, "max_length": 2048},
    }
    model = torch.nn.Module()
    model.language = torch.nn.Linear(1, 1)
    monkeypatch.setattr(c5_certification_budget, "CertificationComputeWindow", Budget)
    monkeypatch.setattr(c5_certification_budget, "WinnerDutyCycle", Duty)
    monkeypatch.setattr(c5_terra_tail, "load_model", lambda: (model, reference))
    monkeypatch.setattr(c5_terra_tail, "native_memory", lambda *args: torch.zeros(1, 3, 1536))
    monkeypatch.setattr(
        tiny_rgb_winner_student, "load_frozen_winner_encoder", lambda ref: Encoder()
    )
    monkeypatch.setattr(transformers.ByT5Tokenizer, "from_pretrained", lambda *args, **kw: None)
    monkeypatch.setattr(evaluate.torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(evaluate, "native_helpers", lambda: None)
    monkeypatch.setattr(evaluate, "legacy_models", lambda plan: (SimpleNamespace(model=None),) * 2)
    monkeypatch.setattr(evaluate, "SmallInputDino", Vision)
    monkeypatch.setattr(evaluate, "state_digest", lambda values: "unchanged")
    monkeypatch.setattr(evaluate, "storage", lambda **kwargs: None)
    monkeypatch.setattr(evaluate, "checked", lambda path: None)
    monkeypatch.setattr(evaluate, "sha256", lambda path: "bound")
    monkeypatch.setattr(evaluate, "load_targets", lambda pair: None)
    monkeypatch.setattr(evaluate, "predicted_streams", lambda *args: {})

    def decode(language, tokenizer, generation, memory):
        if memory.sum() == 0:
            events.append("control")
            return "wrong" if mismatched_control else "fixed phrase", [0, 1]
        events.append("candidate")
        return "candidate phrase", [0, 1]

    def crops(record):
        events.append("candidate_pixels")
        return np.zeros((3, 3, 112, 112, 3), dtype=np.uint8)

    monkeypatch.setattr(evaluate, "decode", decode)
    monkeypatch.setattr(evaluate, "crops", crops)
    rows = [
        {
            "source": "asl_stem_original",
            "example_id": str(i),
            "source_group": f"group-{i}",
            "signer": "P4",
            "reference": "fixed phrase",
            "role": "development",
            "training_allowed": False,
            "original_pair": {},
        }
        for i in range(34)
    ]
    control = tmp_path / "control.json"
    control.write_text(
        json.dumps({"predictions": [evaluate.prediction(row, "fixed phrase") for row in rows]})
    )
    monkeypatch.setattr(evaluate, "CONTROL", control)
    plan = {
        "reference": reference,
        "resource_policy": "runs/policy.json",
        "initial_reference_hashes": {"language": "unchanged", "encoder": "unchanged"},
        "pretrained_state_sha256": "unchanged",
        "records": [{"development": row} for row in rows],
    }
    if mismatched_control:
        with pytest.raises(ValueError, match="control text"):
            evaluate.infer(tmp_path / "plan.json", tmp_path, plan)
        assert events == ["control"]
        assert not (tmp_path / "receipt.json").exists()
    else:
        evaluate.infer(tmp_path / "plan.json", tmp_path, plan)
        assert events == ["control"] * 34 + ["candidate_pixels", "candidate"] * 34
        receipt = json.loads((tmp_path / "receipt.json").read_text())
        assert receipt["clips"] == 34 and receipt["all34_frozen_C5_controls_reproduced"]
        assert receipt["optimizer_updates"] == receipt["new_Terra_requests"] == 0
        assert all(
            row["role"] == "development" and row["training_allowed"] is False
            for row in receipt["predictions"]
        )
