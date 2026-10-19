"""Candidate scope and trained native interface on tiny synthetic backbones."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import c5_temporal_memory_candidate as candidate
import c5_timing_feature_data as data
import qualify_c5_temporal_memory_native as native
import test_c5_timing_semantic_model as semantic_tests
from c5_temporal_memory_model import TemporalMemoryStudent
from test_c5_temporal_memory_model import DiverseEncoder
from test_probe_c5_timing_semantic import Tokenizer
from train_c5_temporal_memory import state_hashes


def student():
    return TemporalMemoryStudent(DiverseEncoder(), semantic_tests.TinyLanguage()).eval()


sample = semantic_tests.sample


def test_original_risk_audit_cannot_release_caption_candidate(tmp_path, monkeypatch):
    import c5_temporal_memory_final as final
    import c5_temporal_memory_model as semantic

    root = Path(__file__).resolve().parents[1]
    path = tmp_path / "old-audit.json"
    import json

    path.write_text(json.dumps({"audit_program": data.artifact(root / "tools/c5_timing_final.py")}))
    monkeypatch.setattr(
        semantic, "load_student", lambda *args: pytest.fail("model loaded before caption audit")
    )
    with pytest.raises(ValueError, match="memory final auditor changed"):
        candidate.load_candidate(path)
    assert final.__name__ == "c5_temporal_memory_final"


@pytest.fixture
def adapter(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "ROOT", tmp_path)
    path = tmp_path / "checkpoints/final.pt"
    path.parent.mkdir()
    model = student()
    weights = {k: v.clone() for k, v in model.memory_adapter.state_dict().items()}
    weights["output_projection.weight"].fill_(0.03)
    trained = student()
    trained.memory_adapter.load_state_dict(weights)
    plan = {"path": "runs/plan.json", "sha256": "0" * 64}
    state = {
        "memory_adapter": weights,
        "configuration": model.memory_adapter.configuration(),
        "optimizer": {},
        "epochs": 4,
        "updates": 268,
        "exposures": 2128,
        "plan": plan,
    }
    torch.save(state, path)
    binding = {
        "schema": "umi-studio-complete-temporal-memory-final-audit/1",
        "complete": True,
        "optimizer_updates_replayed": 268,
        "complete_training_exposures": 2128,
        "full_epochs": 4,
        "adapter_configuration": model.memory_adapter.configuration(),
        "adapter": data.artifact(path),
        "state_hashes": state_hashes(trained.memory_adapter),
        "plan": plan,
    }
    return model, binding, state, path


def test_installs_exact_trained_values_and_preserves_all_synthetic_backbone_tensors(adapter):
    model, binding, _, _ = adapter
    before = {k: v.clone() for k, v in model.state_dict().items()}
    candidate.install_adapter(model, binding)
    assert state_hashes(model.memory_adapter) == binding["state_hashes"]
    assert model.training is False
    for name, value in model.state_dict().items():
        if not name.startswith("memory_adapter."):
            assert torch.equal(value, before[name])
    candidate.require_candidate(model, binding)
    with torch.no_grad():
        model.memory_adapter.output_projection.weight[0, 0] += 0.01
    with pytest.raises(ValueError, match="changed during evaluation"):
        candidate.require_candidate(model, binding)


@pytest.mark.parametrize(
    "change",
    ["partial", "double", "extra", "zero", "bound", "weights", "plan", "already_installed"],
)
def test_candidate_refuses_partial_or_expanded_or_changed_timing_state(adapter, change):
    model, binding, state, path = adapter
    if change == "partial":
        state["updates"] = 133
    elif change == "double":
        state["memory_adapter"]["output_projection.weight"] = state["memory_adapter"][
            "output_projection.weight"
        ].double()
    elif change == "extra":
        state["memory_adapter"]["language.weight"] = torch.ones(1)
    elif change == "zero":
        state["memory_adapter"]["output_projection.weight"].zero_()
    elif change == "bound":
        state["configuration"]["maximum_relative_residual"] = 0.6
    elif change == "weights":
        state["memory_adapter"]["output_projection.weight"][0, 0] += 0.01
    elif change == "plan":
        state["plan"] = {"different": True}
    else:
        with torch.no_grad():
            model.memory_adapter.output_projection.weight[0, 0] = 0.1
    torch.save(state, path)
    binding["adapter"] = data.artifact(path)
    with pytest.raises(ValueError):
        candidate.install_adapter(model, binding)


def test_no_real_candidate_model_load_before_complete_audit(tmp_path, monkeypatch):
    import c5_temporal_memory_final as final
    import c5_temporal_memory_model as semantic

    def refuse(path):
        raise ValueError("actual complete final missing")

    monkeypatch.setattr(final, "checked_final", refuse)
    monkeypatch.setattr(
        semantic, "load_student", lambda *args: pytest.fail("real model loaded early")
    )
    with pytest.raises(ValueError, match="complete final missing"):
        candidate.load_candidate(tmp_path / "missing.json")
    monkeypatch.setattr(data, "ROOT", tmp_path)
    plan = tmp_path / "runs/new/plan.json"
    with pytest.raises(ValueError, match="complete final missing"):
        native.freeze(plan, tmp_path / "missing.json")
    assert not plan.parent.exists()


@pytest.fixture
def native_case(sample):
    values, _, _ = sample
    model = student().eval()
    with torch.no_grad():
        model.memory_adapter.output_projection.weight.fill_(0.03)
    generate = model.language.generate

    def optional_mask_generate(*, encoder_outputs, attention_mask=None, **kwargs):
        if attention_mask is None:
            attention_mask = torch.ones(
                encoder_outputs.last_hidden_state.shape[:2], dtype=torch.long
            )
        return generate(encoder_outputs=encoder_outputs, attention_mask=attention_mask, **kwargs)

    model.language.generate = optional_mask_generate
    values["streams"]["pose_features"][0, 0] = -0.0
    cached = {
        "row": {
            "source": "synthetic",
            "example_id": "0",
            "reference": "a",
            "role": "training",
            "training_allowed": True,
        },
        "streams": values["streams"],
        "valid_intervals": values["valid_intervals"],
        "frame_indices": np.arange(0, 26, 2, dtype=np.int64),
        "header": {"sampled_frames": 13, "source_frames": 26, "source_fps": 25},
    }
    return model, cached, copy.deepcopy(cached)


def test_nonzero_timing_native_interface_matches_memory_logits_and_full_generation(native_case):
    model, cached, fresh = native_case
    generation = {
        "num_beams": 5,
        "max_length": 2048,
        "length_penalty": 0.6,
        "no_repeat_ngram_size": 16,
    }
    before = state_hashes(model.memory_adapter)
    result = native.compare_clip(model, Tokenizer(), cached, fresh, generation)
    assert result["native_generation_tokens_equal"] is True
    assert result["frames"] == 13
    assert result["semantic_memory_maximum_difference"] == 0
    assert result["teacher_forcing_logits_maximum_difference"] == 0
    assert result["base_semantic_memory_maximum_difference"] == 0
    assert state_hashes(model.memory_adapter) == before
    options = model.language.last_generate[-1]
    assert all(options[k] == v for k, v in generation.items())
    assert options["do_sample"] is False and options["early_stopping"] is True


@pytest.mark.parametrize(
    "change", ["row", "gap", "grid", "signed_pose", "feature", "mask_dtype", "missing"]
)
def test_native_comparison_rejects_source_or_pose_or_coverage_changes(native_case, change):
    model, cached, fresh = native_case
    if change == "row":
        fresh["row"]["example_id"] = "another"
    elif change == "gap":
        fresh["valid_intervals"][0] = False
    elif change == "grid":
        fresh["frame_indices"][-1] += 1
    elif change == "signed_pose":
        fresh["streams"]["pose_features"][0, 0] = 0.0
    elif change == "feature":
        fresh["streams"]["face_features"][3, 0] += 1
    elif change == "mask_dtype":
        fresh["valid_intervals"] = fresh["valid_intervals"].astype(np.uint8)
    else:
        del fresh["streams"]["face_features"]
    with pytest.raises(ValueError):
        native.compare_clip(model, Tokenizer(), cached, fresh, {})


def test_complete_native_controller_and_terminal_checker_on_synthetic_backbones(
    native_case, tmp_path, monkeypatch
):
    import json
    import os
    from types import SimpleNamespace

    import c5_small_input_vision as visual
    import c5_temporal_memory_model as semantic
    import c5_timing_native_input as inputs
    import evaluate_c5_words as words
    import prepare_c5_timing_features as features
    import probe_c5_small_input as legacy
    from probe_tiny_rgb_capacity import state_digest

    monkeypatch.setattr(data, "ROOT", tmp_path)
    monkeypatch.setattr(native, "ROOT", tmp_path)
    model, cached, _ = native_case
    appearance = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        appearance.weight.fill_(0.5)
    visual_hash = state_digest(appearance.state_dict())
    monkeypatch.setattr(visual, "SmallInputDino", lambda *args, **kwargs: appearance)
    monkeypatch.setattr(
        legacy,
        "legacy_models",
        lambda plan: (SimpleNamespace(model=None), SimpleNamespace(model=None)),
    )
    monkeypatch.setattr(
        semantic, "require_model", lambda model, plan, configuration: model.require_fixed()
    )
    monkeypatch.setattr(features, "guard", lambda *args: None)
    audit_path = tmp_path / "runs/audit.json"
    audit_path.parent.mkdir()
    audit_path.write_text('{"synthetic":true}')
    adapter_path = tmp_path / "checkpoints/adapter.pt"
    adapter_path.parent.mkdir()
    torch.save({"synthetic": True}, adapter_path)
    binding = {
        "adapter": data.artifact(adapter_path),
        "state_hashes": state_hashes(model.memory_adapter),
        "adapter_configuration": model.memory_adapter.configuration(),
    }
    monkeypatch.setattr(candidate, "load_candidate", lambda path: (model, Tokenizer(), binding))
    cache, pixels, records = {}, {}, []
    for index in native.INDICES:
        value = copy.deepcopy(cached)
        value["row"]["example_id"] = str(index)
        cache[index] = value
        px = {
            "header": value["header"],
            "pixels": np.zeros((13, 3, 224, 224, 3), dtype=np.uint8),
            "pose_features": value["streams"]["pose_features"],
            "frame_indices": value["frame_indices"],
            "valid_intervals": value["valid_intervals"],
            "source_hand_tip_visible": np.ones(26, dtype=np.bool_),
        }
        pixels[index] = px
        receipt = inputs.input_receipt(px)
        original = tmp_path / f"checkpoints/original-{index:04d}.json"
        original.write_text(json.dumps({"input": receipt}))
        records.append(
            {
                "row": value["row"],
                "header": value["header"],
                "receipt": data.artifact(original),
                "arrays": {k: v for k, v in receipt["arrays"].items() if k != "pixels"},
            }
        )

    class Dataset:
        def __init__(self, path):
            pass

        def load(self, index):
            return copy.deepcopy(cache[index])

    monkeypatch.setattr(native, "TimingFeatureDataset", Dataset)
    monkeypatch.setattr(
        inputs, "read_training224", lambda case: copy.deepcopy(pixels[case["index"]])
    )
    monkeypatch.setattr(
        words,
        "original_streams",
        lambda *args: {
            **copy.deepcopy(cached["streams"]),
            "frame_indices": cached["frame_indices"].copy(),
        },
    )
    monkeypatch.setattr(
        features,
        "native_reference",
        lambda *args: {k: v.copy() for k, v in cached["streams"].items() if k != "pose_features"},
    )
    plan = {
        "schema": "umi-studio-trained-temporal-memory-native-qualification-plan/1",
        "adapter": binding["adapter"],
        "adapter_configuration": model.memory_adapter.configuration(),
        "final_audit": data.artifact(audit_path),
        "data_audit": data.artifact(audit_path),
        "model_plan": {
            "pretrained_state_sha256": visual_hash,
            "reference": {
                "generation": {
                    "max_length": 2048,
                    "num_beams": 5,
                    "length_penalty": 0.6,
                    "no_repeat_ngram_size": 16,
                }
            },
        },
        "cases": [{"index": index} for index in native.INDICES],
        "records": records,
    }
    plan_path = tmp_path / "runs/native-plan.json"
    plan_path.write_text(json.dumps(plan))
    monkeypatch.setattr(native, "checked", lambda path: copy.deepcopy(plan))
    output = tmp_path / "checkpoints/native"
    output.mkdir()
    worker = {
        "uid": os.getuid(),
        "pid": 123,
        "argv": [
            sys.executable,
            str(tmp_path / "tools/qualify_c5_temporal_memory_native.py"),
            "--plan",
            str(plan_path),
            "--output",
            str(output),
        ],
    }
    (output / "worker.json").write_text(json.dumps(worker))
    native.qualify(plan_path, output, plan)
    (output / "exit.json").write_text(json.dumps({"pid": 123, "exit_code": 0}))
    poll_path = tmp_path / "runs/poll.json"
    poll = {
        "tool": "write_stdin",
        "requested_session_id": 9,
        "actual_owned_pid": 123,
        "tool_result": {"exit_code": 0},
    }
    poll_path.write_text(json.dumps(poll))
    _, result, evidence = native.checked_qualification(
        plan_path, output / "qualification.json", poll_path
    )
    assert result["trained_native_runtime_qualified"] is True and len(evidence) == 9
    assert result["development_or_confirmation_inputs"] == 0
    first_path = output / "case-00.json"
    first_bytes = first_path.read_bytes()
    result_path = output / "qualification.json"
    result_bytes = result_path.read_bytes()
    for change in ("live", "adapter", "pixels", "generation", "endpoints_only", "extra", "row"):
        changed_result, changed = copy.deepcopy(result), json.loads(first_bytes)
        if change == "live":
            live_poll = copy.deepcopy(poll)
            live_poll["tool_result"]["session_id"] = 9
            poll_path.write_text(json.dumps(live_poll))
        elif change == "adapter":
            changed_result["adapter"] = {"different": True}
        elif change == "pixels":
            changed["input"]["arrays"]["pixels"]["sha256"] = "0" * 64
        elif change == "generation":
            changed["comparison"]["generation"]["max_length"] = 64
        elif change == "endpoints_only":
            changed["complete_native_visual_comparison"]["sample_indices"] = [0, 12]
        elif change == "extra":
            (output / "unbound.json").write_text("{}")
        else:
            changed["comparison"]["row"]["example_id"] = "different"
        first_path.write_text(json.dumps(changed))
        changed_result["comparisons"][0] = data.artifact(first_path)
        result_path.write_text(json.dumps(changed_result))
        with pytest.raises(ValueError):
            native.checked_qualification(plan_path, result_path, poll_path)
        first_path.write_bytes(first_bytes)
        result_path.write_bytes(result_bytes)
        poll_path.write_text(json.dumps(poll))
        (output / "unbound.json").unlink(missing_ok=True)
