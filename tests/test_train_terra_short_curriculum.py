from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import train_terra_short_curriculum as module


@pytest.fixture
def inputs(monkeypatch, tmp_path):
    reference = {"generation": {"num_beams": 5}}
    row = {
        "source": "stem",
        "example_id": "one",
        "source_group": "stem:P4",
        "signer": "P4",
        "reference": "Carbon telephone transmitter",
        "role": "development",
        "training_allowed": False,
    }
    data = {
        "records": [row],
        "reference": reference,
        "source_sha256": {},
        "resource_policy": "budget.json",
        "resource_policy_sha256": "budgethash",
        "training_clips": 116,
    }
    data_path, control_path = tmp_path / "data.json", tmp_path / "receipt.json"
    proof_plan, proof = tmp_path / "proof-plan.json", tmp_path / "proof.json"
    data_path.write_text("{}")
    proof_plan.write_text(
        json.dumps(
            {
                "reference": reference,
                "initial_reference_hashes": {"language": "initial", "encoder": "initial"},
            }
        )
    )
    proof.write_text(
        json.dumps(
            {
                "complete": True,
                "plan_sha256": module.sha256(proof_plan),
                "actual_Terra_only_gradients": True,
                "numerics_passed": True,
                "optimizer_updates": 0,
                "caption_weight_during_proof": 0,
            }
        )
    )
    control = {
        "complete": True,
        "all_weights_unchanged": True,
        "optimizer_updates": 0,
        "GPU_use": False,
        "plan_sha256": module.sha256(data_path),
        "original_C5_generation": reference["generation"],
        "clips": 1,
        "predictions": [{**row, "hypothesis": "Carbon phone transmitter", "WER_diagnostic": 0.5}],
    }
    control_path.write_text(json.dumps(control))
    (tmp_path / "exit.json").write_text(json.dumps({"exit_code": 0}))
    monkeypatch.setattr(module, "checked_data", lambda _: copy.deepcopy(data))
    monkeypatch.setattr(module, "PROOF_PLAN", proof_plan)
    monkeypatch.setattr(module, "PROOF", proof)
    monkeypatch.setattr(module, "artifact", lambda p: {"path": str(p), "sha256": module.sha256(p)})
    return data_path, control_path, proof


def test_pilot_fixes_full_short_coverage_and_semantic_checks_without_WER_veto(inputs):
    data, control, _ = inputs
    result = module.specification(data, control)
    assert result["planned_training_examples"] == 464
    assert result["checks_after_examples"] == [116, 232, 464]
    assert result["WER_is_diagnostic_only"]
    assert not result["new_data_downloads"]
    assert not result["promotion_authorized"]
    assert result["recipe"]["judge_model"] == "gpt-5.6-terra"


@pytest.mark.parametrize(
    "kind",
    ["incomplete", "weights", "updates", "GPU", "generation", "coverage", "role", "terminal"],
)
def test_unfinished_or_changed_C5_control_cannot_authorize_training(inputs, kind):
    data, control, _ = inputs
    value = json.loads(control.read_text())
    field, changed = {
        "incomplete": ("complete", False),
        "weights": ("all_weights_unchanged", False),
        "updates": ("optimizer_updates", 1),
        "GPU": ("GPU_use", True),
        "generation": ("original_C5_generation", {"num_beams": 1}),
        "coverage": ("clips", 0),
        "role": ("complete", True),
        "terminal": ("complete", True),
    }[kind]
    value[field] = changed
    if kind == "role":
        value["predictions"][0]["training_allowed"] = True
    if kind == "terminal":
        control.with_name("exit.json").write_text(json.dumps({"exit_code": 1}))
    control.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        module.specification(data, control)


@pytest.mark.parametrize(
    "field,value",
    [
        ("complete", False),
        ("actual_Terra_only_gradients", False),
        ("numerics_passed", False),
        ("caption_weight_during_proof", 0.1),
        ("optimizer_updates", 1),
        ("plan_sha256", "changed"),
    ],
)
def test_unqualified_or_caption_only_proof_cannot_authorize_training(inputs, field, value):
    data, control, proof = inputs
    saved = json.loads(proof.read_text())
    saved[field] = value
    proof.write_text(json.dumps(saved))
    with pytest.raises(ValueError):
        module.specification(data, control)
