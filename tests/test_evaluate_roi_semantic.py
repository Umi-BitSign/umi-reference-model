from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import evaluate_roi_semantic as evaluate
from score_roi_short_translation import POLICY as ORIGINAL_SCORE_POLICY


def result():
    return {
        "complete": True,
        "plan_sha256": "bound",
        "training_examples": 88,
        "optimizer_updates": 22,
        "all_fixed_backbones_unchanged": True,
        "semantic_translation_evaluated": False,
        "promotion_authorized": False,
        "visual": {"path": "checkpoints/final.safetensors", "sha256": "visual"},
    }


def test_only_completed_fixed_final_fit_is_eligible():
    evaluate.require_final(result(), "bound")
    changed = result()
    changed["optimizer_updates"] = 21  # Whole tied-reward batches legitimately have no step.
    evaluate.require_final(changed, "bound")


@pytest.mark.parametrize(
    "field,value",
    (
        ("complete", False),
        ("plan_sha256", "changed"),
        ("training_examples", 84),
        ("optimizer_updates", 0),
        ("optimizer_updates", 23),
        ("optimizer_updates", True),
        ("all_fixed_backbones_unchanged", False),
        ("semantic_translation_evaluated", True),
        ("promotion_authorized", True),
        ("visual", None),
    ),
)
def test_partial_mutated_or_different_plan_fit_cannot_be_scored(field, value):
    changed = result()
    changed[field] = value
    with pytest.raises(ValueError):
        evaluate.require_final(changed, "bound")


def test_visual_prefix_mapping_preserves_exact_values_without_rewriting_weights():
    expected = {"face.weight": torch.zeros(2, 3), "hands.bias": torch.zeros(2)}
    saved = {f"vision.{name}": torch.ones_like(value) for name, value in expected.items()}
    actual = evaluate.appearance_state(saved, expected)
    assert set(actual) == set(expected)
    assert all(actual[name] is saved[f"vision.{name}"] for name in expected)
    saved["language.weight"] = torch.zeros(1)
    with pytest.raises(ValueError):
        evaluate.appearance_state(saved, expected)


@pytest.mark.parametrize("kind", ("shape", "dtype", "nonfinite", "missing"))
def test_visual_state_validation_cannot_load_partial_or_invalid_parameters(kind):
    expected = {"face.weight": torch.zeros(2, 3)}
    saved = {"vision.face.weight": torch.ones(2, 3)}
    if kind == "shape":
        saved["vision.face.weight"] = torch.zeros(3, 2)
    elif kind == "dtype":
        saved["vision.face.weight"] = saved["vision.face.weight"].double()
    elif kind == "nonfinite":
        saved["vision.face.weight"][0, 0] = float("nan")
    else:
        saved.clear()
    with pytest.raises(ValueError):
        evaluate.appearance_state(saved, expected)


def control():
    row = {
        "source": "asl_stem_original",
        "example_id": "id",
        "source_group": "g",
        "signer": "P4",
        "reference": "Some text",
        "role": "development",
        "training_allowed": False,
    }
    case = {
        **evaluate.prediction(row, "Some text"),
        "tokens": [2, 3, 1],
        "matches_frozen_control": True,
    }
    return row, case


class Tokenizer:
    def decode(self, tokens, skip_special_tokens):
        assert skip_special_tokens is True
        return "Some text"


def test_actual_C5_control_tokens_identity_and_frozen_text_must_agree():
    row, case = control()
    evaluate.verified_control(row, case, "Some text", Tokenizer())
    with pytest.raises(ValueError):
        evaluate.verified_control(row, case, "Different frozen text", Tokenizer())
    changed = copy.deepcopy(case)
    changed["source_group"] = "other"
    with pytest.raises(ValueError):
        evaluate.verified_control(row, changed, "Some text", Tokenizer())


@pytest.mark.parametrize("tokens", ([], [True], [384], [-1], [1] * 2049))
def test_control_cannot_lose_or_fabricate_token_coverage(tokens):
    row, case = control()
    case["tokens"] = tokens
    with pytest.raises(ValueError):
        evaluate.verified_control(row, case, "Some text", Tokenizer())


def test_gate_and_judge_policy_are_the_original_semantic_gate():
    assert evaluate.SCORE_POLICY is ORIGINAL_SCORE_POLICY
    assert evaluate.SCORE_POLICY["passes"] == 2
    assert evaluate.SCORE_POLICY["judge_model"] == "gpt-5.6-terra"
    assert evaluate.SCORE_POLICY["reasoning_effort"] == "medium"
    assert evaluate.SCORE_POLICY["WER_is_diagnostic_only"]
    assert evaluate.POLICY["optimizer_updates"] == 0
    assert evaluate.POLICY["promotion_authorized"] is False


@pytest.mark.parametrize("inference_exit", (0, 1))
def test_pipeline_scores_only_after_terminal_inference_and_never_retries(
    tmp_path, monkeypatch, inference_exit
):
    path = tmp_path / "runs/inference/plan.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"source_sha256": {"tools/evaluate_roi_semantic.py": "bound"}}))
    monkeypatch.setattr(evaluate, "ROOT", tmp_path)
    monkeypatch.setattr(evaluate, "verify_snapshot", lambda *args: None)
    monkeypatch.setattr(evaluate, "sha256", lambda *args: "bound")
    monkeypatch.setattr(evaluate, "storage", lambda: None)
    monkeypatch.setattr(evaluate, "identity", lambda *args: {"pid": 1})
    monkeypatch.setattr(evaluate, "artifact", lambda path: {"path": str(path), "sha256": "bound"})
    calls, freeze_calls = [], []

    def child(command, check):
        assert check is False
        calls.append(command)
        return SimpleNamespace(returncode=inference_exit if len(calls) == 1 else 0)

    monkeypatch.setattr(evaluate.subprocess, "run", child)
    monkeypatch.setattr(evaluate, "score_specification", lambda *args: {"scope": "same34"})
    monkeypatch.setattr(evaluate, "freeze", lambda *args: freeze_calls.append(args))
    output = tmp_path / "checkpoints/job"
    if inference_exit:
        with pytest.raises(RuntimeError, match="scoring not started"):
            evaluate.pipeline(path, output)
        assert len(calls) == 1 and not freeze_calls
    else:
        evaluate.pipeline(path, output)
        assert len(calls) == 2 and len(freeze_calls) == 1
        assert "--score" not in calls[0] and "--score" in calls[1]
        assert json.loads((output / "complete.json").read_text())["promotion_authorized"] is False
    assert json.loads((output / "exit.json").read_text())["exit_code"] == inference_exit
