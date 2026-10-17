from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import evaluate_roi_broad_semantic as evaluation


def fit_and_result():
    final = {"path": "checkpoints/final.safetensors", "sha256": "final-sha"}
    plan = {
        "planned_training_examples": 1668,
        "unique_training_clips": 318,
        "maximum_optimizer_updates": 420,
        "vision_initialization_sha256": "init-sha",
    }
    result = {
        "complete": True,
        "plan_sha256": "plan-sha",
        "training_examples": 1668,
        "unique_training_clips": 318,
        "optimizer_updates": 418,
        "maximum_optimizer_updates": 420,
        "all_fixed_backbones_unchanged": True,
        "semantic_translation_evaluated": False,
        "promotion_authorized": False,
        "visual": final,
        "checkpoints": [final],
        "visual_initialization_sha256": "init-sha",
    }
    return plan, result


def test_fixed_final_allows_legitimate_tied_reward_skips_and_keeps_complete_schedule():
    plan, result = fit_and_result()
    evaluation.require_final(result, plan, "plan-sha")
    assert result["optimizer_updates"] < plan["maximum_optimizer_updates"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("complete", False),
        ("plan_sha256", "wrong"),
        ("training_examples", 396),
        ("unique_training_clips", 99),
        ("optimizer_updates", True),
        ("optimizer_updates", 0),
        ("optimizer_updates", 421),
        ("maximum_optimizer_updates", 22),
        ("all_fixed_backbones_unchanged", False),
        ("semantic_translation_evaluated", True),
        ("promotion_authorized", True),
        ("visual", None),
        ("checkpoints", []),
        ("visual_initialization_sha256", "other-initializer"),
    ],
)
def test_intermediate_incomplete_relabelled_or_promoted_models_are_rejected(field, value):
    plan, result = fit_and_result()
    result[field] = value
    with pytest.raises(ValueError, match="completed fixed-final"):
        evaluation.require_final(result, plan, "plan-sha")


def test_visual_must_be_the_actual_final_checkpoint_not_an_earlier_favorite():
    plan, result = fit_and_result()
    result["visual"] = {"path": "checkpoints/intermediate.safetensors", "sha256": "older"}
    with pytest.raises(ValueError):
        evaluation.require_final(result, plan, "plan-sha")


def complete_steps():
    batches = [
        {"stage": "short", "epoch": 0, "training_record_indices": [0, 1]},
        {"stage": "broad", "epoch": 0, "training_record_indices": [1]},
        {"stage": "broad", "epoch": 1, "training_record_indices": [0, 1]},
    ]
    plan = {"records": [{}, {}], "batches": batches, "planned_training_examples": 5}
    steps = []
    for number, batch in enumerate(batches):
        steps.append(
            {
                **copy.deepcopy(batch),
                "batch": number,
                "Terra_qualities": [[10, 0], [10, 10]] if number != 1 else [[10, 10]],
                "gradient_norm_before_clip": None if number == 1 else 0.3,
                "contrasting_clips": 0 if number == 1 else 1,
                "optimizer_updates": [1, 1, 2][number],
                "training_examples": [2, 3, 5][number],
            }
        )
    return plan, steps, {"checkpoints": ["final"], "optimizer_updates": 2}


def test_complete_step_join_checks_planned_repeats_and_legitimate_zero_contrast_batches():
    plan, steps, result = complete_steps()
    evaluation.verify_steps(plan, steps, result)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "reordered",
        "wrong-epoch",
        "wrong-stage",
        "wrong-row",
        "wrong-batch",
        "missing-quality",
        "noninteger-quality",
        "anchor",
        "contrast-counter",
        "update-counter",
        "exposure-counter",
        "zero-gradient",
        "nonfinite-gradient",
        "tied-gradient",
        "final-counter",
    ],
)
def test_completion_receipt_cannot_replace_full_actual_order_and_update_evidence(mutation):
    plan, steps, result = complete_steps()
    if mutation == "missing":
        steps.pop()
    elif mutation == "reordered":
        steps[0], steps[1] = steps[1], steps[0]
    elif mutation == "wrong-epoch":
        steps[0]["epoch"] = 1
    elif mutation == "wrong-stage":
        steps[0]["stage"] = "other"
    elif mutation == "wrong-row":
        steps[0]["training_record_indices"] = [1, 0]
    elif mutation == "wrong-batch":
        steps[0]["batch"] = 1
    elif mutation == "missing-quality":
        steps[0]["Terra_qualities"].pop()
    elif mutation == "noninteger-quality":
        steps[0]["Terra_qualities"][0][1] = True
    elif mutation == "anchor":
        steps[0]["Terra_qualities"][0][0] = 9
    elif mutation == "contrast-counter":
        steps[0]["contrasting_clips"] = 2
    elif mutation == "update-counter":
        steps[1]["optimizer_updates"] = 2
    elif mutation == "exposure-counter":
        steps[0]["training_examples"] = 3
    elif mutation == "zero-gradient":
        steps[0]["gradient_norm_before_clip"] = 0
    elif mutation == "nonfinite-gradient":
        steps[0]["gradient_norm_before_clip"] = float("nan")
    elif mutation == "tied-gradient":
        steps[1]["gradient_norm_before_clip"] = 1
    else:
        result["optimizer_updates"] = 3
    with pytest.raises(ValueError):
        evaluation.verify_steps(plan, steps, result)


@pytest.mark.parametrize("name", ["infer", "score_specification", "run"])
def test_shared_phase_has_identical_bytecode_and_only_private_control_callback_changes(name):
    original = getattr(evaluation.shared, name)
    previous = dict(original.__globals__)
    bound = evaluation.bound_phase(name)
    assert bound.__code__ is original.__code__
    assert bound.__defaults__ is original.__defaults__
    assert bound.__kwdefaults__ == original.__kwdefaults__
    assert bound.__closure__ is original.__closure__
    replacements = {"checked"} if name != "run" else {"checked", "checked_scores", "infer"}
    assert set(bound.__globals__) == set(previous)
    assert {k for k, v in previous.items() if bound.__globals__[k] is not v} == replacements
    assert all(original.__globals__[k] is v for k, v in previous.items())
    assert bound.__globals__["checked"] is evaluation.checked
    if name == "run":
        assert bound.__globals__["infer"].__code__ is evaluation.shared.infer.__code__
        assert bound.__kwdefaults__["scoring"] is False


def test_gate_crops_generation_rubric_and_pacing_are_not_redefined():
    assert evaluation.shared.SCORE_POLICY["judge_model"] == "gpt-5.6-terra"
    assert evaluation.shared.SCORE_POLICY["reasoning_effort"] == "medium"
    for key, value in evaluation.shared.POLICY.items():
        if key != "candidate":
            assert evaluation.POLICY[key] == value
    assert evaluation.POLICY["WER_is_diagnostic_only"] is True
    assert evaluation.POLICY["promotion_authorized"] is False
    for name in ("infer", "score_specification", "run"):
        bound = evaluation.bound_phase(name)
        for helper in ("decode", "crops", "score_comparison", "semantic_decision", "RecoveryPacer"):
            assert bound.__globals__[helper] is getattr(evaluation.shared, helper)
    with pytest.raises(KeyError):
        evaluation.bound_phase("fallback")


@pytest.mark.parametrize("failure", [None, "inference", "scoring"])
def test_pipeline_scores_once_only_after_complete_inference_and_never_retries(
    tmp_path, monkeypatch, failure
):
    path = tmp_path / "runs/evaluation/plan.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({"source_sha256": {"tools/evaluate_roi_broad_semantic.py": "bound"}})
    )
    monkeypatch.setattr(evaluation, "ROOT", tmp_path)
    monkeypatch.setattr(evaluation, "verify_snapshot", lambda *args: None)
    monkeypatch.setattr(evaluation, "storage", lambda: None)
    monkeypatch.setattr(evaluation, "identity", lambda *args: {"pid": 1})
    monkeypatch.setattr(evaluation, "sha256", lambda *args: "bound")
    monkeypatch.setattr(evaluation, "artifact", lambda p: {"path": str(p), "sha256": "bound"})
    calls, freezes = [], []

    def child(command, check):
        assert check is False
        phase = "scoring" if "--score" in command else "inference"
        calls.append(phase)
        return SimpleNamespace(returncode=1 if failure == phase else 0)

    def score_spec(name):
        assert name == "score_specification"
        assert calls == ["inference"]
        return lambda *args: {"actual_inference_complete": True}

    monkeypatch.setattr(evaluation.subprocess, "run", child)
    monkeypatch.setattr(evaluation, "bound_phase", score_spec)
    monkeypatch.setattr(evaluation, "freeze", lambda p, s: freezes.append((p, s)))
    output = tmp_path / "checkpoints/evaluation"
    if failure:
        with pytest.raises(RuntimeError, match="no retry"):
            evaluation.pipeline(path, output)
        assert not (output / "complete.json").exists()
    else:
        evaluation.pipeline(path, output)
        assert json.loads((output / "complete.json").read_text())["promotion_authorized"] is False
    assert calls == (["inference"] if failure == "inference" else ["inference", "scoring"])
    assert len(freezes) == (0 if failure == "inference" else 1)
