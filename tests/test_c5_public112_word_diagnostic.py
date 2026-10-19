"""No real model, clip or grader execution in resolution-controller tests."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import evaluate_c5_public112_words as diagnostic


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def source(tmp_path, monkeypatch):
    paths = {
        "ORIGINAL_PLAN": tmp_path / "runs/original/plan.json",
        "BASELINE": tmp_path / "checkpoints/baseline/receipt.json",
        "FINAL": tmp_path / "checkpoints/final/complete.json",
        "DECLINE": tmp_path / "checkpoints/scoring/decision.json",
    }
    monkeypatch.setattr(diagnostic, "ROOT", tmp_path)
    monkeypatch.setattr(diagnostic, "artifact", lambda p: {"path": str(p), "sha256": "synthetic"})
    for name, path in paths.items():
        monkeypatch.setattr(diagnostic, name, path)
    source_file = tmp_path / "tools/evaluate_c5_public112_words.py"
    source_file.parent.mkdir()
    source_file.write_text("# synthetic source\n")
    monkeypatch.setattr(diagnostic, "__file__", str(source_file))
    plan = {
        "policy": {"synthetic_original_policy": True},
        "records": [{"training_allowed": False}],
        "reference": {"synthetic": "frozen"},
        "model_plan": {"pretrained_state_sha256": "original-public"},
        "candidate_visual_state_sha256": "rejected-trained",
        "candidate_visual": {"path": "rejected.safetensors"},
        "initial_reference_hashes": {"original": True},
        "source_sha256": {"old-source": "old-hash"},
    }
    monkeypatch.setattr(diagnostic.trained, "checked", lambda p: copy.deepcopy(plan))
    return SimpleNamespace(root=tmp_path, paths=paths, plan=plan)


def test_specification_cannot_run_before_current_serial_evaluation_exits(source):
    with pytest.raises(FileNotFoundError):
        diagnostic.specification()


def test_completed_release_requires_actual_same_worker_exit_and_calibrated_decline(source):
    paths = source.paths
    put(paths["FINAL"], {"complete": True, "evaluations": {"word": {}, "sentence": {}}})
    put(paths["FINAL"].with_name("worker.json"), {"pid": 123})
    put(paths["FINAL"].with_name("exit.json"), {"pid": 124, "exit_code": 0})
    put(paths["DECLINE"], {"calibration_passed": True, "stop_for_semantic_decline": True})
    with pytest.raises(ValueError):
        diagnostic.checked_release()
    put(paths["FINAL"].with_name("exit.json"), {"pid": 123, "exit_code": 0})
    assert diagnostic.checked_release()["complete"]["path"] == str(paths["FINAL"])
    put(paths["DECLINE"], {"calibration_passed": False, "stop_for_semantic_decline": True})
    with pytest.raises(ValueError, match="real regression"):
        diagnostic.checked_release()


def test_specification_preserves_inputs_but_selects_only_original_visual_values(
    source, monkeypatch
):
    monkeypatch.setattr(diagnostic, "checked_release", lambda: {"synthetic": "completed"})
    value = diagnostic.specification()
    assert value["records"] == source.plan["records"]
    assert value["model_plan"] == source.plan["model_plan"]
    assert value["candidate_visual_state_sha256"] == "original-public"
    assert "candidate_visual" not in value
    assert value["trained_or_partial_visual_checkpoint_loaded"] is False
    assert value["post_hoc_diagnostic_only"] is True
    assert source.plan["candidate_visual"]["path"] == "rejected.safetensors"


def test_appearance_uses_public_loader_and_rejects_different_values(monkeypatch):
    calls = []
    value = SimpleNamespace(state_dict=lambda: {"public": True})
    monkeypatch.setattr(
        diagnostic.native, "appearance_model", lambda p, h, f: calls.append((p, h, f)) or value
    )
    monkeypatch.setattr(diagnostic.original, "state_digest", lambda state: "original-public")
    plan = {"model_plan": {"public": True}, "candidate_visual_state_sha256": "original-public"}
    assert diagnostic.appearance(plan, "hand", "face") is value
    assert calls == [(plan["model_plan"], "hand", "face")]
    plan["candidate_visual_state_sha256"] = "trained"
    with pytest.raises(ValueError, match="exact original"):
        diagnostic.appearance(plan, "hand", "face")


def test_reused_baseline_keeps_its_original_plan_and_rejects_changed_inputs(source, monkeypatch):
    plan = {
        **source.plan,
        "baseline_control": diagnostic.artifact(source.paths["BASELINE"]),
        "original_evaluation_plan": diagnostic.artifact(source.paths["ORIGINAL_PLAN"]),
    }
    seen = []
    monkeypatch.setattr(
        diagnostic.original, "checked_baseline", lambda *args: seen.append(args) or "verified"
    )
    assert (
        diagnostic.checked_baseline(source.paths["BASELINE"], Path("new-plan"), plan, "tokenizer")
        == "verified"
    )
    assert seen[0][1] == source.paths["ORIGINAL_PLAN"]
    plan["records"] = [{"substituted": True}]
    with pytest.raises(ValueError, match="baseline inputs"):
        diagnostic.checked_baseline(source.paths["BASELINE"], Path("new-plan"), plan, "tokenizer")
    assert len(seen) == 1


def test_inference_binds_original_public_appearance_and_full_original_control_checks(monkeypatch):
    seen = []

    def binding(function, **kwargs):
        seen.append((function, kwargs))
        return lambda *args: "synthetic-inference"

    monkeypatch.setattr(diagnostic, "bind", binding)
    assert diagnostic.infer("plan", "output", {}, "baseline") == "synthetic-inference"
    assert seen[0][0] is diagnostic.original.infer.__wrapped__
    assert seen[0][1] == {
        "checked": diagnostic.checked,
        "candidate_appearance": diagnostic.appearance,
        "checked_baseline": diagnostic.checked_baseline,
    }


def test_run_keeps_existing_resource_and_gpu_lease_controller(monkeypatch):
    seen = []

    def binding(function, **kwargs):
        return lambda *args: seen.append((function, kwargs, args))

    monkeypatch.setattr(diagnostic, "bind", binding)
    diagnostic.run("plan", "output", "inference")
    assert seen[0][0] is diagnostic.original.run
    assert seen[0][1]["infer"] is diagnostic.infer
    assert seen[0][2] == ("plan", "output", "inference", diagnostic.BASELINE)


def test_scoring_binds_unchanged_polarity_pacer_and_authenticator(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(
        diagnostic, "wait_after_predecessor", lambda: {"synthetic_gap_verified": True}
    )
    monkeypatch.setattr(diagnostic, "artifact", lambda p: {"path": str(p), "sha256": "synthetic"})

    def binding(function, **kwargs):
        seen.append((function, kwargs))
        return lambda *args: None

    monkeypatch.setattr(diagnostic, "bind", binding)
    diagnostic.score("plan", tmp_path, {})
    assert seen[0][0] is diagnostic.original.score
    assert seen[0][1]["RecoveryPacer"] is diagnostic.Pacer
    assert seen[0][1]["authenticated_passes"] is diagnostic.authenticated_passes
    assert json.loads((tmp_path / "API-spacing.json").read_text())["synthetic_gap_verified"] is True
    assert (
        json.loads((tmp_path / "diagnostic-scope.json").read_text())[
            "trained_model_win_established"
        ]
        is False
    )
