from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import evaluate_roi_semantic as shared
import score_c5_small_input as score


@pytest.mark.parametrize("name", ["specification", "run"])
def test_scoring_reuses_original_bytecode_and_every_scientific_global(name):
    original = shared.score_specification if name == "specification" else shared.run
    bound = score.bound_phase(name)
    assert bound.__code__ is original.__code__
    assert bound.__defaults__ == original.__defaults__
    assert bound.__kwdefaults__ == original.__kwdefaults__
    changed = {
        key
        for key in original.__globals__
        if bound.__globals__[key] is not original.__globals__[key]
    }
    assert changed == ({"checked"} if name == "specification" else {"checked_scores"})
    assert bound.__globals__["SCORE_POLICY"] is shared.SCORE_POLICY
    assert bound.__globals__["RecoveryPacer"] is shared.RecoveryPacer
    assert bound.__globals__["semantic_decision"] is shared.semantic_decision


def test_public_runner_can_only_enter_scoring_mode(monkeypatch):
    calls = []
    monkeypatch.setattr(
        score, "bound_phase", lambda name: lambda *a, **kw: calls.append((name, a, kw))
    )
    score.run(Path("plan.json"), Path("output"))
    assert calls == [("run", (Path("plan.json"), Path("output")), {"scoring": True})]


@pytest.mark.parametrize("inference_exit,scoring_exit", [(0, 0), (1, 0), (0, 1)])
def test_pipeline_waits_for_own_children_and_never_retries_or_scores_failed_inference(
    tmp_path, monkeypatch, inference_exit, scoring_exit
):
    plan = tmp_path / "runs/inference/plan.json"
    plan.parent.mkdir(parents=True)
    plan.write_text(json.dumps({"source_sha256": {"tools/score_c5_small_input.py": "source"}}))
    monkeypatch.setattr(score, "ROOT", tmp_path)
    monkeypatch.setattr(score, "verify_snapshot", lambda *args: None)
    monkeypatch.setattr(score, "sha256", lambda path: "source")
    monkeypatch.setattr(score, "storage", lambda: None)
    monkeypatch.setattr(score, "identity", lambda *args: {"pid": 1})
    monkeypatch.setattr(score, "artifact", lambda path: {"path": str(path), "sha256": "bound"})
    commands, freezes = [], []

    def child(command, check):
        assert check is False
        commands.append(command)
        return SimpleNamespace(returncode=inference_exit if len(commands) == 1 else scoring_exit)

    def freeze(*args):
        assert len(commands) == 1 and inference_exit == 0
        freezes.append(args)

    monkeypatch.setattr(score.subprocess, "run", child)
    monkeypatch.setattr(score, "freeze_scores", freeze)
    output = tmp_path / "checkpoints/pipeline"
    if inference_exit or scoring_exit:
        with pytest.raises(RuntimeError):
            score.pipeline(plan, output)
        assert len(commands) == (1 if inference_exit else 2)
        assert len(freezes) == (0 if inference_exit else 1)
        assert not (output / "complete.json").exists()
        assert json.loads((output / "failure.json").read_text())["automatic_retries"] == 0
        assert json.loads((output / "exit.json").read_text())["exit_code"] == 1
    else:
        score.pipeline(plan, output)
        assert len(commands) == 2 and len(freezes) == 1
        assert commands[0][1] == str(tmp_path / "tools/evaluate_c5_small_input.py")
        assert commands[1][1] == str(Path(score.__file__).resolve())
        completed = json.loads((output / "complete.json").read_text())
        assert completed["complete"] is True
        assert completed["optimizer_updates"] == completed["automatic_retries"] == 0
        assert completed["promotion_authorized"] is False
        assert json.loads((output / "exit.json").read_text())["exit_code"] == 0


def test_pipeline_rejects_unpinned_controller_before_any_child_launch(tmp_path, monkeypatch):
    plan = tmp_path / "runs/inference/plan.json"
    plan.parent.mkdir(parents=True)
    plan.write_text(json.dumps({"source_sha256": {"tools/score_c5_small_input.py": "old"}}))
    monkeypatch.setattr(score, "ROOT", tmp_path)
    monkeypatch.setattr(score, "verify_snapshot", lambda *args: None)
    monkeypatch.setattr(score, "sha256", lambda path: "changed")
    monkeypatch.setattr(
        score.subprocess, "run", lambda *args, **kw: pytest.fail("unexpected child")
    )
    with pytest.raises(ValueError, match="pinned source"):
        score.pipeline(plan, tmp_path / "checkpoints/job")


def test_checked_scores_requires_exact_actual_output_binding(tmp_path, monkeypatch):
    plan = {
        "inference_plan": {"path": "runs/inference/plan.json"},
        "decoded": {"path": "checkpoints/inference/receipt.json"},
        "policy": {"model": "gpt-5.6-terra", "effort": "medium"},
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    monkeypatch.setattr(score, "ROOT", tmp_path)
    monkeypatch.setattr(score, "verify_snapshot", lambda *args: None)
    monkeypatch.setattr(score, "bound_phase", lambda name: lambda *args: plan)
    assert score.checked_scores(path) == plan
    changed = {**plan, "policy": {"model": "different"}}
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="rubric or paired gate"):
        score.checked_scores(path)
