from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import train_terra_c5_tail as trainer
from evaluate_terra_c5_tail import decision

ROOT = Path(__file__).resolve().parents[1]


def test_complete_and_partial_quality_gate_operations_are_identical_to_visual_evaluator():
    old = ast.parse((ROOT / "tools/evaluate_terra_visual.py").read_text())
    new = ast.parse((ROOT / "tools/evaluate_terra_c5_tail.py").read_text())
    old_run = next(n for n in old.body if isinstance(n, ast.FunctionDef) and n.name == "run")
    new_decision = next(
        n for n in new.body if isinstance(n, ast.FunctionDef) and n.name == "decision"
    )
    old_gate = next(
        n
        for n in old_run.body
        if isinstance(n, ast.If) and isinstance(n.test, ast.Name) and n.test.id == "complete"
    )
    new_gate = next(
        n
        for n in new_decision.body
        if isinstance(n, ast.If) and isinstance(n.test, ast.Name) and n.test.id == "complete"
    )
    assert ast.dump(old_gate, include_attributes=False) == ast.dump(
        new_gate, include_attributes=False
    )


def test_generation_and_resource_policies_do_not_import_training_horizon_into_evaluation():
    from evaluate_tiny_rgb_feature_winner import CPU_LIMITS
    from terra_teacher_candidates import DECODE

    assert trainer.RECIPE["training_generation"] == DECODE
    assert trainer.RECIPE["training_cpu_limits"] == CPU_LIMITS
    tree = ast.parse((ROOT / "tools/evaluate_terra_c5_tail.py").read_text())
    generate = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "generate"
    )
    arguments = {n.arg: ast.unparse(n.value) for n in generate.keywords}
    assert arguments[None] == "reference['generation']"
    assert "max_length" not in arguments and "num_beams" not in arguments
    assert arguments["early_stopping"] == "True" and arguments["do_sample"] == "False"


def test_previous_gpu_worker_requires_real_terminal_receipt(tmp_path, monkeypatch):
    monkeypatch.setattr(trainer, "ROOT", tmp_path)
    directory = tmp_path / "runs/previous-job"
    directory.mkdir(parents=True)
    monkeypatch.setattr(trainer, "PREDECESSOR", directory)
    directory.joinpath("worker.json").write_text(
        json.dumps({"pid": 123, "uid": 502, "ps_started_local": "old start"})
    )
    with pytest.raises(ValueError, match="terminal receipt"):
        trainer.predecessor()
    directory.joinpath("exit.json").write_text(json.dumps({"pid": 123, "exit_code": 0}))
    monkeypatch.setattr(trainer.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout=""))
    result = trainer.predecessor()
    assert set(result) == {"worker", "terminal"}
    responses = iter(["502\n", "old start\n"])
    monkeypatch.setattr(
        trainer.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout=next(responses))
    )
    with pytest.raises(RuntimeError, match="still present"):
        trainer.predecessor()


def test_os_gpu_lease_refuses_another_owned_tail_job(tmp_path, monkeypatch):
    monkeypatch.setattr(trainer, "ROOT", tmp_path)
    tmp_path.joinpath("runs").mkdir()
    with trainer.gpu_lease():
        with pytest.raises(RuntimeError, match="holds the lease"):
            with trainer.gpu_lease():
                pytest.fail("must not admit overlapping GPU workers")
    with trainer.gpu_lease():
        pass


def test_failed_checkpoint_evaluation_must_be_repeated_after_restore():
    resumed = {"training_examples": 512, "visual": {"sha256": "saved"}}
    assert trainer.missing_resume_evaluation(resumed, [])
    assert trainer.missing_resume_evaluation(resumed, [{"checkpoint_sha256": "other"}])
    assert not trainer.missing_resume_evaluation(resumed, [{"checkpoint_sha256": "saved"}])
    assert not trainer.missing_resume_evaluation({**resumed, "training_examples": 520}, [])


def test_incomplete_evaluation_cannot_claim_a_score_without_mathematical_rejection():
    import audit_winner_student_prefix

    original = audit_winner_student_prefix.quality_bounds
    audit_winner_student_prefix.quality_bounds = lambda *a: {
        "full_quality_gate_mathematically_impossible": False
    }
    try:
        with pytest.raises(ValueError, match="proven frozen-gate rejection"):
            decision(
                {
                    "evaluation": [1],
                    "winner_control": [],
                    "historical_controls": {
                        "original_checkpoint_common_inputs": [],
                        "selected_parent": [],
                    },
                },
                [],
                plan_sha="p",
                checkpoint_sha="c",
            )
    finally:
        audit_winner_student_prefix.quality_bounds = original
