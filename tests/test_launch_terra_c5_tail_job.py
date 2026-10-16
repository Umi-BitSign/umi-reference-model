from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import launch_terra_c5_tail_job as module


def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "ROOT", tmp_path)
    plan = tmp_path / "runs/frozen/plan.json"
    plan.parent.mkdir(parents=True)
    plan.write_text(json.dumps({"schema": "umi-studio-terra-c5-tail-training/1"}))
    plan.with_name("source-snapshot").mkdir()
    plan.with_name("source-snapshot").joinpath("manifest.json").write_text(
        json.dumps({"plan_sha256": module.detached.digest(plan)})
    )
    qualification = tmp_path / "checkpoints/proof/qualification.json"
    qualification.parent.mkdir(parents=True)
    qualification.write_text(
        json.dumps(
            {
                "complete": True,
                "numerics_passed": True,
                "plan_sha256": module.detached.digest(plan),
                "optimizer_updates": 0,
                "caption_weight_during_proof": 0,
            }
        )
    )
    return plan, qualification


def test_initial_native_job_requires_its_actual_completed_proof(tmp_path, monkeypatch):
    plan, qualification = setup(tmp_path, monkeypatch)
    output, job = tmp_path / "checkpoints/full", tmp_path / "runs/job"
    with pytest.raises(ValueError, match="actual completed"):
        module.launch(plan, output, job)
    value = json.loads(qualification.read_text())
    value["plan_sha256"] = "different-model"
    qualification.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="different gradient proof"):
        module.launch(plan, output, job, qualification=qualification)
    assert not job.exists()


def test_native_launch_is_detached_and_retains_no_retry_policy(tmp_path, monkeypatch):
    plan, qualification = setup(tmp_path, monkeypatch)
    calls = []

    def spawn(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(pid=100)

    monkeypatch.setattr(module.subprocess, "Popen", spawn)
    monkeypatch.setattr(module.detached, "identity", lambda pid, argv: {"pid": pid, "argv": argv})
    job = tmp_path / "runs/job"
    module.launch(plan, tmp_path / "checkpoints/full", job, qualification=qualification)
    record = json.loads(job.joinpath("job.json").read_text())
    assert calls[0][1]["start_new_session"] and calls[0][1]["stdin"] == module.subprocess.DEVNULL
    assert record["automatic_restarts"] == 0
    assert record["command"][1].endswith("tools/train_terra_c5_tail.py")
    assert record["command"][-2:] == ["--qualification", str(qualification)]


def test_native_job_refuses_wrong_schema_before_spawning(tmp_path, monkeypatch):
    plan, qualification = setup(tmp_path, monkeypatch)
    plan.write_text(json.dumps({"schema": "visual-model"}))
    with pytest.raises(ValueError, match="own frozen model plan"):
        module.launch(
            plan, tmp_path / "checkpoints/full", tmp_path / "runs/job", qualification=qualification
        )
