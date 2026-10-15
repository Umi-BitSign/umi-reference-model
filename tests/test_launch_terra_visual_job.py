from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import launch_terra_visual_job as module


def setup_plan(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "ROOT", tmp_path)
    plan = tmp_path / "runs/frozen/plan.json"
    plan.parent.mkdir(parents=True)
    plan.write_text(json.dumps({"schema": "umi-studio-terra-visual-training/1", "resume": {}}))
    snapshot = plan.with_name("source-snapshot")
    snapshot.mkdir()
    snapshot.joinpath("manifest.json").write_text(json.dumps({"plan_sha256": module.digest(plan)}))
    return plan


def test_launch_detaches_terminal_and_records_exact_child_intent(tmp_path, monkeypatch):
    plan = setup_plan(tmp_path, monkeypatch)
    calls = []

    def spawn(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(pid=123)

    monkeypatch.setattr(module.subprocess, "Popen", spawn)
    monkeypatch.setattr(module, "identity", lambda pid, command: {"pid": pid, "argv": command})
    job, output = tmp_path / "runs/job", tmp_path / "checkpoints/recovery"
    module.launch(plan, output, job, restore_only=True)
    command, kwargs = calls[0]
    assert kwargs["start_new_session"] is True
    assert kwargs["stdin"] == module.subprocess.DEVNULL
    assert kwargs["stderr"] == module.subprocess.STDOUT
    assert command[-2:] == ["--supervise", str(job)]
    record = json.loads(job.joinpath("job.json").read_text())
    assert record["automatic_restarts"] == 0 and record["restore_only"] is True
    assert record["command"][-1] == "--restore-only"
    assert record["plan_sha256"] == module.digest(plan)
    assert json.loads(job.joinpath("supervisor.json").read_text())["pid"] == 123


@pytest.mark.parametrize("bad_target", ["existing_job", "existing_output", "outside"])
def test_launch_refuses_replay_or_unowned_paths(tmp_path, monkeypatch, bad_target):
    plan = setup_plan(tmp_path, monkeypatch)
    job, output = tmp_path / "runs/job", tmp_path / "checkpoints/recovery"
    if bad_target == "existing_job":
        job.mkdir()
    elif bad_target == "existing_output":
        output.mkdir(parents=True)
    else:
        job = tmp_path / "outside"
    monkeypatch.setattr(module.subprocess, "Popen", lambda *a, **kw: pytest.fail("must not launch"))
    with pytest.raises(ValueError, match="fresh owned"):
        module.launch(plan, output, job)


def test_supervisor_records_terminal_failure_without_retry(tmp_path, monkeypatch):
    plan = setup_plan(tmp_path, monkeypatch)
    job = tmp_path / "runs/job"
    job.mkdir()
    command = [sys.executable, str(tmp_path / "tools/train_terra_visual.py"), "--plan", str(plan)]
    module.write(
        job / "job.json",
        {
            "cwd": str(tmp_path),
            "plan": str(plan),
            "plan_sha256": module.digest(plan),
            "launcher_sha256": module.digest(Path(module.__file__)),
            "command": command,
        },
    )
    calls = []

    def spawn(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(pid=456, wait=lambda: 17)

    monkeypatch.setattr(module.subprocess, "Popen", spawn)
    monkeypatch.setattr(module, "identity", lambda pid, command: {"pid": pid, "argv": command})
    assert module.supervise(job) == 17
    assert len(calls) == 1 and calls[0][1]["start_new_session"] is True
    assert json.loads(job.joinpath("worker.json").read_text())["pid"] == 456
    assert json.loads(job.joinpath("exit.json").read_text())["exit_code"] == 17
    plan.write_text("{}")
    with pytest.raises(ValueError, match="binding changed"):
        module.supervise(job)
