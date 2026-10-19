"""Release tests use synthetic audit claims; no detector or training executes."""
from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import prepare_c5_broad_landmarks as broad


@pytest.fixture
def release(tmp_path, monkeypatch):
    monkeypatch.setattr(broad, "ROOT", tmp_path)
    training = tmp_path / "runs/training/plan.json"
    output = tmp_path / "checkpoints/training"
    training.parent.mkdir(parents=True)
    output.mkdir(parents=True)
    monkeypatch.setattr(broad, "TRAINING", training)
    monkeypatch.setattr(broad, "TRAINING_OUTPUT", output)

    def save(path, value):
        path.write_text(json.dumps(value))

    def artifact(path):
        return {"path": str(path.relative_to(tmp_path)), "sha256": broad.sha256(path)}

    monkeypatch.setattr(broad, "artifact", artifact)
    auditor = training.with_name("auditor.py")
    auditor.write_text("# synthetic auditor\n")
    save(training, {"lineage_audit_source": artifact(auditor)})
    save(output / "exit.json", {"exit_code": 0})
    save(output / "worker.json", {"pid": 123456, "uid": os.getuid()})
    save(output / "result.json", {"synthetic": True})
    poll = training.with_name("poll.json")
    poll_value = {"tool": "write_stdin", "requested_session_id": 31973,
                  "tool_result": {"exit_code": 0, "output": ""}}
    save(poll, poll_value)
    audit = training.with_name("audit.json")
    audit_value = {
        "schema": "umi-studio-complete-word-training-terminal-audit/1",
        "complete": True, "same_worker_exit0": True, "training_examples": 856,
        "all_original_350_tensor_checkpoints_and_actual_reward_evidence_validated": True,
        "all_pinned_sources_and_model_assets_match": True,
        "raw_request_response_pairings_revalidated_by_checked_final": True,
        "audit_source": artifact(auditor), "plan": artifact(training),
        **{key: artifact(output / (key + ".json")) for key in ("worker", "exit", "result")},
    }
    save(audit, audit_value)
    monkeypatch.setattr(broad.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout=""))
    return SimpleNamespace(poll=poll, audit=audit, poll_value=poll_value,
                           audit_value=audit_value, save=save, output=output)


def test_complete_synthetic_release_binds_both_actual_receipt_files(release):
    result = broad.admit_release(release.poll, release.audit)
    assert result == {"terminal_poll": broad.artifact(release.poll),
                      "terminal_audit": broad.artifact(release.audit)}


@pytest.mark.parametrize("changes", [
    {"requested_session_id": 1},
    {"tool": "exec_command"},
    {"tool_result": {"exit_code": None, "session_id": 31973}},
    {"tool_result": {"exit_code": 1}},
    {"tool_result": {"exit_code": False}},
    {"tool_result": {"exit_code": 0, "session_id": 31973}},
])
def test_live_wrong_failed_or_boolean_session_receipt_cannot_release(release, changes):
    release.save(release.poll, {**release.poll_value, **changes})
    with pytest.raises(ValueError, match="actually exit0"):
        broad.admit_release(release.poll, release.audit)


@pytest.mark.parametrize("field,value", [
    ("complete", False), ("training_examples", 686), ("same_worker_exit0", False),
    ("all_original_350_tensor_checkpoints_and_actual_reward_evidence_validated", False),
    ("all_pinned_sources_and_model_assets_match", False),
    ("raw_request_response_pairings_revalidated_by_checked_final", False),
])
def test_incomplete_or_unverified_training_cannot_release(release, field, value):
    release.save(release.audit, {**release.audit_value, field: value})
    with pytest.raises(ValueError, match="completed independent"):
        broad.admit_release(release.poll, release.audit)


def test_live_pid_or_modified_result_cannot_release(release, monkeypatch):
    monkeypatch.setattr(broad.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="123456"))
    with pytest.raises(ValueError, match="still present"):
        broad.admit_release(release.poll, release.audit)
    (release.output / "result.json").write_text('{"changed":true}')
    with pytest.raises(ValueError, match="evidence changed"):
        broad.admit_release(release.poll, release.audit)


def test_release_receipts_cannot_be_aliases(release):
    alias = release.poll.with_name("poll-alias.json")
    alias.symlink_to(release.poll)
    with pytest.raises(ValueError, match="canonical owned"):
        broad.admit_release(alias, release.audit)


def test_failed_batch_keeps_release_binding_with_its_worker(release, monkeypatch):
    def failing_prepare(path, data, receipts, job, limit):
        job.mkdir()
        globals()["write"](job / "worker.json", {"synthetic": True})
        raise RuntimeError("synthetic container failure")

    monkeypatch.setattr(broad.original, "prepare", failing_prepare)
    job = release.poll.parent / "failed-landmark-batch"
    with pytest.raises(RuntimeError, match="synthetic container"):
        broad.prepare(Path("unused"), Path("unused"), Path("unused"), job, 16,
                      release.poll, release.audit)
    assert json.loads((job / "training-release.json").read_text()) == {
        "terminal_poll": broad.artifact(release.poll),
        "terminal_audit": broad.artifact(release.audit),
    }


def test_one_cpu_offline_container_has_no_pull_and_preserves_readonly_sources(tmp_path):
    before = broad.original.container_command(docker_socket="unused", clip=broad.ROOT / "data/clip.mp4",
                                             output_name="clip.npz", landmark_dir=tmp_path)
    command = broad.container_command(docker_socket="unused", clip=broad.ROOT / "data/clip.mp4",
                                      output_name="clip.npz", landmark_dir=tmp_path)
    assert before[before.index("--cpus") + 1] == "2"
    assert command[command.index("--cpus") + 1] == "1"
    assert command[command.index("--network") + 1] == "none"
    assert command[command.index("--pull") + 1] == "never"
    assert command[command.index("--memory") + 1] == "3g"
    assert f"{broad.ROOT}:/workspace:ro" in command
    assert "OMP_NUM_THREADS=1" in command and "PYTHONDONTWRITEBYTECODE=1" in command
    assert broad.POLICY["training_allowed"] is False


def test_sidecar_receipt_keeps_confirmation_role_and_false_training_permission(tmp_path):
    value = {"schema": "umi-studio-word-landmark-receipt/1", "model_training_ready": False}
    original = copy.deepcopy(value)
    path = tmp_path / "receipt.json"
    broad.write(path, value)
    result = json.loads(path.read_text())
    assert value == original
    assert result["schema"] == "umi-studio-broad-confirmation-landmark-receipt/1"
    assert result["training_allowed"] is result["confirmation_consumed"] is False

