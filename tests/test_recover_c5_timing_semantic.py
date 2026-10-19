"""Bounded explicit recovery checks; no network or real model invocation."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import c5_timing_feature_data as data
import recover_c5_timing_semantic as recovery
import test_train_c5_timing as training_tests
import train_c5_timing as training

qualification_fixture = training_tests.qualification_fixture
records = training_tests.records


def save(path, value):
    path.write_text(json.dumps(value))


def test_only_explicit_unanswered_502_is_recoverable():
    recovery.require_failure(
        {"http_status": 502, "ambiguous_completion": False, "automatic_retry_allowed": False}
    )


@pytest.mark.parametrize(
    "change", ["status", "boolean_status", "ambiguous", "missing_ambiguity", "automatic"]
)
def test_ambiguous_or_different_failure_never_releases_recovery(change):
    value = {"http_status": 502, "ambiguous_completion": False, "automatic_retry_allowed": False}
    if change == "status":
        value["http_status"] = 429
    elif change == "boolean_status":
        value["http_status"] = True
    elif change == "ambiguous":
        value["ambiguous_completion"] = True
    elif change == "missing_ambiguity":
        del value["ambiguous_completion"]
    else:
        value["automatic_retry_allowed"] = True
    with pytest.raises(ValueError, match="explicit unanswered"):
        recovery.require_failure(value)


@pytest.fixture
def recovered_qualification(qualification_fixture, monkeypatch):
    plan, result, poll = qualification_fixture
    metadata = data.read(plan)
    metadata["schema"] = recovery.SCHEMA
    predecessor = plan.parent / "predecessor/plan.json"
    predecessor.parent.mkdir()
    save(predecessor, {"synthetic": True})
    metadata["recovery"] = {"predecessor_plan": data.artifact(predecessor), "synthetic": True}
    save(plan, metadata)
    worker_path = result.parent / "worker.json"
    worker = data.read(worker_path)
    worker["argv"][1] = str(training.ROOT / "tools/recover_c5_timing_semantic.py")
    save(worker_path, worker)
    value = data.read(result)
    value.update(
        {
            "plan": data.artifact(plan),
            "worker": data.artifact(worker_path),
            "recovery": metadata["recovery"],
            "orchestrated_failed_request_replays": 1,
            "new_Terra_requests": 2,
            "API_attempts_including_predecessor": 3,
        }
    )
    save(result, value)
    claim = {
        "plan": data.artifact(plan),
        "worker": data.artifact(worker_path),
        "maximum_recovery_executions": 1,
        "orchestrated_failed_request_replays": 1,
    }
    save(plan.with_name("execution-claim.json"), claim)
    save(predecessor.with_name("grading-recovery-child-claim.json"), claim)
    monkeypatch.setattr(recovery, "checked", lambda path: copy.deepcopy(metadata))
    return plan, result, poll


def test_explicit_recovery_has_same_original_training_qualification_checks(recovered_qualification):
    plan, result, _ = training.qualification(*recovered_qualification)
    assert plan["schema"] == recovery.SCHEMA
    assert result["orchestrated_failed_request_replays"] == 1
    assert result["optimizer_updates"] == 0


@pytest.mark.parametrize(
    "change",
    ["hidden_replay", "second_replay", "wrong_attempt_count", "failure", "wrong_scope", "claim"],
)
def test_recovery_cannot_hide_replay_or_admit_second_failure(recovered_qualification, change):
    plan, result, poll = recovered_qualification
    value = data.read(result)
    if change == "hidden_replay":
        value["orchestrated_failed_request_replays"] = 0
    elif change == "second_replay":
        value["orchestrated_failed_request_replays"] = 2
    elif change == "wrong_attempt_count":
        value["API_attempts_including_predecessor"] = 2
    elif change == "failure":
        save(result.parent / "failure.json", {"http_status": 502})
    elif change == "wrong_scope":
        value["recovery"] = {"changed": True}
    else:
        claim = data.read(plan.with_name("execution-claim.json"))
        claim["maximum_recovery_executions"] = 2
        save(plan.with_name("execution-claim.json"), claim)
    save(result, value)
    with pytest.raises(ValueError):
        training.qualification(plan, result, poll)


def test_terminal_review_hash_and_all_predecessor_bytes_must_remain_bound(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "ROOT", tmp_path)
    path = tmp_path / "review.json"
    original = {"synthetic": "source"}
    save(path, {"evidence": original})
    monkeypatch.setattr(recovery, "REVIEW", path)
    monkeypatch.setattr(recovery, "REVIEW_SHA256", data.sha(path))
    assert recovery.verify_review(original) == data.artifact(path)
    with pytest.raises(ValueError, match="actual terminal review"):
        recovery.verify_review({"synthetic": "different"})
    save(path, {"evidence": original, "unbound": True})
    with pytest.raises(ValueError, match="actual terminal review"):
        recovery.verify_review(original)


@pytest.fixture
def runner(tmp_path, monkeypatch):
    import prepare_c5_timing_features as features
    import torch

    monkeypatch.setattr(data, "ROOT", tmp_path)
    monkeypatch.setattr(recovery, "ROOT", tmp_path)
    monkeypatch.setattr(features, "require_environment", lambda: None)
    monkeypatch.setattr(torch, "set_num_threads", lambda value: None)
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda value: None)
    monkeypatch.setattr(recovery.original, "guard", lambda output: None)
    monkeypatch.setattr(
        recovery, "qualify", lambda *args: pytest.fail("C5 model/API reached before release")
    )
    source = tmp_path / "runs/original/plan.json"
    source.parent.mkdir(parents=True)
    save(source, {"synthetic": True})
    monkeypatch.setattr(recovery, "PREDECESSOR_PLAN", source)
    plan = tmp_path / "runs/new/plan.json"
    plan.parent.mkdir()
    save(plan, {"synthetic": True})
    output = tmp_path / "checkpoints/new"
    return plan, output, source


def test_five_minute_cooldown_precedes_model_and_execution_claim(runner, monkeypatch):
    plan, output, source = runner
    monkeypatch.setattr(
        recovery,
        "checked",
        lambda path: {"recovery": {"API_start_not_before": "2100-01-01T00:00:00+00:00"}},
    )
    with pytest.raises(ValueError, match="cooldown has not elapsed"):
        recovery.run(plan, output)
    assert not output.exists()
    assert not source.with_name("grading-recovery-child-claim.json").exists()


def test_existing_child_claim_forbids_another_recovery_before_model_or_API(runner, monkeypatch):
    plan, output, source = runner
    marker = source.with_name("grading-recovery-child-claim.json")
    save(marker, {"original_claim": True})
    monkeypatch.setattr(
        recovery,
        "checked",
        lambda path: {"recovery": {"API_start_not_before": "2000-01-01T00:00:00+00:00"}},
    )
    with pytest.raises(FileExistsError):
        recovery.run(plan, output)
    assert data.read(marker) == {"original_claim": True}
    assert data.read(output / "exit.json")["exit_code"] == 1
