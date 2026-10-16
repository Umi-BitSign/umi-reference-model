from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import roi_appearance_protocol as protocol
import train_roi_appearance as trainer


def metric(loss, gap, teacher=0.2):
    return {"training": {"loss": loss, "temporal_gap": gap, "teacher_temporal_gap": teacher}}


def test_complete_frame_sampler_is_reproducible_without_diagnostic_frames():
    first = protocol.permutation(323, 0)
    assert np.array_equal(np.sort(first), np.arange(323))
    assert np.array_equal(first, protocol.permutation(323, 0))
    assert not np.array_equal(first, protocol.permutation(323, 1))
    for count, epoch in ((0, 0), (3, -1), (3, 64)):
        with pytest.raises(ValueError):
            protocol.permutation(count, epoch)


def test_feature_fit_is_not_a_translation_or_promotion_decision():
    result = protocol.decision(metric(1, 0), metric(0.2, 0.1))
    assert result["feature_feasibility_passed"] is True
    assert result["translation_quality_evaluated"] is False
    assert result["promotion_authorized"] is False
    assert protocol.GATE["later_translation_primary_metric"].startswith("Terra medium")


def test_feature_loss_reduction_cannot_pass_without_motion_correspondence():
    assert not protocol.decision(metric(1, 0), metric(0.01, 0))["feature_feasibility_passed"]
    assert not protocol.decision(metric(1, 0), metric(0.6, 0.2))["feature_feasibility_passed"]
    with pytest.raises(ValueError):
        protocol.decision(metric(1, 0), metric(0.1, 0.1, teacher=0))


def test_targets_copy_only_the_three_native_appearance_streams():
    values = {name: np.ones((3, 384), dtype=np.float32) for name in trainer.STREAMS}
    values["pose_features"] = np.ones((3, 14), dtype=np.float32)
    result = trainer.targets(values, np.array([0, 2]))
    assert set(result) == set(trainer.STREAMS)
    assert all(v.shape == (2, 384) and not v.requires_grad for v in result.values())
    result["face_features"].zero_()
    assert values["face_features"].sum() == 3 * 384


def proof_fixture(tmp_path, monkeypatch):
    proof = tmp_path / "qualification.json"
    comparisons = tmp_path / "comparisons.jsonl"
    row = {
        "record_index": 0,
        "cache": {"path": "cache", "sha256": "cache-hash"},
        "sampled_positions": [0, 4],
        "comparison": {"valid": True},
    }
    comparisons.write_text(json.dumps(row) + "\n")
    data = tmp_path / "data.json"
    data.write_text(json.dumps({"records": [{"cache": row["cache"], "frames": 5}]}))
    result = {
        "complete": True,
        "numerics_passed": True,
        "plan_sha256": "hash",
        "initialization_sha256": "initial",
        "optimizer_updates": 0,
        "new_Terra_requests": 0,
        "comparisons": {"path": "comparisons.jsonl", "sha256": "hash"},
    }
    proof.write_text(json.dumps(result))
    proof.with_name("exit.json").write_text(json.dumps({"pid": 1, "exit_code": 0}))
    proof.with_name("worker.json").write_text(json.dumps({"pid": 1}))
    monkeypatch.setattr(trainer, "DATA", data)
    monkeypatch.setattr(trainer, "sha256", lambda p: "hash")
    monkeypatch.setattr(trainer, "artifact", lambda p: {"path": p.name, "sha256": "hash"})
    monkeypatch.setattr(
        trainer,
        "verified_comparison",
        lambda comparison, n: None
        if comparison["valid"]
        else (_ for _ in ()).throw(ValueError("gradient")),
    )
    plan = {
        "initialization_sha256": "initial",
        "parameter_tensors": 116,
        "proof_training_record_indices": [0],
    }
    return proof, comparisons, result, row, plan


def test_training_proof_requires_terminal_same_worker_zero_update(tmp_path, monkeypatch):
    proof, _, result, _, plan = proof_fixture(tmp_path, monkeypatch)
    trainer.checked_proof(proof, tmp_path / "plan.json", plan)
    for key, value in (
        ("optimizer_updates", 1),
        ("complete", False),
        ("new_Terra_requests", 1),
        ("initialization_sha256", "changed"),
    ):
        changed = copy.deepcopy(result)
        changed[key] = value
        proof.write_text(json.dumps(changed))
        with pytest.raises(ValueError):
            trainer.checked_proof(proof, tmp_path / "plan.json", plan)
    proof.write_text(json.dumps(result))
    proof.with_name("exit.json").write_text(json.dumps({"pid": 2, "exit_code": 0}))
    with pytest.raises(ValueError):
        trainer.checked_proof(proof, tmp_path / "plan.json", plan)


def test_training_proof_rejects_other_frames_and_failed_gradients(tmp_path, monkeypatch):
    proof, comparisons, _, row, plan = proof_fixture(tmp_path, monkeypatch)
    for field, value in (
        ("sampled_positions", [0, 3]),
        ("cache", {"path": "foreign"}),
        ("comparison", {"valid": False}),
    ):
        changed = copy.deepcopy(row)
        changed[field] = value
        comparisons.write_text(json.dumps(changed) + "\n")
        with pytest.raises(ValueError):
            trainer.checked_proof(proof, tmp_path / "plan.json", plan)
