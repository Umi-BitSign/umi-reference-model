from __future__ import annotations

import copy
import hashlib
import json
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import audit_terra_word_landmarks
import terra_word_verified_inputs as verified
from terra_training_recovery import artifact


def arrays(values):
    return {
        name: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
        }
        for name, value in values.items()
        if name != "header"
    }


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(verified, "ROOT", tmp_path)
    descriptor = partial(artifact, root=tmp_path)
    monkeypatch.setattr(verified, "artifact", descriptor)
    monkeypatch.setattr(audit_terra_word_landmarks, "artifact", descriptor)
    plan_path = tmp_path / "runs/plan.json"
    plan_path.parent.mkdir()
    plan_path.write_text("{}")
    output = tmp_path / "checkpoints/preflight"
    output.mkdir(parents=True)
    (output / "worker.json").write_text(json.dumps({"pid": 71}))
    (output / "exit.json").write_text(json.dumps({"pid": 71, "exit_code": 0}))
    records, values = [], []
    for index, (participant, role, frames) in enumerate(
        [("P2", "training", 5), ("P4", "development", 4), ("P0", "training", 3)]
    ):
        record = {
            "original": {
                "row": {
                    "participant": participant,
                    "role": role,
                    "training_allowed": role == "training",
                    "example_id": str(index),
                }
            },
            "ROI_lineage_training_allowed": index == 0,
            "normalized_probe": {"decoded_frames": frames, "height": 480, "width": 640},
        }
        header = verified.expected_header(record)
        records.append(record)
        values.append(
            {
                "pixels": np.full(
                    (header["sampled_frames"], 3, 112, 112, 3), index, dtype=np.uint8
                ),
                "pose_features": np.zeros((header["sampled_frames"], 14), dtype=np.float32),
                "frame_indices": np.arange(0, frames, 2, dtype=np.int64),
                "header": header,
            }
        )
    plan = {"records": records, "eligible_counts": {"training": 1, "development": 1}}
    monkeypatch.setattr(verified, "checked", lambda path: plan)
    monkeypatch.setattr(verified, "eligible_records", lambda rows: (rows[:2], rows[2:]))
    receipts = []
    for index, record in enumerate(records[:2]):
        target = output / f"input-{index:04d}.json"
        target.write_text(
            json.dumps(
                {
                    "record": record,
                    "header": values[index]["header"],
                    "arrays": arrays(values[index]),
                    "streamed_pixels_equal_native_reference": True,
                }
            )
        )
        receipts.append(descriptor(target))
    result_path = output / "data.json"
    result = {
        "complete": True,
        "plan_sha256": verified.sha256(plan_path),
        "receipts": receipts,
        "counts": plan["eligible_counts"],
        "sampled_frames_by_role": {"training": 3, "development": 2},
        "excluded_original_training_identities": ["2"],
        "all_streamed_pixels_equal_native_reference": True,
        "model_training_ready": False,
        "optimizer_updates": 0,
        "new_Terra_requests": 0,
        "translation_quality_evaluated": False,
        "promotion_authorized": False,
    }
    result_path.write_text(json.dumps(result))
    calls = []

    def read(record, purpose):
        calls.append((record, purpose))
        return values[int(record["original"]["row"]["example_id"])]

    monkeypatch.setattr(verified, "read_input", read)
    return plan_path, result_path, result, records, values, calls


def test_completed_receipts_bind_every_role_and_reproduce_odd_and_even_complete_timelines(fixture):
    plan, result, _, records, _, calls = fixture
    bound = verified.checked_result(plan, result)
    assert bound["counts"] == {"training": 1, "development": 1}
    assert [r["record"] for r in bound["records"]] == records[:2]
    assert bound["model_training_qualification_still_required"] is True
    for index, purpose in enumerate(("training", "development")):
        value = verified.load_verified_input(bound["records"][index], purpose=purpose)
        np.testing.assert_array_equal(value["frame_indices"], [0, 2, 4] if index == 0 else [0, 2])
    assert len(calls) == 2


@pytest.mark.parametrize(
    "mutation",
    [
        "complete",
        "plan_sha256",
        "counts",
        "sampled_frames_by_role",
        "excluded_original_training_identities",
        "all_streamed_pixels_equal_native_reference",
        "model_training_ready",
        "optimizer_updates",
        "new_Terra_requests",
        "translation_quality_evaluated",
        "promotion_authorized",
        "receipts",
        "extra_receipt",
        "failed_exit",
        "wrong_worker",
    ],
)
def test_incomplete_or_changed_preflight_cannot_become_training_inputs(fixture, mutation):
    plan, path, result, _, _, calls = fixture
    if mutation == "extra_receipt":
        path.with_name("input-9999.json").write_text("{}")
    elif mutation in ("failed_exit", "wrong_worker"):
        path.with_name("exit.json").write_text(
            json.dumps(
                {
                    "pid": 72 if mutation == "wrong_worker" else 71,
                    "exit_code": 1 if mutation == "failed_exit" else 0,
                }
            )
        )
    else:
        result[mutation] = {
            "complete": False,
            "plan_sha256": "changed",
            "counts": {},
            "sampled_frames_by_role": {},
            "excluded_original_training_identities": [],
            "all_streamed_pixels_equal_native_reference": False,
            "model_training_ready": True,
            "optimizer_updates": 1,
            "new_Terra_requests": 1,
            "translation_quality_evaluated": True,
            "promotion_authorized": True,
            "receipts": [],
        }[mutation]
        path.write_text(json.dumps(result))
    with pytest.raises(ValueError):
        verified.checked_result(plan, path)
    assert not calls


@pytest.mark.parametrize("mutation", ["pixels", "pose_features", "frame_indices", "header"])
def test_training_reread_must_reproduce_the_actual_preflight_hashes(fixture, mutation):
    plan, path, _, _, values, _ = fixture
    bound = verified.checked_result(plan, path)["records"][0]
    if mutation == "header":
        values[0]["header"]["sampled_frames"] = 2
    else:
        values[0][mutation].reshape(-1)[0] += 1
    with pytest.raises(ValueError, match="actual preflight"):
        verified.load_verified_input(bound, purpose="training")


def test_development_reuse_and_changed_receipts_are_rejected_before_pixels(fixture):
    plan, path, _, _, _, calls = fixture
    bound = verified.checked_result(plan, path)["records"]
    with pytest.raises(ValueError, match="roles"):
        verified.load_verified_input(bound[1], purpose="training")
    path.with_name("input-0000.json").write_text("{}")
    with pytest.raises(ValueError, match="receipt"):
        verified.load_verified_input(bound[0], purpose="training")
    assert not calls


@pytest.mark.parametrize("mutation", ["shape", "dtype", "index_hash", "identity", "parity"])
def test_validly_hashed_receipts_must_still_describe_the_full_native_input(fixture, mutation):
    _, path, _, records, _, _ = fixture
    saved = copy.deepcopy(json.loads(path.with_name("input-0000.json").read_text()))
    if mutation == "shape":
        saved["arrays"]["pixels"]["shape"][0] = 2
    elif mutation == "dtype":
        saved["arrays"]["pose_features"]["dtype"] = "float64"
    elif mutation == "index_hash":
        saved["arrays"]["frame_indices"]["sha256"] = "0" * 64
    elif mutation == "identity":
        saved["record"]["original"]["row"]["example_id"] = "other"
    else:
        saved["streamed_pixels_equal_native_reference"] = False
    with pytest.raises(ValueError):
        verified.require_receipt(records[0], saved)
