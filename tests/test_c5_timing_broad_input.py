"""Reserved-role and complete full-rate timing-mask joins, with synthetic media."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import c5_timing_broad_input as inputs
import c5_timing_feature_data as data
from extract_mediapipe_landmarks import write_landmarks


@pytest.fixture
def record(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "ROOT", tmp_path)
    monkeypatch.setattr(inputs, "ROOT", tmp_path)
    video = tmp_path / "data/video.mp4"
    video.parent.mkdir()
    video.write_bytes(b"synthetic video; native pixel reader mocked")
    sidecar = video.with_suffix(".npz")
    hand = {"hand_landmarks": np.ones((1, 21, 3)).tolist()}
    # Missing frame1 must block the0->2 interval even though sampled frames exist.
    write_landmarks(
        sidecar,
        {i: None if i == 1 else hand for i in range(5)},
        source=video,
        width=640,
        height=480,
        fps=25,
        mediapipe_version="synthetic",
    )
    source = tmp_path / "tools/synthetic.py"
    source.parent.mkdir()
    source.write_text("# synthetic")
    plan = tmp_path / "runs/plan.json"
    plan.parent.mkdir()
    plan.write_text("{}")
    row = {
        "original": {
            "row": {"role": "confirmation", "training_allowed": False, "example_id": "synthetic"}
        },
        "example": {
            "clip_id": "synthetic",
            "family": "how2sign",
            "source_group": "how2sign:a",
            "video": data.artifact(video),
            "fps": 25,
            "frames": 5,
            "training_allowed": False,
        },
        "ROI_lineage_training_allowed": False,
        "clip": data.artifact(video),
        "landmarks": data.artifact(sidecar),
        "normalized_probe": {"decoded_frames": 5, "width": 640, "height": 480},
        "landmark_plan": data.artifact(plan),
    }
    audit = plan.with_name("audit.json")
    audit.write_text(
        json.dumps(
            {
                "complete": True,
                "training_allowed": False,
                "records": [row] * 48,
                "plan": data.artifact(plan),
                "source_sha256": {"tools/synthetic.py": data.sha(source)},
            }
        )
    )
    return {**row, "broad_audit": data.artifact(audit)}


def test_original_native224_inputs_keep_hidden_full_rate_gaps_and_signed_pose(record, monkeypatch):
    header, indices, observed, intervals = inputs.source_visibility(record)
    assert indices.tolist() == [0, 2, 4]
    assert observed.tolist() == [True, False, True, True, True]
    assert intervals.tolist() == [False, True]
    pose = np.ones((3, 14), dtype=np.float32)
    pose[0, 0] = -0.0

    def native(row, *, purpose, side):
        assert row == record and purpose == "confirmation" and side == 224
        return {
            "header": header,
            "frame_indices": indices,
            "pixels": np.zeros((3, 3, 224, 224, 3), dtype=np.uint8),
            "pose_features": pose,
        }

    monkeypatch.setattr(inputs.native, "read_input", native)
    values = inputs.read_confirmation224(record)
    assert values["pose_features"].tobytes() == pose.tobytes()
    assert np.signbit(values["pose_features"][0, 0])
    assert values["valid_intervals"].tolist() == [False, True]


@pytest.mark.parametrize("change", ["role", "allowed", "identity", "family"])
def test_other_roles_reject_before_file_or_native_reader_access(record, monkeypatch, change):
    row = copy.deepcopy(record)
    if change == "role":
        row["original"]["row"]["role"] = "training"
    elif change == "allowed":
        row["example"]["training_allowed"] = True
    elif change == "identity":
        row["example"]["clip_id"] = "different"
    else:
        row["example"]["family"] = "asl_stem"
    monkeypatch.setattr(inputs, "bound", lambda *args: pytest.fail("file access before role check"))
    with pytest.raises(ValueError, match="source roles"):
        inputs.read_confirmation224(row)


def test_source_or_audit_mutation_cannot_enter_timing_confirmation(record):
    row = copy.deepcopy(record)
    row["clip"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="membership"):
        inputs.source_visibility(row)
    row = copy.deepcopy(record)
    row["broad_audit"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="artifact bytes"):
        inputs.source_visibility(row)
