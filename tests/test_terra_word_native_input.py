from __future__ import annotations

import copy
import json
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import terra_word_native_input as word
from extract_mediapipe_landmarks import write_landmarks
from terra_training_recovery import artifact
from training_media_artifact import media_artifact


def role_record(participant="P2", role="training"):
    return {
        "original": {
            "row": {
                "participant": participant,
                "role": role,
                "training_allowed": role == "training",
            }
        },
        "ROI_lineage_training_allowed": role == "training" and participant not in word.DIAGNOSTIC,
    }


@pytest.mark.parametrize("participant", sorted(word.RESERVED | word.DIAGNOSTIC))
def test_every_reserved_and_retained_diagnostic_signer_is_rejected_before_reads(participant):
    with pytest.raises(ValueError, match="roles"):
        word.read_input(role_record(participant), purpose="training")


def test_original_development_role_cannot_become_training_or_vice_versa():
    word.require_role(role_record(), "training")
    word.require_role(role_record("P4", "development"), "development")
    for record, purpose in (
        (role_record("P4", "development"), "training"),
        (role_record(), "development"),
        (role_record("P2", "development"), "development"),
        (role_record(), "diagnostic"),
    ):
        with pytest.raises(ValueError):
            word.require_role(record, purpose)


def fixture_record(tmp_path, monkeypatch):
    monkeypatch.setattr(word, "ROOT", tmp_path)
    monkeypatch.setattr(word, "media_artifact", partial(media_artifact, root=tmp_path))
    monkeypatch.setattr(word, "artifact", partial(artifact, root=tmp_path))
    video = tmp_path / "data/clips/training/word.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"synthetic video; decoding is a test fixture")
    landmark = tmp_path / "data/landmarks/training/word.npz"
    frames = {i: None for i in range(5)}
    frames[2] = {"pose_landmarks": None}
    metadata = write_landmarks(
        landmark, frames, source=video, width=640, height=480, fps=25, mediapipe_version="0.10.14"
    )
    record = {
        **role_record(),
        "clip": media_artifact(video, root=tmp_path),
        "normalized_probe": {"width": 640, "height": 480, "decoded_frames": 5},
    }
    receipt = tmp_path / "runs/receipts/training/word.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text(
        json.dumps(
            {
                "plan_sha256": "plan",
                "record": record,
                "landmarks": media_artifact(landmark, root=tmp_path),
                "metadata": metadata,
            }
        )
    )
    return {
        **record,
        "landmarks": media_artifact(landmark, root=tmp_path),
        "landmark_receipt": artifact(receipt, root=tmp_path),
        "landmark_plan": {"path": "runs/plan.json", "sha256": "plan"},
    }


@pytest.mark.parametrize("native_matches", [True, False])
def test_complete_native_RGB_and_pose_get_the_same_stride2_dictionary(
    tmp_path, monkeypatch, native_matches
):
    record = fixture_record(tmp_path, monkeypatch)
    seen = []

    class Reader:
        def __init__(self, path, header):
            assert path == tmp_path / record["clip"]["path"]
            assert header["source_frames"] == 5 and header["sampled_frames"] == 3
            seen.append(header)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    class Extractor:
        def __init__(self, output_size):
            assert output_size == (112, 112)

        def values(self, sampled):
            assert sampled == {
                0: None,
                1: {"face_landmarks": None, "hand_landmarks": None, "pose_landmarks": None},
                2: None,
            }
            seen.append(sampled)
            offset = 0 if native_matches or len(seen) < 5 else 1
            return [np.full((112, 112, 3), [i + offset, 20, 30], dtype=np.uint8) for i in (0, 2, 4)]

        def extract_hand_frames(self, reader, sampled):
            values = self.values(sampled)
            return values, values

        def extract_face_frames(self, reader, sampled):
            return self.values(sampled)

    class Pose:
        def process_landmarks_sequence(self, sampled):
            assert sampled is seen[1] and sampled is seen[3]
            # The original helper owns missing-frame/sentinel behavior.
            return np.full((3, 14), -9999, dtype=np.float64)

    monkeypatch.setattr(word, "SampledRgbReader", Reader)
    monkeypatch.setattr(
        word, "native_helpers", lambda: (Extractor, Extractor, lambda path: object())
    )
    monkeypatch.setattr(word, "NativeSampledReader", lambda reader, header: reader)
    monkeypatch.setattr(word, "pose_processor", Pose)
    if not native_matches:
        with pytest.raises(ValueError, match="streamed/reference"):
            word.read_input(record, purpose="training")
        return
    result = word.read_input(record, purpose="training")
    assert result["pixels"].shape == (3, 3, 112, 112, 3)
    np.testing.assert_array_equal(
        result["pixels"][:, 0, 0, 0], [[30, 20, 0], [30, 20, 2], [30, 20, 4]]
    )
    np.testing.assert_array_equal(result["frame_indices"], [0, 2, 4])
    assert result["pose_features"].dtype == np.float32
    assert (result["pose_features"] == -9999).all()


@pytest.mark.parametrize("mutation", ["video", "landmark", "receipt", "source_role"])
def test_changed_media_receipt_or_role_is_rejected_before_decode(tmp_path, monkeypatch, mutation):
    record = fixture_record(tmp_path, monkeypatch)
    if mutation == "source_role":
        record = copy.deepcopy(record)
        record["original"]["row"]["role"] = "development"
    else:
        key = {"video": "clip", "landmark": "landmarks", "receipt": "landmark_receipt"}[mutation]
        (tmp_path / record[key]["path"]).write_bytes(b"changed")
    with pytest.raises((ValueError, json.JSONDecodeError)):
        word.read_input(record, purpose="training")


def test_pose_loader_uses_the_bound_body_file_instead_of_a_cached_module(tmp_path, monkeypatch):
    monkeypatch.setattr(word, "DEMO", tmp_path)
    (tmp_path / "body_features.py").write_text("class PoseProcessor:\n    bound_file = True\n")
    assert word.pose_processor().bound_file is True


@pytest.mark.parametrize(
    "mutation", [None, "incomplete", "plan", "receipt", "source_record", "worker", "audit_source"]
)
def test_bind_requires_complete_audit_and_unchanged_original_receipts(
    tmp_path, monkeypatch, mutation
):
    bound = fixture_record(tmp_path, monkeypatch)
    original = {
        key: value
        for key, value in bound.items()
        if key not in ("landmarks", "landmark_receipt", "landmark_plan")
    }
    plan = tmp_path / "runs/plan.json"
    plan.write_text("{}")
    receipt = tmp_path / bound["landmark_receipt"]["path"]
    saved = json.loads(receipt.read_text())
    saved["plan_sha256"] = word.sha256(plan)
    if mutation == "source_record":
        saved["record"]["original"]["row"]["participant"] = "P3"
    receipt.write_text(json.dumps(saved))
    monkeypatch.setattr(word, "terminal", lambda directory: {"bound_terminal": str(directory)})
    audit_source = tmp_path / "runs/audit-source.py"
    audit_source.write_text("# frozen audit implementation\n")
    audit = {
        "schema": "umi-studio-word-complete-landmark-audit/1",
        "complete_landmark_audit": mutation != "incomplete",
        "plan": artifact(plan, root=tmp_path),
        "all_native_source_timeline_and_coordinate_checks_passed": True,
        "by_role": {"training": 228, "development": 96},
        "frames": 35245,
        "supervisor": word.terminal(plan.parent / "remaining-batches"),
        "workers": [word.terminal(plan.parent / f"batch-{i:04d}") for i in range(1, 22)],
        "source_sha256": {"runs/audit-source.py": word.sha256(audit_source)},
        "records": [{"receipt": artifact(receipt, root=tmp_path), "landmarks": bound["landmarks"]}],
    }
    if mutation == "plan":
        audit["plan"]["sha256"] = "changed"
    if mutation == "receipt":
        receipt.write_text("{}")
    if mutation == "worker":
        audit["workers"].pop()
    if mutation == "audit_source":
        audit_source.write_text("# changed audit\n")
    audit_path = tmp_path / "runs/audit.json"
    audit_path.write_text(json.dumps(audit))
    monkeypatch.setattr(word, "checked", lambda path: {"records": [original]})
    if mutation is None:
        records = word.bind_records(plan, audit_path)
        assert len(records) == 1 and records[0]["original"] == original["original"]
        assert records[0]["landmarks"] == bound["landmarks"]
    else:
        with pytest.raises(ValueError):
            word.bind_records(plan, audit_path)
