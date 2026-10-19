"""Reserved-role, source-clock and native224 reader boundaries; no real model."""

from __future__ import annotations

import copy
import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import c5_timing_development_input as inputs
import c5_timing_feature_data as data

POLICY = {}  # The sentence reader must supply a private native224 policy binding.


@pytest.fixture
def cases():
    indices = np.array([0, 2, 4], dtype=np.int64)
    header = {
        "source_fps": 25,
        "source_frames": 5,
        "source_shape": [480, 640],
        "sampled_frames": 3,
        "frame_indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
    }
    row = {
        "example_id": "synthetic",
        "source_group": "asl_stem:P6:synthetic",
        "reference": "a",
        "role": "development",
        "training_allowed": False,
    }
    word = {
        "kind": "word",
        "word": {
            "record": {
                "original": {"row": {**row, "participant": "P6"}},
                "ROI_lineage_training_allowed": False,
                "normalized_probe": {"fps": 25, "decoded_frames": 5, "height": 480, "width": 640},
                "clip": {"path": "data/video.mp4", "sha256": "video"},
                "landmarks": {"path": "data/landmarks.npz", "sha256": "landmarks"},
            },
            "verified_input": {"header": header},
        },
    }
    historical = {
        "example_id": "synthetic",
        "source": "asl_stem_original",
        "role": "training",
        "training_allowed": True,
        "video_sha256": "video",
    }
    raw = {
        **historical,
        "reference": "a",
        "signer": "P6",
        "source_group": row["source_group"],
        "video": "data/video.mp4",
    }
    teacher = {
        "id": "synthetic",
        "source": "asl_stem_original",
        "signer": "P6",
        "reference": "a",
        "role": "training",
        "training_allowed": True,
        "cache": "data/cache.npz",
        "cache_sha256": "cache",
        "frames": 3,
    }
    sentence = {
        "kind": "sentence",
        "sentence": {
            "development": {
                **row,
                "source": "asl_stem_original",
                "signer": "P6",
                "original_pair": {
                    "raw": raw,
                    "header": {**header, **historical},
                    "alignment": {
                        **historical,
                        "sampled_frames": 3,
                        "frame_indices_sha256": header["frame_indices_sha256"],
                    },
                    "teacher": teacher,
                },
            },
            "landmarks": {"path": "data/landmarks.npz", "sha256": "landmarks"},
        },
    }
    return word, sentence, header, indices


def test_reserved_outer_roles_and_complete_headers_preserve_historical_provenance(cases):
    word, sentence, header, _ = cases
    assert inputs.development_row(word)["training_allowed"] is False
    assert inputs.development_row(sentence)["training_allowed"] is False
    assert sentence["sentence"]["development"]["original_pair"]["raw"]["training_allowed"] is True
    assert inputs.expected_header(word) == inputs.expected_header(sentence) == header


@pytest.mark.parametrize("role", ["training", "feature_diagnostic", "independent_confirmation"])
def test_other_roles_are_rejected_before_source_access_or_model_import(cases, monkeypatch, role):
    word, sentence, _, _ = cases
    monkeypatch.setattr(
        inputs,
        "source_visibility",
        lambda *args: pytest.fail("source accessed before reserved-role guard"),
    )
    word["word"]["record"]["original"]["row"]["role"] = role
    sentence["sentence"]["development"]["role"] = role
    for case in (word, sentence):
        with pytest.raises(ValueError, match="reserved development"):
            inputs.read_development224(case)
        with pytest.raises(ValueError, match="reserved development"):
            inputs.source_paths(case)


@pytest.mark.parametrize("change", ["signer", "source_join", "clock", "grid"])
def test_altered_reserved_inventory_or_original_clock_is_rejected(cases, change):
    word, sentence, _, _ = cases
    if change == "signer":
        word["word"]["record"]["original"]["row"]["participant"] = "P18"
        case = word
    elif change == "source_join":
        sentence["sentence"]["development"]["original_pair"]["teacher"]["id"] = "different"
        case = sentence
    elif change == "clock":
        word["word"]["record"]["normalized_probe"]["fps"] = 30
        case = word
    else:
        sentence["sentence"]["development"]["original_pair"]["header"]["sampled_frames"] = 2
        case = sentence
    with pytest.raises(ValueError):
        inputs.expected_header(case)


def test_full_rate_source_visibility_and_metadata_are_bound(cases, tmp_path, monkeypatch):
    word, _, header, indices = cases
    monkeypatch.setattr(data, "ROOT", tmp_path)
    video = tmp_path / "data/video.mp4"
    video.parent.mkdir()
    video.write_bytes(b"synthetic video")
    sidecar = tmp_path / "data/landmarks.npz"
    sidecar.write_bytes(b"synthetic landmarks")
    word["word"]["record"]["clip"] = data.artifact(video)
    word["word"]["record"]["landmarks"] = data.artifact(sidecar)
    observed = np.array([1, 0, 1, 1, 1], dtype=np.bool_)
    valid = np.array([0, 1], dtype=np.bool_)
    metadata = {
        "source_sha256": data.sha(video),
        "fps": 25,
        "frames": 5,
        "height": 480,
        "width": 640,
    }
    monkeypatch.setattr(inputs, "unpack_landmarks", lambda path: ({"full_rate": True}, metadata))

    def masks(landmarks, frames, sampled):
        assert landmarks == {"full_rate": True} and frames == 5 and np.array_equal(sampled, indices)
        return observed, valid

    monkeypatch.setattr(inputs, "visibility", masks)
    actual_header, actual_indices, actual_observed, actual_valid = inputs.source_visibility(word)
    assert actual_header == header and np.array_equal(actual_indices, indices)
    assert actual_observed is observed and actual_valid is valid
    metadata["frames"] = 6
    with pytest.raises(ValueError, match="landmark source or clock changed"):
        inputs.source_visibility(word)


@pytest.fixture
def reader(cases, monkeypatch, tmp_path):
    word, sentence, header, indices = cases
    observed = np.array([1, 0, 1, 1, 1], dtype=np.bool_)
    valid = np.array([0, 1], dtype=np.bool_)
    monkeypatch.setattr(
        inputs,
        "source_visibility",
        lambda case: (copy.deepcopy(header), indices.copy(), observed.copy(), valid.copy()),
    )
    pose = np.arange(42, dtype=np.float32).reshape(3, 14)
    pose[0, 0] = -0.0
    pixels = np.zeros((3, 3, 224, 224, 3), dtype=np.uint8)
    values = {"header": header, "frame_indices": indices, "pose_features": pose, "pixels": pixels}
    import c5_word_evaluation_io as words

    monkeypatch.setattr(words, "original_control_input", lambda case: values)
    monkeypatch.setattr(data, "ROOT", tmp_path)
    cache = tmp_path / "data/cache.npz"
    cache.parent.mkdir()
    np.savez(
        cache,
        pose_features=pose,
        frame_indices=indices,
        encoder_hidden_state=np.ones((3, 1536), dtype=np.float32),
        face_features=np.ones((3, 384), dtype=np.float32) * 99,
    )
    teacher = sentence["sentence"]["development"]["original_pair"]["teacher"]
    teacher.update({"cache": data.artifact(cache)["path"], "cache_sha256": data.sha(cache)})
    import evaluate_c5_small_input as native

    seen = []
    original_policy = copy.deepcopy(native.POLICY)

    def crops(record):
        seen.append(POLICY["native_input_side"])
        return pixels.copy()

    monkeypatch.setattr(native, "crops", crops)
    return word, sentence, pose, values, seen, native, original_policy


def test_both_native224_routes_keep_pose_bits_complete_grid_and_hidden_gaps(reader):
    word, sentence, pose, _, seen, native, original_policy = reader
    for case in (word, sentence):
        values = inputs.read_development224(case)
        assert values["pixels"].shape == (3, 3, 224, 224, 3)
        assert values["pose_features"].tobytes() == pose.tobytes()
        assert values["frame_indices"].tolist() == [0, 2, 4]
        assert values["valid_intervals"].tolist() == [False, True]
        assert "encoder_hidden_state" not in values and "face_features" not in values
    assert seen == [224] and native.POLICY == original_policy


def test_changed_word_native_header_is_rejected(reader):
    word, _, _, values, _, _, _ = reader
    values["header"] = {"different": True}
    with pytest.raises(ValueError, match="verified source"):
        inputs.read_development224(word)


def test_old_sentence_cache_without_indices_uses_verified_original_source_lattice(reader):
    _, sentence, pose, _, _, _, _ = reader
    teacher = sentence["sentence"]["development"]["original_pair"]["teacher"]
    path = data.bound({"path": teacher["cache"], "sha256": teacher["cache_sha256"]})
    np.savez(path, pose_features=pose, encoder_hidden_state=np.ones((3, 1536), dtype=np.float32))
    teacher["cache_sha256"] = data.sha(path)
    values = inputs.read_development224(sentence)
    assert values["frame_indices"].tolist() == [0, 2, 4]
    assert values["pose_features"].tobytes() == pose.tobytes()
    assert "encoder_hidden_state" not in values


def test_sentence_alignment_and_teacher_frame_count_are_required(cases):
    _, sentence, _, _ = cases
    pair = sentence["sentence"]["development"]["original_pair"]
    pair["alignment"]["frame_indices_sha256"] = "changed"
    with pytest.raises(ValueError, match="complete stride2"):
        inputs.expected_header(sentence)
