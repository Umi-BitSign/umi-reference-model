"""Synthetic cache integrity plus terminal admission; never a real dataset audit."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import c5_timing_feature_data as data


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "ROOT", tmp_path)
    arrays = {k: np.zeros((3, width), dtype=np.float32) for k, width in data.WIDTHS.items()}
    arrays["pose_features"][0, 0] = -0.0
    arrays.update(
        {
            "frame_indices": np.array([0, 2, 4], dtype=np.int64),
            "source_hand_tip_visible": np.array([1, 0, 1, 1, 1], dtype=np.bool_),
            "valid_intervals": np.array([0, 1], dtype=np.bool_),
        }
    )
    header = {
        "source_fps": 25,
        "source_frames": 5,
        "source_shape": [480, 640],
        "sampled_frames": 3,
        "frame_indices_sha256": data.array_metadata(arrays["frame_indices"])["sha256"],
    }
    inputs = {
        "header": header,
        "native_side": 224,
        "streamed_pixels_equal_native_reference": True,
        "arrays": {
            k: data.array_metadata(arrays[k])
            for k in (
                "pose_features",
                "frame_indices",
                "valid_intervals",
                "source_hand_tip_visible",
            )
        },
    }
    inputs["arrays"]["pixels"] = {
        "shape": [3, 3, 224, 224, 3],
        "dtype": "uint8",
        "sha256": "0" * 64,
    }
    return arrays, inputs


def test_complete_feature_arrays_preserve_pose_bits_and_full_rate_gap(cache):
    arrays, inputs = cache
    result = data.check_arrays(arrays, inputs)
    assert set(result) == set(data.WIDTHS) | {
        "frame_indices",
        "valid_intervals",
        "source_hand_tip_visible",
    }
    assert result["pose_features"] == inputs["arrays"]["pose_features"]


@pytest.mark.parametrize(
    "change",
    [
        "signed_zero",
        "bridge",
        "sampling",
        "truncated",
        "extra",
        "nonfinite",
        "dtype",
        "pixel_digest",
    ],
)
def test_cache_authentication_rejects_substitutions_and_hidden_gaps(cache, change):
    arrays, inputs = cache
    if change == "signed_zero":
        arrays["pose_features"][0, 0] = 0.0
    elif change == "bridge":
        arrays["valid_intervals"][:] = True
    elif change == "sampling":
        arrays["frame_indices"][-1] = 3
    elif change == "truncated":
        arrays["left_hand_features"] = arrays["left_hand_features"][:-1]
    elif change == "extra":
        arrays["encoder_hidden_state"] = np.zeros((3, 1536), dtype=np.float32)
    elif change == "nonfinite":
        arrays["face_features"][0, 0] = np.nan
    elif change == "dtype":
        arrays["pose_features"] = arrays["pose_features"].astype(np.float64)
    else:
        inputs["arrays"]["pixels"]["sha256"] = "unbound"
    with pytest.raises(ValueError):
        data.check_arrays(arrays, inputs)


@pytest.fixture
def dataset(cache, tmp_path, monkeypatch):
    arrays, inputs = cache
    output = tmp_path / "checkpoints/synthetic"
    output.mkdir(parents=True)
    feature = output / "features-0000.npz"
    np.savez(feature, **arrays)
    receipt = output / "clip-0000.json"
    receipt.write_text(json.dumps({"input": inputs}))
    authority = tmp_path / "runs/synthetic-audit.json"
    authority.parent.mkdir()
    authority.write_text("Synthetic reader fixture; not a complete admitted dataset.\n")
    value = {
        "records": [
            {
                "index": 0,
                "row": {
                    "source": "synthetic",
                    "example_id": "synthetic-0",
                    "role": "training",
                    "training_allowed": True,
                },
                "receipt": data.artifact(receipt),
                "features": data.artifact(feature),
                "header": inputs["header"],
                "arrays": data.check_arrays(arrays, inputs),
            }
        ],
        "model_plan": {"synthetic": True},
    }
    monkeypatch.setattr(data, "checked", lambda path: copy.deepcopy(value))
    return data.TimingFeatureDataset(authority), feature, receipt, authority


def test_reader_loads_only_original_feature_streams_and_does_not_reuse_mutated_arrays(dataset):
    reader, *_ = dataset
    first = reader.load(0)
    assert set(first["streams"]) == set(data.WIDTHS)
    first["streams"]["face_features"][0, 0] = 100
    first["row"]["role"] = "development"
    second = reader.load(0)
    assert second["streams"]["face_features"][0, 0] == 0
    assert second["row"]["role"] == "training"


@pytest.mark.parametrize("change", ["cache", "receipt", "authority"])
def test_reader_rejects_changed_bound_bytes(dataset, change):
    reader, feature, receipt, authority = dataset
    {"cache": feature, "receipt": receipt, "authority": authority}[change].write_bytes(b"changed")
    with pytest.raises(ValueError):
        reader.load(0)


def test_reader_rejects_other_roles_or_indices_before_any_artifact_read(dataset, monkeypatch):
    reader, *_ = dataset
    monkeypatch.setattr(
        data, "artifact", lambda path: pytest.fail("read outside admitted training request")
    )
    with pytest.raises(ValueError, match="only its admitted training"):
        reader.load(0, purpose="development")
    for index in (True, -1, 1):
        with pytest.raises(ValueError, match="only its admitted training"):
            reader.load(index)


def test_live_tool_poll_cannot_admit_a_partial_dataset(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "ROOT", tmp_path)
    poll = tmp_path / "runs/live.json"
    poll.parent.mkdir()
    poll.write_text(
        json.dumps(
            {
                "tool": "write_stdin",
                "requested_session_id": 57261,
                "tool_result": {"session_id": 57261},
            }
        )
    )
    with pytest.raises(ValueError, match="actual completed feature-session"):
        data.audit(tmp_path / "runs/plan.json", tmp_path / "checkpoints/result.json", poll)


def test_native_reference_array_and_report_cannot_be_changed(cache):
    arrays, _ = cache
    reference = {
        k: v.copy() for k, v in arrays.items() if k in data.WIDTHS and k != "pose_features"
    }
    indices = np.arange(3, dtype=np.int64)
    report = {
        "sample_indices": [0, 1, 2],
        "streams": {k: {"maximum_absolute_difference": 0.0} for k in reference},
    }
    data.native_reference_check(arrays, reference, report, indices, 1e-4, 1e-4)
    report["streams"]["face_features"]["maximum_absolute_difference"] = 1
    with pytest.raises(ValueError, match="report changed"):
        data.native_reference_check(arrays, reference, report, indices, 1e-4, 1e-4)
    report["streams"]["face_features"]["maximum_absolute_difference"] = 0
    reference["face_features"][0, 0] = 0.1
    with pytest.raises(ValueError, match="reference failed"):
        data.native_reference_check(arrays, reference, report, indices, 1e-4, 1e-4)


@pytest.mark.parametrize("role", ["development", "feature_diagnostic"])
def test_training_role_guard_rejects_reserved_or_moved_word_cases(role):
    case = {
        "kind": "word",
        "word": {
            "record": {
                "ROI_lineage_training_allowed": False,
                "original": {"row": {"role": role, "training_allowed": False, "participant": "P6"}},
            }
        },
    }
    with pytest.raises(ValueError, match="reserved word roles"):
        data.source_binding(case)
