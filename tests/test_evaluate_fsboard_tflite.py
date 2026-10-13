from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from benchmark_fleurs_dev import sha256
from evaluate_fsboard_tflite import checked_landmark_columns, decode_output, model_input


def test_model_input_uses_publisher_part_then_axis_order() -> None:
    points = np.arange(2 * 543 * 3, dtype=np.float32).reshape(2, 543, 3)
    columns = model_input(points)
    assert columns.shape == (2, 1629)
    assert columns.dtype == np.float32
    assert columns.flags.c_contiguous
    for frame in range(2):
        offset = 0
        for start, stop in ((0, 468), (468, 489), (522, 543), (489, 522)):
            for axis in range(3):
                width = stop - start
                np.testing.assert_array_equal(
                    columns[frame, offset : offset + width], points[frame, start:stop, axis]
                )
                offset += width
        assert offset == 1629


def test_model_input_training_tensor_uses_global_axis_blocks() -> None:
    points = np.arange(2 * 543 * 3, dtype=np.float32).reshape(2, 543, 3)
    columns = model_input(points, "training_tensor")
    for frame in range(2):
        for axis in range(3):
            np.testing.assert_array_equal(
                columns[frame, axis * 543 : (axis + 1) * 543], points[frame, :, axis]
            )
    with pytest.raises(ValueError, match="unknown"):
        model_input(points, "other")


def test_model_input_preserves_missing_landmarks_and_rejects_empty_input() -> None:
    points = np.ones((2, 543, 3), dtype=np.float32)
    points[0, 522, 1] = np.nan
    assert np.isnan(model_input(points)[0, 3 * 468 + 3 * 21 + 21])
    with pytest.raises(ValueError, match="no usable"):
        model_input(np.full((1, 543, 3), np.nan, dtype=np.float32))
    with pytest.raises(ValueError, match="shape"):
        model_input(np.zeros((1, 543, 2), dtype=np.float32))


def test_decode_output_maps_graph_indices_to_vocab_without_blank() -> None:
    vocab = ["<blank>"] + [chr(65 + i) for i in range(59)]
    output = np.eye(59, dtype=np.float32)[[0, 2, 58]]
    assert decode_output(output, vocab) == vocab[1] + vocab[3] + vocab[59]
    with pytest.raises(ValueError, match="unexpected"):
        decode_output(np.zeros((1, 59), dtype=np.float32), vocab)


def test_checked_landmarks_reject_changed_receipt(tmp_path: Path) -> None:
    source = tmp_path / "sample.npz"
    np.savez(source, landmarks=np.ones((2, 543, 3), dtype=np.float32))
    row = {"clip_filename": "sample.mp4"}
    receipt = {
        "selection_sha256": "selection",
        "sha256": sha256(source),
        "shape": [2, 543, 3],
    }
    assert checked_landmark_columns(row, receipt, tmp_path, "selection", "training_tensor").shape == (2, 1629)
    with pytest.raises(ValueError, match="receipt mismatch"):
        checked_landmark_columns(row, {**receipt, "sha256": "wrong"}, tmp_path, "selection", "training_tensor")
    with pytest.raises(ValueError, match="shape differs"):
        checked_landmark_columns(row, {**receipt, "shape": [1, 543, 3]}, tmp_path, "selection", "training_tensor")
