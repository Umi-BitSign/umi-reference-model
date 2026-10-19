"""Native224 training pixels and original pose, without role changes or resizing112."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import terra_mixed_native_input as mixed
from c5_timing_visibility import interval_mask, visibility
from extract_mediapipe_landmarks import unpack_landmarks
from prepare_roi_appearance import SampledRgbReader, native_helpers
from prepare_roi_short_corpus import NativeSampledReader
from probe_c5_small_input import pack_rgb
from training_media_artifact import media_artifact

ROOT = Path(__file__).resolve().parents[1]


def source_paths(case):
    mixed.training_row(case)  # Enforce training before any file read.
    if case["kind"] == "word":
        record = case["word"]["record"]
        return record["clip"], record["landmarks"]
    record = case["phrase"]
    raw = record["original_row"]["original_pair"]["raw"]
    return {"path": raw["video"], "sha256": raw["video_sha256"]}, record["landmarks"]


def read_training224(case):
    clip, sidecar = source_paths(case)
    verified = mixed.read_input(case)
    del verified["pixels"]  # Independently decode224 from original full source frames.
    video, landmark = ROOT / clip["path"], ROOT / sidecar["path"]
    if media_artifact(video) != clip or media_artifact(landmark) != sidecar:
        raise ValueError("timing source video or landmark bytes changed")
    landmarks, metadata = unpack_landmarks(landmark)
    header, indices = verified["header"], verified["frame_indices"]
    if (
        header != mixed.expected_header(case)
        or metadata["source_sha256"] != clip["sha256"]
        or metadata["frames"] != header["source_frames"]
        or metadata["fps"] != 25
        or [metadata["height"], metadata["width"]] != header["source_shape"]
    ):
        raise ValueError("timing input changed the complete original source timeline")
    observed, valid = visibility(landmarks, header["source_frames"], indices)
    sampled = {i: landmarks[int(index)] for i, index in enumerate(indices)}
    hands, face, reader_type = native_helpers()
    with SampledRgbReader(video, header) as reader:
        left, right = hands(output_size=(224, 224)).extract_hand_frames(reader, sampled)
    with SampledRgbReader(video, header) as reader:
        facial = face(output_size=(224, 224)).extract_face_frames(reader, sampled)
    pixels = np.stack([pack_rgb(v, len(indices), 224) for v in (facial, left, right)], axis=1)
    del facial, left, right
    reader = reader_type(video)
    try:
        left, right = hands(output_size=(224, 224)).extract_hand_frames(
            NativeSampledReader(reader, header), sampled
        )
        facial = face(output_size=(224, 224)).extract_face_frames(
            NativeSampledReader(reader, header), sampled
        )
    finally:
        del reader
    reference = np.stack([pack_rgb(v, len(indices), 224) for v in (facial, left, right)], axis=1)
    if not np.array_equal(pixels, reference):
        raise ValueError("timing streamed/native224 RGB crop pixels differ")
    return {
        **verified,
        "pixels": pixels,
        "valid_intervals": valid,
        "source_hand_tip_visible": observed,
    }


def input_receipt(values):
    header = values["header"]
    frames = header["sampled_frames"]
    expected = {
        "pixels": ((frames, 3, 224, 224, 3), np.uint8),
        "pose_features": ((frames, 14), np.float32),
        "frame_indices": ((frames,), np.int64),
        "valid_intervals": ((frames - 1,), np.bool_),
        "source_hand_tip_visible": ((header["source_frames"],), np.bool_),
    }
    if set(values) != {*expected, "header"} or header["source_fps"] != 25:
        raise ValueError("timing input array inventory or original FPS changed")
    for name, (shape, dtype) in expected.items():
        value = values[name]
        if (
            not isinstance(value, np.ndarray)
            or value.shape != shape
            or value.dtype != dtype
            or not np.isfinite(value).all()
        ):
            raise ValueError("timing input lost complete finite native array shapes")
    if not np.array_equal(
        values["valid_intervals"],
        interval_mask(values["source_hand_tip_visible"], values["frame_indices"]),
    ):
        raise ValueError("timing input bridged an original visibility gap")
    return {
        "header": header,
        "native_side": 224,
        "streamed_pixels_equal_native_reference": True,
        "arrays": {
            name: {
                "shape": list(values[name].shape),
                "dtype": str(values[name].dtype),
                "sha256": hashlib.sha256(values[name].tobytes()).hexdigest(),
            }
            for name in expected
        },
    }
