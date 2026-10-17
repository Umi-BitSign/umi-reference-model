"""Original224px word controls and authenticated complete feature caches."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import numpy as np
from c5_small_input_vision import STREAMS
from extract_mediapipe_landmarks import unpack_landmarks
from prepare_roi_appearance import SampledRgbReader, native_helpers
from prepare_roi_short_corpus import NativeSampledReader
from probe_c5_small_input import pack_rgb
from terra_training_recovery import artifact
from terra_word_native_input import RESERVED, require_role
from terra_word_verified_inputs import expected_header, load_verified_input
from training_media_artifact import media_artifact

ROOT = Path(__file__).resolve().parents[1]
STORAGE = {
    "physical_floor_bytes": 80 * 1024**3,
    "output_budget_bytes": 128 * 1024**2,
    "headroom_bytes": 128 * 1024**2,
}


def storage(prospective=0, *, output=None, root=ROOT, disk_usage=shutil.disk_usage):
    if type(prospective) is not int or prospective < 0:
        raise ValueError("word evaluation requires a nonnegative byte reservation")
    used = 0
    if output is not None:
        if not output.resolve().is_relative_to(root / "checkpoints"):
            raise ValueError("word evaluation output must remain in owned checkpoints/")
        for path in output.rglob("*"):
            if path.is_symlink():
                raise ValueError("word evaluation cannot follow output aliases")
            if path.is_file():
                used += path.stat().st_size
    remaining = STORAGE["output_budget_bytes"] - used
    free = disk_usage(root).free
    if (
        prospective > remaining
        or free < STORAGE["physical_floor_bytes"] + STORAGE["headroom_bytes"] + remaining
    ):
        raise RuntimeError("word evaluation exceeded its128MiB reserve or unchanged80GiB floor")
    return {
        "output_bytes": used,
        "remaining_output_reserve_bytes": remaining,
        "physical_free_bytes": free,
    }


def evaluation_row(case):
    require_role(case["record"], "development")
    row = case["record"]["original"]["row"]
    return {
        "source": "asl_stem_word",
        "example_id": row["example_id"],
        "source_group": row["source_group"],
        "signer": row["participant"],
        "reference": row["reference"],
        "role": "development",
        "training_allowed": False,
    }


def development_records(binding):
    records = [
        case
        for case in binding["records"]
        if case["record"]["original"]["row"]["role"] == "development"
    ]
    rows = [evaluation_row(case) for case in records]
    if (
        len(records) != 96
        or len({row["example_id"] for row in rows}) != 96
        or {row["signer"] for row in rows} != set(RESERVED)
        or len({case["record"]["original"]["row"]["label"] for case in records}) != 60
        or sum(expected_header(case["record"])["sampled_frames"] for case in records) != 4828
    ):
        raise ValueError("word evaluation changed the complete96-input/8-signer/60-label inventory")
    return records


def original_control_input(case):
    record = case["record"]
    require_role(record, "development")  # Reject role violations before any file access.
    verified = load_verified_input(case, purpose="development")
    # Reproduce authentic112px hashes first;224px crops are decoded separately
    # from original complete frames, never resized from these verified pixels.
    del verified["pixels"]
    video = ROOT / record["clip"]["path"]
    landmark = ROOT / record["landmarks"]["path"]
    if media_artifact(video) != record["clip"] or media_artifact(landmark) != record["landmarks"]:
        raise ValueError("word original control source bytes changed after input verification")
    landmarks, metadata = unpack_landmarks(landmark)
    header, indices = verified["header"], verified["frame_indices"]
    if (
        metadata["frames"] != header["source_frames"]
        or metadata["fps"] != 25
        or metadata["source_sha256"] != record["clip"]["sha256"]
        or [metadata["height"], metadata["width"]] != header["source_shape"]
    ):
        raise ValueError("word original control source timeline or dimensions differ")
    sampled = {index: landmarks[int(source)] for index, source in enumerate(indices)}
    hands, face, native_reader = native_helpers()
    side, frames = 224, len(indices)
    with SampledRgbReader(video, header) as reader:
        left, right = hands(output_size=(side, side)).extract_hand_frames(reader, sampled)
    with SampledRgbReader(video, header) as reader:
        facial = face(output_size=(side, side)).extract_face_frames(reader, sampled)
    pixels = np.stack([pack_rgb(value, frames, side) for value in (facial, left, right)], axis=1)
    del facial, left, right
    reader = native_reader(video)
    try:
        left, right = hands(output_size=(side, side)).extract_hand_frames(
            NativeSampledReader(reader, header), sampled
        )
        facial = face(output_size=(side, side)).extract_face_frames(
            NativeSampledReader(reader, header), sampled
        )
    finally:
        del reader
    reference = np.stack([pack_rgb(value, frames, side) for value in (facial, left, right)], axis=1)
    if not np.array_equal(pixels, reference):
        raise ValueError("word original224 control streamed/reference crop pixels differ")
    return {**verified, "pixels": pixels}


def validate_streams(values, case):
    require_role(case["record"], "development")
    frames = expected_header(case["record"])["sampled_frames"]
    expected = {
        **{name: ((frames, 384), np.dtype("float32")) for name in STREAMS},
        "pose_features": ((frames, 14), np.dtype("float32")),
        "frame_indices": ((frames,), np.dtype("int64")),
    }
    if set(values) != set(expected):
        raise ValueError("word original control cache changed stream coverage")
    for name, (shape, dtype) in expected.items():
        array = values[name]
        if (
            not isinstance(array, np.ndarray)
            or array.shape != shape
            or array.dtype != dtype
            or not np.isfinite(array).all()
        ):
            raise ValueError(
                "word original control cache changed full shapes, dtypes or finite values"
            )
    for name in ("pose_features", "frame_indices"):
        if (
            hashlib.sha256(values[name].tobytes()).hexdigest()
            != case["verified_input"]["arrays"][name]["sha256"]
        ):
            raise ValueError("word original control cache changed verified pose or source timeline")


def save_streams(path, values, case):
    validate_streams(values, case)
    path = path.resolve()
    if not path.is_relative_to(ROOT / "checkpoints") or path.suffix != ".npz":
        raise ValueError("word control caches must remain private in owned checkpoints/")
    storage(sum(value.nbytes for value in values.values()) + 4096, output=path.parent)
    with path.open("xb") as stream:
        np.savez(stream, **values)
    return artifact(path)


def checked_streams(descriptor, case):
    require_role(case["record"], "development")
    path = ROOT / descriptor["path"]
    if not path.resolve().is_relative_to(ROOT / "checkpoints") or artifact(path) != descriptor:
        raise ValueError("word original control cache lost its actual private byte binding")
    with np.load(path, allow_pickle=False) as saved:
        values = {name: saved[name].copy() for name in saved.files}
    validate_streams(values, case)
    return values
