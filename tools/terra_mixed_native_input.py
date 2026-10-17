"""Complete native112 word/phrase inputs with reserved roles checked before reads."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from pathlib import Path

import numpy as np
from benchmark_fleurs_dev import normalize
from c5_word_training import training_records
from extract_mediapipe_landmarks import unpack_landmarks
from prepare_roi_appearance import SampledRgbReader, native_helpers
from prepare_roi_broad_corpus import LAYOUT
from prepare_roi_short_corpus import NativeSampledReader
from probe_c5_small_input import pack_rgb
from terra_word_native_input import DIAGNOSTIC, RESERVED, pose_processor, require_role
from terra_word_verified_inputs import expected_header as word_header
from terra_word_verified_inputs import load_verified_input as load_word
from training_media_artifact import media_artifact

ROOT = Path(__file__).resolve().parents[1]


def phrase_row(record):
    row = record["original_row"]
    pair = row["original_pair"]
    raw, header, alignment, teacher = (
        pair[key] for key in ("raw", "header", "alignment", "teacher")
    )
    identity = (
        "source",
        "example_id",
        "source_group",
        "signer",
        "reference",
        "video",
        "video_sha256",
        "video_fps",
    )
    if (
        record.get("roi_role") != "training"
        or record.get("roi_training_allowed") is not True
        or row.get("role") != "training"
        or row.get("training_allowed") is not True
        or row.get("source") not in LAYOUT
        or any(row[key] != raw[key] for key in identity)
        or any(value.get("role") != "training" for value in (raw, header, alignment, teacher))
        or any(value.get("training_allowed") is not True for value in (raw, alignment, teacher))
        or pair.get("feature_pretraining_allowed") is not True
        or teacher.get("stored_temporal_stride") != 2
        or pair.get("native_teacher_stride") != 2
        or any(value["source"] != row["source"] for value in (header, alignment, teacher))
        or header["example_id"] != row["example_id"]
        or alignment["example_id"] != row["example_id"]
        or alignment["source_group"] != row["source_group"]
        or teacher["signer"] != row["signer"]
        or teacher["reference"] != row["reference"]
        or not 2 <= len(normalize(row["reference"])) <= 6
        or (row["source"].startswith("asl_stem") and row["signer"] in RESERVED | DIAGNOSTIC)
    ):
        raise ValueError("mixed native phrases cannot change training identity or reserved roles")
    return row


def training_row(case):
    if case.get("kind") == "word" and set(case) == {"kind", "word"}:
        record = case["word"]["record"]
        require_role(record, "training")
        row = record["original"]["row"]
        return {
            "source": "asl_stem_word",
            "example_id": row["example_id"],
            "source_group": row["source_group"],
            "signer": row["participant"],
            "reference": row["reference"],
            "role": "training",
            "training_allowed": True,
        }
    if case.get("kind") == "phrase" and set(case) == {"kind", "phrase"}:
        return phrase_row(case["phrase"])
    raise ValueError("mixed native inputs require an explicit word or phrase training binding")


def expected_header(case):
    training_row(case)
    if case["kind"] == "word":
        return word_header(case["word"]["record"])
    pair = case["phrase"]["original_row"]["original_pair"]
    header = pair["header"]
    count = header["source_frames"]
    shape = header["source_shape"]
    if (
        type(count) is not int
        or count < 1
        or len(shape) != 2
        or any(type(value) is not int or value < 1 for value in shape)
        or header["source_fps"] != 25
        or pair["raw"]["video_fps"] != 25
        or header["video_sha256"] != pair["raw"]["video_sha256"]
    ):
        raise ValueError("mixed native phrase changed original source metadata")
    indices = np.arange(0, count, 2, dtype=np.int64)
    digest = hashlib.sha256(indices.tobytes()).hexdigest()
    if (
        not 1 <= len(indices) <= 149
        or header["sampled_frames"] != len(indices)
        or header["frame_indices_sha256"] != digest
        or pair["alignment"]["sampled_frames"] != len(indices)
        or pair["alignment"]["frame_indices_sha256"] != digest
        or pair["teacher"]["frames"] != len(indices)
    ):
        raise ValueError("mixed native phrase cannot drop or replace original stride2 frames")
    return {
        "source_fps": 25,
        "source_frames": count,
        "source_shape": shape,
        "sampled_frames": len(indices),
        "frame_indices_sha256": digest,
    }


def select_inputs(words, phrase_plan, phrase_data, sentence_plan, native_rows):
    cases = [{"kind": "word", "word": case} for case in training_records(words)]
    if (
        phrase_data.get("complete") is not True
        or phrase_data.get("counts") != {"training": 318, "feature_diagnostic": 2}
        or len(phrase_plan["records"]) != 320
        or len(phrase_data["records"]) != 320
        or set(phrase_plan["reserved_STEM_participants"]) != RESERVED
        or set(phrase_plan["ROI_diagnostic_participants"]) != DIAGNOSTIC
    ):
        raise ValueError("mixed native input requires the original complete318/2 phrase inventory")
    diagnostics = []
    for planned, actual in zip(phrase_plan["records"], phrase_data["records"], strict=True):
        if any(actual.get(key) != value for key, value in planned.items()):
            raise ValueError("mixed native phrase changed its original complete data binding")
        if planned["roi_training_allowed"] is True:
            phrase_row(planned)
            cases.append({"kind": "phrase", "phrase": planned})
        else:
            row = planned["original_row"]
            if (
                planned["roi_role"] != "feature_diagnostic"
                or planned["roi_training_allowed"] is not False
                or not row["source"].startswith("asl_stem")
                or row["signer"] not in DIAGNOSTIC
            ):
                raise ValueError(
                    "mixed native input cannot repurpose a retained feature diagnostic"
                )
            diagnostics.append(row)
    if {row["signer"] for row in diagnostics} != DIAGNOSTIC or len(diagnostics) != 2:
        raise ValueError("mixed native input lost the two original feature diagnostics")
    rows = [training_row(case) for case in cases]
    if (
        len(cases) != 532
        or Counter(case["kind"] for case in cases) != {"word": 214, "phrase": 318}
        or len({(row["source"], row["example_id"]) for row in rows}) != 532
        or len({row["example_id"] for row in rows}) != 532
        or sum(expected_header(case)["sampled_frames"] for case in cases) != 38876
    ):
        raise ValueError("mixed native input must retain all214 easier words and318 phrases")
    word_held = []
    for case in words["records"]:
        record = case["record"]
        row = record["original"]["row"]
        if row["role"] == "development":
            require_role(record, "development")
            word_held.append(row)
    sentences = [case["development"] for case in sentence_plan["records"]]
    if (
        len(word_held) != 96
        or len(sentences) != 34
        or len(native_rows) != 142
        or len({row["example_id"] for row in native_rows}) != 142
        or any(
            row["training_allowed"] is not False for row in (*word_held, *sentences, *native_rows)
        )
    ):
        raise ValueError("mixed native input changed protected development inventories")
    boundaries = {}
    for label, held in (
        ("word96", word_held),
        ("sentence34", sentences),
        ("native142", native_rows),
        ("retained_diagnostics2", diagnostics),
    ):
        overlaps = {
            key: sorted({row[key] for row in rows} & {row[key] for row in held})
            for key in ("example_id", "source_group")
        }
        if any(overlaps.values()):
            raise ValueError("mixed native input crosses protected metadata boundaries: " + label)
        boundaries[label] = overlaps
    return cases, boundaries


def read_phrase(record):
    row = phrase_row(record)  # Roles must fail before file reads or helper imports.
    case = {"kind": "phrase", "phrase": record}
    header = expected_header(case)
    pair = row["original_pair"]
    video = ROOT / pair["raw"]["video"]
    landmark = ROOT / record["landmarks"]["path"]
    if (
        media_artifact(video)
        != {"path": pair["raw"]["video"], "sha256": pair["raw"]["video_sha256"]}
        or media_artifact(landmark) != record["landmarks"]
    ):
        raise ValueError("mixed native phrase original video or landmark bytes changed")
    landmarks, metadata = unpack_landmarks(landmark)
    if (
        metadata["source_sha256"] != pair["raw"]["video_sha256"]
        or metadata["frames"] != header["source_frames"]
        or len(landmarks) != header["source_frames"]
        or metadata["fps"] != 25
        or metadata["mediapipe_version"] != "0.10.14"
        or [metadata["height"], metadata["width"]] != header["source_shape"]
    ):
        raise ValueError("mixed native phrase landmark topology or complete timeline changed")
    indices = np.arange(0, header["source_frames"], 2, dtype=np.int64)
    sampled = {index: landmarks[int(source)] for index, source in enumerate(indices)}
    if any(
        not np.isfinite(np.asarray(points)).all()
        for value in sampled.values()
        if value is not None
        for points in value.values()
        if points is not None
    ):
        raise ValueError("mixed native phrase active landmark coordinates are nonfinite")
    hands, face, native_reader = native_helpers()
    side, frames = 112, len(indices)
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
        raise ValueError("mixed native phrase streamed/reference crop pixels differ")
    del reference, facial, left, right
    pose = pose_processor().process_landmarks_sequence(sampled).astype(np.float32)
    if pose.shape != (frames, 14) or not np.isfinite(pose).all():
        raise ValueError("mixed native phrase pose must preserve complete finite14D coverage")
    teacher = pair["teacher"]
    teacher_path = ROOT / teacher["cache"]
    if media_artifact(teacher_path) != {
        "path": teacher["cache"],
        "sha256": teacher["cache_sha256"],
    }:
        raise ValueError("mixed native phrase lost its original pose stream binding")
    with np.load(teacher_path, allow_pickle=False) as saved:
        original_pose = saved["pose_features"]
    if original_pose.dtype != np.float32 or not np.array_equal(pose, original_pose):
        raise ValueError("mixed native phrase recomputed pose differs from original C5 input")
    return {"pixels": pixels, "pose_features": pose, "frame_indices": indices, "header": header}


def read_input(case):
    training_row(case)
    if case["kind"] == "word":
        return load_word(case["word"], purpose="training")
    return read_phrase(case["phrase"])


def input_receipt(case, values):
    header = expected_header(case)
    frames = header["sampled_frames"]
    expected = {
        "pixels": ((frames, 3, 112, 112, 3), np.dtype("uint8")),
        "pose_features": ((frames, 14), np.dtype("float32")),
        "frame_indices": ((frames,), np.dtype("int64")),
    }
    if set(values) != {*expected, "header"} or values["header"] != header:
        raise ValueError("mixed native input changed full array coverage or metadata")
    arrays = {}
    for name, (shape, dtype) in expected.items():
        value = values[name]
        if (
            not isinstance(value, np.ndarray)
            or value.shape != shape
            or value.dtype != dtype
            or not np.isfinite(value).all()
        ):
            raise ValueError("mixed native input changed full shapes, dtypes or finite values")
        arrays[name] = {
            "shape": list(shape),
            "dtype": str(dtype),
            "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
        }
    if arrays["frame_indices"]["sha256"] != header["frame_indices_sha256"]:
        raise ValueError("mixed native input changed the complete stride2 indices")
    if case["kind"] == "word" and arrays != case["word"]["verified_input"]["arrays"]:
        raise ValueError("mixed native input changed an original easier word array")
    return {
        "case": case,
        "header": header,
        "arrays": arrays,
        "streamed_pixels_equal_native_reference": True,
        "original_pose_and_easier_word_inputs_preserved": True,
    }


def require_receipt(case, saved):
    header = expected_header(case)
    frames = header["sampled_frames"]
    expected = {
        "pixels": ([frames, 3, 112, 112, 3], "uint8"),
        "pose_features": ([frames, 14], "float32"),
        "frame_indices": ([frames], "int64"),
    }
    arrays = saved.get("arrays", {})
    if (
        saved.get("case") != case
        or saved.get("header") != header
        or saved.get("streamed_pixels_equal_native_reference") is not True
        or saved.get("original_pose_and_easier_word_inputs_preserved") is not True
        or set(arrays) != set(expected)
        or any(
            arrays[name].get("shape") != shape
            or arrays[name].get("dtype") != dtype
            or re.fullmatch(r"[0-9a-f]{64}", arrays[name].get("sha256", "")) is None
            for name, (shape, dtype) in expected.items()
        )
        or arrays["frame_indices"]["sha256"] != header["frame_indices_sha256"]
        or (case["kind"] == "word" and arrays != case["word"]["verified_input"]["arrays"])
    ):
        raise ValueError("mixed native receipt changed identity, complete arrays or native checks")
