"""Reserved word/sentence native224 inputs with the training timing-mask definition.

Outer development roles govern admission. Historical inner teacher rows are
source provenance, and never grant training eligibility to these inputs.
"""

from __future__ import annotations

import hashlib

import numpy as np
from c5_timing_feature_data import bound, require
from c5_timing_visibility import visibility
from extract_mediapipe_landmarks import unpack_landmarks

RESERVED = {"P10", "P13", "P14", "P21", "P22", "P31", "P4", "P6"}


def development_row(case):
    if case["kind"] == "word":
        record = case["word"]["record"]
        row = record["original"]["row"]
        require(
            row["role"] == "development"
            and row["training_allowed"] is False
            and record["ROI_lineage_training_allowed"] is False
            and row["participant"] in RESERVED,
            "timing evaluation accepts only the reserved development word roles",
        )
        return {
            "source": "asl_stem_word",
            "example_id": row["example_id"],
            "source_group": row["source_group"],
            "signer": row["participant"],
            "reference": row["reference"],
            "role": "development",
            "training_allowed": False,
        }
    require(case["kind"] == "sentence", "timing evaluation accepts only word or sentence inputs")
    row = case["sentence"]["development"]
    require(
        row["role"] == "development"
        and row["training_allowed"] is False
        and row["signer"] in RESERVED
        and row["source"] in {"asl_stem_original", "asl_stem_supplemental"},
        "timing evaluation accepts only the reserved development sentence roles",
    )
    pair = row["original_pair"]
    raw, header, alignment, teacher = (pair[k] for k in ("raw", "header", "alignment", "teacher"))
    require(
        all(value["example_id"] == row["example_id"] for value in (raw, header, alignment))
        and teacher["id"] == row["example_id"]
        and all(value["source"] == row["source"] for value in (raw, header, alignment, teacher))
        and raw["signer"] == teacher["signer"] == row["signer"]
        and raw["source_group"] == row["source_group"]
        and raw["reference"] == teacher["reference"] == row["reference"]
        and raw["video_sha256"] == header["video_sha256"] == alignment["video_sha256"],
        "timing development sentence changed its historical source join",
    )
    return {
        key: row[key]
        for key in (
            "source",
            "example_id",
            "source_group",
            "signer",
            "reference",
            "role",
            "training_allowed",
        )
    }


def expected_header(case):
    development_row(case)
    if case["kind"] == "word":
        probe = case["word"]["record"]["normalized_probe"]
        fps, frames, shape = (
            probe["fps"],
            probe["decoded_frames"],
            [probe["height"], probe["width"]],
        )
    else:
        source = case["sentence"]["development"]["original_pair"]["header"]
        fps, frames, shape = source["source_fps"], source["source_frames"], source["source_shape"]
    require(
        fps == 25
        and type(frames) is int
        and frames > 0
        and len(shape) == 2
        and all(type(v) is int and v > 0 for v in shape),
        "timing development source clock changed",
    )
    indices = np.arange(0, frames, 2, dtype=np.int64)
    header = {
        "source_fps": 25,
        "source_frames": frames,
        "source_shape": shape,
        "sampled_frames": len(indices),
        "frame_indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
    }
    if case["kind"] == "word":
        require(
            header == case["word"]["verified_input"]["header"],
            "timing development word header changed",
        )
    else:
        pair = case["sentence"]["development"]["original_pair"]
        require(
            source["sampled_frames"] == len(indices)
            and source["frame_indices_sha256"] == header["frame_indices_sha256"]
            and pair["alignment"]["sampled_frames"] == pair["teacher"]["frames"] == len(indices)
            and pair["alignment"]["frame_indices_sha256"] == header["frame_indices_sha256"],
            "timing development sentence changed complete stride2 coverage",
        )
    return header


def source_paths(case):
    development_row(case)
    if case["kind"] == "word":
        record = case["word"]["record"]
        return record["clip"], record["landmarks"]
    record = case["sentence"]
    raw = record["development"]["original_pair"]["raw"]
    return {"path": raw["video"], "sha256": raw["video_sha256"]}, record["landmarks"]


def source_visibility(case):
    header = expected_header(case)
    clip, sidecar = source_paths(case)
    bound(clip)
    landmarks, metadata = unpack_landmarks(bound(sidecar))
    require(
        metadata["source_sha256"] == clip["sha256"]
        and metadata["fps"] == 25
        and metadata["frames"] == header["source_frames"]
        and [metadata["height"], metadata["width"]] == header["source_shape"],
        "timing development landmark source or clock changed",
    )
    indices = np.arange(0, header["source_frames"], 2, dtype=np.int64)
    observed, intervals = visibility(landmarks, header["source_frames"], indices)
    return header, indices, observed, intervals


def read_development224(case):
    development_row(
        case
    )  # Reject training/diagnostic/confirmation before any file or model import.
    header, indices, observed, intervals = source_visibility(case)
    from c5_timing_native_input import input_receipt

    if case["kind"] == "word":
        from c5_word_evaluation_io import original_control_input

        values = original_control_input(case["word"])
        require(values["header"] == header, "timing word native input changed its verified source")
        pose, pixels = values["pose_features"], values["pixels"]
        require(
            values["frame_indices"].tobytes() == indices.tobytes(),
            "timing word source grid changed",
        )
    else:
        import evaluate_c5_small_input as native
        from terra_polarity_judge import bind

        record = case["sentence"]
        teacher = record["development"]["original_pair"]["teacher"]
        with np.load(
            bound({"path": teacher["cache"], "sha256": teacher["cache_sha256"]}), allow_pickle=False
        ) as archive:
            pose = archive["pose_features"].copy()
            if "frame_indices" in archive.files:
                require(
                    archive["frame_indices"].dtype == np.int64
                    and archive["frame_indices"].tobytes() == indices.tobytes(),
                    "timing sentence cached source grid changed",
                )
        pixels = bind(native.crops, POLICY={**native.POLICY, "native_input_side": 224})(record)
    result = {
        "header": header,
        "pixels": pixels,
        "pose_features": pose,
        "frame_indices": indices,
        "source_hand_tip_visible": observed,
        "valid_intervals": intervals,
    }
    input_receipt(
        result
    )  # Validate every original frame, pose bit dtype and full-rate visibility gap.
    return result
