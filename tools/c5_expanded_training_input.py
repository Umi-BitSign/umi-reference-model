"""Read authenticated expanded training clips at original native224 geometry.

The caller binds the frozen candidate manifest, full-rate visibility and source
roles. This module admits no feature cache or training recipe on its own.
"""

from __future__ import annotations

import hashlib

import numpy as np
from c5_timing_feature_data import ROOT, bound, require
from c5_timing_visibility import visibility
from extract_mediapipe_landmarks import unpack_landmarks
from prepare_roi_appearance import SampledRgbReader, native_helpers
from prepare_roi_short_corpus import NativeSampledReader
from probe_c5_small_input import pack_rgb
from terra_word_native_input import DIAGNOSTIC, RESERVED, pose_processor

SOURCES = frozenset(
    ("asl_stem_original", "asl_stem_supplemental", "how2sign_precut_expansion", "openasl_scale")
)


def training_row(record):
    """Reject protected roles and whole reserved signers before any source reads."""
    pair, candidate, annotation = (record[key] for key in ("pair", "candidate", "annotation"))
    raw, teacher, header, alignment = (
        pair[key] for key in ("raw", "teacher", "header", "alignment")
    )
    require(
        raw.get("source") in SOURCES
        and all(row.get("role") == "training" for row in (raw, teacher, header, alignment))
        and all(
            row.get("training_allowed") is True
            for row in (raw, teacher, alignment, candidate, annotation)
        )
        and candidate.get("role") == annotation.get("role") == "training"
        and pair.get("feature_pretraining_allowed") is True,
        "expanded input requires original affirmative training roles",
    )
    require(
        not (raw["source"].startswith("asl_stem") and raw["signer"] in RESERVED | DIAGNOSTIC),
        "expanded input cannot use any reserved or retained diagnostic STEM signer",
    )
    require(
        all(
            row[key] == raw[key]
            for row in (candidate, annotation)
            for key in ("source", "example_id", "source_group")
        )
        and all(row["source"] == raw["source"] for row in (teacher, header, alignment))
        and header["example_id"] == alignment["example_id"] == raw["example_id"]
        and alignment["source_group"] == raw["source_group"]
        and teacher["signer"] == raw["signer"]
        and teacher["reference"] == raw["reference"]
        and candidate["original_index"] == annotation["original_index"]
        and candidate["stored_stride"] == pair["native_teacher_stride"] == 2
        and teacher.get("stored_temporal_stride") == 2
        and candidate["canonical_video"]
        == annotation["canonical_video"]
        == {"path": raw["video"], "sha256": raw["video_sha256"]}
        and header["video_sha256"] == alignment["video_sha256"] == raw["video_sha256"],
        "expanded input changed original identities, source groups or canonical bytes",
    )
    return {
        key: raw[key]
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


def expected_header(record):
    training_row(record)
    pair, candidate, annotation = (record[k] for k in ("pair", "candidate", "annotation"))
    header = pair["header"]
    count, shape = header["source_frames"], header["source_shape"]
    require(
        type(count) is int
        and count > 0
        and len(shape) == 2
        and all(type(v) is int and v > 0 for v in shape)
        and header["source_fps"] == pair["raw"]["video_fps"] == 25
        and candidate["source_frames"] == annotation["source_frames"] == count,
        "expanded input changed its complete original source clock",
    )
    indices = np.arange(0, count, 2, dtype=np.int64)
    digest = hashlib.sha256(indices.tobytes()).hexdigest()
    require(
        header["sampled_frames"]
        == pair["alignment"]["sampled_frames"]
        == pair["teacher"]["frames"]
        == candidate["sampled_frames"]
        == annotation["sampled_frames"]
        == len(indices)
        and header["frame_indices_sha256"] == pair["alignment"]["frame_indices_sha256"] == digest,
        "expanded input lost or changed its original stride2 positions",
    )
    return {
        "source_fps": 25,
        "source_frames": count,
        "source_shape": shape,
        "sampled_frames": len(indices),
        "frame_indices_sha256": digest,
    }


def read_training224(record, *, native_reference=False):
    training_row(record)
    header = expected_header(record)
    annotation = record["annotation"]
    video, landmark = bound(annotation["canonical_video"]), bound(annotation["landmarks"])
    require(
        video.resolve().is_relative_to(ROOT / "data")
        and landmark.resolve().is_relative_to(ROOT / "data"),
        "expanded source must remain inside authenticated data",
    )
    landmarks, metadata = unpack_landmarks(landmark)
    require(
        metadata["source_sha256"] == annotation["canonical_video"]["sha256"]
        and metadata["frames"] == len(landmarks) == header["source_frames"]
        and metadata["fps"] == 25
        and metadata["mediapipe_version"] == "0.10.14"
        and [metadata["height"], metadata["width"]] == header["source_shape"],
        "expanded landmarks changed topology, source or complete clock",
    )
    indices = np.arange(0, header["source_frames"], 2, dtype=np.int64)
    observed, intervals = visibility(landmarks, header["source_frames"], indices)
    with np.load(bound(annotation["visibility"]), allow_pickle=False) as archive:
        require(
            np.array_equal(archive["frame_indices"], indices)
            and np.array_equal(archive["source_hand_tip_visible"], observed)
            and np.array_equal(archive["valid_intervals"], intervals),
            "expanded input changed full-rate visibility or bridged a detector gap",
        )
    sampled = {i: landmarks[int(source)] for i, source in enumerate(indices)}
    require(
        all(
            np.isfinite(np.asarray(points)).all()
            for value in sampled.values()
            if value is not None
            for points in value.values()
            if points is not None
        ),
        "expanded active landmark coordinates are nonfinite",
    )
    hands, face, reader_type = native_helpers()
    with SampledRgbReader(video, header) as reader:
        left, right = hands(output_size=(224, 224)).extract_hand_frames(reader, sampled)
    with SampledRgbReader(video, header) as reader:
        facial = face(output_size=(224, 224)).extract_face_frames(reader, sampled)
    pixels = np.stack([pack_rgb(v, len(indices), 224) for v in (facial, left, right)], axis=1)
    del facial, left, right
    if native_reference:
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
        reference = np.stack(
            [pack_rgb(v, len(indices), 224) for v in (facial, left, right)], axis=1
        )
        require(np.array_equal(pixels, reference), "expanded streamed/native crop pixels differ")
    pose = pose_processor().process_landmarks_sequence(sampled).astype(np.float32)
    require(
        pose.shape == (len(indices), 14) and np.isfinite(pose).all(),
        "expanded original pose has incomplete finite14D coverage",
    )
    return {
        "header": header,
        "pixels": pixels,
        "pose_features": pose,
        "frame_indices": indices,
        "source_hand_tip_visible": observed,
        "valid_intervals": intervals,
    }
