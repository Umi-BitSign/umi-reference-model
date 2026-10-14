"""Bind approved raw videos to audited native-frame feature-teacher receipts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "runs/tiny-rgb-full-v3-buckets/plan.json"
DURATION_AUDIT = ROOT / "runs/tiny-rgb-duration-integrity-v1/receipt.json"
CORPUS = ROOT / "runs/c5-all-source-corpus-v2"


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def canonical_input(raw, header, receipt):
    """Use the already-receipted25fps derivative when the teacher used that derivative.

    Old raw inventories remain unchanged. A24fps original and a25fps resampling
    can have equal sample counts while their selected frames differ.
    """
    from benchmark_fleurs_dev import sha256

    teacher_digest = receipt.get("resampled_clip_sha256", receipt.get("clip_sha256"))
    if raw["video_sha256"] == teacher_digest:
        return dict(raw), dict(header)
    if (
        raw["source"] != "how2sign_precut_expansion"
        or receipt.get("source_clip_sha256") != raw["video_sha256"]
    ):
        raise ValueError("teacher video mismatch has no receipted raw-to25fps transform")
    original = (ROOT / raw["video"]).resolve()
    derivative = ROOT / "data/how2sign/precut-expansion-clips-25fps" / f"{raw['example_id']}.mp4"
    if (
        not original.is_relative_to(ROOT / "data")
        or sha256(original) != raw["video_sha256"]
        or sha256(derivative) != teacher_digest
    ):
        raise ValueError("teacher canonical original/derivative bytes differ from receipts")
    import cv2
    import numpy as np

    cv2.setNumThreads(2)
    capture = cv2.VideoCapture(str(derivative))
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        shape = [
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        ]
        if not capture.isOpened() or fps != 25 or count != receipt["source_frames"]:
            raise ValueError("teacher canonical derivative header differs from extraction")
    finally:
        capture.release()
    indices = np.arange(0, count, 2, dtype=np.int64)
    return (
        {
            **raw,
            "video": str(derivative.relative_to(ROOT)),
            "video_sha256": teacher_digest,
            "video_fps": 25.0,
        },
        {
            **header,
            "source_fps": 25.0,
            "source_frames": count,
            "source_shape": shape,
            "video_sha256": teacher_digest,
            "sampled_frames": len(indices),
            "frame_indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
        },
    )


def bind_pair(raw, header, teacher, receipt, duration_digest):
    import numpy as np
    from benchmark_fleurs_dev import normalize
    from tiny_rgb_visual_teacher import verify_training_alignment

    identity = (raw["source"], raw["example_id"])
    if (
        raw["role"] != "training"
        or raw["training_allowed"] is not True
        or teacher["role"] != "training"
        or teacher["training_allowed"] is not True
        or (teacher["source"], teacher["id"]) != identity
        or (header["source"], header["example_id"]) != identity
        or header["role"] != "training"
        or receipt.get("split") != "training"
        or receipt.get("example_id", receipt.get("sentence_name")) != identity[1]
        or normalize(raw["reference"]) != normalize(teacher["reference"])
        or normalize(raw["reference"]) != normalize(receipt["reference"])
        or raw["video_fps"] != 25
        or header["source_fps"] != 25
        or receipt.get("clip_fps") != 25
        or header["video_sha256"] != raw["video_sha256"]
        or receipt["encoder_cache_sha256"] != teacher["cache_sha256"]
    ):
        raise ValueError(
            "teacher source/reference/role/cache binding differs from approved raw data"
        )
    stride = teacher.get("stored_temporal_stride", 1)
    if stride not in (1, 2) or receipt.get("temporal_stride", 1) != stride:
        raise ValueError("teacher temporal policy differs from its native extraction receipt")
    source_frames = header["source_frames"]
    if type(source_frames) is not int or source_frames < 1:
        raise ValueError("teacher requires a complete source frame count")
    if stride == 2:
        if receipt.get("source_frames") != source_frames or receipt.get("effective_fps") != 12.5:
            raise ValueError("stride2 teacher has a different source-frame timeline")
    elif receipt["encoder_shape"][0] != source_frames:
        raise ValueError("full-rate teacher has a different source-frame timeline")
    indices = np.arange(0, source_frames, 2, dtype=np.int64)
    digest = hashlib.sha256(indices.tobytes()).hexdigest()
    if (
        header["sampled_frames"] != len(indices)
        or header["frame_indices_sha256"] != digest
        or teacher["frames"] != (source_frames if stride == 1 else len(indices))
        or receipt["encoder_shape"] != [teacher["frames"], 1536]
    ):
        raise ValueError("teacher and RGB sampled frame indices differ; do not interpolate")
    student_binding = {
        "source": raw["source"],
        "example_id": raw["example_id"],
        "source_group": raw["source_group"],
        "role": "training",
        "training_allowed": True,
        "video_sha256": raw["video_sha256"],
        "sampled_frames": header["sampled_frames"],
        "frame_indices_sha256": header["frame_indices_sha256"],
        "duration_integrity_audit_sha256": duration_digest,
    }
    teacher_binding = {
        **student_binding,
        "video_sha256": receipt.get("resampled_clip_sha256", receipt.get("clip_sha256")),
        "sampled_frames": len(indices),
        "frame_indices_sha256": digest,
    }
    verify_training_alignment(student_binding, teacher_binding)
    return {
        "raw": raw,
        "header": header,
        "teacher": teacher,
        "alignment": teacher_binding,
        "native_teacher_stride": stride,
        "native_receipt_row_sha256": hashlib.sha256(
            json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def inputs():
    from audit_tiny_rgb_duration import checked_duration_audit
    from benchmark_fleurs_dev import sha256
    from c5_full_coverage_mps import load_training
    from freeze_visual_lora_plan import SOURCE_LAYOUT
    from tiny_rgb_capacity_protocol import select_cases
    from tiny_rgb_fit import select_fit_cases
    from tiny_rgb_full_run import checked_full_plan

    parent = checked_full_plan(PARENT)
    audit, raw_rows = checked_duration_audit(DURATION_AUDIT)
    headers = {
        (row["source"], row["example_id"]): row
        for row in read_rows(ROOT / parent["headers"])
        if row["role"] == "training"
    }
    cases = select_cases(raw_rows, [headers[(r["source"], r["example_id"])] for r in raw_rows])
    examples = select_fit_cases(raw_rows)
    selected = {(r["source"], r["example_id"]): r for r in examples}
    selected.update(
        {(c["training"]["source"], c["training"]["example_id"]): c["training"] for c in cases}
    )
    approved = {
        (row["source"], row["id"]): row
        for row in load_training(CORPUS / "audit.json", CORPUS / "training-streams.jsonl")
    }
    indexes = {source: layout["index"] for source, layout in SOURCE_LAYOUT.items()}
    indexes.update(
        {
            "how2sign_precut_expansion": (
                "runs/how2sign-precut-expansion-v1/stream-cache-stride2-index.jsonl"
            ),
            "asl_stem_original": "runs/asl-stem-wiki-v1/stream-cache-stride2.jsonl",
            "asl_stem_supplemental": "runs/asl-stem-wiki-scale-v1/stream-cache-stride2.jsonl",
            "openasl_scale": "runs/openasl-scale-v1/stream-cache-stride2.jsonl",
        }
    )
    indexed = {
        source: {
            row.get("example_id", row.get("sentence_name")): row for row in read_rows(ROOT / name)
        }
        for source, name in indexes.items()
    }
    dependencies = dict(parent["dependency_sha256"])
    dependencies.update(parent["language_asset_sha256"])
    for path in (
        PARENT,
        DURATION_AUDIT,
        ROOT / parent["headers"],
        CORPUS / "audit.json",
        CORPUS / "training-streams.jsonl",
    ):
        dependencies[str(path.relative_to(ROOT))] = sha256(path)
    dependencies.update({name: sha256(ROOT / name) for name in indexes.values()})
    pairs = {}
    for identity, raw in selected.items():
        teacher = approved[identity]
        receipt = indexed[identity[0]][identity[1]]
        canonical, header = canonical_input(raw, headers[identity], receipt)
        pair = bind_pair(canonical, header, teacher, receipt, sha256(DURATION_AUDIT))
        pair["original_raw_binding"] = raw
        pair["native_receipt_index"] = indexes[identity[0]]
        for name, digest in (
            (raw["video"], raw["video_sha256"]),
            (canonical["video"], canonical["video_sha256"]),
            (teacher["cache"], teacher["cache_sha256"]),
            (teacher.get("sidecar"), teacher.get("sidecar_sha256")),
        ):
            if name:
                path = (ROOT / name).resolve()
                if not path.is_relative_to(ROOT / "data") or sha256(path) != digest:
                    raise ValueError("teacher/RGB private source bytes changed")
                dependencies[name] = digest
        # Sidecars must independently bind the same full-rate clip and encoder cache.
        layout = SOURCE_LAYOUT.get(identity[0], {})
        if teacher.get("sidecar"):
            sidecar_index = layout["sidecar_index"]
            sidecar = next(
                row
                for row in read_rows(ROOT / sidecar_index)
                if row["sentence_name"] == identity[1]
            )
            if (
                sidecar["resampled_clip_sha256"] != raw["video_sha256"]
                or sidecar["encoder_cache_sha256"] != teacher["cache_sha256"]
                or sidecar["stream_sha256"] != teacher["sidecar_sha256"]
                or sidecar["original_index_sha256"] != sha256(ROOT / indexes[identity[0]])
            ):
                raise ValueError("native teacher sidecar provenance differs")
            dependencies[sidecar_index] = sha256(ROOT / sidecar_index)
        pairs[identity] = pair
    return {
        "eligible_training_clips": audit["valid_training_clips"],
        "examples": [pairs[(r["source"], r["example_id"])] for r in examples],
        "preflight_cases": [
            pairs[(c["training"]["source"], c["training"]["example_id"])] for c in cases
        ],
        "dependency_sha256": dependencies,
        "parent_plan_sha256": sha256(PARENT),
    }


def load_targets(pair):
    from tiny_rgb_visual_teacher import STREAM_WIDTHS
    from train_c5_stem_continuation import stride2_streams

    row = pair["teacher"]
    bound = {**row, "cache": str(ROOT / row["cache"])}
    if row.get("sidecar"):
        bound["sidecar"] = str(ROOT / row["sidecar"])
    values = stride2_streams(bound)
    frames = pair["alignment"]["sampled_frames"]
    if any(values[name].shape != (frames, width) for name, width in STREAM_WIDTHS.items()):
        raise ValueError("native teacher targets have a different sampled timeline/width")
    return values
