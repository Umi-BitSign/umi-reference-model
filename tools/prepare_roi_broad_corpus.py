"""Prepare a cross-source short-phrase ROI stage without whole-corpus RAM residency."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from audit_roi_training_coverage import eligible_pairs, word_band
from benchmark_fleurs_dev import normalize, sha256
from extract_mediapipe_landmarks import unpack_landmarks
from launch_terra_visual_job import identity, now, write
from prepare_roi_appearance import DEMO, ROOT, SampledRgbReader, native_helpers, rgb_crops
from prepare_roi_short_corpus import NativeSampledReader
from terra_training_recovery import artifact, verify_snapshot
from terra_training_storage import storage
from tiny_rgb_visual_teacher_data import load_targets

CORPUS = ROOT / "runs/tiny-rgb-visual-teacher-corpus-v1/plan.json"
PREVIOUS = ROOT / "runs/roi-short-corpus-data-v1/plan.json"
PREVIOUS_DATA = ROOT / "checkpoints/roi-short-corpus-data-v1/data.json"
LAYOUT = {
    "asl_stem_original": (
        "runs/asl-stem-wiki-v1/landmark-receipts.jsonl",
        "data/asl-stem-wiki/landmarks-25fps",
        "runs/asl-stem-wiki-v1/stream-cache-stride2.jsonl",
    ),
    "asl_stem_supplemental": (
        "runs/asl-stem-wiki-scale-v1/landmark-receipts.jsonl",
        "data/asl-stem-wiki/supplemental-landmarks-25fps",
        "runs/asl-stem-wiki-scale-v1/stream-cache-stride2.jsonl",
    ),
    "how2sign_precut_expansion": (
        "runs/how2sign-precut-expansion-v1/landmark-receipts-prefix-1650.jsonl",
        "data/how2sign/precut-expansion-landmarks-25fps",
        "runs/how2sign-precut-expansion-v1/stream-cache-stride2-index.jsonl",
    ),
    "openasl_scale": (
        "runs/openasl-scale-v1/landmark-receipts.jsonl",
        "data/openasl-scale/landmarks",
        "runs/openasl-scale-v1/stream-cache-stride2.jsonl",
    ),
}
POLICY = {
    "patch_side": 64,
    "maximum_caption_words": 6,
    "selection": "all approved 2-6-word pairs in existing landmark-bound source families",
    "maximum_sampled_frames": 149,
    "maximum_clip_arrays_bytes": 128 * 1024**2,
    "maximum_preparation_live_arrays_bytes": 512 * 1024**2,
    "maximum_total_uncompressed_arrays_bytes": 1536 * 1024**2,
    "cache_access": "one complete clip at a time; no whole-corpus in-memory cache",
    "timeline": "all source indices0,2,4,...; no caption or interior-frame truncation",
    "native_pixel_parity": "every new training clip, every face/left/right crop",
    "diagnostics": "reuse the two original non-training P19/P0 caches with byte verification",
    "missing_landmarks": "older FLEURS/How2Sign families await separately bound landmarks",
    "selection_uses_development_text_or_scores": False,
    "augmentation": "none during data preparation",
    "CPU_threads": 2,
    "GPU_use": False,
    "new_downloads": 0,
    "new_Terra_requests": 0,
    "optimizer_updates": 0,
    "translation_promotion_authorized": False,
}


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def indexed(values):
    result = {}
    for row in values:
        key = row.get("example_id", row.get("sentence_name"))
        if not isinstance(key, str) or not key or key in result:
            raise ValueError("broad ROI index has missing/duplicate identities")
        result[key] = row
    return result


def row_digest(row):
    return hashlib.sha256(
        json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def select(pairs, reserved):
    eligible, excluded = eligible_pairs(pairs, reserved)
    supported = [p for p in eligible if p["raw"]["source"] in LAYOUT]
    selected = [
        p
        for p in supported
        if 2 <= len(normalize(p["raw"]["reference"])) <= POLICY["maximum_caption_words"]
    ]
    if any(
        not 1 <= p["header"]["sampled_frames"] <= POLICY["maximum_sampled_frames"] for p in selected
    ):
        raise ValueError("broad short stage cannot truncate or silently omit a long clip")
    return sorted(selected, key=lambda p: (p["raw"]["source"], p["raw"]["example_id"])), {
        "approved_pairs": len(pairs),
        "excluded_reserved_by_source": excluded,
        "eligible_after_reserved_signers": len(eligible),
        "metadata_pairs_in_existing_landmark_families": len(supported),
        "eligible_pairs_awaiting_landmark_binding": len(eligible) - len(supported),
        "selected_short_training_pairs": len(selected),
        "selected_by_source": dict(Counter(p["raw"]["source"] for p in selected)),
        "selected_by_word_band": dict(
            Counter(word_band(len(normalize(p["raw"]["reference"]))) for p in selected)
        ),
        "selected_sampled_frames": sum(p["header"]["sampled_frames"] for p in selected),
    }


def binding(pair, landmark, stream, landmark_index_sha):
    raw, header, teacher = pair["raw"], pair["header"], pair["teacher"]
    if (
        row_digest(stream) != pair["native_receipt_row_sha256"]
        or stream.get("split") != "training"
        or stream.get("landmark_receipts_sha256") != landmark_index_sha
        or landmark.get("clip_sha256") != raw["video_sha256"]
        or landmark.get("frames") != header["source_frames"]
        or stream.get("landmarks_sha256") != landmark.get("landmarks_sha256")
        or stream.get("encoder_cache_sha256") != teacher["cache_sha256"]
        or stream.get("resampled_clip_sha256", stream.get("clip_sha256")) != raw["video_sha256"]
        or stream.get("temporal_stride") != 2
        or pair["native_teacher_stride"] != 2
        or header["video_sha256"] != raw["video_sha256"]
        or header["source_fps"] != 25
        or raw["video_fps"] != 25
    ):
        raise ValueError("broad ROI landmark/native teacher/canonical video binding differs")
    stem = landmark.get("clip_id", raw["example_id"])
    if Path(raw["video"]).stem != stem:
        raise ValueError("broad ROI canonical video and landmark filename differ")
    return stem


def checked_file(path, digest, *, private=False):
    target = (ROOT / path).resolve()
    if private and not target.is_relative_to(ROOT / "data"):
        raise ValueError("broad ROI private media must remain in data/")
    if sha256(target) != digest:
        raise ValueError(f"broad ROI bound file changed: {path}")
    return target


def specification():
    corpus, previous, previous_data = [
        json.loads(path.read_text()) for path in (CORPUS, PREVIOUS, PREVIOUS_DATA)
    ]
    manifest = checked_file(corpus["native_pairs"], corpus["native_pairs_sha256"])
    if (
        not previous_data["complete"]
        or previous_data["plan_sha256"] != sha256(PREVIOUS)
        or previous_data["counts"] != {"training": 88, "feature_diagnostic": 2}
    ):
        raise ValueError("broad ROI requires the original complete88/2 data receipts")
    reserved = previous["reserved_STEM_participants"] + previous["ROI_diagnostic_participants"]
    pairs, counts = select(rows(manifest), reserved)
    if len(pairs) != 318 or counts["metadata_pairs_in_existing_landmark_families"] != 2843:
        raise ValueError("broad ROI cannot substitute a changed source inventory")
    training_keys = {(p["raw"]["source"], p["raw"]["example_id"]) for p in pairs}
    old_keys = {
        (r["original_row"]["source"], r["original_row"]["example_id"])
        for r in previous_data["records"]
        if r["roi_training_allowed"]
    }
    if len(old_keys) != 88 or not old_keys <= training_keys:
        raise ValueError("broad ROI lost part of the original training inventory")
    assets, indexes, records = {}, {}, []
    for source, (landmark_index, _, stream_index) in LAYOUT.items():
        for name in (landmark_index, stream_index):
            assets[name] = sha256(ROOT / name)
        indexes[source] = (indexed(rows(ROOT / landmark_index)), indexed(rows(ROOT / stream_index)))
    for pair in pairs:
        raw = pair["raw"]
        source, ident = raw["source"], raw["example_id"]
        landmark_index, landmark_dir, stream_index = LAYOUT[source]
        lm, stream = (index[ident] for index in indexes[source])
        if pair["native_receipt_index"] != stream_index:
            raise ValueError("broad ROI native teacher index changed")
        stem = binding(pair, lm, stream, assets[landmark_index])
        lm_path = f"{landmark_dir}/{stem}.npz"
        checked_file(raw["video"], raw["video_sha256"], private=True)
        checked_file(lm_path, lm["landmarks_sha256"], private=True)
        checked_file(pair["teacher"]["cache"], pair["teacher"]["cache_sha256"], private=True)
        records.append(
            {
                "original_row": {**raw, "original_pair": pair},
                "roi_role": "training",
                "roi_training_allowed": True,
                "landmarks": {"path": lm_path, "sha256": lm["landmarks_sha256"]},
            }
        )
    diagnostics = [r for r in previous_data["records"] if not r["roi_training_allowed"]]
    if {r["original_row"]["signer"] for r in diagnostics} != {"P0", "P19"}:
        raise ValueError("broad ROI cannot replace its two feature diagnostics")
    for record in diagnostics:
        checked_file(record["cache"]["path"], record["cache"]["sha256"])
        records.append({**record, "reuse_existing_diagnostic_cache": True})
    source_names = set(previous["source_sha256"]) | {
        "tools/prepare_roi_broad_corpus.py",
        "tools/audit_roi_training_coverage.py",
        "tools/prepare_roi_short_corpus.py",
        "tools/terra_training_storage.py",
    }
    assets.update(
        {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (
                DEMO / "crop_hands.py",
                DEMO / "crop_face.py",
                DEMO / "video_reader.py",
            )
        }
    )
    return {
        "schema": "umi-studio-roi-broad-short-data/1",
        "policy": POLICY,
        "parent_corpus": artifact(CORPUS),
        "native_pairs": artifact(manifest),
        "previous_data_plan": artifact(PREVIOUS),
        "previous_data": artifact(PREVIOUS_DATA),
        "source_sha256": {name: sha256(ROOT / name) for name in sorted(source_names)},
        "assets": assets,
        "records": records,
        "coverage_counts": counts,
        "reserved_STEM_participants": previous["reserved_STEM_participants"],
        "ROI_diagnostic_participants": previous["ROI_diagnostic_participants"],
        "resource_policy": previous["resource_policy"],
        "resource_policy_sha256": previous["resource_policy_sha256"],
        "training_readiness_requires_complete_preparation": True,
        "new_Terra_requests": 0,
        "optimizer_updates": 0,
        "promotion_authorized": False,
    }


def checked(path):
    plan = json.loads(path.read_text())
    if plan != specification():
        raise ValueError("broad ROI immutable data, source, roles or policy changed")
    checked_file(plan["resource_policy"], plan["resource_policy_sha256"])
    verify_snapshot(path, plan)
    return plan


def freeze(path):
    from c5_experiment_sources import archive_sources

    path = path.resolve()
    if path.parent.exists() or not path.is_relative_to(ROOT / "runs"):
        raise ValueError("broad ROI needs a fresh owned plan directory")
    storage(POLICY["maximum_total_uncompressed_arrays_bytes"])
    plan = specification()
    path.parent.mkdir(parents=True)
    write(path, plan)
    archive_sources(path, path.with_name("source-snapshot"))
    print(json.dumps({"plan": artifact(path), "counts": plan["coverage_counts"]}), flush=True)


def sampled_landmarks(record):
    pair = record["original_row"]["original_pair"]
    header = pair["header"]
    landmarks, metadata = unpack_landmarks(
        checked_file(record["landmarks"]["path"], record["landmarks"]["sha256"], private=True)
    )
    if (
        metadata["source_sha256"] != pair["raw"]["video_sha256"]
        or metadata["frames"] != header["source_frames"]
        or len(landmarks) != header["source_frames"]
        or metadata["fps"] != 25
        or metadata["mediapipe_version"] != "0.10.14"
        or [metadata["height"], metadata["width"]] != header["source_shape"]
    ):
        raise ValueError("broad ROI landmark topology/source metadata differs")
    sampled = {j: landmarks[i] for j, i in enumerate(range(0, header["source_frames"], 2))}
    if any(
        not np.isfinite(np.asarray(points)).all()
        for value in sampled.values()
        if value is not None
        for points in value.values()
        if points is not None
    ):
        raise ValueError("broad ROI active landmark coordinates are nonfinite")
    return sampled


def crops_with_native_parity(record, helpers):
    hands, face, reader_type = helpers
    pair = record["original_row"]["original_pair"]
    header = pair["header"]
    video = checked_file(pair["raw"]["video"], pair["raw"]["video_sha256"], private=True)
    sampled, count = sampled_landmarks(record), header["sampled_frames"]
    with SampledRgbReader(video, header) as reader:
        left, right = hands(output_size=(64, 64)).extract_hand_frames(reader, sampled)
    with SampledRgbReader(video, header) as reader:
        facial = face(output_size=(64, 64)).extract_face_frames(reader, sampled)
    pixels = np.stack(
        [rgb_crops(facial, count), rgb_crops(left, count), rgb_crops(right, count)], axis=1
    )
    del facial, left, right
    native = reader_type(video)
    try:
        left, right = hands(output_size=(64, 64)).extract_hand_frames(
            NativeSampledReader(native, header), sampled
        )
        facial = face(output_size=(64, 64)).extract_face_frames(
            NativeSampledReader(native, header), sampled
        )
    finally:
        del native
    for index, region in enumerate((facial, left, right)):
        if not np.array_equal(pixels[:, index], rgb_crops(region, count)):
            raise ValueError("broad ROI streamed pixels differ from the native reference")
    return pixels


def validate_arrays(values, pair):
    from tiny_rgb_visual_teacher import STREAM_WIDTHS

    header = pair["header"]
    frames = header["sampled_frames"]
    indices = np.arange(0, header["source_frames"], 2, dtype=np.int64)
    if (
        set(values) != {"pixels", "frame_indices", *STREAM_WIDTHS}
        or values["pixels"].dtype != np.uint8
        or values["pixels"].shape != (frames, 3, 64, 64, 3)
        or values["frame_indices"].dtype != np.int64
        or not np.array_equal(values["frame_indices"], indices)
        or len(indices) != frames
        or header["frame_indices_sha256"] != hashlib.sha256(indices.tobytes()).hexdigest()
        or pair["alignment"]["sampled_frames"] != frames
        or pair["alignment"]["frame_indices_sha256"]
        != hashlib.sha256(indices.tobytes()).hexdigest()
        or any(
            values[name].dtype != np.float32
            or values[name].shape != (frames, width)
            or not np.isfinite(values[name]).all()
            for name, width in STREAM_WIDTHS.items()
        )
    ):
        raise ValueError("broad ROI cached arrays differ from complete native targets")
    count = sum(value.nbytes for value in values.values())
    if count > POLICY["maximum_clip_arrays_bytes"]:
        raise ValueError("broad ROI complete clip exceeds its bounded resident arrays")
    return count


def prepare(path, output):
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "threads;2")
    if os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] != "threads;2":
        raise ValueError("broad ROI decoding requires the bound two-thread FFmpeg policy")
    import cv2

    cv2.setNumThreads(2)
    path, output = path.resolve(), output.resolve()
    if output.exists() or not output.is_relative_to(ROOT / "checkpoints"):
        raise ValueError("broad ROI preparation needs a fresh owned output")
    plan = checked(path)
    storage(POLICY["maximum_total_uncompressed_arrays_bytes"])
    output.mkdir(parents=True)
    write(output / "worker.json", identity(os.getpid(), sys.argv))
    write(output / "run.json", {"plan_sha256": sha256(path), "started_at": now()})
    code, records, total_bytes = 1, [], 0
    current = None
    try:
        helpers = native_helpers()
        for number, record in enumerate(plan["records"]):
            current = record["original_row"]["example_id"]
            pair = record["original_row"]["original_pair"]
            if record.get("reuse_existing_diagnostic_cache"):
                cache = checked_file(record["cache"]["path"], record["cache"]["sha256"])
                with np.load(cache, allow_pickle=False) as archive:
                    values = {name: archive[name] for name in archive.files}
                arrays_bytes = validate_arrays(values, pair)
            else:
                storage(POLICY["maximum_clip_arrays_bytes"], output=output)
                values = {
                    "pixels": crops_with_native_parity(record, helpers),
                    "frame_indices": np.arange(
                        0, pair["header"]["source_frames"], 2, dtype=np.int64
                    ),
                    **load_targets(pair),
                }
                arrays_bytes = validate_arrays(values, pair)
                cache = output / f"clip-{number:04d}.npz"
                storage(arrays_bytes + 1024**2, output=output)
                with cache.open("xb") as stream:
                    np.savez_compressed(stream, **values)
                    stream.flush()
                    os.fsync(stream.fileno())
            total_bytes += arrays_bytes
            if total_bytes > POLICY["maximum_total_uncompressed_arrays_bytes"]:
                raise ValueError("broad ROI exceeds its predeclared total arrays budget")
            records.append(
                {
                    **record,
                    "cache": artifact(cache),
                    "frames": pair["header"]["sampled_frames"],
                    "arrays_bytes": arrays_bytes,
                    "streamed_pixels_equal_native_reference": True,
                }
            )
            del values
            print(
                json.dumps(
                    {
                        "stage": "broad_short_ROI_data",
                        "clips": number + 1,
                        "source": record["original_row"]["source"],
                        "total_arrays_bytes": total_bytes,
                    }
                ),
                flush=True,
            )
        checked(path)
        write(
            output / "data.json",
            {
                "schema": "umi-studio-roi-broad-short-cache/1",
                "complete": True,
                "plan_sha256": sha256(path),
                "records": records,
                "total_uncompressed_arrays_bytes": total_bytes,
                "all_streamed_pixels_equal_native_reference": True,
                "counts": {"training": 318, "feature_diagnostic": 2},
                "coverage_counts": plan["coverage_counts"],
                "GPU_use": False,
                "new_Terra_requests": 0,
                "optimizer_updates": 0,
                "translation_promotion_authorized": False,
            },
        )
        code = 0
    except Exception as error:
        write(
            output / "failure.json",
            {
                "example_id": current,
                "completed_clips": len(records),
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise
    finally:
        write(output / "exit.json", {"pid": os.getpid(), "exit_code": code, "observed_at": now()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.freeze:
        freeze(args.freeze)
    elif args.plan and args.output:
        prepare(args.plan, args.output)
    else:
        parser.error("provide --freeze or --plan and --output")


if __name__ == "__main__":
    main()
