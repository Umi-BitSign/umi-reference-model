"""Expand the validated64px native ROI path to existing short-STEM training clips."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from benchmark_fleurs_dev import sha256
from extract_mediapipe_landmarks import unpack_landmarks
from launch_terra_visual_job import identity, now, write
from prepare_roi_appearance import (
    ROOT,
    SampledRgbReader,
    native_helpers,
    preview,
    rgb_crops,
)
from prepare_roi_appearance import checked as checked_pilot_plan
from prepare_terra_short_curriculum import checked as checked_parent
from prepare_terra_short_curriculum import training_pairs
from terra_training_recovery import artifact, verify_snapshot
from terra_training_storage import storage
from tiny_rgb_visual_teacher_data import load_targets

PARENT = ROOT / "runs/terra-short-curriculum-v1/plan.json"
PILOT = ROOT / "runs/roi-appearance-data-v1/plan.json"
PILOT_DATA = ROOT / "checkpoints/roi-appearance-data-v1/data.json"
LAYOUT = {
    "asl_stem_original": ("asl-stem-wiki-v1", "data/asl-stem-wiki/landmarks-25fps"),
    "asl_stem_supplemental": (
        "asl-stem-wiki-scale-v1",
        "data/asl-stem-wiki/supplemental-landmarks-25fps",
    ),
}
POLICY = {
    "patch_side": 64,
    "training": (
        "all currently approved short-STEM original/supplemental rows except reserved signers"
    ),
    "diagnostic": "the original two ROI diagnostic clips, never optimized",
    "maximum_cache_RAM_bytes": 512 * 1024**2,
    "maximum_reference_frame_bytes": 128 * 1024**2,
    "reference": (
        "unchanged native VideoReader one sampled source frame at a time; no full-RGB video array"
    ),
    "geometry_color_and_fallback": (
        "unchanged native64px hand/face helpers, square hand intersections, explicit BGR->RGB copy"
    ),
    "complete_timeline": "0,2,4,... through the complete original clip, no caption cuts",
    "new_downloads": 0,
    "new_Terra_requests": 0,
    "optimizer_updates": 0,
    "CPU_threads": 2,
    "GPU_use": False,
    "translation_promotion_authorized": False,
}


def select(parent, pilot):
    training_pairs(parent["records"])
    diagnostic_signers = set(pilot["ROI_diagnostic_participants"])
    reserved = set(parent["reserved_STEM_participants"]) | diagnostic_signers
    rows = [
        row
        for row in parent["records"]
        if row["training_allowed"] and row["source"] in LAYOUT and row["signer"] not in reserved
    ]
    if len(rows) != 88 or len({r["signer"] for r in rows}) != 25:
        raise ValueError("expanded ROI inventory differs from the predeclared88 clips/25 signers")
    ordered = sorted(rows, key=lambda r: (r["source"], r["example_id"]))
    chosen = [
        {"original_row": row, "roi_role": "training", "roi_training_allowed": True}
        for row in ordered
    ]
    diagnostics = [r for r in pilot["records"] if r["roi_training_allowed"] is False]
    if (
        len(diagnostics) != 2
        or {r["original_row"]["signer"] for r in diagnostics} != diagnostic_signers
    ):
        raise ValueError("expanded ROI cannot replace the existing feature diagnostics")
    chosen.extend(
        {k: r[k] for k in ("original_row", "roi_role", "roi_training_allowed")} for r in diagnostics
    )
    return chosen


class NativeSampledReader:
    """One-frame native reference decode with no cropper-visible seek/reset method."""

    def __init__(self, reader, header):
        self.reader, self.header, self.next = reader, header, 0
        if (
            len(reader) != header["source_frames"]
            or reader.get_avg_fps() != 25
            or len(range(0, header["source_frames"], 2)) != header["sampled_frames"]
        ):
            raise ValueError("native ROI reference source count/fps differs")

    def __len__(self):
        return self.header["sampled_frames"]

    def __getitem__(self, index):
        if index != self.next or not 0 <= index < len(self):
            raise ValueError("native ROI reference cannot repeat/reorder sampled frames")
        value = self.reader.get_batch([2 * index]).asnumpy()
        if value.dtype != np.uint8 or value.shape != (1, *self.header["source_shape"], 3):
            raise ValueError("native ROI reference frame differs in shape/dtype")
        if value.nbytes > POLICY["maximum_reference_frame_bytes"]:
            raise ValueError("native one-frame ROI reference exceeds128MiB")
        self.next += 1
        return value[0]


def specification():
    parent, pilot = checked_parent(PARENT), checked_pilot_plan(PILOT)
    data = json.loads(PILOT_DATA.read_text())
    terminal = json.loads(PILOT_DATA.with_name("exit.json").read_text())
    worker = json.loads(PILOT_DATA.with_name("worker.json").read_text())
    if (
        not data["complete"]
        or data["plan_sha256"] != sha256(PILOT)
        or terminal["exit_code"] != 0
        or terminal["pid"] != worker["pid"]
    ):
        raise ValueError("expanded ROI requires the completed original native-parity data audit")
    records, indexes, assets = [], {}, {}
    for source, (run_dir, _) in LAYOUT.items():
        indexes[source] = []
        for name in ("landmark-receipts.jsonl", "stream-cache-stride2.jsonl"):
            path = ROOT / "runs" / run_dir / name
            assets[str(path.relative_to(ROOT))] = sha256(path)
            indexes[source].append(
                {r["example_id"]: r for r in map(json.loads, path.read_text().splitlines())}
            )
    for row in select(parent, pilot):
        original = row["original_row"]
        source, ident, pair = original["source"], original["example_id"], original["original_pair"]
        lm_index, stream_index = indexes[source]
        lm, stream = lm_index[ident], stream_index[ident]
        run_dir, landmark_dir = LAYOUT[source]
        landmark_path = ROOT / landmark_dir / f"{ident}.npz"
        if (
            lm["clip_sha256"] != pair["raw"]["video_sha256"]
            or lm["frames"] != pair["header"]["source_frames"]
            or stream["landmarks_sha256"] != lm["landmarks_sha256"]
            or stream["landmark_receipts_sha256"]
            != assets[f"runs/{run_dir}/landmark-receipts.jsonl"]
            or stream["encoder_cache_sha256"] != pair["teacher"]["cache_sha256"]
            or sha256(landmark_path) != lm["landmarks_sha256"]
        ):
            raise ValueError("expanded ROI landmark, source and native teacher bindings differ")
        records.append(
            {
                **row,
                "landmarks": {
                    "path": str(landmark_path.relative_to(ROOT)),
                    "sha256": lm["landmarks_sha256"],
                },
            }
        )
    source = dict(pilot["source_sha256"])
    source["tools/prepare_roi_short_corpus.py"] = sha256(Path(__file__))
    return {
        "schema": "umi-studio-roi-short-corpus-plan/1",
        "policy": POLICY,
        "parent": artifact(PARENT),
        "pilot": artifact(PILOT),
        "pilot_data": artifact(PILOT_DATA),
        "source_sha256": source,
        "assets": {**pilot["assets"], **assets},
        "records": records,
        "counts": dict(Counter(r["roi_role"] for r in records)),
        "training_sampled_frames": sum(
            r["original_row"]["original_pair"]["header"]["sampled_frames"]
            for r in records
            if r["roi_training_allowed"]
        ),
        "reserved_STEM_participants": parent["reserved_STEM_participants"],
        "ROI_diagnostic_participants": pilot["ROI_diagnostic_participants"],
        "resource_policy": pilot["resource_policy"],
        "resource_policy_sha256": pilot["resource_policy_sha256"],
        "new_downloads": 0,
        "new_Terra_requests": 0,
        "optimizer_updates": 0,
        "primary_translation_development_used": False,
        "translation_promotion_authorized": False,
    }


def checked(path):
    plan = json.loads(path.read_text())
    if plan != specification():
        raise ValueError("expanded ROI source/roles/geometry/resource policy changed")
    verify_snapshot(path, plan)
    return plan


def freeze(path):
    from c5_experiment_sources import archive_sources

    path = path.resolve()
    if path.parent.exists() or not path.is_relative_to(ROOT / "runs"):
        raise ValueError("expanded ROI needs a fresh owned plan")
    storage(POLICY["maximum_cache_RAM_bytes"])
    plan = specification()
    path.parent.mkdir(parents=True)
    write(path, plan)
    archive_sources(path, path.with_name("source-snapshot"))
    print(
        json.dumps(
            {
                "plan": artifact(path),
                "counts": plan["counts"],
                "training_sampled_frames": plan["training_sampled_frames"],
            }
        ),
        flush=True,
    )


def prepare(path, output):
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "threads;2")
    if os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] != "threads;2":
        raise ValueError("expanded ROI requires two-thread FFmpeg")
    import cv2

    cv2.setNumThreads(2)
    plan = checked(path)
    output = output.resolve()
    if output.exists() or not output.is_relative_to(ROOT / "checkpoints"):
        raise ValueError("expanded ROI cache requires a fresh owned output")
    storage(POLICY["maximum_cache_RAM_bytes"])
    output.mkdir(parents=True)
    write(output / "worker.json", identity(os.getpid(), [sys.executable, *sys.argv]))
    code = 1
    try:
        hands, face, reader_type = native_helpers()
        records, cache_bytes = [], 0
        for number, row in enumerate(plan["records"]):
            storage(32 * 1024**2, output=output)
            pair = row["original_row"]["original_pair"]
            header, video = pair["header"], ROOT / pair["raw"]["video"]
            landmarks, metadata = unpack_landmarks(ROOT / row["landmarks"]["path"])
            if (
                metadata["source_sha256"] != pair["raw"]["video_sha256"]
                or metadata["frames"] != header["source_frames"]
                or metadata["fps"] != 25
            ):
                raise ValueError("expanded ROI landmark metadata differs from source")
            sampled = {j: landmarks[i] for j, i in enumerate(range(0, header["source_frames"], 2))}
            if any(
                not np.isfinite(np.asarray(points)).all()
                for value in sampled.values()
                if value is not None
                for points in value.values()
                if points is not None
            ):
                raise ValueError("expanded ROI has nonfinite active landmark coordinates")
            count = header["sampled_frames"]
            with SampledRgbReader(video, header) as reader:
                left, right = hands(output_size=(64, 64)).extract_hand_frames(reader, sampled)
            with SampledRgbReader(video, header) as reader:
                facial = face(output_size=(64, 64)).extract_face_frames(reader, sampled)
            pixels = np.stack(
                [rgb_crops(facial, count), rgb_crops(left, count), rgb_crops(right, count)], axis=1
            )
            del left, right, facial
            native = reader_type(video)
            try:
                ref_left, ref_right = hands(output_size=(64, 64)).extract_hand_frames(
                    NativeSampledReader(native, header), sampled
                )
                ref_face = face(output_size=(64, 64)).extract_face_frames(
                    NativeSampledReader(native, header), sampled
                )
            finally:
                del native
            reference = np.stack(
                [
                    rgb_crops(ref_face, count),
                    rgb_crops(ref_left, count),
                    rgb_crops(ref_right, count),
                ],
                axis=1,
            )
            if not np.array_equal(pixels, reference):
                raise ValueError("expanded streaming ROI differs from independent native reference")
            del reference, ref_left, ref_right, ref_face
            values = {
                "pixels": pixels,
                "frame_indices": np.arange(0, header["source_frames"], 2, dtype=np.int64),
                **load_targets(pair),
            }
            cache_bytes += sum(v.nbytes for v in values.values())
            if cache_bytes > POLICY["maximum_cache_RAM_bytes"]:
                raise ValueError("expanded ROI exceeds its512MiB cache RAM bound")
            cache = output / f"clip-{number:03d}.npz"
            with cache.open("xb") as stream:
                np.savez_compressed(stream, **values)
                stream.flush()
                os.fsync(stream.fileno())
            records.append(
                {
                    **row,
                    "cache": artifact(cache),
                    "frames": count,
                    "streamed_pixels_equal_native_reference": True,
                }
            )
            print(
                json.dumps(
                    {"stage": "expanded_ROI_data", "clips": number + 1, "cache_bytes": cache_bytes}
                ),
                flush=True,
            )
        # Eight-row review sheet is bounded; all90 caches still receive exact native parity.
        preview(output, records[:6] + records[-2:])
        checked(path)
        write(
            output / "data.json",
            {
                "schema": "umi-studio-roi-short-corpus-cache/1",
                "complete": True,
                "plan_sha256": sha256(path),
                "records": records,
                "cache_bytes": cache_bytes,
                "all_streamed_pixels_equal_native_reference": True,
                "preview": artifact(output / "preview.png"),
                "new_Terra_requests": 0,
                "optimizer_updates": 0,
                "GPU_use": False,
                "counts": plan["counts"],
                "primary_translation_development_used": False,
                "translation_promotion_authorized": False,
            },
        )
        code = 0
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
