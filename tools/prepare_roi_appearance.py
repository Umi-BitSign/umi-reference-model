"""Bound existing training clips to small native-geometry hand/face RGB inputs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
from benchmark_fleurs_dev import sha256
from extract_mediapipe_landmarks import unpack_landmarks
from launch_terra_visual_job import identity, now, write
from prepare_terra_short_curriculum import ROOT, training_pairs
from prepare_terra_short_curriculum import checked as checked_parent
from terra_training_recovery import artifact, verify_snapshot
from tiny_rgb_visual_teacher_data import load_targets

PARENT = ROOT / "runs/terra-short-curriculum-v1/plan.json"
LANDMARK_INDEX = ROOT / "runs/asl-stem-wiki-v1/landmark-receipts.jsonl"
STREAM_INDEX = ROOT / "runs/asl-stem-wiki-v1/stream-cache-stride2.jsonl"
DEMO = ROOT / "artifacts/baseline-6fe8df59/vendor/demo"
POLICY = {
    "patch_side": 64,
    "regions": ["face", "left_hand", "right_hand"],
    "training_clips": 6,
    "feature_diagnostic_clips": 2,
    "maximum_source_frames": 128,
    "maximum_ROI_cache_bytes": 128 * 1024**2,
    "maximum_reference_RGB_bytes": 128 * 1024**2,
    "sampling": "complete source indices0,2,4,..., matching existing native target streams",
    "crop_geometry": "pinned native helpers at64; reject nonsquare hand intersections",
    "color": "source RGB -> native helper BGR crops -> explicit RGB copy",
    "new_downloads": False,
    "GPU_use": False,
    "CPU_threads": 2,
    "no_caption_or_word_boundaries_invented": True,
    "feature_diagnostic_not_translation_quality_gate": True,
}


def select(data):
    training_pairs(data["records"])  # Validate strict outer roles, including development rows.
    chosen, signers = [], set()
    ordered = sorted(
        data["records"],
        key=lambda r: hashlib.sha256(
            f"roi64-v1:{r['source']}:{r['example_id']}".encode()
        ).hexdigest(),
    )
    for row in ordered:
        if row["source"] != "asl_stem_original" or row["training_allowed"] is not True:
            continue
        pair = row["original_pair"]
        if row["signer"] in data["reserved_STEM_participants"]:
            raise ValueError("ROI inputs cannot train on reserved STEM participants")
        if pair["header"]["source_frames"] > POLICY["maximum_source_frames"]:
            continue
        if row["signer"] not in signers:
            chosen.append(row)
            signers.add(row["signer"])
        if len(chosen) == POLICY["training_clips"] + POLICY["feature_diagnostic_clips"]:
            return chosen
    raise ValueError("ROI pilot lacks eight eligible distinct training-origin signers")


class SampledRgbReader:
    """Sequential full-resolution decode; only declared stride2 indices reach crop helpers."""

    def __init__(self, path, header, *, capture_factory=None):
        import cv2

        self.cv2, self.header = cv2, header
        self.capture = (capture_factory or cv2.VideoCapture)(str(path))
        self.source_position = self.sample_position = 0
        self.indices = np.arange(0, header["source_frames"], 2, dtype=np.int64)
        if (
            not self.capture.isOpened()
            or header["source_fps"] != 25
            or self.capture.get(cv2.CAP_PROP_FPS) != 25
            or int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT)) != header["source_frames"]
            or len(self.indices) != header["sampled_frames"]
            or hashlib.sha256(self.indices.tobytes()).hexdigest() != header["frame_indices_sha256"]
        ):
            self.capture.release()
            raise ValueError("ROI decode source or sampled timeline differs")

    def __len__(self):
        return len(self.indices)

    def __enter__(self):
        return self

    def _read(self):
        ok, frame = self.capture.read()
        if not ok or frame is None:
            return None
        if frame.dtype != np.uint8 or list(frame.shape) != [*self.header["source_shape"], 3]:
            raise ValueError("ROI source shape, channels or pixel dtype changed")
        self.source_position += 1
        return frame

    def __getitem__(self, index):
        if index != self.sample_position or not 0 <= index < len(self):
            raise ValueError("ROI cropper cannot silently seek, repeat or reorder frames")
        wanted = int(self.indices[index])
        frame = None
        while self.source_position <= wanted:
            frame = self._read()
            if frame is None:
                raise ValueError("ROI video ended before its declared sampled frame")
        self.sample_position += 1
        return self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)

    def __exit__(self, error_type, error, traceback):
        try:
            if error_type is None:
                while self._read() is not None:
                    if self.source_position > self.header["source_frames"]:
                        raise ValueError("ROI decoded more source frames than bound")
                if self.source_position != self.header[
                    "source_frames"
                ] or self.sample_position != len(self):
                    raise ValueError("ROI decoder did not cover the complete bound source")
        finally:
            self.capture.release()


def rgb_crops(values, frames):
    side = POLICY["patch_side"]
    if len(values) != frames or any(
        not isinstance(v, np.ndarray) or v.shape != (side, side, 3) or v.dtype != np.uint8
        for v in values
    ):
        raise ValueError("ROI helper output is not one uint8 crop per sampled frame")
    return np.ascontiguousarray(np.stack(values)[..., ::-1])


def native_helpers():
    old_path = list(sys.path)
    sys.path.insert(0, str(DEMO))
    try:
        modules = []
        for name in ("crop_hands", "crop_face"):
            spec = importlib.util.spec_from_file_location(f"roi_native_{name}", DEMO / f"{name}.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            modules.append(module)
    finally:
        sys.path[:] = old_path
    reader_source = Path(sys.modules[modules[0].VideoReader.__module__].__file__).resolve()
    if reader_source != (DEMO / "video_reader.py").resolve():
        raise ValueError("ROI native helpers imported an unbound video reader")

    class SquareCheckedHand(modules[0].HandExtractor):
        def crop_frame(self, image, bounding_box):
            value = super().crop_frame(image, bounding_box)
            if value.size == 0 or value.shape[0] != value.shape[1]:
                raise ValueError("ROI native hand intersection would distort aspect ratio")
            return value

    return SquareCheckedHand, modules[1].FaceExtractor, modules[0].VideoReader


def specification():
    data = checked_parent(PARENT)
    landmarks = {
        r["example_id"]: r for r in map(json.loads, LANDMARK_INDEX.read_text().splitlines())
    }
    streams = {r["example_id"]: r for r in map(json.loads, STREAM_INDEX.read_text().splitlines())}
    records = []
    for i, row in enumerate(select(data)):
        pair, ident = row["original_pair"], row["example_id"]
        lm, stream = landmarks[ident], streams[ident]
        landmark_path = ROOT / "data/asl-stem-wiki/landmarks-25fps" / f"{ident}.npz"
        if (
            lm["clip_sha256"] != pair["raw"]["video_sha256"]
            or lm["frames"] != pair["header"]["source_frames"]
            or stream["landmarks_sha256"] != lm["landmarks_sha256"]
            or stream["landmark_receipts_sha256"] != sha256(LANDMARK_INDEX)
            or stream["encoder_cache_sha256"] != pair["teacher"]["cache_sha256"]
            or sha256(landmark_path) != lm["landmarks_sha256"]
        ):
            raise ValueError("ROI landmarks, raw clip and feature receipts differ")
        records.append(
            {
                "original_row": row,
                "roi_role": "training" if i < POLICY["training_clips"] else "feature_diagnostic",
                "roi_training_allowed": i < POLICY["training_clips"],
                "landmarks": {
                    "path": str(landmark_path.relative_to(ROOT)),
                    "sha256": lm["landmarks_sha256"],
                },
                "landmark_receipt": lm,
                "stream_receipt": stream,
            }
        )
    sources = dict(data["source_sha256"])
    for name in (
        "prepare_roi_appearance.py",
        "extract_mediapipe_landmarks.py",
        "tiny_rgb_visual_teacher_data.py",
    ):
        sources[f"tools/{name}"] = sha256(ROOT / "tools" / name)
    return {
        "schema": "umi-studio-roi-appearance-data/1",
        "policy": POLICY,
        "parent": artifact(PARENT),
        "source_sha256": sources,
        "assets": {
            str(p.relative_to(ROOT)): sha256(p)
            for p in (
                LANDMARK_INDEX,
                STREAM_INDEX,
                DEMO / "crop_hands.py",
                DEMO / "crop_face.py",
                DEMO / "video_reader.py",
            )
        },
        "records": records,
        "reserved_STEM_participants": data["reserved_STEM_participants"],
        "ROI_diagnostic_participants": [
            r["original_row"]["signer"] for r in records if not r["roi_training_allowed"]
        ],
        "resource_policy": data["resource_policy"],
        "resource_policy_sha256": data["resource_policy_sha256"],
        "new_downloads": False,
        "new_Terra_requests": 0,
        "optimizer_updates": 0,
        "promotion_authorized": False,
        "literal_word_training_claimed": False,
    }


def checked(path):
    plan = json.loads(path.read_text())
    if plan != specification():
        raise ValueError("ROI data selection, bindings or geometry policy changed")
    verify_snapshot(path, plan)
    return plan


def freeze(path):
    from c5_experiment_sources import archive_sources
    from terra_training_storage import storage

    path = path.resolve()
    if path.parent.exists() or not path.is_relative_to(ROOT / "runs"):
        raise ValueError("ROI data requires a fresh owned plan directory")
    storage(128 * 1024**2)
    plan = specification()
    path.parent.mkdir(parents=True)
    write(path, plan)
    archive_sources(path, path.with_name("source-snapshot"))
    print(
        json.dumps({"plan_sha256": sha256(path), "selected_clips": len(plan["records"])}),
        flush=True,
    )


def preview(output, records):
    import cv2
    from PIL import Image, ImageDraw

    sheet = Image.new("RGB", (960, 116 * len(records)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, row in enumerate(records):
        original = row["original_row"]
        capture = cv2.VideoCapture(str(ROOT / original["original_pair"]["raw"]["video"]))
        try:
            ok, bgr = capture.read()
            if not ok:
                raise ValueError("ROI preview source frame is unavailable")
            frame = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        finally:
            capture.release()
        frame.thumbnail((128, 96))
        top = i * 116
        sheet.paste(frame, (0, top + 20))
        draw.text(
            (2, top + 2),
            f"{row['roi_role']} {original['signer']}: {original['reference']}",
            fill="black",
        )
        with np.load(ROOT / row["cache"]["path"], allow_pickle=False) as archive:
            pixels = archive["pixels"]
            indices = [0, len(pixels) // 3, 2 * len(pixels) // 3, len(pixels) - 1]
            for j, index in enumerate(indices):
                for region in range(3):
                    sheet.paste(
                        Image.fromarray(pixels[index, region]),
                        (136 + j * 200 + region * 64, top + 28),
                    )
    sheet.save(output / "preview.png")


def prepare(path, output):
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "threads;2")
    if os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] != "threads;2":
        raise ValueError("ROI decoding requires the bound two-thread FFmpeg policy")
    import cv2
    from terra_training_storage import storage

    cv2.setNumThreads(2)
    plan = checked(path)
    output = output.resolve()
    if output.exists() or not output.is_relative_to(ROOT / "checkpoints"):
        raise ValueError("ROI cache requires a fresh owned output")
    storage(128 * 1024**2)
    output.mkdir(parents=True)
    write(output / "worker.json", identity(os.getpid(), [sys.executable, *sys.argv]))
    code = 1
    try:
        hands, face, reader_type = native_helpers()
        records, cache_bytes = [], 0
        for number, row in enumerate(plan["records"]):
            pair = row["original_row"]["original_pair"]
            header = pair["header"]
            video = ROOT / pair["raw"]["video"]
            landmarks, metadata = unpack_landmarks(ROOT / row["landmarks"]["path"])
            if (
                metadata["source_sha256"] != pair["raw"]["video_sha256"]
                or metadata["frames"] != header["source_frames"]
                or metadata["fps"] != 25
            ):
                raise ValueError("ROI landmark metadata does not bind its original source")
            sampled = {j: landmarks[i] for j, i in enumerate(range(0, header["source_frames"], 2))}
            for value in sampled.values():
                if value is not None and any(
                    not np.isfinite(np.asarray(points)).all()
                    for points in value.values()
                    if points is not None
                ):
                    raise ValueError("ROI active landmark coordinates are nonfinite")
            side, count = POLICY["patch_side"], header["sampled_frames"]
            with SampledRgbReader(video, header) as reader:
                left, right = hands(output_size=(side, side)).extract_hand_frames(reader, sampled)
            with SampledRgbReader(video, header) as reader:
                facial = face(output_size=(side, side)).extract_face_frames(reader, sampled)
            pixels = np.stack(
                [rgb_crops(facial, count), rgb_crops(left, count), rgb_crops(right, count)], axis=1
            )
            if (
                header["source_frames"] * np.prod(header["source_shape"]) * 3
                > POLICY["maximum_reference_RGB_bytes"]
            ):
                raise ValueError("ROI native reference decode exceeds128MiB")
            reference_reader = reader_type(video)
            full = reference_reader.get_batch(range(len(reference_reader))).asnumpy()[::2].copy()
            del reference_reader
            ref_left, ref_right = hands(output_size=(side, side)).extract_hand_frames(full, sampled)
            ref_face = face(output_size=(side, side)).extract_face_frames(full, sampled)
            reference = np.stack(
                [
                    rgb_crops(ref_face, count),
                    rgb_crops(ref_left, count),
                    rgb_crops(ref_right, count),
                ],
                axis=1,
            )
            if not np.array_equal(pixels, reference):
                raise ValueError("streamed ROI pixels differ from the native reference crop path")
            del reference, full, ref_left, ref_right, ref_face, left, right, facial
            targets = load_targets(pair)
            arrays = {
                "pixels": pixels,
                "frame_indices": np.arange(0, header["source_frames"], 2, dtype=np.int64),
                **targets,
            }
            cache_bytes += sum(v.nbytes for v in arrays.values())
            if cache_bytes > POLICY["maximum_ROI_cache_bytes"]:
                raise RuntimeError("ROI cache exceeds its128MiB RAM bound")
            cache = output / f"clip-{number:03d}.npz"
            with cache.open("xb") as stream:
                np.savez_compressed(stream, **arrays)
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
                json.dumps({"stage": "ROI_data", "clips": number + 1, "cache_bytes": cache_bytes}),
                flush=True,
            )
        preview(output, records)
        checked(path)
        write(
            output / "data.json",
            {
                "schema": "umi-studio-roi-appearance-cache/1",
                "complete": True,
                "plan_sha256": sha256(path),
                "records": records,
                "cache_bytes": cache_bytes,
                "all_streamed_pixels_equal_native_reference": True,
                "preview": artifact(output / "preview.png"),
                "new_Terra_requests": 0,
                "optimizer_updates": 0,
                "GPU_use": False,
                "training_clips": POLICY["training_clips"],
                "feature_diagnostic_clips": POLICY["feature_diagnostic_clips"],
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
