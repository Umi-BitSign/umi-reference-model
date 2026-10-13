"""Cache 168px DINO streams paired with frozen native-224 training streams."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from benchmark_fleurs_dev import sha256
from cache_asl_stem_wiki_streams import temporal_subsample_landmarks
from extract_mediapipe_landmarks import unpack_landmarks
from freeze_c5_crop_calibration_plan import SCHEMA as PLAN_SCHEMA
from resolution_policy import set_visual_crop_size

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "umi-studio-c5-crop-calibration-pair/1"


def checked_index(
    path: Path, cache_dir: Path, plan_hash: str, augmentation_profile: str = "clean"
) -> set[str]:
    completed = set()
    if not path.exists():
        return completed
    for line in path.read_text().splitlines():
        row = json.loads(line)
        name = row.get("sentence_name")
        cache = cache_dir / f"{name}.npz"
        if (
            row.get("schema") != SCHEMA
            or row.get("plan_sha256") != plan_hash
            or row.get("visual_crop_size") != 168
            or row.get("augmentation_profile", "clean") != augmentation_profile
            or name in completed
            or not cache.is_file()
            or row.get("crop168_cache_sha256") != sha256(cache)
        ):
            raise ValueError(f"crop calibration pair differs: {name}")
        completed.add(name)
    return completed


def shifted_frames(frames: list[np.ndarray], name: str) -> list[np.ndarray]:
    """Apply one deterministic mild translation to every crop in a clip."""
    import cv2

    digest = hashlib.sha256(f"crop-shift-v1:{name}".encode()).digest()
    dx_fraction = (int.from_bytes(digest[:4], "big") / (2**32 - 1) * 2 - 1) * 0.04
    dy_fraction = (int.from_bytes(digest[4:8], "big") / (2**32 - 1) * 2 - 1) * 0.04
    output = []
    for frame in frames:
        height, width = frame.shape[:2]
        matrix = np.asarray(
            [[1, 0, round(dx_fraction * width)], [0, 1, round(dy_fraction * height)]],
            dtype=np.float32,
        )
        output.append(
            cv2.warpAffine(
                frame,
                matrix,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--clip-dir", type=Path, required=True)
    parser.add_argument("--landmark-dir", type=Path, required=True)
    parser.add_argument("--native-cache-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--mps-idle-fraction", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=169)
    parser.add_argument(
        "--augmentation-profile",
        choices=("clean", "mild_shift_v1"),
        default="clean",
    )
    parser.add_argument("--miner-maintenance-mode", action="store_true")
    args = parser.parse_args()
    if not args.miner_maintenance_mode:
        parser.error("pair extraction requires explicit miner maintenance mode")
    if not 1 <= args.limit <= 169:
        parser.error("limit must be in [1, 169]")
    if (args.device == "mps" and args.mps_idle_fraction != 0.5) or (
        args.device == "cpu" and args.mps_idle_fraction != 0
    ):
        parser.error("device idle fraction violates the resource policy")
    if not all(
        path.is_dir()
        for path in (args.bundle, args.clip_dir, args.landmark_dir, args.native_cache_dir)
    ) or not args.plan.is_file():
        parser.error("pair extraction inputs are missing")
    plan = json.loads(args.plan.read_text())
    if plan.get("schema") != PLAN_SCHEMA or plan.get("examples") != 169:
        raise ValueError("crop calibration plan differs")
    plan_hash = sha256(args.plan)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.index.parent.mkdir(parents=True, exist_ok=True)
    completed = checked_index(
        args.index, args.cache_dir, plan_hash, args.augmentation_profile
    )

    bundle = args.bundle.resolve()
    for key, value in {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "SHUBERT_DINOV2_SOURCE": str(bundle / "vendor/dinov2-source"),
        "PYTORCH_ENABLE_MPS_FALLBACK": "1",
        "OMP_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
    }.items():
        os.environ[key] = value
    sys.path.insert(0, str(bundle))
    import torch
    from runtime import SHuBERTInferenceRuntime, _rgb_crop_frames

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    model = SHuBERTInferenceRuntime(
        bundle,
        device=args.device,
        dino_batch_size=64,
        model_execution_concurrency=1,
        verify_assets=True,
    )
    from umi_video_reader import VideoReader

    model._video_reader = VideoReader
    visual_policy = set_visual_crop_size(model, 168)
    identity_hash = sha256(bundle / "MODEL_IDENTITY.json")
    runtime_hash = sha256(bundle / "runtime.py")
    new_count = 0
    for record in plan["records"]:
        name = record["sentence_name"]
        if name in completed:
            continue
        if new_count >= args.limit:
            break
        started = time.perf_counter()
        clip = args.clip_dir / f"{name}.mp4"
        landmark_path = args.landmark_dir / f"{name}.npz"
        native_path = args.native_cache_dir / f"{name}.npz"
        if (
            sha256(clip) != record["resampled_clip_sha256"]
            or sha256(landmark_path) != record["landmarks_sha256"]
            or sha256(native_path) != record["encoder_cache_sha256"]
        ):
            raise ValueError(f"crop calibration source differs: {name}")
        reader = model._video_reader(clip)
        frames = reader.get_batch(range(len(reader))).asnumpy()
        landmarks, metadata = unpack_landmarks(landmark_path)
        if len(landmarks) != len(frames) or metadata["source_sha256"] != sha256(clip):
            raise ValueError(f"crop calibration landmarks differ: {name}")
        frames = np.ascontiguousarray(frames[::2])
        landmarks = temporal_subsample_landmarks(landmarks, 2)
        left_frames, right_frames = model._hand_extractor.extract_hand_frames(
            frames, landmarks
        )
        face_frames = model._face_extractor.extract_face_frames(frames, landmarks)
        if args.augmentation_profile == "mild_shift_v1":
            left_frames = shifted_frames(left_frames, name)
            right_frames = shifted_frames(right_frames, name)
            face_frames = shifted_frames(face_frames, name)
        streams = {
            "face_features": model._face_model.extract_embeddings_from_frames(
                _rgb_crop_frames(face_frames)
            ).astype(np.float32),
            "left_hand_features": model._hand_model.extract_embeddings_from_frames(
                _rgb_crop_frames(left_frames)
            ).astype(np.float32),
            "right_hand_features": model._hand_model.extract_embeddings_from_frames(
                _rgb_crop_frames(right_frames)
            ).astype(np.float32),
        }
        with np.load(native_path, allow_pickle=False) as native:
            if any(
                value.shape != native[key].shape
                or value.shape != (record["frames"], 384)
                for key, value in streams.items()
            ):
                raise ValueError(f"crop calibration stream shape differs: {name}")
        cache = args.cache_dir / f"{name}.npz"
        temporary = cache.with_suffix(".partial.npz")
        if cache.exists() or temporary.exists():
            raise ValueError(f"unindexed crop calibration cache exists: {cache}")
        try:
            with temporary.open("xb") as output:
                np.savez(output, **streams)
                output.flush()
                os.fsync(output.fileno())
            temporary.rename(cache)
        finally:
            temporary.unlink(missing_ok=True)
        active_seconds = time.perf_counter() - started
        row = {
            "schema": SCHEMA,
            "sentence_name": name,
            "signer": record["signer"],
            "source_group": record["source_group"],
            "frames": record["frames"],
            "plan_sha256": plan_hash,
            "augmentation_profile": args.augmentation_profile,
            "native224_cache_sha256": record["encoder_cache_sha256"],
            "crop168_cache_sha256": sha256(cache),
            "bundle_identity_sha256": identity_hash,
            "bundle_runtime_sha256": runtime_hash,
            **visual_policy,
            "device": args.device,
            "mps_idle_fraction": args.mps_idle_fraction,
            "miner_health_policy": "user_authorized_maintenance",
            "active_seconds": active_seconds,
        }
        with args.index.open("a", encoding="utf-8") as output:
            output.write(json.dumps(row, sort_keys=True) + "\n")
            output.flush()
            os.fsync(output.fileno())
        completed.add(name)
        new_count += 1
        print(f"paired {len(completed)}/169 {name}", flush=True)
        if args.device == "mps":
            time.sleep(
                active_seconds
                * args.mps_idle_fraction
                / (1 - args.mps_idle_fraction)
            )


if __name__ == "__main__":
    main()
