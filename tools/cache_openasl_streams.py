"""Cache stride-2 C5 visual streams for the frozen OpenASL pilot."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from benchmark_fleurs_dev import sha256
from cache_asl_stem_wiki_streams import temporal_subsample_landmarks
from cache_fleurs_encoder import checked_stream_features
from extract_mediapipe_landmarks import unpack_landmarks
from prepare_openasl_landmarks import BBOX_POLICY, FULL_FRAME_POLICY
from openasl_scale_storage import checked_storage

ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def selection_records(selection: dict) -> tuple[list[dict], int]:
    records = selection.get("records", [])
    if selection.get("schema") == "umi-studio-openasl-pilot-selection/1":
        total = selection.get("clips")
        if total != 120 or selection.get("source_videos") != 60:
            raise ValueError("OpenASL pilot selection differs")
    elif selection.get("schema") == "umi-studio-openasl-scale-selection/1":
        total = selection.get("selected_new_clips")
        if total != 360 or selection.get("selected_new_sources") != 180:
            raise ValueError("OpenASL scale selection differs")
    else:
        raise ValueError("unsupported OpenASL selection")
    if len(records) != total or len({row["example_id"] for row in records}) != total:
        raise ValueError("OpenASL selection records differ")
    return records, total


def checked_index(
    path: Path,
    cache_dir: Path,
    selection_hash: str,
    landmark_receipts_hash: str,
    spatial_policy: str = FULL_FRAME_POLICY,
) -> set[str]:
    completed = set()
    for row in read_jsonl(path):
        identifier = row.get("clip_id")
        cache = cache_dir / f"{identifier}.npz"
        if (
            row.get("schema") != "umi-studio-openasl-stream-cache/1"
            or row.get("selection_sha256") != selection_hash
            or row.get("landmark_receipts_sha256") != landmark_receipts_hash
            or row.get("temporal_stride") != 2
            or row.get("frames") != (row.get("source_frames", 0) + 1) // 2
            or row.get("spatial_policy") != spatial_policy
            or identifier in completed
            or not cache.is_file()
            or row.get("encoder_cache_sha256") != sha256(cache)
        ):
            raise ValueError(f"OpenASL stream cache differs: {identifier}")
        completed.add(identifier)
    return completed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--landmark-receipts", type=Path, required=True)
    parser.add_argument("--clip-dir", type=Path, required=True)
    parser.add_argument("--landmark-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--mps-idle-fraction", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--miner-maintenance-mode", action="store_true")
    parser.add_argument(
        "--spatial-policy",
        choices=(FULL_FRAME_POLICY, BBOX_POLICY),
        default=FULL_FRAME_POLICY,
    )
    args = parser.parse_args()
    if not args.miner_maintenance_mode:
        parser.error("OpenASL caching requires user-authorized miner maintenance mode")
    if not 1 <= args.limit <= 120:
        parser.error("limit must be in [1, 120]")
    if args.device == "mps" and args.mps_idle_fraction != 0.5:
        parser.error("MPS extraction must reserve half of average GPU capacity")
    if args.device == "cpu" and args.mps_idle_fraction != 0:
        parser.error("CPU extraction requires --mps-idle-fraction 0")
    if (
        not args.selection.is_file()
        or not args.landmark_receipts.is_file()
        or not args.clip_dir.is_dir()
        or not args.landmark_dir.is_dir()
        or args.baseline.resolve() != ROOT / "artifacts/baseline-6fe8df59"
    ):
        parser.error("OpenASL stream cache inputs are invalid")
    selection = json.loads(args.selection.read_text())
    selection_hash = sha256(args.selection)
    records, total = selection_records(selection)
    landmark_receipts_hash = sha256(args.landmark_receipts)
    landmark_rows = read_jsonl(args.landmark_receipts)
    landmarks_by_example = {row["example_id"]: row for row in landmark_rows}
    if (
        len(landmark_rows) != total
        or len(landmarks_by_example) != total
        or any(
            row.get("schema") != "umi-studio-openasl-landmarks/1"
            or row.get("selection_sha256") != selection_hash
            or row.get("spatial_policy") != args.spatial_policy
            for row in landmark_rows
        )
    ):
        raise ValueError("OpenASL landmark receipts are incomplete")
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.index.parent.mkdir(parents=True, exist_ok=True)
    completed = checked_index(
        args.index,
        args.cache_dir,
        selection_hash,
        landmark_receipts_hash,
        args.spatial_policy,
    )
    baseline = args.baseline.resolve()
    for key, value in {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HOME": str(baseline / "run/hf"),
        "TORCH_HOME": str(baseline / "run/torch"),
        "MPLCONFIGDIR": str(baseline / "run/matplotlib"),
        "SHUBERT_DINOV2_SOURCE": str(baseline / "vendor/dinov2-source"),
        "OMP_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
    }.items():
        os.environ[key] = value
    sys.path.insert(0, str(baseline))
    import torch
    from runtime import SHuBERTInferenceRuntime, _rgb_crop_frames
    from sequential_video_reader import make_sequential_reader

    torch.set_num_threads(2)
    model = SHuBERTInferenceRuntime(baseline, device=args.device)
    model._video_reader = make_sequential_reader(model._video_reader)
    identity_hash = sha256(baseline / "MODEL_IDENTITY.json")
    runtime_hash = sha256(baseline / "runtime.py")
    new_count = 0
    for record in records:
        receipt = landmarks_by_example[record["example_id"]]
        identifier = receipt["clip_id"]
        if identifier in completed:
            continue
        if new_count >= args.limit:
            break
        if selection.get("schema") == "umi-studio-openasl-scale-selection/1":
            checked_storage()
        active_started = time.perf_counter()
        clip = args.clip_dir / f"{identifier}.mp4"
        landmark_path = args.landmark_dir / f"{identifier}.npz"
        if (
            not clip.is_file()
            or not landmark_path.is_file()
            or sha256(clip) != receipt["clip_sha256"]
            or sha256(landmark_path) != receipt["landmarks_sha256"]
        ):
            raise ValueError(f"OpenASL clip or landmarks changed: {identifier}")
        reader = model._video_reader(clip)
        frames = reader.get_batch(range(len(reader))).asnumpy()
        landmarks, metadata = unpack_landmarks(landmark_path)
        if len(landmarks) != len(reader) or metadata["source_sha256"] != sha256(clip):
            raise ValueError(f"OpenASL landmark frame count differs: {identifier}")
        source_frames = len(reader)
        frames = np.ascontiguousarray(frames[::2])
        landmarks = temporal_subsample_landmarks(landmarks, 2)
        left_frames, right_frames = model._hand_extractor.extract_hand_frames(
            frames, landmarks
        )
        face_frames = model._face_extractor.extract_face_frames(frames, landmarks)
        pose = model._process_pose_landmarks(landmarks).astype(np.float32)
        left = model._hand_model.extract_embeddings_from_frames(
            _rgb_crop_frames(left_frames)
        ).astype(np.float32)
        right = model._hand_model.extract_embeddings_from_frames(
            _rgb_crop_frames(right_frames)
        ).astype(np.float32)
        face = model._face_model.extract_embeddings_from_frames(
            _rgb_crop_frames(face_frames)
        ).astype(np.float32)
        streams = checked_stream_features(
            {
                "face_features": face,
                "left_hand_features": left,
                "right_hand_features": right,
                "pose_features": pose,
            },
            len(frames),
        )
        with torch.inference_mode():
            hidden = (
                model._translator.encoder(
                    **{
                        name: torch.from_numpy(value).unsqueeze(0).to(model.device)
                        for name, value in streams.items()
                    },
                    return_dict=True,
                )
                .last_hidden_state[0]
                .cpu()
                .numpy()
            )
        if hidden.shape != (len(frames), 1536) or hidden.dtype != np.float32:
            raise ValueError(f"OpenASL encoder state differs: {identifier}")
        cache = args.cache_dir / f"{identifier}.npz"
        temporary = cache.with_suffix(".partial.npz")
        if cache.exists() or temporary.exists():
            raise ValueError(f"unindexed OpenASL stream cache exists: {cache}")
        try:
            with temporary.open("xb") as output:
                np.savez(output, encoder_hidden_state=hidden, **streams)
                output.flush()
                os.fsync(output.fileno())
            temporary.rename(cache)
        finally:
            temporary.unlink(missing_ok=True)
        if selection.get("schema") == "umi-studio-openasl-scale-selection/1":
            checked_storage(reserve_item=False)
        row = {
            "schema": "umi-studio-openasl-stream-cache/1",
            "clip_id": identifier,
            "example_id": record["example_id"],
            "youtube_id": record["youtube_id"],
            "source_group": record["source_group"],
            "split": "training",
            "reference": record["reference"],
            "selection_sha256": selection_hash,
            "landmark_receipts_sha256": landmark_receipts_hash,
            "clip_sha256": receipt["clip_sha256"],
            "landmarks_sha256": receipt["landmarks_sha256"],
            "encoder_cache_sha256": sha256(cache),
            "encoder_shape": list(hidden.shape),
            "encoder_dtype": "float32",
            "stream_feature_shapes": {
                name: list(value.shape) for name, value in streams.items()
            },
            "stream_feature_dtype": "float32",
            "source_frames": source_frames,
            "frames": len(frames),
            "temporal_stride": 2,
            "clip_fps": 25,
            "effective_fps": 12.5,
            "spatial_policy": receipt["spatial_policy"],
            "baseline_identity_sha256": identity_hash,
            "baseline_runtime_sha256": runtime_hash,
            "miner_model_revision": None,
            "miner_health_policy": "user_authorized_maintenance",
            "device": args.device,
            "mps_idle_fraction": args.mps_idle_fraction,
        }
        with args.index.open("a", encoding="utf-8") as output:
            output.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
            output.flush()
            os.fsync(output.fileno())
        completed.add(identifier)
        new_count += 1
        print(f"cached {len(completed)}/{total} {identifier}", flush=True)
        active_seconds = time.perf_counter() - active_started
        del frames, landmarks, streams, hidden
        gc.collect()
        if args.device == "mps":
            torch.mps.empty_cache()
            time.sleep(
                active_seconds
                * args.mps_idle_fraction
                / (1 - args.mps_idle_fraction)
            )


if __name__ == "__main__":
    main()
