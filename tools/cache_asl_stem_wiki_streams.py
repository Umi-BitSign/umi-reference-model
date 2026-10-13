"""Cache stride-2 C5 visual streams for the frozen ASL STEM training selection."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
from benchmark_fleurs_dev import sha256
from cache_fleurs_encoder import checked_stream_features
from extract_mediapipe_landmarks import unpack_landmarks

ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def temporal_subsample_landmarks(landmarks: dict[int, object], stride: int) -> dict[int, object]:
    if stride < 1 or not landmarks or set(landmarks) != set(range(len(landmarks))):
        raise ValueError("landmark sequence or temporal stride is invalid")
    return {
        output_index: landmarks[source_index]
        for output_index, source_index in enumerate(range(0, len(landmarks), stride))
    }


def checked_index(
    path: Path,
    cache_dir: Path,
    selection_hash: str,
    landmark_receipts_hash: str,
) -> set[str]:
    completed = set()
    for row in read_jsonl(path):
        example_id = row.get("example_id")
        cache = cache_dir / f"{example_id}.npz"
        if (
            row.get("schema") != "umi-studio-asl-stem-wiki-stream-cache/1"
            or row.get("selection_sha256") != selection_hash
            or row.get("landmark_receipts_sha256") != landmark_receipts_hash
            or row.get("temporal_stride") != 2
            or row.get("frames") != (row.get("source_frames", 0) + 1) // 2
            or example_id in completed
            or not cache.is_file()
            or row.get("encoder_cache_sha256") != sha256(cache)
        ):
            raise ValueError(f"ASL STEM stream cache differs: {example_id}")
        completed.add(example_id)
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
    parser.add_argument("--limit", type=int, default=32)
    args = parser.parse_args()
    if not 1 <= args.limit <= 1024:
        parser.error("limit must be in [1, 1024]")
    if (
        not args.selection.is_file()
        or not args.landmark_receipts.is_file()
        or not args.clip_dir.is_dir()
        or not args.landmark_dir.is_dir()
        or args.baseline.resolve() != ROOT / "artifacts/baseline-6fe8df59"
    ):
        parser.error("ASL STEM stream cache inputs are invalid")
    selection = json.loads(args.selection.read_text())
    selection_hash = sha256(args.selection)
    if (
        selection.get("schema") != "umi-studio-asl-stem-wiki-training-selection/1"
        or selection.get("training_only") is not True
        or selection.get("selected_examples") != 1024
        or len(selection.get("records", [])) != 1024
    ):
        raise ValueError("ASL STEM training selection scope differs")
    landmark_receipts_hash = sha256(args.landmark_receipts)
    landmark_rows = read_jsonl(args.landmark_receipts)
    landmarks_by_id = {row["example_id"]: row for row in landmark_rows}
    if (
        len(landmark_rows) != 1024
        or len(landmarks_by_id) != 1024
        or any(
            row.get("schema") != "umi-studio-asl-stem-wiki-landmarks/1"
            or row.get("selection_sha256") != selection_hash
            for row in landmark_rows
        )
    ):
        raise ValueError("ASL STEM landmark receipts are incomplete")
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.index.parent.mkdir(parents=True, exist_ok=True)
    completed = checked_index(
        args.index,
        args.cache_dir,
        selection_hash,
        landmark_receipts_hash,
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
    model = SHuBERTInferenceRuntime(baseline, device="cpu")
    model._video_reader = make_sequential_reader(model._video_reader)
    identity_hash = sha256(baseline / "MODEL_IDENTITY.json")
    runtime_hash = sha256(baseline / "runtime.py")
    new_count = 0
    for record in selection["records"]:
        example_id = record["example_id"]
        if example_id in completed:
            continue
        if new_count >= args.limit:
            break
        receipt = landmarks_by_id[example_id]
        clip = args.clip_dir / f"{example_id}.mp4"
        landmark_path = args.landmark_dir / f"{example_id}.npz"
        if (
            not clip.is_file()
            or not landmark_path.is_file()
            or sha256(clip) != receipt["clip_sha256"]
            or sha256(landmark_path) != receipt["landmarks_sha256"]
        ):
            raise ValueError(f"ASL STEM clip or landmarks changed: {example_id}")
        reader = model._video_reader(clip)
        frames = reader.get_batch(range(len(reader))).asnumpy()
        landmarks, metadata = unpack_landmarks(landmark_path)
        if len(landmarks) != len(reader) or metadata["source_sha256"] != sha256(clip):
            raise ValueError(f"ASL STEM landmark frame count differs: {example_id}")
        source_frames = len(reader)
        frames = np.ascontiguousarray(frames[::2])
        landmarks = temporal_subsample_landmarks(landmarks, 2)
        left_frames, right_frames = model._hand_extractor.extract_hand_frames(frames, landmarks)
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
                        name: torch.from_numpy(value).unsqueeze(0)
                        for name, value in streams.items()
                    },
                    return_dict=True,
                )
                .last_hidden_state[0]
                .cpu()
                .numpy()
            )
        if hidden.shape != (len(frames), 1536) or hidden.dtype != np.float32:
            raise ValueError(f"ASL STEM encoder state differs: {example_id}")
        cache = args.cache_dir / f"{example_id}.npz"
        temporary = cache.with_suffix(".partial.npz")
        if cache.exists() or temporary.exists():
            raise ValueError(f"unindexed ASL STEM stream cache exists: {cache}")
        try:
            with temporary.open("xb") as output:
                np.savez(output, encoder_hidden_state=hidden, **streams)
                output.flush()
                os.fsync(output.fileno())
            temporary.rename(cache)
        finally:
            temporary.unlink(missing_ok=True)
        row = {
            "schema": "umi-studio-asl-stem-wiki-stream-cache/1",
            "example_id": example_id,
            "participant": record["participant"],
            "article": record["article"],
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
            "baseline_identity_sha256": identity_hash,
            "baseline_runtime_sha256": runtime_hash,
            "miner_model_revision": None,
            "miner_health_policy": "user_authorized_maintenance",
        }
        with args.index.open("a", encoding="utf-8") as output:
            output.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
            output.flush()
            os.fsync(output.fileno())
        completed.add(example_id)
        new_count += 1
        print(f"cached {len(completed)}/1024 {example_id}", flush=True)
        del frames, landmarks, streams, hidden
        gc.collect()


if __name__ == "__main__":
    main()
