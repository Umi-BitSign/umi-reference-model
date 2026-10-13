"""Cache frozen SHuBERT encoder states for the public FLEURS development split.

Use the independent baseline environment, not the live miner environment. Each
run is bounded by --limit and checks miner health between expensive stages.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from benchmark_fleurs_dev import extract_clip, miner_revision, sha256


def _selected_rows(selection: dict, partition: dict | None, split: str) -> list[dict]:
    if split == "training":
        return selection["training"]
    if partition is None:
        raise ValueError("validation cache requires a frozen evaluation partition")
    tuning = set(partition["tuning_documents"])
    rows = [
        row
        for row in selection["validation"]
        if (row["document_id"] in tuning) == (split == "tuning")
    ]
    expected = partition[f"{split}_examples"]
    if len(rows) != expected:
        raise ValueError(f"{split} count differs from frozen partition")
    return rows


def checked_stream_features(values: dict[str, object], frames: int) -> dict[str, np.ndarray]:
    dimensions = {
        "face_features": 384,
        "left_hand_features": 384,
        "right_hand_features": 384,
        "pose_features": 14,
    }
    if set(values) != set(dimensions):
        raise ValueError("video stream features are missing or duplicated")
    streams = {}
    for name, width in dimensions.items():
        array = np.asarray(values[name], dtype=np.float32)
        if array.shape != (frames, width) or not np.isfinite(array).all():
            raise ValueError(f"video stream feature has an invalid shape or value: {name}")
        streams[name] = array.copy()
    return streams


def _encoder_state(
    model,
    clip: Path,
    health_url: str | None,
    revision: str | None,
    *,
    return_streams: bool = False,
) -> tuple[np.ndarray, dict] | tuple[np.ndarray, dict, dict[str, np.ndarray]]:
    import torch
    from runtime import _rgb_crop_frames

    timings = {}

    def timed(name: str, operation):
        started = time.perf_counter()
        result = operation()
        timings[name] = time.perf_counter() - started
        return result

    def check_miner() -> None:
        if health_url is not None:
            miner_revision(health_url, revision)

    reader = timed("video_open", lambda: model._video_reader(clip))
    frames = timed("video_decode", lambda: reader.get_batch(range(len(reader))).asnumpy())
    check_miner()
    landmarks = timed(
        "mediapipe_landmarks",
        lambda: model._video_holistic(
            frames,
            str(model.models / "face_landmarker_v2_with_blendshapes.task"),
            str(model.models / "hand_landmarker.task"),
        ),
    )
    check_miner()
    left_frames, right_frames = timed(
        "hand_crops", lambda: model._hand_extractor.extract_hand_frames(frames, landmarks)
    )
    face_frames = timed(
        "face_crops", lambda: model._face_extractor.extract_face_frames(frames, landmarks)
    )
    pose = timed("body_pose", lambda: model._process_pose_landmarks(landmarks))
    left = timed(
        "dino_left_hand",
        lambda: model._hand_model.extract_embeddings_from_frames(_rgb_crop_frames(left_frames)),
    )
    check_miner()
    right = timed(
        "dino_right_hand",
        lambda: model._hand_model.extract_embeddings_from_frames(_rgb_crop_frames(right_frames)),
    )
    face = timed(
        "dino_face",
        lambda: model._face_model.extract_embeddings_from_frames(_rgb_crop_frames(face_frames)),
    )
    check_miner()
    with torch.no_grad():
        face_tensor = torch.tensor(face, dtype=torch.float32).unsqueeze(0).to(model.device)
        left_tensor = torch.tensor(left, dtype=torch.float32).unsqueeze(0).to(model.device)
        right_tensor = torch.tensor(right, dtype=torch.float32).unsqueeze(0).to(model.device)
        pose_tensor = torch.tensor(pose, dtype=torch.float32).unsqueeze(0).to(model.device)
        hidden = timed(
            "frozen_encoder",
            lambda: (
                model._translator.encoder(
                    face_features=face_tensor,
                    left_hand_features=left_tensor,
                    right_hand_features=right_tensor,
                    pose_features=pose_tensor,
                    return_dict=True,
                ).last_hidden_state
            ),
        )
        state = hidden[0].detach().cpu().numpy().copy()
    if state.ndim != 2 or state.shape[0] != len(reader) or state.shape[1] != 1536:
        raise RuntimeError(f"unexpected encoder state shape: {state.shape}")
    if state.dtype != np.float32 or not np.isfinite(state).all():
        raise RuntimeError("frozen encoder returned non-finite or non-float32 state")
    diagnostics = {"frames": len(reader), "timings_seconds": timings}
    if return_streams:
        streams = checked_stream_features(
            {
                "face_features": face,
                "left_hand_features": left,
                "right_hand_features": right,
                "pose_features": pose,
            },
            len(reader),
        )
        return state, diagnostics, streams
    return state, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--evaluation-partition", type=Path)
    parser.add_argument("--source-receipt", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--clip-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--miner-health-url", required=True)
    parser.add_argument("--split", choices=("training", "tuning", "confirmation"), required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--sequential-decode", action="store_true")
    parser.add_argument("--save-stream-features", action="store_true")
    parser.add_argument("--example-ids-file", type=Path)
    parser.add_argument("--limit", type=int, default=1, help="maximum new examples")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("limit must be positive")
    if not args.source_dir.is_dir() or not args.baseline.is_dir():
        parser.error("source directory and independent baseline must exist")
    selection_hash = sha256(args.selection)
    selection = json.loads(args.selection.read_bytes())
    if selection.get("schema") != "umi-studio-fleurs-dev-split/1":
        raise ValueError("unexpected development selection schema")
    partition = None
    if args.split != "training":
        if args.evaluation_partition is None:
            parser.error("tuning/confirmation requires --evaluation-partition")
        partition = json.loads(args.evaluation_partition.read_bytes())
        if (
            partition.get("schema") != "umi-studio-fleurs-dev-evaluation-partition/1"
            or partition.get("selection_sha256") != selection_hash
        ):
            raise ValueError("evaluation partition differs from selection")
    rows = _selected_rows(selection, partition, args.split)
    if args.example_ids_file is not None:
        if args.split != "training":
            parser.error("explicit example selection is only allowed for training")
        requested = json.loads(args.example_ids_file.read_bytes())
        identifiers = requested.get("requested_example_ids")
        if (
            requested.get("schema") != "umi-studio-fleurs-training-pilot/1"
            or requested.get("selection_sha256") != selection_hash
            or not isinstance(identifiers, list)
            or not identifiers
            or any(not isinstance(identifier, str) for identifier in identifiers)
            or len(identifiers) != len(set(identifiers))
        ):
            raise ValueError("explicit training selection is invalid")
        by_id = {row["example_id"]: row for row in rows}
        if not set(identifiers) <= by_id.keys():
            raise ValueError("explicit selection contains a non-training example")
        rows = [by_id[example_id] for example_id in identifiers]
    receipt = json.loads(args.source_receipt.read_bytes())
    receipt_split = "training" if args.split == "training" else "validation"
    if (
        receipt.get("schema") != "umi-studio-fleurs-dev-source-files/1"
        or receipt.get("split") != receipt_split
        or receipt.get("selection_sha256") != selection_hash
    ):
        raise ValueError("source receipt differs from selection or requested split")
    source_records = {row["path"]: row for row in receipt["files"]}
    if len(source_records) != len(receipt["files"]):
        raise ValueError("source receipt contains duplicate files")
    completed = {}
    if args.index.exists():
        for line in args.index.read_text().splitlines():
            record = json.loads(line)
            if (
                record["selection_sha256"] != selection_hash
                or record["split"] != args.split
                or record["clip_fps"] != 25
            ):
                raise ValueError("existing encoder cache index differs from this run")
            if record["example_id"] in completed:
                raise ValueError("duplicate example in encoder cache index")
            cached = args.cache_dir / f"{record['example_id']}.npz"
            if not cached.is_file() or sha256(cached) != record["encoder_cache_sha256"]:
                raise ValueError(f"encoder cache missing or altered: {cached}")
            completed[record["example_id"]] = record

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
    from runtime import SHuBERTInferenceRuntime

    revision = miner_revision(args.miner_health_url, None)
    model = SHuBERTInferenceRuntime(baseline, device=args.device)
    if args.sequential_decode:
        from sequential_video_reader import make_sequential_reader

        model._video_reader = make_sequential_reader(model._video_reader)
    revision = miner_revision(args.miner_health_url, revision)
    baseline_identity_hash = sha256(baseline / "MODEL_IDENTITY.json")
    baseline_runtime_hash = sha256(baseline / "runtime.py")
    args.clip_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.index.parent.mkdir(parents=True, exist_ok=True)
    verified_sources = set()
    new_count = 0
    for row in rows:
        if row["example_id"] in completed:
            continue
        if new_count >= args.limit:
            break
        revision = miner_revision(args.miner_health_url, revision)
        source_name = row["source_video"]
        source = args.source_dir / Path(source_name).name
        source_record = source_records[source_name]
        if source_name not in verified_sources:
            if (
                source.stat().st_size != source_record["size_bytes"]
                or sha256(source) != source_record["sha256"]
            ):
                raise RuntimeError(f"source video no longer matches receipt: {source_name}")
            verified_sources.add(source_name)
        clip = args.clip_dir / f"{row['example_id']}.mp4"
        extract_clip(source, clip, row["start_time_sec"], row["end_time_sec"], "25")
        clip_hash = sha256(clip)
        if args.save_stream_features:
            state, diagnostics, streams = _encoder_state(
                model, clip, args.miner_health_url, revision, return_streams=True
            )
        else:
            state, diagnostics = _encoder_state(model, clip, args.miner_health_url, revision)
            streams = {}
        revision = miner_revision(args.miner_health_url, revision)
        cache = args.cache_dir / f"{row['example_id']}.npz"
        temporary = cache.with_suffix(".partial.npz")
        if cache.exists() or temporary.exists():
            raise RuntimeError(f"unindexed encoder cache must be inspected: {cache}")
        try:
            with temporary.open("xb") as output:
                np.savez(output, encoder_hidden_state=state, **streams)
                output.flush()
                os.fsync(output.fileno())
            temporary.rename(cache)
        finally:
            temporary.unlink(missing_ok=True)
        record = {
            "schema": "umi-studio-fleurs-dev-encoder-cache-example/1",
            "example_id": row["example_id"],
            "document_id": row["document_id"],
            "signer": row["signer"],
            "split": args.split,
            "selection_sha256": selection_hash,
            "source_receipt_sha256": sha256(args.source_receipt),
            "source_video": source_name,
            "source_video_sha256": source_record["sha256"],
            "clip_sha256": clip_hash,
            "clip_fps": 25,
            "encoder_cache_sha256": sha256(cache),
            "encoder_shape": list(state.shape),
            "encoder_dtype": str(state.dtype),
            "video_decode_strategy": (
                "opencv_sequential_full_range"
                if args.sequential_decode
                else "baseline_seek_per_frame"
            ),
            "baseline_identity_sha256": baseline_identity_hash,
            "baseline_runtime_sha256": baseline_runtime_hash,
            "reference": row["reference"],
            "miner_model_revision": revision,
            "diagnostics": diagnostics,
        }
        if streams:
            record["stream_feature_shapes"] = {
                name: list(value.shape) for name, value in streams.items()
            }
            record["stream_feature_dtype"] = "float32"
        with args.index.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
            output.flush()
            os.fsync(output.fileno())
        new_count += 1
        print(f"{len(completed) + new_count}: cached {row['example_id']}", flush=True)
        del state
        gc.collect()


if __name__ == "__main__":
    main()
