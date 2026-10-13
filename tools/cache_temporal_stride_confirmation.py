"""Build frozen C5 visual-stream caches from receipted confirmation landmarks."""

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
from prepare_temporal_stride_confirmation import checked_completed, read_jsonl

ROOT = Path(__file__).resolve().parents[1]


def confirmation_references(
    plan: dict, how2sign_selection: dict, fleurs_selection: dict
) -> dict[str, dict]:
    h_rows = {row["sentence_name"]: row for row in how2sign_selection["rows"]}
    f_rows = {row["example_id"]: row for row in fleurs_selection["validation"]}
    result = {}
    for record in plan["records"]:
        example_id = record["example_id"]
        source = h_rows if record["domain"] == "how2sign" else f_rows
        if example_id in result or example_id not in source:
            raise ValueError("confirmation references differ from the frozen plan")
        row = source[example_id]
        if row["signer"] != record["signer"] or not row.get("reference"):
            raise ValueError("confirmation reference metadata differs")
        result[example_id] = row
    if len(result) != 88:
        raise ValueError("confirmation reference scope differs")
    return result


def verified_indexes(
    paths: list[Path], cache_dir: Path, plan_hash: str, landmark_receipts_hash: str
) -> dict[str, dict]:
    completed = {}
    for path in paths:
        for row in read_jsonl(path):
            example_id = row.get("sentence_name", row.get("example_id"))
            cache = cache_dir / f"{example_id}.npz"
            if (
                row.get("confirmation_plan_sha256") != plan_hash
                or row.get("landmark_receipts_sha256") != landmark_receipts_hash
                or example_id in completed
                or not cache.is_file()
                or row.get("encoder_cache_sha256") != sha256(cache)
                or row.get("stream_feature_dtype") != "float32"
            ):
                raise ValueError(f"confirmation stream cache differs: {example_id}")
            completed[example_id] = row
    return completed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--landmark-receipts", type=Path, required=True)
    parser.add_argument("--clip-dir", type=Path, required=True)
    parser.add_argument("--landmark-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--how2sign-index", type=Path, required=True)
    parser.add_argument("--fleurs-index", type=Path, required=True)
    parser.add_argument("--how2sign-selection", type=Path, required=True)
    parser.add_argument("--fleurs-selection", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--limit", type=int, default=16)
    args = parser.parse_args()
    if not 1 <= args.limit <= 88:
        parser.error("limit must be in [1, 88]")
    required = (
        args.plan,
        args.landmark_receipts,
        args.how2sign_selection,
        args.fleurs_selection,
    )
    if not all(path.is_file() for path in required) or not args.baseline.is_dir():
        parser.error("confirmation cache inputs are missing")
    plan = json.loads(args.plan.read_text())
    if (
        plan.get("schema") != "umi-studio-temporal-stride-confirmation-plan/1"
        or len(plan.get("records", [])) != 88
    ):
        raise ValueError("confirmation plan scope differs")
    plan_hash = sha256(args.plan)
    landmark_receipts_hash = sha256(args.landmark_receipts)
    landmark_receipts = {
        row["example_id"]: row for row in read_jsonl(args.landmark_receipts)
    }
    docker_image = next(iter(landmark_receipts.values()), {}).get("docker_image_id")
    prepared = checked_completed(
        args.landmark_receipts,
        plan_hash=plan_hash,
        docker_image=docker_image,
        clip_dir=args.clip_dir,
        landmark_dir=args.landmark_dir,
    )
    if len(prepared) != 88:
        raise ValueError("all confirmation landmarks must be frozen before stream caching")
    how2sign_selection = json.loads(args.how2sign_selection.read_text())
    fleurs_selection = json.loads(args.fleurs_selection.read_text())
    if (
        sha256(args.how2sign_selection)
        != plan["source_hashes"]["how2sign_selection"]
        or sha256(args.fleurs_selection) != plan["source_hashes"]["fleurs_selection"]
    ):
        raise ValueError("confirmation selections differ from the plan")
    references = confirmation_references(plan, how2sign_selection, fleurs_selection)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.how2sign_index.parent.mkdir(parents=True, exist_ok=True)
    args.fleurs_index.parent.mkdir(parents=True, exist_ok=True)
    completed = verified_indexes(
        [args.how2sign_index, args.fleurs_index],
        args.cache_dir,
        plan_hash,
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
    model = SHuBERTInferenceRuntime(baseline, device=args.device)
    model._video_reader = make_sequential_reader(model._video_reader)
    identity_hash = sha256(baseline / "MODEL_IDENTITY.json")
    runtime_hash = sha256(baseline / "runtime.py")
    new_count = 0
    for record in plan["records"]:
        example_id = record["example_id"]
        if example_id in completed:
            continue
        if new_count >= args.limit:
            break
        clip = args.clip_dir / f"{example_id}.mp4"
        landmark_path = args.landmark_dir / f"{example_id}.npz"
        landmark_receipt = landmark_receipts[example_id]
        if (
            sha256(clip) != landmark_receipt["clip_sha256"]
            or sha256(landmark_path) != landmark_receipt["landmarks_sha256"]
        ):
            raise ValueError(f"confirmation clip or landmarks changed: {example_id}")
        reader = model._video_reader(clip)
        frames = reader.get_batch(range(len(reader))).asnumpy()
        landmarks, metadata = unpack_landmarks(landmark_path)
        if len(landmarks) != len(reader) or metadata["source_sha256"] != sha256(clip):
            raise ValueError(f"confirmation landmark frame count differs: {example_id}")
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
            len(reader),
        )
        with torch.inference_mode():
            hidden = (
                model._translator.encoder(
                    **{
                        name: torch.from_numpy(value).unsqueeze(0).to(args.device)
                        for name, value in streams.items()
                    },
                    return_dict=True,
                )
                .last_hidden_state[0]
                .cpu()
                .numpy()
            )
        if hidden.shape != (len(reader), 1536) or hidden.dtype != np.float32:
            raise ValueError(f"confirmation encoder state differs: {example_id}")
        cache = args.cache_dir / f"{example_id}.npz"
        temporary = cache.with_suffix(".partial.npz")
        if cache.exists() or temporary.exists():
            raise ValueError(f"unindexed confirmation cache exists: {cache}")
        try:
            with temporary.open("xb") as output:
                np.savez(output, encoder_hidden_state=hidden, **streams)
                output.flush()
                os.fsync(output.fileno())
            temporary.rename(cache)
        finally:
            temporary.unlink(missing_ok=True)
        reference_row = references[example_id]
        common = {
            "signer": record["signer"],
            "selection_sha256": (
                plan["source_hashes"]["how2sign_selection"]
                if record["domain"] == "how2sign"
                else plan["source_hashes"]["fleurs_selection"]
            ),
            "encoder_cache_sha256": sha256(cache),
            "encoder_shape": list(hidden.shape),
            "encoder_dtype": "float32",
            "stream_feature_shapes": {
                name: list(value.shape) for name, value in streams.items()
            },
            "stream_feature_dtype": "float32",
            "clip_fps": 25,
            "reference": reference_row["reference"],
            "confirmation_plan_sha256": plan_hash,
            "landmark_receipts_sha256": landmark_receipts_hash,
            "container_landmarks_sha256": landmark_receipt["landmarks_sha256"],
            "baseline_identity_sha256": identity_hash,
            "baseline_runtime_sha256": runtime_hash,
            "miner_model_revision": None,
            "miner_health_policy": "user_authorized_maintenance",
        }
        if record["domain"] == "how2sign":
            index_path = args.how2sign_index
            index_row = {
                **common,
                "schema": "umi-studio-how2sign-encoder-cache-example/1",
                "sentence_name": example_id,
                "split": "validation",
            }
        else:
            index_path = args.fleurs_index
            index_row = {
                **common,
                "schema": "umi-studio-fleurs-dev-encoder-cache-example/1",
                "example_id": example_id,
                "document_id": record["document_id"],
                "split": "confirmation",
            }
        with index_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(index_row, sort_keys=True, ensure_ascii=False) + "\n")
            output.flush()
            os.fsync(output.fileno())
        completed[example_id] = index_row
        new_count += 1
        print(f"cached {len(completed)}/88 {record['domain']} {example_id}", flush=True)
        del frames, landmarks, streams, hidden
        gc.collect()


if __name__ == "__main__":
    main()
