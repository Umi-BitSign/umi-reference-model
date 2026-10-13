"""Cache stride-2 C5 streams for the verified pre-cut How2Sign expansion."""

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

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "umi-studio-how2sign-precut-expansion-stream-cache/1"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def checked_receipted_prefix(selection: dict, receipts: list[dict], selection_hash: str) -> list[dict]:
    records = selection.get("records", [])
    names = [row.get("sentence_name") for row in records]
    received_names = [row.get("sentence_name") for row in receipts]
    if (
        selection.get("schema") != "umi-studio-how2sign-precut-expansion/1"
        or selection.get("training_only") is not True
        or selection.get("clips") != len(records)
        or len(names) != len(set(names))
        or not receipts
        or received_names != names[: len(receipts)]
        or any(
            receipt.get("schema")
            != "umi-studio-how2sign-precut-expansion-clip/1"
            or receipt.get("selection_sha256") != selection_hash
            or receipt.get("archive_member") != record.get("archive_member")
            or receipt.get("archive_crc32") != record.get("archive_crc32")
            or receipt.get("clip_size_bytes") != record.get("archive_file_size")
            or len(receipt.get("clip_sha256", "")) != 64
            for record, receipt in zip(records, receipts, strict=False)
        )
    ):
        raise ValueError("pre-cut expansion receipts are not a verified selection prefix")
    return records[: len(receipts)]


def checked_index(
    path: Path,
    cache_dir: Path,
    selection_hash: str,
    receipts_hash: str,
    landmark_receipts_hash: str,
) -> set[str]:
    completed = set()
    if not path.exists():
        return completed
    for row in read_jsonl(path):
        name = row.get("sentence_name")
        cache = cache_dir / f"{name}.npz"
        if (
            row.get("schema") != SCHEMA
            or row.get("selection_sha256") != selection_hash
            or row.get("clip_receipts_sha256") != receipts_hash
            or row.get("landmark_receipts_sha256") != landmark_receipts_hash
            or row.get("temporal_stride") != 2
            or row.get("frames") != (row.get("source_frames", 0) + 1) // 2
            or name in completed
            or not cache.is_file()
            or row.get("encoder_cache_sha256") != sha256(cache)
        ):
            raise ValueError(f"pre-cut expansion stream cache differs: {name}")
        completed.add(name)
    return completed


def rebind_prior_index(
    *,
    prior_index: Path,
    prior_landmark_receipts: Path,
    index: Path,
    cache_dir: Path,
    records: list[dict],
    landmark_rows: list[dict],
    selection_hash: str,
    receipts_hash: str,
    landmark_receipts_hash: str,
) -> None:
    """Reuse a validated pilot cache under the completed receipt-set hash."""
    if index.exists():
        return
    prior_landmark_rows = read_jsonl(prior_landmark_receipts)
    prior_landmark_hash = sha256(prior_landmark_receipts)
    prior_rows = read_jsonl(prior_index)
    count = len(prior_rows)
    expected_names = [row["sentence_name"] for row in records[:count]]
    if (
        not prior_rows
        or count > len(landmark_rows)
        or [row.get("sentence_name") for row in prior_rows] != expected_names
        or prior_landmark_rows != landmark_rows[: len(prior_landmark_rows)]
        or len(prior_landmark_rows) != count
        or checked_index(
            prior_index,
            cache_dir,
            selection_hash,
            receipts_hash,
            prior_landmark_hash,
        )
        != set(expected_names)
    ):
        raise ValueError("prior stream cache is not a verified landmark prefix")
    rebound = []
    for row, landmark_row in zip(prior_rows, prior_landmark_rows, strict=True):
        if row.get("landmarks_sha256") != landmark_row.get("landmarks_sha256"):
            raise ValueError("prior stream cache landmark hash differs")
        rebound.append({**row, "landmark_receipts_sha256": landmark_receipts_hash})
    index.parent.mkdir(parents=True, exist_ok=True)
    with index.open("x", encoding="utf-8") as output:
        for row in rebound:
            output.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
        output.flush()
        os.fsync(output.fileno())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--clip-receipts", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--clip-dir", type=Path, required=True)
    parser.add_argument("--landmark-dir", type=Path, required=True)
    parser.add_argument("--landmark-receipts", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--prior-index", type=Path)
    parser.add_argument("--prior-landmark-receipts", type=Path)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--mps-idle-fraction", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--miner-maintenance-mode", action="store_true")
    args = parser.parse_args()
    if not args.miner_maintenance_mode:
        parser.error("pre-cut expansion caching requires miner maintenance mode")
    if not 1 <= args.limit <= 1650:
        parser.error("limit must be in [1, 1650]")
    if args.device == "mps" and args.mps_idle_fraction != 0.5:
        parser.error("MPS extraction must reserve half of average GPU capacity")
    if args.device == "cpu" and args.mps_idle_fraction != 0:
        parser.error("CPU extraction requires --mps-idle-fraction 0")
    if bool(args.prior_index) != bool(args.prior_landmark_receipts):
        parser.error("prior index and landmark receipts must be provided together")
    if (
        not args.selection.is_file()
        or not args.clip_receipts.is_file()
        or not args.landmark_receipts.is_file()
        or not args.source_dir.is_dir()
        or not args.clip_dir.is_dir()
        or not args.landmark_dir.is_dir()
        or args.baseline.resolve() != ROOT / "artifacts/baseline-6fe8df59"
    ):
        parser.error("pre-cut expansion cache inputs are invalid")
    selection_hash = sha256(args.selection)
    receipts_hash = sha256(args.clip_receipts)
    landmark_receipts_hash = sha256(args.landmark_receipts)
    selection = json.loads(args.selection.read_text())
    receipts = read_jsonl(args.clip_receipts)
    records = checked_receipted_prefix(selection, receipts, selection_hash)
    receipts_by_name = {row["sentence_name"]: row for row in receipts}
    landmark_rows = read_jsonl(args.landmark_receipts)
    landmark_names = [row.get("sentence_name") for row in landmark_rows]
    if (
        not landmark_rows
        or landmark_names
        != [row["sentence_name"] for row in records[: len(landmark_rows)]]
        or any(
            row.get("schema")
            != "umi-studio-how2sign-precut-expansion-landmarks/1"
            or row.get("selection_sha256") != selection_hash
            or row.get("source_receipts_sha256") != receipts_hash
            for row in landmark_rows
        )
    ):
        raise ValueError("pre-cut expansion landmark receipts are incomplete")
    records = records[: len(landmark_rows)]
    landmarks_by_name = {row["sentence_name"]: row for row in landmark_rows}
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.index.parent.mkdir(parents=True, exist_ok=True)
    if args.prior_index:
        rebind_prior_index(
            prior_index=args.prior_index,
            prior_landmark_receipts=args.prior_landmark_receipts,
            index=args.index,
            cache_dir=args.cache_dir,
            records=records,
            landmark_rows=landmark_rows,
            selection_hash=selection_hash,
            receipts_hash=receipts_hash,
            landmark_receipts_hash=landmark_receipts_hash,
        )
    completed = checked_index(
        args.index,
        args.cache_dir,
        selection_hash,
        receipts_hash,
        landmark_receipts_hash,
    )
    if not completed <= set(receipts_by_name):
        raise ValueError("stream index contains an unreceipted clip")

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
        name = record["sentence_name"]
        if name in completed:
            continue
        if new_count >= args.limit:
            break
        active_started = time.perf_counter()
        receipt = receipts_by_name[name]
        source = args.source_dir / f"{name}.mp4"
        clip = args.clip_dir / f"{name}.mp4"
        landmark_path = args.landmark_dir / f"{name}.npz"
        landmark_receipt = landmarks_by_name[name]
        if (
            not source.is_file()
            or source.stat().st_size != receipt["clip_size_bytes"]
            or sha256(source) != receipt["clip_sha256"]
            or not clip.is_file()
            or not landmark_path.is_file()
            or sha256(clip) != landmark_receipt["clip_sha256"]
            or sha256(landmark_path) != landmark_receipt["landmarks_sha256"]
        ):
            raise ValueError(f"pre-cut source clip changed: {name}")
        reader = model._video_reader(clip)
        frames = reader.get_batch(range(len(reader))).asnumpy()
        landmarks, landmark_metadata = unpack_landmarks(landmark_path)
        if (
            len(landmarks) != len(reader)
            or landmark_metadata.get("source_sha256")
            != landmark_receipt["clip_sha256"]
        ):
            raise ValueError(f"pre-cut landmark frame count differs: {name}")
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
                        key: torch.from_numpy(value).unsqueeze(0).to(model.device)
                        for key, value in streams.items()
                    },
                    return_dict=True,
                )
                .last_hidden_state[0]
                .cpu()
                .numpy()
            )
        if hidden.shape != (len(frames), 1536) or hidden.dtype != np.float32:
            raise ValueError(f"pre-cut encoder state differs: {name}")
        cache = args.cache_dir / f"{name}.npz"
        temporary = cache.with_suffix(".partial.npz")
        if cache.exists() or temporary.exists():
            raise ValueError(f"unindexed pre-cut stream cache exists: {cache}")
        try:
            with temporary.open("xb") as output:
                np.savez(output, encoder_hidden_state=hidden, **streams)
                output.flush()
                os.fsync(output.fileno())
            temporary.rename(cache)
        finally:
            temporary.unlink(missing_ok=True)
        row = {
            "schema": SCHEMA,
            "sentence_name": name,
            "video_id": record["video_id"],
            "source_group": f"how2sign:{record['video_id']}",
            "signer": record["signer"],
            "split": "training",
            "reference": record["reference"],
            "selection_sha256": selection_hash,
            "clip_receipts_sha256": receipts_hash,
            "landmark_receipts_sha256": landmark_receipts_hash,
            "source_clip_sha256": receipt["clip_sha256"],
            "resampled_clip_sha256": sha256(clip),
            "landmarks_sha256": landmark_receipt["landmarks_sha256"],
            "encoder_cache_sha256": sha256(cache),
            "encoder_shape": list(hidden.shape),
            "encoder_dtype": "float32",
            "stream_feature_shapes": {
                key: list(value.shape) for key, value in streams.items()
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
            "device": args.device,
            "mps_idle_fraction": args.mps_idle_fraction,
        }
        with args.index.open("a", encoding="utf-8") as output:
            output.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
            output.flush()
            os.fsync(output.fileno())
        completed.add(name)
        new_count += 1
        print(f"cached {len(completed)}/{len(records)} {name}", flush=True)
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
