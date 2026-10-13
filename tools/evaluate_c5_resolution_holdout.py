"""Evaluate C5 crop resolution on the 134-clip internal How2Sign holdout."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from benchmark_fleurs_dev import distance, normalize, sha256
from evaluate_c5_resolution_tuning import resolution_policy
from extract_mediapipe_landmarks import unpack_landmarks
from resolution_policy import make_short_side_reader, set_visual_crop_size

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "umi-studio-c5-resolution-holdout-example/1"


def load_index(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    seen = set()
    for row in rows:
        name = row.get("sentence_name")
        if (
            row.get("schema")
            != "umi-studio-how2sign-precut-expansion-stream-cache/1"
            or row.get("split") != "training"
            or row.get("temporal_stride") != 2
            or name in seen
        ):
            raise ValueError("internal holdout index differs")
        seen.add(name)
    if len(rows) != 134:
        raise ValueError("internal holdout must contain 134 examples")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--clip-dir", type=Path, required=True)
    parser.add_argument("--landmark-dir", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--source-short-side", type=int)
    parser.add_argument("--visual-crop-size", type=int, default=224)
    parser.add_argument("--calibrator", type=Path)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--mps-idle-fraction", type=float)
    parser.add_argument("--limit", type=int, default=134)
    parser.add_argument("--miner-maintenance-mode", action="store_true")
    args = parser.parse_args()
    idle_fraction = args.mps_idle_fraction
    if idle_fraction is None:
        idle_fraction = 0.5 if args.device == "mps" else 0.0
    if not args.miner_maintenance_mode:
        parser.error("evaluation requires explicit miner maintenance mode")
    if (
        not 1 <= args.limit <= 134
        or not args.index.is_file()
        or not args.bundle.is_dir()
        or not args.clip_dir.is_dir()
        or not args.landmark_dir.is_dir()
        or not args.results.resolve().is_relative_to(ROOT / "runs")
    ):
        parser.error("evaluation paths or bounds are invalid")
    if (args.device == "mps" and idle_fraction != 0.5) or (
        args.device == "cpu" and idle_fraction != 0
    ):
        parser.error("device idle fraction violates the resource policy")
    if args.calibrator is not None and (
        not args.calibrator.is_file() or args.visual_crop_size != 168
    ):
        parser.error("crop calibrator requires an existing 168px artifact")
    try:
        policy = resolution_policy(args.source_short_side, args.visual_crop_size)
    except ValueError as error:
        parser.error(str(error))
    rows = load_index(args.index)
    index_hash = sha256(args.index)
    bundle = args.bundle.resolve()
    identity_hash = sha256(bundle / "MODEL_IDENTITY.json")
    runtime_hash = sha256(bundle / "runtime.py")
    stride_hash = sha256(bundle / "c5_stride2_runtime.py")
    calibrator_hash = sha256(args.calibrator) if args.calibrator is not None else None
    completed = {}
    if args.results.exists():
        for line in args.results.read_text().splitlines():
            record = json.loads(line)
            name = record.get("example_id")
            if (
                record.get("schema") != SCHEMA
                or record.get("evaluation_index_sha256") != index_hash
                or record.get("bundle_identity_sha256") != identity_hash
                or record.get("bundle_runtime_sha256") != runtime_hash
                or record.get("stride_runtime_sha256") != stride_hash
                or record.get("calibrator_sha256") != calibrator_hash
                or any(record.get(key) != value for key, value in policy.items())
                or name in completed
            ):
                raise ValueError("existing holdout results use another frozen input")
            completed[name] = record

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
    from c5_stride2_runtime import install_post_mediapipe_stride
    from runtime import SHuBERTInferenceRuntime

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    model = SHuBERTInferenceRuntime(
        bundle,
        device=args.device,
        generation_num_beams=5,
        generation_max_length=2048,
        dino_batch_size=64,
        model_execution_concurrency=1,
        verify_assets=True,
    )
    from umi_video_reader import VideoReader

    model._video_reader = VideoReader
    if args.source_short_side is not None:
        model._video_reader = make_short_side_reader(model._video_reader, args.source_short_side)
    visual_policy = set_visual_crop_size(model, args.visual_crop_size)
    if any(policy[key] != value for key, value in visual_policy.items()):
        raise RuntimeError("installed visual resolution policy differs")
    if args.calibrator is not None:
        from c5_crop_calibrator import install_crop_calibrator

        install_crop_calibrator(model, args.calibrator)
    install_post_mediapipe_stride(model)
    args.results.parent.mkdir(parents=True, exist_ok=True)
    new_count = 0
    for row in rows:
        name = row["sentence_name"]
        if name in completed:
            continue
        if new_count >= args.limit:
            break
        clip = args.clip_dir / f"{name}.mp4"
        landmark_path = args.landmark_dir / f"{name}.npz"
        if (
            not clip.is_file()
            or not landmark_path.is_file()
            or sha256(clip) != row["resampled_clip_sha256"]
            or sha256(landmark_path) != row["landmarks_sha256"]
        ):
            raise ValueError(f"holdout media differs: {name}")
        landmarks, metadata = unpack_landmarks(landmark_path)
        if metadata.get("source_sha256") != row["resampled_clip_sha256"]:
            raise ValueError(f"holdout landmark provenance differs: {name}")

        def pinned(frames, *_args):
            if len(frames) != len(landmarks):
                raise ValueError("video frame count differs from pinned landmarks")
            return landmarks

        model._video_holistic = pinned
        started = time.perf_counter()
        hypothesis = model.translate_path(clip)
        active_seconds = time.perf_counter() - started
        reference_words = normalize(row["reference"])
        edits = distance(reference_words, normalize(hypothesis))
        record = {
            "schema": SCHEMA,
            "split": "internal_training_holdout",
            "promotion_eligible": False,
            "example_id": name,
            "source_group": row["source_group"],
            "signer": row["signer"],
            "reference": row["reference"],
            "hypothesis": hypothesis,
            "single_reference_wer": edits / len(reference_words),
            "single_reference_word_edits": edits,
            "reference_word_count": len(reference_words),
            "evaluation_index_sha256": index_hash,
            "bundle_identity_sha256": identity_hash,
            "bundle_runtime_sha256": runtime_hash,
            "stride_runtime_sha256": stride_hash,
            "clip_sha256": row["resampled_clip_sha256"],
            "landmarks_sha256": row["landmarks_sha256"],
            "calibrator_sha256": calibrator_hash,
            **policy,
            "device": args.device,
            "mps_idle_fraction": idle_fraction,
            "miner_health_policy": "user_authorized_maintenance",
            "active_seconds": active_seconds,
            "diagnostics": model.last_diagnostics,
        }
        with args.results.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
            output.flush()
            os.fsync(output.fileno())
        completed[name] = record
        new_count += 1
        print(f"{len(completed)}/134 {name} WER={record['single_reference_wer']:.3f}", flush=True)
        if args.device == "mps":
            time.sleep(active_seconds * idle_fraction / (1 - idle_fraction))
    print(json.dumps({"examples": len(completed)}, sort_keys=True))


if __name__ == "__main__":
    main()
