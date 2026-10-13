"""Evaluate confirmed C5 at native or aspect-preserving source resolution.

This uses the frozen 23-clip open tuning inventory. It is not confirmation
evidence and cannot by itself authorize a production-model promotion.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

from benchmark_fleurs_dev import distance, normalize, sha256
from evaluate_fresh_lowres_tuning import load_tuning
from extract_mediapipe_landmarks import unpack_landmarks
from prepare_c5_resolution_tuning_landmarks import (
    SCHEMA as LANDMARK_SCHEMA,
    landmark_filename,
)
from resolution_policy import (
    DINO_PATCH_SIZE,
    make_short_side_reader,
    set_visual_crop_size,
    validate_source_short_side,
    validate_visual_crop_size,
)

SCHEMA = "umi-studio-c5-resolution-tuning-example/1"


def resolution_policy(
    short_side: int | None, visual_crop_size: int = 224
) -> dict[str, object]:
    validate_source_short_side(short_side)
    validate_visual_crop_size(visual_crop_size)
    return {
        "source_short_side": short_side,
        "source_resize_policy": (
            "aspect_preserving_downscale_only" if short_side is not None else "native"
        ),
        "visual_crop_size": visual_crop_size,
        "dino_patch_size": DINO_PATCH_SIZE,
        "patch_tokens_per_stream_frame": (visual_crop_size // DINO_PATCH_SIZE) ** 2,
        "temporal_stride": 2,
    }


def checked_completed(
    path: Path,
    *,
    inventory_hash: str,
    identity_hash: str,
    runtime_hash: str,
    stride_runtime_hash: str,
    landmark_receipts_hash: str,
    calibrator_hash: str | None,
    policy: dict[str, object],
) -> dict[tuple[str, str], dict]:
    completed = {}
    if not path.exists():
        return completed
    for line in path.read_text().splitlines():
        record = json.loads(line)
        identity = (record.get("source"), record.get("example_id"))
        if (
            record.get("schema") != SCHEMA
            or record.get("tuning_inventory_sha256") != inventory_hash
            or record.get("bundle_identity_sha256") != identity_hash
            or record.get("bundle_runtime_sha256") != runtime_hash
            or record.get("stride_runtime_sha256") != stride_runtime_hash
            or record.get("landmark_receipts_sha256") != landmark_receipts_hash
            or record.get("calibrator_sha256") != calibrator_hash
            or any(record.get(key) != value for key, value in policy.items())
            or identity in completed
        ):
            raise ValueError("existing C5 resolution results use another frozen input")
        completed[identity] = record
    return completed


def checked_landmarks(
    receipts: Path,
    *,
    inventory_hash: str,
    rows: list[dict],
    landmark_dir: Path,
) -> dict[tuple[str, str], dict]:
    records = [json.loads(line) for line in receipts.read_text().splitlines()]
    expected = {(row["source"], row["example_id"]): row for row in rows}
    output = {}
    for record in records:
        identity = (record.get("source"), record.get("example_id"))
        row = expected.get(identity)
        path = landmark_dir / record.get("landmark_filename", "")
        if (
            record.get("schema") != LANDMARK_SCHEMA
            or row is None
            or identity in output
            or record.get("tuning_inventory_sha256") != inventory_hash
            or record.get("video_sha256") != row["video_sha256"]
            or record.get("landmark_filename") != landmark_filename(row)
            or not path.is_file()
            or record.get("landmarks_sha256") != sha256(path)
        ):
            raise ValueError(f"resolution landmark receipt differs: {identity}")
        output[identity] = {**record, "path": path}
    if set(output) != set(expected):
        raise ValueError("resolution landmark receipts do not cover the frozen inventory")
    return output


def summarize(records: list[dict]) -> dict[str, dict[str, float | int]]:
    grouped = defaultdict(list)
    for record in records:
        grouped[record["source"]].append(record)
    return {
        source: {
            "examples": len(items),
            "mean_single_reference_wer": sum(
                item["single_reference_wer"] for item in items
            )
            / len(items),
            "word_weighted_single_reference_wer": sum(
                item["single_reference_word_edits"] for item in items
            )
            / sum(item["reference_word_count"] for item in items),
            "distinct_hypotheses": len({item["hypothesis"] for item in items}),
        }
        for source, items in sorted(grouped.items())
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--landmark-dir", type=Path, required=True)
    parser.add_argument("--landmark-receipts", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--source-short-side", type=int)
    parser.add_argument("--visual-crop-size", type=int, default=224)
    parser.add_argument("--calibrator", type=Path)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--mps-idle-fraction", type=float)
    parser.add_argument("--limit", type=int, default=23)
    parser.add_argument("--miner-maintenance-mode", action="store_true")
    args = parser.parse_args()
    workspace = Path(__file__).resolve().parents[1]
    bundle = args.bundle.resolve()
    idle_fraction = (
        args.mps_idle_fraction
        if args.mps_idle_fraction is not None
        else (0.5 if args.device == "mps" else 0.0)
    )
    if not args.miner_maintenance_mode:
        parser.error("this experiment requires explicit miner maintenance mode")
    if (
        args.limit < 1
        or not bundle.is_dir()
        or not args.inventory.is_file()
        or not args.landmark_dir.is_dir()
        or not args.landmark_receipts.is_file()
        or not args.results.resolve().is_relative_to(workspace / "runs")
    ):
        parser.error("evaluation paths or bounds are invalid")
    if args.device == "mps" and idle_fraction != 0.5:
        parser.error("MPS evaluation requires a 0.5 idle fraction")
    if args.device == "cpu" and idle_fraction != 0:
        parser.error("CPU evaluation requires a zero MPS idle fraction")
    if args.calibrator is not None and (
        not args.calibrator.is_file() or args.visual_crop_size != 168
    ):
        parser.error("crop calibrator requires an existing 168px artifact")
    try:
        policy = resolution_policy(args.source_short_side, args.visual_crop_size)
    except ValueError as error:
        parser.error(str(error))

    required = {
        "identity": bundle / "MODEL_IDENTITY.json",
        "runtime": bundle / "runtime.py",
        "stride_runtime": bundle / "c5_stride2_runtime.py",
        "model": bundle / "models/checkpoint-11625/model.safetensors",
    }
    if any(not path.is_file() for path in required.values()):
        parser.error("confirmed C5 bundle is incomplete")
    for key, value in {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HOME": str(bundle / "run/hf"),
        "TORCH_HOME": str(bundle / "run/torch"),
        "MPLCONFIGDIR": str(bundle / "run/matplotlib"),
        "SHUBERT_DINOV2_SOURCE": str(bundle / "vendor/dinov2-source"),
        "PYTORCH_ENABLE_MPS_FALLBACK": "1",
        "OMP_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
    }.items():
        os.environ[key] = value

    rows = load_tuning(args.inventory, workspace)
    inventory_hash = sha256(args.inventory)
    landmark_receipts_hash = sha256(args.landmark_receipts)
    landmarks = checked_landmarks(
        args.landmark_receipts,
        inventory_hash=inventory_hash,
        rows=rows,
        landmark_dir=args.landmark_dir,
    )
    identity_hash = sha256(required["identity"])
    runtime_hash = sha256(required["runtime"])
    stride_runtime_hash = sha256(required["stride_runtime"])
    calibrator_hash = sha256(args.calibrator) if args.calibrator is not None else None
    completed = checked_completed(
        args.results,
        inventory_hash=inventory_hash,
        identity_hash=identity_hash,
        runtime_hash=runtime_hash,
        stride_runtime_hash=stride_runtime_hash,
        landmark_receipts_hash=landmark_receipts_hash,
        calibrator_hash=calibrator_hash,
        policy=policy,
    )

    sys.path.insert(0, str(bundle))
    import torch
    from c5_stride2_runtime import install_post_mediapipe_stride
    from runtime import SHuBERTInferenceRuntime

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable")
    model = SHuBERTInferenceRuntime(
        bundle,
        device=args.device,
        generation_num_beams=5,
        generation_max_length=2048,
        dino_batch_size=64,
        model_execution_concurrency=1,
        verify_assets=True,
    )
    # Runtime construction installs the bundle's pinned vendor paths.
    from umi_video_reader import VideoReader

    model._video_reader = VideoReader
    if args.source_short_side is not None:
        model._video_reader = make_short_side_reader(
            model._video_reader, args.source_short_side
        )
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
        key = (row["source"], row["example_id"])
        if key in completed:
            continue
        if new_count >= args.limit:
            break
        if sha256(row["video_path"]) != row["video_sha256"]:
            raise ValueError(f"tuning video checksum changed: {row['example_id']}")
        landmark_record = landmarks[key]
        pinned_landmarks, landmark_metadata = unpack_landmarks(landmark_record["path"])

        def use_pinned_landmarks(frames, *_args):
            if len(frames) != len(pinned_landmarks):
                raise ValueError("video frame count differs from pinned landmarks")
            return pinned_landmarks

        model._video_holistic = use_pinned_landmarks
        active_started = time.perf_counter()
        hypothesis = model.translate_path(row["video_path"])
        active_seconds = time.perf_counter() - active_started
        if not hypothesis.strip() or len(hypothesis.encode("utf-8")) > 4096:
            raise ValueError("confirmed C5 output is outside the protocol bound")
        reference_words = normalize(row["reference"])
        hypothesis_words = normalize(hypothesis)
        edits = distance(reference_words, hypothesis_words)
        record = {
            "schema": SCHEMA,
            "split": "tuning",
            "confirmation_evaluated": False,
            "source": row["source"],
            "source_group": row["source_group"],
            "signer": row.get("signer"),
            "example_id": row["example_id"],
            "reference": row["reference"],
            "hypothesis": hypothesis,
            "single_reference_wer": edits / len(reference_words),
            "single_reference_word_edits": edits,
            "reference_word_count": len(reference_words),
            "video_sha256": row["video_sha256"],
            "tuning_inventory_sha256": inventory_hash,
            "bundle_identity_sha256": identity_hash,
            "bundle_runtime_sha256": runtime_hash,
            "stride_runtime_sha256": stride_runtime_hash,
            "landmark_receipts_sha256": landmark_receipts_hash,
            "landmarks_sha256": landmark_record["landmarks_sha256"],
            "calibrator_sha256": calibrator_hash,
            "landmark_policy": "pinned_linux_cpu_mediapipe_0.10.14",
            "landmark_frames": landmark_metadata["frames"],
            "model_sha256": sha256(required["model"]),
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
        completed[key] = record
        new_count += 1
        print(
            f"{len(completed)}/23 {row['source']} {row['example_id']} "
            f"WER={record['single_reference_wer']:.3f} active={active_seconds:.1f}s",
            flush=True,
        )
        if args.device == "mps" and idle_fraction:
            time.sleep(active_seconds * idle_fraction / (1 - idle_fraction))

    records = list(completed.values())
    print(json.dumps({"examples": len(records), "summaries": summarize(records)}, sort_keys=True))


if __name__ == "__main__":
    main()
