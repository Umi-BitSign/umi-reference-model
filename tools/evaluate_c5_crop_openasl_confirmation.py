"""Run one frozen OpenASL confirmation variant for the calibrated C5 crop."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from benchmark_fleurs_dev import distance, normalize, sha256
from extract_mediapipe_landmarks import unpack_landmarks
from resolution_policy import set_visual_crop_size

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "umi-studio-c5-crop-openasl-confirmation-example/1"
EXPECTED_PLAN_SCHEMA = "umi-studio-c5-crop-openasl-confirmation-plan/1"
CLIP_DIR = ROOT / "data/openasl/clips-25fps"
LANDMARK_DIR = ROOT / "data/openasl/landmarks"
BUNDLE = ROOT / "artifacts/c5-corrective-stride2-confirmed"


def load_completed(
    path: Path, *, plan_hash: str, variant: str, configuration: dict
) -> dict[str, dict]:
    completed = {}
    if not path.exists():
        return completed
    for line in path.read_text().splitlines():
        row = json.loads(line)
        clip_id = row.get("clip_id")
        if (
            row.get("schema") != SCHEMA
            or row.get("confirmation_plan_sha256") != plan_hash
            or row.get("variant") != variant
            or row.get("visual_crop_size") != configuration["visual_crop_size"]
            or row.get("calibrator_sha256") != configuration.get("calibrator_sha256")
            or clip_id in completed
        ):
            raise ValueError("existing OpenASL confirmation output differs")
        completed[clip_id] = row
    return completed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--variant", choices=("native224", "calibrated168"), required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--mps-idle-fraction", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument("--miner-maintenance-mode", action="store_true")
    args = parser.parse_args()
    if (
        not args.miner_maintenance_mode
        or not args.plan.is_file()
        or not 1 <= args.limit <= 60
        or not args.results.resolve().is_relative_to((ROOT / "runs").resolve())
        or not BUNDLE.is_dir()
    ):
        parser.error("confirmation paths, bounds or maintenance mode are invalid")
    if (args.device == "mps" and args.mps_idle_fraction != 0.5) or (
        args.device == "cpu" and args.mps_idle_fraction != 0.0
    ):
        parser.error("device idle fraction violates the resource policy")
    plan = json.loads(args.plan.read_text())
    if (
        plan.get("schema") != EXPECTED_PLAN_SCHEMA
        or len(plan.get("examples", [])) != 60
        or len({row["source_group"] for row in plan["examples"]}) != 60
        or plan.get("parent_bundle_identity_sha256")
        != sha256(BUNDLE / "MODEL_IDENTITY.json")
        or plan.get("calibrator_runtime_sha256")
        != sha256(ROOT / "tools/c5_crop_calibrator.py")
        or plan.get("evaluator_sha256") != sha256(Path(__file__))
    ):
        raise ValueError("OpenASL confirmation plan differs")
    configuration = dict(plan["variants"][args.variant])
    calibrator = None
    if configuration["calibrator"] is not None:
        calibrator = ROOT / configuration["calibrator"]
        configuration["calibrator_sha256"] = sha256(calibrator)
        if configuration["calibrator_sha256"] != plan["calibrator_sha256"]:
            raise ValueError("OpenASL confirmation calibrator differs")
    else:
        configuration["calibrator_sha256"] = None
    plan_hash = sha256(args.plan)
    completed = load_completed(
        args.results,
        plan_hash=plan_hash,
        variant=args.variant,
        configuration=configuration,
    )

    for key, value in {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HOME": str(BUNDLE / "run/hf"),
        "TORCH_HOME": str(BUNDLE / "run/torch"),
        "MPLCONFIGDIR": str(BUNDLE / "run/matplotlib"),
        "SHUBERT_DINOV2_SOURCE": str(BUNDLE / "vendor/dinov2-source"),
        "PYTORCH_ENABLE_MPS_FALLBACK": "1",
        "OMP_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
    }.items():
        os.environ[key] = value
    sys.path.insert(0, str(BUNDLE))
    import torch
    from c5_stride2_runtime import install_post_mediapipe_stride
    from runtime import SHuBERTInferenceRuntime

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable")
    model = SHuBERTInferenceRuntime(
        BUNDLE,
        device=args.device,
        generation_num_beams=5,
        generation_max_length=2048,
        dino_batch_size=64,
        model_execution_concurrency=1,
        verify_assets=True,
    )
    from umi_video_reader import VideoReader

    model._video_reader = VideoReader
    resolution = set_visual_crop_size(model, configuration["visual_crop_size"])
    if calibrator is not None:
        from c5_crop_calibrator import install_crop_calibrator

        install_crop_calibrator(model, calibrator)
    install_post_mediapipe_stride(model)
    args.results.parent.mkdir(parents=True, exist_ok=True)
    new_count = 0
    for row in plan["examples"]:
        clip_id = row["clip_id"]
        if clip_id in completed:
            continue
        if new_count >= args.limit:
            break
        clip = CLIP_DIR / f"{clip_id}.mp4"
        landmark_path = LANDMARK_DIR / f"{clip_id}.npz"
        if (
            sha256(clip) != row["clip_sha256"]
            or sha256(landmark_path) != row["landmarks_sha256"]
        ):
            raise ValueError(f"OpenASL confirmation media differs: {clip_id}")
        landmarks, metadata = unpack_landmarks(landmark_path)
        if (
            metadata.get("source_sha256") != row["clip_sha256"]
            or len(landmarks) != row["source_frames"]
        ):
            raise ValueError(f"OpenASL confirmation landmark provenance differs: {clip_id}")

        def pinned(frames, *_args):
            if len(frames) != len(landmarks):
                raise ValueError("OpenASL clip frame count differs from frozen landmarks")
            return landmarks

        model._video_holistic = pinned
        started = time.perf_counter()
        hypothesis = model.translate_path(clip)
        active_seconds = time.perf_counter() - started
        if not hypothesis.strip() or len(hypothesis.encode("utf-8")) > 4096:
            raise ValueError("OpenASL confirmation output is outside the protocol bound")
        reference_words = normalize(row["reference"])
        edits = distance(reference_words, normalize(hypothesis))
        record = {
            "schema": SCHEMA,
            "split": plan["split"],
            "confirmation_plan_sha256": plan_hash,
            "variant": args.variant,
            "clip_id": clip_id,
            "example_id": row["example_id"],
            "source_group": row["source_group"],
            "reference": row["reference"],
            "hypothesis": hypothesis,
            "single_reference_wer": edits / len(reference_words),
            "single_reference_word_edits": edits,
            "reference_word_count": len(reference_words),
            "clip_sha256": row["clip_sha256"],
            "landmarks_sha256": row["landmarks_sha256"],
            "bundle_identity_sha256": plan["parent_bundle_identity_sha256"],
            "calibrator_sha256": configuration["calibrator_sha256"],
            **resolution,
            "device": args.device,
            "mps_idle_fraction": args.mps_idle_fraction,
            "miner_health_policy": "user_authorized_maintenance",
            "active_seconds": active_seconds,
            "diagnostics": model.last_diagnostics,
        }
        with args.results.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
            output.flush()
            os.fsync(output.fileno())
        completed[clip_id] = record
        new_count += 1
        print(
            f"{len(completed)}/60 {args.variant} {clip_id} "
            f"WER={record['single_reference_wer']:.3f} active={active_seconds:.1f}s",
            flush=True,
        )
        if args.device == "mps":
            time.sleep(active_seconds * args.mps_idle_fraction / (1 - args.mps_idle_fraction))
    print(json.dumps({"variant": args.variant, "examples": len(completed)}, sort_keys=True))


if __name__ == "__main__":
    main()
