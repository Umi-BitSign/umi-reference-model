"""Produce all532 native224 feature streams with original frozen CPU visual weights.

No translator, optimizer, scorer, exact sign labels or confirmation media is used.
The dataset still needs an actual C5 timing-risk/identity qualification.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from c5_timing_native_input import input_receipt, read_training224
from c5_timing_visibility import POLICY as VISIBILITY
from fetch_c5_broad_how2sign import ROOT, artifact, read, require, sha256
from launch_terra_visual_job import identity, now, write
from terra_training_recovery import verify_snapshot

MIXED = ROOT / "runs/c5-mixed-native112-data-v1/plan.json"
PUBLIC = ROOT / "runs/c5-public112-word-diagnostic-v1/plan.json"
PREDECESSOR = ROOT / "checkpoints/c5-public112-word-diagnostic-v1"
POLICY = {
    "CPU_threads": 2,
    "GPU_use": False,
    "native_side": 224,
    "output_budget_bytes": 512 * 1024**2,
    "physical_free_floor_bytes": 84 * 1024**3,
    "headroom_bytes": 128 * 1024**2,
    "feature_widths": {
        "face_features": 384,
        "left_hand_features": 384,
        "right_hand_features": 384,
        "pose_features": 14,
    },
    "training_clips": {"word": 214, "phrase": 318},
    "sampled_frames": 38876,
    "full_native_feature_reference_indices": [0, 1, 214, 215],
    "native_endpoint_feature_reference": "first and last sampled frame of every clip",
    "native_feature_absolute_tolerance": 1e-4,
    "native_feature_relative_tolerance": 1e-4,
    "legacy_encoder_states_used": False,
    "legacy_visual_features_reused": False,
    "visibility": VISIBILITY,
    "optimizer_updates": 0,
    "new_Terra_requests": 0,
    "training_or_timing_risk_qualified": False,
    "promotion_authorized": False,
}
SOURCES = (
    "prepare_c5_timing_features.py",
    "c5_timing_native_input.py",
    "c5_timing_visibility.py",
    "evaluate_c5_words.py",
    "probe_c5_small_input.py",
    "c5_small_input_vision.py",
)


def release():
    worker, exited = (read(PREDECESSOR / name) for name in ("worker.json", "exit.json"))
    complete = read(PREDECESSOR / "complete.json")
    require(
        worker["uid"] == os.getuid()
        and exited["pid"] == worker["pid"]
        and type(exited["exit_code"]) is int
        and exited["exit_code"] == 0
        and complete["complete"] is True
        and complete["post_hoc_diagnostic_only"] is True,
        "timing features require the actually exited resolution diagnostic",
    )
    live = subprocess.run(
        ["ps", "-p", str(worker["pid"]), "-o", "uid=,lstart="], capture_output=True, text=True
    )
    if live.returncode == 0:
        current = live.stdout.strip().split(maxsplit=1)
        require(
            current != [str(worker["uid"]), worker["ps_started_local"]],
            "previous owned model worker is still present",
        )
    return {
        name: artifact(PREDECESSOR / name) for name in ("worker.json", "exit.json", "complete.json")
    }


def specification():
    from prepare_terra_mixed_native import checked

    predecessor = release()
    mixed = checked(MIXED)
    public = read(PUBLIC)
    verify_snapshot(PUBLIC, public)
    require(
        public["trained_or_partial_visual_checkpoint_installed_for_inference"] is False
        and public["candidate_visual_state_sha256"]
        == public["model_plan"]["pretrained_state_sha256"],
        "timing feature source must be the original public visual model",
    )
    sources = dict(mixed["source_sha256"])
    for name, digest in public["source_sha256"].items():
        require(name not in sources or sources[name] == digest, "conflicting source ancestry")
        sources[name] = digest
    sources.update({"tools/" + name: sha256(ROOT / "tools" / name) for name in SOURCES})
    for name, digest in sources.items():
        require(sha256(ROOT / name) == digest, "timing feature source changed")
    records = mixed["records"]
    require(
        len(records) == 532 and Counter(r["kind"] for r in records) == POLICY["training_clips"],
        "timing feature dataset cannot omit words or phrases",
    )
    return {
        "schema": "umi-studio-complete-native224-timing-feature-plan/1",
        "mixed": artifact(MIXED),
        "public_model": artifact(PUBLIC),
        "predecessor": predecessor,
        "policy": POLICY,
        "records": records,
        "model_plan": public["model_plan"],
        "source_sha256": sources,
        "training_or_timing_risk_qualified": False,
        "promotion_authorized": False,
    }


def checked(path):
    plan = read(path)
    require(
        plan == specification(), "timing feature plan changed complete inventory, values or policy"
    )
    verify_snapshot(path, plan)
    return plan


def owned(path, folder):
    require(
        path.absolute() == path.resolve()
        and path.is_relative_to(ROOT / folder)
        and path != ROOT / folder,
        "timing feature path must be canonical and owned",
    )


def freeze(path):
    from c5_experiment_sources import archive_sources

    owned(path, "runs")
    require(not path.parent.exists(), "timing feature plan needs a fresh directory")
    plan = specification()
    path.parent.mkdir(parents=True)
    write(path, plan)
    archive_sources(path, path.with_name("source-snapshot"))


def guard(output, extra=0):
    paths = list(output.rglob("*")) if output.exists() else []
    require(all(not p.is_symlink() for p in paths), "timing feature output contains aliases")
    used = sum(p.stat().st_size for p in paths if p.is_file())
    require(
        type(extra) is int
        and extra >= 0
        and used + extra <= POLICY["output_budget_bytes"]
        and shutil.disk_usage(ROOT).free
        >= POLICY["physical_free_floor_bytes"]
        + POLICY["headroom_bytes"]
        + POLICY["output_budget_bytes"]
        - used,
        "timing feature output budget or physical reserve failed",
    )


def write_arrays(path, arrays, output):
    temporary = path.with_suffix(".partial.npz")
    require(
        not path.exists() and not temporary.exists(),
        "timing features cannot replace existing bytes",
    )
    guard(output, sum(v.nbytes for v in arrays.values()) + 65536)
    with temporary.open("xb") as f:
        np.savez(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    temporary.rename(path)


def native_reference(hands, face, pixels, indices):
    return {
        "face_features": face.extract_embeddings_from_frames(pixels[indices, 0]),
        "left_hand_features": hands.extract_embeddings_from_frames(pixels[indices, 1]),
        "right_hand_features": hands.extract_embeddings_from_frames(pixels[indices, 2]),
    }


def compare_features(streams, native, indices):
    report = {}
    for name in ("face_features", "left_hand_features", "right_hand_features"):
        actual, reference = streams[name][indices], native[name]
        require(
            actual.dtype == reference.dtype == np.float32
            and actual.shape == reference.shape
            and np.isfinite(actual).all()
            and np.isfinite(reference).all(),
            "native timing feature reference changed shapes or dtype",
        )
        require(
            np.allclose(
                actual,
                reference,
                atol=POLICY["native_feature_absolute_tolerance"],
                rtol=POLICY["native_feature_relative_tolerance"],
            ),
            "original native224 feature comparison failed: " + name,
        )
        report[name] = {
            "maximum_absolute_difference": float(
                np.max(np.abs(actual.astype(np.float64) - reference))
            )
        }
    return {"sample_indices": indices.tolist(), "streams": report}


def require_environment():
    require(
        all(
            os.environ.get(name) == "2"
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        )
        and all(
            os.environ.get(name) == "1"
            for name in ("PYTHONDONTWRITEBYTECODE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
        )
        and os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS") == "threads;2",
        "timing feature extraction requires the pinned offline CPU2 environment",
    )


def run(plan_path, output):
    plan = checked(plan_path)
    owned(output, "checkpoints")
    require(not output.exists(), "timing features require a fresh output")
    require_environment()
    guard(output)
    output.mkdir(parents=True)
    write(output / "worker.json", identity(os.getpid(), [sys.executable, *sys.argv]))
    code = 1
    try:
        import cv2
        import torch
        from c5_small_input_vision import SmallInputDino
        from evaluate_c5_words import original_streams
        from probe_c5_small_input import legacy_models
        from probe_tiny_rgb_capacity import state_digest
        from terra_mixed_native_input import training_row
        from train_terra_c5_tail import gpu_lease

        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        cv2.setNumThreads(2)
        with gpu_lease(), torch.inference_mode():
            hands, face = legacy_models(plan["model_plan"])
            model = SmallInputDino(hands.model, face.model, side=224).requires_grad_(False).eval()
            initial = state_digest(model.state_dict())
            require(
                initial == plan["model_plan"]["pretrained_state_sha256"]
                and all(
                    p.device.type == "cpu" and p.dtype == torch.float32 and not p.requires_grad
                    for p in model.parameters()
                ),
                "timing feature model is not the original frozen CPU FP32 visual model",
            )
            receipts, total_frames, visible_intervals, adjustable_samples = [], 0, 0, 0
            for index, case in enumerate(plan["records"]):
                guard(output)
                row = training_row(case)
                values = read_training224(case)
                pixels = input_receipt(values)
                streams = original_streams(model, values)
                frames = len(values["frame_indices"])
                boundaries = np.r_[0, np.flatnonzero(~values["valid_intervals"]) + 1, frames]
                component_lengths = np.diff(boundaries)
                adjustable = int(np.maximum(component_lengths - 2, 0).sum())
                for name, width in POLICY["feature_widths"].items():
                    require(
                        streams[name].shape == (frames, width)
                        and streams[name].dtype == np.float32
                        and np.isfinite(streams[name]).all(),
                        "timing features lost full finite original streams",
                    )
                require(
                    streams["pose_features"].tobytes() == values["pose_features"].tobytes()
                    and np.array_equal(streams["frame_indices"], values["frame_indices"]),
                    "timing extraction changed original pose bits or sample indices",
                )
                full = index in POLICY["full_native_feature_reference_indices"]
                indices = (
                    np.arange(frames, dtype=np.int64)
                    if full
                    else np.unique(np.asarray([0, frames - 1], dtype=np.int64))
                )
                native = native_reference(hands, face, values["pixels"], indices)
                comparison = compare_features(streams, native, indices)
                reference_path = None
                if full:
                    reference_path = output / f"native-reference-{index:04d}.npz"
                    write_arrays(reference_path, native, output)
                cache = output / f"features-{index:04d}.npz"
                arrays = {
                    **streams,
                    "valid_intervals": values["valid_intervals"],
                    "source_hand_tip_visible": values["source_hand_tip_visible"],
                }
                write_arrays(cache, arrays, output)
                receipt = output / f"clip-{index:04d}.json"
                write(
                    receipt,
                    {
                        "case": case,
                        "row": row,
                        "input": pixels,
                        "features": artifact(cache),
                        "native_feature_comparison": comparison,
                        "visibility_component_lengths": component_lengths.tolist(),
                        "potentially_adjustable_interior_samples": adjustable,
                        "complete_native_feature_reference": artifact(reference_path)
                        if reference_path
                        else None,
                        "decoder_executed": False,
                        "optimizer_updates": 0,
                    },
                )
                receipts.append(artifact(receipt))
                total_frames += frames
                visible_intervals += int(values["valid_intervals"].sum())
                adjustable_samples += adjustable
                del values, streams, arrays, native
                print(
                    json.dumps({"feature_clips": index + 1, "of": 532, "kind": case["kind"]}),
                    flush=True,
                )
            require(
                total_frames == POLICY["sampled_frames"] and len(receipts) == 532,
                "timing feature extraction lost complete532 input coverage",
            )
            require(
                state_digest(model.state_dict()) == initial,
                "timing extraction changed original visual values",
            )
            require(
                plan == checked(plan_path),
                "timing feature source or data changed during extraction",
            )
            require(
                not list(output.glob("*.partial.npz")),
                "timing features contain an incomplete cache",
            )
            guard(output)
            write(
                output / "result.json",
                {
                    "schema": "umi-studio-complete-native224-timing-features/1",
                    "complete": True,
                    "plan": artifact(plan_path),
                    "worker": artifact(output / "worker.json"),
                    "receipts": receipts,
                    "clips": 532,
                    "counts": POLICY["training_clips"],
                    "sampled_frames": total_frames,
                    "timing_visible_intervals": visible_intervals,
                    "potentially_adjustable_interior_samples": adjustable_samples,
                    "original_visual_state_sha256": initial,
                    "original_visual_values_unchanged": True,
                    "native_endpoint_feature_comparisons": 532,
                    "complete_native_feature_comparisons": 4,
                    "CPU_threads": 2,
                    "GPU_use": False,
                    "visual_encoder_executed": True,
                    "decoder_executed": False,
                    "optimizer_updates": 0,
                    "new_Terra_requests": 0,
                    "training_or_timing_risk_qualified": False,
                    "promotion_authorized": False,
                },
            )
        code = 0
    except BaseException as error:
        write(output / "failure.json", {"error": str(error), "automatic_retries": 0})
        raise
    finally:
        write(output / "exit.json", {"pid": os.getpid(), "exit_code": code, "observed_at": now()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("freeze", "plan", "output"):
        parser.add_argument("--" + name, type=lambda p: Path(p).absolute())
    args = parser.parse_args()
    if args.freeze:
        freeze(args.freeze)
    path = args.plan or args.freeze
    if path and args.output:
        run(path, args.output)
    elif not args.freeze:
        parser.error("freeze a timing feature plan, or supply plan and fresh output")


if __name__ == "__main__":
    main()
