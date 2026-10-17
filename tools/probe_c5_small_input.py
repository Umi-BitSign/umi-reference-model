"""Zero-update qualification of real C5 DINO weights on native112px training crops."""

from __future__ import annotations

import argparse
import copy
import gc
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from benchmark_fleurs_dev import sha256
from c5_small_input_vision import STREAMS, SmallInputDino, input_tensor
from launch_terra_visual_job import identity, now, write
from probe_tiny_rgb_capacity import compare, state_digest
from terra_training_recovery import artifact, verify_snapshot
from tiny_rgb_capacity_protocol import LIMITS, verified_comparison
from training_media_artifact import media_artifact

ROOT = Path(__file__).resolve().parents[1]
DATA_PLAN = ROOT / "runs/roi-appearance-data-v1/plan.json"
POLICY = {
    "native_side": 112,
    "control_side": 224,
    "training_record_indices": [0, 1],
    "crop_coverage": "complete stride2 timelines; native streamed/reference pixel equality",
    "numerical_frames": "first and final sampled frame of each complete training clip",
    "numerical_loss": "fixed linear feature cotangent; component numerics only, not Terra loss",
    "maximum_reference_RGB_bytes": 128 * 1024**2,
    "control_maximum_absolute_error": 1e-4,
    "cpu_threads": 2,
    "gpu_duty_fraction": 0.5,
    "optimizer_updates": 0,
    "new_Terra_requests": 0,
    "new_downloads": 0,
    "training_qualification_complete": False,
    "translation_quality_evaluated": False,
    "promotion_authorized": False,
}


def training_cases(parent):
    reserved = set(parent["reserved_STEM_participants"] + parent["ROI_diagnostic_participants"])
    result = [parent["records"][i] for i in POLICY["training_record_indices"]]
    for row in result:
        source = row["original_row"]
        if (
            row.get("roi_role") != "training"
            or row.get("roi_training_allowed") is not True
            or source.get("role") != "training"
            or source.get("training_allowed") is not True
            or source["signer"] in reserved
            or source["original_pair"]["header"]["sampled_frames"] < 2
        ):
            raise ValueError("small DINO probe needs complete, nonreserved training identities")
    if len({r["original_row"]["example_id"] for r in result}) != len(result):
        raise ValueError("small DINO numerical cases cannot be duplicated")
    return result


def specification():
    from c5_winner_reference import load_reference
    from prepare_roi_appearance import checked as checked_data
    from tiny_rgb_full_run import runtime_binding

    parent = checked_data(DATA_PLAN)
    reference = load_reference()
    model = ROOT / reference["bundle_root"] / "model"
    assets = {
        str(path.relative_to(ROOT)): sha256(path)
        for path in (
            ROOT / "C5_BASELINE_REFERENCE.json",
            model / "models/dinov2hand.safetensors",
            model / "models/dinov2face.safetensors",
            model / "vendor/demo/dinov2_features.py",
            model / "vendor/demo/device.py",
            model / "vendor/demo/video_reader.py",
            model / "vendor/demo/crop_hands.py",
            model / "vendor/demo/crop_face.py",
            *(model / "vendor/dinov2-source").rglob("*.py"),
        )
    }
    sources = dict(parent["source_sha256"])
    for name in (
        "probe_c5_small_input.py",
        "c5_small_input_vision.py",
        "c5_certification_budget.py",
        "c5_winner_reference.py",
        "probe_tiny_rgb_capacity.py",
        "tiny_rgb_capacity_protocol.py",
        "tiny_rgb_full_run.py",
        "terra_training_storage.py",
        "training_media_artifact.py",
    ):
        sources[f"tools/{name}"] = sha256(ROOT / "tools" / name)
    return {
        "schema": "umi-studio-c5-native-small-input-qualification/1",
        "policy": POLICY,
        "limits": LIMITS,
        "data_plan": artifact(DATA_PLAN),
        "reference": reference,
        "records": training_cases(parent),
        "source_sha256": sources,
        "assets": assets,
        "runtime_binding": runtime_binding(),
        "resource_policy": parent["resource_policy"],
        "resource_policy_sha256": parent["resource_policy_sha256"],
    }


def checked(path):
    plan = json.loads(path.read_text())
    if plan != specification():
        raise ValueError("small DINO qualification inputs, recipe or source changed")
    verify_snapshot(path, plan)
    return plan


def freeze(path):
    from c5_experiment_sources import archive_sources
    from terra_training_storage import storage

    path = path.resolve()
    if path.parent.exists() or not path.is_relative_to(ROOT / "runs"):
        raise ValueError("small DINO qualification requires a fresh owned plan")
    storage(0)
    if os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS") != "threads;2":
        raise ValueError("small DINO decode requires the declared two-thread FFmpeg policy")
    plan = specification()
    path.parent.mkdir(parents=True)
    write(path, plan)
    archive_sources(path, path.with_name("source-snapshot"))
    print(json.dumps({"plan_sha256": sha256(path)}), flush=True)


def pack_rgb(values, frames, side):
    if len(values) != frames or any(
        v.dtype != np.uint8 or v.shape != (side, side, 3) for v in values
    ):
        raise ValueError("small DINO native crops changed count, side or dtype")
    return np.ascontiguousarray(np.stack(values)[..., ::-1])


def crop_case(row, side):
    from extract_mediapipe_landmarks import unpack_landmarks
    from prepare_roi_appearance import SampledRgbReader, native_helpers

    pair = row["original_row"]["original_pair"]
    header = pair["header"]
    video, landmark = ROOT / pair["raw"]["video"], ROOT / row["landmarks"]["path"]
    if sha256(video) != pair["raw"]["video_sha256"] or media_artifact(landmark) != row["landmarks"]:
        raise ValueError("small DINO video or landmark bytes changed")
    landmarks, metadata = unpack_landmarks(landmark)
    if (
        metadata["source_sha256"] != pair["raw"]["video_sha256"]
        or metadata["frames"] != header["source_frames"]
        or metadata["fps"] != 25
        or header["source_frames"] * np.prod(header["source_shape"]) * 3
        > POLICY["maximum_reference_RGB_bytes"]
    ):
        raise ValueError("small DINO source metadata or native decode bound differs")
    sampled = {j: landmarks[i] for j, i in enumerate(range(0, header["source_frames"], 2))}
    if any(
        not np.isfinite(np.asarray(points)).all()
        for value in sampled.values()
        if value is not None
        for points in value.values()
        if points is not None
    ):
        raise ValueError("small DINO active landmark coordinates must be finite")
    hands, face, reader_type = native_helpers()
    with SampledRgbReader(video, header) as reader:
        left, right = hands(output_size=(side, side)).extract_hand_frames(reader, sampled)
    with SampledRgbReader(video, header) as reader:
        facial = face(output_size=(side, side)).extract_face_frames(reader, sampled)
    count = header["sampled_frames"]
    pixels = np.stack([pack_rgb(v, count, side) for v in (facial, left, right)], axis=1)
    native_reader = reader_type(video)
    full = native_reader.get_batch(range(len(native_reader))).asnumpy()[::2].copy()
    del native_reader
    left, right = hands(output_size=(side, side)).extract_hand_frames(full, sampled)
    facial = face(output_size=(side, side)).extract_face_frames(full, sampled)
    reference = np.stack([pack_rgb(v, count, side) for v in (facial, left, right)], axis=1)
    if not np.array_equal(pixels, reference):
        raise ValueError("small DINO streamed crops differ from pinned native reference")
    return pixels


def legacy_models(plan):
    from prepare_roi_appearance import DEMO, native_helpers

    model = ROOT / plan["reference"]["bundle_root"] / "model"
    demo = model / "vendor/demo"
    if sha256(demo / "video_reader.py") != sha256(DEMO / "video_reader.py"):
        raise ValueError("small DINO original and crop video readers must be byte-identical")
    # Bind the cropper's canonical module before the legacy loader imports its
    # byte-identical VideoReader through Python's shared module cache.
    native_helpers()
    os.environ["SHUBERT_DINOV2_SOURCE"] = str(model / "vendor/dinov2-source")
    previous = list(sys.path)
    sys.path.insert(0, str(demo))
    try:
        spec = importlib.util.spec_from_file_location(
            "small_dino_legacy", demo / "dinov2_features.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return tuple(
            module.DINOEmbedder(str(model / "models" / name), batch_size=2, device="cpu")
            for name in ("dinov2hand.safetensors", "dinov2face.safetensors")
        )
    finally:
        sys.path[:] = previous


def feature_probe(model, pixels):
    model.zero_grad(set_to_none=True)
    features = model(pixels)
    output = torch.cat([features[name] for name in STREAMS])
    cotangent = torch.linspace(-1, 1, 384, device=output.device)[None]
    loss = (output * cotangent).mean()
    loss.backward()
    gradients = {}
    for name, parameter in model.trainable_parameters().items():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise ValueError(f"small DINO lacks a finite actual visual gradient: {name}")
        gradients[name] = parameter.grad.detach().cpu().clone()
    if model.hands.mask_token.grad is not None or model.face.mask_token.grad is not None:
        raise ValueError("small DINO unused mask tokens received gradients")
    return float(loss.detach().cpu()), output.detach().cpu(), gradients


def run(path, output):
    import cv2
    from c5_certification_budget import CertificationComputeWindow, WinnerDutyCycle
    from PIL import Image
    from terra_training_storage import storage

    plan = checked(path)
    output = output.resolve()
    if output.exists() or not output.is_relative_to(ROOT / "checkpoints"):
        raise ValueError("small DINO probe requires fresh owned output")
    storage(0)
    if os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS") != "threads;2":
        raise ValueError("small DINO decode requires the declared two-thread FFmpeg policy")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(2)
    budget = CertificationComputeWindow(ROOT / plan["resource_policy"])
    if budget.duty_fraction() != 0.5 or budget.cpu_threads() != 2:
        raise ValueError("small DINO probe requires the current GPU50%/CPU2 policy")
    if not torch.backends.mps.is_available():
        raise ValueError("small DINO numerical qualification requires native Metal")
    output.mkdir(parents=True)
    write(output / "worker.json", identity(os.getpid(), [sys.executable, *sys.argv]))
    code = 1
    try:
        duty = WinnerDutyCycle(torch, budget)
        hands, face = legacy_models(plan)
        original = state_digest(
            {
                **{f"hands.{k}": v for k, v in hands.model.state_dict().items()},
                **{f"face.{k}": v for k, v in face.model.state_dict().items()},
            }
        )
        # Clone before the224px hook is installed, so112px gets its own strict hook.
        small = SmallInputDino(copy.deepcopy(hands.model), copy.deepcopy(face.model))
        control = SmallInputDino(hands.model, face.model, side=224)
        if (
            len(small.trainable_parameters()) != 350
            or sum(p.numel() for p in small.trainable_parameters().values()) != 43260672
            or state_digest(small.state_dict()) != original
        ):
            raise ValueError("small DINO actual pretrained parameter scope differs")
        metal = copy.deepcopy(small).to("mps")
        reports = []
        for number, row in enumerate(plan["records"]):
            full224, full112 = crop_case(row, 224), crop_case(row, 112)
            selected = [0, len(full112) - 1]
            pixels224, pixels112 = full224[selected], full112[selected]
            with torch.no_grad():
                legacy = {
                    "face_features": torch.from_numpy(
                        face.extract_embeddings_from_frames(list(pixels224[:, 0]))
                    ),
                    "left_hand_features": torch.from_numpy(
                        hands.extract_embeddings_from_frames(list(pixels224[:, 1]))
                    ),
                    "right_hand_features": torch.from_numpy(
                        hands.extract_embeddings_from_frames(list(pixels224[:, 2]))
                    ),
                }
                actual = control(input_tensor(pixels224, side=224))
            error = max(float((legacy[k] - actual[k]).abs().max()) for k in STREAMS)
            if error > POLICY["control_maximum_absolute_error"]:
                raise ValueError("small DINO224px path does not reproduce the original embedder")
            sheet = Image.fromarray(
                np.concatenate([np.concatenate(list(frame), axis=1) for frame in pixels112], axis=0)
            )
            sheet.save(output / f"native112-{number:02d}.png")
            cpu_loss, cpu_output, cpu_gradients = feature_probe(small, input_tensor(pixels112))
            mps_loss, mps_output, mps_gradients = duty.run(
                lambda pixels=pixels112: feature_probe(metal, input_tensor(pixels).to("mps"))
            )
            comparison = compare(
                cpu_gradients, mps_gradients, cpu_loss, mps_loss, cpu_output, mps_output
            )
            report = {
                "source": row["original_row"]["source"],
                "example_id": row["original_row"]["example_id"],
                "complete_crop_frames": len(full112),
                "numerical_sample_positions": selected,
                "both_sides_native_pixel_parity": True,
                "control224_maximum_absolute_error": error,
                "comparison": comparison,
                "resources": duty.receipt(),
            }
            write(output / f"comparison-{number:02d}.json", report)
            verified_comparison(comparison, 350)
            reports.append(report)
            del full224, full112, cpu_gradients, mps_gradients
            gc.collect()
        if (
            state_digest(small.state_dict()) != original
            or state_digest(metal.state_dict()) != original
        ):
            raise ValueError("small DINO numerical qualification changed pretrained weights")
        checked(path)
        write(
            output / "qualification.json",
            {
                "complete": True,
                "plan_sha256": sha256(path),
                "component_numerics_passed": True,
                "training_qualification_complete": False,
                "translation_quality_evaluated": False,
                "pretrained_state_sha256": original,
                "trainable_parameters": 43260672,
                "trainable_tensors": 350,
                "comparisons": reports,
                "resources": duty.receipt(),
                "optimizer_updates": 0,
                "new_Terra_requests": 0,
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
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.freeze:
        freeze(args.freeze)
    if args.plan or args.output:
        if not (args.plan and args.output):
            parser.error("--plan and --output are required together")
        run(args.plan, args.output)
    elif not args.freeze:
        parser.error("choose --freeze or --plan/--output")


if __name__ == "__main__":
    main()
