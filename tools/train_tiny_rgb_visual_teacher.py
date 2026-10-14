"""Qualify and fit a fresh raw-pixel visual student against fixed feature teachers."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUDGET = ROOT / "runs/c5-resource-policy-v2-certification48/gpu-budget.json"
SCHEMA = "umi-studio-raw-rgb-feature-teacher-pilot/1"
RECIPE = {
    "capacity": "medium",
    "seed": 317,
    "short_side": 192,
    "target_fps": 12.5,
    "max_frames": 512,
    "epochs": 64,
    "checks_at_epochs": [0, 1, 4, 8, 16, 32, 64],
    "learning_rate": 3e-4,
    "weight_decay": 0.01,
    "gradient_clip": 1.0,
    "pose_weight": 0.1,
    "augmentation": "none",
    "caption_or_language_updates": 0,
    "initial_diagnostic_weights_reused": False,
    "minimum_training_feature_loss_reduction": 0.5,
    "minimum_wrong_video_feature_loss_increase": 0.05,
}


def sources():
    from benchmark_fleurs_dev import sha256
    from tiny_rgb_capacity_protocol import sources as numerical_sources

    result = numerical_sources()
    for name in (
        "tiny_rgb_visual_teacher.py",
        "tiny_rgb_visual_teacher_data.py",
        "train_tiny_rgb_visual_teacher.py",
        "train_tiny_rgb_capacity_cpu.py",
        "probe_tiny_rgb_capacity.py",
        "prepare_how2sign_precut_expansion_landmarks.py",
        "cache_how2sign_precut_expansion_streams.py",
        "cache_asl_stem_wiki_streams.py",
        "cache_openasl_streams.py",
        "visual_training_data.py",
        "train_visual_lora.py",
        "train_c5_stem_continuation.py",
        "freeze_c5_how2sign_expansion_plan.py",
        "freeze_visual_lora_plan.py",
    ):
        result[f"tools/{name}"] = sha256(ROOT / "tools" / name)
    return result


def checked_plan(path):
    from benchmark_fleurs_dev import sha256
    from tiny_rgb_full_run import runtime_binding
    from tiny_rgb_visual_teacher_data import inputs

    plan = json.loads(path.read_text())
    if (
        plan["schema"] != SCHEMA
        or plan["recipe"] != RECIPE
        or plan["source_sha256"] != sources()
        or plan["runtime_binding"] != runtime_binding()
        or plan["resource_policy"] != str(BUDGET.relative_to(ROOT))
        or plan["miner_modified"] is not False
        or plan["training_only"] is not True
        or plan["caption_or_language_updates"] != 0
    ):
        raise ValueError("feature-teacher immutable implementation/recipe/scope changed")
    for name, digest in plan["data"]["dependency_sha256"].items():
        if sha256(ROOT / name) != digest:
            raise ValueError("feature-teacher source/feature dependency changed")
    if sha256(BUDGET) != plan["resource_policy_sha256"] or inputs() != plan["data"]:
        raise ValueError("feature-teacher protected raw/teacher alignment changed")
    snapshot = json.loads(path.with_name("source-snapshot").joinpath("manifest.json").read_text())
    if snapshot["plan_sha256"] != sha256(path):
        raise ValueError("feature-teacher plan differs from archived source")
    return plan


def freeze(path):
    from benchmark_fleurs_dev import sha256
    from c5_experiment_sources import archive_sources
    from tiny_rgb_full_run import runtime_binding
    from tiny_rgb_visual_teacher_data import inputs, load_targets
    from train_tiny_rgb_capacity_cpu import storage

    path = path.resolve()
    if path.parent.exists() or not path.is_relative_to(ROOT / "runs"):
        raise ValueError("feature teacher needs a fresh runs/ plan directory")
    storage()
    data = inputs()
    for pair in data["examples"] + data["preflight_cases"]:
        load_targets(pair)
    plan = {
        "schema": SCHEMA,
        "recipe": RECIPE,
        "data": data,
        "source_sha256": sources(),
        "runtime_binding": runtime_binding(),
        "resource_policy": str(BUDGET.relative_to(ROOT)),
        "resource_policy_sha256": sha256(BUDGET),
        "training_only": True,
        "caption_or_language_updates": 0,
        "miner_modified": False,
        "qualification": "Fresh192 raw-pixel vision predicts receipted native hand/face/pose "
        "features. All six real CPU activation/padding and Metal gradient proofs must pass "
        "before updates. No language/caption graph, held-out quality, baseline win or promotion.",
    }
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    archive_sources(path, path.with_name("source-snapshot"))
    checked_plan(path)
    print(
        json.dumps(
            {
                "plan_sha256": sha256(path),
                "pilot_clips": len(data["examples"]),
                "preflight_clips": len(data["preflight_cases"]),
                "planned_updates": len(data["examples"]) * RECIPE["epochs"],
            }
        ),
        flush=True,
    )


def build_student():
    import torch
    from tiny_rgb_capacity import visual_config
    from tiny_rgb_capacity_activations import replace_capacity_gelu
    from tiny_rgb_capacity_positions import CanonicalPositionsVision
    from tiny_rgb_language_model import ScratchRgbVision
    from tiny_rgb_visual_teacher import SpatialTeacherHeads

    torch.manual_seed(RECIPE["seed"])
    base = ScratchRgbVision(visual_config(RECIPE["capacity"]))
    vision = CanonicalPositionsVision(base.config)
    vision.load_state_dict(base.state_dict(), strict=True)
    replace_capacity_gelu(vision)
    return torch.nn.ModuleDict(
        {"vision": vision, "heads": SpatialTeacherHeads(vision.config.width)}
    )


def clip_average_diagnostic_loss(features, mask, targets):
    """Compare correct/wrong clips with the same averaging rule, without aligning frames."""
    import torch
    from tiny_rgb_visual_teacher import STREAM_WIDTHS, visual_teacher_loss

    if (
        features.ndim != 3
        or features.shape[-1] != sum(STREAM_WIDTHS.values())
        or mask.shape != features.shape[:2]
        or mask.dtype != torch.bool
        or not (mask.sum(1) > 0).all()
    ):
        raise ValueError("clip-average diagnostic requires complete masked feature sequences")
    safe = features.masked_fill(~mask[..., None], 0)
    average = safe.sum(1, keepdim=True) / mask.sum(1, keepdim=True)[..., None]
    predictions = dict(
        zip(STREAM_WIDTHS, average.split(list(STREAM_WIDTHS.values()), -1), strict=True)
    )
    return visual_teacher_loss(
        predictions,
        {name: value.mean(1, keepdim=True) for name, value in targets.items()},
        torch.ones(features.shape[0], 1, dtype=torch.bool, device=features.device),
        pose_weight=RECIPE["pose_weight"],
    )["loss"]


def run(path, output):
    import torch
    from benchmark_fleurs_dev import sha256
    from c5_certification_budget import CertificationComputeWindow, WinnerDutyCycle
    from probe_tiny_rgb_capacity import compare, gradients, native_gelus, state_digest
    from safetensors.torch import save_file
    from tiny_rgb_capacity_protocol import verified_comparison
    from tiny_rgb_data import pixel_tensor
    from tiny_rgb_fit import rank
    from tiny_rgb_full_training_tools import decode_rgb_video_fast, masked_length_buckets
    from tiny_rgb_visual_teacher import STREAM_WIDTHS, visual_teacher_loss
    from tiny_rgb_visual_teacher_data import load_targets
    from train_tiny_rgb_capacity_cpu import storage

    plan = checked_plan(path)
    output = output.resolve()
    if output.exists() or not output.is_relative_to(ROOT / "checkpoints"):
        raise ValueError("feature teacher requires a fresh checkpoint output")
    storage()
    output.mkdir(parents=True)
    budget = CertificationComputeWindow(BUDGET)
    torch.set_num_threads(budget.cpu_threads())
    torch.set_num_interop_threads(1)
    if not torch.backends.mps.is_available():
        raise RuntimeError("feature-teacher qualification requires native Metal")
    duty = WinnerDutyCycle(torch, budget)
    pairs = {
        (p["raw"]["source"], p["raw"]["example_id"]): p
        for p in plan["data"]["examples"] + plan["data"]["preflight_cases"]
    }
    cache = {}
    for key, pair in pairs.items():
        pixels, indices, info = decode_rgb_video_fast(
            pair["raw"]["video"], RECIPE["short_side"], RECIPE["target_fps"], RECIPE["max_frames"]
        )
        header = pair["header"]
        import hashlib

        if (
            hashlib.sha256(indices.tobytes()).hexdigest() != header["frame_indices_sha256"]
            or len(pixels) != header["sampled_frames"]
            or info["source_frames"] != header["source_frames"]
            or info["source_shape"] != header["source_shape"]
        ):
            raise ValueError("feature-teacher decoded pixels/timeline differ from native receipt")
        video = pixel_tensor(pixels)[None]
        padded, mask, _ = masked_length_buckets(video, None, torch)
        targets = {
            name: torch.from_numpy(value)[None] for name, value in load_targets(pair).items()
        }
        padded_targets = {}
        for name, value in targets.items():
            extra = value.new_zeros(1, padded.shape[1], value.shape[-1])
            extra[:, : value.shape[1]] = value
            padded_targets[name] = extra
        cache[key] = (video, targets, padded, mask, padded_targets)

    def append(name, value):
        with output.joinpath(name).open("a") as stream:
            stream.write(json.dumps(value, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def forward(model, video, mask, targets):
        hidden, token_mask = model["vision"](video, mask)
        predictions, frame_mask = model["heads"](hidden, token_mask)
        objective = visual_teacher_loss(
            predictions, targets, frame_mask, pose_weight=RECIPE["pose_weight"]
        )
        features = torch.cat([predictions[name] for name in STREAM_WIDTHS], dim=-1)
        return objective, features

    records = []
    for pair in plan["data"]["preflight_cases"]:
        torch.set_num_threads(budget.cpu_threads())
        key = pair["raw"]["source"], pair["raw"]["example_id"]
        video, targets, padded, mask, padded_targets = cache[key]
        cpu = build_student().eval()
        native = copy.deepcopy(cpu)
        native_gelus(native["vision"])
        plain_mask = torch.ones(1, video.shape[1], dtype=torch.bool)
        original, original_features = forward(native, video, plain_mask, targets)
        original["loss"].backward()
        before = gradients(native)
        actual, actual_features = forward(cpu, video, plain_mask, targets)
        actual["loss"].backward()
        unpadded = gradients(cpu)
        equivalence = compare(
            before,
            unpadded,
            float(original["loss"]),
            float(actual["loss"]),
            original_features.detach(),
            actual_features.detach(),
        )
        row = {
            "source": key[0],
            "example_id": key[1],
            "stage": "native_cpu_equivalence",
            "comparison": equivalence,
            "compared_output": "predicted_teacher_features",
            "optimizer_updates": 0,
        }
        append("preflight.jsonl", row)
        verified_comparison(equivalence, 50, cpu_equivalence=True)
        cpu.zero_grad(set_to_none=True)
        bucketed, bucket_features = forward(cpu, padded, mask, padded_targets)
        bucketed["loss"].backward()
        reference = gradients(cpu)
        padding = compare(
            unpadded,
            reference,
            float(actual["loss"]),
            float(bucketed["loss"]),
            actual_features.detach(),
            bucket_features[:, : video.shape[1]].detach(),
        )
        append("preflight.jsonl", {**row, "stage": "cpu_padding", "comparison": padding})
        verified_comparison(padding, 50)
        cpu.zero_grad(set_to_none=True)
        candidate = duty.run(lambda cpu=cpu: cpu.to("mps"))
        moments = duty.run(
            lambda candidate=candidate: [
                [torch.zeros_like(p) for p in candidate.parameters()] for _ in range(2)
            ]
        )

        def metal_backward(
            candidate=candidate,
            padded=padded,
            mask=mask,
            padded_targets=padded_targets,
            reference=reference,
            bucketed=bucketed,
            bucket_features=bucket_features,
        ):
            objective, features = forward(
                candidate,
                padded.to("mps"),
                mask.to("mps"),
                {name: value.to("mps") for name, value in padded_targets.items()},
            )
            objective["loss"].backward()
            return compare(
                reference,
                gradients(candidate),
                float(bucketed["loss"]),
                float(objective["loss"]),
                bucket_features.detach(),
                features.detach().cpu(),
            )

        comparison = duty.run(metal_backward)
        record = {**row, "stage": "cpu_mps_comparison", "comparison": comparison}
        append("preflight.jsonl", record)
        verified_comparison(comparison, 50)
        records.append(record)
        del metal_backward
        del cpu, native, candidate, moments, before, unpadded, reference
        del original, actual, bucketed, original_features, actual_features, bucket_features
        gc.collect()
        torch.mps.empty_cache()
    proof = {
        "numerics_passed": True,
        "plan_sha256": sha256(path),
        "real_cases": len(records),
        "parameter_tensors": 50,
        "optimizer_updates": 0,
        "caption_or_language_updates": 0,
        "scope": "vision_and_disposable_teacher_heads_only",
        "resource_receipt": duty.receipt(),
    }
    output.joinpath("numerical-receipt.json").write_text(json.dumps(proof, indent=2) + "\n")
    # Begin from fresh seeded vision/heads after qualification; no trained diagnostic state.
    student = build_student().to("mps")
    initial_digest = state_digest(student.state_dict())
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=RECIPE["learning_rate"],
        weight_decay=RECIPE["weight_decay"],
        foreach=False,
    )
    examples = plan["data"]["examples"]
    history = []

    def evaluate(epoch, mode="frame_matched"):
        student.eval()
        losses = []
        for index, pair in enumerate(examples):
            swapped = mode == "wrong_video_clip_average"
            input_pair = examples[(index + 1) % len(examples)] if swapped else pair
            key = input_pair["raw"]["source"], input_pair["raw"]["example_id"]
            _, _, video, mask, input_targets = cache[key]
            target_key = pair["raw"]["source"], pair["raw"]["example_id"]
            targets = cache[target_key][1]

            def score(
                video=video,
                mask=mask,
                targets=targets,
                input_targets=input_targets,
                mode=mode,
            ):
                with torch.no_grad():
                    objective, features = forward(
                        student,
                        video.to("mps"),
                        mask.to("mps"),
                        {name: value.to("mps") for name, value in input_targets.items()},
                    )
                    if mode != "frame_matched":
                        # Correct/wrong-video controls both compare clip averages. Unequal
                        # timelines are never interpolated or presented as frame-aligned.
                        return float(
                            clip_average_diagnostic_loss(
                                features,
                                mask.to("mps"),
                                {name: value.to("mps") for name, value in targets.items()},
                            )
                        )
                    return float(objective["loss"])

            loss = duty.run(score)
            losses.append(loss)
            append(
                "training-feature-checks.jsonl",
                {
                    "epoch": epoch,
                    "example_id": pair["raw"]["example_id"],
                    "diagnostic": mode,
                    "loss": loss,
                    "training_only": True,
                },
            )
            torch.mps.empty_cache()
        return sum(losses) / len(losses)

    for epoch in range(RECIPE["epochs"] + 1):
        if epoch:
            student.train()
            for index, pair in enumerate(
                sorted(examples, key=lambda p, epoch=epoch: rank(p["raw"], f"epoch-{epoch}"))
            ):
                _, _, video, mask, targets = cache[
                    (pair["raw"]["source"], pair["raw"]["example_id"])
                ]

                def update(video=video, mask=mask, targets=targets):
                    optimizer.zero_grad(set_to_none=True)
                    objective, _ = forward(
                        student,
                        video.to("mps"),
                        mask.to("mps"),
                        {name: value.to("mps") for name, value in targets.items()},
                    )
                    if not torch.isfinite(objective["loss"]):
                        raise RuntimeError("teacher objective is nonfinite")
                    objective["loss"].backward()
                    norms = []
                    for name, parameter in student.named_parameters():
                        if parameter.grad is None:
                            raise RuntimeError(f"teacher parameter lacks gradient: {name}")
                        norms.append(parameter.grad.float().norm())
                    values = torch.stack(norms).detach().cpu().tolist()
                    if len(values) != 50 or any(not math.isfinite(v) for v in values):
                        raise RuntimeError("teacher gradient coverage is nonfinite")
                    norm = float(
                        torch.nn.utils.clip_grad_norm_(
                            student.parameters(), RECIPE["gradient_clip"]
                        )
                    )
                    if not math.isfinite(norm):
                        raise RuntimeError("teacher gradient norm is nonfinite")
                    optimizer.step()
                    return {
                        "loss": float(objective["loss"]),
                        "gradient_norm": norm,
                        "gradient_tensors": len(values),
                        "nonzero_gradient_tensors": sum(v > 0 for v in values),
                    }

                measured = duty.run(update)
                append(
                    "steps.jsonl",
                    {
                        "epoch": epoch,
                        "update": (epoch - 1) * len(examples) + index + 1,
                        "example_id": pair["raw"]["example_id"],
                        **measured,
                    },
                )
                torch.mps.empty_cache()
        if epoch not in RECIPE["checks_at_epochs"]:
            continue
        storage()
        matched = evaluate(epoch)
        correct_average = evaluate(epoch, mode="correct_video_clip_average")
        swapped = evaluate(epoch, mode="wrong_video_clip_average")
        history.append(
            {
                "epoch": epoch,
                "training_feature_loss": matched,
                "correct_video_clip_average_feature_loss": correct_average,
                "wrong_video_clip_average_feature_loss": swapped,
                "wrong_video_clip_average_feature_loss_increase": swapped - correct_average,
                "held_out_quality_verified": False,
            }
        )
        output.joinpath("feature-history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(json.dumps(history[-1]), flush=True)
    storage(sum(p.numel() for p in student.parameters()) * 4 + 1024**2)
    weight_path = output / "vision-and-teacher-heads.safetensors"
    save_file(
        {name: value.detach().cpu().contiguous() for name, value in student.state_dict().items()},
        str(weight_path),
    )
    checked_plan(path)
    reduction = 1 - history[-1]["training_feature_loss"] / history[0]["training_feature_loss"]
    decision = {
        "complete": True,
        "plan_sha256": sha256(path),
        "optimizer_updates": len(examples) * RECIPE["epochs"],
        "initial_weights_sha256": initial_digest,
        "checkpoint_sha256": sha256(weight_path),
        "feature_history": history,
        "training_feature_loss_reduction": reduction,
        "training_feature_loss_reduction_passes": reduction
        >= RECIPE["minimum_training_feature_loss_reduction"],
        "passes_feature_learning_gate": (
            reduction >= RECIPE["minimum_training_feature_loss_reduction"]
            and history[-1]["wrong_video_clip_average_feature_loss_increase"]
            >= RECIPE["minimum_wrong_video_feature_loss_increase"]
        ),
        "wrong_video_diagnostic_is_not_frame_matched": True,
        "held_out_quality_verified": False,
        "caption_or_language_updates": 0,
        "baseline_beaten": False,
        "miner_modified": False,
        "resource_receipt": duty.receipt(),
    }
    output.joinpath("decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    print(
        json.dumps({"complete": True, "decision_sha256": sha256(output / "decision.json")}),
        flush=True,
    )


def main():
    for name, value in {
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTORCH_ENABLE_MPS_FALLBACK": "1",
        "OMP_NUM_THREADS": "8",
        "OPENBLAS_NUM_THREADS": "8",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }.items():
        os.environ[name] = value
    sys.dont_write_bytecode = True
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.freeze:
        freeze(args.freeze)
    elif args.plan and args.output_dir:
        run(args.plan, args.output_dir)
    else:
        parser.error("use --freeze or --plan with --output-dir")


if __name__ == "__main__":
    main()
