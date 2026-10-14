"""Zero-update real-ASL qualification of raw vision plus frozen winner language."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import sys
from pathlib import Path


def gradients(model):
    import torch

    values = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            if parameter.grad is not None:
                raise RuntimeError("winner frozen language received parameter gradients")
            continue
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise RuntimeError(f"student lacks finite gradient: {name}")
        values[name] = parameter.grad.detach().cpu().clone()
    if len(values) != 53:
        raise RuntimeError("student meaningful optimizer topology differs")
    return values


def run(plan_path, output):
    import hashlib

    import torch
    from benchmark_fleurs_dev import sha256
    from c5_certification_budget import CertificationComputeWindow, WinnerDutyCycle
    from c5_winner_language_decoder import load_winner_language
    from probe_tiny_rgb_capacity import compare, native_gelus, state_digest
    from tiny_rgb_capacity_protocol import verified_comparison
    from tiny_rgb_data import pixel_tensor
    from tiny_rgb_full_training_tools import decode_rgb_video_fast, masked_length_buckets
    from tiny_rgb_visual_teacher_data import load_targets
    from tiny_rgb_winner_student import (
        RawRgbWinnerStudent,
        load_frozen_winner_encoder,
        student_objective,
    )
    from tiny_rgb_winner_student_protocol import BUDGET, RECIPE, ROOT, checked_plan
    from transformers import ByT5Tokenizer

    plan = checked_plan(plan_path)
    output = output.resolve()
    if output.exists() or not output.is_relative_to(ROOT / "runs"):
        raise ValueError("student proof needs fresh private runs/ output")
    output.mkdir(parents=True)
    budget = CertificationComputeWindow(BUDGET)
    torch.set_num_threads(budget.cpu_threads())
    torch.set_num_interop_threads(1)
    if not torch.backends.mps.is_available():
        raise RuntimeError("winner student proof requires Metal")
    duty = WinnerDutyCycle(torch, budget)
    language, reference = load_winner_language()
    language_digest = state_digest(language.state_dict())
    teacher = load_frozen_winner_encoder(reference)
    tokenizer = ByT5Tokenizer.from_pretrained(
        ROOT / reference["bundle_root"] / "model/models/byt5_base", local_files_only=True
    )
    results = []

    def append(row):
        with output.joinpath("comparisons.jsonl").open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    for pair in plan["cases"]:
        row = pair["raw"]
        pixels, indices, info = decode_rgb_video_fast(
            row["video"], RECIPE["short_side"], RECIPE["target_fps"], RECIPE["max_frames"]
        )
        if (
            len(pixels) != pair["header"]["sampled_frames"]
            or hashlib.sha256(indices.tobytes()).hexdigest()
            != pair["header"]["frame_indices_sha256"]
            or info["source_shape"] != pair["header"]["source_shape"]
        ):
            raise ValueError("student proof RGB/native timeline differs")
        labels = tokenizer(row["reference"], return_tensors="pt", truncation=False).input_ids
        if labels.shape[1] > RECIPE["max_text_tokens"]:
            raise ValueError("student proof cannot truncate winner byte targets")
        video = pixel_tensor(pixels)[None]
        feature_targets = {
            name: torch.from_numpy(values)[None] for name, values in load_targets(pair).items()
        }
        teacher.cpu().eval()
        with torch.no_grad():
            latent = teacher(**feature_targets, return_dict=True).last_hidden_state
        if latent.shape != (1, video.shape[1], 1536) or not torch.isfinite(latent).all():
            raise RuntimeError("winner semantic teacher has invalid native states")
        duty.run(lambda teacher=teacher: teacher.to("mps"))

        def semantic_target(feature_targets=feature_targets):
            with torch.no_grad():
                return teacher(
                    **{name: value.to("mps") for name, value in feature_targets.items()},
                    return_dict=True,
                ).last_hidden_state.cpu()

        actual_latent = duty.run(semantic_target)
        teacher_error = float((actual_latent - latent).abs().max())
        identity = {
            "source": row["source"],
            "example_id": row["example_id"],
            "frames": video.shape[1],
            "byte_target_tokens": labels.shape[1],
            "training_only": True,
            "optimizer_updates": 0,
        }
        append(
            {
                **identity,
                "stage": "frozen_winner_teacher_forward",
                "maximum_absolute_difference": teacher_error,
            }
        )
        if teacher_error > 1e-3 or not torch.isfinite(actual_latent).all():
            raise RuntimeError("winner semantic teacher CPU/Metal forward mismatch")
        padded, mask, target_labels = masked_length_buckets(video, labels, torch)
        latent_padded = latent.new_zeros(1, padded.shape[1], 1536)
        latent_padded[:, : video.shape[1]] = latent
        targets_padded = {}
        for name, value in feature_targets.items():
            targets_padded[name] = value.new_zeros(1, padded.shape[1], value.shape[-1])
            targets_padded[name][:, : video.shape[1]] = value
        cpu = RawRgbWinnerStudent(copy.deepcopy(language)).eval()
        if (
            sum(p.numel() for p in cpu.trainable_parameters().values())
            != RECIPE["trainable_parameters"]
        ):
            raise RuntimeError("winner student visual/bridge parameter count changed")
        native = copy.deepcopy(cpu)
        native_gelus(native.vision)
        plain_mask = torch.ones(1, video.shape[1], dtype=torch.bool)

        def objective(model, video, mask, labels, latent, features):
            predictions = model(video, mask, labels)
            measured = student_objective(
                predictions,
                latent,
                features,
                latent_coefficient=RECIPE["latent_coefficient"],
                feature_coefficient=RECIPE["feature_coefficient"],
            )
            measured["loss"].backward()
            return (
                float(measured["loss"]),
                predictions["language"].logits.detach().cpu(),
                gradients(model),
            )

        before_loss, before_logits, before_gradients = objective(
            native, video, plain_mask, labels, latent, feature_targets
        )
        cpu_loss, cpu_logits, cpu_gradients = objective(
            cpu, video, plain_mask, labels, latent, feature_targets
        )
        equivalent = compare(
            before_gradients, cpu_gradients, before_loss, cpu_loss, before_logits, cpu_logits
        )
        append({**identity, "stage": "native_cpu_equivalence", "comparison": equivalent})
        verified_comparison(equivalent, 53, cpu_equivalence=True)
        cpu.zero_grad(set_to_none=True)
        padded_loss, padded_logits, padded_gradients = objective(
            cpu, padded, mask, target_labels, latent_padded, targets_padded
        )
        padding = compare(
            cpu_gradients,
            padded_gradients,
            cpu_loss,
            padded_loss,
            cpu_logits,
            padded_logits[:, : labels.shape[1]],
        )
        append({**identity, "stage": "cpu_padding", "comparison": padding})
        verified_comparison(padding, 53)
        cpu.zero_grad(set_to_none=True)
        candidate = duty.run(lambda cpu=cpu: cpu.to("mps"))
        moments = duty.run(
            lambda candidate=candidate: [
                [torch.zeros_like(p) for p in candidate.trainable_parameters().values()]
                for _ in range(2)
            ]
        )

        def metal(
            candidate=candidate,
            padded=padded,
            mask=mask,
            target_labels=target_labels,
            latent_padded=latent_padded,
            targets_padded=targets_padded,
        ):
            return objective(
                candidate,
                padded.to("mps"),
                mask.to("mps"),
                target_labels.to("mps"),
                latent_padded.to("mps"),
                {name: value.to("mps") for name, value in targets_padded.items()},
            )

        metal_loss, metal_logits, metal_gradients = duty.run(metal)
        comparison = compare(
            padded_gradients, metal_gradients, padded_loss, metal_loss, padded_logits, metal_logits
        )
        append({**identity, "stage": "cpu_mps_comparison", "comparison": comparison})
        worst = verified_comparison(comparison, 53)
        if state_digest(candidate.language.state_dict()) != language_digest:
            raise RuntimeError("student proof changed winner language weights")
        results.append(
            {
                **identity,
                "worst_relative_gradient_error": worst,
                "semantic_teacher_forward_maximum_difference": teacher_error,
            }
        )
        print(json.dumps(results[-1]), flush=True)
        del cpu, native, candidate, moments, metal, semantic_target
        del before_gradients, cpu_gradients, padded_gradients, metal_gradients
        gc.collect()
        torch.mps.empty_cache()
    receipt = {
        "complete": True,
        "numerics_passed": True,
        "plan_sha256": sha256(plan_path),
        "real_cases": results,
        "trainable_parameter_tensors": 53,
        "optimizer_updates": 0,
        "weights_saved": False,
        "language_weights_changed": False,
        "teacher_weights_trainable": False,
        "held_out_quality_verified": False,
        "miner_modified": False,
        "resource_receipt": duty.receipt(),
    }
    output.joinpath("receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        json.dumps({"complete": True, "receipt_sha256": sha256(output / "receipt.json")}),
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
        from tiny_rgb_winner_student_protocol import freeze

        freeze(args.freeze)
    elif args.plan and args.output_dir:
        run(args.plan, args.output_dir)
    else:
        parser.error("use --freeze or --plan with --output-dir")


if __name__ == "__main__":
    main()
