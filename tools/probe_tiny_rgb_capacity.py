"""Zero-update CPU/Metal qualification of larger vision with the same small raw RGB."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import sys
from pathlib import Path


def state_digest(values):
    digest = hashlib.sha256()
    for name, tensor in sorted(values.items()):
        digest.update(name.encode())
        digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def native_gelus(model):
    from tiny_rgb_mps_safe import PortableGelu
    from torch import nn

    replacements = 0
    for name, child in tuple(model.named_children()):
        if isinstance(child, PortableGelu):
            setattr(model, name, nn.GELU())
            replacements += 1
        else:
            replacements += native_gelus(child)
    return replacements


def gradients(model):
    import torch

    result = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise RuntimeError(f"capacity lacks finite gradient: {name}")
        result[name] = parameter.grad.detach().cpu().clone()
    return result


def compare(reference, actual, reference_loss, actual_loss, reference_logits, actual_logits):
    import torch

    if reference.keys() != actual.keys():
        raise ValueError("capacity gradient topology differs")
    rows, before, after = [], 0.0, 0.0
    for name in reference:
        old, new = reference[name].double(), actual[name].double()
        old_norm, new_norm = float(old.norm()), float(new.norm())
        before += old_norm**2
        after += new_norm**2
        rows.append(
            {
                "parameter": name,
                "cpu_norm": old_norm,
                "actual_norm": new_norm,
                "finite": bool(torch.isfinite(new).all()),
                "relative_l2_error": float((old - new).norm()) / max(old_norm, 1e-12),
            }
        )
    if before <= 0 or not math.isfinite(before) or not math.isfinite(after):
        raise ValueError("capacity has invalid total gradient signal")
    return {
        "loss_absolute_difference": abs(reference_loss - actual_loss),
        "logit_maximum_absolute_difference": float((reference_logits - actual_logits).abs().max()),
        "total_gradient_norm_relative_error": abs(math.sqrt(after / before) - 1),
        "gradient_differences": rows,
    }


def main():
    for key, value in {
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTORCH_ENABLE_MPS_FALLBACK": "1",
        "OMP_NUM_THREADS": "8",
        "OPENBLAS_NUM_THREADS": "8",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }.items():
        os.environ[key] = value
    sys.dont_write_bytecode = True
    import torch
    from benchmark_fleurs_dev import sha256
    from c5_certification_budget import CertificationComputeWindow, WinnerDutyCycle
    from tiny_rgb_capacity import build_capacity_translator
    from tiny_rgb_capacity_protocol import ROOT, checked_plan, verified_comparison
    from tiny_rgb_data import pixel_tensor
    from tiny_rgb_full_training_tools import decode_rgb_video_fast, masked_length_buckets
    from transformers import AutoTokenizer, T5ForConditionalGeneration

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    plan = checked_plan(args.plan)
    output = args.output_dir.resolve()
    if output.exists() or not output.is_relative_to(ROOT / "runs"):
        raise ValueError("capacity probe needs a fresh runs/ output directory")
    output.mkdir(parents=True)
    budget = CertificationComputeWindow(ROOT / plan["resource_policy"])
    torch.set_num_threads(budget.cpu_threads())
    torch.set_num_interop_threads(1)
    if not torch.backends.mps.is_available():
        raise RuntimeError("capacity qualification requires native Metal")
    duty = WinnerDutyCycle(torch, budget)
    tokenizer = AutoTokenizer.from_pretrained(ROOT / plan["language_path"], local_files_only=True)
    inputs, records = [], []
    for case in plan["cases"]:
        row, header = case["training"], case["header"]
        if row["role"] != "training" or row["training_allowed"] is not True:
            raise ValueError("capacity probe rejects evaluation identities")
        if sha256(ROOT / row["video"]) != row["video_sha256"]:
            raise ValueError("capacity source clip changed")
        label = tokenizer(row["reference"], return_tensors="pt", truncation=False).input_ids
        if label.shape[1] != header["target_tokens"] or label.shape[1] > plan["max_text_tokens"]:
            raise ValueError("capacity complete target differs from its audit")
        for side in plan["short_sides"]:
            pixels, indices, info = decode_rgb_video_fast(row["video"], side)
            if (
                len(pixels) != header["sampled_frames"]
                or hashlib.sha256(indices.tobytes()).hexdigest() != header["frame_indices_sha256"]
                or info["source_shape"] != header["source_shape"]
            ):
                raise ValueError("capacity pixel timeline/geometry differs from its audit")
            inputs.append((row, side, pixels, label))
    for capacity, expected in zip(plan["capacities"], plan["visual_budgets"], strict=True):
        for row, side, pixels, label in inputs:
            language = T5ForConditionalGeneration.from_pretrained(
                ROOT / plan["language_path"], local_files_only=True, use_safetensors=True
            )
            language_digest = state_digest(
                {n: v for n, v in language.state_dict().items() if not n.startswith("encoder.")}
            )
            torch.manual_seed(plan["seed"])
            model = build_capacity_translator(language, capacity).eval()
            if state_digest(model.language_model.state_dict()) != language_digest:
                raise RuntimeError("capacity changed supplied language weights")
            counts = model.parameter_counts()
            if counts["scratch_visual_encoder"] != expected["scratch_visual_parameters"]:
                raise RuntimeError("capacity visual parameter budget changed")
            initialization = state_digest(model.state_dict())
            native = copy.deepcopy(model)
            if native_gelus(native.vision) != expected["portable_gelu_replacements"]:
                raise RuntimeError("capacity activation topology changed")
            video = pixel_tensor(pixels)[None]
            mask = torch.ones(1, len(pixels), dtype=torch.bool)
            result = native(video, mask, label)
            native_loss, native_logits = float(result.loss), result.logits.detach().clone()
            result.loss.backward()
            original_gradients = gradients(native)
            del result, native
            gc.collect()
            padded, padded_mask, target = masked_length_buckets(video, label, torch)
            result = model(padded, padded_mask, target)
            cpu_loss = float(result.loss)
            cpu_logits = result.logits[:, : label.shape[1]].detach().clone()
            result.loss.backward()
            cpu_gradients = gradients(model)
            equivalence = compare(
                original_gradients, cpu_gradients, native_loss, cpu_loss, native_logits, cpu_logits
            )
            parameter_tensors = len(list(model.parameters()))
            report = {
                "capacity": capacity,
                "short_side": side,
                "source": row["source"],
                "example_id": row["example_id"],
                "role": "training",
                "training_allowed": True,
                "frames": len(pixels),
                "target_tokens": label.shape[1],
                "parameter_counts": counts,
                "parameter_tensors": parameter_tensors,
                "initialization_sha256": initialization,
                "language_state_sha256": language_digest,
                "cpu_equivalence": equivalence,
                "numerics_passed": False,
                "optimizer_updates": 0,
                "weights_saved": False,
            }
            records.append(report)
            output.joinpath("comparisons.json").write_text(
                json.dumps(records, indent=2, sort_keys=True) + "\n"
            )
            cpu_error = verified_comparison(equivalence, parameter_tensors, cpu_equivalence=True)
            model.zero_grad(set_to_none=True)
            del result, original_gradients
            gc.collect()

            def metal_probe(
                model=model,
                padded=padded,
                padded_mask=padded_mask,
                target=target,
                cpu_gradients=cpu_gradients,
                cpu_loss=cpu_loss,
                cpu_logits=cpu_logits,
                target_length=label.shape[1],
            ):
                model.to("mps")
                # Reserve the two full float32 AdamW moments without an optimizer update.
                moments = [torch.zeros_like(p) for p in model.parameters() for _ in range(2)]
                result = model(padded.to("mps"), padded_mask.to("mps"), target.to("mps"))
                result.loss.backward()
                actual = gradients(model)
                comparison = compare(
                    cpu_gradients,
                    actual,
                    cpu_loss,
                    float(result.loss),
                    cpu_logits,
                    result.logits[:, :target_length].detach().cpu(),
                )
                # Synchronize and capture driver allocation while moments are still resident.
                torch.mps.synchronize()
                allocation = torch.mps.driver_allocated_memory()
                del moments
                return comparison, allocation

            comparison, moment_peak = duty.run(metal_probe)
            report.update(
                {
                    "cpu_mps_comparison": comparison,
                    "future_moment_driver_allocation": moment_peak,
                    **duty.receipt(),
                }
            )
            output.joinpath("comparisons.json").write_text(
                json.dumps(records, indent=2, sort_keys=True) + "\n"
            )
            mps_error = verified_comparison(comparison, parameter_tensors)
            report["numerics_passed"] = True
            output.joinpath("comparisons.json").write_text(
                json.dumps(records, indent=2, sort_keys=True) + "\n"
            )
            print(
                json.dumps(
                    {
                        "capacity": capacity,
                        "short_side": side,
                        "frames": len(pixels),
                        "parameter_tensors": parameter_tensors,
                        "cpu_maximum_relative_error": cpu_error,
                        "mps_maximum_relative_error": mps_error,
                        "moment_driver_bytes": moment_peak,
                    }
                ),
                flush=True,
            )
            del (
                metal_probe,
                model,
                language,
                cpu_gradients,
                cpu_logits,
                native_logits,
                video,
                mask,
                padded,
                padded_mask,
                target,
            )
            gc.collect()
            torch.mps.empty_cache()
    checked_plan(args.plan)
    if len(records) != len(plan["capacities"]) * len(plan["short_sides"]) * len(plan["cases"]):
        raise RuntimeError("capacity qualification coverage is incomplete")
    receipt = {
        "schema": "umi-studio-tiny-rgb-capacity-numerics/1",
        "plan_sha256": sha256(args.plan),
        "comparisons_sha256": sha256(output / "comparisons.json"),
        "numerics_passed": True,
        "training_only": True,
        "optimizer_updates": 0,
        "weights_saved": False,
        "held_out_quality_verified": False,
        "miner_modified": False,
        **duty.receipt(),
    }
    output.joinpath("receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
