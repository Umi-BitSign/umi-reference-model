"""Qualify decoder or full-translator training from the verified C5 winner."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

from c5_winner_reference import ROOT, load_reference, sha256

UNUSED_PREFIX = "encoder.adapter.signhubert_adapter.signhubert."
UNUSED_FEATURE_PARAMETERS = {
    UNUSED_PREFIX + name
    for name in (
        "mask_emb",
        "layer_norm.weight",
        "layer_norm.bias",
        "encoder.layers.11.final_layer_norm.weight",
        "encoder.layers.11.final_layer_norm.bias",
        *(f"heads.{index}.{kind}" for index in range(4) for kind in ("weight", "bias")),
    )
}


def trainable_full(model):
    selected = {}
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name not in UNUSED_FEATURE_PARAMETERS)
        if parameter.requires_grad:
            selected[name] = parameter
    if not selected:
        raise ValueError("winner has no trainable translator parameters")
    return selected


def trainable_decoder(model):
    """Keep the winning visual encoder fixed; update the complete language decoder."""
    selected = {}
    for name, parameter in model.named_parameters():
        allowed = (
            name.startswith("decoder.")
            or name.startswith("decoder_emb.")
            or name.startswith("lm_head.")
        )
        parameter.requires_grad_(allowed)
        if allowed:
            selected[name] = parameter
    if not selected or not any(name.startswith("decoder.") for name in selected):
        raise ValueError("winner has no supported trainable decoder")
    return selected


def decoder_gradient_guard(model, selected, torch):
    if any(
        parameter.grad is not None
        for name, parameter in model.named_parameters()
        if name not in selected
    ):
        raise RuntimeError("frozen winner encoder received gradients")
    if any(parameter.grad is None for parameter in selected.values()):
        raise RuntimeError("winner decoder gradient coverage is incomplete")
    norms = (
        torch.stack([parameter.grad.float().norm() for parameter in selected.values()])
        .detach()
        .cpu()
        .tolist()
    )
    if any(not math.isfinite(norm) for norm in norms) or not any(norm > 0 for norm in norms):
        raise RuntimeError("winner decoder gradients are nonfinite or absent")
    return {
        "parameter_tensors": len(norms),
        "nonzero_gradient_tensors": sum(norm > 0 for norm in norms),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "mps"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scope", choices=("decoder", "full"), default="decoder")
    parser.add_argument("--resource-policy", type=Path)
    args = parser.parse_args()
    output = args.output.absolute()
    if output.exists() or not output.is_relative_to(ROOT / "runs"):
        raise ValueError("winner preflight requires a new private runs/ output")
    reference = load_reference()
    sys.dont_write_bytecode = True
    budget = None
    if args.resource_policy:
        from c5_certification_budget import CertificationComputeWindow

        budget = CertificationComputeWindow(args.resource_policy)
    if args.scope == "full" and budget is None:
        raise ValueError("full winner qualification needs the separate48-hour resource policy")
    model_root = ROOT / reference["bundle_root"] / "model"
    for key, value in {
        "OMP_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
        "PYTORCH_ENABLE_MPS_FALLBACK": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "SHUBERT_DINOV2_SOURCE": str(model_root / "vendor/dinov2-source"),
    }.items():
        os.environ[key] = value
    sys.path.insert(0, str(model_root / "vendor"))
    sys.path.insert(0, str(model_root / "vendor/demo"))
    import torch
    from c5_full_coverage_mps import load_training
    from inference import SignLanguageByT5ForConditionalGeneration
    from tiny_rgb_fit import RgbDutyCycle, metal_working_set_bytes
    from train_c5_stem_continuation import stride2_streams
    from transformers import ByT5Tokenizer

    torch.set_num_threads(budget.cpu_threads() if budget else 2)
    torch.set_num_interop_threads(1)
    if args.device == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("winner preflight requires native Metal")
        if budget:
            from c5_certification_budget import WinnerDutyCycle

            duty = WinnerDutyCycle(torch, budget)
        else:
            torch.mps.set_per_process_memory_fraction(6 * 1024**3 / metal_working_set_bytes())
            duty = RgbDutyCycle(torch, ROOT / "runs/c5-resource-policy-v1/gpu-budget.json")
        run = duty.run
    else:
        duty = None

        def run(operation):
            return operation()

    training = load_training(
        ROOT / "runs/c5-all-source-corpus-v2/audit.json",
        ROOT / "runs/c5-all-source-corpus-v2/training-streams.jsonl",
    )

    def sampled_frames(row):
        return row["frames"] if row.get("stored_temporal_stride") == 2 else (row["frames"] + 1) // 2

    cases = {
        "longest_frames": max(training, key=sampled_frames),
        "longest_target": max(training, key=lambda row: len(row["reference"].encode())),
        "fleurs": next(row for row in training if row["source"] == "fleurs"),
    }
    model = SignLanguageByT5ForConditionalGeneration.from_pretrained(
        model_root / "models/checkpoint-11625", local_files_only=True
    )
    selected = trainable_full(model) if args.scope == "full" else trainable_decoder(model)
    if args.scope == "full" and not UNUSED_FEATURE_PARAMETERS.issubset(
        dict(model.named_parameters())
    ):
        raise ValueError("winner unused-feature topology differs from the reviewed architecture")
    counts = {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(parameter.numel() for parameter in selected.values()),
    }
    run(lambda: model.to(args.device))
    model.eval()
    reserve = run(lambda: torch.empty(counts["trainable"] * 2, device=args.device))
    tokenizer = ByT5Tokenizer.from_pretrained(
        model_root / "models/byt5_base", local_files_only=True
    )
    records = []
    for name, row in cases.items():
        bound = {**row, "cache": str(ROOT / row["cache"])}
        if row.get("sidecar"):
            bound["sidecar"] = str(ROOT / row["sidecar"])
        arrays = stride2_streams(bound)
        labels = tokenizer(row["reference"], return_tensors="pt", truncation=False).input_ids
        model.zero_grad(set_to_none=True)
        cpu_reference = None
        if args.scope == "full" and args.device == "mps":
            run(lambda: model.to("cpu"))
            torch.mps.empty_cache()
            selected = {
                name: parameter
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            }
            inputs = {key: torch.from_numpy(value).unsqueeze(0) for key, value in arrays.items()}
            result = model(**inputs, labels=labels, use_cache=False, return_dict=True)
            result.loss.backward()
            decoder_gradient_guard(model, selected, torch)
            cpu_reference = {
                "loss": float(result.loss),
                "logits": result.logits.detach().clone(),
                "gradients": {
                    name: parameter.grad.detach().clone() for name, parameter in selected.items()
                },
            }
            del inputs, result
            model.zero_grad(set_to_none=True)
            run(lambda: model.to("mps"))
            selected = {
                name: parameter
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            }

        def backward(arrays=arrays, labels=labels, cpu_reference=cpu_reference, selected=selected):
            features = {
                key: torch.from_numpy(value).unsqueeze(0).to(args.device)
                for key, value in arrays.items()
            }
            target = labels.to(args.device)
            result = model(**features, labels=target, use_cache=False, return_dict=True)
            if not torch.isfinite(result.loss):
                raise RuntimeError("winner decoder preflight loss is nonfinite")
            result.loss.backward()
            coverage = decoder_gradient_guard(model, selected, torch)
            loss = float(result.loss)
            parity = {}
            if cpu_reference is not None:
                loss_difference = abs(loss - cpu_reference["loss"])
                logit_difference = float(
                    (result.logits.detach().cpu() - cpu_reference["logits"]).abs().max()
                )
                differences = []
                for parameter_name, parameter in selected.items():
                    expected = cpu_reference["gradients"][parameter_name]
                    actual = parameter.grad.detach().cpu()
                    relative = float((expected - actual).norm()) / max(float(expected.norm()), 1e-6)
                    differences.append({"parameter": parameter_name, "relative_l2_error": relative})
                worst = max(row["relative_l2_error"] for row in differences)
                if (
                    loss_difference > 5e-4
                    or logit_difference > 1e-3
                    or not math.isfinite(worst)
                    or worst > 0.005
                ):
                    raise RuntimeError(
                        f"full winner CPU/Metal parity failed: loss={loss_difference}, "
                        f"logits={logit_difference}, gradients={worst}"
                    )
                parity = {
                    "cpu_loss": cpu_reference["loss"],
                    "loss_absolute_difference": loss_difference,
                    "logit_maximum_absolute_difference": logit_difference,
                    "worst_gradient_relative_l2_error": worst,
                    "gradient_differences": differences,
                }
            del result, features, target
            if args.device == "mps":
                torch.mps.synchronize()
                torch.mps.empty_cache()
            return {"loss": loss, "gradient_coverage": coverage, **parity}

        record = run(backward)
        records.append(
            {
                "case": name,
                "source": row["source"],
                "example_id": row["id"],
                "training_allowed": True,
                "role": "training",
                "frames": sampled_frames(row),
                "target_tokens": labels.shape[1],
                **record,
                **(duty.receipt() if duty else {}),
            }
        )
    del reserve
    report = {
        "schema": "umi-studio-c5-winner-translator-preflight/1",
        "training_scope": args.scope,
        "frozen_unused_parameters": sorted(UNUSED_FEATURE_PARAMETERS)
        if args.scope == "full"
        else [],
        "resource_policy": budget.record if budget else None,
        "model_bundle_sha256": reference["model_bundle_sha256"],
        "checkpoint_sha256": reference["translator_checkpoint_sha256"],
        "reference_registry_sha256": sha256(ROOT / "C5_BASELINE_REFERENCE.json"),
        "probe_source_sha256": sha256(Path(__file__)),
        "device": args.device,
        "parameter_counts": counts,
        "optimizer_moment_memory_reserved_bytes": counts["trainable"] * 8,
        "optimizer_updates": 0,
        "weights_saved": False,
        "miner_modified": False,
        "quality_verified": False,
        "input_policy": "verified_native224_legacy_post_mediapipe_stride2_training_only",
        "records": records,
        **(duty.receipt() if duty else {}),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "preflight_passed": True,
                "device": args.device,
                "parameter_counts": counts,
                "optimizer_updates": 0,
                **(duty.receipt() if duty else {}),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
