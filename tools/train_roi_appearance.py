"""Qualify and fit fresh small-ROI appearance, without a translation-quality claim."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from benchmark_fleurs_dev import sha256
from c5_certification_budget import CertificationComputeWindow, WinnerDutyCycle
from launch_terra_visual_job import identity, now, write
from probe_tiny_rgb_capacity import compare, gradients, state_digest
from roi_appearance_model import STREAMS, RoiAppearance, feature_loss, pixel_tensor, temporal_gap
from roi_appearance_protocol import (
    DATA,
    RECIPE,
    ROOT,
    checked,
    checked_data,
    decision,
    freeze,
    permutation,
)
from terra_training_recovery import CheckpointRequest, artifact, save_checkpoint
from terra_training_storage import storage
from tiny_rgb_capacity_protocol import verified_comparison
from train_terra_c5_tail import append, gpu_lease


def targets(values, indices=None):
    if indices is None:
        indices = slice(None)
    return {name: torch.from_numpy(values[name][indices].copy()) for name in STREAMS}


def qualify(path, output, plan, data, arrays, duty):
    cpu, gpu = RoiAppearance().eval(), RoiAppearance().eval()
    initial = plan["initialization_sha256"]
    if any(state_digest(m.state_dict()) != initial for m in (cpu, gpu)):
        raise ValueError("fresh ROI proof initialization differs")
    gpu = duty.run(lambda: gpu.to("mps"))
    if state_digest(gpu.state_dict()) != initial:
        raise ValueError("ROI Metal transfer changed fresh weights")
    comparisons = []
    for index in plan["proof_training_record_indices"]:
        row, values = data["records"][index], arrays[index]
        if row["roi_training_allowed"] is not True or row["roi_role"] != "training":
            raise ValueError("ROI proof cannot use diagnostic frames")
        selection = np.array([0, len(values["pixels"]) - 1])
        pixels, target = pixel_tensor(values["pixels"][selection]), targets(values, selection)
        cpu.zero_grad(set_to_none=True)
        predicted = cpu(pixels)
        loss = feature_loss(predicted, target)["loss"]
        loss.backward()
        reference = gradients(cpu)
        expected_loss = float(loss)
        expected_output = torch.cat([predicted[n].detach() for n in STREAMS], dim=-1)
        del predicted, loss

        def metal(
            pixels=pixels,
            target=target,
            reference=reference,
            expected_loss=expected_loss,
            expected_output=expected_output,
        ):
            gpu.zero_grad(set_to_none=True)
            result = gpu(pixels.to("mps"))
            loss = feature_loss(result, target)["loss"]
            loss.backward()
            return compare(
                reference,
                gradients(gpu),
                expected_loss,
                float(loss),
                expected_output,
                torch.cat([result[n].detach().cpu() for n in STREAMS], dim=-1),
            )

        comparison = duty.run(metal)
        record = {
            "record_index": index,
            "cache": row["cache"],
            "sampled_positions": selection.tolist(),
            "comparison": comparison,
        }
        append(output, "comparisons.jsonl", record)
        verified_comparison(comparison, plan["parameter_tensors"])
        comparisons.append(record)
        del reference
        gc.collect()
    if any(state_digest(m.state_dict()) != initial for m in (cpu, gpu)):
        raise ValueError("zero-update ROI proof changed weights")
    checked(path)
    write(
        output / "qualification.json",
        {
            "complete": True,
            "numerics_passed": True,
            "plan_sha256": sha256(path),
            "initialization_sha256": initial,
            "parameter_tensors": plan["parameter_tensors"],
            "comparisons": artifact(output / "comparisons.jsonl"),
            "proof_training_record_indices": plan["proof_training_record_indices"],
            "optimizer_updates": 0,
            "new_Terra_requests": 0,
            "resources": duty.receipt(),
        },
    )
    print(
        json.dumps(
            {
                "stage": "ROI_qualification_complete",
                "cases": len(comparisons),
                "optimizer_updates": 0,
            }
        ),
        flush=True,
    )


def checked_proof(receipt, plan_path, plan):
    result = json.loads(receipt.read_text())
    terminal = json.loads(receipt.with_name("exit.json").read_text())
    worker = json.loads(receipt.with_name("worker.json").read_text())
    comparisons = receipt.with_name("comparisons.jsonl")
    rows = [json.loads(s) for s in comparisons.read_text().splitlines()]
    if (
        result.get("complete") is not True
        or result.get("numerics_passed") is not True
        or result.get("plan_sha256") != sha256(plan_path)
        or result.get("initialization_sha256") != plan["initialization_sha256"]
        or result.get("optimizer_updates") != 0
        or result.get("new_Terra_requests") != 0
        or terminal.get("exit_code") != 0
        or terminal.get("pid") != worker.get("pid")
        or result.get("comparisons") != artifact(comparisons)
        or [r["record_index"] for r in rows] != plan["proof_training_record_indices"]
    ):
        raise ValueError("ROI training requires a complete bound terminal zero-update proof")
    data = json.loads(DATA.read_text())
    for row in rows:
        cache = data["records"][row["record_index"]]
        if row["cache"] != cache["cache"] or row["sampled_positions"] != [0, cache["frames"] - 1]:
            raise ValueError("ROI proof used different real frame inputs")
        verified_comparison(row["comparison"], plan["parameter_tensors"])
    return artifact(receipt)


def evaluate(model, data, arrays, duty, epoch):
    records = []
    model.eval()
    with torch.no_grad():
        for row, values in zip(data["records"], arrays, strict=True):
            parts = {n: [] for n in STREAMS}
            for start in range(0, len(values["pixels"]), RECIPE["batch_frames"]):
                pixels = pixel_tensor(values["pixels"][start : start + RECIPE["batch_frames"]])

                def forward(pixels=pixels):
                    return {n: v.cpu() for n, v in model(pixels.to("mps")).items()}

                for n, value in duty.run(forward).items():
                    parts[n].append(value)
            predicted = {n: torch.cat(v) for n, v in parts.items()}
            target = targets(values)
            measured = feature_loss(predicted, target)
            records.append(
                {
                    "cache": row["cache"],
                    "role": row["roi_role"],
                    "frames": len(values["pixels"]),
                    "loss": float(measured["loss"]),
                    "component_losses": {n: float(v) for n, v in measured["components"].items()},
                    "usable_frames": measured["usable_frames"],
                    "temporal_gap": temporal_gap(predicted, target),
                    "teacher_temporal_gap": temporal_gap(target, target),
                }
            )
    result = {"epoch": epoch, "records": records}
    for role in ("training", "feature_diagnostic"):
        group = [r for r in records if r["role"] == role]
        result[role] = {
            key: float(np.mean([r[key] for r in group]))
            for key in ("loss", "temporal_gap", "teacher_temporal_gap")
        }
    return result


def fit(path, output, plan, data, arrays, duty, qualification):
    proof = checked_proof(qualification, path, plan)
    model = RoiAppearance()
    if state_digest(model.state_dict()) != plan["initialization_sha256"]:
        raise ValueError("ROI training did not start fresh")
    model = duty.run(lambda: model.to("mps"))
    if state_digest(model.state_dict()) != plan["initialization_sha256"]:
        raise ValueError("ROI training transfer changed initialization")
    torch.manual_seed(RECIPE["seed"])
    torch.mps.manual_seed(RECIPE["seed"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=RECIPE["learning_rate"], weight_decay=RECIPE["weight_decay"]
    )
    stop = CheckpointRequest()
    train = [v for v, r in zip(arrays, data["records"], strict=True) if r["roi_training_allowed"]]
    pixels = np.concatenate([v["pixels"] for v in train])
    labels = {n: np.concatenate([v[n] for v in train]) for n in STREAMS}
    if len(pixels) != plan["training_frames_per_epoch"]:
        raise ValueError("ROI complete training frame count differs")
    write(
        output / "run.json",
        {
            "plan_sha256": sha256(path),
            "qualification": proof,
            "new_Terra_requests": 0,
            "frame_exposures_not_clip_exposures": True,
        },
    )
    initial = evaluate(model, data, arrays, duty, 0)
    append(output, "evaluations.jsonl", initial)
    if initial["training"]["teacher_temporal_gap"] <= 0:
        raise ValueError("ROI teacher lacks positive within-clip hand-motion signal")
    exposures = updates = 0
    safe_to_save = True
    cursor = {"epoch": 0, "next_frame_offset": 0}

    def save(recovery):
        storage(plan["visual_parameters"] * 12 + 1024**2, output=output)
        reference = json.loads((ROOT / "C5_BASELINE_REFERENCE.json").read_text())
        checkpoint = save_checkpoint(
            model,
            optimizer,
            output,
            path,
            reference["translator_checkpoint_sha256"],
            exposures,
            updates,
            recovery=recovery,
        )
        write(
            checkpoint.with_name(checkpoint.stem + "-sampler.json"),
            {
                **cursor,
                "frame_exposures": exposures,
                "optimizer_updates": updates,
                "recipe": RECIPE,
                "training_frame_count": len(pixels),
            },
        )
        return artifact(checkpoint)

    try:
        for epoch in range(RECIPE["epochs"]):
            order = permutation(len(pixels), epoch)
            for offset in range(0, len(order), RECIPE["batch_frames"]):
                cursor = {"epoch": epoch, "next_frame_offset": offset}
                if stop.requested:
                    checkpoint = save(True) if updates else None
                    write(
                        output / "checkpoint-stop.json",
                        {
                            "checkpoint": checkpoint,
                            "frame_exposures": exposures,
                            "optimizer_updates": updates,
                        },
                    )
                    return
                storage(output=output)
                indices = order[offset : offset + RECIPE["batch_frames"]]
                batch = pixel_tensor(pixels[indices])
                target = {n: torch.from_numpy(v[indices].copy()) for n, v in labels.items()}
                safe_to_save = False

                def update(batch=batch, target=target):
                    model.train()
                    optimizer.zero_grad(set_to_none=True)
                    result = feature_loss(model(batch.to("mps")), target)
                    result["loss"].backward()
                    if any(
                        p.grad is None or not torch.isfinite(p.grad).all()
                        for p in model.parameters()
                    ):
                        raise RuntimeError("ROI update lacks complete finite visual gradients")
                    norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), RECIPE["clip_gradient_norm"], error_if_nonfinite=True
                    )
                    optimizer.step()
                    torch.mps.synchronize()
                    if any(not torch.isfinite(p).all() for p in model.parameters()):
                        raise RuntimeError("ROI optimizer produced nonfinite weights")
                    return float(result["loss"]), float(norm)

                loss, norm = duty.run(update)
                safe_to_save = True
                updates += 1
                exposures += len(indices)
                cursor = {"epoch": epoch, "next_frame_offset": offset + len(indices)}
                append(
                    output,
                    "steps.jsonl",
                    {
                        "epoch": epoch,
                        "optimizer_update": updates,
                        "frame_exposures": exposures,
                        "batch_frames": len(indices),
                        "loss": loss,
                        "gradient_norm": norm,
                        "finite_gradient_tensors": plan["parameter_tensors"],
                    },
                )
            if epoch + 1 in RECIPE["evaluation_epochs"]:
                measured = evaluate(model, data, arrays, duty, epoch + 1)
                append(output, "evaluations.jsonl", measured)
                print(
                    json.dumps(
                        {
                            "stage": "ROI_epoch",
                            "epoch": epoch + 1,
                            "optimizer_updates": updates,
                            "frame_exposures": exposures,
                            "training": measured["training"],
                            "feature_diagnostic": measured["feature_diagnostic"],
                        }
                    ),
                    flush=True,
                )
        if (
            updates != plan["planned_optimizer_updates"]
            or exposures != plan["planned_frame_exposures"]
        ):
            raise ValueError("ROI final coverage differs from fixed complete-frame recipe")
        checked(path)
        checkpoint = save(False)
        write(
            output / "decision.json",
            {
                "complete": True,
                "plan_sha256": sha256(path),
                "visual": checkpoint,
                "optimizer_updates": updates,
                "frame_exposures": exposures,
                "initial": initial,
                "final": measured,
                **decision(initial, measured),
                "resources": duty.receipt(),
                "new_Terra_requests": 0,
                "C5_selected": True,
            },
        )
    except BaseException as error:
        checkpoint = save(True) if updates and safe_to_save else None
        write(
            output / "failure.json",
            {
                "error": repr(error),
                "optimizer_updates": updates,
                "frame_exposures": exposures,
                "ambiguous_uncommitted_update": not safe_to_save,
                "recovery_checkpoint": checkpoint,
                "automatic_retries": 0,
            },
        )
        raise


def run(path, output, *, qualification_only=False, qualification=None):
    if qualification_only == (qualification is not None):
        raise ValueError("choose zero-update qualification or supply completed qualification")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    plan = checked(path)
    _, data, arrays = checked_data()
    budget = CertificationComputeWindow(ROOT / plan["resource_policy"])
    if (
        budget.duty_fraction() != 0.5
        or budget.cpu_threads() != 2
        or not torch.backends.mps.is_available()
    ):
        raise ValueError("ROI pilot requires native Metal,50% duty,two CPU threads")
    output = output.resolve()
    if output.exists() or not output.is_relative_to(ROOT / "checkpoints"):
        raise ValueError("ROI pilot requires a fresh owned output")
    storage(plan["visual_parameters"] * 12 + 1024**2)
    output.mkdir(parents=True)
    write(output / "worker.json", identity(os.getpid(), [sys.executable, *sys.argv]))
    code = 1
    try:
        with gpu_lease():
            duty = WinnerDutyCycle(torch, budget)
            if qualification_only:
                qualify(path, output, plan, data, arrays, duty)
            else:
                fit(path, output, plan, data, arrays, duty, qualification)
        code = 0
    finally:
        write(output / "exit.json", {"pid": os.getpid(), "exit_code": code, "observed_at": now()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--qualification-only", action="store_true")
    parser.add_argument("--qualification", type=Path)
    args = parser.parse_args()
    if args.freeze:
        freeze(args.freeze)
    elif args.plan and args.output:
        run(
            args.plan,
            args.output,
            qualification_only=args.qualification_only,
            qualification=args.qualification,
        )
    else:
        parser.error("provide --freeze or --plan and --output")


if __name__ == "__main__":
    main()
