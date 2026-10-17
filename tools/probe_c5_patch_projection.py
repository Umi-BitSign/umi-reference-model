"""Isolate the failed C5 patch-weight backward using actual full-backbone cotangents."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from functools import partial
from pathlib import Path

import torch
from benchmark_fleurs_dev import sha256
from c5_patch_projection import LinearPatchProjection
from c5_small_input_vision import SmallInputDino, input_tensor
from launch_terra_visual_job import identity, now, write
from probe_c5_small_input import checked as checked_component
from probe_c5_small_input import crop_case, feature_probe, legacy_models
from probe_tiny_rgb_capacity import compare, state_digest
from terra_training_recovery import artifact, verify_snapshot
from terra_training_storage import storage
from tiny_rgb_capacity_protocol import verified_comparison

ROOT = Path(__file__).resolve().parents[1]
POLICY = {
    "component_training_case_index": 0,
    "numerical_frames": "first and final sampled frame of the complete original training clip",
    "cotangents": "actual CPU full-backbone fixed feature-loss derivatives at each patch output",
    "variants": ["native_contiguous", "native_channels_last", "linear_contiguous"],
    "additional_full_backbone_comparison": (
        "case0 native CPU versus linear CPU and linear Metal; all350 tensors"
    ),
    "CPU_threads": 2,
    "GPU_duty_fraction": 0.5,
    "optimizer_updates": 0,
    "new_Terra_requests": 0,
    "full_component_numerics_qualified": False,
    "promotion_authorized": False,
}


def specification(component_path):
    from tiny_rgb_full_run import runtime_binding

    component = checked_component(component_path)
    output = ROOT / "checkpoints" / component_path.parent.name
    worker = json.loads((output / "worker.json").read_text())
    terminal = json.loads((output / "exit.json").read_text())
    comparison = json.loads((output / "comparison-00.json").read_text())
    gradients = comparison["comparison"]["gradient_differences"]
    failed = {r["parameter"] for r in gradients if r["relative_l2_error"] > 0.005}
    if (
        terminal.get("pid") != worker.get("pid")
        or terminal.get("exit_code") != 1
        or failed != {"hands.patch_embed.proj.weight", "face.patch_embed.proj.weight"}
        or len(gradients) != 350
        or any(not r["finite"] for r in gradients)
    ):
        raise ValueError("patch isolation requires the actual two-weight component failure")
    sources = dict(component["source_sha256"])
    for name in ("probe_c5_patch_projection.py", "c5_patch_projection.py"):
        sources[f"tools/{name}"] = sha256(ROOT / "tools" / name)
    return {
        "schema": "umi-studio-C5-patch-backward-isolation/1",
        "component_plan": artifact(component_path),
        "actual_failure": artifact(output / "comparison-00.json"),
        "terminal": artifact(output / "exit.json"),
        "policy": POLICY,
        "source_sha256": sources,
        "assets": component["assets"],
        "runtime_binding": runtime_binding(),
        "resource_policy": component["resource_policy"],
        "resource_policy_sha256": component["resource_policy_sha256"],
    }


def checked(path):
    plan = json.loads(path.read_text())
    if plan != specification(ROOT / plan["component_plan"]["path"]):
        raise ValueError("patch isolation changed its actual failure, sources or fixed variants")
    verify_snapshot(path, plan)
    return plan


def freeze(path, component_path):
    from c5_experiment_sources import archive_sources

    path = path.resolve()
    if path.parent.exists() or not path.is_relative_to(ROOT / "runs"):
        raise ValueError("patch isolation requires a fresh owned plan")
    storage()
    plan = specification(component_path.resolve())
    path.parent.mkdir(parents=True)
    write(path, plan)
    archive_sources(path, path.with_name("source-snapshot"))
    print(json.dumps({"plan": artifact(path)}), flush=True)


def isolated(original, pixels, cotangent, variant, device):
    if variant not in POLICY["variants"]:
        raise ValueError("unknown predeclared patch isolation variant")
    model = copy.deepcopy(original)
    if variant == "linear_contiguous":
        model = LinearPatchProjection(model)
    model = model.to(device)
    model.zero_grad(set_to_none=True)
    pixels, cotangent = pixels.to(device), cotangent.to(device)
    pixels = pixels.contiguous(
        memory_format=(
            torch.channels_last if variant == "native_channels_last" else torch.contiguous_format
        )
    )
    output = model(pixels)
    loss = (output * cotangent).sum()
    loss.backward()
    return (
        float(loss.detach().cpu()),
        output.detach().cpu(),
        {
            name: parameter.grad.detach().cpu().clone()
            for name, parameter in model.named_parameters()
        },
    )


def run(path, output):
    import cv2
    from c5_certification_budget import CertificationComputeWindow, WinnerDutyCycle
    from train_terra_c5_tail import gpu_lease

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(2)
    if os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS") != "threads;2":
        raise ValueError("patch isolation requires the two-thread FFmpeg policy")
    plan = checked(path)
    output = output.resolve()
    if output.exists() or not output.is_relative_to(ROOT / "checkpoints"):
        raise ValueError("patch isolation requires a fresh owned output")
    budget = CertificationComputeWindow(ROOT / plan["resource_policy"])
    if (
        budget.duty_fraction() != 0.5
        or budget.cpu_threads() != 2
        or not torch.backends.mps.is_available()
    ):
        raise ValueError("patch isolation requires native Metal50% and CPU2")
    storage()
    output.mkdir(parents=True)
    write(output / "worker.json", identity(os.getpid(), [sys.executable, *sys.argv]))
    code = 1
    try:
        component = checked_component(ROOT / plan["component_plan"]["path"])
        hands, face = legacy_models(component)
        model = SmallInputDino(hands.model, face.model).eval()
        initial = state_digest(model.state_dict())
        crops = crop_case(component["records"][0], 112)
        pixels = input_tensor(crops[[0, len(crops) - 1]])
        captures, hooks = {}, []

        def capture(name):
            def hook(module, arguments, value):
                value.retain_grad()
                captures[name] = (arguments[0], value)

            return hook

        for name in ("hands", "face"):
            hooks.append(getattr(model, name).patch_embed.proj.register_forward_hook(capture(name)))
        full_loss, full_output, full_gradients = feature_probe(model, pixels)
        for hook in hooks:
            hook.remove()
        duty, reports = WinnerDutyCycle(torch, budget), []
        with gpu_lease():
            for name in ("hands", "face"):
                original = getattr(model, name).patch_embed.proj
                actual_input, actual_output = captures[name]
                cotangent = actual_output.grad.detach().clone()
                reference = isolated(
                    original, actual_input.detach(), cotangent, "native_contiguous", "cpu"
                )
                errors = {}
                for parameter in ("weight", "bias"):
                    full = full_gradients[f"{name}.patch_embed.proj.{parameter}"]
                    errors[parameter] = float((reference[2][parameter] - full).norm() / full.norm())
                if any(error > 1e-6 for error in errors.values()):
                    raise ValueError(
                        "isolated cotangent does not reproduce actual full-backbone gradients"
                    )
                for variant in POLICY["variants"]:
                    cpu = isolated(original, actual_input.detach(), cotangent, variant, "cpu")
                    cpu_comparison = compare(
                        reference[2], cpu[2], reference[0], cpu[0], reference[1], cpu[1]
                    )
                    verified_comparison(cpu_comparison, 2, cpu_equivalence=True)
                    metal = duty.run(
                        partial(
                            isolated, original, actual_input.detach(), cotangent, variant, "mps"
                        )
                    )
                    measured = compare(
                        reference[2], metal[2], reference[0], metal[0], reference[1], metal[1]
                    )
                    try:
                        verified_comparison(measured, 2)
                        passed = True
                    except ValueError:
                        passed = False
                    report = {
                        "backbone": name,
                        "variant": variant,
                        "CPU_native_comparison": cpu_comparison,
                        "CPU_Metal_comparison": measured,
                        "isolated_original_gradient_errors": errors,
                        "input_shape": list(actual_input.shape),
                        "input_stride": list(actual_input.stride()),
                        "cotangent_stride": list(cotangent.stride()),
                        "passes_unchanged_limits": passed,
                    }
                    write(output / f"{name}-{variant}.json", report)
                    reports.append(report)
                    print(
                        json.dumps({"backbone": name, "variant": variant, "passed": passed}),
                        flush=True,
                    )
            candidate = copy.deepcopy(model)
            for name in ("hands", "face"):
                patch = getattr(candidate, name).patch_embed
                patch.proj = LinearPatchProjection(patch.proj)
                patch.proj.register_forward_pre_hook(candidate._first_layer)
            cpu_loss, cpu_output, cpu_gradients = feature_probe(candidate, pixels)
            cpu_equivalence = compare(
                full_gradients, cpu_gradients, full_loss, cpu_loss, full_output, cpu_output
            )
            write(output / "full-backbone-CPU-equivalence.json", cpu_equivalence)
            verified_comparison(cpu_equivalence, 350, cpu_equivalence=True)
            duty.run(lambda: candidate.to("mps"))
            metal_loss, metal_output, metal_gradients = duty.run(
                lambda: feature_probe(candidate, pixels.to("mps"))
            )
            full_comparison = compare(
                full_gradients, metal_gradients, full_loss, metal_loss, full_output, metal_output
            )
            write(output / "full-backbone-CPU-Metal.json", full_comparison)
            try:
                verified_comparison(full_comparison, 350)
                corrected_case_passes = True
            except ValueError:
                corrected_case_passes = False
            if duty.run(lambda: state_digest(candidate.state_dict())) != initial:
                raise ValueError("patch isolation changed the candidate's original visual values")
        if state_digest(model.state_dict()) != initial:
            raise ValueError("patch isolation changed an original visual value")
        checked(path)
        write(
            output / "diagnosis.json",
            {
                "complete": True,
                "plan_sha256": sha256(path),
                "comparisons": reports,
                "pretrained_values_unchanged": True,
                "pretrained_state_sha256": initial,
                "resources": duty.receipt(),
                "optimizer_updates": 0,
                "new_Terra_requests": 0,
                "full_component_numerics_qualified": False,
                "corrected_first_case_passes": corrected_case_passes,
                "promotion_authorized": False,
            },
        )
        code = 0
    except BaseException as error:
        write(output / "failure.json", {"error": str(error), "automatic_retries": 0})
        raise
    finally:
        write(output / "exit.json", {"pid": os.getpid(), "exit_code": code, "observed_at": now()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("freeze", "component-plan", "plan", "output"):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args()
    if args.freeze:
        if not args.component_plan:
            parser.error("--freeze requires --component-plan")
        freeze(args.freeze, args.component_plan)
    if args.plan and args.output:
        run(args.plan, args.output)
    elif args.plan or args.output or not args.freeze:
        parser.error("choose --freeze or --plan/--output")
