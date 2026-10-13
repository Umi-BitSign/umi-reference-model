"""Freeze one half-strength interpolation of the trained visual adapter."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from benchmark_fleurs_dev import miner_revision, sha256
from freeze_visual_lora_plan import RECIPE, TUNING_GATE

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "artifacts/baseline-6fe8df59"
STRENGTH = 0.5
DIRECT_NAMES = {
    "encoder.adapter.layer_weights",
    "encoder.adapter.final_layer.weight",
    "encoder.adapter.final_layer.bias",
}


def interpolated_weights(
    trained: dict[str, torch.Tensor],
    baseline: dict[str, torch.Tensor],
    strength: float,
) -> dict[str, torch.Tensor]:
    if not 0 < strength < 1 or not DIRECT_NAMES.issubset(trained):
        raise ValueError("visual interpolation needs a fractional strength and direct weights")
    a_names = {name for name in trained if name.endswith(".lora_a.weight")}
    b_names = {name for name in trained if name.endswith(".lora_b.weight")}
    if (
        not a_names
        or {name.replace(".lora_a.weight", ".lora_b.weight") for name in a_names}
        != b_names
        or set(trained) != DIRECT_NAMES | a_names | b_names
    ):
        raise ValueError("visual interpolation adapter topology differs")
    output = {}
    for name, value in trained.items():
        if not torch.isfinite(value).all():
            raise ValueError(f"visual interpolation has a non-finite tensor: {name}")
        if name in DIRECT_NAMES:
            prior = baseline[name]
            if (
                prior.shape != value.shape
                or prior.dtype != value.dtype
                or not torch.isfinite(prior).all()
            ):
                raise ValueError(f"visual baseline parameter differs: {name}")
            output[name] = torch.lerp(prior, value, strength).contiguous()
        elif name in b_names:
            output[name] = (value * strength).contiguous()
        else:
            output[name] = value.contiguous().clone()
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-plan", type=Path, required=True)
    parser.add_argument("--source-adapter", type=Path, required=True)
    parser.add_argument("--plan-output", type=Path, required=True)
    parser.add_argument("--adapter-output", type=Path, required=True)
    parser.add_argument("--miner-health-url", required=True)
    args = parser.parse_args()
    if (
        not args.source_plan.resolve().is_relative_to(ROOT / "runs")
        or not args.source_adapter.resolve().is_relative_to(ROOT / "checkpoints")
        or not args.plan_output.resolve().is_relative_to(ROOT / "runs")
        or not args.adapter_output.resolve().is_relative_to(ROOT / "checkpoints")
        or args.plan_output.exists()
        or args.adapter_output.exists()
    ):
        parser.error("visual interpolation needs new workspace plan and adapter outputs")
    source_plan = json.loads(args.source_plan.read_text())
    source_plan_hash = sha256(args.source_plan)
    baseline_checkpoint = BASELINE / "models/checkpoint-11625/model.safetensors"
    baseline_hash = sha256(baseline_checkpoint)
    if (
        source_plan.get("schema") != "umi-studio-visual-lora-plan/1"
        or source_plan.get("recipe") != RECIPE
        or source_plan.get("tuning_gate") != TUNING_GATE
        or source_plan.get("baseline_checkpoint_sha256") != baseline_hash
        or source_plan.get("trainer_source_sha256")
        != sha256(ROOT / "tools/train_visual_lora.py")
    ):
        raise ValueError("visual interpolation source plan differs from the frozen run")
    with safe_open(str(args.source_adapter), framework="pt", device="cpu") as source:
        metadata = source.metadata()
        trained = {name: source.get_tensor(name) for name in source.keys()}
    if (
        metadata.get("schema") != "umi-studio-shubert-visual-lora/1"
        or metadata.get("plan_sha256") != source_plan_hash
        or metadata.get("recipe_id") != RECIPE["id"]
        or metadata.get("example_step") != "4096"
        or metadata.get("baseline_checkpoint_sha256") != baseline_hash
        or metadata.get("trainer_source_sha256") != source_plan["trainer_source_sha256"]
        or metadata.get("adapter_source_sha256")
        != source_plan["visual_adapter_source_sha256"]
    ):
        raise ValueError("visual interpolation source adapter differs from the completed run")
    with safe_open(str(baseline_checkpoint), framework="pt", device="cpu") as original:
        baseline = {name: original.get_tensor(name) for name in DIRECT_NAMES}
    weights = interpolated_weights(trained, baseline, STRENGTH)
    revision = miner_revision(args.miner_health_url, source_plan["miner_model_revision"])
    plan = {
        "schema": "umi-studio-visual-adapter-interpolation-plan/1",
        "parent_plan": str(args.source_plan.resolve().relative_to(ROOT)),
        "parent_plan_sha256": source_plan_hash,
        "source_adapter": str(args.source_adapter.resolve().relative_to(ROOT)),
        "source_adapter_sha256": sha256(args.source_adapter),
        "source_example_step": 4096,
        "baseline_checkpoint_sha256": baseline_hash,
        "strength": STRENGTH,
        "interpolation_rule": (
            "Keep trained LoRA A, multiply trained LoRA B by 0.5, and linearly "
            "interpolate visual layer weights and final projection from baseline to trained."
        ),
        "tuning": source_plan["tuning"],
        "tuning_gate": source_plan["tuning_gate"],
        "interpolator_source_sha256": sha256(Path(__file__)),
        "miner_model_revision": revision,
        "confirmation_scored": False,
    }
    args.plan_output.parent.mkdir(parents=True, exist_ok=True)
    args.adapter_output.parent.mkdir(parents=True, exist_ok=True)
    plan_temp = args.plan_output.with_suffix(".partial.json")
    adapter_temp = args.adapter_output.with_suffix(".partial.safetensors")
    if plan_temp.exists() or adapter_temp.exists():
        parser.error("unfinished visual interpolation output needs inspection")
    try:
        with plan_temp.open("x", encoding="utf-8") as output:
            json.dump(plan, output, sort_keys=True, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        plan_temp.rename(args.plan_output)
        save_file(
            weights,
            str(adapter_temp),
            metadata={
                "schema": "umi-studio-shubert-visual-lora/1",
                "plan_sha256": sha256(args.plan_output),
                "recipe_id": "visual-lora-midpoint-v1",
                "rank": str(RECIPE["rank"]),
                "alpha": str(RECIPE["alpha"]),
                "visual_last_layers": str(RECIPE["visual_last_layers"]),
                "baseline_checkpoint_sha256": baseline_hash,
                "source_adapter_sha256": plan["source_adapter_sha256"],
                "interpolation_strength": str(STRENGTH),
            },
        )
        adapter_temp.rename(args.adapter_output)
    finally:
        plan_temp.unlink(missing_ok=True)
        adapter_temp.unlink(missing_ok=True)
    miner_revision(args.miner_health_url, revision)
    print(f"frozen midpoint adapter sha256={sha256(args.adapter_output)}")


if __name__ == "__main__":
    main()
