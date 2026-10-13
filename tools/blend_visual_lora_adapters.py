"""Blend two projection-matched visual LoRA adapters exactly at the delta level."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from benchmark_fleurs_dev import sha256
from interpolate_visual_adapter import DIRECT_NAMES
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]


def blended_weights(
    first: dict[str, torch.Tensor],
    second: dict[str, torch.Tensor],
    second_weight: float,
    *,
    blend_direct: bool = False,
) -> dict[str, torch.Tensor]:
    """Represent a weighted sum of two equal-rank LoRA deltas at double rank."""
    if not 0 < second_weight < 1 or set(first) != set(second):
        raise ValueError("visual LoRA blend needs matching tensors and fractional weight")
    a_names = {name for name in first if name.endswith(".lora_a.weight")}
    b_names = {name for name in first if name.endswith(".lora_b.weight")}
    if (
        not a_names
        or {name.replace(".lora_a.weight", ".lora_b.weight") for name in a_names}
        != b_names
        or set(first) != DIRECT_NAMES | a_names | b_names
    ):
        raise ValueError("visual LoRA blend topology differs")
    output = {}
    for name in sorted(first):
        left = first[name]
        right = second[name]
        if (
            left.shape != right.shape
            or left.dtype != right.dtype
            or not torch.isfinite(left).all()
            or not torch.isfinite(right).all()
        ):
            raise ValueError(f"visual LoRA blend tensor differs: {name}")
        if name in DIRECT_NAMES:
            if torch.equal(left, right):
                output[name] = left.contiguous().clone()
            elif blend_direct:
                output[name] = torch.lerp(left, right, second_weight).contiguous()
            else:
                raise ValueError(f"visual LoRA blend projection differs: {name}")
        elif name in a_names:
            output[name] = torch.cat((left, right), dim=0).contiguous()
        else:
            output[name] = torch.cat(
                ((1 - second_weight) * left, second_weight * right), dim=1
            ).contiguous()
    return output


def _read_adapter(path: Path) -> tuple[dict[str, str], dict[str, torch.Tensor]]:
    with safe_open(str(path), framework="pt", device="cpu") as source:
        return source.metadata(), {name: source.get_tensor(name) for name in source.keys()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-adapter", type=Path, required=True)
    parser.add_argument("--second-adapter", type=Path, required=True)
    parser.add_argument("--second-weight", type=float, required=True)
    parser.add_argument("--blend-direct", action="store_true")
    parser.add_argument("--plan-output", type=Path, required=True)
    parser.add_argument("--adapter-output", type=Path, required=True)
    args = parser.parse_args()
    if (
        not args.first_adapter.resolve().is_relative_to(ROOT / "checkpoints")
        or not args.second_adapter.resolve().is_relative_to(ROOT / "checkpoints")
        or not args.plan_output.resolve().is_relative_to(ROOT / "runs")
        or not args.adapter_output.resolve().is_relative_to(ROOT / "checkpoints")
        or args.plan_output.exists()
        or args.adapter_output.exists()
    ):
        parser.error("visual LoRA blend needs new workspace outputs and checkpoint inputs")
    first_metadata, first = _read_adapter(args.first_adapter)
    second_metadata, second = _read_adapter(args.second_adapter)
    required = ("schema", "rank", "alpha", "visual_last_layers", "baseline_checkpoint_sha256")
    if (
        any(first_metadata.get(key) != second_metadata.get(key) for key in required)
        or first_metadata.get("schema") != "umi-studio-shubert-visual-lora/1"
    ):
        raise ValueError("visual LoRA blend metadata differs")
    rank = int(first_metadata["rank"])
    alpha = float(first_metadata["alpha"])
    weights = blended_weights(
        first,
        second,
        args.second_weight,
        blend_direct=args.blend_direct,
    )
    plan = {
        "schema": "umi-studio-visual-lora-blend-plan/1",
        "first_adapter": str(args.first_adapter.resolve().relative_to(ROOT)),
        "first_adapter_sha256": sha256(args.first_adapter),
        "second_adapter": str(args.second_adapter.resolve().relative_to(ROOT)),
        "second_adapter_sha256": sha256(args.second_adapter),
        "second_weight": args.second_weight,
        "blend_rule": (
            "Exact weighted sum of LoRA deltas using concatenated rank; "
            "direct visual parameters are linearly interpolated when enabled."
        ),
        "direct_parameter_rule": (
            "linear_interpolation" if args.blend_direct else "must_match"
        ),
        "rank": rank * 2,
        "alpha": alpha * 2,
        "visual_last_layers": int(first_metadata["visual_last_layers"]),
        "baseline_checkpoint_sha256": first_metadata["baseline_checkpoint_sha256"],
        "blender_source_sha256": sha256(Path(__file__)),
        "confirmation_scored": False,
    }
    args.plan_output.parent.mkdir(parents=True, exist_ok=True)
    args.adapter_output.parent.mkdir(parents=True, exist_ok=True)
    plan_temp = args.plan_output.with_suffix(".partial.json")
    adapter_temp = args.adapter_output.with_suffix(".partial.safetensors")
    if plan_temp.exists() or adapter_temp.exists():
        parser.error("unfinished visual LoRA blend output needs inspection")
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
                "recipe_id": "visual-lora-delta-blend-v1",
                "rank": str(plan["rank"]),
                "alpha": str(plan["alpha"]),
                "visual_last_layers": str(plan["visual_last_layers"]),
                "baseline_checkpoint_sha256": plan["baseline_checkpoint_sha256"],
                "first_adapter_sha256": plan["first_adapter_sha256"],
                "second_adapter_sha256": plan["second_adapter_sha256"],
                "second_weight": str(args.second_weight),
                "direct_parameter_rule": plan["direct_parameter_rule"],
            },
        )
        adapter_temp.rename(args.adapter_output)
    finally:
        plan_temp.unlink(missing_ok=True)
        adapter_temp.unlink(missing_ok=True)
    print(f"frozen visual LoRA blend sha256={sha256(args.adapter_output)}")


if __name__ == "__main__":
    main()
