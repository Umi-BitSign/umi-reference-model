"""Evaluate projection-frozen STEM checkpoints on the frozen open tuning sets."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import torch
from benchmark_fleurs_dev import sha256
from c5_retention import ALL_SOURCE_IDS, RETENTION_RECIPES
from c5_scale_learning_rates import SPLIT_RATES, split_recipe
from freeze_c5_stem_continuation_plan import ROOT
from freeze_c5_stem_projection_frozen_plan import RECIPE as PROJECTION_FROZEN_RECIPE
from freeze_c5_stem_scale_plan import RECIPE as SCALE_RECIPE
from interpolate_visual_adapter import DIRECT_NAMES
from run_c5_stem_tuning import CHECKPOINT, LAYOUT, RUNTIME, read_jsonl, summarize_candidate
from safetensors import safe_open


def select_step(summaries: dict[str, dict], screen: dict) -> int | None:
    passing = []
    for label, row in summaries.items():
        h_delta = row["how2sign"]["candidate_minus_baseline_mean_wer"]
        f_delta = row["fleurs"]["candidate_minus_baseline_mean_wer"]
        improved_signers = sum(
            value["candidate_mean_wer"] < value["baseline_mean_wer"]
            for value in row["how2sign"]["signer_means"].values()
        )
        row["equal_domain_mean_wer_change"] = (h_delta + f_delta) / 2
        row["how2sign_signers_improved"] = improved_signers
        row["passes_screen"] = (
            h_delta <= screen["how2sign_max_mean_wer_change"]
            and f_delta <= screen["fleurs_max_mean_wer_change"]
            and row["equal_domain_mean_wer_change"]
            <= screen["equal_domain_max_mean_wer_change"]
            and improved_signers >= screen["minimum_improved_how2sign_signers"]
        )
        if row["passes_screen"]:
            passing.append(int(label))
    return (
        min(
            passing,
            key=lambda step: (summaries[str(step)]["equal_domain_mean_wer_change"], step),
        )
        if passing
        else None
    )


def direct_parameters(path: Path) -> dict[str, torch.Tensor]:
    with safe_open(str(path), framework="pt", device="cpu") as source:
        return {name: source.get_tensor(name) for name in DIRECT_NAMES}


def tuning_config(plan: dict) -> tuple[dict, str]:
    configs = {
        "umi-studio-c5-stem-projection-frozen-plan/1": (
            PROJECTION_FROZEN_RECIPE,
            "umi-studio-c5-stem-projection-frozen-tuning-decision/1",
        ),
        "umi-studio-c5-stem-scale-plan/1": (
            SCALE_RECIPE,
            "umi-studio-c5-stem-scale-tuning-decision/1",
        ),
    }
    if plan.get("schema") == "umi-studio-c5-frozen-replay-plan/1":
        recipe = plan.get("recipe", {})
        if recipe.get("id") not in {
            "c5-signer-recovery-projection-frozen-v1",
            "c5-signer-recovery-projection-frozen-v2",
            "c5-signer-recovery-projection-frozen-v3",
            "c5-signer3-recovery-projection-frozen-v1",
            "c5-expanded-visual-lora-v1",
            "c5-openasl-diversity-projection-frozen-v1",
            "c5-openasl-diversity-projection-frozen-v2",
            "c5-openasl-bbox-projection-frozen-v1",
            "c5-how2sign-precut-expansion-projection-frozen-v1",
            "c5-how2sign-precut-expansion-projection-frozen-v2",
            "c5-how2sign-precut-expansion-projection-frozen-v3",
            "c5-how2sign-precut-expansion-scale-v1",
            "c5-how2sign-precut-expansion-scale-v2",
            "c5-openasl-scale-projection-frozen-v1",
            "c5-openasl-scale-fused-lora-fixed-v1",
            "c5-openasl-scale-fused-lora-high-lr-v1",
            "c5-openasl-scale-fused-lora-split-visual-fast-v1",
            "c5-openasl-scale-fused-lora-split-decoder-fast-v1",
            "c5-all-source-fused-lora-v1",
            "c5-all-source-fused-lora-retention-v1",
            "c5-all-source-fused-lora-retention-coverage-v1",
        }:
            raise ValueError("projection-frozen tuning replay recipe differs")
        if recipe.get("id") in ALL_SOURCE_IDS:
            from c5_all_source_training import RECIPE as ALL_RECIPE
            expected_recipe = RETENTION_RECIPES.get(recipe["id"], ALL_RECIPE)
            if recipe != expected_recipe:
                raise ValueError("frozen full-corpus recipe changed")
            output_schema = "umi-studio-c5-all-source-tuning-decision/1"
        elif recipe.get("id") in {
            "c5-openasl-scale-projection-frozen-v1",
            "c5-openasl-scale-fused-lora-fixed-v1",
            "c5-openasl-scale-fused-lora-high-lr-v1",
            "c5-openasl-scale-fused-lora-split-visual-fast-v1",
            "c5-openasl-scale-fused-lora-split-decoder-fast-v1",
        }:
            from freeze_c5_openasl_scale_protocol import RECIPE as OPENASL_SCALE_RECIPE

            expected_recipe = {**OPENASL_SCALE_RECIPE, "id": recipe["id"]}
            if recipe["id"] in SPLIT_RATES:
                expected_recipe = split_recipe(OPENASL_SCALE_RECIPE, recipe["id"])
            if recipe["id"] == "c5-openasl-scale-fused-lora-high-lr-v1":
                expected_recipe["learning_rate"] = 1e-4
            if recipe != expected_recipe:
                raise ValueError("frozen OpenASL scale tuning recipe changed")
            output_schema = "umi-studio-c5-openasl-scale-tuning-decision/1"
        elif recipe.get("id", "").startswith("c5-openasl-"):
            output_schema = "umi-studio-c5-openasl-pilot-tuning-decision/1"
        elif recipe.get("id", "").startswith("c5-how2sign-precut-expansion-"):
            output_schema = "umi-studio-c5-how2sign-expansion-tuning-decision/1"
        else:
            output_schema = "umi-studio-c5-signer-recovery-tuning-decision/1"
        return recipe, output_schema
    try:
        recipe, output_schema = configs[plan["schema"]]
    except (KeyError, TypeError) as error:
        raise ValueError("projection-frozen tuning plan schema differs") from error
    if plan.get("recipe") != recipe:
        raise ValueError("projection-frozen tuning recipe differs")
    return recipe, output_schema


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        not args.plan.is_file()
        or not args.checkpoint_dir.is_dir()
        or args.output.exists()
        or not args.output.resolve().is_relative_to((ROOT / "runs").resolve())
    ):
        parser.error("projection-frozen tuning needs a plan, checkpoints and new output")
    plan = json.loads(args.plan.read_text())
    recipe, output_schema = tuning_config(plan)
    confirmation_policy_valid = "Do not evaluate" in plan.get("confirmation_policy", "")
    if recipe.get("id") in {
        "c5-openasl-scale-projection-frozen-v1",
        "c5-openasl-scale-fused-lora-fixed-v1",
        "c5-openasl-scale-fused-lora-high-lr-v1",
        "c5-openasl-scale-fused-lora-split-visual-fast-v1",
        "c5-openasl-scale-fused-lora-split-decoder-fast-v1",
        "c5-all-source-fused-lora-v1",
        "c5-all-source-fused-lora-retention-v1",
        "c5-all-source-fused-lora-retention-coverage-v1",
    }:
        partition_path = ROOT / plan["scale_partition"]
        partition = json.loads(partition_path.read_text())
        confirmation_policy_valid = (
            sha256(partition_path) == plan["scale_partition_sha256"]
            and plan.get("confirmation_policy") == partition.get("confirmation_policy")
            and "Do not decode sealed_confirmation_records" in plan["confirmation_policy"]
        )
    if (recipe.get("id", "").startswith("c5-openasl-scale-fused-lora-") or recipe.get("id") in ALL_SOURCE_IDS) and (
        plan.get("adapter_implementation_sha256") != sha256(ROOT / "tools/shubert_lora.py")
    ):
        raise ValueError("fixed visual LoRA evaluation implementation changed")
    starting_adapter = ROOT / plan.get("starting_adapter", "")
    if (
        not starting_adapter.is_file()
        or not starting_adapter.resolve().is_relative_to((ROOT / "checkpoints").resolve())
        or plan.get("starting_adapter_sha256") != sha256(starting_adapter)
        or not confirmation_policy_valid
    ):
        raise ValueError("projection-frozen tuning plan differs")
    checkpoint_rows = read_jsonl(args.checkpoint_dir / "checkpoint-index.jsonl")
    by_step = {row["example_step"]: row for row in checkpoint_rows}
    if set(by_step) != set(recipe["checkpoint_after_examples"]):
        raise ValueError("projection-frozen checkpoints are incomplete")
    baseline_rows = {
        domain: read_jsonl(ROOT / plan["open_tuning"][domain]["results"])
        for domain in LAYOUT
    }
    for domain in LAYOUT:
        if sha256(ROOT / plan["open_tuning"][domain]["results"]) != plan[
            "open_tuning"
        ][domain]["results_sha256"]:
            raise ValueError(f"open {domain} tuning result changed")
    frozen_direct = direct_parameters(starting_adapter)
    summaries = {}
    result_hashes = {}
    for step in recipe["checkpoint_after_examples"]:
        adapter = args.checkpoint_dir / f"visual-lora-after-{step:06d}.safetensors"
        if sha256(adapter) != by_step[step]["checkpoint_sha256"]:
            raise ValueError(f"projection-frozen checkpoint changed: {step}")
        if any(
            not torch.equal(value, direct_parameters(adapter)[name])
            for name, value in frozen_direct.items()
        ):
            raise ValueError(f"direct projection changed at step {step}")
        summaries[str(step)] = {}
        result_hashes[str(step)] = {}
        for domain, layout in LAYOUT.items():
            results = args.output.parent / f"tuning-{domain}-{step:06d}.jsonl"
            subprocess.run(
                [
                    "nice",
                    "-n",
                    "20",
                    str(RUNTIME),
                    "tools/evaluate_fleurs_cache.py",
                    "--index",
                    str(layout["index"]),
                    "--cache-dir",
                    str(layout["cache"]),
                    "--split",
                    str(layout["split"]),
                    "--baseline",
                    str(ROOT / "artifacts/baseline-6fe8df59"),
                    "--checkpoint",
                    str(CHECKPOINT),
                    "--adapter",
                    str(adapter),
                    "--results",
                    str(results),
                    "--miner-maintenance-mode",
                    "--device",
                    "cpu",
                    "--use-stream-features",
                    "--temporal-stride",
                    "2",
                    "--limit",
                    str(layout["examples"]),
                ],
                cwd=ROOT,
                check=True,
                stdout=subprocess.DEVNULL,
            )
            candidate = read_jsonl(results)
            if {row.get("adapter_sha256") for row in candidate} != {sha256(adapter)}:
                raise ValueError(f"projection-frozen result differs: {step} {domain}")
            summaries[str(step)][domain] = summarize_candidate(
                baseline_rows[domain], candidate
            )
            result_hashes[str(step)][domain] = sha256(results)
    selected = select_step(summaries, plan["screen"])
    result = {
        "schema": output_schema,
        "plan_sha256": sha256(args.plan),
        "starting_adapter_sha256": sha256(starting_adapter),
        "screen": plan["screen"],
        "checkpoints": summaries,
        "result_sha256": result_hashes,
        "selected_step": selected,
        "decision": "collect_fresh_heldout" if selected else "retain_confirmed_candidate",
        "prior_confirmation_reused": False,
        "confirmation_variants_evaluated": 0,
        "direct_parameters_preserved": True,
        "miner_changed": False,
    }
    temporary = args.output.with_suffix(".partial.json")
    try:
        with temporary.open("x", encoding="utf-8") as output:
            json.dump(result, output, sort_keys=True, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        temporary.rename(args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(result["decision"])


if __name__ == "__main__":
    main()
