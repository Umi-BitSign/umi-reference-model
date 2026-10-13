"""Freeze a second visual-LoRA run with baseline projection weights held fixed."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from safetensors import safe_open

from benchmark_fleurs_dev import miner_revision, sha256
from freeze_visual_lora_plan import RECIPE as PARENT_RECIPE
from freeze_visual_lora_plan import TUNING_GATE, freeze_plan

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "artifacts/baseline-6fe8df59"
RECIPE = {
    **PARENT_RECIPE,
    "id": "visual-lora-projection-frozen-v1",
    "learning_rate": 0.00005,
    "projection_training": "frozen_baseline",
    "starting_adapter": "projection_frozen_ablation_of_stage1_final",
}


def build_plan(
    parent_path: Path,
    ablation_path: Path,
    decision_path: Path,
    start_adapter: Path,
    health_url: str,
) -> dict:
    parent = json.loads(parent_path.read_text())
    reproduced = json.loads(json.dumps(freeze_plan(ROOT, BASELINE, health_url), sort_keys=True))
    if parent != reproduced or parent.get("recipe") != PARENT_RECIPE:
        raise ValueError("projection-frozen run needs the unchanged source-complete parent plan")
    ablation = json.loads(ablation_path.read_text())
    decision = json.loads(decision_path.read_text())
    with safe_open(str(start_adapter), framework="pt", device="cpu") as source:
        metadata = source.metadata()
    if (
        ablation.get("schema") != "umi-studio-visual-projection-ablation-plan/1"
        or ablation.get("parent_plan_sha256") != sha256(parent_path)
        or ablation.get("tuning_gate") != TUNING_GATE
        or ablation.get("baseline_checkpoint_sha256")
        != parent["baseline_checkpoint_sha256"]
        or decision.get("schema") != "umi-studio-visual-projection-ablation-decision/1"
        or decision.get("plan_sha256") != sha256(ablation_path)
        or decision.get("candidate_adapter_sha256") != sha256(start_adapter)
        or decision.get("decision") != "retain_frozen_baseline"
        or decision.get("confirmation_evaluated") is not False
        or metadata.get("schema") != "umi-studio-shubert-visual-lora/1"
        or metadata.get("plan_sha256") != sha256(ablation_path)
        or metadata.get("recipe_id") != ablation["recipe_id"]
        or metadata.get("ablation") != "baseline_visual_projection"
        or metadata.get("baseline_checkpoint_sha256")
        != parent["baseline_checkpoint_sha256"]
    ):
        raise ValueError("projection-frozen starting adapter lacks frozen ablation provenance")
    revision = miner_revision(health_url, parent["miner_model_revision"])
    return {
        "schema": "umi-studio-visual-lora-projection-frozen-plan/1",
        "parent_plan": str(parent_path.resolve().relative_to(ROOT)),
        "parent_plan_sha256": sha256(parent_path),
        "ablation_plan": str(ablation_path.resolve().relative_to(ROOT)),
        "ablation_plan_sha256": sha256(ablation_path),
        "ablation_decision": str(decision_path.resolve().relative_to(ROOT)),
        "ablation_decision_sha256": sha256(decision_path),
        "starting_adapter": str(start_adapter.resolve().relative_to(ROOT)),
        "starting_adapter_sha256": sha256(start_adapter),
        "baseline_checkpoint_sha256": parent["baseline_checkpoint_sha256"],
        "sources": parent["sources"],
        "tuning": parent["tuning"],
        "tuning_gate": parent["tuning_gate"],
        "stage_one_schedule_sha256": parent["stage_one_schedule_sha256"],
        "stage_one_source_counts": parent["stage_one_source_counts"],
        "stage_one_how2sign_signer_counts": parent["stage_one_how2sign_signer_counts"],
        "seed": parent["seed"],
        "recipe": RECIPE,
        "adapter_source_sha256": parent["visual_adapter_source_sha256"],
        "trainer_source_sha256": sha256(ROOT / "tools/train_projection_frozen_lora.py"),
        "plan_source_sha256": sha256(Path(__file__)),
        "miner_model_revision": revision,
        "confirmation_scored": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-plan", type=Path, required=True)
    parser.add_argument("--ablation-plan", type=Path, required=True)
    parser.add_argument("--ablation-decision", type=Path, required=True)
    parser.add_argument("--starting-adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--miner-health-url", required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(ROOT / "runs"):
        parser.error("projection-frozen plan needs a new run-ledger output")
    plan = build_plan(
        args.parent_plan, args.ablation_plan, args.ablation_decision,
        args.starting_adapter, args.miner_health_url,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".partial.json")
    if temporary.exists():
        parser.error("unfinished projection-frozen plan needs inspection")
    try:
        with temporary.open("x", encoding="utf-8") as output:
            json.dump(plan, output, sort_keys=True, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        temporary.rename(args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"frozen projection-frozen plan sha256={sha256(args.output)}")


if __name__ == "__main__":
    main()
