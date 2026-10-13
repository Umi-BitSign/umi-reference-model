"""Freeze a full replay after expanding visual LoRA from four to twelve layers."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

from benchmark_fleurs_dev import sha256
from freeze_c5_stem_scale_plan import ROOT

SOURCE_PLAN = ROOT / "runs/c5-stem-scale-v1/plan.json"
SOURCE_SCHEDULE = SOURCE_PLAN.with_name("schedule.jsonl")
PARENT_DECISION = ROOT / "runs/c5-signer-recovery-v2/tuning-decision.json"
STARTING_ADAPTER = (
    ROOT / "checkpoints/c5-signer-recovery-v2/visual-lora-after-000512.safetensors"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() or not args.output_dir.resolve().is_relative_to(
        (ROOT / "runs").resolve()
    ):
        parser.error("expanded visual plan needs a new run-ledger directory")
    source = json.loads(SOURCE_PLAN.read_text())
    parent = json.loads(PARENT_DECISION.read_text())
    checkpoint_rows = [
        json.loads(line)
        for line in (
            ROOT / "checkpoints/c5-signer-recovery-v2/checkpoint-index.jsonl"
        ).read_text().splitlines()
    ]
    expected_hash = next(
        row["checkpoint_sha256"] for row in checkpoint_rows if row["example_step"] == 512
    )
    if (
        source.get("schema") != "umi-studio-c5-stem-scale-plan/1"
        or source.get("recipe", {}).get("visual_last_layers") != 4
        or source.get("recipe", {}).get("train_examples") != 3_112
        or parent.get("decision") != "retain_confirmed_candidate"
        or parent["checkpoints"]["512"]["fleurs"][
            "candidate_minus_baseline_mean_wer"
        ]
        >= 0
        or parent["checkpoints"]["512"]["how2sign_signers_improved"] != 2
        or sha256(STARTING_ADAPTER) != expected_hash
    ):
        raise ValueError("expanded visual starting point is not frozen")
    plan = copy.deepcopy(source)
    plan["schema"] = "umi-studio-c5-frozen-replay-plan/1"
    plan["objective"] = "test early visual-layer capacity on the complete combined replay"
    plan["seed"] = 49
    plan["starting_adapter"] = str(STARTING_ADAPTER.relative_to(ROOT))
    plan["starting_adapter_sha256"] = sha256(STARTING_ADAPTER)
    plan["starting_visual_last_layers"] = 4
    plan["recipe"]["id"] = "c5-expanded-visual-lora-v1"
    plan["recipe"]["visual_last_layers"] = 12
    plan["recipe"]["learning_rate"] = 2e-6
    plan["parent_decision"] = str(PARENT_DECISION.relative_to(ROOT))
    plan["parent_decision_sha256"] = sha256(PARENT_DECISION)
    plan["source_plan_sha256"] = sha256(SOURCE_PLAN)
    plan["trainer_source_sha256"] = sha256(ROOT / "tools/train_c5_frozen_replay.py")
    plan["plan_source_sha256"] = sha256(Path(__file__))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    schedule_path = args.output_dir / "schedule.jsonl"
    with schedule_path.open("xb") as output:
        output.write(SOURCE_SCHEDULE.read_bytes())
        output.flush()
        os.fsync(output.fileno())
    plan["schedule_sha256"] = sha256(schedule_path)
    plan_path = args.output_dir / "plan.json"
    with plan_path.open("x", encoding="utf-8") as output:
        json.dump(plan, output, sort_keys=True, indent=2)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    print(json.dumps({"plan_sha256": sha256(plan_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
