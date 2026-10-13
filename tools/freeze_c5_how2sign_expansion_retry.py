"""Freeze a higher-learning-rate retry of the How2Sign expansion schedule."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

from benchmark_fleurs_dev import sha256
from freeze_c5_stem_scale_plan import ROOT

SOURCE_DIR = ROOT / "runs/c5-how2sign-precut-expansion-v1"
SOURCE_PLAN = SOURCE_DIR / "plan.json"
SOURCE_DECISION = SOURCE_DIR / "tuning-decision.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() or not args.output_dir.resolve().is_relative_to(
        (ROOT / "runs").resolve()
    ):
        parser.error("expansion retry needs a new run-ledger directory")
    source = json.loads(SOURCE_PLAN.read_text())
    decision = json.loads(SOURCE_DECISION.read_text())
    if (
        source.get("schema") != "umi-studio-c5-frozen-replay-plan/1"
        or source.get("recipe", {}).get("id")
        != "c5-how2sign-precut-expansion-projection-frozen-v1"
        or source["recipe"].get("learning_rate") != 1e-6
        or decision.get("schema")
        != "umi-studio-c5-how2sign-expansion-tuning-decision/1"
        or decision.get("plan_sha256") != sha256(SOURCE_PLAN)
        or decision.get("decision") != "retain_confirmed_candidate"
        or decision.get("selected_step") is not None
        or decision["checkpoints"]["512"]["how2sign"][
            "candidate_minus_baseline_mean_wer"
        ]
        >= 0
        or decision["checkpoints"]["512"]["fleurs"][
            "candidate_minus_baseline_mean_wer"
        ]
        != 0
    ):
        raise ValueError("the first expansion result does not justify a magnitude retry")
    plan = copy.deepcopy(source)
    plan["objective"] = "test a larger update on the same signer-balanced expansion schedule"
    plan["recipe"]["id"] = "c5-how2sign-precut-expansion-projection-frozen-v2"
    plan["recipe"]["learning_rate"] = 2e-6
    plan["source_plan_sha256"] = sha256(SOURCE_PLAN)
    plan["source_decision_sha256"] = sha256(SOURCE_DECISION)
    plan["trainer_source_sha256"] = sha256(ROOT / "tools/train_c5_frozen_replay.py")
    plan["plan_source_sha256"] = sha256(Path(__file__))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    schedule_path = args.output_dir / "schedule.jsonl"
    with schedule_path.open("xb") as output:
        output.write((SOURCE_DIR / "schedule.jsonl").read_bytes())
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
