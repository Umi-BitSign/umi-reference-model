"""Freeze one stronger-learning-rate retry of the OpenASL pilot."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from benchmark_fleurs_dev import sha256
from freeze_c5_openasl_pilot_plan import RECIPE as PARENT_RECIPE
from freeze_c5_stem_scale_plan import ROOT

RECIPE = {
    **PARENT_RECIPE,
    "id": "c5-openasl-diversity-projection-frozen-v2",
    "learning_rate": 4e-6,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-plan", type=Path, required=True)
    parser.add_argument("--parent-decision", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if (
        not args.parent_plan.is_file()
        or not args.parent_decision.is_file()
        or args.output_dir.exists()
        or not args.output_dir.resolve().is_relative_to((ROOT / "runs").resolve())
    ):
        parser.error("OpenASL retry needs frozen parent inputs and a new output")
    parent = json.loads(args.parent_plan.read_text())
    decision = json.loads(args.parent_decision.read_text())
    parent_schedule = args.parent_plan.with_name("schedule.jsonl")
    if (
        parent.get("schema") != "umi-studio-c5-frozen-replay-plan/1"
        or parent.get("recipe") != PARENT_RECIPE
        or parent.get("schedule_sha256") != sha256(parent_schedule)
        or decision.get("schema") != "umi-studio-c5-openasl-pilot-tuning-decision/1"
        or decision.get("plan_sha256") != sha256(args.parent_plan)
        or decision.get("decision") != "retain_confirmed_candidate"
        or decision.get("selected_step") is not None
        or decision.get("prior_confirmation_reused") is not False
    ):
        raise ValueError("OpenASL retry parent differs")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    schedule_path = args.output_dir / "schedule.jsonl"
    schedule_path.write_bytes(parent_schedule.read_bytes())
    plan = {
        **parent,
        "objective": "test whether the OpenASL pilot was update-limited at 2e-6",
        "recipe": RECIPE,
        "schedule_sha256": sha256(schedule_path),
        "parent_plan": str(args.parent_plan.resolve().relative_to(ROOT)),
        "parent_plan_sha256": sha256(args.parent_plan),
        "parent_decision": str(args.parent_decision.resolve().relative_to(ROOT)),
        "parent_decision_sha256": sha256(args.parent_decision),
        "trainer_source_sha256": sha256(ROOT / "tools/train_c5_frozen_replay.py"),
        "plan_source_sha256": sha256(Path(__file__)),
    }
    plan_path = args.output_dir / "plan.json"
    with plan_path.open("x", encoding="utf-8") as output:
        json.dump(plan, output, sort_keys=True, indent=2)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    print(json.dumps({"plan_sha256": sha256(plan_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
