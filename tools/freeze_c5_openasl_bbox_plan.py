"""Freeze the signer-isolated OpenASL replay against its full-frame parent."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from benchmark_fleurs_dev import sha256
from freeze_c5_openasl_pilot_plan import load_openasl_examples
from freeze_c5_openasl_pilot_retry import RECIPE as PARENT_RECIPE
from freeze_c5_stem_scale_plan import ROOT

BBOX_INDEX = ROOT / "runs/openasl-bbox-v1/stream-cache-stride2.jsonl"
BBOX_CACHE = ROOT / "data/openasl/bbox-stream-cache-stride2"
BBOX_POLICY = "annotated_bbox_preserve_aspect_ratio"
RECIPE = {
    **PARENT_RECIPE,
    "id": "c5-openasl-bbox-projection-frozen-v1",
}


def load_openasl_bbox_examples() -> tuple[list[dict], dict[str, str]]:
    return load_openasl_examples(BBOX_INDEX, BBOX_CACHE, BBOX_POLICY)


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
        parser.error("OpenASL bbox plan needs frozen parent inputs and a new output")
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
        raise ValueError("OpenASL bbox parent differs")
    bbox_examples, bbox_hashes = load_openasl_bbox_examples()
    scheduled = [json.loads(line) for line in parent_schedule.read_text().splitlines()]
    scheduled_openasl = {
        row["example_id"] for row in scheduled if row["source"] == "openasl"
    }
    if scheduled_openasl != {row["id"] for row in bbox_examples}:
        raise ValueError("OpenASL bbox examples differ from the frozen schedule")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    schedule_path = args.output_dir / "schedule.jsonl"
    schedule_path.write_bytes(parent_schedule.read_bytes())
    plan = {
        **parent,
        "objective": "test annotated signer isolation on the same OpenASL replay",
        "recipe": RECIPE,
        "schedule_sha256": sha256(schedule_path),
        "parent_plan": str(args.parent_plan.resolve().relative_to(ROOT)),
        "parent_plan_sha256": sha256(args.parent_plan),
        "parent_decision": str(args.parent_decision.resolve().relative_to(ROOT)),
        "parent_decision_sha256": sha256(args.parent_decision),
        "source_hashes": {**parent["source_hashes"], **bbox_hashes},
        "openasl_spatial_policy": BBOX_POLICY,
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
