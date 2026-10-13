"""Freeze a projection-frozen repeat of the stride-2 ASL STEM continuation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from benchmark_fleurs_dev import sha256
from freeze_c5_stem_continuation_plan import RECIPE as PARENT_RECIPE
from freeze_c5_stem_continuation_plan import ROOT
from interpolate_visual_adapter import DIRECT_NAMES

RECIPE = {
    **PARENT_RECIPE,
    "id": "c5-stem-projection-frozen-v1",
    "frozen_direct_parameter_names": sorted(DIRECT_NAMES),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if (
        not args.parent_plan.is_file()
        or args.output_dir.exists()
        or not args.parent_plan.resolve().is_relative_to((ROOT / "runs").resolve())
        or not args.output_dir.resolve().is_relative_to((ROOT / "runs").resolve())
    ):
        parser.error("projection-frozen plan needs a parent plan and new run directory")
    parent = json.loads(args.parent_plan.read_text())
    parent_schedule = args.parent_plan.with_name("schedule.jsonl")
    if (
        parent.get("schema") != "umi-studio-c5-stem-continuation-plan/1"
        or parent.get("recipe") != PARENT_RECIPE
        or parent.get("schedule_sha256") != sha256(parent_schedule)
        or parent.get("miner_health_policy") != "user_authorized_maintenance"
        or "Do not evaluate" not in parent.get("confirmation_policy", "")
    ):
        raise ValueError("parent STEM continuation plan differs")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    schedule_path = args.output_dir / "schedule.jsonl"
    schedule_path.write_bytes(parent_schedule.read_bytes())
    plan = {
        "schema": "umi-studio-c5-stem-projection-frozen-plan/1",
        "objective": (
            "isolate whether direct visual-projection drift caused the STEM "
            "continuation's signer regression"
        ),
        "parent_plan": str(args.parent_plan.resolve().relative_to(ROOT)),
        "parent_plan_sha256": sha256(args.parent_plan),
        "recipe": RECIPE,
        "seed": parent["seed"],
        "starting_adapter": parent["starting_adapter"],
        "starting_adapter_sha256": parent["starting_adapter_sha256"],
        "baseline_checkpoint_sha256": parent["baseline_checkpoint_sha256"],
        "schedule_sha256": sha256(schedule_path),
        "source_counts": parent["source_counts"],
        "how2sign_signer_counts": parent["how2sign_signer_counts"],
        "asl_stem_participants": parent["asl_stem_participants"],
        "source_hashes": parent["source_hashes"],
        "open_tuning": parent["open_tuning"],
        "screen": parent["screen"],
        "confirmation_decision_sha256": parent["confirmation_decision_sha256"],
        "merged_equivalence_sha256": parent["merged_equivalence_sha256"],
        "confirmation_policy": parent["confirmation_policy"],
        "trainer_source_sha256": sha256(
            ROOT / "tools/train_c5_stem_projection_frozen.py"
        ),
        "plan_source_sha256": sha256(Path(__file__)),
        "miner_health_policy": "user_authorized_maintenance",
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
