"""Freeze a matched full-corpus retention experiment before training/evaluation."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from benchmark_fleurs_dev import sha256
from c5_all_source_training import load_corpus
from c5_retention import RECIPE
from freeze_c5_stem_continuation_plan import ROOT
from train_c5_frozen_replay import materialize_schedule


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() or not args.output_dir.resolve().is_relative_to(ROOT / "runs"):
        parser.error("retention plan needs a new runs/ output")
    control_path = ROOT / "runs/c5-all-source-v1/plan.json"
    control = json.loads(control_path.read_text())
    plan = {
        **control,
        "recipe": RECIPE,
        "objective": "Preserve starting-model token distributions during the identical audited full-corpus adaptation.",
        "single_experimental_change": "Add coefficient-1 teacher KL at temperature 2 on training-only reference prefixes.",
        "matched_control_plan": str(control_path.relative_to(ROOT)),
        "matched_control_plan_sha256": sha256(control_path),
        "trainer_source_sha256": sha256(ROOT / "tools/train_c5_frozen_replay.py"),
        "optimizer_groups_source_sha256": sha256(ROOT / "tools/c5_scale_learning_rates.py"),
        "retention_implementation_sha256": sha256(ROOT / "tools/c5_retention.py"),
        "freezer_source_sha256": sha256(Path(__file__)),
        "teacher_adapter_sha256": control["starting_adapter_sha256"],
        "teacher_evaluation_data_used": False,
    }
    rows, stem, hashes = load_corpus(plan)
    frozen = [json.loads(line) for line in control_path.with_name("schedule.jsonl").read_text().splitlines()]
    schedule = materialize_schedule(rows, stem, frozen)
    if hashes != plan["source_hashes"] or len(schedule) != RECIPE["train_examples"]:
        raise ValueError("matched retention corpus differs from control")
    if sha256(control_path.with_name("schedule.jsonl")) != plan["schedule_sha256"]:
        raise ValueError("matched retention schedule changed")
    args.output_dir.mkdir(parents=True)
    shutil.copyfile(control_path.with_name("schedule.jsonl"), args.output_dir / "schedule.jsonl")
    plan_path = args.output_dir / "plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"plan_sha256": sha256(plan_path), "examples": len(schedule), "recipe": RECIPE}))


if __name__ == "__main__":
    main()
