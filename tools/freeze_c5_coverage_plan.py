"""Freeze a single-change coverage-order test against the completed retention run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark_fleurs_dev import sha256
from c5_all_source_training import load_corpus
from c5_coverage_schedule import coverage_schedule, prefix_counts
from c5_experiment_sources import archive_sources
from c5_retention import COVERAGE_RECIPE, RECIPE as CONTROL_RECIPE
from freeze_c5_stem_continuation_plan import ROOT
from train_c5_frozen_replay import materialize_schedule


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() or not args.output_dir.resolve().is_relative_to(ROOT / "runs"):
        parser.error("coverage plan needs a new runs/ output")
    control_path = ROOT / "runs/c5-all-source-retention-v1/plan.json"
    control = json.loads(control_path.read_text())
    if control["recipe"] != CONTROL_RECIPE:
        raise ValueError("coverage control recipe changed")
    rows, stem, hashes = load_corpus(control)
    schedule_path = control_path.with_name("schedule.jsonl")
    frozen = [json.loads(line) for line in schedule_path.read_text().splitlines()]
    original = materialize_schedule(rows, stem, frozen)
    if hashes != control["source_hashes"] or sha256(schedule_path) != control["schedule_sha256"]:
        raise ValueError("coverage control data changed")
    schedule = coverage_schedule(original, COVERAGE_RECIPE["domain_mass"])
    if len(schedule) != COVERAGE_RECIPE["train_examples"]:
        raise ValueError("coverage schedule lost training clips")
    plan = {
        **control,
        "recipe": COVERAGE_RECIPE,
        "objective": "Test whether earlier distinct signer coverage broadens the retention run's gains.",
        "single_experimental_change": "Reorder the same unique clips with weighted-fair domain/signer coverage; preserve within-group order and all objective weights.",
        "matched_control_plan": str(control_path.relative_to(ROOT)),
        "matched_control_plan_sha256": sha256(control_path),
        "matched_control_schedule_sha256": sha256(schedule_path),
        "trainer_source_sha256": sha256(ROOT / "tools/train_c5_frozen_replay.py"),
        "optimizer_groups_source_sha256": sha256(ROOT / "tools/c5_scale_learning_rates.py"),
        "retention_implementation_sha256": sha256(ROOT / "tools/c5_retention.py"),
        "coverage_implementation_sha256": sha256(ROOT / "tools/c5_coverage_schedule.py"),
        "freezer_source_sha256": sha256(Path(__file__)),
        "source_snapshot_dir": str((args.output_dir.resolve() / "source-snapshot").relative_to(ROOT)),
        "coverage_qualification": "Early clip counts are balanced until groups exhaust. Existing global loss weights are unchanged, so prefix objective masses are not asserted balanced; every clip occurs exactly once across the full run.",
        "prefix_coverage": {
            str(limit): {"control": prefix_counts(original, limit), "candidate": prefix_counts(schedule, limit)}
            for limit in COVERAGE_RECIPE["checkpoint_after_examples"]
        },
    }
    args.output_dir.mkdir(parents=True)
    target = args.output_dir / "schedule.jsonl"
    with target.open("x") as output:
        for position, row in enumerate(schedule):
            output.write(json.dumps({
                "position": position, "source": row["source"], "example_id": row["id"],
                "signer": row["signer"], "stored_temporal_stride": row.get("stored_temporal_stride", 1),
            }, sort_keys=True) + "\n")
    plan["schedule_sha256"] = sha256(target)
    plan_path = args.output_dir / "plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    archive_sources(plan_path, args.output_dir / "source-snapshot")
    print(json.dumps({"plan_sha256": sha256(plan_path), "examples": len(schedule), "prefix1024": plan["prefix_coverage"]["1024"]}))


if __name__ == "__main__":
    main()
