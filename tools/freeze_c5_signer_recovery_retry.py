"""Freeze a stronger learning-rate retry of the signer-recovery schedule."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

from benchmark_fleurs_dev import sha256
from freeze_c5_stem_scale_plan import ROOT

SOURCE_DIR = ROOT / "runs/c5-signer-recovery-v1"
SOURCE_PLAN = SOURCE_DIR / "plan.json"
SOURCE_DECISION = SOURCE_DIR / "tuning-decision.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() or not args.output_dir.resolve().is_relative_to(
        (ROOT / "runs").resolve()
    ):
        parser.error("signer recovery retry needs a new run-ledger directory")
    source = json.loads(SOURCE_PLAN.read_text())
    decision = json.loads(SOURCE_DECISION.read_text())
    if (
        source.get("schema") != "umi-studio-c5-frozen-replay-plan/1"
        or source.get("recipe", {}).get("learning_rate") != 1e-6
        or decision.get("decision") != "retain_confirmed_candidate"
        or decision.get("selected_step") is not None
        or decision.get("plan_sha256") != sha256(SOURCE_PLAN)
        or any(
            row["how2sign"]["candidate_minus_baseline_mean_wer"]
            != decision["checkpoints"]["128"]["how2sign"][
                "candidate_minus_baseline_mean_wer"
            ]
            for row in decision["checkpoints"].values()
        )
    ):
        raise ValueError("signer recovery source run is not a stable no-op")
    plan = copy.deepcopy(source)
    plan["objective"] = "retry signer recovery with enough update magnitude to change decoding"
    plan["seed"] = 46
    plan["recipe"]["id"] = "c5-signer-recovery-projection-frozen-v2"
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
