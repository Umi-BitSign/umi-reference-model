"""Freeze a final higher-magnitude signer-recovery probe."""

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
V2_DECISION = ROOT / "runs/c5-signer-recovery-v2/tuning-decision.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() or not args.output_dir.resolve().is_relative_to(
        (ROOT / "runs").resolve()
    ):
        parser.error("signer recovery v3 needs a new run-ledger directory")
    source = json.loads(SOURCE_PLAN.read_text())
    v2 = json.loads(V2_DECISION.read_text())
    if (
        source.get("schema") != "umi-studio-c5-frozen-replay-plan/1"
        or source.get("recipe", {}).get("learning_rate") != 1e-6
        or v2.get("decision") != "retain_confirmed_candidate"
        or v2.get("selected_step") is not None
        or v2["checkpoints"]["512"]["fleurs"][
            "candidate_minus_baseline_mean_wer"
        ]
        >= 0
        or v2["checkpoints"]["512"]["how2sign_signers_improved"] != 2
    ):
        raise ValueError("signer recovery v2 does not justify the magnitude probe")
    plan = copy.deepcopy(source)
    plan["objective"] = "test the final signer-recovery update magnitude"
    plan["seed"] = 47
    plan["recipe"]["id"] = "c5-signer-recovery-projection-frozen-v3"
    plan["recipe"]["learning_rate"] = 4e-6
    plan["source_plan_sha256"] = sha256(SOURCE_PLAN)
    plan["v2_decision_sha256"] = sha256(V2_DECISION)
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
