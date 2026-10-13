"""Freeze one signer-3-emphasized recovery from the best balanced replay."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

from benchmark_fleurs_dev import sha256
from freeze_c5_stem_scale_plan import ROOT, load_examples

SEED = 48
SOURCE_PLAN = ROOT / "runs/c5-signer-recovery-v1/plan.json"
PARENT_DECISION = ROOT / "runs/c5-signer-recovery-v2/tuning-decision.json"
STARTING_ADAPTER = (
    ROOT / "checkpoints/c5-signer-recovery-v2/visual-lora-after-000512.safetensors"
)
FLEURS_EXAMPLES = 128
FOCUS_SIGNER = 3
FOCUS_COUNT = 192
OTHER_COUNT = 64


def stable_key(*values: object) -> str:
    return hashlib.sha256(
        "\0".join(str(value) for value in (SEED, *values)).encode()
    ).hexdigest()


def select_schedule(replay: list[dict]) -> list[dict]:
    by_signer = defaultdict(list)
    fleurs = []
    for row in replay:
        if row["source"] == "fleurs":
            fleurs.append(row)
        else:
            by_signer[row["signer"]].append(row)
    if set(by_signer) != {1, 2, 3, 4, 5, 8, 9, 11}:
        raise ValueError("signer-3 recovery lacks a How2Sign training signer")
    selected = []
    for signer, rows in sorted(by_signer.items()):
        target = FOCUS_COUNT if signer == FOCUS_SIGNER else OTHER_COUNT
        ordered = sorted(
            rows,
            key=lambda row: stable_key("h", signer, row["source"], row["id"]),
        )
        if len(ordered) < target:
            raise ValueError(f"signer-3 recovery lacks examples for signer {signer}")
        selected.extend(ordered[:target])
    if len(fleurs) < FLEURS_EXAMPLES:
        raise ValueError("signer-3 recovery lacks FLEURS examples")
    selected.extend(
        sorted(fleurs, key=lambda row: stable_key("f", row["id"]))[:FLEURS_EXAMPLES]
    )
    if len(selected) != 768:
        raise ValueError("signer-3 recovery schedule count differs")
    return sorted(selected, key=lambda row: stable_key("schedule", row["source"], row["id"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() or not args.output_dir.resolve().is_relative_to(
        (ROOT / "runs").resolve()
    ):
        parser.error("signer-3 recovery needs a new run-ledger directory")
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
        source.get("schema") != "umi-studio-c5-frozen-replay-plan/1"
        or parent.get("decision") != "retain_confirmed_candidate"
        or parent.get("selected_step") is not None
        or parent["checkpoints"]["512"]["fleurs"][
            "candidate_minus_baseline_mean_wer"
        ]
        >= 0
        or parent["checkpoints"]["512"]["how2sign_signers_improved"] != 2
        or sha256(STARTING_ADAPTER) != expected_hash
    ):
        raise ValueError("best balanced recovery checkpoint is not frozen")
    replay, _stem, source_hashes = load_examples()
    schedule = select_schedule(replay)
    schedule_rows = [
        {
            "position": position,
            "source": row["source"],
            "example_id": row["id"],
            "signer": row["signer"],
            "stored_temporal_stride": row.get("stored_temporal_stride", 1),
        }
        for position, row in enumerate(schedule)
    ]
    plan = copy.deepcopy(source)
    plan["objective"] = "test whether added signer-3 diversity broadens the aggregate gain"
    plan["seed"] = SEED
    plan["recipe"]["id"] = "c5-signer3-recovery-projection-frozen-v1"
    plan["recipe"]["learning_rate"] = 2e-6
    plan["starting_adapter"] = str(STARTING_ADAPTER.relative_to(ROOT))
    plan["starting_adapter_sha256"] = sha256(STARTING_ADAPTER)
    plan["parent_decision"] = str(PARENT_DECISION.relative_to(ROOT))
    plan["parent_decision_sha256"] = sha256(PARENT_DECISION)
    plan["source_plan_sha256"] = sha256(SOURCE_PLAN)
    plan["source_hashes"] = source_hashes
    plan["source_counts"] = dict(
        sorted(Counter(row["source"] for row in schedule).items())
    )
    plan["how2sign_signer_counts"] = dict(
        sorted(
            Counter(
                str(row["signer"])
                for row in schedule
                if row["source"].startswith("how2sign")
            ).items()
        )
    )
    plan["trainer_source_sha256"] = sha256(ROOT / "tools/train_c5_frozen_replay.py")
    plan["plan_source_sha256"] = sha256(Path(__file__))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    schedule_path = args.output_dir / "schedule.jsonl"
    with schedule_path.open("x", encoding="utf-8") as output:
        for row in schedule_rows:
            output.write(json.dumps(row, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())
    plan["schedule_sha256"] = sha256(schedule_path)
    plan_path = args.output_dir / "plan.json"
    with plan_path.open("x", encoding="utf-8") as output:
        json.dump(plan, output, sort_keys=True, indent=2)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    print(
        json.dumps(
            {
                "plan_sha256": sha256(plan_path),
                "signer_counts": plan["how2sign_signer_counts"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
