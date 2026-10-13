"""Freeze a broader signer-balanced continuation over the How2Sign expansion."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

from benchmark_fleurs_dev import sha256
from freeze_c5_how2sign_expansion_plan import (
    FLEURS_EXAMPLES,
    REPLAY_PER_SIGNER,
    load_expansion_examples,
)
from freeze_c5_stem_scale_plan import ROOT, load_examples
from interpolate_visual_adapter import DIRECT_NAMES

SOURCE_PLAN = ROOT / "runs/c5-how2sign-precut-expansion-v1/plan.json"
PROBE_DECISION = ROOT / "runs/c5-how2sign-precut-expansion-v3/tuning-decision.json"
SEED = 52
EXPANSION_TARGETS = {1: 92, 2: 20, 3: 72, 4: 72, 5: 93, 8: 93, 9: 20, 11: 50}
RECIPE = {
    "id": "c5-how2sign-precut-expansion-scale-v1",
    "rank": 16,
    "alpha": 32.0,
    "visual_last_layers": 4,
    "learning_rate": 2e-6,
    "weight_decay": 0.01,
    "label_smoothing": 0.1,
    "gradient_accumulation": 4,
    "cpu_threads": 4,
    "train_examples": 768,
    "checkpoint_after_examples": [128, 256, 512, 768],
    "frozen_direct_parameter_names": sorted(DIRECT_NAMES),
}


def stable_key(*values: object) -> str:
    return hashlib.sha256(
        "\0".join(str(value) for value in (SEED, *values)).encode()
    ).hexdigest()


def select_scale_schedule(replay: list[dict], expansion: list[dict]) -> list[dict]:
    expansion_by_signer = defaultdict(list)
    for row in expansion:
        expansion_by_signer[row["signer"]].append(row)
    if set(expansion_by_signer) != set(EXPANSION_TARGETS):
        raise ValueError("scaled expansion lacks a How2Sign training signer")
    selected = []
    for signer, target in EXPANSION_TARGETS.items():
        ordered = sorted(
            expansion_by_signer[signer],
            key=lambda row: stable_key("expansion", signer, row["id"]),
        )
        if len(ordered) < target:
            raise ValueError(f"scaled expansion signer {signer} lacks unique examples")
        selected.extend(ordered[:target])

    replay_by_signer = defaultdict(list)
    fleurs = []
    for row in replay:
        if row["source"] == "fleurs":
            fleurs.append(row)
        else:
            replay_by_signer[row["signer"]].append(row)
    if set(replay_by_signer) != set(EXPANSION_TARGETS):
        raise ValueError("scaled legacy replay lacks a How2Sign training signer")
    for signer, rows in sorted(replay_by_signer.items()):
        ordered = sorted(
            rows,
            key=lambda row: stable_key("replay", signer, row["source"], row["id"]),
        )
        if len(ordered) < REPLAY_PER_SIGNER:
            raise ValueError(f"scaled replay signer {signer} lacks unique examples")
        selected.extend(ordered[:REPLAY_PER_SIGNER])
    if len(fleurs) < FLEURS_EXAMPLES:
        raise ValueError("scaled replay lacks FLEURS examples")
    selected.extend(
        sorted(fleurs, key=lambda row: stable_key("fleurs", row["id"]))[
            :FLEURS_EXAMPLES
        ]
    )
    if len(selected) != RECIPE["train_examples"]:
        raise ValueError("scaled expansion schedule count differs")
    return sorted(
        selected,
        key=lambda row: stable_key("schedule", row["source"], row["id"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() or not args.output_dir.resolve().is_relative_to(
        (ROOT / "runs").resolve()
    ):
        parser.error("scaled expansion plan needs a new run-ledger directory")
    source = json.loads(SOURCE_PLAN.read_text())
    probe = json.loads(PROBE_DECISION.read_text())
    probe_final = probe.get("checkpoints", {}).get("512", {})
    if (
        source.get("schema") != "umi-studio-c5-frozen-replay-plan/1"
        or source.get("recipe", {}).get("id")
        != "c5-how2sign-precut-expansion-projection-frozen-v1"
        or probe.get("decision") != "retain_confirmed_candidate"
        or probe.get("selected_step") is not None
        or probe_final.get("how2sign", {}).get(
            "candidate_minus_baseline_mean_wer", 0
        )
        > -0.02
        or probe_final.get("fleurs", {}).get("candidate_minus_baseline_mean_wer")
        != 0
        or probe_final.get("how2sign_signers_improved") != 2
    ):
        raise ValueError("the magnitude probe does not justify broader data coverage")
    replay, _stem, source_hashes = load_examples()
    expansion, expansion_hashes = load_expansion_examples()
    schedule = select_scale_schedule(replay, expansion)
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
    args.output_dir.mkdir(parents=True, exist_ok=False)
    schedule_path = args.output_dir / "schedule.jsonl"
    with schedule_path.open("x", encoding="utf-8") as output:
        for row in schedule_rows:
            output.write(json.dumps(row, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())
    plan = copy.deepcopy(source)
    plan["objective"] = "test broader signer coverage from the pre-cut expansion"
    plan["seed"] = SEED
    plan["recipe"] = RECIPE
    plan["schedule_sha256"] = sha256(schedule_path)
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
    plan["expansion_signer_counts"] = {
        str(key): value for key, value in EXPANSION_TARGETS.items()
    }
    plan["source_hashes"] = {**source_hashes, **expansion_hashes}
    plan["source_plan_sha256"] = sha256(SOURCE_PLAN)
    plan["probe_decision_sha256"] = sha256(PROBE_DECISION)
    plan["trainer_source_sha256"] = sha256(ROOT / "tools/train_c5_frozen_replay.py")
    plan["plan_source_sha256"] = sha256(Path(__file__))
    plan_path = args.output_dir / "plan.json"
    with plan_path.open("x", encoding="utf-8") as output:
        json.dump(plan, output, sort_keys=True, indent=2)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    print(
        json.dumps(
            {"plan_sha256": sha256(plan_path), "source_counts": plan["source_counts"]},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
