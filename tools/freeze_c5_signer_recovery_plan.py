"""Freeze a signer-balanced recovery replay from the scaled STEM endpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

from benchmark_fleurs_dev import sha256
from freeze_c5_stem_continuation_plan import CONFIRMATION, MERGED_EQUIVALENCE, TUNING
from freeze_c5_stem_scale_plan import ROOT, load_examples
from interpolate_visual_adapter import DIRECT_NAMES

SEED = 45
STARTING_ADAPTER = (
    ROOT / "checkpoints/c5-stem-scale-v1/visual-lora-after-003112.safetensors"
)
PARENT_DECISION = ROOT / "runs/c5-stem-scale-v1/tuning-decision.json"
PARENT_PLAN = ROOT / "runs/c5-stem-scale-v1/plan.json"
HOW2SIGN_PER_SIGNER = 80
FLEURS_EXAMPLES = 128
RECIPE = {
    "id": "c5-signer-recovery-projection-frozen-v1",
    "rank": 16,
    "alpha": 32.0,
    "visual_last_layers": 4,
    "learning_rate": 1e-6,
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


def select_schedule(replay: list[dict]) -> list[dict]:
    by_signer_source = defaultdict(lambda: defaultdict(list))
    fleurs = []
    for row in replay:
        if row["source"] == "fleurs":
            fleurs.append(row)
        else:
            by_signer_source[row["signer"]][row["source"]].append(row)
    if set(by_signer_source) != {1, 2, 3, 4, 5, 8, 9, 11}:
        raise ValueError("signer recovery lacks a How2Sign training signer")
    selected = []
    for signer, source_rows in sorted(by_signer_source.items()):
        ordered = []
        for source, rows in source_rows.items():
            ordered.extend(
                sorted(
                    rows,
                    key=lambda row: stable_key("h", signer, source, row["id"]),
                )
            )
        ordered.sort(key=lambda row: stable_key("merged", signer, row["source"], row["id"]))
        if len(ordered) < HOW2SIGN_PER_SIGNER:
            raise ValueError(f"signer recovery lacks examples for signer {signer}")
        selected.extend(ordered[:HOW2SIGN_PER_SIGNER])
    if len(fleurs) < FLEURS_EXAMPLES:
        raise ValueError("signer recovery lacks FLEURS examples")
    selected.extend(
        sorted(fleurs, key=lambda row: stable_key("f", row["id"]))[:FLEURS_EXAMPLES]
    )
    if len(selected) != RECIPE["train_examples"]:
        raise ValueError("signer recovery schedule count differs")
    return sorted(selected, key=lambda row: stable_key("schedule", row["source"], row["id"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() or not args.output_dir.resolve().is_relative_to(
        (ROOT / "runs").resolve()
    ):
        parser.error("signer recovery needs a new run-ledger directory")
    parent = json.loads(PARENT_DECISION.read_text())
    parent_plan = json.loads(PARENT_PLAN.read_text())
    if (
        parent.get("schema") != "umi-studio-c5-stem-scale-tuning-decision/1"
        or parent.get("plan_sha256") != sha256(PARENT_PLAN)
        or parent_plan.get("schema") != "umi-studio-c5-stem-scale-plan/1"
        or parent.get("decision") != "retain_confirmed_candidate"
        or parent.get("selected_step") is not None
        or parent.get("direct_parameters_preserved") is not True
        or parent.get("prior_confirmation_reused") is not False
        or sha256(STARTING_ADAPTER)
        != next(
            row["checkpoint_sha256"]
            for row in map(
                json.loads,
                (ROOT / "checkpoints/c5-stem-scale-v1/checkpoint-index.jsonl")
                .read_text()
                .splitlines(),
            )
            if row["example_step"] == 3_112
        )
    ):
        raise ValueError("scaled STEM endpoint is not frozen")
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
    args.output_dir.mkdir(parents=True, exist_ok=False)
    schedule_path = args.output_dir / "schedule.jsonl"
    with schedule_path.open("x", encoding="utf-8") as output:
        for row in schedule_rows:
            output.write(json.dumps(row, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())
    plan = {
        "schema": "umi-studio-c5-frozen-replay-plan/1",
        "objective": "recover broad signer gains after the full STEM replay",
        "seed": SEED,
        "recipe": RECIPE,
        "starting_adapter": str(STARTING_ADAPTER.relative_to(ROOT)),
        "starting_adapter_sha256": sha256(STARTING_ADAPTER),
        "parent_decision": str(PARENT_DECISION.relative_to(ROOT)),
        "parent_decision_sha256": sha256(PARENT_DECISION),
        "parent_plan_sha256": sha256(PARENT_PLAN),
        "baseline_checkpoint_sha256": parent_plan["baseline_checkpoint_sha256"],
        "confirmation_decision_sha256": sha256(CONFIRMATION),
        "merged_equivalence_sha256": sha256(MERGED_EQUIVALENCE),
        "confirmation_policy": (
            "The consumed confirmation results are final. Do not evaluate this recovery "
            "experiment on either confirmation set."
        ),
        "schedule_sha256": sha256(schedule_path),
        "source_counts": dict(sorted(Counter(row["source"] for row in schedule).items())),
        "how2sign_signer_counts": dict(
            sorted(
                Counter(
                    str(row["signer"])
                    for row in schedule
                    if row["source"].startswith("how2sign")
                ).items()
            )
        ),
        "source_hashes": source_hashes,
        "open_tuning": {
            domain: {
                "results": str(path.relative_to(ROOT)),
                "results_sha256": sha256(path),
            }
            for domain, path in TUNING.items()
        },
        "screen": {
            "how2sign_max_mean_wer_change": 0.01,
            "fleurs_max_mean_wer_change": 0.01,
            "equal_domain_max_mean_wer_change": -0.005,
            "minimum_improved_how2sign_signers": 3,
            "next_action": (
                "Only a passing checkpoint may justify fresh held-out collection; it cannot "
                "replace the confirmed candidate from open tuning alone."
            ),
        },
        "trainer_source_sha256": sha256(ROOT / "tools/train_c5_frozen_replay.py"),
        "plan_source_sha256": sha256(Path(__file__)),
        "miner_health_policy": "user_authorized_maintenance",
    }
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
                "source_counts": plan["source_counts"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
