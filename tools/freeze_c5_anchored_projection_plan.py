"""Freeze a diverse C5 continuation with gently anchored visual projection."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

from benchmark_fleurs_dev import sha256
from freeze_c5_how2sign_expansion_plan import load_expansion_examples
from freeze_c5_openasl_pilot_plan import load_openasl_examples
from freeze_c5_stem_continuation_plan import (
    BASELINE,
    CONFIRMATION,
    MERGED_EQUIVALENCE,
    ROOT,
    STARTING_ADAPTER,
    TUNING,
)
from freeze_c5_stem_scale_plan import load_examples
from interpolate_visual_adapter import DIRECT_NAMES

SEED = 89
EXPANSION_TARGETS = {1: 32, 2: 20, 3: 32, 4: 32, 5: 32, 8: 56, 9: 20, 11: 32}
REPLAY_PER_SIGNER = 32
FLEURS_COUNT = 136
STEM_PER_SOURCE = 128
OPENASL_COUNT = 120
RECIPE = {
    "id": "c5-anchored-projection-diverse-v1",
    "rank": 16,
    "alpha": 32.0,
    "visual_last_layers": 4,
    "lora_learning_rate": 2e-6,
    "direct_learning_rate": 2e-7,
    "direct_anchor_strength": 0.1,
    "direct_anchor_variance_floor": 1e-4,
    "weight_decay": 0.01,
    "label_smoothing": 0.1,
    "gradient_accumulation": 4,
    "cpu_threads": 4,
    "train_examples": 1_024,
    "checkpoint_after_examples": [128, 256, 512, 1_024],
    "anchored_direct_parameter_names": sorted(DIRECT_NAMES),
}


def stable_key(*values: object) -> str:
    return hashlib.sha256(
        "\0".join(str(value) for value in (SEED, *values)).encode()
    ).hexdigest()


def take_balanced(rows: list[dict], group_key: str, count: int, label: str) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row[group_key]].append(row)
    for group in grouped:
        grouped[group].sort(key=lambda row: stable_key(label, group, row["id"]))
    groups = sorted(grouped, key=lambda group: stable_key(label, "group", group))
    selected = []
    position = 0
    while len(selected) < count:
        advanced = False
        for group in groups:
            if position < len(grouped[group]):
                selected.append(grouped[group][position])
                advanced = True
                if len(selected) == count:
                    break
        if not advanced:
            raise ValueError(f"{label} cannot fill its balanced target")
        position += 1
    return selected


def select_schedule(
    replay: list[dict], stem: list[dict], expansion: list[dict], openasl: list[dict]
) -> list[dict]:
    selected = []
    expansion_by_signer = defaultdict(list)
    for row in expansion:
        expansion_by_signer[row["signer"]].append(row)
    if set(expansion_by_signer) != set(EXPANSION_TARGETS):
        raise ValueError("anchored schedule lacks an expansion signer")
    for signer, count in EXPANSION_TARGETS.items():
        rows = sorted(
            expansion_by_signer[signer],
            key=lambda row: stable_key("expansion", signer, row["id"]),
        )
        if len(rows) < count:
            raise ValueError(f"anchored expansion lacks signer {signer}")
        selected.extend(rows[:count])

    replay_by_signer = defaultdict(list)
    fleurs = []
    for row in replay:
        if row["source"] == "fleurs":
            fleurs.append(row)
        else:
            replay_by_signer[row["signer"]].append(row)
    if set(replay_by_signer) != set(EXPANSION_TARGETS):
        raise ValueError("anchored replay lacks a How2Sign signer")
    for signer, rows in sorted(replay_by_signer.items()):
        ordered = sorted(
            rows,
            key=lambda row: stable_key("replay", signer, row["source"], row["id"]),
        )
        if len(ordered) < REPLAY_PER_SIGNER:
            raise ValueError(f"anchored replay lacks signer {signer}")
        selected.extend(ordered[:REPLAY_PER_SIGNER])
    if len(fleurs) < FLEURS_COUNT:
        raise ValueError("anchored replay lacks FLEURS examples")
    selected.extend(sorted(fleurs, key=lambda row: stable_key("fleurs", row["id"]))[:FLEURS_COUNT])

    for source in ("asl_stem_original", "asl_stem_supplemental"):
        source_rows = [row for row in stem if row["source"] == source]
        selected.extend(
            take_balanced(source_rows, "participant", STEM_PER_SOURCE, source)
        )
    if len(openasl) != OPENASL_COUNT:
        raise ValueError("anchored schedule needs the complete OpenASL pilot")
    selected.extend(openasl)
    if len(selected) != RECIPE["train_examples"]:
        raise ValueError("anchored schedule count differs")
    if len({(row["source"], row["id"]) for row in selected}) != len(selected):
        raise ValueError("anchored schedule repeats an example")
    return sorted(
        selected, key=lambda row: stable_key("schedule", row["source"], row["id"])
    )


def frozen_rows(schedule: list[dict]) -> list[dict]:
    return [
        {
            "position": position,
            "source": row["source"],
            "example_id": row["id"],
            "signer": row["signer"],
            "stored_temporal_stride": row.get("stored_temporal_stride", 1),
        }
        for position, row in enumerate(schedule)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() or not args.output_dir.resolve().is_relative_to(
        (ROOT / "runs").resolve()
    ):
        parser.error("anchored projection plan needs a new run directory")
    confirmation = json.loads(CONFIRMATION.read_text())
    equivalence = json.loads(MERGED_EQUIVALENCE.read_text())
    if (
        confirmation.get("decision") != "package_candidate"
        or confirmation.get("candidate_adapter_sha256") != sha256(STARTING_ADAPTER)
        or equivalence.get("all_hypotheses_equal") is not True
        or equivalence.get("examples") != 88
    ):
        raise ValueError("anchored projection needs the confirmed starting adapter")
    replay, stem, source_hashes = load_examples()
    expansion, expansion_hashes = load_expansion_examples()
    openasl, openasl_hashes = load_openasl_examples()
    schedule = select_schedule(replay, stem, expansion, openasl)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    schedule_path = args.output_dir / "schedule.jsonl"
    with schedule_path.open("x", encoding="utf-8") as output:
        for row in frozen_rows(schedule):
            output.write(json.dumps(row, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())
    plan = {
        "schema": "umi-studio-c5-anchored-projection-plan/1",
        "objective": (
            "test whether gently anchored visual projection adaptation broadens "
            "signer and source generalization without decoder forgetting"
        ),
        "seed": SEED,
        "recipe": RECIPE,
        "starting_adapter": str(STARTING_ADAPTER.relative_to(ROOT)),
        "starting_adapter_sha256": sha256(STARTING_ADAPTER),
        "baseline_checkpoint_sha256": confirmation["how2sign"]["checkpoint_sha256"],
        "confirmation_decision_sha256": sha256(CONFIRMATION),
        "merged_equivalence_sha256": sha256(MERGED_EQUIVALENCE),
        "schedule_sha256": sha256(schedule_path),
        "source_counts": dict(sorted(Counter(row["source"] for row in schedule).items())),
        "source_hashes": {**source_hashes, **expansion_hashes, **openasl_hashes},
        "open_tuning": {
            domain: {"results": str(path.relative_to(ROOT)), "results_sha256": sha256(path)}
            for domain, path in TUNING.items()
        },
        "screen": {
            "how2sign_max_mean_wer_change": 0.0,
            "fleurs_max_mean_wer_change": 0.0,
            "equal_domain_max_mean_wer_change": -0.01,
            "minimum_improved_how2sign_signers": 3,
        },
        "confirmation_policy": (
            "Do not reuse either consumed confirmation set. A passing open screen must "
            "next be checked with the calibrated raw-video runtime and new evidence."
        ),
        "trainer_source_sha256": sha256(ROOT / "tools/train_c5_anchored_projection.py"),
        "plan_source_sha256": sha256(Path(__file__)),
        "miner_health_policy": "user_authorized_maintenance",
    }
    plan_path = args.output_dir / "plan.json"
    with plan_path.open("x", encoding="utf-8") as output:
        json.dump(plan, output, indent=2, sort_keys=True)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    print(json.dumps({"plan_sha256": sha256(plan_path), "source_counts": plan["source_counts"]}, sort_keys=True))


if __name__ == "__main__":
    main()
