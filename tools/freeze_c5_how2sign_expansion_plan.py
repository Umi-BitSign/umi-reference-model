"""Freeze a signer-balanced C5 continuation over pre-cut How2Sign expansion data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

from benchmark_fleurs_dev import sha256
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

SEED = 51
EXPANSION_INDEX = (
    ROOT / "runs/how2sign-precut-expansion-v1/stream-cache-stride2-index.jsonl"
)
EXPANSION_CACHE = ROOT / "data/how2sign/precut-expansion-stream-cache-stride2"
EXPANSION_TARGETS = {1: 32, 2: 20, 3: 32, 4: 32, 5: 32, 8: 56, 9: 20, 11: 32}
REPLAY_PER_SIGNER = 16
FLEURS_EXAMPLES = 128
RECIPE = {
    "id": "c5-how2sign-precut-expansion-projection-frozen-v1",
    "rank": 16,
    "alpha": 32.0,
    "visual_last_layers": 4,
    "learning_rate": 1e-6,
    "weight_decay": 0.01,
    "label_smoothing": 0.1,
    "gradient_accumulation": 4,
    "cpu_threads": 4,
    "train_examples": 512,
    "checkpoint_after_examples": [128, 256, 384, 512],
    "frozen_direct_parameter_names": sorted(DIRECT_NAMES),
}


def stable_key(*values: object) -> str:
    return hashlib.sha256(
        "\0".join(str(value) for value in (SEED, *values)).encode()
    ).hexdigest()


def load_expansion_examples() -> tuple[list[dict], dict[str, str]]:
    rows = [json.loads(line) for line in EXPANSION_INDEX.read_text().splitlines()]
    names = [row.get("sentence_name") for row in rows]
    if len(rows) != 1_650 or len(set(names)) != 1_650:
        raise ValueError("pre-cut How2Sign expansion stream index is incomplete")
    examples = []
    for row in rows:
        cache = EXPANSION_CACHE / f"{row['sentence_name']}.npz"
        if (
            row.get("schema")
            != "umi-studio-how2sign-precut-expansion-stream-cache/1"
            or row.get("split") != "training"
            or row.get("temporal_stride") != 2
            or row.get("frames") != row.get("encoder_shape", [None])[0]
            or row.get("signer") not in EXPANSION_TARGETS
            or not cache.is_file()
            or sha256(cache) != row.get("encoder_cache_sha256")
        ):
            raise ValueError(
                f"pre-cut How2Sign expansion stream differs: {row.get('sentence_name')}"
            )
        examples.append(
            {
                "id": row["sentence_name"],
                "source": "how2sign_precut_expansion",
                "signer": row["signer"],
                "reference": row["reference"],
                "cache": str(cache),
                "cache_sha256": row["encoder_cache_sha256"],
                "sidecar": None,
                "sidecar_sha256": None,
                "frames": row["frames"],
                "stored_temporal_stride": 2,
            }
        )
    return examples, {"how2sign_precut_expansion_index_sha256": sha256(EXPANSION_INDEX)}


def select_schedule(replay: list[dict], expansion: list[dict]) -> list[dict]:
    expansion_by_signer = defaultdict(list)
    for row in expansion:
        expansion_by_signer[row["signer"]].append(row)
    if set(expansion_by_signer) != set(EXPANSION_TARGETS):
        raise ValueError("pre-cut expansion lacks a How2Sign training signer")
    selected = []
    for signer, target in EXPANSION_TARGETS.items():
        ordered = sorted(
            expansion_by_signer[signer],
            key=lambda row: stable_key("expansion", signer, row["id"]),
        )
        if len(ordered) < target:
            raise ValueError(f"pre-cut expansion signer {signer} lacks unique examples")
        selected.extend(ordered[:target])

    replay_by_signer = defaultdict(list)
    fleurs = []
    for row in replay:
        if row["source"] == "fleurs":
            fleurs.append(row)
        else:
            replay_by_signer[row["signer"]].append(row)
    if set(replay_by_signer) != set(EXPANSION_TARGETS):
        raise ValueError("legacy replay lacks a How2Sign training signer")
    for signer, rows in sorted(replay_by_signer.items()):
        ordered = sorted(
            rows,
            key=lambda row: stable_key("replay", signer, row["source"], row["id"]),
        )
        if len(ordered) < REPLAY_PER_SIGNER:
            raise ValueError(f"legacy replay signer {signer} lacks unique examples")
        selected.extend(ordered[:REPLAY_PER_SIGNER])
    if len(fleurs) < FLEURS_EXAMPLES:
        raise ValueError("legacy replay lacks FLEURS examples")
    selected.extend(
        sorted(fleurs, key=lambda row: stable_key("fleurs", row["id"]))[
            :FLEURS_EXAMPLES
        ]
    )
    if len(selected) != RECIPE["train_examples"]:
        raise ValueError("pre-cut expansion schedule count differs")
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
        parser.error("pre-cut expansion plan needs a new run-ledger directory")
    confirmation = json.loads(CONFIRMATION.read_text())
    equivalence = json.loads(MERGED_EQUIVALENCE.read_text())
    if (
        confirmation.get("decision") != "package_candidate"
        or confirmation.get("candidate_adapter_sha256") != sha256(STARTING_ADAPTER)
        or equivalence.get("all_hypotheses_equal") is not True
        or equivalence.get("examples") != 88
    ):
        raise ValueError("confirmed C5 starting candidate is not frozen")
    replay, _stem, source_hashes = load_examples()
    expansion, expansion_hashes = load_expansion_examples()
    schedule = select_schedule(replay, expansion)
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
        "objective": "test signer-balanced pre-cut How2Sign expansion on confirmed C5",
        "seed": SEED,
        "recipe": RECIPE,
        "starting_adapter": str(STARTING_ADAPTER.relative_to(ROOT)),
        "starting_adapter_sha256": sha256(STARTING_ADAPTER),
        "baseline_checkpoint_sha256": confirmation["how2sign"]["checkpoint_sha256"],
        "confirmation_decision_sha256": sha256(CONFIRMATION),
        "merged_equivalence_sha256": sha256(MERGED_EQUIVALENCE),
        "confirmation_policy": (
            "The consumed confirmation results are final. Do not evaluate this experiment "
            "on either confirmation set."
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
        "expansion_signer_counts": {
            str(key): value for key, value in EXPANSION_TARGETS.items()
        },
        "source_hashes": {**source_hashes, **expansion_hashes},
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
                "Only a passing checkpoint may justify a larger expansion continuation; "
                "it cannot replace the confirmed candidate from open tuning alone."
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
            {"plan_sha256": sha256(plan_path), "source_counts": plan["source_counts"]},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
