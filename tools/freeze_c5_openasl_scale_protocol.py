"""Freeze the C5 OpenASL-scale schedule and gates before acquiring its media."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

from benchmark_fleurs_dev import sha256
from freeze_c5_stem_continuation_plan import (
    CONFIRMATION,
    MERGED_EQUIVALENCE,
    STARTING_ADAPTER,
    TUNING,
)
from freeze_c5_stem_scale_plan import ROOT, load_examples
from interpolate_visual_adapter import DIRECT_NAMES

SEED = 223
COUNTS = {
    "openasl_scale": 240,
    "how2sign": 256,
    "fleurs": 136,
    "asl_stem_original": 68,
    "asl_stem_supplemental": 68,
}
RECIPE = {
    "id": "c5-openasl-scale-projection-frozen-v1",
    "rank": 16,
    "alpha": 32.0,
    "visual_last_layers": 4,
    "learning_rate": 4e-6,
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


def take_balanced(rows: list[dict], count: int, source: str) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[row["participant"]].append(row)
    for participant in groups:
        groups[participant].sort(
            key=lambda row: stable_key(source, participant, row["id"])
        )
    participants = sorted(groups, key=lambda item: stable_key(source, "group", item))
    selected = []
    offset = 0
    while len(selected) < count:
        advanced = False
        for participant in participants:
            if offset < len(groups[participant]):
                selected.append(groups[participant][offset])
                advanced = True
                if len(selected) == count:
                    break
        if not advanced:
            raise ValueError(f"{source} cannot fill its balanced replay count")
        offset += 1
    return selected


def scale_examples(partition: dict) -> list[dict]:
    records = partition.get("training_records", [])
    if (
        partition.get("schema") != "umi-studio-openasl-scale-partition/1"
        or partition.get("training_sources") != 120
        or partition.get("training_clips") != 240
        or partition.get("sealed_confirmation_sources") != 60
        or partition.get("sealed_confirmation_clips") != 60
        or len(records) != 240
    ):
        raise ValueError("OpenASL scale partition differs")
    return [
        {
            "id": row["example_id"],
            "source": "openasl_scale",
            "signer": row["source_group"],
            "source_group": row["source_group"],
            "youtube_id": row["youtube_id"],
            "reference": row["reference"],
            "stored_temporal_stride": 2,
        }
        for row in records
    ]


def select_schedule(
    replay: list[dict], stem: list[dict], openasl_scale: list[dict]
) -> list[dict]:
    by_signer = defaultdict(list)
    fleurs = []
    for row in replay:
        if row["source"] == "fleurs":
            fleurs.append(row)
        else:
            by_signer[row["signer"]].append(row)
    if set(by_signer) != {1, 2, 3, 4, 5, 8, 9, 11}:
        raise ValueError("OpenASL scale replay lacks a How2Sign signer")
    selected = []
    for signer, rows in sorted(by_signer.items()):
        ordered = sorted(
            rows,
            key=lambda row: stable_key("how2sign", signer, row["source"], row["id"]),
        )
        if len(ordered) < 32:
            raise ValueError(f"OpenASL scale replay lacks signer {signer}")
        selected.extend(ordered[:32])
    if len(fleurs) < COUNTS["fleurs"]:
        raise ValueError("OpenASL scale replay lacks FLEURS examples")
    selected.extend(
        sorted(fleurs, key=lambda row: stable_key("fleurs", row["id"]))[
            : COUNTS["fleurs"]
        ]
    )
    for source in ("asl_stem_original", "asl_stem_supplemental"):
        selected.extend(
            take_balanced(
                [row for row in stem if row["source"] == source], COUNTS[source], source
            )
        )
    if (
        len(openasl_scale) != COUNTS["openasl_scale"]
        or len({row["source_group"] for row in openasl_scale}) != 120
    ):
        raise ValueError("OpenASL scale training examples differ")
    selected.extend(openasl_scale)
    if len(selected) != RECIPE["train_examples"] or len(
        {(row["source"], row["id"]) for row in selected}
    ) != len(selected):
        raise ValueError("OpenASL scale protocol schedule differs")
    return sorted(
        selected, key=lambda row: stable_key("schedule", row["source"], row["id"])
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if (
        not args.partition.is_file()
        or args.output_dir.exists()
        or not args.output_dir.resolve().is_relative_to((ROOT / "runs").resolve())
    ):
        parser.error("OpenASL scale protocol needs a partition and new runs/ output")
    confirmation = json.loads(CONFIRMATION.read_text())
    equivalence = json.loads(MERGED_EQUIVALENCE.read_text())
    if (
        confirmation.get("decision") != "package_candidate"
        or confirmation.get("candidate_adapter_sha256") != sha256(STARTING_ADAPTER)
        or equivalence.get("all_hypotheses_equal") is not True
        or equivalence.get("examples") != 88
    ):
        raise ValueError("confirmed C5 starting candidate is not frozen")
    partition = json.loads(args.partition.read_text())
    replay, stem, source_hashes = load_examples()
    schedule = select_schedule(replay, stem, scale_examples(partition))
    frozen = [
        {
            "position": position,
            "source": row["source"],
            "example_id": row["id"],
            "signer": row["signer"],
            "stored_temporal_stride": row.get("stored_temporal_stride", 1),
        }
        for position, row in enumerate(schedule)
    ]
    args.output_dir.mkdir(parents=True)
    schedule_path = args.output_dir / "schedule.jsonl"
    with schedule_path.open("x", encoding="utf-8") as output:
        for row in frozen:
            output.write(json.dumps(row, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())
    plan = {
        "schema": "umi-studio-c5-openasl-scale-protocol/1",
        "objective": (
            "test whether 120 new independent OpenASL sources break the repeated "
            "two-signer C5 adaptation ceiling"
        ),
        "seed": SEED,
        "recipe": RECIPE,
        "starting_adapter": str(STARTING_ADAPTER.relative_to(ROOT)),
        "starting_adapter_sha256": sha256(STARTING_ADAPTER),
        "baseline_checkpoint_sha256": confirmation["how2sign"]["checkpoint_sha256"],
        "confirmation_decision_sha256": sha256(CONFIRMATION),
        "merged_equivalence_sha256": sha256(MERGED_EQUIVALENCE),
        "scale_partition": str(args.partition.resolve().relative_to(ROOT)),
        "scale_partition_sha256": sha256(args.partition),
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
        "openasl_scale_training_sources": len(
            {row["source_group"] for row in schedule if row["source"] == "openasl_scale"}
        ),
        "source_hashes": source_hashes,
        "cache_binding": {
            "required": True,
            "expected_index": "runs/openasl-scale-v1/stream-cache-stride2.jsonl",
            "expected_cache_dir": "data/openasl-scale/stream-cache-stride2",
            "policy": (
                "Bind only the 240 training records after all receipts and caches exist; "
                "sealed confirmation and reserve records are forbidden from training."
            ),
        },
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
        "confirmation_policy": partition["confirmation_policy"],
        "plan_source_sha256": sha256(Path(__file__)),
        "miner_health_policy": "user_authorized_maintenance",
    }
    plan_path = args.output_dir / "protocol.json"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "protocol_sha256": sha256(plan_path),
                "schedule_sha256": plan["schedule_sha256"],
                "source_counts": plan["source_counts"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
