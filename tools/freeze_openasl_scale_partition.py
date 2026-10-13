"""Freeze source-disjoint training and sealed confirmation partitions for OpenASL scale."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from benchmark_fleurs_dev import sha256

ROOT = Path(__file__).resolve().parents[1]
SEED = 211
TRAIN_SOURCES = 120
CONFIRMATION_SOURCES = 60


def stable_key(*values: object) -> str:
    return hashlib.sha256(
        "\0".join(str(value) for value in (SEED, *values)).encode()
    ).hexdigest()


def partition_rows(rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    by_source = defaultdict(list)
    for row in rows:
        by_source[row["youtube_id"]].append(row)
    if len(by_source) != TRAIN_SOURCES + CONFIRMATION_SOURCES or any(
        len(source_rows) != 2 for source_rows in by_source.values()
    ):
        raise ValueError("OpenASL scale partition requires 180 two-clip sources")
    ordered_sources = sorted(by_source, key=lambda source: stable_key("source", source))
    training_sources = set(ordered_sources[:TRAIN_SOURCES])
    confirmation_sources = set(ordered_sources[TRAIN_SOURCES:])
    training = sorted(
        [row for row in rows if row["youtube_id"] in training_sources],
        key=lambda row: stable_key("train", row["youtube_id"], row["example_id"]),
    )
    confirmation = []
    reserve = []
    for source in sorted(confirmation_sources, key=lambda item: stable_key("confirm", item)):
        source_rows = sorted(
            by_source[source],
            key=lambda row: stable_key("clip", source, row["example_id"]),
        )
        confirmation.append(source_rows[0])
        reserve.append(source_rows[1])
    if (
        len(training) != 240
        or len(confirmation) != 60
        or len(reserve) != 60
        or {row["youtube_id"] for row in training}
        & {row["youtube_id"] for row in (*confirmation, *reserve)}
    ):
        raise ValueError("OpenASL scale partition overlaps or has another size")
    return training, confirmation, reserve


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        not args.selection.is_file()
        or args.output.exists()
        or not args.output.resolve().is_relative_to((ROOT / "runs").resolve())
    ):
        parser.error("OpenASL partition requires a frozen selection and new runs/ output")
    selection = json.loads(args.selection.read_text())
    rows = selection.get("records", [])
    if (
        selection.get("schema") != "umi-studio-openasl-scale-selection/1"
        or selection.get("selected_new_sources") != 180
        or selection.get("selected_new_clips") != 360
        or len(rows) != 360
    ):
        raise ValueError("OpenASL scale selection differs")
    training, confirmation, reserve = partition_rows(rows)
    result = {
        "schema": "umi-studio-openasl-scale-partition/1",
        "seed": SEED,
        "selection": str(args.selection.resolve().relative_to(ROOT)),
        "selection_sha256": sha256(args.selection),
        "training_sources": len({row["youtube_id"] for row in training}),
        "training_clips": len(training),
        "training_duration_hours": sum(row["duration_sec"] for row in training) / 3600,
        "sealed_confirmation_sources": len(
            {row["youtube_id"] for row in confirmation}
        ),
        "sealed_confirmation_clips": len(confirmation),
        "sealed_confirmation_duration_hours": sum(
            row["duration_sec"] for row in confirmation
        )
        / 3600,
        "unused_reserve_clips": len(reserve),
        "source_disjoint": True,
        "training_records": training,
        "sealed_confirmation_records": confirmation,
        "unused_reserve_records": reserve,
        "confirmation_policy": (
            "Do not decode sealed_confirmation_records or unused_reserve_records during "
            "training, checkpoint selection or open tuning. Evaluate the single frozen "
            "confirmation clip per source once, only after a candidate passes every open gate."
        ),
        "freezer_sha256": sha256(Path(__file__)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "training_sources": result["training_sources"],
                "training_clips": result["training_clips"],
                "confirmation_sources": result["sealed_confirmation_sources"],
                "confirmation_clips": result["sealed_confirmation_clips"],
                "reserve_clips": result["unused_reserve_clips"],
                "output_sha256": sha256(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
