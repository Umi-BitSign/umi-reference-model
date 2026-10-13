"""Bind the frozen OpenASL-scale protocol to completed stride-2 caches."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from benchmark_fleurs_dev import sha256
from freeze_c5_openasl_scale_protocol import RECIPE, ROOT


def load_scale_examples(
    index: Path, cache_dir: Path, partition: dict
) -> tuple[list[dict], dict[str, str]]:
    rows = [json.loads(line) for line in index.read_text().splitlines()]
    by_example = {row.get("example_id"): row for row in rows}
    training_records = partition.get("training_records", [])
    training_ids = {row["example_id"] for row in training_records}
    sealed_ids = {
        row["example_id"]
        for key in ("sealed_confirmation_records", "unused_reserve_records")
        for row in partition.get(key, [])
    }
    if (
        partition.get("schema") != "umi-studio-openasl-scale-partition/1"
        or len(rows) != 360
        or len(by_example) != 360
        or len(training_ids) != 240
        or len(sealed_ids) != 120
        or training_ids & sealed_ids
        or set(by_example) != training_ids | sealed_ids
    ):
        raise ValueError("OpenASL scale cache partition is incomplete or overlaps")
    examples = []
    for record in training_records:
        row = by_example[record["example_id"]]
        cache = cache_dir / f"{row['clip_id']}.npz"
        if (
            row.get("schema") != "umi-studio-openasl-stream-cache/1"
            or row.get("split") != "training"
            or row.get("temporal_stride") != 2
            or row.get("spatial_policy")
            != "full_source_frame_preserve_aspect_ratio"
            or row.get("youtube_id") != record["youtube_id"]
            or row.get("source_group") != record["source_group"]
            or row.get("reference") != record["reference"]
            or row.get("frames") != row.get("encoder_shape", [None])[0]
            or not cache.is_file()
            or sha256(cache) != row.get("encoder_cache_sha256")
        ):
            raise ValueError(f"OpenASL scale stream row differs: {record['example_id']}")
        examples.append(
            {
                "id": record["example_id"],
                "source": "openasl_scale",
                "signer": record["source_group"],
                "source_group": record["source_group"],
                "youtube_id": record["youtube_id"],
                "reference": record["reference"],
                "cache": str(cache),
                "cache_sha256": row["encoder_cache_sha256"],
                "frames": row["frames"],
                "stored_temporal_stride": 2,
            }
        )
    return examples, {"openasl_scale_index_sha256": sha256(index)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if (
        not all(path.is_file() for path in (args.protocol, args.index))
        or not args.cache_dir.is_dir()
        or args.output_dir.exists()
        or not args.output_dir.resolve().is_relative_to((ROOT / "runs").resolve())
    ):
        parser.error("cache binding needs complete inputs and a new runs/ output")
    protocol = json.loads(args.protocol.read_text())
    if (
        protocol.get("schema") != "umi-studio-c5-openasl-scale-protocol/1"
        or protocol.get("recipe") != RECIPE
        or protocol.get("cache_binding", {}).get("required") is not True
    ):
        raise ValueError("OpenASL scale protocol differs")
    partition_path = ROOT / protocol["scale_partition"]
    partition = json.loads(partition_path.read_text())
    if sha256(partition_path) != protocol["scale_partition_sha256"]:
        raise ValueError("OpenASL scale partition changed")
    examples, scale_hashes = load_scale_examples(args.index, args.cache_dir, partition)
    if len(examples) != 240 or len({row["source_group"] for row in examples}) != 120:
        raise ValueError("OpenASL scale training cache binding differs")
    source_schedule = args.protocol.with_name("schedule.jsonl")
    if sha256(source_schedule) != protocol["schedule_sha256"]:
        raise ValueError("OpenASL scale protocol schedule changed")
    args.output_dir.mkdir(parents=True)
    schedule_path = args.output_dir / "schedule.jsonl"
    shutil.copyfile(source_schedule, schedule_path)
    plan = {
        "schema": "umi-studio-c5-frozen-replay-plan/1",
        "objective": protocol["objective"],
        "seed": protocol["seed"],
        "recipe": protocol["recipe"],
        "starting_adapter": protocol["starting_adapter"],
        "starting_adapter_sha256": protocol["starting_adapter_sha256"],
        "baseline_checkpoint_sha256": protocol["baseline_checkpoint_sha256"],
        "confirmation_decision_sha256": protocol["confirmation_decision_sha256"],
        "merged_equivalence_sha256": protocol["merged_equivalence_sha256"],
        "protocol": str(args.protocol.resolve().relative_to(ROOT)),
        "protocol_sha256": sha256(args.protocol),
        "scale_partition": protocol["scale_partition"],
        "scale_partition_sha256": protocol["scale_partition_sha256"],
        "scale_index": str(args.index.resolve().relative_to(ROOT)),
        "scale_cache_dir": str(args.cache_dir.resolve().relative_to(ROOT)),
        "schedule_sha256": sha256(schedule_path),
        "source_counts": protocol["source_counts"],
        "how2sign_signer_counts": protocol["how2sign_signer_counts"],
        "openasl_scale_training_sources": protocol["openasl_scale_training_sources"],
        "source_hashes": {**protocol["source_hashes"], **scale_hashes},
        "open_tuning": protocol["open_tuning"],
        "screen": protocol["screen"],
        "confirmation_policy": protocol["confirmation_policy"],
        "trainer_source_sha256": sha256(ROOT / "tools/train_c5_frozen_replay.py"),
        "binder_source_sha256": sha256(Path(__file__)),
        "miner_health_policy": "user_authorized_maintenance",
    }
    plan_path = args.output_dir / "plan.json"
    with plan_path.open("x", encoding="utf-8") as output:
        json.dump(plan, output, indent=2, sort_keys=True)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    print(
        json.dumps(
            {
                "plan_sha256": sha256(plan_path),
                "scale_index_sha256": scale_hashes["openasl_scale_index_sha256"],
                "training_examples": len(examples),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
