"""Audit one public FSboard training landmark shard against split metadata."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter
from pathlib import Path

import pyarrow.feather as feather
from tfrecord import example_pb2

from benchmark_fleurs_dev import miner_revision, sha256
from audit_fsboard_coordinate_integrity import PARTS, axis_shifted

ROOT = Path(__file__).resolve().parents[1]
PUBLISHER_NAME = re.compile(r"(.+)_(\d+)\.(\d{1,3})_(\d+)\.(\d{1,3})\.mp4\Z")
TRAIN_SHARD = re.compile(r"daun_v3-train\.arrow-\d{5}-of-00100\Z")


def metadata_clip_name(publisher_name: str) -> str:
    """Convert arrow video_name seconds.millis into metadata millisecond fields."""
    match = PUBLISHER_NAME.fullmatch(publisher_name)
    if match is None:
        raise ValueError(f"unexpected FSboard arrow video name: {publisher_name}")
    prefix, start_s, start_ms, end_s, end_ms = match.groups()
    start = int(start_s) * 1000 + int(start_ms.ljust(3, "0"))
    end = int(end_s) * 1000 + int(end_ms.ljust(3, "0"))
    return f"{prefix}_{start}_{end}.mp4"


def context_bytes(example: example_pb2.SequenceExample, key: str) -> str:
    values = example.context.feature[key].bytes_list.value
    if len(values) != 1:
        raise ValueError(f"FSboard arrow context missing {key}")
    return values[0].decode("utf-8")


def context_frames(example: example_pb2.SequenceExample) -> int:
    values = example.context.feature["num_frames"].int64_list.value
    if len(values) != 1 or not 0 < values[0] <= 4096:
        raise ValueError("FSboard arrow frame count is invalid")
    return int(values[0])


def hand_fraction(example: example_pb2.SequenceExample, frames: int, hand: str) -> float:
    values = example.feature_lists.feature_list[f"x_{hand}"].feature
    if len(values) != frames:
        raise ValueError("FSboard arrow hand stream has a different frame count")
    counts = [len(frame.float_list.value) for frame in values]
    if any(count not in (0, 21) for count in counts):
        raise ValueError("FSboard arrow hand landmarks have an unexpected width")
    return sum(count == 21 for count in counts) / frames


def validate_coordinate_streams(example: example_pb2.SequenceExample, frames: int) -> None:
    widths = {
        "face": {0, 478},
        "left_hand": {0, 21},
        "right_hand": {0, 21},
        "pose": {0, 25, 33},
    }
    for part, allowed in widths.items():
        by_axis = []
        for axis in "xyz":
            values = example.feature_lists.feature_list[f"{axis}_{part}"].feature
            if len(values) != frames:
                raise ValueError(f"FSboard {part} coordinate stream has a different frame count")
            counts = [len(frame.float_list.value) for frame in values]
            if any(count not in allowed for count in counts):
                raise ValueError(f"FSboard {part} coordinate stream has an unexpected width")
            by_axis.append(counts)
        if by_axis[0] != by_axis[1] or by_axis[0] != by_axis[2]:
            raise ValueError(f"FSboard {part} coordinate axes disagree")


def assert_clean_coordinate_streams(example: example_pb2.SequenceExample) -> None:
    for part in PARTS:
        streams = [example.feature_lists.feature_list[f"{axis}_{part}"].feature for axis in "xyz"]
        for x_frame, y_frame, z_frame in zip(*streams, strict=True):
            x = x_frame.float_list.value
            y = y_frame.float_list.value
            z = z_frame.float_list.value
            if (
                len(x) >= 6
                and len(x) == len(y) == len(z)
                and x[1] == y[0]
                and y[1] == z[0]
                and axis_shifted(list(x), list(y), list(z))
            ):
                raise ValueError(f"FSboard {part} coordinate axes are shifted interleaved views")


def audit(
    training: list[dict], validation: list[dict], shard: Path, health_url: str
) -> tuple[dict, list[dict]]:
    train_by_name = {row["clipFilename"]: row for row in training}
    if len(train_by_name) != len(training):
        raise ValueError("FSboard training metadata has duplicate clip names")
    train_signers = {row["signerId"] for row in training}
    validation_signers = {row["signerId"] for row in validation}
    train_phrases = {row["phrase"] for row in training if row["sensitiveContent"] is False}
    validation_phrases = {row["phrase"] for row in validation if row["sensitiveContent"] is False}
    if train_signers & validation_signers or train_phrases & validation_phrases:
        raise ValueError("FSboard training and validation split identities overlap")
    revision = miner_revision(health_url, None)
    serialized = feather.read_table(shard, columns=["serialized"], memory_map=True)["serialized"]
    counts: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    frame_counts = []
    hand_counts = []
    index_rows = []
    seen = set()
    excluded_sensitive = 0
    excluded_other = 0
    for index, value in enumerate(serialized):
        example = example_pb2.SequenceExample.FromString(value.as_py())
        name = metadata_clip_name(context_bytes(example, "video_name"))
        signer = context_bytes(example, "signer_id")
        phrase = context_bytes(example, "prompt")
        row = train_by_name.get(name)
        if (
            row is None
            or name in seen
            or row["signerId"] != signer
            or row["phrase"] != phrase
            or signer in validation_signers
        ):
            raise ValueError(f"FSboard arrow row differs from training metadata: {name}")
        seen.add(name)
        if row["sensitiveContent"] is not False:
            excluded_sensitive += 1
            continue
        if row["signingType"] != "fs":
            excluded_other += 1
            continue
        if phrase in validation_phrases:
            raise ValueError(f"FSboard safe training phrase appears in validation: {name}")
        frames = context_frames(example)
        validate_coordinate_streams(example, frames)
        assert_clean_coordinate_streams(example)
        left = hand_fraction(example, frames, "left_hand")
        right = hand_fraction(example, frames, "right_hand")
        frame_counts.append(frames)
        hand_counts.append(max(left, right))
        index_rows.append({
            "schema": "umi-studio-fsboard-train-shard-row/1",
            "shard_row": index,
            "clip_filename": name,
            "reference": phrase,
            "signer_id": signer,
            "frames": frames,
            "left_hand_detection_fraction": left,
            "right_hand_detection_fraction": right,
        })
        counts[signer] += 1
        categories[row["collectionId"]] += 1
        if (index + 1) % 64 == 0:
            revision = miner_revision(health_url, revision)
    revision = miner_revision(health_url, revision)
    if len(serialized) != len(seen):
        raise ValueError("FSboard arrow shard has duplicate training clips")
    if not frame_counts:
        raise ValueError("FSboard arrow shard has no eligible training clips")
    summary = {
        "schema": "umi-studio-fsboard-train-arrow-audit/1",
        "source_split": "FSboard daun_v3 train",
        "test_used": False,
        "shard_name": shard.name,
        "shard_sha256": sha256(shard),
        "training_metadata_rows": len(training),
        "training_metadata_signers": len(train_signers),
        "validation_metadata_rows": len(validation),
        "validation_metadata_signers": len(validation_signers),
        "train_validation_signer_overlap": 0,
        "train_validation_exact_phrase_overlap": 0,
        "coordinate_integrity": "verified_clean",
        "shard_examples": len(seen),
        "shard_safe_examples": len(frame_counts),
        "shard_excluded_sensitive_examples": excluded_sensitive,
        "shard_excluded_other_examples": excluded_other,
        "shard_signers": len(counts),
        "shard_examples_by_collection": dict(sorted(categories.items())),
        "shard_min_examples_per_signer": min(counts.values()),
        "shard_max_examples_per_signer": max(counts.values()),
        "shard_frame_count_median": statistics.median(frame_counts),
        "shard_max_hand_detection_fraction_median": statistics.median(hand_counts),
        "shard_examples_hand_detection_at_least_half": sum(fraction >= 0.5 for fraction in hand_counts),
        "miner_model_revision": revision,
        "model_training_started": False,
    }
    return summary, index_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-metadata", type=Path, required=True)
    parser.add_argument("--validation-metadata", type=Path, required=True)
    parser.add_argument("--shard", type=Path, required=True)
    parser.add_argument("--miner-health-url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--index-output", type=Path, required=True)
    args = parser.parse_args()
    if (
        not TRAIN_SHARD.fullmatch(args.shard.name)
        or args.output.exists()
        or args.index_output.exists()
        or not args.output.resolve().is_relative_to(ROOT / "runs")
        or not args.index_output.resolve().is_relative_to(ROOT / "runs")
        or args.output.resolve() == args.index_output.resolve()
    ):
        parser.error("expected one train shard and a new output under runs/")
    training = json.loads(args.training_metadata.read_text())
    validation = json.loads(args.validation_metadata.read_text())
    result, index_rows = audit(training, validation, args.shard, args.miner_health_url)
    result["training_metadata_sha256"] = sha256(args.training_metadata)
    result["validation_metadata_sha256"] = sha256(args.validation_metadata)
    args.index_output.parent.mkdir(parents=True, exist_ok=True)
    with args.index_output.open("x", encoding="utf-8") as stream:
        for row in index_rows:
            stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    result["training_index_sha256"] = sha256(args.index_output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({key: result[key] for key in (
        "shard_examples", "shard_safe_examples", "shard_excluded_sensitive_examples",
        "shard_signers", "shard_examples_hand_detection_at_least_half"
    )}, sort_keys=True))


if __name__ == "__main__":
    main()
