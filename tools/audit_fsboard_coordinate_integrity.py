"""Fingerprint coordinate-axis corruption in published FSboard Arrow shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.feather as feather
from tfrecord import example_pb2

from benchmark_fleurs_dev import miner_revision, sha256

ROOT = Path(__file__).resolve().parents[1]
PARTS = ("face", "left_hand", "right_hand", "pose")


def axis_shifted(x: list[float], y: list[float], z: list[float]) -> bool:
    """Detect three stride-one views of the same interleaved buffer."""
    return (
        len(x) == len(y) == len(z)
        and len(x) >= 6
        and len(set(x)) >= 6
        and x[1:] == y[:-1]
        and y[1:] == z[:-1]
    )


def first_observed_frame(example: example_pb2.SequenceExample, part: str) -> tuple[int, list[float], list[float], list[float]] | None:
    streams = [example.feature_lists.feature_list[f"{axis}_{part}"].feature for axis in "xyz"]
    if len({len(stream) for stream in streams}) != 1:
        raise ValueError("coordinate streams have different frame counts")
    for frame in range(len(streams[0])):
        axes = [list(streams[axis][frame].float_list.value) for axis in range(3)]
        if any(axes):
            if not all(axes):
                raise ValueError("coordinate stream is present on only some axes")
            return frame, axes[0], axes[1], axes[2]
    return None


def sample_indices(count: int, samples: int) -> list[int]:
    if count < 1 or samples < 1:
        raise ValueError("positive shard and sample sizes are required")
    return sorted({min(count - 1, (index * count) // samples) for index in range(samples)})


def audit_shard(shard: Path, samples: int) -> dict:
    serialized = feather.read_table(shard, columns=["serialized"], memory_map=True)["serialized"]
    indices = sample_indices(len(serialized), samples)
    counts = {part: {"observed": 0, "shifted": 0} for part in PARTS}
    witnesses = []
    for index in indices:
        example = example_pb2.SequenceExample.FromString(serialized[index].as_py())
        witness = {"source_row": index, "parts": {}}
        for part in PARTS:
            observed = first_observed_frame(example, part)
            if observed is None:
                continue
            frame, x, y, z = observed
            shifted = axis_shifted(x, y, z)
            counts[part]["observed"] += 1
            counts[part]["shifted"] += int(shifted)
            witness["parts"][part] = {
                "frame": frame,
                "width": len(x),
                "shifted": shifted,
            }
        witnesses.append(witness)
    return {
        "shard_name": shard.name,
        "shard_sha256": sha256(shard),
        "source_rows": len(serialized),
        "sampled_rows": indices,
        "counts": counts,
        "witnesses": witnesses,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=Path, action="append", required=True)
    parser.add_argument("--samples-per-shard", type=int, default=32)
    parser.add_argument("--miner-health-url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        not 1 <= args.samples_per_shard <= 256
        or args.output.exists()
        or not args.output.resolve().is_relative_to(ROOT / "runs")
        or len({path.resolve() for path in args.shard}) != len(args.shard)
    ):
        parser.error("use distinct shards and a new output under runs/")
    revision = miner_revision(args.miner_health_url, None)
    shards = []
    for shard in args.shard:
        shards.append(audit_shard(shard, args.samples_per_shard))
        revision = miner_revision(args.miner_health_url, revision)
    result = {
        "schema": "umi-studio-fsboard-arrow-coordinate-integrity/1",
        "source_split": "FSboard daun_v3 train",
        "validation_or_test_used": False,
        "samples_per_shard": args.samples_per_shard,
        "shards": shards,
        "all_sampled_observed_face_frames_shifted": all(
            shard["counts"]["face"]["observed"] > 0
            and shard["counts"]["face"]["observed"] == shard["counts"]["face"]["shifted"]
            for shard in shards
        ),
        "audit_script_sha256": sha256(Path(__file__)),
        "miner_model_revision": revision,
        "live_miner_changed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({
        "shards": len(shards),
        "face_shifted": sum(shard["counts"]["face"]["shifted"] for shard in shards),
        "face_observed": sum(shard["counts"]["face"]["observed"] for shard in shards),
        "all_face_shifted": result["all_sampled_observed_face_frames_shifted"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
