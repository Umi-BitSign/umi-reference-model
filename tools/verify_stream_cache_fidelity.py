"""Verify that visual-stream caches preserve the frozen tuning encoder states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from benchmark_fleurs_dev import miner_revision, sha256
from cache_fleurs_encoder import checked_stream_features

COMMON_FIELDS = (
    "schema",
    "split",
    "selection_sha256",
    "reference",
    "signer",
    "baseline_identity_sha256",
    "baseline_runtime_sha256",
    "encoder_shape",
    "encoder_dtype",
    "video_decode_strategy",
)
SOURCE_FIELDS = {
    "umi-studio-fleurs-dev-encoder-cache-example/1": (
        "example_id",
        "document_id",
        "source_receipt_sha256",
        "source_video",
        "source_video_sha256",
        "clip_sha256",
    ),
    "umi-studio-how2sign-encoder-cache-example/1": (
        "sentence_name",
        "clip_receipts_sha256",
        "source_clip_sha256",
        "resampled_clip_sha256",
    ),
}


def verify_rows(
    original_rows: list[dict],
    stream_rows: list[dict],
    original_dir: Path,
    stream_dir: Path,
    partition: dict,
) -> dict:
    if not original_rows or len(original_rows) != len(stream_rows):
        raise ValueError("stream and original tuning index counts differ")
    schema = original_rows[0]["schema"]
    if schema not in SOURCE_FIELDS or any(
        row["schema"] != schema for row in [*original_rows, *stream_rows]
    ):
        raise ValueError("stream tuning index has an unsupported schema")
    if schema == "umi-studio-fleurs-dev-encoder-cache-example/1":
        if any(row["split"] != "tuning" for row in [*original_rows, *stream_rows]):
            raise ValueError("FLEURS stream comparison is not tuning-only")
    elif any(row["split"] != "validation" for row in [*original_rows, *stream_rows]):
        raise ValueError("How2Sign stream comparison is not validation-only")
    identifier = SOURCE_FIELDS[schema][0]
    identifiers = [row[identifier] for row in original_rows]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("tuning index repeats an example")
    if schema == "umi-studio-fleurs-dev-encoder-cache-example/1":
        if (
            partition.get("schema") != "umi-studio-fleurs-dev-evaluation-partition/1"
            or partition.get("selection_sha256") != original_rows[0]["selection_sha256"]
            or len(identifiers) != partition.get("tuning_examples")
            or {row["document_id"] for row in original_rows}
            != set(partition.get("tuning_documents", []))
        ):
            raise ValueError("FLEURS stream comparison is outside the frozen tuning partition")
    elif (
        partition.get("schema") != "umi-studio-how2sign-validation-partition/1"
        or partition.get("selection_sha256") != original_rows[0]["selection_sha256"]
        or identifiers != partition.get("tuning_sentence_names")
    ):
        raise ValueError("How2Sign stream comparison is outside the frozen tuning partition")
    for original, streamed in zip(original_rows, stream_rows, strict=True):
        for field in (*COMMON_FIELDS, *SOURCE_FIELDS[schema]):
            if field != "video_decode_strategy" and (
                field not in original or field not in streamed
            ):
                raise ValueError(f"stream tuning source lacks field: {field}")
            old_value = original.get(field)
            new_value = streamed.get(field)
            if field == "video_decode_strategy" and schema == (
                "umi-studio-fleurs-dev-encoder-cache-example/1"
            ):
                old_value = old_value or "baseline_seek_per_frame"
                new_value = new_value or "baseline_seek_per_frame"
            elif field == "video_decode_strategy" and (
                old_value is None or new_value is None
            ):
                raise ValueError("How2Sign stream cache lacks decode strategy")
            if old_value != new_value:
                raise ValueError(f"stream tuning source differs: {field}")
        if streamed.get("stream_feature_dtype") != "float32":
            raise ValueError("stream cache metadata lacks float32 features")
        name = original[identifier]
        source = original_dir / f"{name}.npz"
        target = stream_dir / f"{name}.npz"
        if (
            sha256(source) != original["encoder_cache_sha256"]
            or sha256(target) != streamed["encoder_cache_sha256"]
        ):
            raise ValueError(f"stream comparison cache checksum differs: {name}")
        with np.load(source, allow_pickle=False) as old, np.load(
            target, allow_pickle=False
        ) as new:
            if set(old.files) != {"encoder_hidden_state"} or set(new.files) != {
                "encoder_hidden_state",
                "face_features",
                "left_hand_features",
                "right_hand_features",
                "pose_features",
            }:
                raise ValueError(f"stream cache payload has unexpected fields: {name}")
            prior = old["encoder_hidden_state"]
            current = new["encoder_hidden_state"]
            if (
                prior.shape != tuple(original["encoder_shape"])
                or prior.dtype != np.float32
                or not np.array_equal(prior, current)
            ):
                raise ValueError(f"frozen encoder state changed: {name}")
            streams = checked_stream_features(
                {key: new[key] for key in new.files if key != "encoder_hidden_state"},
                prior.shape[0],
            )
            if streamed.get("stream_feature_shapes") != {
                key: list(value.shape) for key, value in streams.items()
            }:
                raise ValueError(f"stream cache metadata shapes differ: {name}")
    return {"schema": "umi-studio-stream-cache-fidelity/1", "examples": len(identifiers)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-index", type=Path, required=True)
    parser.add_argument("--stream-index", type=Path, required=True)
    parser.add_argument("--original-dir", type=Path, required=True)
    parser.add_argument("--stream-dir", type=Path, required=True)
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--miner-health-url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    revision = miner_revision(args.miner_health_url, None)
    original_rows = [json.loads(line) for line in args.original_index.read_text().splitlines()]
    stream_rows = [json.loads(line) for line in args.stream_index.read_text().splitlines()]
    partition = json.loads(args.partition.read_text())
    if (
        partition.get("schema") == "umi-studio-how2sign-validation-partition/1"
        and partition.get("tuning_index_sha256") != sha256(args.original_index)
    ):
        raise ValueError("How2Sign original tuning index differs from the frozen partition")
    result = verify_rows(original_rows, stream_rows, args.original_dir, args.stream_dir, partition)
    result.update(
        original_index_sha256=sha256(args.original_index),
        stream_index_sha256=sha256(args.stream_index),
        partition_sha256=sha256(args.partition),
        original_cache_dir=str(args.original_dir.resolve()),
        stream_cache_dir=str(args.stream_dir.resolve()),
        miner_model_revision=miner_revision(args.miner_health_url, revision),
    )
    if args.output.exists():
        if json.loads(args.output.read_text()) != result:
            raise ValueError("frozen stream fidelity record differs")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as output:
            json.dump(result, output, sort_keys=True, indent=2)
            output.write("\n")
    print(f"verified {result['examples']} byte-identical encoder states with visual streams")


if __name__ == "__main__":
    main()
