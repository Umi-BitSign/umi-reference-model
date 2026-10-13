"""Compare FLEURS training visual streams with frozen encoder states."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np

from benchmark_fleurs_dev import miner_revision, sha256
from cache_fleurs_encoder import checked_stream_features
from verify_stream_cache_fidelity import COMMON_FIELDS, SOURCE_FIELDS


def verify_training_rows(
    original_rows: list[dict],
    stream_rows: list[dict],
    original_dir: Path,
    stream_dir: Path,
    selection: dict,
    health_url: str | None = None,
) -> dict:
    selected = selection.get("training", [])
    selected_ids = [row["example_id"] for row in selected]
    old_ids = [row["example_id"] for row in original_rows]
    new_ids = [row["example_id"] for row in stream_rows]
    if (
        selection.get("schema") != "umi-studio-fleurs-dev-split/1"
        or len(selected_ids) != 313
        or len(set(selected_ids)) != 313
        or len(old_ids) != 313
        or len(set(old_ids)) != 313
        or set(old_ids) != set(selected_ids)
        or not stream_rows
        or len(stream_rows) > 313
        or new_ids != selected_ids[: len(stream_rows)]
    ):
        raise ValueError("FLEURS training streams do not follow the frozen selection")
    selection_hash = None
    old_by_id = {row["example_id"]: row for row in original_rows}
    prior_decode_strategies = Counter()
    revision = miner_revision(health_url, None) if health_url else None
    fields = (*COMMON_FIELDS, *SOURCE_FIELDS["umi-studio-fleurs-dev-encoder-cache-example/1"])
    for position, streamed in enumerate(stream_rows):
        if health_url and position % 16 == 0:
            revision = miner_revision(health_url, revision)
        identifier = streamed["example_id"]
        original = old_by_id[identifier]
        if (
            original.get("schema") != "umi-studio-fleurs-dev-encoder-cache-example/1"
            or streamed.get("schema") != original["schema"]
            or original.get("split") != "training"
            or streamed.get("split") != "training"
            or original.get("signer") != 3
            or streamed.get("signer") != 3
            or streamed.get("stream_feature_dtype") != "float32"
        ):
            raise ValueError(f"FLEURS training stream row is invalid: {identifier}")
        if selection_hash is None:
            selection_hash = original["selection_sha256"]
        for field in fields:
            old_value = original.get(field)
            new_value = streamed.get(field)
            if field == "video_decode_strategy":
                if old_value not in {
                    None,
                    "baseline_seek_per_frame",
                    "opencv_sequential_full_range",
                } or new_value != "baseline_seek_per_frame":
                    raise ValueError(f"FLEURS training decode strategy is invalid: {identifier}")
                prior_decode_strategies[old_value or "unspecified"] += 1
                continue
            elif field not in original or field not in streamed:
                raise ValueError(f"FLEURS training source lacks {field}: {identifier}")
            if old_value != new_value:
                raise ValueError(f"FLEURS training source differs: {field}: {identifier}")
        old_path = original_dir / f"{identifier}.npz"
        new_path = stream_dir / f"{identifier}.npz"
        if (
            sha256(old_path) != original["encoder_cache_sha256"]
            or sha256(new_path) != streamed["encoder_cache_sha256"]
        ):
            raise ValueError(f"FLEURS training cache checksum differs: {identifier}")
        with np.load(old_path, allow_pickle=False) as old, np.load(
            new_path, allow_pickle=False
        ) as new:
            if set(old.files) != {"encoder_hidden_state"} or set(new.files) != {
                "encoder_hidden_state",
                "face_features",
                "left_hand_features",
                "right_hand_features",
                "pose_features",
            }:
                raise ValueError(f"FLEURS training cache fields differ: {identifier}")
            prior = old["encoder_hidden_state"]
            current = new["encoder_hidden_state"]
            if (
                prior.shape != tuple(original["encoder_shape"])
                or prior.dtype != np.float32
                or not np.array_equal(prior, current)
            ):
                raise ValueError(f"FLEURS frozen encoder state changed: {identifier}")
            values = {
                name: new[name]
                for name in ("face_features", "left_hand_features", "right_hand_features", "pose_features")
            }
            if any(value.dtype != np.float32 for value in values.values()):
                raise ValueError(f"FLEURS visual stream dtype differs: {identifier}")
            streams = checked_stream_features(values, prior.shape[0])
            if streamed.get("stream_feature_shapes") != {
                name: list(value.shape) for name, value in streams.items()
            }:
                raise ValueError(f"FLEURS training stream shapes differ: {identifier}")
    if health_url:
        miner_revision(health_url, revision)
    return {
        "schema": "umi-studio-fleurs-training-stream-fidelity/1",
        "examples_verified": len(stream_rows),
        "complete": len(stream_rows) == 313,
        "selection_sha256": selection_hash,
        "prior_decode_strategies": dict(sorted(prior_decode_strategies.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-index", type=Path, required=True)
    parser.add_argument("--stream-index", type=Path, required=True)
    parser.add_argument("--original-dir", type=Path, required=True)
    parser.add_argument("--stream-dir", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--miner-health-url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.output.exists() or not args.output.resolve().is_relative_to(root / "runs"):
        parser.error("fidelity output must be new and under the run ledger")
    original_rows = [json.loads(line) for line in args.original_index.read_text().splitlines()]
    stream_rows = [json.loads(line) for line in args.stream_index.read_text().splitlines()]
    result = verify_training_rows(
        original_rows,
        stream_rows,
        args.original_dir,
        args.stream_dir,
        json.loads(args.selection.read_text()),
        args.miner_health_url,
    )
    if not result["complete"] or result["selection_sha256"] != sha256(args.selection):
        raise ValueError("FLEURS training stream fidelity is not source-complete")
    result.update(
        original_index_sha256=sha256(args.original_index),
        stream_index_sha256=sha256(args.stream_index),
        selection_sha256=sha256(args.selection),
        original_cache_dir=str(args.original_dir.resolve()),
        stream_cache_dir=str(args.stream_dir.resolve()),
        verifier_source_sha256=sha256(Path(__file__)),
        miner_model_revision=miner_revision(args.miner_health_url, None),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".partial.json")
    if temporary.exists():
        parser.error("unfinished FLEURS training fidelity output needs inspection")
    try:
        with temporary.open("x", encoding="utf-8") as output:
            json.dump(result, output, sort_keys=True, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        temporary.rename(args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"verified {result['examples_verified']} byte-identical FLEURS training states")


if __name__ == "__main__":
    main()
