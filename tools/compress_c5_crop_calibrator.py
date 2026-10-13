"""Compress a dense crop calibrator into low-rank residual maps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open
from safetensors.numpy import load_file, save_file

from benchmark_fleurs_dev import sha256
from c5_crop_calibrator import SCHEMA

LOW_RANK_SCHEMA = "umi-studio-c5-crop-calibrator-low-rank/1"


def factors(weight: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
    delta = weight.astype(np.float64) - np.eye(384)
    left, singular, right = np.linalg.svd(delta, full_matrices=False)
    return (
        (left[:, :rank] * singular[:rank]).astype(np.float32),
        right[:rank].astype(np.float32),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=32)
    args = parser.parse_args()
    if args.output.exists() or args.report.exists() or not 1 <= args.rank <= 384:
        parser.error("low-rank output or rank is invalid")
    with safe_open(str(args.input), framework="numpy") as handle:
        metadata = handle.metadata()
    if metadata.get("schema") != SCHEMA:
        raise ValueError("dense crop calibrator schema differs")
    tensors = load_file(str(args.input))
    hand_left, hand_right = factors(tensors["hand_weight"], args.rank)
    face_left, face_right = factors(tensors["face_weight"], args.rank)
    output_metadata = {
        **metadata,
        "schema": LOW_RANK_SCHEMA,
        "dense_calibrator_sha256": sha256(args.input),
        "rank": str(args.rank),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "hand_left": hand_left,
            "hand_right": hand_right,
            "hand_bias": tensors["hand_bias"],
            "face_left": face_left,
            "face_right": face_right,
            "face_bias": tensors["face_bias"],
        },
        str(args.output),
        metadata=output_metadata,
    )
    report = {
        "schema": "umi-studio-c5-crop-calibrator-compression/1",
        "dense_calibrator_sha256": sha256(args.input),
        "low_rank_calibrator_sha256": sha256(args.output),
        "rank": args.rank,
        "dense_parameter_count": 2 * (384 * 384 + 384),
        "low_rank_parameter_count": 2 * (384 * args.rank * 2 + 384),
        "dense_bytes": args.input.stat().st_size,
        "low_rank_bytes": args.output.stat().st_size,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
