"""Verify every full-corpus raw clip, frame bound and complete language target on CPU."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
from benchmark_fleurs_dev import sha256
from c5_full_coverage_mps import load_training
from tiny_rgb_fit import ROOT, private_path
from tiny_rgb_full_corpus import FULL_RECIPE, read_rows, validate_split


def frame_indices(count: int, fps: float, target: float, cap: int) -> np.ndarray:
    if count < 1 or not math.isfinite(fps) or fps < target:
        raise ValueError("full RGB source cannot support complete temporal sampling")
    indices = np.unique(np.floor(np.arange(0, count, fps / target)).astype(np.int64))
    if not 1 <= len(indices) <= cap:
        raise ValueError("full RGB frame limit would truncate signing")
    return indices


def main() -> None:
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OPENCV_FFMPEG_THREADS"):
        os.environ[key] = "2"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import cv2
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() or not output.is_relative_to(ROOT / "runs"):
        raise ValueError("full RGB header audit needs a new private runs/ directory")
    proposal = json.loads(args.proposal.read_text())
    if proposal["recipe"] != FULL_RECIPE or not proposal["planning_only"]:
        raise ValueError("full RGB header audit proposal recipe changed")
    dependencies = {**proposal["dependency_sha256"], **proposal["language_asset_sha256"]}
    if any(sha256(ROOT / name) != digest for name, digest in dependencies.items()):
        raise ValueError("full RGB header audit dependencies changed")
    corpus = ROOT / "runs/c5-all-source-corpus-v2"
    approved = load_training(corpus / "audit.json", corpus / "training-streams.jsonl")
    training = read_rows(ROOT / proposal["training_inventory"])
    if {(r["source"], r["example_id"]) for r in training} != {
        (r["source"], r["id"]) for r in approved
    }:
        raise ValueError("full RGB header audit eligibility changed")
    evaluation = proposal["evaluation"]
    validate_split(training, evaluation)
    tokenizer = AutoTokenizer.from_pretrained(
        ROOT / proposal["language_path"], local_files_only=True
    )
    cv2.setNumThreads(2)
    output.mkdir(parents=True)
    records = []
    ledger = output / "headers.jsonl"
    with ledger.open("x") as stream:
        for index, row in enumerate(training + evaluation):
            video = private_path(row["video"], "data")
            if sha256(video) != row["video_sha256"]:
                raise ValueError("full RGB header audit clip bytes changed")
            capture = cv2.VideoCapture(str(video), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 2])
            try:
                if not capture.isOpened() or int(capture.get(cv2.CAP_PROP_N_THREADS)) != 2:
                    raise ValueError("full RGB audit decoder cannot honor CPU2")
                fps = float(capture.get(cv2.CAP_PROP_FPS))
                count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            finally:
                capture.release()
            indices = frame_indices(
                count, fps, FULL_RECIPE["target_fps"], FULL_RECIPE["max_frames"]
            )
            if min(height, width) < 16 or abs(fps - row["video_fps"]) > 0.02:
                raise ValueError("full RGB source shape/fps differs from the approved binding")
            target = tokenizer(row["reference"], truncation=False).input_ids
            if not 1 <= len(target) <= FULL_RECIPE["max_text_tokens"]:
                raise ValueError("full RGB language bound would truncate a target")
            record = {
                "source": row["source"],
                "example_id": row["example_id"],
                "role": row["role"],
                "video_sha256": row["video_sha256"],
                "source_fps": fps,
                "source_frames": count,
                "source_shape": [height, width],
                "sampled_frames": len(indices),
                "frame_indices_sha256": __import__("hashlib").sha256(indices.tobytes()).hexdigest(),
                "target_tokens": len(target),
            }
            records.append(record)
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
            if (index + 1) % 512 == 0:
                print(
                    json.dumps(
                        {"verified_headers": index + 1, "total": len(training) + len(evaluation)}
                    ),
                    flush=True,
                )
    receipt = {
        "schema": "umi-studio-tiny-rgb-full-header-audit/1",
        "proposal_sha256": sha256(args.proposal),
        "source_sha256": sha256(Path(__file__)),
        "headers_sha256": sha256(ledger),
        "training_headers": len(training),
        "evaluation_headers": len(evaluation),
        "maximum_sampled_frames": max(r["sampled_frames"] for r in records),
        "maximum_target_tokens": max(r["target_tokens"] for r in records),
        "minimum_sampled_frames": min(r["sampled_frames"] for r in records),
        "clip_bytes_verified": True,
        "complete_targets": True,
        "pixels_decoded": False,
        "gpu_used": False,
        "pretrained_weights_loaded": False,
        "qualification": (
            "All approved clip bytes, headers and complete targets only. Pixel decode, model "
            "preflight, training and held-out quality remain required."
        ),
    }
    output.joinpath("receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
