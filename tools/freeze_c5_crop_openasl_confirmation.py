"""Freeze a one-shot, source-diverse OpenASL crop confirmation plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

from benchmark_fleurs_dev import sha256

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "runs/openasl-v1/stream-cache-stride2.jsonl"
LANDMARK_RECEIPTS = ROOT / "runs/openasl-v1/landmark-receipts.jsonl"
CLIP_DIR = ROOT / "data/openasl/clips-25fps"
LANDMARK_DIR = ROOT / "data/openasl/landmarks"
PARENT_BUNDLE = ROOT / "artifacts/c5-corrective-stride2-confirmed"
CANDIDATE_BUNDLE = ROOT / "artifacts/c5-corrective-stride2-crop168-calibrated-v1"
CALIBRATOR = ROOT / "checkpoints/c5-crop-calibration-v4-cap32/calibrator.safetensors"
CALIBRATOR_RUNTIME = ROOT / "tools/c5_crop_calibrator.py"
EVALUATOR = ROOT / "tools/evaluate_c5_crop_openasl_confirmation.py"
SEED = 83


def stable_key(row: dict) -> str:
    return hashlib.sha256(
        f"{SEED}\0{row['source_group']}\0{row['clip_id']}".encode()
    ).hexdigest()


def select_one_per_source(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["source_group"]].append(row)
    if len(grouped) != 60 or Counter(map(len, grouped.values())) != Counter({2: 60}):
        raise ValueError("OpenASL pilot must contain two clips from each of 60 sources")
    return sorted(
        (min(source_rows, key=stable_key) for source_rows in grouped.values()),
        key=lambda row: row["source_group"],
    )


def load_examples() -> list[dict]:
    rows = [json.loads(line) for line in INDEX.read_text().splitlines()]
    receipts = {
        row["clip_id"]: row
        for row in (
            json.loads(line) for line in LANDMARK_RECEIPTS.read_text().splitlines()
        )
    }
    if len(rows) != 120 or len(receipts) != 120:
        raise ValueError("OpenASL pilot inputs are incomplete")
    selected = []
    for row in select_one_per_source(rows):
        receipt = receipts.get(row["clip_id"])
        clip = CLIP_DIR / f"{row['clip_id']}.mp4"
        landmarks = LANDMARK_DIR / f"{row['clip_id']}.npz"
        if (
            row.get("schema") != "umi-studio-openasl-stream-cache/1"
            or row.get("split") != "training"
            or row.get("temporal_stride") != 2
            or row.get("spatial_policy")
            != "full_source_frame_preserve_aspect_ratio"
            or receipt is None
            or receipt.get("schema") != "umi-studio-openasl-landmarks/1"
            or receipt.get("example_id") != row.get("example_id")
            or receipt.get("source_group") != row.get("source_group")
            or receipt.get("clip_sha256") != row.get("clip_sha256")
            or receipt.get("landmarks_sha256") != row.get("landmarks_sha256")
            or not clip.is_file()
            or not landmarks.is_file()
            or sha256(clip) != row.get("clip_sha256")
            or sha256(landmarks) != row.get("landmarks_sha256")
        ):
            raise ValueError(f"OpenASL confirmation input differs: {row.get('clip_id')}")
        selected.append(
            {
                "clip_id": row["clip_id"],
                "example_id": row["example_id"],
                "source_group": row["source_group"],
                "reference": row["reference"],
                "clip_sha256": row["clip_sha256"],
                "landmarks_sha256": row["landmarks_sha256"],
                "source_frames": row["source_frames"],
            }
        )
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.output.exists()
        or not args.output.resolve().is_relative_to((ROOT / "runs").resolve())
        or not all(
            path.is_file()
            for path in (
                INDEX,
                LANDMARK_RECEIPTS,
                PARENT_BUNDLE / "MODEL_IDENTITY.json",
                CANDIDATE_BUNDLE / "MODEL_IDENTITY.json",
                CANDIDATE_BUNDLE / "CROP168_CANDIDATE_EVIDENCE.json",
                CALIBRATOR,
                CALIBRATOR_RUNTIME,
                EVALUATOR,
            )
        )
    ):
        parser.error("confirmation output or required artifacts are invalid")
    candidate_identity = json.loads(
        (CANDIDATE_BUNDLE / "MODEL_IDENTITY.json").read_text()
    )
    evidence = json.loads(
        (CANDIDATE_BUNDLE / "CROP168_CANDIDATE_EVIDENCE.json").read_text()
    )
    if (
        candidate_identity.get("calibrator_sha256") != sha256(CALIBRATOR)
        or candidate_identity.get("visual_crop", {}).get("pixels") != 168
        or evidence.get("fresh_external_confirmation_required") is not True
        or evidence.get("miner_installed") is not False
    ):
        raise ValueError("crop candidate is not the frozen research candidate")
    plan = {
        "schema": "umi-studio-c5-crop-openasl-confirmation-plan/1",
        "seed": SEED,
        "selection_policy": "lowest seeded hash; one clip from each OpenASL source video",
        "split": "public_openasl_training_cross_domain_confirmation",
        "examples": load_examples(),
        "source_index_sha256": sha256(INDEX),
        "landmark_receipts_sha256": sha256(LANDMARK_RECEIPTS),
        "parent_bundle_identity_sha256": sha256(
            PARENT_BUNDLE / "MODEL_IDENTITY.json"
        ),
        "candidate_bundle_identity_sha256": sha256(
            CANDIDATE_BUNDLE / "MODEL_IDENTITY.json"
        ),
        "calibrator_sha256": sha256(CALIBRATOR),
        "calibrator_runtime_sha256": sha256(CALIBRATOR_RUNTIME),
        "evaluator_sha256": sha256(EVALUATOR),
        "variants": {
            "native224": {"visual_crop_size": 224, "calibrator": None},
            "calibrated168": {
                "visual_crop_size": 168,
                "calibrator": str(CALIBRATOR.relative_to(ROOT)),
            },
        },
        "quality_gate": {
            "maximum_mean_wer_delta": -0.005,
            "maximum_word_weighted_wer_delta": 0.0,
            "minimum_paired_improved_minus_regressed": 0,
        },
        "confirmation_policy": (
            "This plan was frozen before either variant was evaluated. Evaluate both "
            "variants exactly once on all 60 clips. The remaining clip from each source "
            "is not part of this confirmation."
        ),
        "candidate_training_overlap": (
            "The crop calibrator used only How2Sign training clips. OpenASL did not "
            "select the parent confirmed adapter or this calibrator."
        ),
        "miner_health_policy": "user_authorized_maintenance",
        "miner_installed": False,
        "freezer_sha256": sha256(Path(__file__)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(plan, output, indent=2, sort_keys=True)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    print(
        json.dumps(
            {
                "output": str(args.output),
                "examples": len(plan["examples"]),
                "sources": len({row["source_group"] for row in plan["examples"]}),
                "plan_sha256": sha256(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
