"""Fit tiny affine maps from 168px DINO streams to native-224 streams."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from benchmark_fleurs_dev import sha256

SCHEMA = "umi-studio-c5-crop-calibrator/1"
RIDGE_RATIOS = (0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)
SEED = 127


def is_validation_group(group: str) -> bool:
    score = hashlib.sha256(f"{SEED}:{group}".encode()).digest()
    return int.from_bytes(score[:8], "big") % 5 == 0


def fit_affine(
    inputs: list[np.ndarray], targets: list[np.ndarray], ridge_ratio: float
) -> tuple[np.ndarray, np.ndarray]:
    x = np.concatenate(inputs).astype(np.float64)
    y = np.concatenate(targets).astype(np.float64)
    augmented = np.concatenate((x, np.ones((len(x), 1))), axis=1)
    identity = np.zeros((385, 384), dtype=np.float64)
    identity[:384] = np.eye(384)
    ridge = ridge_ratio * len(x)
    system = augmented.T @ augmented + ridge * np.eye(385)
    target = augmented.T @ y + ridge * identity
    solution = np.linalg.solve(system, target)
    return solution[:384].astype(np.float32), solution[384].astype(np.float32)


def capped_arrays(arrays: list[np.ndarray], cap: int | None) -> list[np.ndarray]:
    if cap is None:
        return arrays
    output = []
    for values in arrays:
        if len(values) <= cap:
            output.append(values)
        else:
            indices = np.linspace(0, len(values) - 1, cap, dtype=np.int64)
            output.append(values[indices])
    return output


def metrics(
    inputs: list[np.ndarray], targets: list[np.ndarray], weight=None, bias=None
) -> dict[str, float]:
    squared = 0.0
    cosine = 0.0
    count = 0
    for source, target in zip(inputs, targets, strict=True):
        prediction = source if weight is None else source @ weight + bias
        delta = prediction.astype(np.float64) - target.astype(np.float64)
        squared += float(np.square(delta).sum())
        numerator = np.sum(prediction * target, axis=1)
        denominator = np.linalg.norm(prediction, axis=1) * np.linalg.norm(target, axis=1)
        cosine += float(np.divide(numerator, np.maximum(denominator, 1e-12)).sum())
        count += source.shape[0]
    return {
        "frames": count,
        "mse": squared / (count * 384),
        "mean_cosine_similarity": cosine / count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-index", type=Path, required=True)
    parser.add_argument("--crop-cache-dir", type=Path, required=True)
    parser.add_argument("--native-cache-dir", type=Path, required=True)
    parser.add_argument("--extra-pair-index", type=Path)
    parser.add_argument("--extra-crop-cache-dir", type=Path)
    parser.add_argument("--per-stream-frame-cap", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        parser.error("calibrator output already exists")
    rows = [json.loads(line) for line in args.pair_index.read_text().splitlines()]
    if bool(args.extra_pair_index) != bool(args.extra_crop_cache_dir):
        parser.error("extra pair index and cache directory must be supplied together")
    if args.per_stream_frame_cap is not None and args.per_stream_frame_cap < 4:
        parser.error("per-stream frame cap must be at least four")
    if len(rows) != 169:
        raise ValueError("crop pair index must contain 169 examples")
    partitions = {
        "train": {"hand_x": [], "hand_y": [], "face_x": [], "face_y": []},
        "validation": {"hand_x": [], "hand_y": [], "face_x": [], "face_y": []},
    }
    groups = {"train": set(), "validation": set()}
    for row in rows:
        name = row["sentence_name"]
        split = "validation" if is_validation_group(row["source_group"]) else "train"
        groups[split].add(row["source_group"])
        crop_path = args.crop_cache_dir / f"{name}.npz"
        native_path = args.native_cache_dir / f"{name}.npz"
        if (
            sha256(crop_path) != row["crop168_cache_sha256"]
            or sha256(native_path) != row["native224_cache_sha256"]
        ):
            raise ValueError(f"calibration pair cache differs: {name}")
        with np.load(crop_path, allow_pickle=False) as crop, np.load(
            native_path, allow_pickle=False
        ) as native:
            partitions[split]["face_x"].append(crop["face_features"])
            partitions[split]["face_y"].append(native["face_features"])
            partitions[split]["hand_x"].extend(
                (crop["left_hand_features"], crop["right_hand_features"])
            )
            partitions[split]["hand_y"].extend(
                (native["left_hand_features"], native["right_hand_features"])
            )
    extra_hash = None
    extra_train_examples = 0
    if args.extra_pair_index is not None:
        extra_rows = [
            json.loads(line) for line in args.extra_pair_index.read_text().splitlines()
        ]
        if len(extra_rows) != len(rows):
            raise ValueError("extra crop pair index scope differs")
        clean_by_name = {row["sentence_name"]: row for row in rows}
        for row in extra_rows:
            name = row["sentence_name"]
            clean = clean_by_name.get(name)
            if (
                clean is None
                or row.get("source_group") != clean.get("source_group")
                or row.get("native224_cache_sha256")
                != clean.get("native224_cache_sha256")
                or row.get("augmentation_profile") != "mild_shift_v1"
            ):
                raise ValueError(f"extra crop pair provenance differs: {name}")
            if is_validation_group(row["source_group"]):
                continue
            crop_path = args.extra_crop_cache_dir / f"{name}.npz"
            native_path = args.native_cache_dir / f"{name}.npz"
            if sha256(crop_path) != row["crop168_cache_sha256"]:
                raise ValueError(f"extra crop pair cache differs: {name}")
            with np.load(crop_path, allow_pickle=False) as crop, np.load(
                native_path, allow_pickle=False
            ) as native:
                partitions["train"]["face_x"].append(crop["face_features"])
                partitions["train"]["face_y"].append(native["face_features"])
                partitions["train"]["hand_x"].extend(
                    (crop["left_hand_features"], crop["right_hand_features"])
                )
                partitions["train"]["hand_y"].extend(
                    (native["left_hand_features"], native["right_hand_features"])
                )
            extra_train_examples += 1
        extra_hash = sha256(args.extra_pair_index)
    if not groups["train"] or not groups["validation"] or groups["train"] & groups["validation"]:
        raise ValueError("calibration source-group partition differs")

    candidates = []
    train_hand_x = capped_arrays(
        partitions["train"]["hand_x"], args.per_stream_frame_cap
    )
    train_hand_y = capped_arrays(
        partitions["train"]["hand_y"], args.per_stream_frame_cap
    )
    train_face_x = capped_arrays(
        partitions["train"]["face_x"], args.per_stream_frame_cap
    )
    train_face_y = capped_arrays(
        partitions["train"]["face_y"], args.per_stream_frame_cap
    )
    for ratio in RIDGE_RATIOS:
        hand_w, hand_b = fit_affine(
            train_hand_x, train_hand_y, ratio
        )
        face_w, face_b = fit_affine(
            train_face_x, train_face_y, ratio
        )
        hand = metrics(
            partitions["validation"]["hand_x"],
            partitions["validation"]["hand_y"],
            hand_w,
            hand_b,
        )
        face = metrics(
            partitions["validation"]["face_x"],
            partitions["validation"]["face_y"],
            face_w,
            face_b,
        )
        candidates.append((hand["mse"] + face["mse"], ratio, hand_w, hand_b, face_w, face_b, hand, face))
    _, ratio, hand_w, hand_b, face_w, face_b, hand_validation, face_validation = min(
        candidates, key=lambda item: (item[0], item[1])
    )
    from safetensors.numpy import save_file

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "hand_weight": hand_w,
            "hand_bias": hand_b,
            "face_weight": face_w,
            "face_bias": face_b,
        },
        str(args.output),
        metadata={
            "schema": SCHEMA,
            "pair_index_sha256": sha256(args.pair_index),
            "ridge_ratio": str(ratio),
            "extra_pair_index_sha256": extra_hash or "none",
            "per_stream_frame_cap": str(args.per_stream_frame_cap or 0),
            "source_crop_size": "168",
            "target_crop_size": "224",
        },
    )
    report = {
        "schema": "umi-studio-c5-crop-calibrator-report/1",
        "pair_index_sha256": sha256(args.pair_index),
        "calibrator_sha256": sha256(args.output),
        "ridge_ratio": ratio,
        "extra_pair_index_sha256": extra_hash,
        "extra_train_examples": extra_train_examples,
        "per_stream_frame_cap": args.per_stream_frame_cap,
        "fit_hand_frames": sum(len(values) for values in train_hand_x),
        "fit_face_frames": sum(len(values) for values in train_face_x),
        "train_groups": len(groups["train"]),
        "validation_groups": len(groups["validation"]),
        "uncalibrated_validation": {
            "hand": metrics(partitions["validation"]["hand_x"], partitions["validation"]["hand_y"]),
            "face": metrics(partitions["validation"]["face_x"], partitions["validation"]["face_y"]),
        },
        "calibrated_validation": {
            "hand": hand_validation,
            "face": face_validation,
        },
        "candidate_validation_mse": {
            str(candidate[1]): {
                "hand": candidate[6]["mse"],
                "face": candidate[7]["mse"],
            }
            for candidate in candidates
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
