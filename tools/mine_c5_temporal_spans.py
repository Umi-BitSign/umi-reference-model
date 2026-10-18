"""Build unverified motion-span proposals and separate landmark feature branches."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "runs/c5-mixed-native112-data-v1/plan.json"
INPUT_SHA = "12135fba5be61387e1c806d32a3d746bb36f9ab248b64c0bd03de905b8142cf0"
TIPS = [4, 8, 12, 16, 20]
FACE = [1, 33, 61, 70, 105, 133, 145, 152, 159, 263, 291, 300, 334, 362, 374, 386]
EXCLUDED = {"P10", "P13", "P14", "P21", "P22", "P31", "P4", "P6", "P0", "P19"}
POLICY = {
    "schema": "umi-studio-landmark-motion-span-pilot/1",
    "scope": "all532 retained training clips; no development or diagnostic inputs",
    "timing": "all original canonical25fps landmark frames; no resampling",
    "hand_tracking": "tip-only nearest trajectory; unnamed tracks, not verified handedness",
    "ambiguous_more_than_two_hands": "mask both hand tracks; preserve the frame and counts",
    "missingness": "NaN plus explicit masks; no gap interpolation",
    "coordinates": "original detector coordinates; no palm/body normalization",
    "z_coordinates": "preserve original values; no shared world3D coordinate claim",
    "feature_sets": {
        "fingertips_only": ["fingertips"],
        "full_hands": ["full_hands"],
        "hands_and_arms": ["full_hands", "arms"],
        "hands_arms_face_body": ["full_hands", "pose", "face_sparse"],
    },
    "motion_coordinates": "xy scaled by image diagonal only; z excluded from motion score",
    "minimum_motion_diagonals_per_second": 0.005,
    "relative_motion_threshold": 0.2,
    "minimum_bout_frames": 3,
    "context_frames": 3,
    "span_label_status": "heuristic_unverified_motion_bout; not a sign boundary annotation",
    "caption": "weak whole-clip caption only; no text assigned to individual proposals",
    "output_budget_bytes": 256 * 1024**2,
    "physical_floor_bytes": 80 * 1024**3,
    "active_training_full_reserve_bytes": 4 * 1024**3,
    "headroom_bytes": 128 * 1024**2,
    "CPU_threads": 2,
    "GPU_or_API_or_model_execution": False,
    "optimizer_training_authorized": False,
}


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def identity(case):
    if case["kind"] == "word":
        record = case["word"]["record"]
        row = record["original"]["row"]
        require(record["ROI_lineage_training_allowed"] is True, "word is not training eligible")
        signer = row["participant"]
        landmark = record["landmarks"]
        clip_sha = record["clip"]["sha256"]
        video = record["clip"]
        frames = record["normalized_probe"]["decoded_frames"]
        source = "asl_stem_word"
    else:
        require(case["kind"] == "phrase", "unknown input kind")
        record = case["phrase"]
        row = record["original_row"]
        require(
            record["roi_role"] == "training" and record["roi_training_allowed"] is True,
            "phrase is not training eligible",
        )
        signer, source = row["signer"], row["source"]
        landmark = record["landmarks"]
        clip_sha = row["video_sha256"]
        video = {"path": row["video"], "sha256": clip_sha}
        frames = row["original_pair"]["header"]["source_frames"]
    require(
        row["role"] == "training" and row["training_allowed"] is True,
        "reject held-out input before landmark access",
    )
    require(
        not source.startswith("asl_stem") or signer not in EXCLUDED,
        "reject reserved or diagnostic signer before landmark access",
    )
    return {
        "kind": case["kind"],
        "source": source,
        "example_id": row["example_id"],
        "source_group": row["source_group"],
        "signer": signer,
        "weak_clip_caption": row["reference"],
        "landmarks": landmark,
        "canonical_video_sha256": clip_sha,
        "canonical_video": video,
        "frames": frames,
    }


def feature_branches(arrays):
    """Track hands using only fingertip xy; other branches never enter this decision."""
    frames = len(arrays["frame_valid"])
    hands = np.full((frames, 2, 21, 3), np.nan, dtype=np.float32)
    continuity = np.zeros((frames, 2), dtype=bool)
    previous = np.full((2, 5, 2), np.nan)
    for t in range(frames):
        count = int(arrays["hand_counts"][t])
        if not arrays["frame_valid"][t] or count not in (1, 2):
            previous[:] = np.nan
            continue
        incoming = arrays["hand_landmarks"][t, :count]
        tips = incoming[:, TIPS, :2]
        if not np.isfinite(tips).all():
            previous[:] = np.nan
            continue
        known = np.isfinite(previous).all(axis=(1, 2))
        choices = list(itertools.permutations(range(2), count))
        if known.any():

            def cost(slots, tips=tips, known=known):
                return sum(
                    float(np.mean((tips[i] - previous[slot]) ** 2))
                    for i, slot in enumerate(slots)
                    if known[slot]
                ) + sum(1.0 for slot in slots if not known[slot])

            slots = min(choices, key=cost)
        else:
            order = np.argsort(tips[:, :, 0].mean(axis=1), kind="stable")
            slots = tuple(int(np.flatnonzero(order == i)[0]) for i in range(count))
        for i, slot in enumerate(slots):
            hands[t, slot] = incoming[i]
            continuity[t, slot] = (
                known[slot] and np.sqrt(np.mean((tips[i] - previous[slot]) ** 2)) <= 0.35
            )
        previous[:] = np.nan
        for i, slot in enumerate(slots):
            previous[slot] = tips[i]
    pose = arrays["pose_landmarks"].astype(np.float32, copy=True)
    pose[~(arrays["frame_valid"] & arrays["pose_present"])] = np.nan
    face = np.full((frames, len(FACE), 3), np.nan, dtype=np.float32)
    single = arrays["frame_valid"] & (arrays["face_counts"] == 1)
    face[single] = arrays["face_landmarks"][single, 0][:, FACE]
    return {
        "fingertips": hands[:, :, TIPS].reshape(frames, 10, 3),
        "full_hands": hands.reshape(frames, 42, 3),
        "pose": pose,
        "arms": pose[:, [11, 12, 13, 14, 15, 16]],
        "face_sparse": face,
        "hand_continuity": continuity,
    }


def motion(values, fps, width, height, continuity=None):
    points = values[..., :2].astype(np.float64)
    points *= np.array([width, height]) / np.hypot(width, height)
    valid = np.isfinite(points).all(axis=2)
    both = valid[1:] & valid[:-1]
    if continuity is not None:
        both &= continuity[1:]
    distance = np.linalg.norm(np.diff(points, axis=0), axis=2) * fps
    result = np.full(len(values), np.nan)
    for t in range(1, len(values)):
        if both[t - 1].any():
            result[t] = np.median(distance[t - 1, both[t - 1]])
    return result


def proposals(curve, fps):
    finite = curve[np.isfinite(curve)]
    if not len(finite):
        return [], None
    threshold = max(
        POLICY["minimum_motion_diagonals_per_second"],
        POLICY["relative_motion_threshold"] * float(np.quantile(finite, 0.9)),
    )
    active = np.isfinite(curve) & (curve > threshold)
    edges = np.flatnonzero(np.diff(np.r_[False, active, False].astype(np.int8)))
    spans = []
    for start, end in zip(edges[::2], edges[1::2], strict=True):
        if end - start < POLICY["minimum_bout_frames"]:
            continue
        first = max(0, int(start) - 1 - POLICY["context_frames"])
        last = min(len(curve), int(end) + POLICY["context_frames"])
        spans.append(
            {
                "start_frame": first,
                "end_frame_exclusive": last,
                "start_seconds": first / fps,
                "end_seconds": last / fps,
                "unknown_motion_samples": int(np.count_nonzero(~np.isfinite(curve[first:last]))),
                "label": None,
                "verified_sign_boundary": False,
            }
        )
    return spans, threshold


def guard(output):
    used = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    require(used <= POLICY["output_budget_bytes"], "pilot output budget exceeded")
    required = sum(
        POLICY[key]
        for key in (
            "physical_floor_bytes",
            "active_training_full_reserve_bytes",
            "headroom_bytes",
            "output_budget_bytes",
        )
    )
    require(shutil.disk_usage(ROOT).free >= required, "preserve floor and active training reserve")


def run(output):
    require(sha(INPUT) == INPUT_SHA, "frozen input authority changed")
    plan = read(INPUT)
    records = [identity(case) for case in plan["records"]]
    require(
        len({(row["source"], row["example_id"]) for row in records}) == len(records),
        "duplicate training identity",
    )
    require(
        Counter(row["kind"] for row in records) == {"word": 214, "phrase": 318},
        "complete retained input inventory changed",
    )
    output = output.resolve()
    require(
        not output.exists() and output.is_relative_to(ROOT / "checkpoints"),
        "fresh private pilot output required",
    )
    output.mkdir(parents=True)
    guard(output)
    requested_threads = {
        name: os.environ.get(name)
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
    }
    require(set(requested_threads.values()) == {"2"}, "pilot requires its two-thread environment")
    write(
        output / "worker.json",
        {
            "pid": os.getpid(),
            "uid": os.getuid(),
            "argv": [sys.executable, *sys.argv],
            "requested_threads": requested_threads,
        },
    )
    source_sha = sha(Path(__file__))
    with (output / "source.py").open("xb") as stream:
        stream.write(Path(__file__).read_bytes())
    require(sha(output / "source.py") == source_sha, "source changed during archival")
    write(
        output / "plan.json",
        {
            "input_plan": {"path": str(INPUT.relative_to(ROOT)), "sha256": INPUT_SHA},
            "policy": POLICY,
            "source_sha256": source_sha,
            "numpy_version": np.__version__,
            "records": records,
        },
    )
    completed = []
    code = 1
    try:
        for index, row in enumerate(records):
            guard(output)
            path = ROOT / row["landmarks"]["path"]
            require(
                path.resolve() == path
                and path.is_relative_to(ROOT / "data")
                and sha(path) == row["landmarks"]["sha256"],
                "landmark bytes/path changed",
            )
            with np.load(path, allow_pickle=False) as archive:
                meta = json.loads(str(archive["metadata"].item()))
                arrays = {key: archive[key] for key in archive.files if key != "metadata"}
            count = row["frames"]
            shapes = {
                "face_landmarks": (count, 6, 478, 3),
                "hand_landmarks": (count, 6, 21, 3),
                "pose_landmarks": (count, 33, 3),
                "face_counts": (count,),
                "hand_counts": (count,),
                "pose_present": (count,),
                "frame_valid": (count,),
            }
            require(
                meta["schema"] == "umi-studio-mediapipe-landmarks/1"
                and meta["fps"] == 25
                and meta["frames"] == count
                and meta["source_sha256"] == row["canonical_video_sha256"]
                and set(arrays) == set(shapes)
                and all(arrays[key].shape == shape for key, shape in shapes.items()),
                "landmark source, full timeline or topology changed",
            )
            require(
                all(
                    np.all((arrays[key] >= 0) & (arrays[key] <= 6))
                    for key in ("face_counts", "hand_counts")
                ),
                "invalid detector counts",
            )
            features = feature_branches(arrays)
            curves, spans, thresholds = {}, {}, {}
            for name in ("fingertips", "full_hands", "arms", "pose", "face_sparse"):
                continuity = (
                    np.repeat(
                        features["hand_continuity"], 5 if name == "fingertips" else 21, axis=1
                    )
                    if name in ("fingertips", "full_hands")
                    else None
                )
                curve = motion(features[name], 25, meta["width"], meta["height"], continuity)
                curves[name] = [float(value) if np.isfinite(value) else None for value in curve]
                spans[name], thresholds[name] = proposals(curve, 25)
                features[name + "_valid"] = np.isfinite(features[name]).all(axis=2)
            features["source_frame_indices"] = np.arange(count, dtype=np.int64)
            features["canonical_seconds"] = np.arange(count, dtype=np.float64) / 25
            features["source_hand_counts"] = arrays["hand_counts"]
            features["source_face_counts"] = arrays["face_counts"]
            prefix = output / f"clip-{index:04d}"
            with prefix.with_suffix(".npz").open("xb") as stream:
                np.savez_compressed(stream, **features)
            record = {
                **row,
                "fps": 25,
                "frames": count,
                "motion": curves,
                "motion_thresholds": thresholds,
                "unverified_motion_bouts": spans,
                "full_clip_available": True,
                "clip_caption_is_weak_supervision": True,
                "annotation": {
                    "status": "unreviewed",
                    "verified_sign_intervals": [],
                    "reviewer": None,
                },
                "features": {
                    "path": str(prefix.with_suffix(".npz").relative_to(ROOT)),
                    "sha256": sha(prefix.with_suffix(".npz")),
                },
            }
            write(prefix.with_suffix(".json"), record)
            completed.append(
                {
                    "path": str(prefix.with_suffix(".json").relative_to(ROOT)),
                    "sha256": sha(prefix.with_suffix(".json")),
                }
            )
            if (index + 1) % 32 == 0:
                print(json.dumps({"clips_completed": index + 1}), flush=True)
        guard(output)
        require(sha(INPUT) == INPUT_SHA, "input authority changed during mining")
        require(
            sha(Path(__file__)) == source_sha and sha(output / "source.py") == source_sha,
            "source changed during mining",
        )
        write(
            output / "complete.json",
            {
                "complete": True,
                "records": completed,
                "clips": len(records),
                "ground_truth_sign_spans": 0,
                "training_authorized": False,
                "policy": POLICY,
            },
        )
        print(
            json.dumps({"clips": len(records), "complete": str(output / "complete.json")}),
            flush=True,
        )
        code = 0
    except BaseException as error:
        write(
            output / "failure.json",
            {"error": str(error), "completed_clips": len(completed), "automatic_retries": 0},
        )
        raise
    finally:
        write(output / "exit.json", {"pid": os.getpid(), "exit_code": code})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)


if __name__ == "__main__":
    main()
