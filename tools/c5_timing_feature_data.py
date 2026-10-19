"""Authenticate complete532 training features and load them without Torch or a model."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from c5_timing_visibility import POLICY as VISIBILITY
from c5_timing_visibility import interval_mask, visibility
from extract_mediapipe_landmarks import unpack_landmarks

ROOT = Path(__file__).resolve().parents[1]
WIDTHS = {
    "face_features": 384,
    "left_hand_features": 384,
    "right_hand_features": 384,
    "pose_features": 14,
}
RESERVED = {"P0", "P19", "P10", "P13", "P14", "P21", "P22", "P31", "P4", "P6"}


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(1024**2):
            digest.update(block)
    return digest.hexdigest()


def artifact(path):
    return {"path": str(path.relative_to(ROOT)), "sha256": sha(path)}


def owned(path, directory):
    require(
        path.absolute() == path.resolve()
        and path.is_relative_to(ROOT / directory)
        and path != ROOT / directory,
        "timing data paths must be canonical owned artifacts",
    )


def bound(descriptor):
    path = ROOT / descriptor["path"]
    require(
        path.is_relative_to(ROOT) and path.resolve() == path and artifact(path) == descriptor,
        "timing data artifact bytes or canonical path changed",
    )
    return path


def array_metadata(value):
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
    }


def training_row(case):
    if case["kind"] == "word":
        record = case["word"]["record"]
        row = record["original"]["row"]
        require(
            row["role"] == "training"
            and row["training_allowed"] is True
            and record["ROI_lineage_training_allowed"] is True
            and row["participant"] not in RESERVED,
            "timing data cannot train on reserved word roles",
        )
        return {
            "source": "asl_stem_word",
            "example_id": row["example_id"],
            "source_group": row["source_group"],
            "signer": row["participant"],
            "reference": row["reference"],
            "role": "training",
            "training_allowed": True,
        }
    require(case["kind"] == "phrase", "unknown timing training input kind")
    record, row = case["phrase"], case["phrase"]["original_row"]
    require(
        record["roi_role"] == row["role"] == "training"
        and record["roi_training_allowed"] is row["training_allowed"] is True
        and (not row["source"].startswith("asl_stem") or row["signer"] not in RESERVED),
        "timing data cannot train on reserved phrase roles",
    )
    pair = row["original_pair"]
    raw, header, alignment, teacher = (pair[k] for k in ("raw", "header", "alignment", "teacher"))
    require(
        pair["feature_pretraining_allowed"] is True
        and all(
            v["role"] == "training" and v["source"] == row["source"]
            for v in (raw, header, alignment, teacher)
        )
        and all(v["training_allowed"] is True for v in (raw, alignment, teacher))
        and raw["example_id"]
        == header["example_id"]
        == alignment["example_id"]
        == teacher["id"]
        == row["example_id"]
        and raw["source_group"] == alignment["source_group"] == row["source_group"]
        and teacher["signer"] == row["signer"]
        and teacher["reference"] == row["reference"]
        and teacher["stored_temporal_stride"] == pair["native_teacher_stride"] == 2,
        "timing phrase inner identities or training roles changed",
    )
    return row


def source_binding(case):
    training_row(case)  # Reject roles before looking up paths.
    if case["kind"] == "word":
        record = case["word"]["record"]
        return record["clip"], record["landmarks"], case["word"]["verified_input"]["header"]
    record = case["phrase"]
    pair = record["original_row"]["original_pair"]
    header = pair["header"]
    canonical = {
        k: header[k]
        for k in (
            "source_fps",
            "source_frames",
            "source_shape",
            "sampled_frames",
            "frame_indices_sha256",
        )
    }
    return (
        {"path": pair["raw"]["video"], "sha256": pair["raw"]["video_sha256"]},
        record["landmarks"],
        canonical,
    )


def check_arrays(arrays, inputs):
    header = inputs["header"]
    source_frames, frames = header["source_frames"], header["sampled_frames"]
    require(
        type(source_frames) is int
        and source_frames >= 1
        and type(frames) is int
        and frames >= 1
        and header["source_fps"] == 25
        and inputs["native_side"] == 224
        and inputs["streamed_pixels_equal_native_reference"] is True,
        "timing cache lost original25fps native224 pixel admission",
    )
    expected = {
        **{k: ((frames, width), np.float32) for k, width in WIDTHS.items()},
        "frame_indices": ((frames,), np.int64),
        "valid_intervals": ((frames - 1,), np.bool_),
        "source_hand_tip_visible": ((source_frames,), np.bool_),
    }
    require(set(arrays) == set(expected), "timing cache contains missing or extra feature streams")
    for name, (shape, dtype) in expected.items():
        value = arrays[name]
        require(
            isinstance(value, np.ndarray)
            and value.shape == shape
            and value.dtype == dtype
            and np.isfinite(value).all(),
            "timing cache changed complete finite FP32 shapes",
        )
    indices = np.arange(0, source_frames, 2, dtype=np.int64)
    require(
        np.array_equal(indices, arrays["frame_indices"])
        and array_metadata(indices)["sha256"] == header["frame_indices_sha256"]
        and np.array_equal(
            interval_mask(arrays["source_hand_tip_visible"], indices), arrays["valid_intervals"]
        ),
        "timing cache changed complete sampling or bridged a full-rate visibility gap",
    )
    for name in ("pose_features", "frame_indices", "valid_intervals", "source_hand_tip_visible"):
        require(
            array_metadata(arrays[name]) == inputs["arrays"][name],
            "timing cache changed admitted input bits",
        )
    require(
        inputs["arrays"]["pixels"]["shape"] == [frames, 3, 224, 224, 3]
        and inputs["arrays"]["pixels"]["dtype"] == "uint8"
        and isinstance(inputs["arrays"]["pixels"]["sha256"], str)
        and len(inputs["arrays"]["pixels"]["sha256"]) == 64
        and set(inputs["arrays"]["pixels"]["sha256"]) <= set("0123456789abcdef"),
        "timing cache changed native pixel surface",
    )
    return {name: array_metadata(v) for name, v in arrays.items()}


def terminal(directory):
    worker, exited = read(directory / "worker.json"), read(directory / "exit.json")
    require(
        worker["uid"] == os.getuid()
        and worker["pid"] == exited["pid"]
        and type(exited["exit_code"]) is int
        and exited["exit_code"] == 0,
        "timing data requires a successful same-worker terminal extraction",
    )
    return {name: artifact(directory / name) for name in ("worker.json", "exit.json")}


def native_reference_check(arrays, reference, reported, indices, atol, rtol):
    require(
        set(reference) == {"face_features", "left_hand_features", "right_hand_features"}
        and reported["sample_indices"] == indices.tolist(),
        "native timing reference coverage changed",
    )
    for name, native in reference.items():
        actual = arrays[name][indices]
        require(
            native.dtype == np.float32
            and native.shape == actual.shape
            and np.isfinite(native).all()
            and np.allclose(actual, native, atol=atol, rtol=rtol),
            "native timing feature reference failed",
        )
        maximum = float(np.max(np.abs(actual.astype(np.float64) - native)))
        require(
            maximum == reported["streams"][name]["maximum_absolute_difference"],
            "native timing feature report changed",
        )


def audit(plan_path, result_path, poll_path):
    for path, directory in ((plan_path, "runs"), (result_path, "checkpoints"), (poll_path, "runs")):
        owned(path, directory)
    poll = read(poll_path)
    require(
        poll.get("tool") == "write_stdin"
        and type(poll.get("requested_session_id")) is int
        and type(poll["tool_result"].get("exit_code")) is int
        and poll["tool_result"]["exit_code"] == 0
        and poll["tool_result"].get("session_id") is None,
        "timing audit requires the actual completed feature-session tool poll",
    )
    authority = terminal(result_path.parent)
    plan, result = read(plan_path), read(result_path)
    policy = plan["policy"]
    require(
        plan["schema"] == "umi-studio-complete-native224-timing-feature-plan/1"
        and result["schema"] == "umi-studio-complete-native224-timing-features/1"
        and result["complete"] is True
        and result["plan"] == artifact(plan_path)
        and result["worker"] == authority["worker.json"]
        and result_path.name == "result.json"
        and len(plan["records"]) == len(result["receipts"]) == result["clips"] == 532
        and Counter(v["kind"] for v in plan["records"])
        == result["counts"]
        == {"word": 214, "phrase": 318}
        and policy["feature_widths"] == WIDTHS
        and policy["visibility"] == VISIBILITY
        and policy["native_side"] == 224
        and policy["CPU_threads"] == 2
        and policy["GPU_use"] is False
        and policy["full_native_feature_reference_indices"] == [0, 1, 214, 215]
        and policy["native_feature_absolute_tolerance"]
        == policy["native_feature_relative_tolerance"]
        == 1e-4,
        "timing audit lost the complete fixed532 native224 feature contract",
    )
    worker = read(result_path.with_name("worker.json"))
    require(
        poll.get("actual_owned_pid") == worker["pid"],
        "timing terminal tool poll lost its actual owned feature worker",
    )
    argv = worker["argv"]
    require(
        Path(argv[1]).absolute() == ROOT / "tools/prepare_c5_timing_features.py"
        and Path(argv[argv.index("--plan") + 1]).absolute() == plan_path
        and Path(argv[argv.index("--output") + 1]).absolute() == result_path.parent,
        "timing feature worker changed its actual extraction program or paths",
    )
    manifest = read(plan_path.with_name("source-snapshot") / "manifest.json")
    require(manifest["plan_sha256"] == sha(plan_path), "timing source snapshot lost its plan")
    for name, digest in plan["source_sha256"].items():
        require(
            sha(ROOT / name) == sha(plan_path.with_name("source-snapshot") / name) == digest
            and manifest["source_sha256"][name] == digest,
            "timing extraction source changed",
        )
    mixed = read(bound(plan["mixed"]))
    public = read(bound(plan["public_model"]))
    require(
        plan["records"] == mixed["records"]
        and plan["model_plan"] == public["model_plan"]
        and public["trained_or_partial_visual_checkpoint_installed_for_inference"] is False
        and result["original_visual_state_sha256"] == plan["model_plan"]["pretrained_state_sha256"]
        and result["original_visual_values_unchanged"] is True,
        "timing data lost original public values or unchanged training inventory",
    )
    for name, digest in plan["model_plan"]["assets"].items():
        require(sha(ROOT / name) == digest, "original C5 model asset changed")
    for key, expected in (
        ("CPU_threads", 2),
        ("GPU_use", False),
        ("visual_encoder_executed", True),
        ("decoder_executed", False),
        ("optimizer_updates", 0),
        ("new_Terra_requests", 0),
        ("training_or_timing_risk_qualified", False),
        ("promotion_authorized", False),
    ):
        require(
            type(result[key]) is type(expected) and result[key] == expected,
            "timing data reclassified extraction as training",
        )
    expected_files = {
        result_path,
        result_path.with_name("worker.json"),
        result_path.with_name("exit.json"),
    }
    records, frames, visible, adjustable = [], 0, 0, 0
    identities = set()
    for index, (case, descriptor) in enumerate(
        zip(plan["records"], result["receipts"], strict=True)
    ):
        row = training_row(case)
        identity = row["source"], row["example_id"]
        require(identity not in identities, "duplicate timing training identity")
        identities.add(identity)
        path = bound(descriptor)
        require(
            path == result_path.with_name(f"clip-{index:04d}.json"), "timing receipt order changed"
        )
        saved = read(path)
        require(
            saved["case"] == case
            and saved["row"] == row
            and saved["decoder_executed"] is False
            and saved["optimizer_updates"] == 0,
            "timing clip changed its source role or operation",
        )
        clip, landmark, header = source_binding(case)
        bound(clip)
        landmark_path = bound(landmark)
        require(saved["input"]["header"] == header, "timing clip changed its original header")
        cache = bound(saved["features"])
        require(
            cache == result_path.with_name(f"features-{index:04d}.npz"),
            "timing feature cache path changed",
        )
        with np.load(cache, allow_pickle=False) as archive:
            arrays = dict(archive)
        metadata = check_arrays(arrays, saved["input"])
        original, details = unpack_landmarks(landmark_path)
        require(
            details["source_sha256"] == clip["sha256"]
            and details["fps"] == 25
            and details["frames"] == header["source_frames"]
            and [details["height"], details["width"]] == header["source_shape"],
            "timing landmark source changed",
        )
        observed, valid = visibility(original, header["source_frames"], arrays["frame_indices"])
        require(
            np.array_equal(observed, arrays["source_hand_tip_visible"])
            and np.array_equal(valid, arrays["valid_intervals"]),
            "timing mask differs from full-rate source landmarks",
        )
        if case["kind"] == "word":
            require(
                metadata["pose_features"]
                == case["word"]["verified_input"]["arrays"]["pose_features"],
                "word timing pose bits changed",
            )
        else:
            teacher = case["phrase"]["original_row"]["original_pair"]["teacher"]
            legacy = bound({"path": teacher["cache"], "sha256": teacher["cache_sha256"]})
            with np.load(legacy, allow_pickle=False) as old:
                require(
                    old["pose_features"].tobytes() == arrays["pose_features"].tobytes(),
                    "phrase timing pose bits changed",
                )
        count = len(arrays["frame_indices"])
        lengths = np.diff(np.r_[0, np.flatnonzero(~valid) + 1, count])
        interior = int(np.maximum(lengths - 2, 0).sum())
        require(
            saved["visibility_component_lengths"] == lengths.tolist()
            and saved["potentially_adjustable_interior_samples"] == interior,
            "timing visibility support changed",
        )
        full = index in policy["full_native_feature_reference_indices"]
        indices = (
            np.arange(count, dtype=np.int64)
            if full
            else np.unique(np.array([0, count - 1], dtype=np.int64))
        )
        reported = saved["native_feature_comparison"]
        require(
            reported["sample_indices"] == indices.tolist()
            and set(reported["streams"]) == set(WIDTHS) - {"pose_features"}
            and all(
                type(v["maximum_absolute_difference"]) in (int, float)
                and math.isfinite(v["maximum_absolute_difference"])
                and v["maximum_absolute_difference"] >= 0
                for v in reported["streams"].values()
            ),
            "native endpoint report changed",
        )
        if full:
            reference_path = bound(saved["complete_native_feature_reference"])
            require(
                reference_path == result_path.with_name(f"native-reference-{index:04d}.npz"),
                "native timing reference path changed",
            )
            with np.load(reference_path, allow_pickle=False) as reference:
                native_reference_check(arrays, dict(reference), reported, indices, 1e-4, 1e-4)
            expected_files.add(reference_path)
        else:
            require(
                saved["complete_native_feature_reference"] is None,
                "extra complete native reference",
            )
        expected_files.update((path, cache))
        records.append(
            {
                "index": index,
                "kind": case["kind"],
                "row": {
                    k: row[k]
                    for k in (
                        "source",
                        "example_id",
                        "source_group",
                        "signer",
                        "reference",
                        "role",
                        "training_allowed",
                    )
                },
                "receipt": descriptor,
                "features": saved["features"],
                "header": header,
                "arrays": metadata,
            }
        )
        frames += count
        visible += int(valid.sum())
        adjustable += interior
    actual = set(result_path.parent.rglob("*"))
    require(
        all(not p.is_symlink() and (not p.is_file() or p.stat().st_nlink == 1) for p in actual)
        and {p for p in actual if p.is_file()} == expected_files,
        "timing extraction inventory has aliases, partials or extras",
    )
    require(
        frames == result["sampled_frames"] == 38876
        and visible == result["timing_visible_intervals"]
        and adjustable == result["potentially_adjustable_interior_samples"]
        and result["native_endpoint_feature_comparisons"] == 532
        and result["complete_native_feature_comparisons"] == 4,
        "timing extraction coverage or support totals changed",
    )
    return {
        "schema": "umi-studio-complete-timing-feature-data-audit/1",
        "complete": True,
        "plan": artifact(plan_path),
        "result": artifact(result_path),
        "terminal_poll": artifact(poll_path),
        "authority": authority,
        "audit_program": artifact(Path(__file__)),
        "records": records,
        "clips": 532,
        "sampled_frames": frames,
        "timing_visible_intervals": visible,
        "potentially_adjustable_interior_samples": adjustable,
        "model_plan": plan["model_plan"],
        "source_sha256": plan["source_sha256"],
        "native_endpoint_reports_authenticated": 532,
        "complete_native_reference_arrays_recompared": 4,
        "native_pixels_or_neural_features_recomputed_by_audit": False,
        "visibility_and_pose_bits_revalidated": True,
        "model_or_scorer_executed": False,
        "training_risk_qualified": False,
        "quality_gain_established": False,
        "promotion_authorized": False,
    }


def checked(path):
    owned(path, "runs")
    saved = read(path)
    require(saved["audit_program"] == artifact(Path(__file__)), "timing data auditor changed")
    require(
        saved == audit(bound(saved["plan"]), bound(saved["result"]), bound(saved["terminal_poll"])),
        "complete timing feature audit changed",
    )
    return saved


class TimingFeatureDataset:
    def __init__(self, audit_path):
        self._value = checked(audit_path)
        self._path, self._authority = audit_path, artifact(audit_path)

    def __len__(self):
        return len(self._value["records"])

    @property
    def model_plan(self):
        return copy.deepcopy(self._value["model_plan"])

    def load(self, index, *, purpose="training"):
        require(
            purpose == "training" and type(index) is int and 0 <= index < len(self),
            "timing dataset permits only its admitted training indices",
        )
        require(
            artifact(self._path) == self._authority, "timing dataset authority changed during use"
        )
        record = self._value["records"][index]
        require(
            record["index"] == index
            and record["row"]["role"] == "training"
            and record["row"]["training_allowed"] is True,
            "timing dataset changed training roles",
        )
        saved = read(bound(record["receipt"]))
        with np.load(bound(record["features"]), allow_pickle=False) as archive:
            arrays = dict(archive)
        require(
            record["header"] == saved["input"]["header"]
            and check_arrays(arrays, saved["input"]) == record["arrays"],
            "timing dataset feature binding changed",
        )
        return {
            "row": copy.deepcopy(record["row"]),
            "streams": {name: arrays[name] for name in WIDTHS},
            "valid_intervals": arrays["valid_intervals"],
            "frame_indices": arrays["frame_indices"],
            "header": copy.deepcopy(record["header"]),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "result", "terminal-poll", "output"):
        parser.add_argument("--" + name, type=lambda p: Path(p).absolute(), required=True)
    args = parser.parse_args()
    owned(args.output, "runs")
    require(not args.output.exists(), "timing data audit needs a fresh output")
    require(
        all(
            os.environ.get(name) == "1"
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "PYTHONDONTWRITEBYTECODE",
            )
        ),
        "data-only timing audit requires one CPU and disabled bytecode",
    )
    value = audit(args.plan, args.result, args.terminal_poll)
    require("torch" not in sys.modules, "standalone timing data audit must not import Torch")
    data = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    require(
        len(data) <= 32 * 1024**2
        and shutil.disk_usage(ROOT).free >= 84 * 1024**3 + 128 * 1024**2 + len(data),
        "timing data audit storage reserve failed",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as f:
        f.write(data)
    print(json.dumps({"audit": artifact(args.output), "clips": value["clips"]}), flush=True)


if __name__ == "__main__":
    main()
