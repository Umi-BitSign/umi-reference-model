"""Audit reserved sidecars and read complete native224/native112 confirmation inputs.

No training role, neural model, scorer, download or translation is executed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np

import prepare_c5_broad_landmarks as preparation
from audit_terra_word_landmarks import terminal
from extract_mediapipe_landmarks import unpack_landmarks
from fetch_c5_broad_how2sign import ROOT, artifact, read, require, sha256, write
from prepare_roi_appearance import DEMO, SampledRgbReader, native_helpers
from prepare_roi_short_corpus import NativeSampledReader
from training_media_artifact import media_artifact

PLAN = ROOT / "runs/c5-broad-confirmation-landmarks-v1/plan.json"
DATA = ROOT / "data/c5-broad-landmarks-v1"
RECEIPTS = PLAN.parent / "landmark-receipts"
EXTRAS = ("landmarks", "landmark_receipt", "landmark_plan", "broad_audit")


def owned(path, folder):
    require(path.absolute() == path.resolve() and path.is_relative_to(ROOT / folder)
            and path != ROOT / folder, "confirmation paths must be canonical owned files/directories")


def sources(parent):
    result = dict(parent["source_sha256"])
    for name in ("tools/c5_broad_native_input.py", "tools/prepare_roi_appearance.py",
                 "tools/prepare_roi_short_corpus.py", "tools/training_media_artifact.py",
                 *(str((DEMO / name).relative_to(ROOT)) for name in
                   ("body_features.py", "crop_hands.py", "crop_face.py", "video_reader.py"))):
        result[name] = sha256(ROOT / name)
    return result


def require_role(record, purpose):
    row = record.get("original", {}).get("row", {})
    example = record.get("example", {})
    require(purpose == "confirmation" and row.get("role") == "confirmation"
            and row.get("training_allowed") is False and example.get("training_allowed") is False
            and record.get("ROI_lineage_training_allowed") is False
            and isinstance(example.get("clip_id"), str) and bool(example["clip_id"])
            and row.get("example_id") == example["clip_id"]
            and example.get("family") in ("how2sign", "openasl")
            and example.get("source_group", "").startswith(example["family"] + ":"),
            "confirmation inputs cannot become training/development or change source roles")


def audit(plan_path, data, receipts, jobs):
    for path, folder in ((plan_path, "runs"), (data, "data"), (receipts, "runs")):
        owned(path, folder)
    require(len(jobs) == len(set(jobs)) == 3, "complete confirmation sidecars require three serial batches")
    parent = preparation.checked(plan_path)
    require(parent["training_allowed"] is False and len(parent["records"]) == 48,
            "confirmation sidecar plan changed its protected inventory")
    authority, produced = [], set()
    by_name = {r["example"]["clip_id"] + ".npz": r for r in parent["records"]}
    for job in jobs:
        owned(job, "runs")
        evidence = terminal(job)
        worker = read(job / "worker.json")
        require(worker["uid"] == os.getuid(), "confirmation extraction worker ownership changed")
        argv = worker["argv"]
        require(Path(argv[1]).absolute() == ROOT / "tools/prepare_c5_broad_landmarks.py",
                "confirmation sidecars came from an unrelated extraction worker")
        for name, expected in (("--plan", plan_path), ("--data", data), ("--receipts", receipts), ("--job", job)):
            require(name in argv and Path(argv[argv.index(name) + 1]).absolute() == expected,
                    "confirmation extraction worker changed paths")
        require("--limit" not in argv or argv[argv.index("--limit") + 1] == "16",
                "confirmation extraction batch changed its bound")
        release_path = job / "training-release.json"
        release = read(release_path)
        for descriptor in release.values():
            require(artifact(ROOT / descriptor["path"]) == descriptor, "training release bytes changed")
        require(preparation.admit_release(ROOT / release["terminal_poll"]["path"],
                                          ROOT / release["terminal_audit"]["path"]) == release,
                "sidecar worker lost actual complete-training release")
        commands = sorted(job.glob("container-*.json"))
        require([p.name for p in commands] == [f"container-{i:02d}.json" for i in range(16)],
                "confirmation extraction batch lacks exactly sixteen detector calls")
        for command_path in commands:
            command_record = read(command_path)
            command = command_record["command"]
            name = Path(command[command.index("--output") + 1]).name
            require(name in by_name and name not in produced and command_record["automatic_retries"] == 0,
                    "confirmation detector command repeated or changed a source")
            expected = preparation.container_command(docker_socket=preparation.DOCKER_SOCKET,
                clip=ROOT / by_name[name]["clip"]["path"], output_name=name, landmark_dir=data / "confirmation")
            expected[expected.index(preparation.IMAGE)] = parent["docker_image_id"]
            require(command == expected and command_path.with_suffix(".log").is_file(),
                    "confirmation detector command or raw log changed")
            produced.add(name)
        authority.append({**evidence, "release": artifact(release_path)})
    require(produced == set(by_name), "confirmation workers did not extract all48 frozen identities")
    expected_data, expected_receipts, bound = set(), set(), []
    frames = total = 0
    families, groups, signers = Counter(), set(), Counter()
    for record in parent["records"]:
        require_role(record, "confirmation")
        identity = record["example"]["clip_id"]
        sidecar = data / "confirmation" / (identity + ".npz")
        receipt = receipts / "confirmation" / (identity + ".json")
        saved = read(receipt)
        require(sidecar.stat().st_size <= parent["policy"]["maximum_landmark_clip_bytes"]
                and saved.get("schema") == "umi-studio-broad-confirmation-landmark-receipt/1"
                and saved.get("plan_sha256") == sha256(plan_path)
                and saved.get("record") == record
                and saved.get("landmarks") == media_artifact(sidecar)
                and saved.get("training_allowed") is False
                and saved.get("confirmation_consumed") is False
                and saved.get("model_training_ready") is False
                and saved.get("service_checks_performed") is False,
                "confirmation sidecar receipt changed source, role or bytes")
        metadata = preparation.original.checked_landmarks(sidecar, record)
        require(saved.get("metadata") == metadata, "confirmation detector metadata changed")
        bound.append({**record, "landmarks": media_artifact(sidecar),
                      "landmark_receipt": artifact(receipt), "landmark_plan": artifact(plan_path)})
        expected_data.add(sidecar)
        expected_receipts.add(receipt)
        frames += metadata["frames"]
        total += sidecar.stat().st_size
        example = record["example"]
        families[example["family"]] += 1
        groups.add(example["source_group"])
        if example["family"] == "how2sign":
            signers[example["signer"]] += 1
    require(all(not p.is_symlink() for base in (data, receipts) for p in base.rglob("*"))
            and {p for p in data.rglob("*") if p.is_file()} == expected_data
            and {p for p in receipts.rglob("*") if p.is_file()} == expected_receipts,
            "confirmation sidecar inventory contains aliases, extras, partials or missing clips")
    require(families == {"how2sign": 24, "openasl": 24} and len(groups) == 36
            and signers == {1: 6, 2: 6, 3: 6, 5: 6} and frames == 8126
            and total <= parent["policy"]["maximum_total_landmark_bytes"],
            "complete confirmation family/signer/source/timeline coverage changed")
    pinned = sources(parent)
    require(all(sha256(ROOT / name) == digest for name, digest in parent["source_sha256"].items()),
            "frozen landmark source changed during audit")
    return {"schema": "umi-studio-broad-confirmation-complete-landmark-audit/1",
            "complete": True, "plan": artifact(plan_path), "data_dir": str(data.relative_to(ROOT)),
            "receipts_dir": str(receipts.relative_to(ROOT)), "jobs": [str(p.relative_to(ROOT)) for p in jobs],
            "worker_authority": authority, "records": bound, "frames": frames, "landmark_bytes": total,
            "families": dict(families), "source_groups": len(groups), "source_sha256": pinned,
            "training_allowed": False, "candidate_bound_by_audit": False,
            "confirmation_consumed_by_audit": False, "model_or_scorer_execution": False}


def checked_audit(path):
    owned(path, "runs")
    saved = read(path)
    require(saved == audit(ROOT / saved["plan"]["path"], ROOT / saved["data_dir"],
                           ROOT / saved["receipts_dir"], [ROOT / p for p in saved["jobs"]]),
            "complete confirmation landmark audit changed")
    return saved


def bind_records(path):
    complete = checked_audit(path)
    return [{**record, "broad_audit": artifact(path)} for record in complete["records"]]


def pose_processor():
    specification = importlib.util.spec_from_file_location("broad_native_body_features", DEMO / "body_features.py")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module.PoseProcessor()


def pack_rgb(values, frames, side):
    require(len(values) == frames and all(isinstance(v, np.ndarray) and v.dtype == np.uint8
            and v.shape == (side, side, 3) for v in values),
            "confirmation native crops changed complete count, side or dtype")
    return np.ascontiguousarray(np.stack(values)[..., ::-1])


def read_input(record, *, purpose="confirmation", side=112):
    require_role(record, purpose)  # Reject role changes before any file read.
    require(type(side) is int and side in (112, 224), "confirmation permits only native112/native224 input")
    source = {k: v for k, v in record.items() if k not in EXTRAS}
    audit_path = ROOT / record["broad_audit"]["path"]
    require(artifact(audit_path) == record["broad_audit"], "confirmation audit bytes changed")
    complete = read(audit_path)
    expected = {k: v for k, v in record.items() if k != "broad_audit"}
    require(complete["complete"] is True and complete["training_allowed"] is False
            and len(complete["records"]) == 48 and bool(complete["source_sha256"])
            and expected in complete["records"] and complete["plan"] == record["landmark_plan"]
            and all(sha256(ROOT / name) == digest for name, digest in complete["source_sha256"].items()),
            "confirmation input lost its audited complete-inventory membership")
    video = ROOT / record["clip"]["path"]
    landmark = ROOT / record["landmarks"]["path"]
    receipt = ROOT / record["landmark_receipt"]["path"]
    saved = read(receipt)
    require(artifact(receipt) == record["landmark_receipt"]
            and media_artifact(video) == record["clip"] and media_artifact(landmark) == record["landmarks"]
            and saved["schema"] == "umi-studio-broad-confirmation-landmark-receipt/1"
            and saved["plan_sha256"] == record["landmark_plan"]["sha256"] and saved["record"] == source
            and saved["landmarks"] == record["landmarks"] and saved["training_allowed"] is False,
            "confirmation clip/landmark receipt identity or role changed")
    metadata = preparation.original.checked_landmarks(landmark, source)
    require(saved["metadata"] == metadata, "confirmation actual landmark metadata changed")
    landmarks, _ = unpack_landmarks(landmark)
    indices = np.arange(0, metadata["frames"], 2, dtype=np.int64)
    sampled = {i: landmarks[int(index)] for i, index in enumerate(indices)}
    header = {"source_fps": 25, "source_frames": metadata["frames"],
              "source_shape": [metadata["height"], metadata["width"]], "sampled_frames": len(indices),
              "frame_indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest()}
    require(len(indices) * 3 * side * side * 3 <= 128 * 1024**2,
            "confirmation input exceeds its per-clip RGB bound")
    hands, face, reader_type = native_helpers()
    with SampledRgbReader(video, header) as reader:
        left, right = hands(output_size=(side, side)).extract_hand_frames(reader, sampled)
    with SampledRgbReader(video, header) as reader:
        facial = face(output_size=(side, side)).extract_face_frames(reader, sampled)
    pixels = np.stack([pack_rgb(v, len(indices), side) for v in (facial, left, right)], axis=1)
    del facial, left, right
    reader = reader_type(video)
    try:
        left, right = hands(output_size=(side, side)).extract_hand_frames(NativeSampledReader(reader, header), sampled)
        facial = face(output_size=(side, side)).extract_face_frames(NativeSampledReader(reader, header), sampled)
    finally:
        del reader
    reference = np.stack([pack_rgb(v, len(indices), side) for v in (facial, left, right)], axis=1)
    require(np.array_equal(pixels, reference), "confirmation streamed/native RGB crop pixels differ")
    pose = pose_processor().process_landmarks_sequence(sampled).astype(np.float32)
    require(pose.shape == (len(indices), 14) and np.isfinite(pose).all(),
            "confirmation complete native pose stream changed")
    return {"pixels": pixels, "pose_features": pose, "frame_indices": indices, "header": header}


def array_receipt(values, side):
    frames = values["header"]["source_frames"]
    require(type(frames) is int and frames > 0 and values["header"]["source_fps"] == 25
            and set(values) == {"pixels", "pose_features", "frame_indices", "header"},
            "confirmation input header or array inventory changed")
    expected = np.arange(0, frames, 2, dtype=np.int64)
    require(isinstance(values["frame_indices"], np.ndarray) and values["frame_indices"].dtype == np.int64
            and np.array_equal(values["frame_indices"], expected)
            and values["header"]["sampled_frames"] == len(expected)
            and values["pixels"].shape == (len(expected), 3, side, side, 3)
            and values["pixels"].dtype == np.uint8 and values["pose_features"].shape == (len(expected), 14)
            and values["pose_features"].dtype == np.float32 and np.isfinite(values["pose_features"]).all(),
            "confirmation preflight arrays lost complete native shapes or timeline")
    return {"header": values["header"], "side": side,
            "arrays": {name: {"shape": list(value.shape), "dtype": str(value.dtype),
                              "sha256": hashlib.sha256(value.tobytes()).hexdigest()}
                       for name, value in values.items() if isinstance(value, np.ndarray)}}


def preflight(audit_path, output):
    output = output.absolute()
    owned(output, "checkpoints")
    require(not output.exists(), "confirmation input preflight needs a fresh output")
    require(all(os.environ.get(name) == "1" for name in
                ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "PYTHONDONTWRITEBYTECODE")),
            "confirmation data preflight requires the pinned one-CPU environment")
    import cv2
    cv2.setNumThreads(1)
    records = bind_records(audit_path)
    complete = read(audit_path)
    def guard(extra=0):
        paths = list(output.rglob("*")) if output.exists() else []
        require(all(not p.is_symlink() for p in paths), "confirmation preflight cannot follow output aliases")
        used = sum(p.stat().st_size for p in paths if p.is_file())
        require(used + extra <= 32 * 1024**2
                and shutil.disk_usage(ROOT).free >= 84 * 1024**3 + 128 * 1024**2 + 32 * 1024**2 - used,
                "confirmation preflight output cap or storage reserve failed")
    guard()
    output.mkdir(parents=True)
    from launch_terra_visual_job import identity, now
    from launch_terra_visual_job import write as write_receipt
    write_receipt(output / "worker.json", identity(os.getpid(), [sys.executable, *sys.argv]))
    code = 1
    try:
        receipts = []
        for i, record in enumerate(records):
            guard()
            result = {}
            for side in (224, 112):
                values = read_input(record, side=side)
                result[str(side)] = array_receipt(values, side)
                del values
            require(result["112"]["header"] == result["224"]["header"]
                    and result["112"]["arrays"]["pose_features"] == result["224"]["arrays"]["pose_features"]
                    and result["112"]["arrays"]["frame_indices"] == result["224"]["arrays"]["frame_indices"],
                    "confirmation arms changed pose or complete sampling grid")
            path = output / f"input-{i:04d}.json"
            write_receipt(path, {"record": record, "native_inputs": result})
            receipts.append(artifact(path))
            print(json.dumps({"verified_clips": i + 1, "of": len(records)}), flush=True)
        guard()
        require(all(sha256(ROOT / name) == digest for name, digest in complete["source_sha256"].items()),
                "confirmation input source changed during preflight")
        write_receipt(output / "result.json", {"schema": "umi-studio-broad-confirmation-native-inputs/1",
            "complete": True, "landmark_audit": artifact(audit_path), "receipts": receipts,
            "clips": 48, "source_sha256": complete["source_sha256"],
            "training_allowed": False, "candidate_bound_by_preflight": False, "confirmation_consumed_by_preflight": False,
            "model_or_scorer_execution": False})
        code = 0
    except BaseException as error:
        write_receipt(output / "failure.json", {"error": str(error), "automatic_retries": 0})
        raise
    finally:
        write_receipt(output / "exit.json", {"pid": os.getpid(), "exit_code": code, "observed_at": now()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-output", type=Path)
    parser.add_argument("--preflight-output", type=Path)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--plan", type=Path, default=PLAN)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--receipts", type=Path, default=RECEIPTS)
    parser.add_argument("--jobs", nargs=3, type=Path)
    args = parser.parse_args()
    if args.audit_output and args.jobs:
        owned(args.audit_output.absolute(), "runs")
        write(args.audit_output.absolute(), audit(args.plan.absolute(), args.data.absolute(),
                                                 args.receipts.absolute(), [p.absolute() for p in args.jobs]))
    elif args.audit and args.preflight_output:
        preflight(args.audit.absolute(), args.preflight_output)
    else:
        parser.error("audit three completed extraction batches or preflight the complete audited inputs")


if __name__ == "__main__":
    main()

