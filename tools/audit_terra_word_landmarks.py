"""Audit the fixed complete word sidecars after all extraction workers exit0."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from benchmark_fleurs_dev import sha256
from extract_mediapipe_landmarks import unpack_landmarks
from prepare_terra_word_landmarks import checked, checked_landmarks
from terra_training_recovery import artifact
from training_media_artifact import media_artifact

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "runs/terra-word-landmarks-v1/plan.json"
DATA = ROOT / "data/asl-stem-wiki/word-pilot-v1/landmarks-25fps"
RECEIPTS = PLAN.parent / "landmark-receipts"


def terminal(directory):
    worker = json.loads((directory / "worker.json").read_text())
    exit_record = json.loads((directory / "exit.json").read_text())
    if (
        exit_record.get("pid") != worker.get("pid")
        or exit_record.get("exit_code") != 0
        or (directory / "failure.json").exists()
    ):
        raise ValueError("word sidecar audit requires same-worker exit0 without failure")
    return {
        "worker": artifact(directory / "worker.json"),
        "terminal": artifact(directory / "exit.json"),
    }


def audit(path=PLAN, data=DATA, receipts=RECEIPTS):
    plan = checked(path)
    plan_sha = sha256(path)
    supervisor_dir = path.parent / "remaining-batches"
    supervisor = terminal(supervisor_dir)
    worker = json.loads((supervisor_dir / "worker.json").read_text())
    if (
        worker.get("plan_sha256") != plan_sha
        or worker.get("controller_sha256") != sha256(path.parent / "extract_remaining.py")
        or worker.get("planned_total_clips") != len(plan["records"])
        or worker.get("already_complete_clips") != 16
        or worker.get("maximum_batch_clips") != 16
        or worker.get("automatic_retries") != 0
        or len(plan["records"]) != 324
    ):
        raise ValueError("word sidecar supervisor changed its fixed inventory or controller")
    workers = [terminal(path.parent / f"batch-{i:04d}") for i in range(1, 22)]
    rows, counts, detections = [], Counter(), Counter()
    expected_data, expected_receipts = set(), set()
    frames = total_bytes = 0
    for record in plan["records"]:
        original = record["original"]["row"]
        target = data / original["role"] / (original["example_id"] + ".npz")
        receipt = receipts / original["role"] / (original["example_id"] + ".json")
        saved = json.loads(receipt.read_text())
        if (
            saved.get("plan_sha256") != plan_sha
            or saved.get("record") != record
            or saved.get("landmarks") != media_artifact(target)
            or saved.get("model_training_ready") is not False
            or saved.get("service_checks_performed") is not False
        ):
            raise ValueError("word sidecar receipt, original role or source bytes changed")
        metadata = checked_landmarks(target, record)
        if saved.get("metadata") != metadata:
            raise ValueError("word sidecar receipt metadata differs from actual sidecar")
        landmarks, _ = unpack_landmarks(target)
        for value in landmarks.values():
            if value is not None:
                for label, key in (
                    ("face", "face_landmarks"),
                    ("hands", "hand_landmarks"),
                    ("pose", "pose_landmarks"),
                ):
                    detections[f"frames_with_{label}"] += bool(value.get(key))
        frames += metadata["frames"]
        total_bytes += target.stat().st_size
        counts[original["role"]] += 1
        rows.append({"receipt": artifact(receipt), "landmarks": media_artifact(target)})
        expected_data.add(target)
        expected_receipts.add(receipt)
    if (
        set(data.rglob("*.npz")) != expected_data
        or set(receipts.rglob("*.json")) != expected_receipts
        or counts != {"training": 228, "development": 96}
        or frames != 35245
        or total_bytes > plan["policy"]["maximum_total_landmark_bytes"]
    ):
        raise ValueError("word sidecar audit lacks the exact complete frozen inventory")
    return {
        "schema": "umi-studio-word-complete-landmark-audit/1",
        "complete_landmark_audit": True,
        "plan": artifact(path),
        "supervisor": supervisor,
        "workers": workers,
        "by_role": dict(counts),
        "frames": frames,
        "landmark_bytes": total_bytes,
        "detection_frame_counts": dict(detections),
        "records": rows,
        "all_native_source_timeline_and_coordinate_checks_passed": True,
        "model_training_ready": False,
        "optimizer_updates": 0,
        "new_Terra_requests": 0,
        "source_sha256": {str(Path(__file__).relative_to(ROOT)): sha256(Path(__file__))},
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.absolute().is_relative_to(PLAN.parent):
        raise ValueError("word sidecar audit needs a fresh owned receipt")
    result = audit()
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"audit": artifact(args.output), "frames": result["frames"]}), flush=True)
