"""Synthetic audits and tiny video/coordinate inputs; no neural model or translation."""
from __future__ import annotations

import copy
import json
import os
import shutil
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import c5_broad_native_input as broad
from extract_mediapipe_landmarks import write_landmarks
from training_media_artifact import media_artifact


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def source_record(identity="synthetic-clip", family="how2sign", group="synthetic-group", signer=1):
    return {"original": {"row": {"role": "confirmation", "example_id": identity, "training_allowed": False}},
            "example": {"clip_id": identity, "family": family, "source_group": family + ":" + group,
                        "signer": signer, "training_allowed": False},
            "ROI_lineage_training_allowed": False}


@pytest.fixture
def owned(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    monkeypatch.setattr(broad, "ROOT", root)

    def artifact(path):
        return {"path": str(path.relative_to(root)), "sha256": broad.sha256(path)}

    monkeypatch.setattr(broad, "artifact", artifact)
    monkeypatch.setattr(broad, "media_artifact", partial(media_artifact, root=root))
    monkeypatch.setattr(broad, "terminal", broad.preparation.bind(broad.terminal, artifact=artifact))
    return root


@pytest.mark.parametrize("kind", ["purpose", "role", "allowed", "roi", "identity", "family"])
def test_confirmation_can_never_be_training_or_development_before_file_reads(kind):
    record = source_record()
    purpose = "confirmation"
    if kind == "purpose":
        purpose = "training"
    elif kind == "role":
        record["original"]["row"]["role"] = "development"
    elif kind == "allowed":
        record["example"]["training_allowed"] = True
    elif kind == "roi":
        record["ROI_lineage_training_allowed"] = True
    elif kind == "identity":
        record["original"]["row"]["example_id"] = "other"
    else:
        record["example"]["family"] = "other"
    with pytest.raises(ValueError, match="source roles"):
        broad.read_input(record, purpose=purpose)


def test_rgb_packing_keeps_every_frame_and_restores_native_channel_order():
    frames = [np.tile(np.array([i, i + 1, i + 2], dtype=np.uint8), (112, 112, 1)) for i in (1, 10)]
    actual = broad.pack_rgb(frames, 2, 112)
    assert actual.dtype == np.uint8 and actual.flags.c_contiguous
    assert actual[:, 0, 0].tolist() == [[3, 2, 1], [12, 11, 10]]
    with pytest.raises(ValueError):
        broad.pack_rgb(frames[:1], 2, 112)


@pytest.mark.parametrize("side", [112, 224])
def test_receipt_accepts_only_complete_original_grid_and_finite_fp32_pose(side):
    indices = np.array([0, 2, 4], dtype=np.int64)
    values = {"pixels": np.zeros((3, 3, side, side, 3), dtype=np.uint8),
              "pose_features": np.full((3, 14), -9999, dtype=np.float32),
              "frame_indices": indices,
              "header": {"source_frames": 5, "source_fps": 25, "sampled_frames": 3}}
    assert broad.array_receipt(values, side)["arrays"]["frame_indices"]["dtype"] == "int64"
    for kind in ("fractional_grid", "short", "nonfinite_pose", "wrong_pixel_dtype"):
        changed = copy.deepcopy(values)
        if kind == "fractional_grid":
            changed["frame_indices"] = indices.astype(np.float64)
        elif kind == "short":
            changed["pixels"] = changed["pixels"][:2]
        elif kind == "nonfinite_pose":
            changed["pose_features"][0, 0] = np.nan
        else:
            changed["pixels"] = changed["pixels"].astype(np.float32)
        with pytest.raises(ValueError):
            broad.array_receipt(changed, side)


@pytest.fixture
def complete_sidecars(owned, monkeypatch):
    root = owned
    data, receipts = root / "data/sidecars", root / "runs/receipts"
    plan_path = root / "runs/plan/plan.json"
    save(plan_path, {"synthetic": True})
    helper = root / "tools/synthetic.py"
    helper.parent.mkdir()
    helper.write_text("# synthetic source")
    pinned = {"tools/synthetic.py": broad.sha256(helper)}
    monkeypatch.setattr(broad, "sources", lambda parent: dict(pinned))
    video = root / "data/clips/input.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"synthetic source bytes, no video decoding in sidecar audit")
    templates = {}
    for count in (169, 170):
        path = root / f"template-{count}.npz"
        write_landmarks(path, {i: None for i in range(count)}, source=video,
                        width=4, height=4, fps=25, mediapipe_version="0.10.14")
        templates[count] = path
    records = []
    for i in range(48):
        family = "how2sign" if i < 24 else "openasl"
        signer = (1, 2, 3, 5)[i // 6] if i < 24 else None
        group = f"synthetic-group-{i // 2}" if i < 24 else f"synthetic-group-{i}"
        record = source_record(f"synthetic-{i:04d}", family, group, signer)
        count = 170 if i < 14 else 169
        record.update(clip=broad.media_artifact(video),
                      normalized_probe={"width": 4, "height": 4, "decoded_frames": count})
        target = data / "confirmation" / (record["example"]["clip_id"] + ".npz")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(templates[count], target)
        _, metadata = broad.unpack_landmarks(target)
        receipt = receipts / "confirmation" / (record["example"]["clip_id"] + ".json")
        save(receipt, {"schema": "umi-studio-broad-confirmation-landmark-receipt/1",
                       "plan_sha256": broad.sha256(plan_path), "record": record,
                       "landmarks": broad.media_artifact(target), "metadata": metadata,
                       "training_allowed": False, "confirmation_consumed": False,
                       "model_training_ready": False, "service_checks_performed": False})
        records.append(record)
    parent = {"records": records, "training_allowed": False, "source_sha256": pinned,
              "docker_image_id": "sha256:synthetic", "policy": {
                  "maximum_landmark_clip_bytes": 16 * 1024**2, "maximum_total_landmark_bytes": 1024**3}}
    monkeypatch.setattr(broad.preparation, "checked", lambda _: parent)
    poll, audit = root / "runs/release/poll.json", root / "runs/release/audit.json"
    save(poll, {"synthetic": True})
    save(audit, {"synthetic": True})
    release = {"terminal_poll": broad.artifact(poll), "terminal_audit": broad.artifact(audit)}
    monkeypatch.setattr(broad.preparation, "admit_release", lambda *args: release)
    image = broad.preparation.IMAGE

    def command(*, clip, output_name, **kwargs):
        return ["docker", "run", "--network", "none", "--cpus", "1", "--pull", "never",
                image, "--input", str(clip), "--output", "/output/" + output_name]

    monkeypatch.setattr(broad.preparation, "container_command", command)
    jobs = [root / f"runs/batch-{i}" for i in range(3)]
    for i, job in enumerate(jobs):
        argv = ["python", str(root / "tools/prepare_c5_broad_landmarks.py"),
                "--plan", str(plan_path), "--data", str(data), "--receipts", str(receipts),
                "--job", str(job), "--limit", "16"]
        save(job / "worker.json", {"uid": os.getuid(), "pid": 100 + i, "argv": argv})
        save(job / "exit.json", {"pid": 100 + i, "exit_code": 0})
        save(job / "training-release.json", release)
        for j, record in enumerate(records[i * 16:(i + 1) * 16]):
            arguments = command(clip=video, output_name=record["example"]["clip_id"] + ".npz")
            arguments[arguments.index(image)] = parent["docker_image_id"]
            save(job / f"container-{j:02d}.json", {"command": arguments, "automatic_retries": 0})
            (job / f"container-{j:02d}.log").write_text("synthetic detector log; no detector executed")
    return plan_path, data, receipts, jobs, records


def test_all48_complete_sidecars_and_three_worker_histories_are_bound(complete_sidecars):
    plan, data, receipts, jobs, _ = complete_sidecars
    result = broad.audit(plan, data, receipts, jobs)
    assert result["frames"] == 8126 and len(result["records"]) == 48
    assert result["source_groups"] == 36 and result["families"] == {"how2sign": 24, "openasl": 24}
    assert result["confirmation_consumed_by_audit"] is False
    assert len(result["worker_authority"]) == 3


@pytest.mark.parametrize("kind", ["failed_worker", "missing_command", "repeated_command", "changed_model", "orphan"])
def test_partial_or_changed_worker_sidecar_inventory_cannot_pass(complete_sidecars, kind):
    plan, data, receipts, jobs, _ = complete_sidecars
    if kind == "failed_worker":
        save(jobs[0] / "exit.json", {"pid": 100, "exit_code": 1})
    elif kind == "missing_command":
        (jobs[0] / "container-00.json").unlink()
    elif kind in ("repeated_command", "changed_model"):
        path = jobs[0] / "container-00.json"
        record = json.loads(path.read_text())
        if kind == "repeated_command":
            record = json.loads((jobs[0] / "container-01.json").read_text())
        else:
            record["command"][record["command"].index("sha256:synthetic")] = "other"
        save(path, record)
    else:
        (data / "confirmation/orphan.partial.npz").write_bytes(b"synthetic partial")
    with pytest.raises(ValueError):
        broad.audit(plan, data, receipts, jobs)


@pytest.fixture
def readable(owned):
    import cv2
    root = owned
    video = root / "data/tiny.mp4"
    video.parent.mkdir()
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 25, (16, 16))
    assert writer.isOpened()
    for i in range(5):
        writer.write(np.full((16, 16, 3), i * 20, dtype=np.uint8))
    writer.release()
    landmark = root / "data/tiny.npz"
    rows = {i: None for i in range(5)}
    rows[2] = {"pose_landmarks": None, "face_landmarks": None, "hand_landmarks": None}
    metadata = write_landmarks(landmark, rows, source=video, width=16, height=16,
                               fps=25, mediapipe_version="0.10.14")
    plan = root / "runs/tiny-plan.json"
    save(plan, {"synthetic": True})
    record = {**source_record(), "clip": broad.media_artifact(video),
              "normalized_probe": {"width": 16, "height": 16, "decoded_frames": 5}}
    receipt = root / "runs/tiny-receipt.json"
    save(receipt, {"schema": "umi-studio-broad-confirmation-landmark-receipt/1",
                   "plan_sha256": broad.sha256(plan), "record": record, "metadata": metadata,
                   "landmarks": broad.media_artifact(landmark), "training_allowed": False})
    bound = {**record, "landmarks": broad.media_artifact(landmark),
             "landmark_receipt": broad.artifact(receipt), "landmark_plan": broad.artifact(plan)}
    helper = root / "tools/synthetic.py"
    helper.parent.mkdir()
    helper.write_text("# synthetic binding")
    audit = root / "runs/tiny-audit.json"
    save(audit, {"complete": True, "training_allowed": False, "plan": broad.artifact(plan),
                 "records": [bound] * 48, "source_sha256": {"tools/synthetic.py": broad.sha256(helper)}})
    return {**bound, "broad_audit": broad.artifact(audit)}


def test_actual_tiny_video_native224_and_native112_get_identical_complete_grid_and_pose(readable):
    original = broad.read_input(readable, side=224)
    candidate = broad.read_input(readable, side=112)
    assert original["pixels"].shape == (3, 3, 224, 224, 3)
    assert candidate["pixels"].shape == (3, 3, 112, 112, 3)
    assert original["header"] == candidate["header"]
    assert candidate["frame_indices"].tolist() == [0, 2, 4]
    assert np.array_equal(original["pose_features"], candidate["pose_features"])
    assert np.isfinite(candidate["pose_features"]).all()


def test_orphan_membership_or_changed_receipt_is_rejected_before_decode(readable):
    bad = copy.deepcopy(readable)
    bad["example"]["source_group"] = "how2sign:other"
    with pytest.raises(ValueError, match="membership"):
        broad.read_input(bad)
    receipt = broad.ROOT / readable["landmark_receipt"]["path"]
    saved = json.loads(receipt.read_text())
    saved["record"]["ROI_lineage_training_allowed"] = True
    save(receipt, saved)
    with pytest.raises(ValueError, match="receipt identity"):
        broad.read_input(readable)

