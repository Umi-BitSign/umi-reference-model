"""Freeze and prepare a source-held-out paired face-learning pilot."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import numpy as np

from audit_c5_all_source_corpus import exclusion_reason, protected_sets
from benchmark_fleurs_dev import normalize, sha256
from c5_experiment_sources import archive_sources
from c5_fresh_face_benchmark import calibrate
from extract_mediapipe_landmarks import unpack_landmarks
from prepare_asl_stem_wiki_landmarks import DOCKER_SOCKET, docker_image_id

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "artifacts/c5-corrective-stride2-crop168-confirmed"
BASELINE = ROOT / "artifacts/baseline-6fe8df59"
SIGNERS = (1, 2, 3, 4, 5, 8, 9, 11)
SEED = 281
ARMS = ("legacy_faces", "recovered_faces")
RECIPE = {"rank": 16, "alpha": 32., "visual_last_layers": 4,
    "visual_learning_rate": 1e-4, "decoder_learning_rate": 4e-6,
    "weight_decay": .01, "label_smoothing": .1, "gradient_accumulation": 4,
    "epochs": 4, "evaluation_epochs": [0, 1, 2, 4], "warmup_updates": 8,
    "minimum_learning_rate_ratio": .1, "retention_coefficient": 1., "retention_temperature": 2.,
    "device": "mps", "gpu_idle_fraction": .5, "cpu_threads": 2,
    "maximum_process_mps_bytes": 6 * 1024**3, "generation_num_beams": 5, "generation_max_length": 2048}
GATES = {"minimum_relative_mean_wer_improvement_vs_same_input_parent": .05,
         "minimum_improved_signers": 3, "maximum_signer_regression": .05,
         "maximum_word_weighted_delta": 0., "maximum_source_bootstrap_95_upper_delta": 0.,
         "bootstrap_samples": 5000, "bootstrap_seed": 1729}


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def key(value: str) -> str:
    return hashlib.sha256(f"face-learning-pilot-{SEED}:{value}".encode()).hexdigest()


def source_group(row: dict) -> str:
    if row["source"].startswith("how2sign"):
        match = re.match(r"^([A-Za-z0-9_-]{11})_", row["id"])
        if match is None: raise ValueError("How2Sign source identity missing")
        return "how2sign:" + match.group(1)
    match = re.match(r"^(dev_document_\d+)_sentence_", row["id"])
    if row["source"] != "fleurs" or match is None: raise ValueError("unknown pilot source")
    return "fleurs:" + match.group(1)


def source_fair_take(rows: list[dict], count: int, cap: int) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows: grouped[source_group(row)].append(row)
    for name in grouped: grouped[name].sort(key=lambda row: key(row["id"]))
    names = sorted(grouped, key=key); selected = []
    for offset in range(cap):
        for name in names:
            if offset < len(grouped[name]): selected.append(grouped[name][offset])
            if len(selected) == count: return selected
    raise ValueError("source-capped pilot capacity is insufficient")


def select(rows: list[dict]) -> tuple[list[dict], list[dict], set[str]]:
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["source"].startswith("how2sign"): grouped[row["signer"]][source_group(row)].append(row)
    if set(grouped) != set(SIGNERS): raise ValueError("pilot lacks a How2Sign signer")
    held_groups = set(); designated = {}
    # Reserve scarce actors first. Exclude held content IDs globally across
    # actors, not merely the interpreter-specific physical recording.
    for signer in sorted(grouped, key=lambda s: (len(grouped[s]), s)):
        candidates = sorted((name for name, items in grouped[signer].items() if len(items) >= 4), key=key)
        for pair in itertools.combinations(candidates, 2):
            proposed = held_groups | set(pair)
            capacity = [sum(min(10, len(items)) for name, items in groups.items() if name not in proposed) for groups in grouped.values()]
            if min(capacity) >= 32:
                held_groups = proposed; designated[signer] = pair; break
        else: raise ValueError(f"no source-separated pilot partition for signer {signer}")
    holdout = []
    for signer in SIGNERS:
        for name in designated[signer]:
            holdout.extend(sorted(grouped[signer][name], key=lambda row: key(row["id"]))[:4])
    held_texts = {tuple(normalize(row["reference"])) for row in holdout}
    eligible = [r for r in rows if source_group(r) not in held_groups and tuple(normalize(r["reference"])) not in held_texts]
    training = []
    for signer in SIGNERS:
        training.extend(source_fair_take([r for r in eligible if r["source"].startswith("how2sign") and r["signer"] == signer], 32, 10))
    training.extend(source_fair_take([r for r in eligible if r["source"] == "fleurs"], 64, 4))
    if len(training) != 320 or len(holdout) != 64 or {source_group(r) for r in training} & held_groups:
        raise ValueError("pilot source split changed")
    if len({r["id"] for r in training + holdout}) != 384: raise ValueError("pilot repeats a canonical clip")
    return training, holdout, held_groups


def balanced_epoch(rows: list[dict], epoch: int) -> list[dict]:
    grouped = {s: [] for s in SIGNERS}; fleurs = []
    for row in rows:
        if row["domain"] == "fleurs": fleurs.append(row)
        else: grouped[row["signer"]].append(row)
    if any(len(items) != 32 for items in grouped.values()) or len(fleurs) != 64: raise ValueError("pilot schedule counts differ")
    def order(row): return key(f"epoch-{epoch}:{row['example_id']}")
    for items in grouped.values(): items.sort(key=order)
    fleurs.sort(key=order); schedule = []
    actor_order = sorted(SIGNERS, key=lambda s: key(f"epoch-{epoch}:signer-{s}"))
    for index in range(32):
        for offset, signer in enumerate(actor_order):
            schedule.append(grouped[signer][index])
            if offset in (3, 7): schedule.append(fleurs[index * 2 + (offset == 7)])
    return schedule


def sources() -> dict:
    names = ("c5_face_learning_pilot.py", "train_c5_face_learning_pilot.py", "shubert_lora.py", "shubert_visual_lora.py",
             "c5_retention.py", "c5_face_recovery_runtime.py", "c5_parallel_face_recovery_runtime.py", "probe_c5_face_detection.py",
             "c5_fresh_face_benchmark.py", "extract_mediapipe_landmarks.py", "c5_stride2_runtime.py", "recover_c5_pilot_cached_landmarks.py")
    return {name: sha256(ROOT / "tools" / name) for name in names}


def freeze(output: Path) -> None:
    output = output.resolve()
    if output.exists() or not output.is_relative_to(ROOT / "runs"): raise ValueError("pilot requires a new runs/ output")
    manifest = ROOT / "runs/c5-all-source-corpus-v1/training-streams.jsonl"
    media_path = ROOT / "runs/fresh-lowres-v4/training-inventory.jsonl"
    media = {row["example_id"]: row for row in read_rows(media_path)}
    normalized_index = ROOT / "runs/how2sign-precut-expansion-v1/stream-cache-stride2-index.jsonl"
    normalized_rows = {row["sentence_name"]: row for row in read_rows(normalized_index)}
    groups, protected_texts = protected_sets()
    fresh_path = ROOT / "runs/c5-fresh-face-benchmark-v1/plan.json"
    fresh = json.loads(fresh_path.read_text())
    groups["how2sign"].update(row["video_id"] for row in fresh["examples"])
    protected_texts.update(tuple(normalize(row["reference"])) for row in fresh["examples"])
    rows = [row for row in read_rows(manifest) if row["source"].startswith("how2sign") or row["source"] == "fleurs"]
    rows = [row for row in rows if not exclusion_reason(row, groups, protected_texts)]
    training, holdout, held_groups = select(rows)
    examples = []
    for role, items in (("training", training), ("pilot_source_holdout", holdout)):
        for row in items:
            raw = media[row["id"]]; clip = ROOT / raw["video"]; clip_hash = raw["video_sha256"]; fps = raw["video_fps"]
            cached_landmarks = None
            if row["source"] == "how2sign_precut_expansion":
                normalized = normalized_rows[row["id"]]
                if normalized["reference"] != row["reference"] or normalized["signer"] != row["signer"] or normalized["encoder_cache_sha256"] != row["cache_sha256"]:
                    raise ValueError("verified normalized expansion differs from the eligible corpus")
                clip = ROOT / "data/how2sign/precut-expansion-clips-25fps" / f"{row['id']}.mp4"
                clip_hash = normalized["resampled_clip_sha256"]; fps = normalized["clip_fps"]
                landmark_path = ROOT / "data/how2sign/precut-expansion-landmarks-25fps" / f"{row['id']}.npz"
                _, metadata = unpack_landmarks(landmark_path)
                if sha256(landmark_path) != normalized["landmarks_sha256"] or metadata["source_sha256"] != clip_hash or metadata["fps"] != 25. or metadata["mediapipe_version"] != "0.10.14":
                    raise ValueError("reusable normalized landmark provenance differs")
                cached_landmarks = {"path": str(landmark_path.relative_to(ROOT)), "sha256": normalized["landmarks_sha256"]}
            if raw["reference"] != row["reference"] or raw["signer"] != row["signer"] or fps != 25. or sha256(clip) != clip_hash:
                raise ValueError("pilot media/eligible-stream join differs")
            examples.append({"example_id": row["id"], "source": row["source"], "signer": row["signer"],
                "domain": "fleurs" if row["source"] == "fleurs" else "how2sign", "source_group": source_group(row),
                "reference": row["reference"], "role": role, "training_allowed": role == "training",
                "clip": str(clip.relative_to(ROOT)), "clip_sha256": clip_hash, "clip_fps": fps,
                "original_media_fps": raw["video_fps"], "reusable_legacy_landmarks": cached_landmarks})
    if docker_image_id(DOCKER_SOCKET) != fresh["image_id"]: raise ValueError("pilot CPU image changed")
    assets = ("MODEL_IDENTITY.json", "models/checkpoint-11625/model.safetensors", "crop_calibrator.safetensors",
              "models/dinov2hand.safetensors", "models/dinov2face.safetensors", "models/face_landmarker_v2_with_blendshapes.task", "models/hand_landmarker.task",
              "vendor/demo/inference.py", "vendor/demo/body_features.py", "vendor/demo/crop_hands.py", "vendor/demo/crop_face.py", "vendor/demo/dinov2_features.py")
    asset_hashes = {name: sha256(BUNDLE / name) for name in assets}
    if asset_hashes["models/checkpoint-11625/model.safetensors"] != fresh["selected_checkpoint_sha256"] or asset_hashes["crop_calibrator.safetensors"] != fresh["calibrator_sha256"]:
        raise ValueError("selected parent model or crop calibrator changed")
    probe_path = ROOT / "runs/c5-face-learning-mps-backend-probe-v1.json"
    probe = json.loads(probe_path.read_text())
    if not probe["passes"] or probe["optimizer_updates"] != 0 or probe["parent_checkpoint_sha256"] != asset_hashes["models/checkpoint-11625/model.safetensors"]:
        raise ValueError("the selected-parent MPS gradient probe did not pass")
    plan = {"schema": "umi-studio-c5-face-learning-pilot-plan/1", "seed": SEED, "arms": ARMS,
        "examples": examples, "recipe": RECIPE, "gates": GATES, "source_sha256": sources(), "asset_sha256": asset_hashes,
        "source_manifest_sha256": sha256(manifest), "raw_media_manifest_sha256": sha256(media_path),
        "normalized_expansion_index_sha256": sha256(normalized_index),
        "backend_probe": str(probe_path.relative_to(ROOT)), "backend_probe_sha256": sha256(probe_path),
        "fresh_development_plan_sha256": sha256(fresh_path), "image_id": fresh["image_id"],
        "data_dir": str((ROOT / "data" / output.name).relative_to(ROOT)), "data_budget_bytes": 4 * 1024**3,
        "minimum_free_bytes": 80 * 1024**3, "prepare_device": "cpu", "cpu_threads": 2,
        "training_examples": 320, "holdout_examples": 64, "held_content_groups": sorted(held_groups),
        "preparation_bounds": {"maximum_how2sign_clips_per_process": 128, "maximum_fleurs_clips_per_process": 32},
        "training_policy": "32 How2Sign clips per eight signers, 64 FLEURS replay clips; source-fair within signer; all arms use identical four-pass balanced order, rates and teacher KL",
        "feature_policy": "Native-aspect 25fps, CPU MediaPipe, post-MediaPipe stride2, DINO crop168 plus frozen calibrator. Arms differ only in sampled missing-face recovery; hands, pose and unrecovered faces stay exact.",
        "teacher_policy": "Cache frozen initial-parent logits once on training-reference prefixes before any update in each arm; never cache teacher targets for the diagnostic holdout.",
        "evaluation_policy": "Decode only the 64 pilot source-holdout clips at epochs 0,1,2,4, always using recovered-face inputs for both arms. This isolates learning from the existing fixed-input quality gain.",
        "selection_policy": "Lowest passing mean WER across either training arm and epochs 1,2,4; earlier epoch then lexical arm on exact ties. Report matched-epoch recovery-versus-control deltas separately; do not assume recovered-face training wins.",
        "qualification": "Diagnostic source-held-out relative to this pilot, not guaranteed novel to historical models or upstream pretraining; no public confirmation, reserve, test, devtest or private material. The consumed OpenASL confirmation is forbidden for training and further selection.",
        "miner_health_policy": "user_authorized_maintenance", "miner_changed": False,
        "gpu_policy": "Training uses equal active/idle MPS duty cycle, bounding average occupancy; it cannot cap an individual kernel's instantaneous utilization. Six-GiB process allocation guard leaves memory headroom."}
    output.mkdir(); path = output / "plan.json"
    path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    archive_sources(path, output / "source-snapshot")
    print(json.dumps({"plan_sha256": sha256(path), "training_examples": 320, "holdout_examples": 64, "held_source_groups": len(held_groups)}))


def storage(plan: dict) -> None:
    data = ROOT / plan["data_dir"]
    if shutil.disk_usage(ROOT).free < plan["minimum_free_bytes"] + 128 * 1024**2:
        raise RuntimeError("face pilot storage floor reached")
    if sum(p.stat().st_size for p in data.rglob("*") if p.is_file()) > plan["data_budget_bytes"] - 128 * 1024**2:
        raise RuntimeError("face pilot data budget reached")


def checked_plan(path: Path) -> dict:
    plan = json.loads(path.read_text())
    if plan["schema"] != "umi-studio-c5-face-learning-pilot-plan/1" or plan["recipe"] != RECIPE or plan["gates"] != GATES or plan["source_sha256"] != sources():
        raise ValueError("frozen face pilot implementation or recipe changed")
    if any(sha256(BUNDLE / name) != value for name, value in plan["asset_sha256"].items()): raise ValueError("face pilot parent asset changed")
    snapshot = json.loads((path.parent / "source-snapshot/manifest.json").read_text())
    if snapshot["plan_sha256"] != sha256(path) or sha256(ROOT / plan["backend_probe"]) != plan["backend_probe_sha256"]:
        raise ValueError("pilot plan or backend evidence changed after freezing")
    return plan


def prepare(plan_path: Path) -> None:
    plan = checked_plan(plan_path)
    data = ROOT / plan["data_dir"]
    data.mkdir(exist_ok=True); (data / "features").mkdir(exist_ok=True); (data / "landmarks").mkdir(exist_ok=True)
    receipt_path = plan_path.parent / "features.jsonl"
    completed = read_rows(receipt_path) if receipt_path.exists() else []
    for row, expected in zip(completed, plan["examples"]):
        if row["plan_sha256"] != sha256(plan_path) or any(row[k] != value for k, value in expected.items()):
            raise ValueError("completed pilot preparation is not the frozen prefix")
        if any(sha256(ROOT / target["path"]) != target["sha256"] for target in row["features"].values()):
            raise ValueError("completed pilot feature cache changed")
    if len(completed) > len(plan["examples"]): raise ValueError("pilot preparation exceeds the frozen cohort")
    for name, value in {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
                        "SHUBERT_DINOV2_SOURCE": str(BUNDLE / "vendor/dinov2-source"), "TORCH_HOME": str(BASELINE / "run/torch")}.items(): os.environ[name] = value
    sys.path.insert(0, str(BUNDLE / "vendor/demo")); sys.path.insert(0, str(BUNDLE / "vendor"))
    import cv2
    import torch
    from body_features import process_pose_landmarks
    from crop_face import FaceExtractor
    from crop_hands import HandExtractor
    from dinov2_features import DINOEmbedder
    from safetensors.numpy import load_file
    from torchvision import transforms
    from c5_stride2_runtime import temporal_subsample_landmarks
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    hand = DINOEmbedder(str(BUNDLE / "models/dinov2hand.safetensors"), batch_size=32, device="cpu")
    face = DINOEmbedder(str(BUNDLE / "models/dinov2face.safetensors"), batch_size=32, device="cpu")
    transform = transforms.Compose([transforms.Resize((168, 168)), transforms.ToTensor(), transforms.Normalize(mean=[.485,.456,.406], std=[.229,.224,.225])])
    hand.transform = transform; face.transform = transform
    hand_crop = HandExtractor(output_size=(168, 168)); face_crop = FaceExtractor(output_size=(168, 168))
    calibration = load_file(str(BUNDLE / "crop_calibrator.safetensors"))
    env = {**os.environ, "DOCKER_HOST": DOCKER_SOCKET}
    new_counts = Counter()
    for index, row in enumerate(plan["examples"]):
        if index < len(completed): continue
        if new_counts[row["domain"]] >= (128 if row["domain"] == "how2sign" else 32): break
        storage(plan); clip = ROOT / row["clip"]
        if sha256(clip) != row["clip_sha256"]: raise ValueError("pilot clip changed")
        directory = data / "landmarks" / f"request-{index:03d}"
        command = ["docker", "run", "--rm", "--network", "none", "--cpus", "2", "--memory", "3g",
            "--volume", f"{ROOT}:/workspace:ro", "--volume", f"{directory.parent}:/output:rw", plan["image_id"], "python",
            "/workspace/tools/c5_parallel_face_recovery_runtime.py", "--input", f"/workspace/{row['clip']}",
            "--output-dir", f"/output/{directory.name}", "--recovery",
            "--face-model", "/workspace/artifacts/c5-corrective-stride2-crop168-confirmed/models/face_landmarker_v2_with_blendshapes.task",
            "--hand-model", "/workspace/artifacts/c5-corrective-stride2-crop168-confirmed/models/hand_landmarker.task"]
        if row["reusable_legacy_landmarks"] is not None:
            cached = row["reusable_legacy_landmarks"]
            if sha256(ROOT / cached["path"]) != cached["sha256"]: raise ValueError("reusable pilot landmarks changed")
            command = ["docker", "run", "--rm", "--network", "none", "--cpus", "2", "--memory", "3g",
                "--volume", f"{ROOT}:/workspace:ro", "--volume", f"{directory.parent}:/output:rw", plan["image_id"], "python",
                "/workspace/tools/recover_c5_pilot_cached_landmarks.py", "--input", f"/workspace/{row['clip']}",
                "--legacy", f"/workspace/{cached['path']}", "--output-dir", f"/output/{directory.name}",
                "--face-model", "/workspace/artifacts/c5-corrective-stride2-crop168-confirmed/models/face_landmarker_v2_with_blendshapes.task"]
        result = subprocess.run(command, env=env, capture_output=True, text=True)
        (plan_path.parent / f"prepare-{index:03d}.log").write_text(result.stdout + result.stderr)
        if result.returncode: raise RuntimeError(f"pilot landmark extraction failed: {result.stderr[-2000:]}")
        landmarks, metadata = unpack_landmarks(directory / "legacy.npz")
        if metadata["source_sha256"] != row["clip_sha256"] or metadata["fps"] != 25.: raise ValueError("pilot landmark source differs")
        with np.load(directory / "recovery.npz", allow_pickle=False) as payload:
            mask = payload["recovery_mask"][::2]; points = payload["recovered_faces"][::2]
        capture = cv2.VideoCapture(str(clip)); frames = []
        try:
            while True:
                valid, bgr = capture.read()
                if not valid: break
                frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        finally: capture.release()
        if len(frames) != metadata["frames"]: raise ValueError("pilot decoded frame count differs")
        frames = np.ascontiguousarray(np.asarray(frames)[::2]); landmarks = temporal_subsample_landmarks(landmarks, 2)
        left, right = hand_crop.extract_hand_frames(frames, landmarks); faces = face_crop.extract_face_frames(frames, landmarks)
        streams = {"left_hand_features": calibrate(hand.extract_embeddings_from_frames([f[..., ::-1].copy() for f in left]), calibration, "hand"),
                   "right_hand_features": calibrate(hand.extract_embeddings_from_frames([f[..., ::-1].copy() for f in right]), calibration, "hand"),
                   "face_features": calibrate(face.extract_embeddings_from_frames([f[..., ::-1].copy() for f in faces]), calibration, "face"),
                   "pose_features": np.asarray(process_pose_landmarks(landmarks), dtype=np.float32)}
        recovered = {name: values.copy() for name, values in streams.items()}
        crops = [face_crop.resize_frame(face_crop.cues_on_grey_background(frames[i], points[i].tolist()), (168, 168)) for i in np.flatnonzero(mask)]
        if crops: recovered["face_features"][mask] = calibrate(face.extract_embeddings_from_frames(crops), calibration, "face")
        if any(not np.array_equal(streams[name], recovered[name]) for name in streams if name != "face_features") or not np.array_equal(streams["face_features"][~mask], recovered["face_features"][~mask]):
            raise RuntimeError("pilot recovery changed an unrelated input")
        cache_paths = {}
        for arm, values in zip(ARMS, (streams, recovered)):
            if any(v.dtype != np.float32 or not np.isfinite(v).all() for v in values.values()): raise ValueError("pilot features are invalid")
            path = data / "features" / f"{index:03d}--{arm}.npz"
            with path.open("xb") as output: np.savez_compressed(output, **values)
            cache_paths[arm] = {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
        receipt = {**row, "schema": "umi-studio-c5-face-learning-pilot-features/1", "plan_sha256": sha256(plan_path),
            "source_frames": metadata["frames"], "frames": len(mask), "recovered_frames": int(mask.sum()), "features": cache_paths,
            "legacy_landmarks_sha256": sha256(directory / "legacy.npz"), "recovery_landmarks_sha256": sha256(directory / "recovery.npz"),
            "nonface_and_unrecovered_preserved": True}
        with (plan_path.parent / "features.jsonl").open("a") as output:
            output.write(json.dumps(receipt, sort_keys=True) + "\n"); output.flush(); os.fsync(output.fileno())
        print(f"prepared {index + 1}/384 {row['role']} {row['domain']} signer={row['signer']} recovered={int(mask.sum())}/{len(mask)}", flush=True)
        new_counts[row["domain"]] += 1


def prepare_all(plan_path: Path) -> None:
    checked_plan(plan_path)
    receipts = plan_path.parent / "features.jsonl"
    for _ in range(8):
        before = len(read_rows(receipts)) if receipts.exists() else 0
        if before == 384: return
        result = subprocess.run([sys.executable, str(Path(__file__)), "--prepare", str(plan_path)])
        if result.returncode: raise RuntimeError("bounded pilot preparation batch failed; inspect its exact committed prefix")
        after = len(read_rows(receipts))
        if after <= before: raise RuntimeError("pilot preparation made no committed progress")
    raise RuntimeError("bounded preparation exceeded the expected batch count")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze", type=Path); group.add_argument("--prepare", type=Path); group.add_argument("--prepare-all", type=Path)
    args = parser.parse_args()
    if args.freeze: freeze(args.freeze)
    elif args.prepare: prepare(args.prepare.resolve())
    else: prepare_all(args.prepare_all.resolve())


if __name__ == "__main__": main()
