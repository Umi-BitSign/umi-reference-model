"""Freeze and run complete hybrid CPU inference, parity and open FLEURS checks.

Linux MediaPipe graphs run freshly for every request; model execution runs on
the host. No cached landmarks substitute for detection in the measured path.
This cannot prove native macOS detector portability or official UMI scores.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

import numpy as np

from benchmark_fleurs_dev import distance, normalize, sha256
from c5_experiment_sources import archive_sources
from c5_face_recovery_runtime import RecoveryLandmarks, install_face_recovery
from evaluate_fresh_lowres_tuning import load_tuning
from extract_mediapipe_landmarks import packed_landmarks, unpack_landmarks
from prepare_asl_stem_wiki_landmarks import DOCKER_SOCKET, docker_image_id

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "artifacts/c5-corrective-stride2-crop168-confirmed"
BASELINE = ROOT / "artifacts/baseline-6fe8df59"
IMAGE = "sha256:5cad3554c60a6b0bf7d0b059dde0fed09c5a357cd0142c6dfa0b6fa9cfe63378"
ARMS = ("confirmed_crop168", "confirmed_crop168_recovery")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def sources() -> dict[str, str]:
    return {name: sha256(ROOT / "tools" / name) for name in (
        "verify_c5_face_recovery_runtime.py", "c5_face_recovery_runtime.py", "probe_c5_face_detection.py",
        "extract_mediapipe_landmarks.py", "c5_stride2_runtime.py", "resolution_policy.py", "c5_crop_calibrator.py",
        "c5_parallel_face_recovery_runtime.py",
    )}


def landmark_fingerprint(landmarks: dict) -> str:
    digest = hashlib.sha256()
    for name, values in sorted(packed_landmarks(landmarks).items()):
        digest.update(name.encode()); digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()


def freeze(output: Path, *, parallel_legacy: bool = False) -> None:
    output = output.resolve()
    if output.exists() or not output.is_relative_to(ROOT / "runs"):
        raise ValueError("runtime validation needs a new runs/ directory")
    fresh_dir = ROOT / "runs/c5-fresh-face-benchmark-v1"
    fresh = json.loads((fresh_dir / "plan.json").read_text())
    decision = json.loads((fresh_dir / "decision.json").read_text())
    if decision["selected_research_arm"] != ARMS[1] or not decision["comparisons"][ARMS[1]]["passes"]:
        raise ValueError("fresh face-recovery development gate did not pass")
    fresh_results = {row["example_id"]: row for row in read_rows(fresh_dir / "results.jsonl")}
    prepared = {row["example_id"]: row for row in read_rows(fresh_dir / "prepared.jsonl")}
    if len(fresh_results) != 71 or len(prepared) != 71: raise ValueError("fresh benchmark is incomplete")
    examples = []
    for signer in (1, 2, 3, 5, 8):
        candidates = [row for row in fresh["examples"] if row["signer"] == signer]
        chosen = sorted(candidates, key=lambda row: hashlib.sha256(row["sentence_name"].encode()).hexdigest())[:2]
        for row in chosen:
            key = row["sentence_name"]; receipt = prepared[key]
            clip = ROOT / fresh["data_dir"] / "clips-25fps" / f"{key}.mp4"
            if sha256(clip) != receipt["clip_sha256"]: raise ValueError("fresh runtime clip changed")
            caches = {arm: ROOT / fresh["data_dir"] / "features" / f"{key}--{arm}.npz" for arm in ARMS}
            landmark_path = ROOT / fresh["data_dir"] / "landmarks" / f"{key}.npz"
            examples.append({"example_id": key, "domain": "how2sign", "signer": signer,
                "source_group": row["video_id"], "reference": row["reference"], "clip": str(clip.relative_to(ROOT)),
                "clip_sha256": receipt["clip_sha256"], "expected_hypotheses": {arm: fresh_results[key]["arms"][arm]["hypothesis"] for arm in ARMS},
                "expected_features": {arm: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)} for arm, path in caches.items()},
                "expected_legacy_landmarks": {"path": str(landmark_path.relative_to(ROOT)), "sha256": sha256(landmark_path)}})
    tuning_path = ROOT / "runs/fresh-lowres-v1/tuning-inventory-v2.jsonl"
    tuning = load_tuning(tuning_path, ROOT)
    old_path = ROOT / "runs/c5-resolution-tuning-v1/visual-crop-168-calibrated-v4-cap32.jsonl"
    old = {row["example_id"]: row for row in read_rows(old_path) if row["source"] == "fleurs"}
    for row in tuning:
        if row["source"] != "fleurs": continue
        key = row["example_id"]; clip = row["video_path"]
        if sha256(clip) != row["video_sha256"] or old[key]["calibrator_sha256"] != sha256(BUNDLE / "crop_calibrator.safetensors"):
            raise ValueError("open FLEURS clip or selected-control policy changed")
        landmark_path = ROOT / "data/c5-resolution-tuning-v1/landmarks" / f"fleurs--{key}.npz"
        examples.append({"example_id": key, "domain": "fleurs", "signer": row["signer"], "source_group": row["source_group"],
            "reference": row["reference"], "clip": str(clip.relative_to(ROOT)), "clip_sha256": row["video_sha256"],
            "expected_hypotheses": {ARMS[0]: old[key]["hypothesis"]}, "expected_features": {},
            "expected_legacy_landmarks": {"path": str(landmark_path.relative_to(ROOT)), "sha256": sha256(landmark_path)}})
    if len(examples) != 17 or docker_image_id(DOCKER_SOCKET) != IMAGE: raise ValueError("runtime cohort or CPU image differs")
    assets = ("MODEL_IDENTITY.json", "runtime.py", "umi_video_reader.py", "crop_calibrator.safetensors",
              "models/checkpoint-11625/model.safetensors", "models/dinov2hand.safetensors", "models/dinov2face.safetensors",
              "models/face_landmarker_v2_with_blendshapes.task", "models/hand_landmarker.task", "vendor/demo/crop_face.py")
    asset_hashes = {name: sha256(BUNDLE / name) for name in assets}
    for name, key in (("MODEL_IDENTITY.json", "selected_identity_sha256"), ("crop_calibrator.safetensors", "calibrator_sha256"),
                      ("models/checkpoint-11625/model.safetensors", "selected_checkpoint_sha256")):
        if asset_hashes[name] != fresh[key]: raise ValueError("selected bundle changed since fresh evidence")
    plan = {"schema": "umi-studio-c5-face-runtime-validation-plan/1", "examples": examples, "arms": ARMS,
        "repeats": 2, "cpu_threads": 2, "device": "cpu", "dino_batch_size": 32, "candidate_parallel_legacy": parallel_legacy,
        "generation": {"num_beams": 5, "max_length": 2048}, "image_id": IMAGE, "source_sha256": sources(),
        "bundle_asset_sha256": asset_hashes, "fresh_decision_sha256": sha256(fresh_dir / "decision.json"),
        "tuning_inventory_sha256": sha256(tuning_path), "prior_fleurs_control_sha256": sha256(old_path),
        "data_dir": str((ROOT / "data" / output.name).relative_to(ROOT)), "data_budget_bytes": 1024**3,
        "minimum_free_bytes": 80 * 1024**3, "training_allowed": False, "miner_health_policy": "user_authorized_maintenance",
        "confirmation_policy": "Only ten deterministic fresh development parity clips and seven already-open FLEURS clips; no sealed confirmation, reserve, test, devtest or private data.",
        "gates": {"exact_hypothesis_parity_required": True, "legacy_landmark_parity_required": True,
                  "nonface_inputs_exactly_preserved": True, "maximum_fleurs_mean_wer_regression": 0.,
                  "maximum_fleurs_word_weighted_wer_regression": 0., "empty_hypotheses_allowed": 0,
                  "maximum_geometric_latency_ratio_for_speed_keep": 1.02},
        "runtime_qualification": "Complete warm hybrid CPU request timings include fresh container startup/detection, host decode/crops/DINO/calibration/encoder/beam decode and integrity checks; no cached-landmark substitution. Model initialization is excluded. Native macOS FaceLandmarker initialization failed with NSOpenGLPixelFormat/kGpuService, so native portability and production deadline compliance remain unverified."}
    output.mkdir(parents=True); path = output / "plan.json"
    path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    archive_sources(path, output / "source-snapshot")
    print(json.dumps({"examples": len(examples), "requests": len(examples) * 2 * 2, "plan_sha256": sha256(path)}))


class ObservedEmbedder:
    def __init__(self, base): self.base = base; self.values = []; self.frame_counts = []
    def __getattr__(self, name): return getattr(self.base, name)
    def extract_embeddings_from_frames(self, frames):
        values = self.base.extract_embeddings_from_frames(frames)
        self.values.append(values.copy()); self.frame_counts.append(len(frames))
        return values
    def reset(self): self.values.clear(); self.frame_counts.clear()


def summarize(records: list[dict], plan: dict) -> dict:
    groups = defaultdict(list)
    for row in records: groups[(row["example_id"], row["arm"])].append(row)
    if len(records) != len(plan["examples"]) * len(ARMS) * plan["repeats"]:
        raise ValueError("runtime request coverage is incomplete")
    ratios = []; paired = []; summaries = {}
    for example in plan["examples"]:
        key = example["example_id"]
        current = groups[(key, ARMS[0])]; candidate = groups[(key, ARMS[1])]
        if any(len(items) != plan["repeats"] or len({r["hypothesis"] for r in items}) != 1 for items in (current, candidate)):
            raise ValueError("runtime predictions vary across repeated identical requests")
        if len({r["legacy_fingerprint"] for r in current + candidate}) != 1:
            raise ValueError("recovery altered a legacy landmark graph")
        if len({r["nonface_fingerprint"] for r in current + candidate}) != 1:
            raise ValueError("recovery altered hand or pose model inputs")
        for before, after in zip(sorted(current, key=lambda r: r["repeat"]), sorted(candidate, key=lambda r: r["repeat"])):
            replaced = set(after["recovered_source_indices"])
            if len(before["face_frame_sha256"]) != len(after["face_frame_sha256"]) or any(
                old != new for index, (old, new) in enumerate(zip(before["face_frame_sha256"], after["face_frame_sha256"])) if index * 2 not in replaced
            ):
                raise ValueError("recovery altered an unrecovered face input")
        times = [statistics.median(r["end_to_end_seconds"] for r in items) for items in (current, candidate)]
        ratios.append(times[1] / times[0])
        paired.append({"domain": example["domain"], "example_id": key, "control": current[0], "candidate": candidate[0], "median_latency_ratio": ratios[-1]})
    for domain in ("how2sign", "fleurs"):
        cases = [r for r in paired if r["domain"] == domain]
        summaries[domain] = {side: {"examples": len(cases), "mean_wer": statistics.fmean(r[side]["wer"] for r in cases),
            "word_weighted_wer": sum(r[side]["edits"] for r in cases) / sum(r[side]["reference_words"] for r in cases),
            "empty_hypotheses": sum(not r[side]["hypothesis"].strip() for r in cases)} for side in ("control", "candidate")}
    fleurs = summaries["fleurs"]
    quality = fleurs["candidate"]["mean_wer"] <= fleurs["control"]["mean_wer"] and fleurs["candidate"]["word_weighted_wer"] <= fleurs["control"]["word_weighted_wer"]
    parity = all(r["expected_hypothesis_parity"] and r["expected_legacy_parity"] and r["expected_feature_parity"] for r in records)
    ratio = math.exp(statistics.fmean(math.log(r) for r in ratios))
    return {"schema": "umi-studio-c5-face-runtime-validation-decision/1", "requests": len(records), "summaries": summaries,
        "runtime_parity_passes": parity, "open_cross_domain_passes": quality,
        "geometric_candidate_to_current_latency_ratio": ratio, "speed_keep_passes": ratio <= 1.02,
        "decision": "freeze_untouched_confirmation_after_runtime_budget_review" if parity and quality else "reject_runtime_or_cross_domain_regression",
        "runtime_qualification": plan["runtime_qualification"], "original_baseline_runtime_latency": "Not measured here; retain immutable fresh original-checkpoint quality control separately.",
        "candidate_parallel_legacy": plan.get("candidate_parallel_legacy", False),
        "training_allowed": False, "confirmation_decoded": False, "miner_changed": False, "promoted": False}


def run(plan_path: Path) -> None:
    plan = json.loads(plan_path.read_text())
    if plan["source_sha256"] != sources() or any(sha256(BUNDLE / name) != value for name, value in plan["bundle_asset_sha256"].items()):
        raise ValueError("frozen runtime implementation or model assets changed")
    if docker_image_id(DOCKER_SOCKET) != plan["image_id"]: raise ValueError("frozen CPU image changed")
    data = ROOT / plan["data_dir"]
    if data.exists() or (plan_path.parent / "results.jsonl").exists(): raise ValueError("runtime validation outputs already exist")
    data.mkdir()
    for name, value in {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
                        "HF_HOME": str(BASELINE / "run/hf"), "TORCH_HOME": str(BASELINE / "run/torch"),
                        "MPLCONFIGDIR": str(BASELINE / "run/matplotlib"), "SHUBERT_DINOV2_SOURCE": str(BUNDLE / "vendor/dinov2-source")}.items():
        os.environ[name] = value
    sys.path.insert(0, str(BUNDLE))
    import torch
    from runtime import SHuBERTInferenceRuntime
    from umi_video_reader import VideoReader
    from c5_stride2_runtime import install_post_mediapipe_stride
    from c5_crop_calibrator import install_crop_calibrator
    from resolution_policy import set_visual_crop_size

    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    model = SHuBERTInferenceRuntime(BUNDLE, device="cpu", generation_num_beams=5, generation_max_length=2048,
                                   dino_batch_size=32, model_execution_concurrency=1, verify_assets=True)
    model._video_reader = VideoReader
    set_visual_crop_size(model, 168); install_crop_calibrator(model, BUNDLE / "crop_calibrator.safetensors")
    install_post_mediapipe_stride(model)
    control_extractor = model._face_extractor; install_face_recovery(model); recovery_extractor = model._face_extractor
    hand = ObservedEmbedder(model._hand_model); face = ObservedEmbedder(model._face_model)
    model._hand_model = hand; model._face_model = face
    records = []; environment = {**os.environ, "DOCKER_HOST": DOCKER_SOCKET}
    for repeat in range(plan["repeats"]):
        for position, example in enumerate(plan["examples"]):
            clip = ROOT / example["clip"]
            if sha256(clip) != example["clip_sha256"]: raise ValueError("runtime clip bytes changed")
            arm_order = ARMS if (repeat + position) % 2 == 0 else tuple(reversed(ARMS))
            for arm in arm_order:
                if shutil.disk_usage(ROOT).free < plan["minimum_free_bytes"] + 128 * 1024**2 or sum(p.stat().st_size for p in data.rglob("*") if p.is_file()) > plan["data_budget_bytes"] - 128 * 1024**2:
                    raise RuntimeError("runtime validation storage guard reached")
                directory = data / f"request-{len(records):03d}"
                provider_state = {}
                def provider(frames, face_asset, hand_asset):
                    command = ["docker", "run", "--rm", "--network", "none", "--cpus", "2", "--memory", "3g",
                        "--volume", f"{ROOT}:/workspace:ro", "--volume", f"{data}:/output:rw", plan["image_id"], "python",
                        "/workspace/tools/c5_face_recovery_runtime.py", "--input", f"/workspace/{example['clip']}",
                        "--output-dir", f"/output/{directory.name}", "--face-model", "/workspace/artifacts/c5-corrective-stride2-crop168-confirmed/models/face_landmarker_v2_with_blendshapes.task",
                        "--hand-model", "/workspace/artifacts/c5-corrective-stride2-crop168-confirmed/models/hand_landmarker.task"]
                    if arm == ARMS[1]: command.append("--recovery")
                    if arm == ARMS[1] and plan.get("candidate_parallel_legacy", False):
                        command[command.index("/workspace/tools/c5_face_recovery_runtime.py")] = "/workspace/tools/c5_parallel_face_recovery_runtime.py"
                    result = subprocess.run(command, env=environment, capture_output=True, text=True)
                    (plan_path.parent / f"request-{len(records):03d}.log").write_text(result.stdout + result.stderr)
                    if result.returncode: raise RuntimeError(f"fresh CPU landmark request failed: {result.stderr[-2000:]}")
                    legacy, metadata = unpack_landmarks(directory / "legacy.npz")
                    with np.load(directory / "recovery.npz", allow_pickle=False) as payload:
                        mask = payload["recovery_mask"]; points = payload["recovered_faces"]
                        extra = json.loads(str(payload["metadata"].item()))
                    if metadata["source_sha256"] != example["clip_sha256"] or len(legacy) != len(frames) or extra["runtime_source_sha256"] != plan["source_sha256"]["c5_face_recovery_runtime.py"]:
                        raise ValueError("fresh landmark runtime source or clip identity differs")
                    if arm == ARMS[1] and plan.get("candidate_parallel_legacy", False) and extra.get("parallel_source_sha256") != plan["source_sha256"]["c5_parallel_face_recovery_runtime.py"]:
                        raise ValueError("parallel landmark implementation differs")
                    reference = example["expected_legacy_landmarks"]; old_path = ROOT / reference["path"]
                    if sha256(old_path) != reference["sha256"]: raise ValueError("pinned legacy parity target changed")
                    old, _ = unpack_landmarks(old_path)
                    provider_state.update(legacy_fingerprint=landmark_fingerprint(legacy), expected_legacy_parity=landmark_fingerprint(legacy) == landmark_fingerprint(old),
                                          recovered_frames=int(mask.sum()), recovered_source_indices=np.flatnonzero(mask).tolist(), detector_metadata=extra)
                    return RecoveryLandmarks(legacy, {int(i): points[i] for i in np.flatnonzero(mask)})
                model._video_holistic = provider
                model._face_extractor = recovery_extractor if arm == ARMS[1] else control_extractor
                hand.reset(); face.reset(); started = time.perf_counter()
                hypothesis = model.translate_path(clip); seconds = time.perf_counter() - started
                if len(hand.values) != 2 or len(face.values) != 1: raise ValueError("runtime did not use exactly one DINO pass per stream")
                features = {"left_hand_features": hand.values[0], "right_hand_features": hand.values[1], "face_features": face.values[0]}
                metrics = {}; feature_parity = True
                target = example["expected_features"].get(arm)
                if target:
                    target_path = ROOT / target["path"]
                    if sha256(target_path) != target["sha256"]: raise ValueError("pinned feature parity target changed")
                    with np.load(target_path, allow_pickle=False) as payload:
                        for name, values in features.items():
                            delta = np.abs(values - payload[name]); maximum = float(delta.max())
                            matches = bool(np.allclose(values, payload[name], rtol=1e-5, atol=1e-4))
                            metrics[name] = {"maximum_absolute_difference": maximum, "matches": matches}; feature_parity &= matches
                expected = example["expected_hypotheses"].get(arm)
                edits = distance(normalize(example["reference"]), normalize(hypothesis)); count = len(normalize(example["reference"]))
                digest = hashlib.sha256()
                for name in ("left_hand_features", "right_hand_features"): digest.update(features[name].tobytes())
                pose = np.asarray(model._process_pose_landmarks(RecoveryLandmarks(unpack_landmarks(directory / "legacy.npz")[0], {})), dtype=np.float32)
                digest.update(pose.tobytes())
                record = {"schema": "umi-studio-c5-face-runtime-validation-example/1", "plan_sha256": sha256(plan_path),
                    "example_id": example["example_id"], "domain": example["domain"], "arm": arm, "repeat": repeat,
                    "hypothesis": hypothesis, "reference_words": count, "edits": edits, "wer": edits / count,
                    "expected_hypothesis_parity": expected is None or hypothesis == expected, "expected_feature_parity": feature_parity,
                    "feature_parity_metrics": metrics, "nonface_fingerprint": digest.hexdigest(), "end_to_end_seconds": seconds,
                    "face_frame_sha256": [hashlib.sha256(values.tobytes()).hexdigest() for values in features["face_features"]],
                    "diagnostics": model.last_diagnostics, "dino_calls": {"hand": len(hand.values), "face": len(face.values)}, **provider_state}
                records.append(record)
                with (plan_path.parent / "results.jsonl").open("a") as output:
                    output.write(json.dumps(record, sort_keys=True) + "\n"); output.flush(); os.fsync(output.fileno())
                total_requests = len(plan["examples"]) * len(ARMS) * plan["repeats"]
                print(f"runtime {len(records)}/{total_requests} {example['domain']} {example['example_id']} {arm} seconds={seconds:.2f} parity={record['expected_hypothesis_parity']}", flush=True)
    decision = {**summarize(records, plan), "plan_sha256": sha256(plan_path)}
    (plan_path.parent / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    print(json.dumps(decision, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze", type=Path); group.add_argument("--run", type=Path)
    parser.add_argument("--parallel-legacy", action="store_true", help="freeze the candidate's independent-graph scheduling optimization")
    args = parser.parse_args()
    if args.parallel_legacy and not args.freeze: parser.error("scheduling is chosen only when freezing a new plan")
    if args.freeze: freeze(args.freeze, parallel_legacy=args.parallel_legacy)
    else: run(args.run.resolve())


if __name__ == "__main__": main()
