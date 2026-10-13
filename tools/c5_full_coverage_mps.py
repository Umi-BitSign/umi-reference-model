"""Freeze audited full-corpus MPS retention training and open evaluation inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import statistics
from pathlib import Path

import numpy as np
from audit_c5_all_source_corpus import PROTECTED, exclusion_reason, protected_sets
from benchmark_fleurs_dev import normalize, sha256
from c5_all_source_training import objective_weights
from c5_coverage_schedule import coverage_schedule, prefix_counts
from c5_experiment_sources import archive_sources
from c5_face_learning_pilot import BASELINE, BUNDLE, ROOT, read_rows

RECIPE = {
    "rank": 16,
    "alpha": 32.0,
    "visual_last_layers": 4,
    "visual_learning_rate": 1e-5,
    "decoder_learning_rate": 3e-5,
    "weight_decay": 0.01,
    "label_smoothing": 0.1,
    "gradient_accumulation": 4,
    "passes": 1,
    "warmup_updates": 100,
    "minimum_learning_rate_ratio": 0.1,
    "retention_coefficient": 1.0,
    "retention_temperature": 2.0,
    "domain_mass": {"how2sign": 0.6, "fleurs": 0.2, "asl_stem": 0.15, "openasl": 0.05},
    "checkpoint_after_examples": [512, 1024, 2048, 4096, 7529],
    "cpu_threads": 2,
    "maximum_process_mps_bytes": 6 * 1024**3,
    "generation_num_beams": 5,
    "generation_max_length": 2048,
}
GATES = {
    "minimum_how_relative_mean_improvement": 0.05,
    "minimum_improved_how_signers": 3,
    "maximum_how_signer_delta": 0.05,
    "maximum_word_weighted_delta": 0.0,
    "maximum_fleurs_mean_delta": 0.0,
    "maximum_how_source_bootstrap_upper": 0.0,
    "maximum_original_mean_delta": 0.0,
    "maximum_original_word_weighted_delta": 0.0,
    "bootstrap_samples": 5000,
    "bootstrap_seed": 1729,
}


def sources() -> dict:
    names = (
        "c5_full_coverage_mps.py",
        "train_c5_full_coverage_mps.py",
        "launch_c5_full_coverage_mps.py",
        "c5_gpu_budget.py",
        "audit_c5_all_source_corpus.py",
        "c5_all_source_training.py",
        "c5_coverage_schedule.py",
        "c5_retention.py",
        "train_c5_face_learning_pilot.py",
        "c5_face_learning_pilot.py",
        "train_c5_stem_continuation.py",
        "train_visual_lora.py",
        "visual_training_data.py",
        "cache_fleurs_encoder.py",
        "shubert_lora.py",
        "shubert_visual_lora.py",
        "train_c5_stem_projection_frozen.py",
        "interpolate_visual_adapter.py",
        "extract_mediapipe_landmarks.py",
        "c5_stride2_runtime.py",
        "c5_fresh_face_benchmark.py",
    )
    return {name: sha256(ROOT / "tools" / name) for name in names}


def data_path(value: str) -> Path:
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT / "data"):
        raise ValueError("training/evaluation cache leaves private data/")
    return path


def load_training(audit_path: Path, manifest: Path) -> list[dict]:
    audit = json.loads(audit_path.read_text())
    if audit["training_manifest_sha256"] != sha256(manifest):
        raise ValueError("audited training manifest changed")
    if audit["protected_source_hashes"] != {name: sha256(path) for name, path in PROTECTED.items()}:
        raise ValueError("protected inventories changed after audit")
    rows = read_rows(manifest)
    if len(rows) != audit["eligible_training_examples"] or len(rows) != 7529:
        raise ValueError("new full-corpus count differs")
    if len({(row["source"], row["id"]) for row in rows}) != len(rows):
        raise ValueError("training corpus repeats an example")
    groups, references = protected_sets()
    if any(exclusion_reason(row, groups, references) for row in rows):
        raise ValueError("protected material entered training")
    weights = objective_weights(rows)
    for row, weight in zip(rows, weights, strict=True):
        data_path(row["cache"])
        if row.get("sidecar"):
            data_path(row["sidecar"])
        row.update(loss_weight=weight, role="training", training_allowed=True)
    return rows


def ordered_training(rows: list[dict], seed: int) -> list[dict]:
    shuffled = sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"coverage-mps-{seed}:{row['source']}:{row['id']}".encode()
        ).hexdigest(),
    )
    return coverage_schedule(shuffled, RECIPE["domain_mass"])


def learning_rate_factor(step: int, total: int) -> float:
    warmup = RECIPE["warmup_updates"]
    if total <= warmup:
        raise ValueError("full-corpus schedule must exceed warmup")
    if step < warmup:
        return (step + 1) / warmup
    progress = min(1.0, (step - warmup) / (total - warmup))
    floor = RECIPE["minimum_learning_rate_ratio"]
    return floor + (1 - floor) * (1 + math.cos(math.pi * progress)) / 2


def freeze(output: Path, budget_path: Path) -> None:
    output = output.resolve()
    if output.exists() or not output.is_relative_to(ROOT / "runs"):
        raise ValueError("full-corpus run needs a new private runs/ directory")
    audit_path = ROOT / "runs/c5-all-source-corpus-v2/audit.json"
    manifest = audit_path.with_name("training-streams.jsonl")
    rows = ordered_training(load_training(audit_path, manifest), 287)
    fresh_path = ROOT / "runs/c5-fresh-face-benchmark-v1/plan.json"
    fresh = json.loads(fresh_path.read_text())
    pilot_path = ROOT / "runs/c5-face-learning-pilot-v1/plan.json"
    pilot = json.loads(pilot_path.read_text())
    pilot_features = {r["example_id"]: r for r in read_rows(pilot_path.with_name("features.jsonl"))}
    runtime_path = ROOT / "runs/c5-face-runtime-validation-parallel-v1/plan.json"
    runtime = json.loads(runtime_path.read_text())
    evaluation = []
    for row in fresh["examples"]:
        path = data_path(
            f"{fresh['data_dir']}/features/{row['sentence_name']}--confirmed_crop168.npz"
        )
        evaluation.append(
            {
                "example_id": row["sentence_name"],
                "domain": "how2sign",
                "source_group": "how2sign:" + row["video_id"],
                "signer": row["signer"],
                "reference": row["reference"],
                "cache": str(path.relative_to(ROOT)),
                "cache_sha256": sha256(path),
                "role": "open_development",
                "training_allowed": False,
            }
        )
    for row in pilot["examples"]:
        if row["role"] != "pilot_source_holdout":
            continue
        target = pilot_features[row["example_id"]]["features"]["legacy_faces"]
        path = data_path(target["path"])
        if sha256(path) != target["sha256"]:
            raise ValueError("pilot holdout cache changed")
        evaluation.append(
            {
                "example_id": row["example_id"],
                "domain": "how2sign",
                "source_group": row["source_group"],
                "signer": row["signer"],
                "reference": row["reference"],
                "cache": target["path"],
                "cache_sha256": target["sha256"],
                "role": "open_development",
                "training_allowed": False,
            }
        )
    for row in runtime["examples"]:
        if row["domain"] != "fleurs":
            continue
        if sha256(ROOT / row["clip"]) != row["clip_sha256"]:
            raise ValueError("open FLEURS source changed")
        landmark = row["expected_legacy_landmarks"]
        if sha256(ROOT / landmark["path"]) != landmark["sha256"]:
            raise ValueError("open FLEURS landmarks changed")
        evaluation.append(
            {
                "example_id": row["example_id"],
                "domain": "fleurs",
                "source_group": "fleurs:" + row["example_id"].split("_sentence_")[0],
                "signer": row["signer"],
                "reference": row["reference"],
                "clip": row["clip"],
                "clip_sha256": row["clip_sha256"],
                "landmarks": landmark,
                "expected_parent_cpu_hypothesis": row["expected_hypotheses"]["confirmed_crop168"],
                "role": "open_development",
                "training_allowed": False,
            }
        )
    if len(evaluation) != 142 or len({r["example_id"] for r in evaluation}) != 142:
        raise ValueError("open evaluation coverage changed")
    held_references = {tuple(normalize(row["reference"])) for row in evaluation}
    if any(tuple(normalize(row["reference"])) in held_references for row in rows):
        raise ValueError("training repeats an open evaluation reference")
    assets = {
        "selected_model": BUNDLE / "models/checkpoint-11625/model.safetensors",
        "original_model": BASELINE / "models/checkpoint-11625/model.safetensors",
        "selected_identity": BUNDLE / "MODEL_IDENTITY.json",
        "calibrator": BUNDLE / "crop_calibrator.safetensors",
        "dino_hand": BUNDLE / "models/dinov2hand.safetensors",
        "dino_face": BUNDLE / "models/dinov2face.safetensors",
    }
    if (
        sha256(assets["selected_model"])
        != "419315ded8438bb59fe1b1a7e3c774012e098067db21335621789b56e9f98210"
        or sha256(assets["original_model"])
        != "92c6c77ed7f2343f346c2a0cc7c025d733d47fcb1adc55150939f934d5f3b3b9"
    ):
        raise ValueError("selected or original checkpoint identity changed")
    for name, digest in pilot["asset_sha256"].items():
        target = BUNDLE / name
        if sha256(target) != digest:
            raise ValueError("selected vendor/configuration asset changed")
        assets["bundle:" + name] = target
    output.mkdir(parents=True)
    schedule_path = output / "schedule.jsonl"
    with schedule_path.open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    plan = {
        "schema": "umi-studio-c5-full-coverage-mps-plan/1",
        "seed": 287,
        "recipe": RECIPE,
        "gates": GATES,
        "training_examples": len(rows),
        "optimizer_updates": math.ceil(len(rows) / 4),
        "training_input_policy": "verified_native224_legacy_post_mediapipe_stride2",
        "evaluation_input_policy": "selected_calibrated_crop168_legacy_post_mediapipe_stride2",
        "corpus_audit": str(audit_path.relative_to(ROOT)),
        "corpus_audit_sha256": sha256(audit_path),
        "training_manifest": str(manifest.relative_to(ROOT)),
        "training_manifest_sha256": sha256(manifest),
        "schedule_sha256": sha256(schedule_path),
        "evaluation": evaluation,
        "source_sha256": sources(),
        "assets": {
            k: {"path": str(p.relative_to(ROOT)), "sha256": sha256(p)} for k, p in assets.items()
        },
        "evaluation_inventory_hashes": {
            str(p.relative_to(ROOT)): sha256(p)
            for p in (fresh_path, pilot_path, runtime_path, pilot_path.with_name("features.jsonl"))
        },
        "gpu_budget_policy": str(budget_path.resolve().relative_to(ROOT)),
        "gpu_budget_policy_sha256": sha256(budget_path),
        "data_dir": f"data/{output.name}",
        "data_budget_bytes": 4 * 1024**3,
        "minimum_free_bytes": 80 * 1024**3,
        "miner_health_policy": "user_authorized_maintenance",
        "prefix_coverage": {
            str(n): prefix_counts(rows, n) for n in RECIPE["checkpoint_after_examples"]
        },
        "qualification": (
            "Open development only. Current training excludes all declared development groups "
            "and exact references; older models/upstream pretraining may have seen pilot groups. "
            "The original control shares crop168 calibration and stride2; it is not untouched "
            "full-rate inference. This new corpus/merged-parent/zero-delta/MPS run is not an "
            "isolated order-only comparison against the old CPU retention experiment. "
            "Prefix clip counts, not prefix weighted objective masses, are balanced. "
            "Confirmation/reserve/test/devtest/private data are forbidden; no automatic promotion."
        ),
    }
    path = output / "plan.json"
    path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    archive_sources(path, output / "source-snapshot")
    print(
        json.dumps(
            {
                "plan_sha256": sha256(path),
                "training_examples": len(rows),
                "evaluation_examples": len(evaluation),
                "prefix512": plan["prefix_coverage"]["512"],
            }
        )
    )


def checked_plan(path: Path) -> dict:
    plan = json.loads(path.read_text())
    if (
        plan["schema"] != "umi-studio-c5-full-coverage-mps-plan/1"
        or plan["recipe"] != RECIPE
        or plan["gates"] != GATES
        or plan["source_sha256"] != sources()
    ):
        raise ValueError("frozen full-coverage implementation or recipe changed")
    snapshot = json.loads((path.parent / "source-snapshot/manifest.json").read_text())
    if snapshot["plan_sha256"] != sha256(path):
        raise ValueError("full-coverage plan changed after freezing")
    dependencies = {
        plan["corpus_audit"]: plan["corpus_audit_sha256"],
        plan["training_manifest"]: plan["training_manifest_sha256"],
        plan["gpu_budget_policy"]: plan["gpu_budget_policy_sha256"],
        **plan["evaluation_inventory_hashes"],
    }
    dependencies.update({row["path"]: row["sha256"] for row in plan["assets"].values()})
    if any(sha256(ROOT / name) != digest for name, digest in dependencies.items()):
        raise ValueError("frozen full-coverage dependency changed")
    load_training(ROOT / plan["corpus_audit"], ROOT / plan["training_manifest"])
    if sha256(path.with_name("schedule.jsonl")) != plan["schedule_sha256"]:
        raise ValueError("frozen full-coverage order changed")
    return plan


def storage(plan: dict) -> None:
    if shutil.disk_usage(ROOT).free < plan["minimum_free_bytes"] + 128 * 1024**2:
        raise RuntimeError("full-coverage disk floor reached")
    data = ROOT / plan["data_dir"]
    if (
        sum(p.stat().st_size for p in data.rglob("*") if p.is_file())
        > plan["data_budget_bytes"] - 128 * 1024**2
    ):
        raise RuntimeError("full-coverage private data budget reached")


def prepare_evaluation(path: Path) -> None:
    plan = checked_plan(path)
    from train_c5_face_learning_pilot import environment

    environment()
    import cv2
    import torch
    from body_features import process_pose_landmarks
    from c5_fresh_face_benchmark import calibrate
    from c5_stride2_runtime import temporal_subsample_landmarks
    from crop_face import FaceExtractor
    from crop_hands import HandExtractor
    from dinov2_features import DINOEmbedder
    from extract_mediapipe_landmarks import unpack_landmarks
    from safetensors.numpy import load_file
    from torchvision import transforms

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    destination = ROOT / plan["data_dir"] / "evaluation"
    if destination.exists() or path.with_name("evaluation-features.jsonl").exists():
        raise ValueError("evaluation preparation already exists")
    destination.mkdir(parents=True)
    hand = DINOEmbedder(str(BUNDLE / "models/dinov2hand.safetensors"), batch_size=32, device="cpu")
    face = DINOEmbedder(str(BUNDLE / "models/dinov2face.safetensors"), batch_size=32, device="cpu")
    transform = transforms.Compose(
        [
            transforms.Resize((168, 168)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    hand.transform = transform
    face.transform = transform
    hand_cropper = HandExtractor()
    face_cropper = FaceExtractor()
    hand_cropper.output_size = (168, 168)
    face_cropper.output_size = (168, 168)
    calibrator = load_file(str(BUNDLE / "crop_calibrator.safetensors"))
    with path.with_name("evaluation-features.jsonl").open("x") as output:
        for row in plan["evaluation"]:
            storage(plan)
            if "cache" in row:
                record = row
            else:
                if (
                    sha256(data_path(row["clip"])) != row["clip_sha256"]
                    or sha256(data_path(row["landmarks"]["path"])) != row["landmarks"]["sha256"]
                ):
                    raise ValueError("FLEURS source changed after plan freezing")
                capture = cv2.VideoCapture(str(ROOT / row["clip"]))
                frames = []
                try:
                    while True:
                        valid, image = capture.read()
                        if not valid:
                            break
                        frames.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                finally:
                    capture.release()
                landmarks, metadata = unpack_landmarks(ROOT / row["landmarks"]["path"])
                if (
                    len(frames) != metadata["frames"]
                    or metadata["fps"] != 25.0
                    or metadata["source_sha256"] != row["clip_sha256"]
                ):
                    raise ValueError("FLEURS frame/landmark identity differs")
                frames = np.ascontiguousarray(np.asarray(frames)[::2])
                landmarks = temporal_subsample_landmarks(landmarks, 2)
                left, right = hand_cropper.extract_hand_frames(frames, landmarks)
                faces = face_cropper.extract_face_frames(frames, landmarks)
                values = {
                    "left_hand_features": calibrate(
                        hand.extract_embeddings_from_frames([f[..., ::-1].copy() for f in left]),
                        calibrator,
                        "hand",
                    ),
                    "right_hand_features": calibrate(
                        hand.extract_embeddings_from_frames([f[..., ::-1].copy() for f in right]),
                        calibrator,
                        "hand",
                    ),
                    "face_features": calibrate(
                        face.extract_embeddings_from_frames([f[..., ::-1].copy() for f in faces]),
                        calibrator,
                        "face",
                    ),
                    "pose_features": np.asarray(
                        process_pose_landmarks(landmarks), dtype=np.float32
                    ),
                }
                target = destination / f"{row['example_id']}.npz"
                with target.open("xb") as stream:
                    np.savez_compressed(stream, **values)
                record = {
                    **row,
                    "cache": str(target.relative_to(ROOT)),
                    "cache_sha256": sha256(target),
                }
            output.write(json.dumps(record, sort_keys=True) + "\n")
            output.flush()
            os.fsync(output.fileno())
            if row["domain"] == "fleurs":
                print(f"prepared open FLEURS {row['example_id']}", flush=True)


def comparison(parent: list[dict], candidate: list[dict]) -> dict:
    before = {r["example_id"]: r for r in parent}
    after = {r["example_id"]: r for r in candidate}
    if set(before) != set(after) or len(before) != 142 or len(after) != 142:
        raise ValueError("full-coverage evaluation coverage differs")
    output = {}
    actor_delta = {}
    by_source = {}
    for domain in ("how2sign", "fleurs"):
        old = [r for r in parent if r["domain"] == domain]
        new = [r for r in candidate if r["domain"] == domain]
        output[domain] = {
            "parent_mean_wer": statistics.fmean(r["wer"] for r in old),
            "mean_wer": statistics.fmean(r["wer"] for r in new),
            "parent_word_weighted_wer": sum(r["edits"] for r in old)
            / sum(r["reference_words"] for r in old),
            "word_weighted_wer": sum(r["edits"] for r in new)
            / sum(r["reference_words"] for r in new),
        }
    for key, row in before.items():
        if row["domain"] != "how2sign":
            continue
        delta = after[key]["wer"] - row["wer"]
        actor_delta.setdefault(str(row["signer"]), []).append(delta)
        by_source.setdefault(row["source_group"], []).append(delta)
    deltas = {key: statistics.fmean(values) for key, values in actor_delta.items()}
    sums = np.array([sum(values) for values in by_source.values()])
    counts = np.array([len(values) for values in by_source.values()])
    sample = np.random.default_rng(GATES["bootstrap_seed"]).integers(
        0, len(sums), size=(GATES["bootstrap_samples"], len(sums))
    )
    interval = np.quantile(
        sums[sample].sum(axis=1) / counts[sample].sum(axis=1), [0.025, 0.975]
    ).tolist()
    how = output["how2sign"]
    fleurs = output["fleurs"]
    relative = 1 - how["mean_wer"] / how["parent_mean_wer"] if how["parent_mean_wer"] else 0.0
    passed = (
        relative >= 0.05
        and sum(d < 0 for d in deltas.values()) >= 3
        and max(deltas.values()) <= 0.05
        and interval[1] <= 0
        and fleurs["mean_wer"] <= fleurs["parent_mean_wer"]
        and all(v["word_weighted_wer"] <= v["parent_word_weighted_wer"] for v in output.values())
        and all(r["hypothesis"].strip() for r in candidate)
    )
    return {
        "domains": output,
        "how_relative_improvement": relative,
        "how_signer_mean_deltas": deltas,
        "how_source_bootstrap_95_interval": interval,
        "passes_open_development_gate": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze", type=Path)
    group.add_argument("--prepare-evaluation", type=Path)
    parser.add_argument("--gpu-budget-policy", type=Path)
    args = parser.parse_args()
    if args.freeze:
        if args.gpu_budget_policy is None:
            parser.error("freeze requires the explicit GPU budget policy")
        freeze(args.freeze, args.gpu_budget_policy)
    else:
        prepare_evaluation(args.prepare_evaluation)


if __name__ == "__main__":
    main()
