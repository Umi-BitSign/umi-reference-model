"""Frozen, training-origin-only feasibility test for fresh64px ROI vision."""

from __future__ import annotations

import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
import torchvision.models.resnet as resnet_module
from benchmark_fleurs_dev import sha256
from launch_terra_visual_job import write
from prepare_roi_appearance import POLICY, ROOT
from prepare_roi_appearance import checked as checked_data_plan
from probe_tiny_rgb_capacity import state_digest
from roi_appearance_model import SEED, STREAMS, RoiAppearance
from terra_training_recovery import artifact, verify_snapshot
from tiny_rgb_full_run import runtime_binding
from tiny_rgb_visual_teacher_data import load_targets

DATA_PLAN = ROOT / "runs/roi-appearance-data-v1/plan.json"
DATA = ROOT / "checkpoints/roi-appearance-data-v1/data.json"
RECIPE = {
    "epochs": 64,
    "batch_frames": 8,
    "learning_rate": 3e-4,
    "weight_decay": 0.01,
    "clip_gradient_norm": 1.0,
    "seed": SEED,
    "loss": "mean three appearance cosine losses; missing zero targets excluded; detached targets",
    "frame_order": "complete training frame permutation each epoch using seed+epoch",
    "augmentation": "none in this feasibility pilot",
    "evaluation_epochs": [0, 1, 4, 16, 64],
    "checkpoint_selection": "fixed final epoch, never WER or auxiliary diagnostic selection",
}
GATE = {
    "minimum_training_loss_fractional_reduction": 0.5,
    "minimum_training_hand_temporal_gap_fraction_of_teacher": 0.1,
    "positive_training_teacher_temporal_gap_required": True,
    "translation_quality_claim_authorized": False,
    "automatic_promotion_authorized": False,
    "later_translation_primary_metric": "Terra medium quality0-10; WER diagnostic only",
}


def checked_data():
    plan = checked_data_plan(DATA_PLAN)
    data = json.loads(DATA.read_text())
    terminal = json.loads(DATA.with_name("exit.json").read_text())
    worker = json.loads(DATA.with_name("worker.json").read_text())
    if (
        not data.get("complete")
        or data.get("plan_sha256") != sha256(DATA_PLAN)
        or data.get("all_streamed_pixels_equal_native_reference") is not True
        or terminal.get("exit_code") != 0
        or terminal.get("pid") != worker.get("pid")
        or len(data["records"]) != 8
        or data.get("primary_translation_development_used") is not False
        or data.get("translation_promotion_authorized") is not False
    ):
        raise ValueError("ROI data is not a complete bound native-parity artifact")
    total = 0
    arrays = []
    for expected, row in zip(plan["records"], data["records"], strict=True):
        if any(row.get(k) != v for k, v in expected.items()):
            raise ValueError("ROI cache changed an original role or source binding")
        cache = ROOT / row["cache"]["path"]
        if not cache.resolve().is_relative_to(DATA.parent) or artifact(cache) != row["cache"]:
            raise ValueError("ROI cache path/hash changed")
        pair = row["original_row"]["original_pair"]
        count = pair["header"]["sampled_frames"]
        with np.load(cache, allow_pickle=False) as archive:
            values = {k: archive[k].copy() for k in archive.files}
        if (
            set(values) != {"pixels", "frame_indices", *STREAMS, "pose_features"}
            or values["pixels"].shape != (count, 3, 64, 64, 3)
            or values["pixels"].dtype != np.uint8
            or values["frame_indices"].dtype != np.int64
            or not np.array_equal(
                values["frame_indices"], np.arange(0, pair["header"]["source_frames"], 2)
            )
            or row["frames"] != count
            or row["streamed_pixels_equal_native_reference"] is not True
        ):
            raise ValueError("ROI cache geometry, frames or array topology changed")
        for name, target in load_targets(pair).items():
            if not np.array_equal(values[name], target):
                raise ValueError("ROI target is not the original complete native stream")
        total += sum(v.nbytes for v in values.values())
        arrays.append(values)
    if total != data["cache_bytes"] or total > POLICY["maximum_ROI_cache_bytes"]:
        raise ValueError("ROI cache RAM accounting differs")
    if artifact(ROOT / data["preview"]["path"]) != data["preview"]:
        raise ValueError("ROI preview changed")
    return plan, data, arrays


def permutation(frames, epoch):
    if frames < 1 or not 0 <= epoch < RECIPE["epochs"]:
        raise ValueError("ROI epoch/frame count outside fixed recipe")
    return np.random.default_rng(SEED + epoch).permutation(frames)


def decision(initial, final):
    before, after = initial["training"], final["training"]
    if before["loss"] <= 0 or after["teacher_temporal_gap"] <= 0:
        raise ValueError("ROI feasibility has no positive reference loss/teacher motion control")
    reduction = 1 - after["loss"] / before["loss"]
    gap_fraction = after["temporal_gap"] / after["teacher_temporal_gap"]
    return {
        "training_loss_fractional_reduction": reduction,
        "training_hand_temporal_gap_fraction_of_teacher": gap_fraction,
        "feature_feasibility_passed": reduction
        >= GATE["minimum_training_loss_fractional_reduction"]
        and gap_fraction >= GATE["minimum_training_hand_temporal_gap_fraction_of_teacher"],
        "translation_quality_evaluated": False,
        "promotion_authorized": False,
    }


def specification():
    data_plan, data, arrays = checked_data()
    model = RoiAppearance()
    train_frames = sum(
        len(a["pixels"])
        for a, r in zip(arrays, data["records"], strict=True)
        if r["roi_training_allowed"]
    )
    source = dict(data_plan["source_sha256"])
    for name in (
        "roi_appearance_model.py",
        "roi_appearance_protocol.py",
        "train_roi_appearance.py",
        "tiny_rgb_model.py",
        "tiny_rgb_mps_safe.py",
        "probe_tiny_rgb_capacity.py",
        "tiny_rgb_capacity_protocol.py",
        "terra_training_storage.py",
        "terra_training_recovery.py",
        "c5_certification_budget.py",
        "train_terra_c5_tail.py",
        "tiny_rgb_full_run.py",
        "launch_terra_visual_job.py",
        "train_tiny_rgb_feature_winner_full.py",
    ):
        source[f"tools/{name}"] = sha256(ROOT / "tools" / name)
    return {
        "schema": "umi-studio-roi-appearance-feasibility/1",
        "data_plan": artifact(DATA_PLAN),
        "data": artifact(DATA),
        "recipe": RECIPE,
        "feature_gate": GATE,
        "source_sha256": source,
        "runtime_binding": runtime_binding(),
        "torchvision_version": version("torchvision"),
        "resnet_source_path": str(Path(resnet_module.__file__).resolve()),
        "resnet_source_sha256": sha256(Path(resnet_module.__file__)),
        "C5_reference_sha256": sha256(ROOT / "C5_BASELINE_REFERENCE.json"),
        "C5_selected": True,
        "initialization_sha256": state_digest(model.state_dict()),
        "visual_parameters": sum(p.numel() for p in model.parameters()),
        "parameter_tensors": len(list(model.parameters())),
        "training_frames_per_epoch": train_frames,
        "planned_frame_exposures": train_frames * RECIPE["epochs"],
        "planned_optimizer_updates": (
            (train_frames + RECIPE["batch_frames"] - 1) // RECIPE["batch_frames"]
        )
        * RECIPE["epochs"],
        "resource_policy": data_plan["resource_policy"],
        "resource_policy_sha256": data_plan["resource_policy_sha256"],
        "proof_training_record_indices": [0, 1],
        "proof_frames": (
            "first and final sampled frame of each complete training clip; "
            "independent framewise forward/backward"
        ),
        "ROI_diagnostic_participants": data_plan["ROI_diagnostic_participants"],
        "reserved_STEM_participants": data_plan["reserved_STEM_participants"],
        "new_downloads": 0,
        "new_Terra_requests": 0,
        "pretrained_visual_weights_loaded": False,
        "language_or_sequence_weights_loaded": False,
        "translation_promotion_authorized": False,
        "feature_diagnostic_is_not_translation_development": True,
        "observed_crop_preview_review": (
            "native hand placement and RGB accepted; absent hands remain native "
            "black/fallback crops, zero targets masked"
        ),
    }


def checked(path):
    plan = json.loads(path.read_text())
    if plan != specification():
        raise ValueError("ROI feasibility recipe/data/sources/initialization/runtime changed")
    verify_snapshot(path, plan)
    if sha256(ROOT / plan["resource_policy"]) != plan["resource_policy_sha256"]:
        raise ValueError("ROI resource policy changed")
    return plan


def freeze(path):
    from c5_experiment_sources import archive_sources
    from terra_training_storage import storage

    path = path.resolve()
    if path.parent.exists() or not path.is_relative_to(ROOT / "runs"):
        raise ValueError("ROI feasibility needs a fresh owned plan")
    torch.set_num_threads(2)
    plan = specification()
    storage(plan["visual_parameters"] * 12 + 1024**2)
    path.parent.mkdir(parents=True)
    write(path, plan)
    archive_sources(path, path.with_name("source-snapshot"))
    print(
        json.dumps(
            {
                "plan": artifact(path),
                "parameters": plan["visual_parameters"],
                "updates": plan["planned_optimizer_updates"],
            }
        ),
        flush=True,
    )
