"""Freeze and verify the executable full-corpus RGB experiment without changing its gates."""

from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
import sys
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

from benchmark_fleurs_dev import normalize, sha256
from c5_experiment_sources import archive_sources
from c5_full_coverage_mps import GATES, load_training
from tiny_rgb_fit import ROOT, SIDES
from tiny_rgb_full_corpus import FULL_RECIPE, full_schedule, read_rows, validate_split

MEMORY_POLICY = {
    "temporal_and_target_buckets": True,
    "preserve_spatial_aspect_and_dimensions": True,
    "clear_unused_mps_allocations": "before_and_after_each_backward_and_after_each_evaluation_clip",
    "initial_untrained_generation": False,
    "real_frames_or_targets_dropped": False,
}


def sources() -> dict[str, str]:
    names = (
        "tiny_rgb_full_run.py",
        "train_tiny_rgb_full.py",
        "launch_tiny_rgb_full.py",
        "tiny_rgb_pixel_cache.py",
        "tiny_rgb_full_training_tools.py",
        "audit_tiny_rgb_full.py",
        "tiny_rgb_full_corpus.py",
        "tiny_rgb_fit.py",
        "tiny_rgb_model.py",
        "tiny_rgb_language_model.py",
        "tiny_rgb_mps_safe.py",
        "tiny_rgb_data.py",
        "train_tiny_rgb_fit.py",
        "benchmark_fleurs_dev.py",
        "c5_gpu_budget.py",
        "c5_full_coverage_mps.py",
        "audit_c5_all_source_corpus.py",
        "c5_all_source_training.py",
        "c5_stationary_schedule.py",
        "c5_coverage_schedule.py",
        "c5_face_learning_pilot.py",
        "fresh_lowres_model.py",
        "train_fresh_lowres.py",
        "c5_experiment_sources.py",
    )
    return {f"tools/{name}": sha256(ROOT / "tools" / name) for name in names}


def runtime_binding() -> dict:
    packages = {}
    paths = {}
    for name in (
        "torch",
        "numpy",
        "transformers",
        "safetensors",
        "opencv-python",
        "opencv-contrib-python",
    ):
        try:
            package = distribution(name)
        except PackageNotFoundError:
            continue
        packages[name] = package.version
        for relative in (
            "torch/mps/__init__.py",
            "transformers/models/t5/modeling_t5.py",
            "transformers/generation/utils.py",
            "safetensors/torch.py",
            "cv2/__init__.py",
        ):
            path = Path(package.locate_file(relative))
            if path.is_file():
                paths[str(path.resolve())] = sha256(path)
    if not {"torch", "numpy", "transformers", "safetensors"}.issubset(packages):
        raise ValueError("full RGB training runtime is incomplete")
    return {
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": sys.version,
        "package_versions": packages,
        "critical_runtime_source_sha256": paths,
    }


def lr_factor(update: int, total: int) -> float:
    warmup = FULL_RECIPE["warmup_updates"]
    if total <= warmup:
        raise ValueError("full RGB horizon must exceed warmup")
    if update < warmup:
        return (update + 1) / warmup
    progress = min(1.0, (update - warmup) / (total - warmup))
    floor = FULL_RECIPE["minimum_learning_rate_ratio"]
    return floor + (1 - floor) * (1 + math.cos(math.pi * progress)) / 2


def training_inventory(plan: dict) -> list[dict]:
    corpus = ROOT / "runs/c5-all-source-corpus-v2"
    approved = load_training(corpus / "audit.json", corpus / "training-streams.jsonl")
    rows = read_rows(ROOT / plan["training_inventory"])
    original = {(r["source"], r["id"]): r for r in approved}
    if {(r["source"], r["example_id"]) for r in rows} != set(original) or len(rows) != 7529:
        raise ValueError("full RGB training inventory differs from protected eligibility")
    for row in rows:
        if normalize(row["reference"]) != normalize(
            original[(row["source"], row["example_id"])]["reference"]
        ):
            raise ValueError("full RGB source/reference binding changed")
    validate_split(rows, plan["evaluation"])
    return rows


def checked_full_plan(path: Path) -> dict:
    plan = json.loads(path.read_text())
    if (
        plan["schema"] != "umi-studio-tiny-rgb-full-run/1"
        or plan["recipe"] != FULL_RECIPE
        or plan["gates"] != GATES
        or plan["short_sides"] != list(SIDES)
        or plan["source_sha256"] != sources()
        or plan["initial_diagnostic_weights_reused"]
        or plan["pretrained_visual_weights"]
        or plan["learned_feature_inputs"]
        or plan["c5_teacher_targets"]
        or plan["runtime_binding"] != runtime_binding()
        or plan.get("memory_policy") != MEMORY_POLICY
    ):
        raise ValueError("full RGB immutable implementation or scientific plan changed")
    snapshot = json.loads(path.with_name("source-snapshot").joinpath("manifest.json").read_text())
    if snapshot["plan_sha256"] != sha256(path):
        raise ValueError("full RGB plan changed after source snapshot")
    for name, digest in {**plan["dependency_sha256"], **plan["language_asset_sha256"]}.items():
        if sha256(ROOT / name) != digest:
            raise ValueError("full RGB frozen source/data/control dependency changed")
    rows = training_inventory(plan)
    schedule = path.with_name("schedule.jsonl")
    if sha256(schedule) != plan["schedule_sha256"]:
        raise ValueError("full RGB exposure order changed")
    frozen = read_rows(schedule)
    expected = full_schedule(rows)
    if frozen != expected:
        raise ValueError("full RGB repeated exposures do not reproduce the approved corpus")
    return plan


def freeze(proposal_path: Path, audit_dir: Path, budget_path: Path, output: Path) -> None:
    output = output.resolve()
    if output.exists() or not output.is_relative_to(ROOT / "runs"):
        raise ValueError("full RGB execution plan needs a new private runs/ directory")
    proposal = json.loads(proposal_path.read_text())
    if (
        proposal["recipe"] != FULL_RECIPE
        or proposal["gates"] != GATES
        or not proposal["planning_only"]
    ):
        raise ValueError("full RGB proposal changed")
    header_path, receipt_path = audit_dir / "headers.jsonl", audit_dir / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    if (
        receipt["proposal_sha256"] != sha256(proposal_path)
        or receipt["headers_sha256"] != sha256(header_path)
        or receipt["training_headers"] != 7529
        or receipt["evaluation_headers"] != 142
        or not receipt["clip_bytes_verified"]
        or not receipt["complete_targets"]
    ):
        raise ValueError("full RGB header/target audit does not bind the entire experiment")
    plan = copy.deepcopy(proposal)
    plan.update(
        {
            "schema": "umi-studio-tiny-rgb-full-run/1",
            "planning_only": False,
            "data_dir": f"data/{output.name}",
            "source_sha256": sources(),
            "runtime_binding": runtime_binding(),
            "headers": str(header_path.resolve().relative_to(ROOT)),
            "headers_sha256": sha256(header_path),
            "audit": receipt,
            "gpu_budget_policy": str(budget_path.resolve().relative_to(ROOT)),
            "miner_mode": "user_authorized_maintenance",
            "miner_health_claim": False,
            "memory_policy": dict(MEMORY_POLICY),
            "cache_policy": {
                "codec": "uint8_modulo_temporal_delta",
                "training_minimum_repeats": 12,
                "cache_evaluation": True,
                "maximum_private_bytes": 4 * 1024**3,
                "eviction": "owned_derived_pixel_files_only_LRU_originals_preserved",
            },
        }
    )
    control_plan = json.loads((ROOT / "runs/c5-full-coverage-mps-v1/plan.json").read_text())
    for asset in control_plan["assets"].values():
        plan["dependency_sha256"][asset["path"]] = asset["sha256"]
    for path in (proposal_path, header_path, receipt_path, budget_path):
        plan["dependency_sha256"][str(path.resolve().relative_to(ROOT))] = sha256(path)
    for name, digest in {**plan["dependency_sha256"], **plan["language_asset_sha256"]}.items():
        if sha256(ROOT / name) != digest:
            raise ValueError("full RGB freezing found changed data/control/language bytes")
    if shutil.disk_usage(ROOT).free < FULL_RECIPE["minimum_free_bytes"] + 128 * 1024**2:
        raise RuntimeError("full RGB freezing reached physical disk floor")
    training_inventory(plan)
    plan["qualification"] = (
        "Executable full-corpus experiment with all7529 eligible training identities and142 "
        "separate already-open evaluation clips. Both resolutions start with identical fresh "
        "visual/bridge weights and pinned generic T5 language, not diagnostic checkpoints. "
        "Unit-weight stationary replay; minority repeats can overfit. Complete12.5fps RGB and "
        "complete language targets, no learned visual inputs or C5 teacher. Cache evictions "
        "remove only owned derived pixels and preserve original videos. Frozen original/selected "
        "controls use earlier verified predictions on the exact same clip bytes with their "
        "declared common-input policy; not untouched full-rate inference or official scores. "
        "Preflight must pass before updates. Passing open gates authorizes only new independent "
        "confirmation, never automatic promotion or miner modification."
    )
    output.mkdir(parents=True)
    shutil.copyfile(proposal_path.with_name("schedule.jsonl"), output / "schedule.jsonl")
    path = output / "plan.json"
    path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    archive_sources(path, output / "source-snapshot")
    checked_full_plan(path)
    print(
        json.dumps(
            {
                "plan_sha256": sha256(path),
                "training_examples": 7529,
                "evaluation_examples": 142,
                "exposures_per_arm": 48800,
            }
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--gpu-budget-policy", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    freeze(args.proposal, args.audit_dir, args.gpu_budget_policy, args.output_dir)


if __name__ == "__main__":
    main()
