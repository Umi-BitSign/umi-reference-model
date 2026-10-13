"""Freeze stationary training-only replay, teacher bindings and unchanged open gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path

import numpy as np
from benchmark_fleurs_dev import sha256
from c5_experiment_sources import archive_sources
from c5_face_learning_pilot import ROOT, read_rows
from c5_full_coverage_mps import GATES, data_path
from c5_full_coverage_mps import RECIPE as PARENT_RECIPE
from c5_full_coverage_mps import checked_plan as checked_parent
from c5_full_coverage_mps import sources as parent_sources
from c5_stationary_schedule import mixture

PARENT = ROOT / "runs/c5-full-coverage-mps-v1/plan.json"
COMPLETED = ROOT / "checkpoints/c5-full-coverage-mps-v1"
PROPOSAL = ROOT / "runs/c5-stationary-mixture-proposal-v1/report.json"
RECIPE = {
    **PARENT_RECIPE,
    "schedule_policy": "cyclic_balanced_domains_and_how_signers_unit_weights/1",
    "unique_training_examples": 7529,
    "training_exposures": 48800,
    "checkpoint_after_examples": [512, 1024, 2048, 4096, 8192, 16384, 32768, 48800],
    "teacher_validation": "all_unique_training_prefixes_before_any_optimizer_update",
    "teacher_maximum_absolute_difference": 1e-5,
    "teacher_maximum_initial_kl": 1e-5,
}
EXPOSURE_FIELDS = {
    "loss_weight",
    "original_corpus_loss_weight",
    "exposure",
    "group_exposure",
    "repeat_cycle",
}


def sources() -> dict:
    return {
        **parent_sources(),
        **{
            name: sha256(ROOT / "tools" / name)
            for name in (
                "c5_stationary_schedule.py",
                "c5_stationary_mps.py",
                "train_c5_stationary_mps.py",
                "launch_c5_stationary_mps.py",
                "c5_experiment_sources.py",
                "benchmark_fleurs_dev.py",
            )
        },
    }


def unique_training(schedule: list[dict], original: list[dict]) -> dict:
    unique = {(r["source"], r["id"]): r for r in original}
    if len(unique) != len(original):
        raise ValueError("unique teacher corpus repeats a clip")
    canonical = {
        key: {k: v for k, v in row.items() if k != "loss_weight"} for key, row in unique.items()
    }
    seen = set()
    for index, row in enumerate(schedule, 1):
        key = row["source"], row["id"]
        if (
            row.get("role") != "training"
            or row.get("training_allowed") is not True
            or row.get("loss_weight") != 1.0
            or row.get("exposure") != index
            or key not in unique
            or {k: v for k, v in row.items() if k not in EXPOSURE_FIELDS} != canonical[key]
            or row.get("original_corpus_loss_weight") != unique[key]["loss_weight"]
        ):
            raise ValueError("stationary exposure differs from its protected training identity")
        seen.add(key)
    if seen != set(unique):
        raise ValueError("stationary schedule omits a unique training clip")
    return unique


def validate_teacher_binding(record: dict, row: dict, parent_sha256: str) -> None:
    expected = {
        "source": row["source"],
        "id": row["id"],
        "cache": row["cache"],
        "cache_sha256": row["cache_sha256"],
        "reference_sha256": hashlib.sha256(row["reference"].encode()).hexdigest(),
        "parent_checkpoint_sha256": parent_sha256,
        "input_policy": "verified_native224_legacy_post_mediapipe_stride2",
        "role": "training",
        "training_allowed": True,
    }
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError("reused teacher identity, source, reference or parent differs")


def read_teacher(record: dict) -> np.ndarray:
    target = data_path(record["teacher_path"])
    if sha256(target) != record["teacher_sha256"]:
        raise ValueError("frozen teacher tensor bytes changed")
    with np.load(target, allow_pickle=False) as cache:
        if cache.files != ["logits"]:
            raise ValueError("teacher tensor has undeclared arrays")
        values = cache["logits"]
    if (
        list(values.shape) != record["shape"]
        or values.dtype != np.float32
        or not np.isfinite(values).all()
    ):
        raise ValueError("teacher tensor shape, dtype or finiteness differs")
    return values


def freeze(output: Path) -> None:
    parent = checked_parent(PARENT)
    completed = json.loads((COMPLETED / "decision.json").read_text())
    run = json.loads((COMPLETED / "run.json").read_text())
    proposal = json.loads(PROPOSAL.read_text())
    parent_sha = parent["assets"]["selected_model"]["sha256"]
    if (
        completed["plan_sha256"] != sha256(PARENT)
        or completed["decision"] != "reject_full_coverage_no_open_generalization_gain"
        or completed["promoted"]
        or completed["confirmation_decoded"]
        or run["plan_sha256"] != sha256(PARENT)
        or run["parent_checkpoint_sha256"] != parent_sha
        or proposal["parent_plan_sha256"] != sha256(PARENT)
        or proposal["completed_decision_sha256"] != sha256(COMPLETED / "decision.json")
        or proposal["source_sha256"]
        != {
            **parent_sources(),
            "c5_stationary_schedule.py": sha256(ROOT / "tools/c5_stationary_schedule.py"),
        }
        or not proposal["planning_only"]
        or proposal["training_launched"]
    ):
        raise ValueError("completed parent experiment or stationary proposal changed")
    schedule_path = PROPOSAL.with_name("proposed-schedule.jsonl")
    if sha256(schedule_path) != proposal["proposed_schedule_sha256"]:
        raise ValueError("stationary proposal schedule changed")
    schedule = read_rows(schedule_path)
    original = read_rows(PARENT.with_name("schedule.jsonl"))
    unique = unique_training(schedule, original)
    if (
        len(schedule) != RECIPE["training_exposures"]
        or len(unique) != RECIPE["unique_training_examples"]
    ):
        raise ValueError("stationary exposure/unique count differs")
    evaluation_path = PARENT.with_name("evaluation-features.jsonl")
    evaluation = read_rows(evaluation_path)
    expected = {r["example_id"]: r for r in parent["evaluation"]}
    if len(evaluation) != 142 or len({r["example_id"] for r in evaluation}) != 142:
        raise ValueError("parent evaluation preparation is incomplete")
    for row in evaluation:
        if any(row.get(k) != value for k, value in expected[row["example_id"]].items()):
            raise ValueError("prepared parent evaluation identity changed")
        if sha256(data_path(row["cache"])) != row["cache_sha256"]:
            raise ValueError("prepared parent evaluation tensor changed")
    output = output.resolve()
    if output.exists() or not output.is_relative_to(ROOT / "runs"):
        raise ValueError("stationary plan requires a new private runs/ directory")
    output.mkdir(parents=True)
    shutil.copyfile(schedule_path, output / "schedule.jsonl")
    teachers = output / "teachers.jsonl"
    teacher_dir = ROOT / "data/c5-full-coverage-mps-v1/teacher-logits"
    with teachers.open("x") as stream:
        for index, row in enumerate(original):
            target = teacher_dir / f"{index:04d}.npz"
            with np.load(target, allow_pickle=False) as cache:
                if cache.files != ["logits"]:
                    raise ValueError("parent teacher arrays differ")
                values = cache["logits"]
            if (
                values.ndim != 3
                or values.shape[0] != 1
                or values.dtype != np.float32
                or not np.isfinite(values).all()
            ):
                raise ValueError("parent teacher shape/dtype/finiteness failed")
            record = {
                "source": row["source"],
                "id": row["id"],
                "cache": row["cache"],
                "cache_sha256": row["cache_sha256"],
                "reference_sha256": hashlib.sha256(row["reference"].encode()).hexdigest(),
                "parent_checkpoint_sha256": parent_sha,
                "input_policy": parent["training_input_policy"],
                "role": "training",
                "training_allowed": True,
                "teacher_path": str(target.relative_to(ROOT)),
                "teacher_sha256": sha256(target),
                "shape": list(values.shape),
            }
            validate_teacher_binding(record, row, parent_sha)
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    dependencies = {
        str(PARENT.relative_to(ROOT)): sha256(PARENT),
        str(PARENT.with_name("schedule.jsonl").relative_to(ROOT)): sha256(
            PARENT.with_name("schedule.jsonl")
        ),
        str((COMPLETED / "run.json").relative_to(ROOT)): sha256(COMPLETED / "run.json"),
        str((COMPLETED / "decision.json").relative_to(ROOT)): sha256(COMPLETED / "decision.json"),
        str((COMPLETED / "evaluation-results.jsonl").relative_to(ROOT)): sha256(
            COMPLETED / "evaluation-results.jsonl"
        ),
        str(PROPOSAL.relative_to(ROOT)): sha256(PROPOSAL),
        str(evaluation_path.relative_to(ROOT)): sha256(evaluation_path),
    }
    plan = {
        "schema": "umi-studio-c5-stationary-mps-plan/1",
        "seed": parent["seed"],
        "recipe": RECIPE,
        "gates": GATES,
        "source_sha256": sources(),
        "parent_plan": str(PARENT.relative_to(ROOT)),
        "dependencies": dependencies,
        "schedule_sha256": sha256(output / "schedule.jsonl"),
        "teachers_sha256": sha256(teachers),
        "assets": parent["assets"],
        "training_input_policy": parent["training_input_policy"],
        "evaluation_input_policy": parent["evaluation_input_policy"],
        "evaluation_features": str(evaluation_path.relative_to(ROOT)),
        "prior_control_predictions": str(
            (COMPLETED / "evaluation-results.jsonl").relative_to(ROOT)
        ),
        "training_exposures": len(schedule),
        "unique_training_examples": len(unique),
        "optimizer_updates": math.ceil(len(schedule) / 4),
        "gpu_budget_policy": parent["gpu_budget_policy"],
        "gpu_budget_policy_sha256": parent["gpu_budget_policy_sha256"],
        "minimum_free_bytes": parent["minimum_free_bytes"],
        "data_budget_bytes": parent["data_budget_bytes"],
        "data_dir": f"data/{output.name}",
        "miner_health_policy": "user_authorized_maintenance",
        "first512": mixture(schedule[:512]),
        "last512": mixture(schedule[-512:]),
        "teacher_provenance": (
            "The completed parent run did not persist per-file teacher hashes. Bytes are newly "
            "bound here; every tensor must reproduce the current fixed parent's training-prefix "
            "logits before any optimizer update. Teacher inputs are training-only, "
            "never hypotheses "
            "or labels from evaluation. No teacher tensor is copied or overwritten."
        ),
        "qualification": (
            parent["qualification"] + " Stationary replay uses unit exposure weights, repeated "
            "minority clips and a longer 48800-exposure schedule. The sampling, "
            "loss normalization, "
            "within-group order and learning-rate horizon change together; this is not an isolated "
            "equal-update comparison. Minority clips repeat up to88 times and may overfit. "
            "All original quality gates and the independent future confirmation requirement remain."
        ),
    }
    path = output / "plan.json"
    path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    archive_sources(path, output / "source-snapshot")
    print(
        json.dumps(
            {
                "plan_sha256": sha256(path),
                "training_exposures": len(schedule),
                "unique_training_examples": len(unique),
                "optimizer_updates": plan["optimizer_updates"],
            }
        )
    )


def checked_plan(path: Path) -> tuple[dict, list[dict], dict, dict]:
    plan = json.loads(path.read_text())
    if (
        plan["schema"] != "umi-studio-c5-stationary-mps-plan/1"
        or plan["recipe"] != RECIPE
        or plan["gates"] != GATES
        or plan["source_sha256"] != sources()
    ):
        raise ValueError("stationary implementation or scientific recipe changed")
    snapshot = json.loads(path.with_name("source-snapshot").joinpath("manifest.json").read_text())
    if snapshot["plan_sha256"] != sha256(path):
        raise ValueError("stationary plan changed after freezing")
    if any(sha256(ROOT / name) != digest for name, digest in plan["dependencies"].items()):
        raise ValueError("stationary parent dependencies changed")
    parent = checked_parent(ROOT / plan["parent_plan"])
    if (
        plan["assets"] != parent["assets"]
        or plan["gpu_budget_policy_sha256"] != parent["gpu_budget_policy_sha256"]
    ):
        raise ValueError("stationary model identity or GPU policy changed")
    if (
        sha256(path.with_name("schedule.jsonl")) != plan["schedule_sha256"]
        or sha256(path.with_name("teachers.jsonl")) != plan["teachers_sha256"]
    ):
        raise ValueError("stationary exposures or teacher bindings changed")
    rows = read_rows(path.with_name("schedule.jsonl"))
    original = read_rows((ROOT / plan["parent_plan"]).with_name("schedule.jsonl"))
    unique = unique_training(rows, original)
    if len(rows) != plan["training_exposures"] or len(unique) != plan["unique_training_examples"]:
        raise ValueError("stationary coverage counts differ")
    targets = read_rows(path.with_name("teachers.jsonl"))
    teachers = {(r["source"], r["id"]): r for r in targets}
    if len(teachers) != len(targets) or set(teachers) != set(unique):
        raise ValueError("stationary teacher coverage differs")
    for key, target in teachers.items():
        validate_teacher_binding(target, unique[key], plan["assets"]["selected_model"]["sha256"])
    return plan, rows, unique, teachers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    args = parser.parse_args()
    freeze(args.freeze)


if __name__ == "__main__":
    main()
