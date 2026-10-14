"""Freeze winner-initialized full-corpus fine-tuning with separate open controls."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

from benchmark_fleurs_dev import normalize, sha256
from c5_experiment_sources import archive_sources
from c5_full_coverage_mps import GATES, load_training
from c5_winner_reference import ROOT, load_reference
from tiny_rgb_full_corpus import full_schedule, read_rows, validate_split
from tiny_rgb_full_run import runtime_binding

RECIPE = {
    "learning_rate": 1e-5,
    "weight_decay": 0.01,
    "gradient_clip": 1.0,
    "gradient_accumulation": 8,
    "warmup_updates": 128,
    "minimum_learning_rate_ratio": 0.1,
    "retention_coefficient": 1.0,
    "retention_temperature": 2.0,
    "training_exposures": 48800,
    "checkpoint_after_exposures": [512, 2048, 8192, 16384, 32768, 48800],
    "domain_mass": {"how2sign": 0.60, "fleurs": 0.20, "asl_stem": 0.15, "openasl": 0.05},
    "training_input_policy": "native224_legacy_post_mediapipe_stride2",
    "evaluation_input_policy": "matched_calibrated_crop168_legacy_post_mediapipe_stride2",
    "max_text_tokens": 512,
    "augmentation": "none",
    "minimum_free_bytes": 80 * 1024**3,
    "maximum_derived_data_bytes": 4 * 1024**3,
}
PROOF = ROOT / "runs/c5-winner-full-preflight-v7-functional/mps.json"
CORPUS = ROOT / "runs/c5-all-source-corpus-v2"
RAW = ROOT / "runs/tiny-rgb-corpus-v1/training-inventory.jsonl"
EVALUATION = ROOT / "runs/c5-full-coverage-mps-v1/evaluation-features.jsonl"
HISTORICAL = ROOT / "checkpoints/c5-full-coverage-mps-v1/evaluation-results.jsonl"


def sources():
    names = (
        "c5_winner_training.py",
        "train_c5_winner_full.py",
        "launch_c5_winner_full.py",
        "c5_winner_reference.py",
        "c5_winner_mps_safe.py",
        "probe_c5_winner_decoder.py",
        "c5_certification_budget.py",
        "tiny_rgb_mps_safe.py",
        "c5_full_coverage_mps.py",
        "c5_retention.py",
        "train_c5_stem_continuation.py",
        "train_visual_lora.py",
        "visual_training_data.py",
        "cache_fleurs_encoder.py",
        "benchmark_fleurs_dev.py",
        "tiny_rgb_full_corpus.py",
        "tiny_rgb_full_run.py",
        "c5_stationary_schedule.py",
        "audit_c5_all_source_corpus.py",
        "c5_experiment_sources.py",
    )
    return {f"tools/{name}": sha256(ROOT / "tools" / name) for name in names}


def checked_proof():
    proof = json.loads(PROOF.read_text())
    if (
        not proof["numerics_passed"]
        or proof["optimizer_updates"] != 0
        or proof["weights_saved"]
        or proof["training_scope"] != "full"
        or proof["parameter_counts"] != {"total": 670950728, "trainable": 670149192}
        or proof["probe_source_sha256"] != sha256(ROOT / "tools/probe_c5_winner_decoder.py")
        or sha256(PROOF.with_name("activation-source.py"))
        != sha256(ROOT / "tools/c5_winner_mps_safe.py")
        or (
            proof["portable_fairseq_gelu_replacements"],
            proof["portable_native_gelu_replacements"],
            proof["portable_layer_norm_replacements"],
        )
        != (12, 1, 30)
        or len(proof["records"]) != 3
    ):
        raise ValueError("winner full-training numerical qualification differs")
    for row in proof["records"]:
        if (
            not row["passes_cpu_mps_parity"]
            or row["role"] != "training"
            or row["training_allowed"] is not True
            or row["gradient_coverage"]["parameter_tensors"] != 456
            or row["worst_gradient_relative_l2_error"] > 0.005
            or row["loss_absolute_difference"] > 5e-4
            or row["logit_maximum_absolute_difference"] > 1e-3
            or row["native_portable_cpu_logit_difference"] > 1e-4
        ):
            raise ValueError("winner full-training proof no longer passes its original limits")
    return proof


def lr_factor(update, total=6100):
    warmup = RECIPE["warmup_updates"]
    if update < warmup:
        return (update + 1) / warmup
    progress = min(1.0, (update - warmup) / (total - warmup))
    floor = RECIPE["minimum_learning_rate_ratio"]
    return floor + (1 - floor) * (1 + math.cos(math.pi * progress)) / 2


def inventories():
    training = load_training(CORPUS / "audit.json", CORPUS / "training-streams.jsonl")
    raw = read_rows(RAW)
    originals = {(row["source"], row["id"]): row for row in training}
    if len(raw) != 7529 or {(row["source"], row["example_id"]) for row in raw} != set(originals):
        raise ValueError("winner training identities differ from protected eligibility")
    evaluation = read_rows(EVALUATION)
    if len(evaluation) != 142:
        raise ValueError("winner open-control coverage changed")
    validation_rows = [{**row, "source": row["domain"]} for row in evaluation]
    validate_split(raw, validation_rows)
    for row in raw:
        if normalize(row["reference"]) != normalize(
            originals[(row["source"], row["example_id"])]["reference"]
        ):
            raise ValueError("winner source-reference binding changed")
    return training, raw, evaluation


def private_data(value):
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT / "data"):
        raise ValueError("winner input leaves the private data tree")
    return path


def storage(plan, prospective=0, derived_prospective=0):
    if shutil.disk_usage(ROOT).free < RECIPE["minimum_free_bytes"] + 128 * 1024**2 + prospective:
        raise RuntimeError("winner training reached physical disk floor")
    data = ROOT / plan["data_dir"]
    used = (
        sum(path.stat().st_size for path in data.rglob("*") if path.is_file())
        if data.exists()
        else 0
    )
    if used + derived_prospective > RECIPE["maximum_derived_data_bytes"] - 128 * 1024**2:
        raise RuntimeError("winner teacher cache reached its derived-data budget")


def checked_plan(path):
    plan = json.loads(path.read_text())
    if (
        plan["schema"] != "umi-studio-c5-winner-full-training/1"
        or plan["recipe"] != RECIPE
        or plan["gates"] != GATES
        or plan["source_sha256"] != sources()
        or plan["runtime_binding"] != runtime_binding()
    ):
        raise ValueError("winner immutable training implementation or recipe changed")
    if sha256(ROOT / "C5_BASELINE_REFERENCE.json") != plan["reference_registry_sha256"]:
        raise ValueError("winner working reference changed after freezing")
    for name, digest in plan["dependency_sha256"].items():
        if sha256(ROOT / name) != digest:
            raise ValueError("winner frozen data/control/policy dependency changed")
    checked_proof()
    _, raw, evaluation = inventories()
    if (
        read_rows(path.with_name("schedule.jsonl")) != full_schedule(raw)
        or plan["evaluation"] != evaluation
    ):
        raise ValueError("winner exposure order or open controls changed")
    if sha256(path.with_name("schedule.jsonl")) != plan["schedule_sha256"]:
        raise ValueError("winner schedule bytes changed")
    snapshot = json.loads(path.with_name("source-snapshot").joinpath("manifest.json").read_text())
    if snapshot["plan_sha256"] != sha256(path):
        raise ValueError("winner plan changed after source snapshot")
    return plan


def freeze(output, resource_policy):
    output = output.resolve()
    if output.exists() or not output.is_relative_to(ROOT / "runs"):
        raise ValueError("winner training needs a new private execution directory")
    reference, proof = load_reference(), checked_proof()
    training, raw, evaluation = inventories()
    dependencies = {
        str(path.relative_to(ROOT)): sha256(path)
        for path in (
            CORPUS / "audit.json",
            CORPUS / "training-streams.jsonl",
            RAW,
            EVALUATION,
            HISTORICAL,
            PROOF,
            PROOF.with_name("activation-source.py"),
            resource_policy.resolve(),
        )
    }
    for row in evaluation:
        if sha256(private_data(row["cache"])) != row["cache_sha256"]:
            raise ValueError("winner evaluation input cache changed")
    controls = {
        label: [
            row
            for row in read_rows(HISTORICAL)
            if row["model"] == label and row["checkpoint_examples"] == 0
        ]
        for label in ("selected_parent", "original_checkpoint_common_inputs")
    }
    if any(len(rows) != 142 for rows in controls.values()):
        raise ValueError("winner historical comparison coverage changed")
    schedule = full_schedule(raw)
    plan = {
        "schema": "umi-studio-c5-winner-full-training/1",
        "seed": 317,
        "recipe": RECIPE,
        "gates": GATES,
        "reference": reference,
        "reference_registry_sha256": sha256(ROOT / "C5_BASELINE_REFERENCE.json"),
        "numerical_proof_sha256": sha256(PROOF),
        "parameter_counts": proof["parameter_counts"],
        "training_examples": len(training),
        "exposures": len(schedule),
        "planned_updates": len(schedule) // 8,
        "evaluation": evaluation,
        "historical_controls": controls,
        "source_sha256": sources(),
        "runtime_binding": runtime_binding(),
        "dependency_sha256": dependencies,
        "resource_policy": str(resource_policy.resolve().relative_to(ROOT)),
        "data_dir": f"data/{output.name}",
        "miner_mode": "user_authorized_maintenance",
        "miner_modified": False,
        "confirmation_decoded": False,
        "promotion_authorized": False,
        "qualification": (
            "Full eligible-corpus winner-initialized translator experiment. All functional "
            "translator weights train; unused/invariant parameters, separate fingerspelling "
            "weights and router stay fixed. Human-reference CE plus frozen winner logits on "
            "training prefixes only. Stationary unit-weight replay covers all 7529 identities; "
            "minority repeats can overfit. 142 already-open diagnostic clips use matched "
            "calibrated168 stride2 inputs, not native full-rate complete-composite inference "
            "or official C5 scoring. Passing gates authorizes native-input/FS/router validation "
            "and then a new independent confirmation only. Historical controls remain separate; "
            "no automatic promotion or miner changes."
        ),
    }
    storage(plan)
    output.mkdir(parents=True)
    schedule_path = output / "schedule.jsonl"
    schedule_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in schedule))
    plan["schedule_sha256"] = sha256(schedule_path)
    path = output / "plan.json"
    path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    archive_sources(path, output / "source-snapshot")
    checked_plan(path)
    print(
        json.dumps(
            {
                "plan_sha256": sha256(path),
                "training_identities": len(training),
                "exposures": len(schedule),
                "updates": plan["planned_updates"],
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resource-policy", type=Path, required=True)
    args = parser.parse_args()
    freeze(args.output_dir, args.resource_policy)


if __name__ == "__main__":
    main()
