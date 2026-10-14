"""Freeze training-only visual-capacity numerical qualification with unchanged guards."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PLAN = ROOT / "runs/tiny-rgb-full-v3-buckets/plan.json"
BUDGET = ROOT / "runs/c5-resource-policy-v2-certification48/gpu-budget.json"
DURATION_AUDIT = ROOT / "runs/tiny-rgb-duration-integrity-v1/receipt.json"
CAPACITIES = ("small", "medium", "large")
SIDES = (128, 192)
LIMITS = {
    "maximum_relative_gradient_error": 5e-3,
    "maximum_total_gradient_norm_relative_error": 5e-3,
    "maximum_loss_difference": 5e-4,
    "maximum_logit_difference": 1e-3,
    "maximum_native_portable_cpu_logit_difference": 1e-4,
}
STAGES = {
    "cpu_equivalence": "native versus capacity activation at same unpadded shapes; 0.0001 logits",
    "cpu_padding": "capacity unpadded versus masked inputs; existing full-RGB 0.001 padding logits",
    "cpu_mps_comparison": "masked CPU versus masked Metal; 0.001 logits",
}


def select_cases(rows, headers):
    if any(
        row.get("role") != "training" or row.get("training_allowed") is not True for row in rows
    ):
        raise ValueError("capacity qualification accepts training identities only")
    inventory = {(row["source"], row["example_id"]): row for row in rows}
    training = {
        (row["source"], row["example_id"]): row for row in headers if row["role"] == "training"
    }
    if len(inventory) != len(rows) or set(inventory) != set(training):
        raise ValueError("capacity qualification headers do not match approved training")
    keys = sorted(training)
    if not keys or any(training[key]["sampled_frames"] < 1 for key in keys):
        raise ValueError("capacity qualification requires complete positive frame sequences")
    chosen = [
        min(keys, key=lambda key: training[key]["sampled_frames"]),
        max(keys, key=lambda key: training[key]["sampled_frames"]),
        max(keys, key=lambda key: training[key]["target_tokens"]),
    ]
    fleurs = [key for key in keys if key[0].startswith("fleurs")]
    if not fleurs:
        raise ValueError("capacity qualification lacks an eligible FLEURS training clip")
    chosen.append(fleurs[0])
    for prefix in ("asl_stem", "openasl"):
        family = [key for key in keys if key[0].startswith(prefix)]
        if not family:
            raise ValueError(f"capacity qualification lacks eligible {prefix} training")
        chosen.append(family[0])
    return [{"training": inventory[key], "header": training[key]} for key in sorted(set(chosen))]


def verified_comparison(comparison, parameter_tensors, *, cpu_equivalence=False):
    gradients = comparison["gradient_differences"]
    numbers = (
        comparison["loss_absolute_difference"],
        comparison["logit_maximum_absolute_difference"],
        comparison["total_gradient_norm_relative_error"],
    )
    logit_limit = LIMITS[
        "maximum_native_portable_cpu_logit_difference"
        if cpu_equivalence
        else "maximum_logit_difference"
    ]
    if (
        len(gradients) != parameter_tensors
        or len({row["parameter"] for row in gradients}) != parameter_tensors
        or any(not math.isfinite(value) or value < 0 for value in numbers)
        or numbers[0] > LIMITS["maximum_loss_difference"]
        or numbers[1] > logit_limit
        or numbers[2] > LIMITS["maximum_total_gradient_norm_relative_error"]
    ):
        raise ValueError("capacity loss/logit/gradient coverage parity failed")
    for gradient in gradients:
        error = gradient["relative_l2_error"]
        if (
            not gradient["finite"]
            or not math.isfinite(error)
            or not 0 <= error <= LIMITS["maximum_relative_gradient_error"]
        ):
            raise ValueError("capacity per-parameter gradient parity failed")
    return max(row["relative_l2_error"] for row in gradients)


def sources():
    from benchmark_fleurs_dev import sha256
    from tiny_rgb_full_run import sources as original_sources

    result = original_sources()
    for name in (
        "tiny_rgb_capacity.py",
        "tiny_rgb_capacity_activations.py",
        "tiny_rgb_capacity_positions.py",
        "tiny_rgb_capacity_protocol.py",
        "probe_tiny_rgb_capacity.py",
        "c5_certification_budget.py",
        "audit_tiny_rgb_duration.py",
    ):
        result[f"tools/{name}"] = sha256(ROOT / "tools" / name)
    return result


def inputs():
    from audit_tiny_rgb_duration import checked_duration_audit
    from tiny_rgb_full_corpus import read_rows
    from tiny_rgb_full_run import checked_full_plan

    parent = checked_full_plan(REFERENCE_PLAN)
    _, rows = checked_duration_audit(DURATION_AUDIT)
    allowed = {(row["source"], row["example_id"]) for row in rows}
    headers = [
        row
        for row in read_rows(ROOT / parent["headers"])
        if row["role"] != "training" or (row["source"], row["example_id"]) in allowed
    ]
    return parent, select_cases(rows, headers)


def checked_plan(path):
    from benchmark_fleurs_dev import sha256
    from tiny_rgb_capacity import visual_budget
    from tiny_rgb_full_run import runtime_binding

    plan = json.loads(path.read_text())
    if (
        plan["schema"] != "umi-studio-tiny-rgb-capacity-qualification/2"
        or plan["limits"] != LIMITS
        or plan["comparison_stages"] != STAGES
        or plan["capacities"] != list(CAPACITIES)
        or plan["short_sides"] != list(SIDES)
        or plan["source_sha256"] != sources()
        or plan["runtime_binding"] != runtime_binding()
        or plan["reference_plan_sha256"] != sha256(REFERENCE_PLAN)
        or plan["resource_policy_sha256"] != sha256(BUDGET)
        or plan["duration_audit_sha256"] != sha256(DURATION_AUDIT)
        or plan["visual_budgets"] != [visual_budget(name) for name in CAPACITIES]
    ):
        raise ValueError("capacity immutable qualification settings changed")
    for name, digest in plan["dependency_sha256"].items():
        if sha256(ROOT / name) != digest:
            raise ValueError("capacity qualification dependency changed")
    parent, cases = inputs()
    if cases != plan["cases"] or parent["language_path"] != plan["language_path"]:
        raise ValueError("capacity qualification no longer binds approved training cases")
    snapshot = json.loads(path.with_name("source-snapshot").joinpath("manifest.json").read_text())
    if snapshot["plan_sha256"] != sha256(path):
        raise ValueError("capacity plan differs from its source snapshot")
    return plan


def freeze(path):
    from benchmark_fleurs_dev import sha256
    from c5_certification_budget import CertificationComputeWindow
    from c5_experiment_sources import archive_sources
    from tiny_rgb_capacity import visual_budget
    from tiny_rgb_full_run import runtime_binding

    path = path.resolve()
    if path.parent.exists() or not path.is_relative_to(ROOT / "runs"):
        raise ValueError("capacity qualification needs a new private runs/ directory")
    if shutil.disk_usage(ROOT).free < 80 * 1024**3 + 128 * 1024**2:
        raise RuntimeError("capacity qualification reached physical disk floor")
    CertificationComputeWindow(BUDGET)
    parent, cases = inputs()
    dependencies = dict(parent["language_asset_sha256"])
    for relative in (
        str(REFERENCE_PLAN.relative_to(ROOT)),
        str(BUDGET.relative_to(ROOT)),
        str(DURATION_AUDIT.relative_to(ROOT)),
        "data/how2sign/source/how2sign_realigned_train.csv",
        parent["training_inventory"],
        parent["headers"],
        "C5_BASELINE_REFERENCE.json",
    ):
        dependencies[relative] = sha256(ROOT / relative)
    for case in cases:
        row = case["training"]
        dependencies[row["video"]] = row["video_sha256"]
    for relative, digest in dependencies.items():
        if sha256(ROOT / relative) != digest:
            raise ValueError("capacity freezing found changed source/language bytes")
    plan = {
        "schema": "umi-studio-tiny-rgb-capacity-qualification/2",
        "capacities": list(CAPACITIES),
        "short_sides": list(SIDES),
        "seed": 317,
        "limits": dict(LIMITS),
        "comparison_stages": dict(STAGES),
        "cases": cases,
        "visual_budgets": [visual_budget(name) for name in CAPACITIES],
        "source_sha256": sources(),
        "runtime_binding": runtime_binding(),
        "dependency_sha256": dependencies,
        "reference_plan_sha256": sha256(REFERENCE_PLAN),
        "resource_policy": str(BUDGET.relative_to(ROOT)),
        "resource_policy_sha256": sha256(BUDGET),
        "duration_audit_sha256": sha256(DURATION_AUDIT),
        "duration_audit": str(DURATION_AUDIT.relative_to(ROOT)),
        "eligible_training_clips": 7523,
        "quarantined_truncated_clips": 6,
        "language_path": parent["language_path"],
        "target_fps": 12.5,
        "max_frames": 512,
        "max_text_tokens": 256,
        "optimizer_updates": 0,
        "weights_saved": False,
        "training_only": True,
        "miner_modified": False,
        "qualification": "Real pinned language assets and approved training-only RGB extrema; "
        "separate same-shape native/activation, masked-padding and CPU/Metal all-parameter checks. "
        "No ASL fitting, held-out quality, native composite or promotion claim.",
    }
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    archive_sources(path, path.with_name("source-snapshot"))
    checked_plan(path)
    print(json.dumps({"plan_sha256": sha256(path), "training_cases": len(cases)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    freeze(parser.parse_args().freeze)
