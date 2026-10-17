"""Independently inventory complete word training before fixed-final evaluation.

Run only after the scheduled observation confirms the original session exited0.
This audit does not execute a model, restore an optimizer, or call Terra.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = (
    Path(__file__).resolve().parents[2]
    if Path(__file__).parent.parent.name == "runs"
    else Path(__file__).resolve().parents[1]
)
sys.path.insert(0, str(ROOT / "tools"))

import torch
from c5_word_final import checked_final
from safetensors import safe_open
from terra_training_recovery import artifact


def read(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def optimizer_inventory(path, shapes, updates, recipe):
    expected = {
        f"{name}.{field}" for name in shapes for field in ("step", "exp_avg", "exp_avg_sq")
    } | {"rng.cpu", "rng.mps"}
    with safe_open(str(path), framework="pt", device="cpu") as saved:
        metadata = saved.metadata()
        order = json.loads(metadata["parameter_order"])
        groups = json.loads(metadata["groups_without_parameters"])
        require(set(saved.keys()) == expected, "Adam/RNG tensor inventory changed")
        require(
            len(order) == len(set(order)) == 350 and set(order) == set(shapes),
            "Adam original350-name order is incomplete",
        )
        require(len(groups) == 1, "Adam parameter group count changed")
        group = groups[0]
        require(
            group["lr"] == recipe["learning_rate"]
            and group["weight_decay"] == recipe["weight_decay"]
            and group["betas"] == [0.9, 0.999]
            and group["eps"] == 1e-8
            and group["amsgrad"] is False
            and group["maximize"] is False
            and group["capturable"] is False
            and group["differentiable"] is False
            and group["foreach"] is False
            and group["fused"] in (False, None),
            "Adam hyperparameters changed",
        )
        for name, shape in shapes.items():
            step = saved.get_tensor(f"{name}.step")
            require(
                step.shape == () and step.dtype == torch.float32 and step.item() == updates,
                "Adam tensor step differs from ledger",
            )
            for field in ("exp_avg", "exp_avg_sq"):
                value = saved.get_tensor(f"{name}.{field}")
                require(
                    list(value.shape) == shape
                    and value.dtype == torch.float32
                    and bool(torch.isfinite(value).all()),
                    "invalid Adam moment",
                )
                if field == "exp_avg_sq":
                    require(not bool((value < 0).any()), "negative Adam second moment")
        rng = {}
        for name in ("cpu", "mps"):
            value = saved.get_tensor(f"rng.{name}")
            require(
                value.dtype == torch.uint8 and value.ndim == 1 and value.numel() > 0,
                "RNG state is not a complete byte vector",
            )
            if name == "cpu":
                torch.Generator(device="cpu").set_state(value)
            rng[name] = {
                "bytes": value.numel(),
                "sha256": hashlib.sha256(value.numpy().tobytes()).hexdigest(),
            }
    return {
        "optimizer": artifact(path),
        "parameter_tensors": len(shapes),
        "all_Adam_steps": updates,
        "RNG": rng,
        "groups": groups,
    }


def main():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "result", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    plan_path, result_path, target = (
        path.resolve() for path in (args.plan, args.result, args.output)
    )
    output = result_path.parent
    require(
        not target.exists() and target.is_relative_to(ROOT / "runs"),
        "terminal audit needs a fresh owned receipt",
    )
    require(
        Path(__file__).resolve().is_relative_to(ROOT / "runs"),
        "execute the frozen run's archived terminal auditor",
    )
    plan, result, evidence = checked_final(plan_path, result_path)
    require(
        "resume" in plan and artifact(Path(__file__)) == plan["lineage_audit_source"],
        "word lineage audit differs from its frozen program",
    )
    require(
        not list(output.rglob("failure.json")) and not list(output.rglob("stopped.json")),
        "training failure evidence exists",
    )
    pinned = {**plan["source_sha256"], **plan["model_plan"]["assets"]}
    for name, expected in pinned.items():
        require(
            hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected,
            "pinned source or asset changed: " + name,
        )
    require(
        len(plan["records"]) == 214
        and len(plan["batches"]) == 216
        and plan["planned_training_examples"] == 856
        and plan["recipe"]["epochs"] == 4,
        "complete fixed curriculum changed",
    )
    rows = [case["record"]["original"]["row"] for case in plan["records"]]
    require(
        len({row["example_id"] for row in rows}) == 214
        and len({row["label"] for row in rows}) == 60
        and all(
            row["role"] == "training"
            and row["training_allowed"] is True
            and row["participant"]
            not in {"P10", "P13", "P14", "P21", "P22", "P31", "P4", "P6", "P0", "P19"}
            for row in rows
        ),
        "original training boundary changed",
    )
    for index, donor in enumerate(plan["donor_indices"]):
        first = next(
            (index + offset) % 214
            for offset in range(1, 214)
            if rows[(index + offset) % 214]["label"] != rows[index]["label"]
        )
        require(donor == first, "fixed cyclic different-label donor changed")
    steps = [json.loads(line) for line in (output / "steps.jsonl").read_text().splitlines()]
    require(len(steps) == 216, "complete216-batch ledger missing")
    epochs, exposures, updates, requests, max_pairs = [], 0, 0, 0, 0
    calibration_counts = Counter()
    for epoch in range(4):
        expected = list(range(214))
        random.Random(plan["recipe"]["seed"] + epoch).shuffle(expected)
        actual = []
        initial_updates = updates
        for offset in range(0, 214, 4):
            number = epoch * 54 + offset // 4
            step, batch = steps[number], plan["batches"][number]
            indices = expected[offset : offset + 4]
            require(
                batch == {"epoch": epoch, "training_record_indices": indices}
                and step["batch"] == number
                and step["epoch"] == epoch
                and step["training_record_indices"] == indices,
                "independent deterministic order mismatch",
            )
            actual.extend(indices)
            paired = read(output / f"batch-{number:04d}/blinded-inputs.json")
            rewards = read(output / f"batch-{number:04d}/rewards.json")
            quality = [
                [rewards["scores"][index] for index in case["indices"]] for case in paired["cases"]
            ]
            require(
                quality == rewards["qualities"] == step["Terra_qualities"],
                "actual reward-to-step mapping changed",
            )
            contrast = sum(len(set(values)) > 1 for values in quality)
            updates += bool(contrast)
            exposures += len(indices)
            require(
                step["training_examples"] == exposures
                and step["optimizer_updates"] == updates
                and step["contrasting_clips"] == contrast,
                "actual exposure/update count changed",
            )
            for control in paired["controls"]:
                score = rewards["scores"][control["index"]]
                require(
                    control["range"][0] <= score <= control["range"][1],
                    "actual original calibration failed",
                )
                calibration_counts[str(score)] += 1
            dirs = sorted((output / f"batch-{number:04d}").glob("request-*"))
            requests += len(dirs)
            for directory in dirs:
                started = read(directory / "started.json")
                max_pairs = max(max_pairs, started["pairs"])
                require(
                    1 <= started["pairs"] <= 16 and started["automatic_retries"] == 0,
                    "actual request bound or no-retry policy changed",
                )
        require(
            actual == expected and Counter(actual) == Counter(range(214)),
            "epoch omitted or repeated an original clip",
        )
        epochs.append(
            {
                "epoch": epoch,
                "unique_clips": len(set(actual)),
                "exposures": len(actual),
                "optimizer_updates": updates - initial_updates,
                "tail_clips": len(steps[epoch * 54 + 53]["training_record_indices"]),
            }
        )
    require(
        exposures == result["training_examples"] == 856
        and updates == result["optimizer_updates"]
        and 1 <= updates <= 216,
        "fixed final counter differs from all actual batches",
    )
    scope = read(ROOT / plan["model_plan"]["visual_scope"]["path"])
    shapes = {"vision." + row["parameter"]: row["shape"] for row in scope["parameters"]}
    require(
        len(shapes) == 350
        and set(shapes) == set(plan["model_plan"]["expected_visual_gradient_names"]),
        "original visual scope changed",
    )
    checkpoints = []
    for descriptor in result["checkpoints"]:
        visual = ROOT / descriptor["path"]
        commit = read(visual.with_suffix(".json"))
        optimizer = visual.with_name(visual.stem + "-optimizer.safetensors")
        checkpoints.append(
            {
                "visual": descriptor,
                "commit": artifact(visual.with_suffix(".json")),
                "training_examples": commit["training_examples"],
                **optimizer_inventory(
                    optimizer, shapes, commit["optimizer_updates"], plan["recipe"]
                ),
            }
        )
    audit = {
        "schema": "umi-studio-complete-word-training-terminal-audit/1",
        "complete": True,
        "observed_at": datetime.now(timezone.utc).isoformat(),  # noqa: UP017 (Python3.10 runtime)
        "audit_source": artifact(Path(__file__)),
        "plan": artifact(plan_path),
        "result": artifact(output / "result.json"),
        "same_worker_exit0": True,
        "inherited_source_exit": plan["resume"]["source_exit"],
        "updates_performed_by_current_worker": updates - plan["resume"]["optimizer_updates"],
        "inherited_last_completed_batch_resources": json.loads(
            (ROOT / plan["resume"]["source_steps"]["path"]).read_text().splitlines()[-1]
        )["resources"],
        "worker": artifact(output / "worker.json"),
        "exit": artifact(output / "exit.json"),
        "all_original_350_tensor_checkpoints_and_actual_reward_evidence_validated": True,
        "training_examples": exposures,
        "optimizer_updates": updates,
        "epochs": epochs,
        "checkpoints": checkpoints,
        "authenticated_Terra_requests": requests,
        "maximum_pairs_per_actual_request": max_pairs,
        "calibration_score_counts": dict(calibration_counts),
        "raw_request_response_pairings_revalidated_by_checked_final": True,
        "training_evidence": evidence,
        "pinned_source_count": len(plan["source_sha256"]),
        "pinned_model_asset_count": len(plan["model_plan"]["assets"]),
        "all_pinned_sources_and_model_assets_match": True,
        "visual_initialization_sha256": result["visual_initialization_sha256"],
        "visual_final_sha256": result["visual_final_sha256"],
        "resources": result["resources"],
        "held_out_semantic_quality_evaluated": False,
        "new_Terra_requests_during_audit": 0,
        "optimizer_updates_during_audit": 0,
        "promotion_authorized": False,
    }
    with target.open("x") as stream:
        json.dump(audit, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(
        json.dumps(
            {
                "audit": artifact(target),
                "training_examples": exposures,
                "optimizer_updates": updates,
                "epochs": epochs,
                "authenticated_Terra_requests": requests,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
