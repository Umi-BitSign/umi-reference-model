"""Recompute the completed matched CPU learning trial; no new model predictions."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def audit(plan_path, output):
    from benchmark_fleurs_dev import distance, normalize, sha256
    from tiny_rgb_capacity_protocol import verified_comparison
    from train_tiny_rgb_capacity_cpu import RECIPE, ROOT, checked_plan
    from train_tiny_rgb_fit import fit_summary

    plan = checked_plan(plan_path)
    output = output.resolve()
    if not output.is_relative_to(ROOT / "checkpoints"):
        raise ValueError("capacity audit stays within private checkpoint outputs")

    def lines(name):
        return [json.loads(line) for line in output.joinpath(name).read_text().splitlines()]

    run = json.loads(output.joinpath("run.json").read_text())
    decision = json.loads(output.joinpath("decision.json").read_text())
    checks = json.loads(output.joinpath("fit-decisions.json").read_text())
    if (
        run["plan_sha256"] != sha256(plan_path)
        or decision["plan_sha256"] != run["plan_sha256"]
        or decision["complete"] is not True
        or decision["checks"] != checks
        or run["recipe"] != RECIPE
        or run["device"] != "cpu"
        or any(
            decision[name] is not False
            for name in (
                "held_out_quality_verified",
                "selected_model_changed",
                "miner_modified",
                "gpu_used",
                "metal_training_qualified",
            )
        )
    ):
        raise ValueError("capacity completion/source/scope receipt differs")
    steps, predictions, proofs, weights = (
        lines(name)
        for name in ("steps.jsonl", "fit-predictions.jsonl", "preflight.jsonl", "checkpoints.jsonl")
    )
    initializations = lines("initializations.jsonl")
    if (
        len(initializations) != 3
        or len({r["language_weights_sha256"] for r in initializations}) != 1
    ):
        raise ValueError("capacity arms did not retain the same initial language weights")
    examples = plan["examples"]
    identities = [row["example_id"] for row in examples]
    if len(set(identities)) != len(identities):
        raise ValueError("capacity audit requires unique training identities")
    references = {row["example_id"]: normalize(row["reference"]) for row in examples}
    capacities = RECIPE["capacities"]
    if len(steps) != 3 * RECIPE["epochs"] * len(examples) or len(predictions) != 3 * 9 * 2 * len(
        examples
    ):
        raise ValueError("capacity trial lacks complete update/prediction coverage")
    if len(checks) != 27 or len(weights) != 9 or len(proofs) != 36:
        raise ValueError("capacity trial lacks complete fit/checkpoint/numerical coverage")
    results = []
    for capacity in capacities:
        selected = [row for row in steps if row["capacity"] == capacity]
        expected_updates = RECIPE["epochs"] * len(examples)
        if [row["update"] for row in selected] != list(range(1, expected_updates + 1)):
            raise ValueError("capacity optimizer steps are missing, duplicated or out of order")
        tensors = 166 if capacity == "large" else 126
        for row in selected:
            coverage = row["gradient_coverage"]
            if (
                row["epoch"] != (row["update"] - 1) // len(examples) + 1
                or not math.isfinite(row["loss"])
                or not math.isfinite(row["gradient_norm"])
                or sum(group["parameter_tensors"] for group in coverage.values()) != tensors
                or any(
                    not 0 < group["nonzero_gradient_tensors"] <= group["parameter_tensors"]
                    for group in coverage.values()
                )
            ):
                raise ValueError("capacity step has invalid epoch/loss/gradient coverage")
        paired_checks = [row for row in checks if row["capacity"] == capacity]
        if [row["epoch"] for row in paired_checks] != RECIPE["checks_at_epochs"]:
            raise ValueError("capacity fit schedule changed")
        for check in paired_checks:
            rows = [
                r for r in predictions if r["capacity"] == capacity and r["epoch"] == check["epoch"]
            ]
            correct = [r for r in rows if r["wrong_video"] is False]
            swapped = [r for r in rows if r["wrong_video"] is True]
            if [r["example_id"] for r in correct] != identities or [
                r["example_id"] for r in swapped
            ] != identities:
                raise ValueError("capacity correct/swapped predictions have different coverage")
            for wrong, records in ((False, correct), (True, swapped)):
                for index, row in enumerate(records):
                    expected_input = (
                        identities[(index + 1) % len(examples)] if wrong else identities[index]
                    )
                    reference = references[row["example_id"]]
                    edits = distance(reference, normalize(row["hypothesis"]))
                    if (
                        row["input_example_id"] != expected_input
                        or row["training_only"] is not True
                        or (row["edits"], row["reference_words"], row["wer"])
                        != (edits, len(reference), edits / len(reference))
                    ):
                        raise ValueError("capacity prediction metric/input pairing differs")
            if any(check[name] != value for name, value in fit_summary(correct, swapped).items()):
                raise ValueError("capacity fit decision does not reproduce")
        cases = {case["training"]["example_id"] for case in plan["preflight_cases"]}
        stages = {"native_cpu_equivalence", "cpu_padding"}
        selected_proofs = [r for r in proofs if r["capacity"] == capacity]
        if {(r["example_id"], r["stage"]) for r in selected_proofs} != {
            (identity, stage) for identity in cases for stage in stages
        }:
            raise ValueError("capacity real CPU proof cases/stages differ")
        for row in selected_proofs:
            if row["optimizer_updates"] != 0 or row["gpu_used"] is not False:
                raise ValueError("capacity preflight performed unauthorized updates/GPU work")
            verified_comparison(
                row["comparison"], tensors, cpu_equivalence=row["stage"] == "native_cpu_equivalence"
            )
        selected_weights = [row for row in weights if row["capacity"] == capacity]
        if [row["epoch"] for row in selected_weights] != RECIPE["checkpoints_at_epochs"]:
            raise ValueError("capacity saved-weight schedule changed")
        for row in selected_weights:
            checkpoint = (ROOT / row["checkpoint"]).resolve()
            if checkpoint.parent != output or sha256(checkpoint) != row["sha256"]:
                raise ValueError("capacity checkpoint path/hash differs")
        final = paired_checks[-1]
        results.append(
            {
                "capacity": capacity,
                "optimizer_updates": len(selected),
                "final_training_fit": final,
                "final_checkpoint_sha256": selected_weights[-1]["sha256"],
                "minimum_nonzero_gradient_tensors": min(
                    sum(g["nonzero_gradient_tensors"] for g in r["gradient_coverage"].values())
                    for r in selected
                ),
            }
        )
    return {
        "schema": "umi-studio-capacity-cpu-postrun-audit/1",
        "plan_sha256": sha256(plan_path),
        "decision_sha256": sha256(output / "decision.json"),
        "auditor_sha256": sha256(Path(__file__)),
        "verified_steps": len(steps),
        "recomputed_predictions": len(predictions),
        "recomputed_fit_decisions": len(checks),
        "verified_checkpoint_hashes": len(weights),
        "verified_cpu_preflight_stages": len(proofs),
        "results": results,
        "held_out_quality_verified": False,
        "baseline_beaten": False,
    }


def main():
    sys.dont_write_bytecode = True
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.receipt.exists() or not args.receipt.resolve().is_relative_to(root / "runs"):
        parser.error("audit receipt requires a new runs/ file")
    receipt = audit(args.plan, args.checkpoints)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
