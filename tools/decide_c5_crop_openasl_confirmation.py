"""Compare the frozen native and calibrated OpenASL crop confirmation results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark_fleurs_dev import sha256
from evaluate_c5_crop_openasl_confirmation import SCHEMA

ROOT = Path(__file__).resolve().parents[1]


def load(path: Path, *, plan_hash: str, variant: str) -> dict[str, dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    output = {}
    for row in rows:
        clip_id = row.get("clip_id")
        if (
            row.get("schema") != SCHEMA
            or row.get("confirmation_plan_sha256") != plan_hash
            or row.get("variant") != variant
            or clip_id in output
        ):
            raise ValueError(f"{variant} confirmation results differ")
        output[clip_id] = row
    if len(output) != 60:
        raise ValueError(f"{variant} confirmation must contain 60 examples")
    return output


def summarize(rows: dict[str, dict]) -> dict[str, float | int]:
    values = list(rows.values())
    return {
        "examples": len(values),
        "mean_wer": sum(row["single_reference_wer"] for row in values) / len(values),
        "word_weighted_wer": sum(row["single_reference_word_edits"] for row in values)
        / sum(row["reference_word_count"] for row in values),
        "active_seconds": sum(row["active_seconds"] for row in values),
        "distinct_hypotheses": len({row["hypothesis"] for row in values}),
    }


def decide(plan: dict, native: dict[str, dict], candidate: dict[str, dict]) -> dict:
    if set(native) != set(candidate) or set(native) != {
        row["clip_id"] for row in plan["examples"]
    }:
        raise ValueError("OpenASL confirmation variants are not paired")
    baseline = summarize(native)
    calibrated = summarize(candidate)
    paired = {"improved": 0, "tied": 0, "regressed": 0}
    for clip_id in native:
        delta = (
            candidate[clip_id]["single_reference_wer"]
            - native[clip_id]["single_reference_wer"]
        )
        key = "improved" if delta < 0 else "regressed" if delta > 0 else "tied"
        paired[key] += 1
    mean_delta = calibrated["mean_wer"] - baseline["mean_wer"]
    weighted_delta = calibrated["word_weighted_wer"] - baseline["word_weighted_wer"]
    gate = plan["quality_gate"]
    passes = (
        mean_delta <= gate["maximum_mean_wer_delta"]
        and weighted_delta <= gate["maximum_word_weighted_wer_delta"]
        and paired["improved"] - paired["regressed"]
        >= gate["minimum_paired_improved_minus_regressed"]
    )
    return {
        "baseline": baseline,
        "calibrated": calibrated,
        "mean_wer_delta": mean_delta,
        "word_weighted_wer_delta": weighted_delta,
        "active_time_ratio": calibrated["active_seconds"] / baseline["active_seconds"],
        "paired": paired,
        "quality_gate_passed": passes,
        "decision": "promote_crop_candidate" if passes else "retain_native_candidate",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--calibrated", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        not all(path.is_file() for path in (args.plan, args.native, args.calibrated))
        or args.output.exists()
        or not args.output.resolve().is_relative_to((ROOT / "runs").resolve())
    ):
        parser.error("confirmation inputs or output are invalid")
    plan = json.loads(args.plan.read_text())
    plan_hash = sha256(args.plan)
    native = load(args.native, plan_hash=plan_hash, variant="native224")
    calibrated = load(
        args.calibrated, plan_hash=plan_hash, variant="calibrated168"
    )
    result = {
        "schema": "umi-studio-c5-crop-openasl-confirmation-decision/1",
        "confirmation_plan_sha256": plan_hash,
        "native_results_sha256": sha256(args.native),
        "calibrated_results_sha256": sha256(args.calibrated),
        "calibrator_sha256": plan["calibrator_sha256"],
        "source_videos": 60,
        "fresh_external_confirmation": True,
        "miner_installed": False,
        **decide(plan, native, calibrated),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
