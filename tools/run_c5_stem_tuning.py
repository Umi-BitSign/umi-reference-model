"""Evaluate the four frozen ASL STEM continuation checkpoints on open tuning."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path

from benchmark_fleurs_dev import sha256
from freeze_c5_stem_continuation_plan import BASELINE, RECIPE, ROOT, STARTING_ADAPTER, TUNING

RUNTIME = BASELINE / ".runtime/bin/python"
CHECKPOINT = BASELINE / "models/checkpoint-11625"
LAYOUT = {
    "how2sign": {
        "index": ROOT / "runs/how2sign-validation-v1/stream-encoder-cache.jsonl",
        "cache": ROOT / "data/how2sign/validation-stream-encoder-cache-25fps",
        "split": "validation",
        "examples": 16,
    },
    "fleurs": {
        "index": ROOT / "runs/fleurs-dev-v1/tuning-stream-encoder-cache.jsonl",
        "cache": ROOT / "data/fleurs-asl/encoder-cache-25fps/tuning-stream",
        "split": "tuning",
        "examples": 7,
    },
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def summarize_candidate(baseline: list[dict], candidate: list[dict]) -> dict:
    if not baseline or len(baseline) != len(candidate):
        raise ValueError("continuation tuning counts differ")
    by_signer = defaultdict(lambda: {"baseline": [], "candidate": []})
    for first, second in zip(baseline, candidate, strict=True):
        if (
            first["example_id"] != second["example_id"]
            or first["reference"] != second["reference"]
            or first["index_sha256"] != second["index_sha256"]
            or first["checkpoint_sha256"] != second["checkpoint_sha256"]
            or first.get("adapter_sha256") is None
            or second.get("adapter_sha256") is None
            or first.get("temporal_stride") != 2
            or second.get("temporal_stride") != 2
            or not math.isfinite(first["single_reference_wer"])
            or not math.isfinite(second["single_reference_wer"])
        ):
            raise ValueError("continuation tuning inputs differ")
        signer = str(first["signer"])
        by_signer[signer]["baseline"].append(first["single_reference_wer"])
        by_signer[signer]["candidate"].append(second["single_reference_wer"])
    baseline_mean = statistics.fmean(row["single_reference_wer"] for row in baseline)
    candidate_mean = statistics.fmean(row["single_reference_wer"] for row in candidate)
    signer_means = {
        signer: {
            "baseline_mean_wer": statistics.fmean(values["baseline"]),
            "candidate_mean_wer": statistics.fmean(values["candidate"]),
            "examples": len(values["baseline"]),
        }
        for signer, values in sorted(by_signer.items())
    }
    return {
        "examples": len(baseline),
        "baseline_mean_wer": baseline_mean,
        "candidate_mean_wer": candidate_mean,
        "candidate_minus_baseline_mean_wer": candidate_mean - baseline_mean,
        "signer_means": signer_means,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(
        (ROOT / "runs").resolve()
    ):
        parser.error("tuning decision needs a new run-ledger output")
    plan = json.loads(args.plan.read_text())
    if (
        plan.get("schema") != "umi-studio-c5-stem-continuation-plan/1"
        or plan.get("recipe") != RECIPE
        or plan.get("starting_adapter_sha256") != sha256(STARTING_ADAPTER)
        or plan.get("confirmation_decision_sha256")
        != sha256(ROOT / "runs/temporal-stride-v1/confirmation-decision.json")
    ):
        raise ValueError("continuation tuning plan differs")
    checkpoint_rows = read_jsonl(args.checkpoint_dir / "checkpoint-index.jsonl")
    by_step = {row["example_step"]: row for row in checkpoint_rows}
    if set(by_step) != set(RECIPE["checkpoint_after_examples"]):
        raise ValueError("continuation checkpoints are incomplete")
    baseline_rows = {domain: read_jsonl(path) for domain, path in TUNING.items()}
    summaries = {}
    result_hashes = {}
    for step in RECIPE["checkpoint_after_examples"]:
        adapter = args.checkpoint_dir / f"visual-lora-after-{step:06d}.safetensors"
        if sha256(adapter) != by_step[step]["checkpoint_sha256"]:
            raise ValueError(f"continuation checkpoint changed: {step}")
        summaries[str(step)] = {}
        result_hashes[str(step)] = {}
        for domain, layout in LAYOUT.items():
            results = args.output.parent / f"tuning-{domain}-{step:06d}.jsonl"
            subprocess.run(
                [
                    "nice",
                    "-n",
                    "20",
                    str(RUNTIME),
                    "tools/evaluate_fleurs_cache.py",
                    "--index",
                    str(layout["index"]),
                    "--cache-dir",
                    str(layout["cache"]),
                    "--split",
                    str(layout["split"]),
                    "--baseline",
                    str(BASELINE),
                    "--checkpoint",
                    str(CHECKPOINT),
                    "--adapter",
                    str(adapter),
                    "--results",
                    str(results),
                    "--miner-maintenance-mode",
                    "--device",
                    "cpu",
                    "--use-stream-features",
                    "--temporal-stride",
                    "2",
                    "--limit",
                    str(layout["examples"]),
                ],
                cwd=ROOT,
                check=True,
                stdout=subprocess.DEVNULL,
            )
            candidate = read_jsonl(results)
            if {row["adapter_sha256"] for row in candidate} != {sha256(adapter)}:
                raise ValueError(f"continuation {domain} result adapter differs: {step}")
            summaries[str(step)][domain] = summarize_candidate(
                baseline_rows[domain], candidate
            )
            result_hashes[str(step)][domain] = sha256(results)
    screen = plan["screen"]
    passing = []
    for step in RECIPE["checkpoint_after_examples"]:
        row = summaries[str(step)]
        h_delta = row["how2sign"]["candidate_minus_baseline_mean_wer"]
        f_delta = row["fleurs"]["candidate_minus_baseline_mean_wer"]
        improved_signers = sum(
            values["candidate_mean_wer"] < values["baseline_mean_wer"]
            for values in row["how2sign"]["signer_means"].values()
        )
        row["equal_domain_mean_wer_change"] = (h_delta + f_delta) / 2
        row["how2sign_signers_improved"] = improved_signers
        row["passes_screen"] = (
            h_delta <= screen["how2sign_max_mean_wer_change"]
            and f_delta <= screen["fleurs_max_mean_wer_change"]
            and row["equal_domain_mean_wer_change"]
            <= screen["equal_domain_max_mean_wer_change"]
            and improved_signers >= screen["minimum_improved_how2sign_signers"]
        )
        if row["passes_screen"]:
            passing.append(step)
    selected = (
        min(
            passing,
            key=lambda step: (
                summaries[str(step)]["equal_domain_mean_wer_change"],
                step,
            ),
        )
        if passing
        else None
    )
    result = {
        "schema": "umi-studio-c5-stem-continuation-tuning-decision/1",
        "plan_sha256": sha256(args.plan),
        "starting_adapter_sha256": sha256(STARTING_ADAPTER),
        "screen": screen,
        "checkpoints": summaries,
        "result_sha256": result_hashes,
        "selected_step": selected,
        "decision": (
            "collect_fresh_heldout"
            if selected is not None
            else "retain_confirmed_candidate"
        ),
        "prior_confirmation_reused": False,
        "confirmation_variants_evaluated": 0,
        "miner_changed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".partial.json")
    try:
        with temporary.open("x", encoding="utf-8") as output:
            json.dump(result, output, sort_keys=True, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        temporary.rename(args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(result["decision"])


if __name__ == "__main__":
    main()
