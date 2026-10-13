"""Evaluate anchored-projection C5 checkpoints on frozen open tuning."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from benchmark_fleurs_dev import sha256
from freeze_c5_anchored_projection_plan import BASELINE, RECIPE, ROOT, TUNING
from run_c5_stem_tuning import LAYOUT, read_jsonl, summarize_candidate

RUNTIME = BASELINE / ".runtime/bin/python"
CHECKPOINT = BASELINE / "models/checkpoint-11625"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.output.exists()
        or not args.output.resolve().is_relative_to((ROOT / "runs").resolve())
    ):
        parser.error("anchored tuning needs a new decision output")
    plan = json.loads(args.plan.read_text())
    if (
        plan.get("schema") != "umi-studio-c5-anchored-projection-plan/1"
        or plan.get("recipe") != RECIPE
    ):
        raise ValueError("anchored tuning plan differs")
    checkpoint_rows = read_jsonl(args.checkpoint_dir / "checkpoint-index.jsonl")
    by_step = {row["example_step"]: row for row in checkpoint_rows}
    if set(by_step) != set(RECIPE["checkpoint_after_examples"]):
        raise ValueError("anchored projection checkpoints are incomplete")
    baseline_rows = {domain: read_jsonl(path) for domain, path in TUNING.items()}
    summaries = {}
    result_hashes = {}
    for step in RECIPE["checkpoint_after_examples"]:
        adapter = args.checkpoint_dir / f"visual-lora-after-{step:06d}.safetensors"
        if sha256(adapter) != by_step[step]["checkpoint_sha256"]:
            raise ValueError(f"anchored checkpoint changed: {step}")
        summaries[str(step)] = {}
        result_hashes[str(step)] = {}
        for domain, layout in LAYOUT.items():
            results = args.output.parent / f"tuning-{domain}-{step:06d}.jsonl"
            subprocess.run(
                [
                    "nice", "-n", "20", str(RUNTIME),
                    "tools/evaluate_fleurs_cache.py",
                    "--index", str(layout["index"]),
                    "--cache-dir", str(layout["cache"]),
                    "--split", str(layout["split"]),
                    "--baseline", str(BASELINE),
                    "--checkpoint", str(CHECKPOINT),
                    "--adapter", str(adapter),
                    "--results", str(results),
                    "--miner-maintenance-mode",
                    "--device", "cpu",
                    "--use-stream-features",
                    "--temporal-stride", "2",
                    "--limit", str(layout["examples"]),
                ],
                cwd=ROOT,
                check=True,
                stdout=subprocess.DEVNULL,
            )
            candidate = read_jsonl(results)
            summaries[str(step)][domain] = summarize_candidate(
                baseline_rows[domain], candidate
            )
            result_hashes[str(step)][domain] = sha256(results)
    passing = []
    for step in RECIPE["checkpoint_after_examples"]:
        row = summaries[str(step)]
        h_delta = row["how2sign"]["candidate_minus_baseline_mean_wer"]
        f_delta = row["fleurs"]["candidate_minus_baseline_mean_wer"]
        improved_signers = sum(
            value["candidate_mean_wer"] < value["baseline_mean_wer"]
            for value in row["how2sign"]["signer_means"].values()
        )
        row["equal_domain_mean_wer_change"] = (h_delta + f_delta) / 2
        row["how2sign_signers_improved"] = improved_signers
        screen = plan["screen"]
        row["passes_screen"] = (
            h_delta <= screen["how2sign_max_mean_wer_change"]
            and f_delta <= screen["fleurs_max_mean_wer_change"]
            and row["equal_domain_mean_wer_change"]
            <= screen["equal_domain_max_mean_wer_change"]
            and improved_signers >= screen["minimum_improved_how2sign_signers"]
        )
        if row["passes_screen"]:
            passing.append(step)
    selected = min(
        passing,
        key=lambda step: (summaries[str(step)]["equal_domain_mean_wer_change"], step),
        default=None,
    )
    result = {
        "schema": "umi-studio-c5-anchored-projection-tuning-decision/1",
        "plan_sha256": sha256(args.plan),
        "starting_adapter_sha256": plan["starting_adapter_sha256"],
        "screen": plan["screen"],
        "checkpoints": summaries,
        "result_sha256": result_hashes,
        "selected_step": selected,
        "decision": "evaluate_calibrated_runtime" if selected is not None else "reject_anchored_projection",
        "prior_confirmation_reused": False,
        "confirmation_variants_evaluated": 0,
        "miner_changed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2, sort_keys=True)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    print(json.dumps({"decision": result["decision"], "selected_step": selected}, sort_keys=True))


if __name__ == "__main__":
    main()
