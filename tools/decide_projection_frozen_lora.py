"""Apply the frozen two-domain gate to projection-frozen LoRA checkpoints."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from benchmark_fleurs_dev import miner_revision, sha256
from decide_aligned_mixed import build_decision
from decide_visual_lora import select_checkpoint
from freeze_projection_frozen_lora_plan import RECIPE, ROOT
from freeze_visual_lora_plan import TUNING_GATE, TUNING_LAYOUT
from run_projection_frozen_lora_tuning import checked_checkpoints
from run_visual_lora_tuning import DOMAINS, checked_results, result_path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def decide(plan_path: Path, checkpoint_dir: Path, results_dir: Path, health_url: str) -> dict:
    plan = json.loads(plan_path.read_text())
    plan_hash = sha256(plan_path)
    if (
        plan.get("schema") != "umi-studio-visual-lora-projection-frozen-plan/1"
        or plan.get("recipe") != RECIPE
        or plan.get("tuning_gate") != TUNING_GATE
        or sha256(ROOT / plan["parent_plan"]) != plan["parent_plan_sha256"]
        or sha256(ROOT / plan["starting_adapter"]) != plan["starting_adapter_sha256"]
        or sha256(ROOT / "tools/train_projection_frozen_lora.py") != plan["trainer_source_sha256"]
    ):
        raise ValueError("projection-frozen tuning decision differs from the frozen plan")
    revision = miner_revision(health_url, plan["miner_model_revision"])
    checkpoints = checked_checkpoints(checkpoint_dir, plan, plan_hash)
    baselines = {}
    for domain, layout in TUNING_LAYOUT.items():
        tuning = plan["tuning"][domain]
        if (
            tuning["index"] != layout["index"]
            or tuning["baseline"] != layout["baseline"]
            or sha256(ROOT / tuning["index"]) != tuning["index_sha256"]
            or sha256(ROOT / tuning["baseline"]) != tuning["baseline_sha256"]
        ):
            raise ValueError(f"projection-frozen {domain} tuning baseline changed")
        baselines[domain] = read_jsonl(ROOT / tuning["baseline"])
    compatible_plan = {
        "schema": "umi-studio-aligned-mixed-plan/1",
        "fleurs_tuning_baseline_sha256": plan["tuning"]["fleurs"]["baseline_sha256"],
        "how2sign_tuning_baseline_sha256": plan["tuning"]["how2sign"]["baseline_sha256"],
        "baseline_mean_single_reference_wer": {
            domain: plan["tuning"][domain]["baseline_mean_single_reference_wer"]
            for domain in DOMAINS
        },
    }
    decisions = []
    for step, adapter in checkpoints.items():
        candidate_paths = {
            domain: result_path(results_dir, domain, step)
            for domain in DOMAINS
        }
        adapter_hash = sha256(adapter)
        candidates = {}
        for domain, path in candidate_paths.items():
            count = checked_results(
                path,
                ROOT / TUNING_LAYOUT[domain]["index"],
                DOMAINS[domain]["split"],
                plan["baseline_checkpoint_sha256"],
                adapter_hash,
            )
            if count != TUNING_LAYOUT[domain]["examples"]:
                raise ValueError(f"projection-frozen {domain} tuning incomplete at {step}")
            candidates[domain] = read_jsonl(path)
        hashes = {
            "plan": plan_hash,
            "adapter": adapter_hash,
            "fleurs_baseline": plan["tuning"]["fleurs"]["baseline_sha256"],
            "how2sign_baseline": plan["tuning"]["how2sign"]["baseline_sha256"],
            "fleurs_candidate": sha256(candidate_paths["fleurs"]),
            "how2sign_candidate": sha256(candidate_paths["how2sign"]),
        }
        decision = build_decision(
            compatible_plan,
            baselines["fleurs"], candidates["fleurs"],
            baselines["how2sign"], candidates["how2sign"],
            hashes,
        )
        if decision["fleurs"]["checkpoint_sha256"] != plan["baseline_checkpoint_sha256"]:
            raise ValueError("projection-frozen evaluation used another baseline checkpoint")
        decisions.append((step, decision))
    selection = select_checkpoint(decisions)
    selection.update(
        schema="umi-studio-visual-lora-projection-frozen-tuning-decision/1",
        plan_sha256=plan_hash,
        starting_adapter_sha256=plan["starting_adapter_sha256"],
        decision_source_sha256=sha256(Path(__file__)),
        miner_model_revision=revision,
    )
    return selection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--miner-health-url", required=True)
    args = parser.parse_args()
    if (
        args.output.exists()
        or not args.plan.resolve().is_relative_to(ROOT / "runs")
        or not args.checkpoint_dir.resolve().is_relative_to(ROOT / "checkpoints")
        or not args.results_dir.resolve().is_relative_to(ROOT / "runs")
        or not args.output.resolve().is_relative_to(ROOT / "runs")
    ):
        parser.error("projection-frozen decision needs existing ledgers and a new output")
    selection = decide(
        args.plan, args.checkpoint_dir, args.results_dir, args.miner_health_url
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".partial.json")
    if temporary.exists():
        parser.error("unfinished projection-frozen decision needs inspection")
    try:
        with temporary.open("x", encoding="utf-8") as output:
            json.dump(selection, output, sort_keys=True, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        temporary.rename(args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(selection["decision"])


if __name__ == "__main__":
    main()
