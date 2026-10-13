"""Apply the frozen tuning gate to the projection-frozen visual ablation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from safetensors import safe_open

from ablate_visual_projection import RECIPE_ID
from benchmark_fleurs_dev import sha256
from decide_aligned_mixed import build_decision
from decide_visual_interpolation import read_jsonl
from freeze_visual_lora_plan import TUNING_GATE, TUNING_LAYOUT
from run_visual_lora_tuning import DOMAINS, checked_results

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--fleurs-candidate", type=Path, required=True)
    parser.add_argument("--how2sign-candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(ROOT / "runs"):
        parser.error("projection ablation decision needs a new run-ledger output")
    plan = json.loads(args.plan.read_text())
    plan_hash = sha256(args.plan)
    adapter_hash = sha256(args.adapter)
    parent = ROOT / plan["parent_plan"]
    if (
        plan.get("schema") != "umi-studio-visual-projection-ablation-plan/1"
        or plan.get("recipe_id") != RECIPE_ID
        or plan.get("tuning_gate") != TUNING_GATE
        or plan.get("parent_plan_sha256") != sha256(parent)
        or plan.get("ablation_source_sha256")
        != sha256(ROOT / "tools/ablate_visual_projection.py")
        or plan.get("baseline_checkpoint_sha256")
        != sha256(ROOT / "artifacts/baseline-6fe8df59/models/checkpoint-11625/model.safetensors")
    ):
        raise ValueError("projection ablation plan differs from its frozen parent")
    with safe_open(str(args.adapter), framework="pt", device="cpu") as source:
        metadata = source.metadata()
    if (
        metadata.get("schema") != "umi-studio-shubert-visual-lora/1"
        or metadata.get("plan_sha256") != plan_hash
        or metadata.get("recipe_id") != RECIPE_ID
        or metadata.get("ablation") != "baseline_visual_projection"
        or metadata.get("source_adapter_sha256") != plan["source_adapter_sha256"]
        or metadata.get("baseline_checkpoint_sha256") != plan["baseline_checkpoint_sha256"]
    ):
        raise ValueError("projection ablation adapter metadata differs")
    candidates = {
        "fleurs": args.fleurs_candidate,
        "how2sign": args.how2sign_candidate,
    }
    baselines = {}
    for domain, path in candidates.items():
        tuning = plan["tuning"][domain]
        index = ROOT / TUNING_LAYOUT[domain]["index"]
        baseline = ROOT / TUNING_LAYOUT[domain]["baseline"]
        if (
            tuning["index_sha256"] != sha256(index)
            or tuning["baseline_sha256"] != sha256(baseline)
            or checked_results(
                path, index, DOMAINS[domain]["split"],
                plan["baseline_checkpoint_sha256"], adapter_hash,
            ) != TUNING_LAYOUT[domain]["examples"]
        ):
            raise ValueError(f"projection ablation {domain} results are incomplete")
        baselines[domain] = read_jsonl(baseline)
    compatible_plan = {
        "schema": "umi-studio-aligned-mixed-plan/1",
        "fleurs_tuning_baseline_sha256": plan["tuning"]["fleurs"]["baseline_sha256"],
        "how2sign_tuning_baseline_sha256": plan["tuning"]["how2sign"]["baseline_sha256"],
        "baseline_mean_single_reference_wer": {
            domain: plan["tuning"][domain]["baseline_mean_single_reference_wer"]
            for domain in candidates
        },
    }
    hashes = {
        "plan": plan_hash,
        "adapter": adapter_hash,
        "fleurs_baseline": sha256(ROOT / TUNING_LAYOUT["fleurs"]["baseline"]),
        "how2sign_baseline": sha256(ROOT / TUNING_LAYOUT["how2sign"]["baseline"]),
        "fleurs_candidate": sha256(args.fleurs_candidate),
        "how2sign_candidate": sha256(args.how2sign_candidate),
    }
    decision = build_decision(
        compatible_plan,
        baselines["fleurs"],
        read_jsonl(args.fleurs_candidate),
        baselines["how2sign"],
        read_jsonl(args.how2sign_candidate),
        hashes,
    )
    if decision["fleurs"]["checkpoint_sha256"] != plan["baseline_checkpoint_sha256"]:
        raise ValueError("projection ablation uses another baseline checkpoint")
    decision.update(
        schema="umi-studio-visual-projection-ablation-decision/1",
        source_adapter_sha256=plan["source_adapter_sha256"],
        ablation="baseline_visual_projection",
        decision_source_sha256=sha256(Path(__file__)),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".partial.json")
    if temporary.exists():
        parser.error("unfinished projection ablation decision needs inspection")
    try:
        with temporary.open("x", encoding="utf-8") as output:
            json.dump(decision, output, sort_keys=True, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        temporary.rename(args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(decision["decision"])


if __name__ == "__main__":
    main()
