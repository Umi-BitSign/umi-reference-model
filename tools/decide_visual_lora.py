"""Apply the frozen two-domain tuning gate to visual-adapter checkpoints."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from benchmark_fleurs_dev import sha256
from decide_aligned_mixed import build_decision
from freeze_visual_lora_plan import RECIPE, TUNING_GATE, TUNING_LAYOUT


def select_checkpoint(decisions: list[tuple[int, dict]]) -> dict:
    if [step for step, _ in decisions] != RECIPE["checkpoint_after_examples"]:
        raise ValueError("visual tuning lacks a scheduled checkpoint")
    passed = [
        (step, decision)
        for step, decision in decisions
        if decision["decision"] == "proceed_to_confirmation"
    ]
    best = min(
        passed,
        key=lambda item: (item[1]["equal_domain_mean_wer_change"], item[0]),
        default=None,
    )
    return {
        "schema": "umi-studio-visual-lora-tuning-decision/1",
        "checkpoint_selection_rule": (
            "Evaluate the four scheduled checkpoints with the frozen two-domain gate. "
            "Among passing checkpoints, choose the lowest equal-domain mean WER change; "
            "break ties in favor of the earlier checkpoint. Otherwise retain the baseline."
        ),
        "selected_example_step": best[0] if best is not None else None,
        "selected_adapter_sha256": (
            best[1]["candidate_adapter_sha256"] if best is not None else None
        ),
        "decision": "proceed_to_confirmation" if best is not None else "retain_frozen_baseline",
        "confirmation_evaluated": False,
        "live_miner_changed": False,
        "checkpoints": [
            {
                "example_step": step,
                "adapter_sha256": decision["candidate_adapter_sha256"],
                "decision": decision["decision"],
                "fleurs": decision["fleurs"],
                "how2sign": decision["how2sign"],
                "how2sign_signers_improved": decision["how2sign_signers_improved"],
                "equal_domain_mean_wer_change": decision["equal_domain_mean_wer_change"],
                "fleurs_candidate_results_sha256": decision["fleurs_candidate_results_sha256"],
                "how2sign_candidate_results_sha256": decision[
                    "how2sign_candidate_results_sha256"
                ],
            }
            for step, decision in decisions
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--fleurs-candidate", type=Path, action="append", required=True)
    parser.add_argument("--how2sign-candidate", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    expected = RECIPE["checkpoint_after_examples"]
    plan = json.loads(args.plan.read_text())
    if (
        plan.get("schema") != "umi-studio-visual-lora-plan/1"
        or plan.get("recipe") != RECIPE
        or plan.get("tuning_gate") != TUNING_GATE
        or plan.get("tuning_gate_source_sha256")
        != sha256(root / "tools/decide_aligned_mixed.py")
        or plan.get("checkpoint_decision_source_sha256") != sha256(Path(__file__))
        or len(args.checkpoint) != len(expected)
        or len(args.fleurs_candidate) != len(expected)
        or len(args.how2sign_candidate) != len(expected)
        or not args.output.resolve().is_relative_to(root / "runs")
        or args.output.exists()
    ):
        parser.error("visual tuning needs the frozen plan, four checkpoints and new run output")
    from safetensors import safe_open

    def read_jsonl(path: Path) -> list[dict]:
        return [json.loads(line) for line in path.read_text().splitlines()]

    baselines = {}
    for domain, layout in TUNING_LAYOUT.items():
        record = plan["tuning"][domain]
        if (
            record["baseline"] != layout["baseline"]
            or record["index"] != layout["index"]
            or record["partition"] != layout["partition"]
            or sha256(root / record["partition"]) != record["partition_sha256"]
            or sha256(root / record["baseline"]) != record["baseline_sha256"]
            or sha256(root / record["index"]) != record["index_sha256"]
        ):
            raise ValueError(f"visual {domain} tuning baseline differs from plan")
        baselines[domain] = read_jsonl(root / record["baseline"])
    compatible_plan = {
        "schema": "umi-studio-aligned-mixed-plan/1",
        "fleurs_tuning_baseline_sha256": plan["fleurs_tuning_baseline_sha256"],
        "how2sign_tuning_baseline_sha256": plan["how2sign_tuning_baseline_sha256"],
        "baseline_mean_single_reference_wer": plan["baseline_mean_single_reference_wer"],
    }
    decisions = []
    plan_hash = sha256(args.plan)
    for step, checkpoint, fleurs_path, how2sign_path in zip(
        expected,
        args.checkpoint,
        args.fleurs_candidate,
        args.how2sign_candidate,
        strict=True,
    ):
        with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
            metadata = handle.metadata()
        if (
            metadata.get("schema") != "umi-studio-shubert-visual-lora/1"
            or metadata.get("plan_sha256") != plan_hash
            or metadata.get("recipe_id") != RECIPE["id"]
            or metadata.get("example_step") != str(step)
            or metadata.get("rank") != str(RECIPE["rank"])
            or metadata.get("alpha") != str(RECIPE["alpha"])
            or metadata.get("visual_last_layers") != str(RECIPE["visual_last_layers"])
            or metadata.get("baseline_checkpoint_sha256")
            != plan["baseline_checkpoint_sha256"]
            or metadata.get("trainer_source_sha256")
            != plan["trainer_source_sha256"]
            or metadata.get("adapter_source_sha256")
            != plan["visual_adapter_source_sha256"]
        ):
            raise ValueError(f"visual checkpoint metadata differs: {checkpoint}")
        candidates = {
            "fleurs": read_jsonl(fleurs_path),
            "how2sign": read_jsonl(how2sign_path),
        }
        if any(
            row.get("evaluation_input") != "visual_streams"
            or row.get("index_sha256") != plan["tuning"][domain]["index_sha256"]
            for domain, rows in candidates.items()
            for row in rows
        ):
            raise ValueError("visual tuning candidate lacks the frozen stream evaluation")
        hashes = {
            "plan": plan_hash,
            "fleurs_baseline": sha256(root / TUNING_LAYOUT["fleurs"]["baseline"]),
            "how2sign_baseline": sha256(root / TUNING_LAYOUT["how2sign"]["baseline"]),
            "fleurs_candidate": sha256(fleurs_path),
            "how2sign_candidate": sha256(how2sign_path),
            "adapter": sha256(checkpoint),
        }
        decision = build_decision(
            compatible_plan,
            baselines["fleurs"],
            candidates["fleurs"],
            baselines["how2sign"],
            candidates["how2sign"],
            hashes,
        )
        if decision["fleurs"]["checkpoint_sha256"] != plan["baseline_checkpoint_sha256"]:
            raise ValueError("visual tuning uses another baseline checkpoint")
        decisions.append((step, decision))
    selected = select_checkpoint(decisions)
    selected["plan_sha256"] = plan_hash
    selected["decision_source_sha256"] = sha256(Path(__file__))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".partial.json")
    if temporary.exists():
        parser.error("unfinished visual decision needs inspection")
    try:
        with temporary.open("x", encoding="utf-8") as output:
            json.dump(selected, output, sort_keys=True, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        temporary.rename(args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(selected["decision"])


if __name__ == "__main__":
    main()
