"""Select a full-decoder checkpoint using the frozen two-domain tuning gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark_fleurs_dev import sha256
from decide_aligned_mixed import build_decision
from freeze_full_decoder_plan import RECIPE


def select_checkpoint(decisions: list[tuple[int, dict]]) -> dict:
    expected = RECIPE["checkpoint_after_examples"]
    if [step for step, _ in decisions] != expected:
        raise ValueError("full decoder tuning lacks a scheduled checkpoint")
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
        "schema": "umi-studio-full-decoder-tuning-decision/1",
        "checkpoint_selection_rule": (
            "Evaluate all four scheduled checkpoints. Among those passing the frozen "
            "two-domain gate, choose the lowest equal-domain mean WER change; "
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
    parser.add_argument("--fleurs-baseline", type=Path, required=True)
    parser.add_argument("--how2sign-baseline", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--fleurs-candidate", type=Path, action="append", required=True)
    parser.add_argument("--how2sign-candidate", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    expected = RECIPE["checkpoint_after_examples"]
    if (
        plan.get("schema") != "umi-studio-full-decoder-plan/1"
        or plan.get("recipes") != [RECIPE]
        or len(args.checkpoint) != len(expected)
        or len(args.fleurs_candidate) != len(expected)
        or len(args.how2sign_candidate) != len(expected)
    ):
        raise ValueError("full decoder tuning inputs differ from the frozen plan")
    from safetensors import safe_open

    def read_jsonl(path: Path) -> list[dict]:
        return [json.loads(line) for line in path.read_text().splitlines()]

    fleurs_baseline = read_jsonl(args.fleurs_baseline)
    how2sign_baseline = read_jsonl(args.how2sign_baseline)
    compatible_plan = {**plan, "schema": "umi-studio-aligned-mixed-plan/1"}
    decisions = []
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
            metadata.get("schema") != "umi-studio-shubert-full-decoder/1"
            or metadata.get("plan_sha256") != sha256(args.plan)
            or metadata.get("recipe_id") != RECIPE["id"]
            or metadata.get("example_step") != str(step)
            or metadata.get("fleurs_index_sha256") != plan["fleurs_training_index_sha256"]
            or metadata.get("how2sign_index_sha256")
            != plan["how2sign_training_index_sha256"]
        ):
            raise ValueError(f"full decoder checkpoint metadata differs: {checkpoint}")
        hashes = {
            "plan": sha256(args.plan),
            "fleurs_baseline": sha256(args.fleurs_baseline),
            "how2sign_baseline": sha256(args.how2sign_baseline),
            "fleurs_candidate": sha256(fleurs_path),
            "how2sign_candidate": sha256(how2sign_path),
            "adapter": sha256(checkpoint),
        }
        decision = build_decision(
            compatible_plan,
            fleurs_baseline,
            read_jsonl(fleurs_path),
            how2sign_baseline,
            read_jsonl(how2sign_path),
            hashes,
        )
        if metadata.get("baseline_checkpoint_sha256") != decision["fleurs"][
            "checkpoint_sha256"
        ]:
            raise ValueError("full decoder adapter uses another baseline checkpoint")
        decisions.append((step, decision))
    selected = select_checkpoint(decisions)
    selected["plan_sha256"] = sha256(args.plan)
    selected["decision_source_sha256"] = sha256(Path(__file__))
    if args.output.exists():
        if json.loads(args.output.read_text()) != selected:
            raise ValueError("frozen full decoder decision differs from current results")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as output:
            json.dump(selected, output, sort_keys=True, indent=2)
            output.write("\n")
    print(selected["decision"])


if __name__ == "__main__":
    main()
