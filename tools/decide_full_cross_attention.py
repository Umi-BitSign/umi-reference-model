"""Apply the frozen two-domain tuning gate to a full cross-attention checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark_fleurs_dev import sha256
from decide_aligned_mixed import build_decision


def build_full_decision(
    plan: dict,
    fleurs_baseline: list[dict],
    fleurs_candidate: list[dict],
    how2sign_baseline: list[dict],
    how2sign_candidate: list[dict],
    hashes: dict[str, str],
) -> dict:
    if plan.get("schema") != "umi-studio-full-cross-attention-plan/1" or plan.get(
        "training_examples"
    ) != {"fleurs": 313, "how2sign": 1024, "flores2m": 0}:
        raise ValueError("unexpected full cross-attention training plan")
    compatible_plan = {**plan, "schema": "umi-studio-aligned-mixed-plan/1"}
    decision = build_decision(
        compatible_plan,
        fleurs_baseline,
        fleurs_candidate,
        how2sign_baseline,
        how2sign_candidate,
        hashes,
    )
    decision["schema"] = "umi-studio-full-cross-attention-tuning-decision/1"
    decision["prior_rejected_pilot_decision_sha256"] = plan["prior_rejected_pilot_decision_sha256"]
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--fleurs-baseline", type=Path, required=True)
    parser.add_argument("--fleurs-candidate", type=Path, required=True)
    parser.add_argument("--how2sign-baseline", type=Path, required=True)
    parser.add_argument("--how2sign-candidate", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sources = {
        "plan": args.plan,
        "fleurs_baseline": args.fleurs_baseline,
        "fleurs_candidate": args.fleurs_candidate,
        "how2sign_baseline": args.how2sign_baseline,
        "how2sign_candidate": args.how2sign_candidate,
        "adapter": args.checkpoint,
    }
    hashes = {name: sha256(path) for name, path in sources.items()}
    from safetensors import safe_open

    with safe_open(str(args.checkpoint), framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
    plan = json.loads(args.plan.read_text())
    if (
        metadata.get("schema") != "umi-studio-shubert-cross-attention/1"
        or metadata.get("plan_sha256") != hashes["plan"]
        or metadata.get("recipe_id") != plan["recipes"][0]["id"]
        or metadata.get("how2sign_index_sha256") != plan["how2sign_training_index_sha256"]
    ):
        raise ValueError("cross-attention checkpoint metadata differs from the frozen plan")

    def read_jsonl(path: Path) -> list[dict]:
        return [json.loads(line) for line in path.read_text().splitlines()]

    decision = build_full_decision(
        plan,
        read_jsonl(args.fleurs_baseline),
        read_jsonl(args.fleurs_candidate),
        read_jsonl(args.how2sign_baseline),
        read_jsonl(args.how2sign_candidate),
        hashes,
    )
    if metadata.get("baseline_checkpoint_sha256") != decision["fleurs"]["checkpoint_sha256"]:
        raise ValueError("candidate was trained from another frozen baseline")
    if args.output.exists():
        if json.loads(args.output.read_text()) != decision:
            raise ValueError("frozen tuning decision differs from current results")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as output:
            json.dump(decision, output, sort_keys=True, indent=2)
            output.write("\n")
    print(decision["decision"])


if __name__ == "__main__":
    main()
