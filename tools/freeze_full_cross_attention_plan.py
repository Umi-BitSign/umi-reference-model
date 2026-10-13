"""Freeze the full aligned-data cross-attention fine-tuning experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark_fleurs_dev import sha256
from freeze_aligned_mixed_plan import build_plan


def cross_attention_plan(base: dict, prior_decision_hash: str) -> dict:
    if base["schema"] != "umi-studio-aligned-mixed-plan/1" or base["training_examples"] != {
        "fleurs": 313,
        "how2sign": 1024,
        "flores2m": 0,
    }:
        raise ValueError("cross-attention experiment requires all aligned training states")
    plan = dict(base)
    plan.pop("rank")
    plan.pop("alpha")
    plan.pop("flores2m_repeat")
    plan["schema"] = "umi-studio-full-cross-attention-plan/1"
    plan["prior_rejected_pilot_decision_sha256"] = prior_decision_hash
    plan["trainable_scope"] = "decoder cross-attention q, k, v and o projection weights only"
    plan["recipes"] = [
        {
            "id": "full_cross_attention_e1",
            "epochs": 1,
            "learning_rate": 0.00003,
            "weight_decay": 0.1,
            "label_smoothing": 0.2,
        }
    ]
    plan["training_policy"] = (
        "Use all 1,024 alignment-audited How2Sign training states and 313 public "
        "FLEURS training states. Keep the video encoder and non-cross-attention "
        "decoder weights frozen. No validation or service material enters training."
    )
    plan["paper_context"] = (
        "The published SHuBERT target-data recipe fine-tunes longer and updates more "
        "weights. This one-epoch cross-attention run is a bounded diagnostic, not a "
        "reproduction of the published recipe."
    )
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fleurs-index", type=Path, required=True)
    parser.add_argument("--how2sign-index", type=Path, required=True)
    parser.add_argument("--how2sign-selection", type=Path, required=True)
    parser.add_argument("--fleurs-partition", type=Path, required=True)
    parser.add_argument("--how2sign-partition", type=Path, required=True)
    parser.add_argument("--fleurs-baseline", type=Path, required=True)
    parser.add_argument("--how2sign-baseline", type=Path, required=True)
    parser.add_argument("--prior-decision", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prior = json.loads(args.prior_decision.read_text())
    if (
        prior.get("schema") != "umi-studio-aligned-mixed-tuning-decision/1"
        or prior.get("decision") != "retain_frozen_baseline"
        or prior.get("confirmation_evaluated") is not False
        or prior.get("live_miner_changed") is not False
    ):
        raise ValueError("prior mixed pilot is not a rejected offline tuning run")
    sources = {
        "fleurs_index": args.fleurs_index,
        "how2sign_index": args.how2sign_index,
        "how2sign_selection": args.how2sign_selection,
        "fleurs_partition": args.fleurs_partition,
        "how2sign_partition": args.how2sign_partition,
        "fleurs_baseline": args.fleurs_baseline,
        "how2sign_baseline": args.how2sign_baseline,
    }
    hashes = {name: sha256(path) for name, path in sources.items()}

    def read_jsonl(path: Path) -> list[dict]:
        return [json.loads(line) for line in path.read_text().splitlines()]

    base = build_plan(
        read_jsonl(args.fleurs_index),
        read_jsonl(args.how2sign_index),
        json.loads(args.how2sign_selection.read_text()),
        read_jsonl(args.fleurs_baseline),
        read_jsonl(args.how2sign_baseline),
        json.loads(args.how2sign_partition.read_text()),
        hashes,
        1024,
    )
    plan = cross_attention_plan(base, sha256(args.prior_decision))
    if args.output.exists():
        if json.loads(args.output.read_text()) != plan:
            raise ValueError("frozen cross-attention plan differs from current inputs")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as output:
            json.dump(plan, output, sort_keys=True, indent=2)
            output.write("\n")
    print("frozen 1,337-example full cross-attention training plan")


if __name__ == "__main__":
    main()
