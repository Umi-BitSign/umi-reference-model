"""Freeze a full-decoder follow-up to the rejected cross-attention run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark_fleurs_dev import sha256

RECIPE = {
    "id": "full_decoder_e1_lr1e5",
    "epochs": 1,
    "learning_rate": 0.00001,
    "weight_decay": 0.1,
    "label_smoothing": 0.2,
    "gradient_accumulation": 4,
    "checkpoint_after_examples": [336, 672, 1008, 1337],
}


def full_decoder_plan(
    previous: dict, decision: dict, previous_hash: str, decision_hash: str
) -> dict:
    if (
        previous.get("schema") != "umi-studio-full-cross-attention-plan/1"
        or previous.get("training_examples") != {"fleurs": 313, "how2sign": 1024, "flores2m": 0}
        or previous.get("seed") != 17
        or previous.get("fleurs_repeat") != 1
        or previous.get("how2sign_repeat") != 1
        or decision.get("schema") != "umi-studio-full-cross-attention-tuning-decision/1"
        or decision.get("decision") != "retain_frozen_baseline"
        or decision.get("plan_sha256") != previous_hash
        or decision.get("confirmation_evaluated") is not False
        or decision.get("live_miner_changed") is not False
    ):
        raise ValueError("full decoder run needs the rejected, undeployed prior experiment")
    plan = {
        **previous,
        "schema": "umi-studio-full-decoder-plan/1",
        "previous_plan_sha256": previous_hash,
        "previous_decision_sha256": decision_hash,
        "trainable_scope": "complete ByT5 decoder, decoder embedding and output head",
        "recipes": [RECIPE],
        "training_policy": (
            "Use the same frozen 1,024 aligned How2Sign and 313 FLEURS training states "
            "as the rejected cross-attention run. Keep the video encoder frozen, train "
            "the complete translation decoder, and keep all validation and service data out."
        ),
        "paper_context": (
            "This tests whether the previous cross-attention-only scope was too narrow. "
            "It is not a reproduction of published joint SHuBERT and ByT5 fine-tuning."
        ),
    }
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-plan", type=Path, required=True)
    parser.add_argument("--previous-decision", type=Path, required=True)
    parser.add_argument("--fleurs-index", type=Path, required=True)
    parser.add_argument("--how2sign-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    previous_hash = sha256(args.previous_plan)
    previous = json.loads(args.previous_plan.read_text())
    if previous["fleurs_training_index_sha256"] != sha256(args.fleurs_index) or previous[
        "how2sign_training_index_sha256"
    ] != sha256(args.how2sign_index):
        raise ValueError("full decoder inputs differ from the prior verified training indexes")
    plan = full_decoder_plan(
        previous,
        json.loads(args.previous_decision.read_text()),
        previous_hash,
        sha256(args.previous_decision),
    )
    if args.output.exists():
        if json.loads(args.output.read_text()) != plan:
            raise ValueError("frozen full decoder plan differs from its source decisions")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as output:
            json.dump(plan, output, sort_keys=True, indent=2)
            output.write("\n")
    print("frozen 1,337-example full-decoder follow-up plan")


if __name__ == "__main__":
    main()
