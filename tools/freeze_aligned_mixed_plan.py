"""Freeze a first aligned How2Sign/FLEURS LoRA experiment.

The output pins the exact training cache snapshot and both tuning baselines.
Confirmation data stays sealed until the candidate passes the frozen gate.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from benchmark_fleurs_dev import sha256


def build_plan(
    fleurs: list[dict],
    how2sign: list[dict],
    how2sign_selection: dict,
    fleurs_baseline: list[dict],
    how2sign_baseline: list[dict],
    how2sign_partition: dict,
    hashes: dict[str, str],
    expected_how2sign: int,
) -> dict:
    if (
        len(fleurs) != 313
        or len({row["example_id"] for row in fleurs}) != len(fleurs)
        or any(
            row["schema"] != "umi-studio-fleurs-dev-encoder-cache-example/1"
            or row["split"] != "training"
            or row["signer"] != 3
            for row in fleurs
        )
    ):
        raise ValueError("FLEURS training cache is not the 313-example signer-3 set")
    selection_hash = hashes["how2sign_selection"]
    if (
        how2sign_selection.get("schema") != "umi-studio-how2sign-training-aligned/1"
        or len(how2sign_selection["rows"]) != 1024
        or len(how2sign) != expected_how2sign
        or [row["sentence_name"] for row in how2sign]
        != [row["sentence_name"] for row in how2sign_selection["rows"][:expected_how2sign]]
        or any(
            row["schema"] != "umi-studio-how2sign-encoder-cache-example/1"
            or row["split"] != "training"
            or row["selection_sha256"] != selection_hash
            or row["reference"] != selected["reference"]
            for row, selected in zip(
                how2sign, how2sign_selection["rows"][:expected_how2sign], strict=True
            )
        )
    ):
        raise ValueError("aligned How2Sign cache is incomplete or differs from frozen selection")
    if (
        how2sign_partition.get("schema") != "umi-studio-how2sign-validation-partition/1"
        or how2sign_partition.get("confirmation_predictions_inspected") is not False
        or len(how2sign_partition["tuning_sentence_names"]) != 16
        or len(how2sign_partition["confirmation_sentence_names"]) != 42
        or {row["sentence_name"] for row in how2sign}
        & set(how2sign_partition["tuning_sentence_names"])
        or {row["sentence_name"] for row in how2sign}
        & set(how2sign_partition["confirmation_sentence_names"])
    ):
        raise ValueError("How2Sign validation partition is not sealed or overlaps training")
    if (
        len(fleurs_baseline) != 7
        or len(how2sign_baseline) != 16
        or [row["example_id"] for row in how2sign_baseline]
        != how2sign_partition["tuning_sentence_names"]
        or any(row["split"] != "tuning" for row in fleurs_baseline)
        or any(row["split"] != "validation" for row in how2sign_baseline)
    ):
        raise ValueError("tuning baselines are incomplete or use another partition")
    return {
        "schema": "umi-studio-aligned-mixed-plan/1",
        "fleurs_training_index_sha256": hashes["fleurs_index"],
        "how2sign_training_index_sha256": hashes["how2sign_index"],
        "how2sign_selection_sha256": selection_hash,
        "flores2m_selection_sha256": None,
        "fleurs_evaluation_partition_sha256": hashes["fleurs_partition"],
        "how2sign_evaluation_partition_sha256": hashes["how2sign_partition"],
        "fleurs_tuning_baseline_sha256": hashes["fleurs_baseline"],
        "how2sign_tuning_baseline_sha256": hashes["how2sign_baseline"],
        "training_examples": {
            "fleurs": 313,
            "how2sign": expected_how2sign,
            "flores2m": 0,
        },
        "training_policy": (
            "Use only verified public source-training encoder states. No How2Sign or FLEURS "
            "validation, confirmation, test, service, miner or cohort data enters training."
        ),
        "seed": 17,
        "rank": 8,
        "alpha": 16,
        "fleurs_repeat": 1,
        "how2sign_repeat": 1,
        "flores2m_repeat": 1,
        "recipes": [
            {
                "id": "aligned_decoder_all_e1",
                "targets": "decoder_all",
                "epochs": 1,
                "learning_rate": 0.00003,
                "weight_decay": 0.1,
                "label_smoothing": 0.1,
            }
        ],
        "baseline_mean_single_reference_wer": {
            "fleurs": statistics.fmean(row["single_reference_wer"] for row in fleurs_baseline),
            "how2sign": statistics.fmean(row["single_reference_wer"] for row in how2sign_baseline),
        },
        "selection_metric": (
            "Paired mean single-reference WER on the frozen seven FLEURS and 16 How2Sign "
            "tuning clips; report both domains and How2Sign signer means. This is not an "
            "official UMI score."
        ),
        "next_action_rule": (
            "Proceed to sealed confirmation only if How2Sign mean WER falls by at least 0.05, "
            "at least three of five How2Sign signers improve, FLEURS mean WER rises by no "
            "more than 0.03, and the equal-domain mean WER falls by at least 0.025. "
            "Otherwise retain the frozen baseline and scale aligned data before another recipe."
        ),
        "confirmation_policy": (
            "Keep the 42 pre-cut How2Sign clips, 34 full-video How2Sign clips and 46 FLEURS "
            "clips unscored until the candidate and tuning decision are frozen."
        ),
        "miner_priority": (
            "Run offline at low CPU priority and check miner health around every step."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fleurs-index", type=Path, required=True)
    parser.add_argument("--how2sign-index", type=Path, required=True)
    parser.add_argument("--how2sign-selection", type=Path, required=True)
    parser.add_argument("--fleurs-partition", type=Path, required=True)
    parser.add_argument("--how2sign-partition", type=Path, required=True)
    parser.add_argument("--fleurs-baseline", type=Path, required=True)
    parser.add_argument("--how2sign-baseline", type=Path, required=True)
    parser.add_argument("--expected-how2sign", type=int, default=168)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.expected_how2sign < 1:
        parser.error("expected aligned training count must be positive")
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

    plan = build_plan(
        read_jsonl(args.fleurs_index),
        read_jsonl(args.how2sign_index),
        json.loads(args.how2sign_selection.read_text()),
        read_jsonl(args.fleurs_baseline),
        read_jsonl(args.how2sign_baseline),
        json.loads(args.how2sign_partition.read_text()),
        hashes,
        args.expected_how2sign,
    )
    if args.output.exists():
        if json.loads(args.output.read_text()) != plan:
            raise ValueError("frozen aligned-mixed plan differs from current inputs")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as output:
            json.dump(plan, output, sort_keys=True, indent=2)
            output.write("\n")
    print(f"frozen {313 + args.expected_how2sign}-step aligned-mixed training plan")


if __name__ == "__main__":
    main()
