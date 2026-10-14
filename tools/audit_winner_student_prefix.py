"""Prove rejection from nonnegative WER bounds without claiming a complete score."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from benchmark_fleurs_dev import distance, normalize, sha256


def quality_bounds(evaluation, control, partial):
    before = {row["example_id"]: row for row in control}
    inventory = {row["example_id"]: row for row in evaluation}
    if (
        len(evaluation) != 142
        or len(inventory) != 142
        or len(control) != 142
        or set(before) != set(inventory)
        or not partial
        or len({row["example_id"] for row in partial}) != len(partial)
    ):
        raise ValueError("partial WER bound coverage differs")
    for row in partial:
        expected = inventory.get(row["example_id"])
        if (
            expected is None
            or row["role"] != "open_development"
            or row["training_allowed"] is not False
            or expected["training_allowed"] is not False
            or any(row[key] != expected[key] for key in ("domain", "signer", "source_group"))
        ):
            raise ValueError("partial WER bound identity/role differs")
        words = len(normalize(expected["reference"]))
        edits = distance(normalize(expected["reference"]), normalize(row["hypothesis"]))
        if (
            not words
            or row["reference_words"] != words
            or row["edits"] != edits
            or not math.isfinite(row["wer"])
            or row["wer"] != edits / words
        ):
            raise ValueError("partial WER bound requires recomputed valid nonnegative errors")
    output = {}
    for domain in ("how2sign", "fleurs"):
        full = [row for row in evaluation if row["domain"] == domain]
        observed = [row for row in partial if row["domain"] == domain]
        baseline = [before[row["example_id"]] for row in full]
        words = sum(len(normalize(row["reference"])) for row in full)
        if (
            not full
            or not words
            or any(
                row["reference_words"] != len(normalize(inventory[row["example_id"]]["reference"]))
                or not math.isfinite(row["wer"])
                or row["edits"] < 0
                or row["wer"] != row["edits"] / row["reference_words"]
                for row in baseline
            )
        ):
            raise ValueError("partial WER bound fixed-control metrics differ")
        mean = statistics.fmean(row["wer"] for row in baseline)
        weighted = sum(row["edits"] for row in baseline) / words
        lower_mean = sum(row["wer"] for row in observed) / len(full)
        lower_weighted = sum(row["edits"] for row in observed) / words
        limit = mean * (0.95 if domain == "how2sign" else 1)
        output[domain] = {
            "full_clip_count": len(full),
            "completed_prefix_clips": len(observed),
            "winner_mean_wer": mean,
            "unchanged_gate_mean_wer_limit": limit,
            "mean_wer_lower_bound_even_if_remaining_perfect": lower_mean,
            "winner_word_weighted_wer": weighted,
            "word_weighted_wer_lower_bound": lower_weighted,
            "gate_impossible": lower_mean > limit + 1e-12 or lower_weighted > weighted + 1e-12,
        }
    return {
        "verified_prefix_clips": len(partial),
        "domains": output,
        "full_quality_gate_mathematically_impossible": any(
            row["gate_impossible"] for row in output.values()
        ),
        "complete_benchmark_score": False,
        "reason": "Unobserved word errors are nonnegative. Perfect remaining predictions "
        "are the most optimistic possible completion; an already exceeded frozen mean "
        "or weighted gate cannot recover. This is a rejection proof, not a full WER estimate.",
    }


def main():
    from evaluate_tiny_rgb_winner_cpu import checked_plan
    from train_tiny_rgb_capacity_cpu import storage

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--prefix", type=int, default=14)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(
        Path(__file__).resolve().parents[1] / "runs"
    ):
        raise ValueError("partial WER audit requires new private runs/ output")
    _, parent = checked_plan(args.plan)
    predictions = [json.loads(line) for line in args.predictions.read_text().splitlines()]
    if args.prefix < 1 or len(predictions) < args.prefix:
        raise ValueError("partial WER audit prefix unavailable")
    result = quality_bounds(
        parent["evaluation"], parent["winner_control"], predictions[: args.prefix]
    )
    result.update(
        {
            "plan_sha256": sha256(args.plan),
            "prediction_ledger_sha256": sha256(args.predictions),
            "audit_source_sha256": sha256(Path(__file__)),
            "optimizer_updates": 0,
            "promotion_authorized": False,
            "miner_modified": False,
        }
    )
    storage(len(json.dumps(result).encode()) + 1024**2)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"audit_sha256": sha256(args.output), **result}), flush=True)


if __name__ == "__main__":
    main()
