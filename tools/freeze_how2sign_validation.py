"""Freeze the How2Sign tuning set and untouched confirmation set."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

from benchmark_fleurs_dev import sha256


def build_partition(
    selection: dict,
    index: list[dict],
    baseline: list[dict],
    selection_hash: str,
    index_hash: str,
    baseline_hash: str,
) -> dict:
    if (
        selection.get("schema") != "umi-studio-how2sign-validation-selection/1"
        or selection.get("split") != "validation"
        or selection.get("training_excluded") is not True
    ):
        raise ValueError("selection is not protected How2Sign validation")
    rows = selection["rows"]
    names = [row["sentence_name"] for row in rows]
    if len(set(names)) != len(names) or len({row["video_id"] for row in rows}) != len(rows):
        raise ValueError("validation selection repeats a sentence or source video")
    tuning = names[:16]
    confirmation = names[16:]
    if len(tuning) != 16 or len(confirmation) < 1:
        raise ValueError("validation selection is too small for the frozen partition")
    if (
        len(index) != len(tuning)
        or len(baseline) != len(tuning)
        or [row["sentence_name"] for row in index] != tuning
        or [row["example_id"] for row in baseline] != tuning
        or any(
            row["schema"] != "umi-studio-how2sign-encoder-cache-example/1"
            or row["split"] != "validation"
            or row["selection_sha256"] != selection_hash
            for row in index
        )
        or any(
            row["schema"] != "umi-studio-how2sign-cached-evaluation-example/1"
            or row["split"] != "validation"
            or row["index_sha256"] != index_hash
            or row["selection_sha256"] != selection_hash
            for row in baseline
        )
    ):
        raise ValueError("tuning cache or baseline is incomplete or uses other inputs")
    if any(row["single_reference_wer"] < 0 for row in baseline):
        raise ValueError("baseline WER is invalid")
    return {
        "schema": "umi-studio-how2sign-validation-partition/1",
        "selection_sha256": selection_hash,
        "tuning_index_sha256": index_hash,
        "tuning_baseline_results_sha256": baseline_hash,
        "rule": (
            "First 16 clips in frozen selection order are tuning; remaining clips are sealed "
            "confirmation. Source videos are distinct across both sets."
        ),
        "tuning_sentence_names": tuning,
        "confirmation_sentence_names": confirmation,
        "tuning_signer_counts": dict(Counter(str(row["signer"]) for row in rows[:16])),
        "confirmation_signer_counts": dict(Counter(str(row["signer"]) for row in rows[16:])),
        "tuning_baseline_mean_single_reference_wer": statistics.fmean(
            row["single_reference_wer"] for row in baseline
        ),
        "confirmation_predictions_inspected": False,
        "confirmation_policy": (
            "Do not encode or score confirmation until the candidate, recipe and tuning "
            "decision are frozen."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--tuning-index", type=Path, required=True)
    parser.add_argument("--baseline-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text())
    index = [json.loads(line) for line in args.tuning_index.read_text().splitlines()]
    baseline = [json.loads(line) for line in args.baseline_results.read_text().splitlines()]
    partition = build_partition(
        selection,
        index,
        baseline,
        sha256(args.selection),
        sha256(args.tuning_index),
        sha256(args.baseline_results),
    )
    if args.output.exists():
        if json.loads(args.output.read_text()) != partition:
            raise ValueError("frozen How2Sign validation partition differs from current inputs")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as output:
            json.dump(partition, output, sort_keys=True, indent=2)
            output.write("\n")
    print(
        f"frozen {len(partition['tuning_sentence_names'])} tuning and "
        f"{len(partition['confirmation_sentence_names'])} confirmation clips"
    )


if __name__ == "__main__":
    main()
