"""Compare an offline ByT5 candidate with pinned TFLite on matched FSboard clips.

Both inputs are public-validation diagnostics. The comparison does not authorize
release of either model or change the live miner.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

from benchmark_fleurs_dev import distance, sha256
from evaluate_fsboard_coreml import comparable_units
from evaluate_fsboard_tflite import MODEL_SHA256 as TFLITE_SHA256

ROOT = Path(__file__).resolve().parents[1]


def checked_rows(path: Path, schema: str) -> dict[str, dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    by_name = {row["clip_filename"]: row for row in rows}
    if len(rows) != 60 and schema == "umi-studio-fsboard-byt5-validation/1":
        raise ValueError("ByT5 result needs all 60 validation rows")
    if len(rows) != len(by_name) or any(row.get("schema") != schema for row in rows):
        raise ValueError("FSboard result has duplicate names or an unexpected schema")
    for row in rows:
        reference = comparable_units(row["reference"])
        hypothesis = comparable_units(row["hypothesis"])
        cer = distance(reference, hypothesis) / max(1, len(reference))
        if abs(cer - row["single_reference_cer"]) > 1e-12:
            raise ValueError("FSboard stored CER differs from its transcript")
    return by_name


def paired_metrics(candidate: dict[str, dict], comparator: dict[str, dict]) -> dict:
    if len(candidate) != 60 or candidate.keys() != comparator.keys():
        raise ValueError("FSboard candidate and comparator must cover the same 60 clips")
    wins = ties = losses = 0
    candidate_errors = []
    comparator_errors = []
    by_signer = defaultdict(lambda: [[], []])
    by_hand = defaultdict(lambda: [[], []])
    candidate_exact = comparator_exact = 0
    for name, row in candidate.items():
        baseline = comparator[name]
        if any(row[key] != baseline[key] for key in (
            "reference", "signer_id", "selection_sha256", "landmarks_sha256"
        )):
            raise ValueError(f"FSboard candidate and comparator differ in source: {name}")
        left = row["single_reference_cer"]
        right = baseline["single_reference_cer"]
        candidate_errors.append(left)
        comparator_errors.append(right)
        by_signer[row["signer_id"]][0].append(left)
        by_signer[row["signer_id"]][1].append(right)
        by_hand[row["hand_dominance"]][0].append(left)
        by_hand[row["hand_dominance"]][1].append(right)
        wins += left < right
        ties += left == right
        losses += left > right
        reference = comparable_units(row["reference"])
        candidate_exact += comparable_units(row["hypothesis"]) == reference
        comparator_exact += comparable_units(baseline["hypothesis"]) == reference
    return {
        "clips": len(candidate_errors),
        "candidate_mean_cer": sum(candidate_errors) / len(candidate_errors),
        "tflite_mean_cer": sum(comparator_errors) / len(comparator_errors),
        "mean_cer_delta_candidate_minus_tflite": (
            sum(candidate_errors) - sum(comparator_errors)
        ) / len(candidate_errors),
        "candidate_cer_wins": wins,
        "cer_ties": ties,
        "tflite_cer_wins": losses,
        "candidate_exact_transcriptions": candidate_exact,
        "tflite_exact_transcriptions": comparator_exact,
        "by_signer": {
            signer: {
                "clips": len(values[0]),
                "candidate_mean_cer": sum(values[0]) / len(values[0]),
                "tflite_mean_cer": sum(values[1]) / len(values[1]),
            }
            for signer, values in sorted(by_signer.items())
        },
        "by_hand_dominance": {
            hand: {
                "clips": len(values[0]),
                "candidate_mean_cer": sum(values[0]) / len(values[0]),
                "tflite_mean_cer": sum(values[1]) / len(values[1]),
            }
            for hand, values in sorted(by_hand.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-results", type=Path, required=True)
    parser.add_argument("--candidate-summary", type=Path, required=True)
    parser.add_argument("--tflite-pilot", type=Path, required=True)
    parser.add_argument("--tflite-followup", type=Path, required=True)
    parser.add_argument("--tflite-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(ROOT / "runs"):
        parser.error("comparison output must be new and under runs/")
    candidate_summary = json.loads(args.candidate_summary.read_text())
    tflite_summary = json.loads(args.tflite_summary.read_text())
    if (
        candidate_summary.get("schema") != "umi-studio-fsboard-byt5-validation-summary/1"
        or candidate_summary.get("results_sha256") != sha256(args.candidate_results)
        or tflite_summary.get("schema") != "umi-studio-fsboard-tflite-parity/1"
        or tflite_summary.get("input_layout") != "training_tensor"
        or tflite_summary.get("model_sha256") != TFLITE_SHA256
        or tflite_summary.get("source_sha256", {}).get("pilot", {}).get("tflite")
        != sha256(args.tflite_pilot)
        or tflite_summary.get("source_sha256", {}).get("followup", {}).get("tflite")
        != sha256(args.tflite_followup)
    ):
        raise ValueError("FSboard validation summaries differ from their result files")
    candidate = checked_rows(args.candidate_results, "umi-studio-fsboard-byt5-validation/1")
    pilot = checked_rows(args.tflite_pilot, "umi-studio-fsboard-tflite-diagnostic/2")
    followup = checked_rows(args.tflite_followup, "umi-studio-fsboard-tflite-diagnostic/2")
    if len(pilot) != 30 or len(followup) != 30 or pilot.keys() & followup.keys():
        raise ValueError("TFLite comparator needs two disjoint 30-clip selections")
    if any(row.get("input_layout") != "training_tensor" for row in (*pilot.values(), *followup.values())):
        raise ValueError("TFLite comparator has the wrong input layout")
    metrics = paired_metrics(candidate, pilot | followup)
    if (
        abs(metrics["candidate_mean_cer"] - candidate_summary["single_reference_mean_cer"]) > 1e-12
        or abs(metrics["tflite_mean_cer"] - tflite_summary["combined"]["tflite_mean_cer"]) > 1e-12
    ):
        raise ValueError("FSboard paired CER differs from pinned summaries")
    result = {
        "schema": "umi-studio-fsboard-byt5-paired-comparison/1",
        "split": "FSboard daun_v3 public validation, 60 clips",
        "test_used": False,
        "candidate_checkpoint_sha256": candidate_summary["checkpoint_sha256"],
        "candidate_summary_sha256": sha256(args.candidate_summary),
        "tflite_summary_sha256": sha256(args.tflite_summary),
        "tflite_model_sha256": TFLITE_SHA256,
        "comparison_script_sha256": sha256(Path(__file__)),
        "metrics": metrics,
        "candidate_promoted": False,
        "live_miner_changed": False,
        "limits": [
            "This is a public single-reference development diagnostic, not a UMI score.",
            "Both selections contain the same 15 signers and informed earlier specialist work.",
            "The TFLite file's redistribution rights and native task routing remain unresolved.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({key: metrics[key] for key in (
        "candidate_mean_cer", "tflite_mean_cer", "candidate_cer_wins",
        "tflite_cer_wins", "cer_ties"
    )}, sort_keys=True))


if __name__ == "__main__":
    main()
