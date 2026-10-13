"""Summarize the fixed candidate on the unseen internal expansion holdout."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from collections import defaultdict
from pathlib import Path

from benchmark_fleurs_dev import sha256
from freeze_c5_stem_scale_plan import ROOT
from run_c5_stem_tuning import read_jsonl

CONFIRMATION = ROOT / "runs/temporal-stride-v1/confirmation-decision.json"


def summarize_holdout(baseline: list[dict], candidate: list[dict]) -> dict:
    if not baseline or len(baseline) != len(candidate):
        raise ValueError("internal holdout counts differ")
    by_signer = defaultdict(lambda: {"baseline": [], "candidate": []})
    for first, second in zip(baseline, candidate, strict=True):
        if (
            first["example_id"] != second["example_id"]
            or first["reference"] != second["reference"]
            or first["index_sha256"] != second["index_sha256"]
            or first["checkpoint_sha256"] != second["checkpoint_sha256"]
            or first.get("adapter_sha256") is None
            or second.get("adapter_sha256") is None
            or first.get("temporal_stride") != 1
            or second.get("temporal_stride") != 1
            or not math.isfinite(first["single_reference_wer"])
            or not math.isfinite(second["single_reference_wer"])
        ):
            raise ValueError("internal holdout paired inputs differ")
        signer = str(first["signer"])
        by_signer[signer]["baseline"].append(first["single_reference_wer"])
        by_signer[signer]["candidate"].append(second["single_reference_wer"])
    baseline_mean = statistics.fmean(row["single_reference_wer"] for row in baseline)
    candidate_mean = statistics.fmean(row["single_reference_wer"] for row in candidate)
    return {
        "examples": len(baseline),
        "baseline_mean_wer": baseline_mean,
        "candidate_mean_wer": candidate_mean,
        "candidate_minus_baseline_mean_wer": candidate_mean - baseline_mean,
        "signer_means": {
            signer: {
                "baseline_mean_wer": statistics.fmean(values["baseline"]),
                "candidate_mean_wer": statistics.fmean(values["candidate"]),
                "examples": len(values["baseline"]),
            }
            for signer, values in sorted(by_signer.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--baseline-results", type=Path, required=True)
    parser.add_argument("--candidate-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.output.exists()
        or not args.output.resolve().is_relative_to((ROOT / "runs").resolve())
        or not all(
            path.is_file()
            for path in (args.selection, args.baseline_results, args.candidate_results)
        )
    ):
        parser.error("holdout decision needs fixed results and a new runs output")
    selection = json.loads(args.selection.read_text())
    confirmation = json.loads(CONFIRMATION.read_text())
    baseline = read_jsonl(args.baseline_results)
    candidate = read_jsonl(args.candidate_results)
    if (
        selection.get("schema") != "umi-studio-c5-how2sign-expansion-holdout/1"
        or selection.get("purpose")
        != "internal unseen-training-split diagnostic; not promotion confirmation"
        or len(baseline) != selection.get("examples")
        or len(candidate) != selection.get("examples")
        or {row.get("index_sha256") for row in baseline + candidate}
        != {selection.get("index_sha256")}
        or {row.get("adapter_sha256") for row in baseline}
        != {confirmation.get("candidate_adapter_sha256")}
        or {row.get("adapter_sha256") for row in candidate}
        != {selection.get("candidate_adapter_sha256")}
        or {row.get("temporal_stride") for row in baseline + candidate} != {1}
        or {
            row.get("split") for row in baseline + candidate
        }
        != {"how2sign_expansion_holdout"}
    ):
        raise ValueError("internal expansion holdout inputs differ")
    summary = summarize_holdout(baseline, candidate)
    improved = sum(
        second["single_reference_wer"] < first["single_reference_wer"]
        for first, second in zip(baseline, candidate, strict=True)
    )
    regressed = sum(
        second["single_reference_wer"] > first["single_reference_wer"]
        for first, second in zip(baseline, candidate, strict=True)
    )
    baseline_words = sum(row["reference_word_count"] for row in baseline)
    candidate_words = sum(row["reference_word_count"] for row in candidate)
    if baseline_words != candidate_words:
        raise ValueError("internal holdout reference word counts differ")
    baseline_weighted = sum(row["single_reference_word_edits"] for row in baseline) / baseline_words
    candidate_weighted = sum(row["single_reference_word_edits"] for row in candidate) / candidate_words
    improved_signers = sum(
        row["candidate_mean_wer"] < row["baseline_mean_wer"]
        for row in summary["signer_means"].values()
    )
    promising = (
        summary["candidate_minus_baseline_mean_wer"] < 0
        and candidate_weighted < baseline_weighted
        and improved_signers >= 5
    )
    result = {
        "schema": "umi-studio-c5-how2sign-expansion-holdout-decision/1",
        "selection_sha256": sha256(args.selection),
        "baseline_results_sha256": sha256(args.baseline_results),
        "candidate_results_sha256": sha256(args.candidate_results),
        "candidate_adapter_sha256": selection["candidate_adapter_sha256"],
        "summary": summary,
        "paired_examples_improved": improved,
        "paired_examples_tied": len(baseline) - improved - regressed,
        "paired_examples_regressed": regressed,
        "improved_signers": improved_signers,
        "baseline_word_weighted_wer": baseline_weighted,
        "candidate_word_weighted_wer": candidate_weighted,
        "candidate_minus_baseline_word_weighted_wer": (
            candidate_weighted - baseline_weighted
        ),
        "decision": "promising_internal_only" if promising else "reject_internal_candidate",
        "promotion_eligible": False,
        "next_action": (
            "A favorable result can justify a genuinely fresh held-out evaluation, but this "
            "official-training-split diagnostic cannot replace confirmation."
        ),
        "decision_source_sha256": sha256(Path(__file__)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".partial.json")
    try:
        with temporary.open("x", encoding="utf-8") as output:
            json.dump(result, output, sort_keys=True, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        temporary.rename(args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(result["decision"])


if __name__ == "__main__":
    main()
