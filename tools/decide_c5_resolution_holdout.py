"""Freeze the internal-holdout decision for a C5 visual-crop candidate."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from benchmark_fleurs_dev import sha256


def load(path: Path) -> dict[str, dict]:
    rows = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        name = row["example_id"]
        if name in rows:
            raise ValueError("resolution holdout results contain duplicates")
        rows[name] = row
    if len(rows) != 134:
        raise ValueError("resolution holdout results must contain 134 examples")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--native-validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("decision output already exists")
    baseline = load(args.baseline)
    candidate = load(args.candidate)
    validation = [json.loads(line) for line in args.native_validation.read_text().splitlines()]
    if (
        set(baseline) != set(candidate)
        or len(validation) != 10
        or any(
            baseline[row["example_id"]]["hypothesis"] != row["hypothesis"]
            for row in validation
        )
    ):
        raise ValueError("raw native inference does not reproduce the cached control")
    signer_rows = defaultdict(list)
    for name, base in baseline.items():
        cand = candidate[name]
        if (
            base["reference"] != cand["reference"]
            or base["signer"] != cand["signer"]
            or cand.get("visual_crop_size") != 168
            or cand.get("source_short_side") is not None
        ):
            raise ValueError(f"paired holdout inputs differ: {name}")
        signer_rows[str(base["signer"])].append((base, cand))

    def mean(rows, key):
        return sum(row[key] for row in rows) / len(rows)

    base_rows = list(baseline.values())
    cand_rows = list(candidate.values())
    deltas = [
        candidate[name]["single_reference_wer"] - row["single_reference_wer"]
        for name, row in baseline.items()
    ]
    base_mean = mean(base_rows, "single_reference_wer")
    cand_mean = mean(cand_rows, "single_reference_wer")
    base_weighted = sum(row["single_reference_word_edits"] for row in base_rows) / sum(
        row["reference_word_count"] for row in base_rows
    )
    cand_weighted = sum(row["single_reference_word_edits"] for row in cand_rows) / sum(
        row["reference_word_count"] for row in cand_rows
    )
    signer_summary = {}
    for signer, pairs in sorted(signer_rows.items(), key=lambda item: int(item[0])):
        signer_summary[signer] = {
            "examples": len(pairs),
            "baseline_mean_wer": mean([pair[0] for pair in pairs], "single_reference_wer"),
            "candidate_mean_wer": mean([pair[1] for pair in pairs], "single_reference_wer"),
        }
    quality_pass = (
        cand_mean - base_mean <= 0.01
        and cand_weighted - base_weighted <= 0.01
        and sum(delta < 0 for delta in deltas) >= sum(delta > 0 for delta in deltas)
    )
    result = {
        "schema": "umi-studio-c5-resolution-holdout-decision/1",
        "split": "internal_training_holdout",
        "promotion_eligible": False,
        "baseline_results_sha256": sha256(args.baseline),
        "candidate_results_sha256": sha256(args.candidate),
        "native_validation_sha256": sha256(args.native_validation),
        "native_validation_exact_hypotheses": 10,
        "candidate_visual_crop_size": 168,
        "candidate_patch_tokens_per_stream_frame": 144,
        "baseline_mean_wer": base_mean,
        "candidate_mean_wer": cand_mean,
        "candidate_minus_baseline_mean_wer": cand_mean - base_mean,
        "baseline_word_weighted_wer": base_weighted,
        "candidate_word_weighted_wer": cand_weighted,
        "candidate_minus_baseline_word_weighted_wer": cand_weighted - base_weighted,
        "paired_examples_improved": sum(delta < 0 for delta in deltas),
        "paired_examples_tied": sum(delta == 0 for delta in deltas),
        "paired_examples_regressed": sum(delta > 0 for delta in deltas),
        "signer_means": signer_summary,
        "candidate_active_seconds": sum(row["active_seconds"] for row in cand_rows),
        "quality_gate_passed": quality_pass,
        "decision": (
            "retain_for_crop_adaptation_training"
            if not quality_pass
            else "retain_runtime_candidate"
        ),
        "next_action": (
            (
                "Freeze a standalone candidate bundle and seek genuinely fresh held-out "
                "evidence; do not alter the confirmed production bundle yet."
            )
            if quality_pass
            else (
                "Train crop-resolution robustness on training-only clips before another "
                "runtime evaluation; do not alter the confirmed production bundle."
            )
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
