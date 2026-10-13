"""Summarize a matched FSboard validation pilot without touching the miner.

CER uses the UMI 0.1 text normalization but only one public FSboard reference.
The signer bootstrap describes this small development sample, not UMI quality.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

from benchmark_fleurs_dev import sha256
from fetch_fsboard_validation import validate_selection


def _load_unique(paths: list[Path], schema: str, selection_hash: str) -> dict[str, dict]:
    records = {}
    for path in paths:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row["schema"] != schema or row["selection_sha256"] != selection_hash:
                raise ValueError("FSboard evaluation inputs have mixed identity")
            if (
                schema == "umi-studio-fsboard-coreml-diagnostic/2"
                and row["status"] != "right_hand_evaluated"
            ):
                continue
            name = row["clip_filename"]
            if name in records:
                raise ValueError("FSboard evaluation inputs contain duplicate clips")
            records[name] = row
    return records


def summarize(
    selection: dict,
    candidate: dict[str, dict],
    baseline: dict[str, dict],
    *,
    seed: int = 17,
    bootstrap_draws: int = 10_000,
) -> dict:
    rows = validate_selection(selection)
    names = {row["clip_filename"] for row in rows}
    if names != candidate.keys() or names != baseline.keys():
        raise ValueError("matched FSboard comparison must cover every selected clip")
    pairs = []
    by_signer = {}
    for row in rows:
        name = row["clip_filename"]
        left = candidate[name]
        right = baseline[name]
        if (
            left["status"] != "right_hand_evaluated"
            or left["reference"] != row["reference"]
            or right["reference"] != row["reference"]
            or left["signer_id"] != row["signer_id"]
        ):
            raise ValueError("FSboard clip results do not match the frozen selection")
        candidate_cer = float(left["single_reference_cer"])
        baseline_cer = float(right["single_reference_cer"])
        if candidate_cer < 0 or baseline_cer < 0:
            raise ValueError("CER cannot be negative")
        improvement = max(0.0, 1.0 - candidate_cer) - max(0.0, 1.0 - baseline_cer)
        item = {
            "signer_id": row["signer_id"],
            "candidate_cer": candidate_cer,
            "baseline_cer": baseline_cer,
            "clamped_similarity_improvement": improvement,
        }
        pairs.append(item)
        by_signer.setdefault(row["signer_id"], []).append(item)
    if len(by_signer) != selection["signer_count"] or any(
        len(items) != selection["clips_per_signer"] for items in by_signer.values()
    ):
        raise ValueError("FSboard signer balance was lost")
    signer_means = [
        statistics.mean(item["clamped_similarity_improvement"] for item in items)
        for _, items in sorted(by_signer.items())
    ]
    generator = random.Random(seed)
    bootstrap = sorted(
        statistics.mean(generator.choices(signer_means, k=len(signer_means)))
        for _ in range(bootstrap_draws)
    )
    return {
        "clips": len(pairs),
        "signers": len(by_signer),
        "candidate_mean_cer": statistics.mean(item["candidate_cer"] for item in pairs),
        "baseline_mean_cer": statistics.mean(item["baseline_cer"] for item in pairs),
        "candidate_mean_clamped_similarity": statistics.mean(
            max(0.0, 1.0 - item["candidate_cer"]) for item in pairs
        ),
        "baseline_mean_clamped_similarity": statistics.mean(
            max(0.0, 1.0 - item["baseline_cer"]) for item in pairs
        ),
        "candidate_wins": sum(item["candidate_cer"] < item["baseline_cer"] for item in pairs),
        "ties": sum(item["candidate_cer"] == item["baseline_cer"] for item in pairs),
        "signer_bootstrap_seed": seed,
        "signer_bootstrap_draws": bootstrap_draws,
        "clamped_similarity_improvement_95pct_signer_bootstrap": [
            bootstrap[int(0.025 * bootstrap_draws)],
            bootstrap[int(0.975 * bootstrap_draws)],
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--candidate-right", type=Path, required=True)
    parser.add_argument("--candidate-mirrored-left", type=Path, required=True)
    parser.add_argument("--baseline-right", type=Path, required=True)
    parser.add_argument("--baseline-left", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("summary output already exists")
    selection = json.loads(args.selection.read_text())
    selection_hash = sha256(args.selection)
    candidate_paths = [args.candidate_right, args.candidate_mirrored_left]
    baseline_paths = [args.baseline_right, args.baseline_left]
    candidate = _load_unique(
        candidate_paths, "umi-studio-fsboard-coreml-diagnostic/2", selection_hash
    )
    baseline = _load_unique(
        baseline_paths, "umi-studio-fsboard-baseline-diagnostic/1", selection_hash
    )
    for candidate_path, baseline_path, mirrored in (
        (args.candidate_right, args.baseline_right, False),
        (args.candidate_mirrored_left, args.baseline_left, True),
    ):
        candidate_hash = sha256(candidate_path)
        for line in baseline_path.read_text().splitlines():
            row = json.loads(line)
            if (
                row["candidate_results_sha256"] != candidate_hash
                or candidate[row["clip_filename"]].get("mirror_input", False) != mirrored
            ):
                raise ValueError("baseline result is not paired with its candidate input")
    result = {
        "schema": "umi-studio-fsboard-matched-pilot/1",
        "source_split": "validation",
        "test_used": False,
        "selection_sha256": selection_hash,
        "candidate_results_sha256": [sha256(path) for path in candidate_paths],
        "baseline_results_sha256": [sha256(path) for path in baseline_paths],
        "metric": "single-reference UMI-normalized grapheme CER",
        "model_promotion": False,
        "summary": summarize(selection, candidate, baseline),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2, sort_keys=True)
        output.write("\n")
    print("summarized matched FSboard validation pilot")


if __name__ == "__main__":
    main()
