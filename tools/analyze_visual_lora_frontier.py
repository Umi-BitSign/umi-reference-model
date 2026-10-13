"""Measure the best possible WER obtainable from existing candidate hypotheses."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

from benchmark_fleurs_dev import sha256

ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def build_analysis(baseline: list[dict], candidates: dict[str, list[dict]]) -> dict:
    if not baseline or not candidates:
        raise ValueError("baseline and at least one candidate are required")
    baseline_by_id = {row["example_id"]: row for row in baseline}
    if len(baseline_by_id) != len(baseline):
        raise ValueError("baseline example IDs are not unique")
    candidates_by_name: dict[str, dict[str, dict]] = {}
    for name, rows in candidates.items():
        by_id = {row["example_id"]: row for row in rows}
        if len(by_id) != len(rows) or set(by_id) != set(baseline_by_id):
            raise ValueError(f"candidate {name} has different example IDs")
        for example_id, row in by_id.items():
            reference_fields = ("reference", "signer", "split")
            if any(
                row.get(field) != baseline_by_id[example_id].get(field)
                for field in reference_fields
            ):
                raise ValueError(f"candidate {name} has different inputs")
        candidates_by_name[name] = by_id

    records = []
    source_wers: dict[str, list[float]] = defaultdict(list)
    source_wers["baseline"] = []
    oracle_signer_wers: dict[str, list[float]] = defaultdict(list)
    for row in baseline:
        example_id = row["example_id"]
        choices = {"baseline": float(row["single_reference_wer"])}
        source_wers["baseline"].append(choices["baseline"])
        for name, by_id in candidates_by_name.items():
            choices[name] = float(by_id[example_id]["single_reference_wer"])
            source_wers[name].append(choices[name])
        oracle_wer = min(choices.values())
        selected = sorted(name for name, wer in choices.items() if wer == oracle_wer)
        signer = str(row["signer"])
        oracle_signer_wers[signer].append(oracle_wer)
        records.append(
            {
                "example_id": example_id,
                "signer": row["signer"],
                "wers": choices,
                "oracle_wer": oracle_wer,
                "oracle_sources": selected,
            }
        )
    means = {
        name: sum(wers) / len(wers) for name, wers in source_wers.items()
    }
    oracle_mean = sum(row["oracle_wer"] for row in records) / len(records)
    required_mean = means["baseline"] - 0.05
    return {
        "examples": len(records),
        "mean_single_reference_wer": means,
        "oracle_mean_single_reference_wer": oracle_mean,
        "oracle_signer_means": {
            signer: sum(wers) / len(wers)
            for signer, wers in sorted(oracle_signer_wers.items())
        },
        "required_how2sign_mean_wer": required_mean,
        "existing_hypothesis_oracle_passes_gate": oracle_mean <= required_mean,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        action="append",
        nargs=2,
        metavar=("NAME", "PATH"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(ROOT / "runs"):
        parser.error("frontier analysis needs a new run-ledger output")
    candidate_paths = {name: Path(path) for name, path in args.candidate}
    if len(candidate_paths) != len(args.candidate):
        parser.error("candidate names must be unique")
    analysis = build_analysis(
        read_jsonl(args.baseline),
        {name: read_jsonl(path) for name, path in candidate_paths.items()},
    )
    analysis.update(
        schema="umi-studio-visual-lora-frontier-analysis/1",
        baseline_results_sha256=sha256(args.baseline),
        candidate_results_sha256={
            name: sha256(path) for name, path in candidate_paths.items()
        },
        analysis_source_sha256=sha256(Path(__file__)),
        confirmation_evaluated=False,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".partial.json")
    try:
        with temporary.open("x", encoding="utf-8") as output:
            json.dump(analysis, output, sort_keys=True, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        temporary.rename(args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(
        json.dumps(
            {
                "oracle_mean_single_reference_wer": analysis[
                    "oracle_mean_single_reference_wer"
                ],
                "required_how2sign_mean_wer": analysis[
                    "required_how2sign_mean_wer"
                ],
                "existing_hypothesis_oracle_passes_gate": analysis[
                    "existing_hypothesis_oracle_passes_gate"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
