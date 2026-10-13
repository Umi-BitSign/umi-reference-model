"""Freeze source-stratified train/holdout rows for a compact C5 text corrector."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

from benchmark_fleurs_dev import sha256

ROOT = Path(__file__).resolve().parents[1]
SEED = 131
RECIPE = {
    "id": "c5-t5-small-text-corrector-v1",
    "pretrained_model": "artifacts/t5-small-df1b051c",
    "epochs": 4,
    "batch_size": 4,
    "gradient_accumulation": 4,
    "learning_rate": 5e-5,
    "weight_decay": 0.01,
    "max_input_tokens": 96,
    "max_target_tokens": 96,
    "identity_examples_per_correction": 1,
    "holdout_fraction": 0.2,
    "device": "mps",
    "mps_active_fraction": 0.5,
    "generation_beams": 4,
}


def stable_key(row: dict) -> str:
    return hashlib.sha256(
        f"{SEED}\0{row['source']}\0{row['example_id']}".encode()
    ).hexdigest()


def split_rows(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["source"]].append(row)
    train, holdout = [], []
    for source, source_rows in sorted(grouped.items()):
        ordered = sorted(source_rows, key=stable_key)
        holdout_count = max(1, round(len(ordered) * RECIPE["holdout_fraction"]))
        holdout.extend(ordered[:holdout_count])
        train.extend(ordered[holdout_count:])
    train.sort(key=stable_key)
    holdout.sort(key=stable_key)
    return train, holdout


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
        output.flush()
        os.fsync(output.fileno())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if (
        not args.pairs.is_file()
        or args.output_dir.exists()
        or not args.output_dir.resolve().is_relative_to((ROOT / "runs").resolve())
    ):
        parser.error("freeze requires completed pairs and a new runs/ directory")
    rows = [json.loads(line) for line in args.pairs.read_text().splitlines()]
    keys = {(row["source"], row["example_id"]) for row in rows}
    if (
        len(rows) != 1_024
        or len(keys) != len(rows)
        or any(row.get("schema") != "umi-studio-c5-text-corrector-pair/1" for row in rows)
    ):
        raise ValueError("text-corrector pairs are incomplete or invalid")
    train, holdout = split_rows(rows)
    if set((row["source"], row["example_id"]) for row in train) & set(
        (row["source"], row["example_id"]) for row in holdout
    ):
        raise ValueError("text-corrector split overlaps")
    args.output_dir.mkdir(parents=True)
    train_path = args.output_dir / "train.jsonl"
    holdout_path = args.output_dir / "holdout.jsonl"
    write_rows(train_path, train)
    write_rows(holdout_path, holdout)
    plan = {
        "schema": "umi-studio-c5-text-corrector-plan/1",
        "objective": "repair systematic C5 language errors without changing the visual model",
        "seed": SEED,
        "recipe": RECIPE,
        "pairs": str(args.pairs.resolve().relative_to(ROOT)),
        "pairs_sha256": sha256(args.pairs),
        "train_sha256": sha256(train_path),
        "holdout_sha256": sha256(holdout_path),
        "train_examples": len(train),
        "holdout_examples": len(holdout),
        "train_source_counts": dict(sorted(Counter(row["source"] for row in train).items())),
        "holdout_source_counts": dict(
            sorted(Counter(row["source"] for row in holdout).items())
        ),
        "selection": {
            "maximum_holdout_mean_wer_change": -0.01,
            "maximum_source_mean_wer_regression": 0.02,
            "minimum_sources_improved": 5,
            "tie_break": "lower_identity_change_rate_then_earlier_epoch",
        },
        "open_screen": {
            "inputs": "runs/c5-resolution-tuning-v1/visual-crop-168-calibrated.jsonl",
            "inputs_sha256": sha256(
                ROOT / "runs/c5-resolution-tuning-v1/visual-crop-168-calibrated.jsonl"
            ),
            "maximum_equal_domain_mean_wer_change": -0.005,
            "maximum_domain_mean_wer_change": 0.0,
            "minimum_how2sign_signers_improved": 3,
        },
        "confirmation_policy": (
            "Do not reuse either consumed confirmation set. A passing open screen needs "
            "new confirmation evidence before packaging."
        ),
        "miner_health_policy": "user_authorized_maintenance",
        "plan_source_sha256": sha256(Path(__file__)),
        "trainer_source_sha256": sha256(ROOT / "tools/train_c5_text_corrector.py"),
    }
    plan_path = args.output_dir / "plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"plan_sha256": sha256(plan_path), "train": len(train), "holdout": len(holdout)}, sort_keys=True))


if __name__ == "__main__":
    main()
