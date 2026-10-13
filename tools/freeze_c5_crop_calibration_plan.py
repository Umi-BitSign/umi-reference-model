"""Freeze training-only pairs for calibrating 168px C5 DINO features."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

from benchmark_fleurs_dev import sha256

SCHEMA = "umi-studio-c5-crop-calibration-plan/1"
SEED = 109
PER_SIGNER_CAP = 24


def select(source_rows: list[dict], holdout_rows: list[dict]) -> list[dict]:
    holdout = {row["sentence_name"] for row in holdout_rows}
    grouped = {}
    for row in source_rows:
        name = row.get("sentence_name")
        if (
            row.get("schema")
            != "umi-studio-how2sign-precut-expansion-stream-cache/1"
            or row.get("split") != "training"
            or row.get("temporal_stride") != 2
            or name in holdout
        ):
            continue
        grouped.setdefault(row["signer"], []).append(row)
    selected = []
    for signer, rows in sorted(grouped.items()):
        rows.sort(
            key=lambda row: (
                hashlib.sha256(f"{SEED}:{row['sentence_name']}".encode()).hexdigest(),
                row["sentence_name"],
            )
        )
        selected.extend(rows[:PER_SIGNER_CAP])
    selected.sort(key=lambda row: (row["signer"], row["sentence_name"]))
    counts = Counter(row["signer"] for row in selected)
    if counts != Counter({1: 24, 2: 14, 3: 24, 4: 24, 5: 24, 8: 24, 9: 11, 11: 24}):
        raise ValueError(f"crop calibration signer capacity differs: {counts}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--holdout-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("crop calibration plan already exists")
    source = [json.loads(line) for line in args.source_index.read_text().splitlines()]
    holdout = [json.loads(line) for line in args.holdout_index.read_text().splitlines()]
    rows = select(source, holdout)
    payload = {
        "schema": SCHEMA,
        "training_only": True,
        "seed": SEED,
        "per_signer_cap": PER_SIGNER_CAP,
        "examples": len(rows),
        "source_index_sha256": sha256(args.source_index),
        "holdout_index_sha256": sha256(args.holdout_index),
        "holdout_examples_excluded": len(holdout),
        "per_signer": dict(sorted(Counter(str(row["signer"]) for row in rows).items())),
        "records": [
            {
                key: row[key]
                for key in (
                    "sentence_name",
                    "signer",
                    "source_group",
                    "reference",
                    "resampled_clip_sha256",
                    "landmarks_sha256",
                    "encoder_cache_sha256",
                    "frames",
                )
            }
            for row in rows
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".partial.json")
    try:
        with temporary.open("x", encoding="utf-8") as output:
            output.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            output.flush()
            os.fsync(output.fileno())
        temporary.rename(args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps({"examples": len(rows), "per_signer": payload["per_signer"]}, sort_keys=True))


if __name__ == "__main__":
    main()
