"""Freeze a balanced 16-example memorization gate for the fresh stream model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from benchmark_fleurs_dev import sha256
from fresh_stream_data import load_inventory

GROUPS = {
    "how2sign": {"how2sign_original", "how2sign_new", "how2sign_raw"},
    "fleurs": {"fleurs"},
    "stem": {"asl_stem_original", "asl_stem_supplemental"},
    "openasl": {"openasl"},
}


def stable_key(group: str, row: dict) -> str:
    return hashlib.sha256(f"fresh-stream-overfit-v1\0{group}\0{row['source']}\0{row['example_id']}".encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.output.exists() or not args.output.resolve().is_relative_to((root / "runs").resolve()):
        parser.error("selection must be a new run-ledger file")
    rows = load_inventory(args.inventory, root)
    selected = []
    for group, sources in GROUPS.items():
        candidates = [row for row in rows if row["source"] in sources]
        chosen = sorted(candidates, key=lambda row: stable_key(group, row))[:4]
        if len(chosen) != 4:
            raise ValueError(f"fresh stream overfit group is incomplete: {group}")
        selected.extend({"group": group, "source": row["source"], "example_id": row["example_id"]} for row in chosen)
    result = {
        "schema": "umi-studio-fresh-stream-overfit-selection/1",
        "inventory_sha256": sha256(args.inventory),
        "examples": selected,
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
    print(sha256(args.output))


if __name__ == "__main__":
    main()
