"""Freeze all available generic visual streams for fresh-model training."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

from benchmark_fleurs_dev import sha256
from freeze_c5_openasl_bbox_plan import load_openasl_bbox_examples
from freeze_c5_stem_scale_plan import ROOT, load_examples
from fresh_stream_data import SCHEMA

EXPECTED_COUNTS = {
    "asl_stem_original": 1024,
    "asl_stem_supplemental": 1024,
    "fleurs": 313,
    "how2sign_new": 2335,
    "how2sign_original": 1024,
    "how2sign_raw": 882,
    "openasl": 120,
}


def inventory_rows() -> list[dict]:
    replay, stem, _ = load_examples()
    openasl, _ = load_openasl_bbox_examples()
    examples = replay + stem + openasl
    if Counter(row["source"] for row in examples) != Counter(EXPECTED_COUNTS):
        raise ValueError("fresh stream source counts differ")
    rows = []
    for row in sorted(examples, key=lambda item: (item["source"], item["id"])):
        cache = Path(row["cache"])
        sidecar = Path(row["sidecar"]) if row.get("sidecar") else None
        rows.append(
            {
                "schema": SCHEMA,
                "split": "training",
                "source": row["source"],
                "source_group": row.get("source_group", f"{row['source']}:{row['signer']}"),
                "signer": row["signer"],
                "example_id": row["id"],
                "reference": row["reference"],
                "cache": str(cache.resolve().relative_to(ROOT)),
                "cache_sha256": row["cache_sha256"],
                "sidecar": str(sidecar.resolve().relative_to(ROOT)) if sidecar else None,
                "sidecar_sha256": row.get("sidecar_sha256"),
                "frames": row["frames"],
                "stored_temporal_stride": row.get("stored_temporal_stride", 1),
            }
        )
    if len(rows) != 6722 or len({(row["source"], row["example_id"]) for row in rows}) != 6722:
        raise ValueError("fresh stream inventory identity count differs")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to((ROOT / "runs").resolve()):
        parser.error("inventory must be a new run-ledger file")
    rows = inventory_rows()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".partial.jsonl")
    try:
        with temporary.open("x", encoding="utf-8") as output:
            for row in rows:
                output.write(json.dumps(row, sort_keys=True) + "\n")
            output.flush()
            os.fsync(output.fileno())
        temporary.rename(args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps({"examples": len(rows), "sha256": sha256(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()

