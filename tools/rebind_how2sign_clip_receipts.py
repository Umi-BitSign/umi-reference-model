"""Rebind verified How2Sign clip receipts to an expanded training selection.

The clip bytes stay in place. The new receipt ledger records their original
download provenance and is written only after every prior clip is rehashed.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from benchmark_fleurs_dev import miner_revision, sha256

SOURCE_FIELDS = (
    "source_url",
    "archive_google_drive_file_id",
    "archive_size_bytes",
    "archive_fingerprint_sha256",
    "metadata_sha256",
    "license_sha256",
    "alignment_metadata_sha256",
    "min_bidirectional_overlap",
    "seed",
)


def rebound_receipts(
    prior_selection: dict,
    expanded_selection: dict,
    prior_receipts: list[dict],
    source_dir: Path,
    prior_selection_hash: str,
    expanded_selection_hash: str,
    prior_receipts_hash: str,
    health_url: str | None = None,
) -> list[dict]:
    if (
        prior_selection.get("schema") != "umi-studio-how2sign-training-aligned/1"
        or expanded_selection.get("schema") != "umi-studio-how2sign-training-aligned/1"
        or any(
            key not in prior_selection
            or key not in expanded_selection
            or prior_selection[key] != expanded_selection[key]
            for key in SOURCE_FIELDS
        )
    ):
        raise ValueError("expanded selection uses different training sources")
    prior_rows = prior_selection["rows"]
    expanded_rows = expanded_selection["rows"]
    prior_by_name = {row["sentence_name"]: row for row in prior_rows}
    expanded_by_name = {row["sentence_name"]: row for row in expanded_rows}
    if (
        len(prior_by_name) != len(prior_rows)
        or len(expanded_by_name) != len(expanded_rows)
        or len(expanded_rows) <= len(prior_rows)
        or any(expanded_by_name.get(name) != row for name, row in prior_by_name.items())
    ):
        raise ValueError("prior training rows are not an exact subset of expansion")
    receipt_by_name = {record["sentence_name"]: record for record in prior_receipts}
    if len(receipt_by_name) != len(prior_receipts) or set(receipt_by_name) != set(prior_by_name):
        raise ValueError("prior download receipts are incomplete or duplicated")
    revision = miner_revision(health_url, None) if health_url else None
    rebound = []
    for row in expanded_rows:
        name = row["sentence_name"]
        if name not in prior_by_name:
            continue
        if Path(name).name != name:
            raise ValueError("sentence name contains a path separator")
        if health_url and len(rebound) % 32 == 0:
            revision = miner_revision(health_url, revision)
        receipt = receipt_by_name[name]
        if (
            receipt["schema"] != "umi-studio-how2sign-training-clip/1"
            or receipt["selection_sha256"] != prior_selection_hash
            or receipt["archive_member"] != row["archive_member"]
            or receipt["archive_crc32"] != row["archive_crc32"]
            or receipt["clip_size_bytes"] != row["archive_file_size"]
        ):
            raise ValueError(f"prior receipt differs from expanded selection: {name}")
        clip = source_dir / f"{name}.mp4"
        if (
            not clip.is_file()
            or clip.stat().st_size != row["archive_file_size"]
            or sha256(clip) != receipt["clip_sha256"]
        ):
            raise ValueError(f"downloaded clip differs from prior receipt: {clip}")
        rebound.append(
            {
                **receipt,
                "selection_sha256": expanded_selection_hash,
                "reuse_source": {
                    "prior_selection_sha256": prior_selection_hash,
                    "prior_clip_receipts_sha256": prior_receipts_hash,
                    "reuse_miner_model_revision": revision,
                },
            }
        )
    if health_url:
        miner_revision(health_url, revision)
    return rebound


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-selection", type=Path, required=True)
    parser.add_argument("--prior-receipts", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--miner-health-url", required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("expanded receipt ledger already exists")
    prior_selection = json.loads(args.prior_selection.read_text())
    expanded_selection = json.loads(args.selection.read_text())
    prior_receipts = [json.loads(line) for line in args.prior_receipts.read_text().splitlines()]
    rebound = rebound_receipts(
        prior_selection,
        expanded_selection,
        prior_receipts,
        args.source_dir,
        sha256(args.prior_selection),
        sha256(args.selection),
        sha256(args.prior_receipts),
        args.miner_health_url,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".partial.jsonl")
    if temporary.exists():
        parser.error("unfinished expanded receipt ledger needs inspection")
    try:
        with temporary.open("x", encoding="utf-8") as output:
            for record in rebound:
                output.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
            output.flush()
            os.fsync(output.fileno())
        temporary.rename(args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"rebound {len(rebound)} verified clip receipts")


if __name__ == "__main__":
    main()
