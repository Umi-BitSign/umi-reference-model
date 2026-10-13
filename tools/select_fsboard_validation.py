"""Freeze a small signer-balanced FSboard validation diagnostic.

The public validation split is for offline model selection only. The test split
is not read, and no source phrase or video enters the repository or live miner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from benchmark_fleurs_dev import sha256

CLIP_NAME = re.compile(r"[A-Za-z0-9_.-]+\.mp4\Z")
MD5 = re.compile(r"[0-9a-f]{32}\Z")
SOURCE_PREFIX = "daun_v3/video_clips/daun_v3-validation"


def select_rows(
    metadata: list[dict],
    seed: int,
    clips_per_signer: int,
    excluded_rows: list[dict] | None = None,
) -> list[dict]:
    if clips_per_signer < 1:
        raise ValueError("clips per signer must be positive")
    eligible = {}
    for row in metadata:
        name = row["clipFilename"]
        phrase = row["phrase"]
        duration = row["clipEndTimeS"] - row["clipStartTimeS"]
        if (
            row["sensitiveContent"] is not False
            or row["signingType"] != "fs"
            or not isinstance(phrase, str)
            or not 3 <= len(phrase.strip()) <= 25
            or not 2 <= duration <= 8
            or not CLIP_NAME.fullmatch(name)
            or not MD5.fullmatch(row["clipFileMd5"])
            or not 0 < row["clipFileSize"] <= 30_000_000
        ):
            continue
        eligible.setdefault(row["signerId"], []).append(row)
    selected = []
    excluded_rows = excluded_rows or []
    used_names = {row["clip_filename"] for row in excluded_rows}
    used_phrases = {row["reference"].strip().casefold() for row in excluded_rows}
    used_recordings = {row["source_recording"] for row in excluded_rows}
    for signer, rows in sorted(eligible.items()):
        ordered = sorted(
            rows,
            key=lambda row: (
                hashlib.sha256(f"{seed}:{row['clipFilename']}".encode()).hexdigest(),
                row["clipFilename"],
            ),
        )
        recordings = set()
        count = 0
        for row in ordered:
            name = row["clipFilename"]
            phrase_key = row["phrase"].strip().casefold()
            recording = row["fullVideoFilename"]
            if (
                name in used_names
                or phrase_key in used_phrases
                or recording in recordings
                or recording in used_recordings
            ):
                continue
            selected.append(
                {
                    "clip_filename": name,
                    "source_path": f"{SOURCE_PREFIX}/{name}",
                    "size_bytes": row["clipFileSize"],
                    "md5": row["clipFileMd5"],
                    "reference": row["phrase"],
                    "signer_id": signer,
                    "source_recording": recording,
                    "duration_seconds": round(row["clipEndTimeS"] - row["clipStartTimeS"], 6),
                }
            )
            used_names.add(name)
            used_phrases.add(phrase_key)
            recordings.add(recording)
            used_recordings.add(recording)
            count += 1
            if count == clips_per_signer:
                break
        if count != clips_per_signer:
            raise ValueError(f"not enough distinct validation recordings for signer {signer}")
    if not selected:
        raise ValueError("no eligible validation clips")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--clips-per-signer", type=int, default=2)
    parser.add_argument("--expected-signers", type=int, default=15)
    parser.add_argument("--exclude-selection", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("selection output already exists")
    metadata = json.loads(args.metadata.read_text())
    if not isinstance(metadata, list):
        raise ValueError("FSboard metadata must be a JSON array")
    excluded = None
    exclusion_hash = None
    if args.exclude_selection is not None:
        previous = json.loads(args.exclude_selection.read_text())
        if (
            previous.get("schema") != "umi-studio-fsboard-validation-pilot/1"
            or previous.get("dataset") != "googleai/fsboard"
            or previous.get("source_split") != "validation"
            or previous.get("test_used") is not False
            or previous.get("metadata_sha256") != sha256(args.metadata)
            or not isinstance(previous.get("rows"), list)
            or not previous["rows"]
        ):
            raise ValueError("excluded selection differs from public validation metadata")
        excluded = previous["rows"]
        exclusion_hash = sha256(args.exclude_selection)
    rows = select_rows(metadata, args.seed, args.clips_per_signer, excluded)
    signers = sorted({row["signer_id"] for row in rows})
    if len(signers) != args.expected_signers:
        raise ValueError("validation signer count differs from the pinned expectation")
    result = {
        "schema": "umi-studio-fsboard-validation-pilot/1",
        "dataset": "googleai/fsboard",
        "dataset_license": "CC BY 4.0",
        "source_subset": "daun_v3",
        "source_split": "validation",
        "test_used": False,
        "metadata_sha256": sha256(args.metadata),
        "seed": args.seed,
        "clips_per_signer": args.clips_per_signer,
        "signer_count": len(signers),
        "rows": rows,
    }
    if exclusion_hash is not None:
        result["excluded_selection_sha256"] = exclusion_hash
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, sort_keys=True, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(f"selected {len(rows)} validation clips from {len(signers)} signers")


if __name__ == "__main__":
    main()
